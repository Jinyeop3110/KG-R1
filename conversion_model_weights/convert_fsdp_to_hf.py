#!/usr/bin/env python3
# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert FSDP sharded checkpoints to a single HuggingFace model.

Usage:
    # Single GPU conversion (if model fits in memory)
    python convert_fsdp_to_hf.py \
        --checkpoint_path /path/to/checkpoint/global_step_150 \
        --output_path /path/to/output_merged

    # Multi-GPU conversion (recommended for large models)
    torchrun --nproc_per_node=4 convert_fsdp_to_hf.py \
        --checkpoint_path /path/to/checkpoint/global_step_150 \
        --output_path /path/to/output_merged
"""

import argparse
import json
import os
import shutil
import warnings
from pathlib import Path

import torch
import torch.distributed as dist
from accelerate import init_empty_weights
from torch.distributed.fsdp import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, GenerationConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Convert FSDP sharded checkpoint to HuggingFace format")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to checkpoint directory (e.g., global_step_150)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for merged model. Defaults to {checkpoint_path}_merged",
    )
    parser.add_argument(
        "--actor_subdir",
        type=str,
        default="actor",
        help="Subdirectory containing model checkpoints (default: actor)",
    )
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Torch dtype for the merged model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Trust remote code when loading model",
    )
    return parser.parse_args()


def get_world_size_from_checkpoint(checkpoint_dir):
    """Infer world_size from checkpoint filenames."""
    model_files = list(Path(checkpoint_dir).glob("model_world_size_*_rank_*.pt"))
    if not model_files:
        raise ValueError(f"No FSDP checkpoint files found in {checkpoint_dir}")

    # Extract world_size from filename like "model_world_size_4_rank_0.pt"
    filename = model_files[0].name
    world_size = int(filename.split("world_size_")[1].split("_rank_")[0])

    # Verify all ranks exist
    for rank in range(world_size):
        rank_file = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        if not rank_file.exists():
            raise ValueError(f"Missing checkpoint for rank {rank}: {rank_file}")

    print(f"Detected world_size={world_size} from checkpoint")
    return world_size


def setup_distributed(world_size):
    """Setup distributed training if running with multiple processes."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Running with torchrun
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size_env = int(os.environ["WORLD_SIZE"])

        if world_size_env != world_size:
            raise ValueError(
                f"Mismatch: checkpoint has world_size={world_size} but "
                f"running with {world_size_env} processes. "
                f"Please run with: torchrun --nproc_per_node={world_size}"
            )

        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
        torch.cuda.set_device(local_rank)
        return rank, local_rank, world_size_env, True
    else:
        # Single process - will load sequentially
        return 0, 0, 1, False


def load_and_merge_sharded_checkpoints(checkpoint_dir, world_size, torch_dtype, is_distributed):
    """Load FSDP sharded checkpoints and merge into full state dict."""
    print(f"\n{'='*60}")
    print(f"Loading and merging {world_size} checkpoint shards...")
    print(f"{'='*60}\n")

    checkpoint_dir = Path(checkpoint_dir)

    if is_distributed:
        # Multi-GPU: each rank loads its shard, FSDP merges
        rank = dist.get_rank()

        # Load config from checkpoint
        config = AutoConfig.from_pretrained(checkpoint_dir)

        # Create FSDP-wrapped model
        if rank == 0:
            print("Creating FSDP model...")

        # Initialize model with FSDP
        from torch.distributed.device_mesh import init_device_mesh
        from verl.utils.fsdp_utils import get_fsdp_wrap_policy, MixedPrecisionPolicy

        device_mesh = init_device_mesh("cuda", mesh_shape=(world_size,), mesh_dim_names=["fsdp"])

        # Create model
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(config)

        # Wrap with FSDP
        wrap_policy = get_fsdp_wrap_policy(model)
        mixed_precision = MixedPrecisionPolicy.get_mixed_precision_policy(
            param_dtype=getattr(torch, torch_dtype),
            reduce_dtype=torch.float32,
            buffer_dtype=getattr(torch, torch_dtype)
        )

        model = FSDP(
            model,
            auto_wrap_policy=wrap_policy,
            mixed_precision=mixed_precision,
            device_mesh=device_mesh,
            use_orig_params=True,
        )

        # Load sharded checkpoint
        shard_path = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"[Rank {rank}] Loading shard from {shard_path}")

        shard_state_dict = torch.load(shard_path, map_location="cpu", weights_only=False)

        # Load into FSDP model
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True)
        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg):
            model.load_state_dict(shard_state_dict)

        dist.barrier()

        # Gather full state dict on rank 0
        if rank == 0:
            print("\nGathering full state dict on rank 0...")

        full_state_config = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_config):
            full_state_dict = model.state_dict()

        dist.barrier()

        return full_state_dict if rank == 0 else None, config

    else:
        # Single GPU/CPU: manually merge shards
        print("⚠️  Running in single-process mode - will merge shards manually")
        print("   This is slower than multi-GPU conversion but works without torchrun")

        # Load config
        config = AutoConfig.from_pretrained(checkpoint_dir)

        # Load all shards
        shards = []
        for rank in range(world_size):
            shard_path = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
            print(f"Loading shard {rank+1}/{world_size} from {shard_path}")
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            shards.append(shard)

        # Merge shards
        print("\nMerging shards into full state dict...")
        full_state_dict = {}

        # FSDP shards have keys that need to be cleaned up
        # Example: "model.layers.0._fsdp_wrapped_module.self_attn.q_proj.weight"
        # Should become: "model.layers.0.self_attn.q_proj.weight"

        for key in shards[0].keys():
            # Get tensors from all shards
            tensors = [shard[key] for shard in shards]

            # Clean up FSDP-specific key naming
            clean_key = key.replace("_fsdp_wrapped_module.", "")

            # For FSDP, different parameters are sharded differently:
            # - Some are replicated (same across all ranks)
            # - Some are sharded (need concatenation)

            if all(torch.equal(tensors[0], t) for t in tensors[1:]):
                # Replicated parameter - use from rank 0
                full_state_dict[clean_key] = tensors[0]
            else:
                # Sharded parameter - concatenate along dim 0 (FSDP default)
                full_state_dict[clean_key] = torch.cat(tensors, dim=0)

        return full_state_dict, config


def save_merged_checkpoint(checkpoint_dir, output_dir, full_state_dict, config, torch_dtype, trust_remote_code):
    """Save merged checkpoint as HuggingFace model."""
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)

    print(f"\n{'='*60}")
    print(f"Saving merged checkpoint to {output_dir}")
    print(f"{'='*60}\n")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create model and load merged state dict
    print("Creating model from config...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype_map[torch_dtype])

    model.to_empty(device="cpu")

    print("Loading merged state dict into model...")
    model.load_state_dict(full_state_dict, assign=True)

    # Copy tokenizer and other files
    print("Copying tokenizer and config files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja"
    ]

    for filename in tokenizer_files:
        src = checkpoint_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)
            print(f"  Copied {filename}")

    # Copy/update config
    src_config = checkpoint_dir / "config.json"
    if src_config.exists():
        shutil.copy2(src_config, output_dir / "config.json")
        print(f"  Copied config.json")

    # Copy generation config if exists
    src_gen_config = checkpoint_dir / "generation_config.json"
    if src_gen_config.exists():
        shutil.copy2(src_gen_config, output_dir / "generation_config.json")
        print(f"  Copied generation_config.json")

    # Save model
    print(f"\nSaving merged model with dtype={torch_dtype}...")
    model.save_pretrained(output_dir, max_shard_size="5GB")

    # Merge and save extra_state (lr_scheduler, rng) from all ranks
    print("\nMerging extra_state files...")
    extra_states = []
    world_size = get_world_size_from_checkpoint(checkpoint_dir)

    for rank in range(world_size):
        extra_path = checkpoint_dir / f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        if extra_path.exists():
            extra_state = torch.load(extra_path, map_location="cpu", weights_only=False)
            extra_states.append(extra_state)

    if extra_states:
        # All ranks should have identical lr_scheduler and rng states
        # Just use rank 0's extra_state
        merged_extra_state = extra_states[0]

        # Save merged extra_state
        extra_output_path = output_dir / "extra_state_merged.pt"
        torch.save(merged_extra_state, extra_output_path)
        print(f"  Saved merged extra_state to {extra_output_path}")

        # Also save as human-readable JSON for inspection
        if "lr_scheduler" in merged_extra_state:
            lr_state_json = output_dir / "lr_scheduler_state.json"
            with open(lr_state_json, "w") as f:
                json.dump(merged_extra_state["lr_scheduler"], f, indent=2, default=str)
            print(f"  Saved lr_scheduler state to {lr_state_json}")

    print(f"\n✅ Successfully saved merged checkpoint to {output_dir}")
    print(f"\nYou can now load this model with:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_dir}')")


def main():
    args = parse_args()

    # Resolve paths
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    # Get actor subdirectory
    actor_dir = checkpoint_path / args.actor_subdir
    if not actor_dir.exists():
        raise ValueError(f"Actor directory does not exist: {actor_dir}")

    # Determine output path
    if args.output_path is None:
        output_path = Path(str(checkpoint_path) + "_merged")
    else:
        output_path = Path(args.output_path).resolve()

    print(f"\n{'='*60}")
    print(f"FSDP to HuggingFace Checkpoint Converter")
    print(f"{'='*60}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"Actor directory: {actor_dir}")
    print(f"Output path:     {output_path}")
    print(f"Torch dtype:     {args.torch_dtype}")
    print(f"{'='*60}\n")

    # Detect world_size from checkpoint
    world_size = get_world_size_from_checkpoint(actor_dir)

    # Setup distributed if needed
    rank, local_rank, env_world_size, is_distributed = setup_distributed(world_size)

    # Load and merge checkpoints
    full_state_dict, config = load_and_merge_sharded_checkpoints(
        actor_dir, world_size, args.torch_dtype, is_distributed
    )

    # Save merged checkpoint (only rank 0)
    if not is_distributed or rank == 0:
        if full_state_dict is not None:
            save_merged_checkpoint(
                actor_dir,
                output_path,
                full_state_dict,
                config,
                args.torch_dtype,
                args.trust_remote_code,
            )

    if is_distributed:
        dist.barrier()
        dist.destroy_process_group()

    print("\n✅ Conversion complete!")


if __name__ == "__main__":
    main()
