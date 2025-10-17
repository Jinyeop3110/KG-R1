#!/usr/bin/env python3
"""
Simple FSDP checkpoint converter - CPU/Single GPU version.

This script merges FSDP sharded checkpoints into a single HuggingFace model
without requiring distributed training setup.

Usage:
    python convert_checkpoint_simple.py \
        /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def get_world_size_from_checkpoint(checkpoint_dir):
    """Infer world_size from checkpoint filenames."""
    checkpoint_dir = Path(checkpoint_dir)
    model_files = list(checkpoint_dir.glob("model_world_size_*_rank_*.pt"))

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

    print(f"✓ Detected world_size={world_size} from checkpoint")
    return world_size


def merge_fsdp_shards(checkpoint_dir, world_size):
    """
    Merge FSDP sharded checkpoints into a full state dict.

    This uses a simple heuristic:
    - If a parameter is identical across all shards, it's replicated (use any copy)
    - If a parameter differs, it's sharded (concatenate along dim 0)
    """
    checkpoint_dir = Path(checkpoint_dir)

    print(f"\nLoading {world_size} checkpoint shards...")
    shards = []
    for rank in range(world_size):
        shard_path = checkpoint_dir / f"model_world_size_{world_size}_rank_{rank}.pt"
        print(f"  [{rank+1}/{world_size}] Loading {shard_path.name}...", end=" ")
        shard = torch.load(shard_path, map_location="cpu", weights_only=False)

        # Convert DTensor to regular tensor if needed
        cleaned_shard = {}
        for key, value in shard.items():
            if hasattr(value, '_local_tensor'):
                # This is a DTensor - extract the local tensor
                cleaned_shard[key] = value._local_tensor
            else:
                cleaned_shard[key] = value

        print(f"({len(cleaned_shard)} keys)")
        shards.append(cleaned_shard)

    print(f"\nMerging shards into full state dict...")
    full_state_dict = {}

    all_keys = set(shards[0].keys())

    # Verify all shards have the same keys
    for i, shard in enumerate(shards[1:], 1):
        if set(shard.keys()) != all_keys:
            print(f"⚠️  Warning: Shard {i} has different keys than shard 0")

    replicated_count = 0
    sharded_count = 0

    for key in sorted(all_keys):
        # Get tensors from all shards
        tensors = [shard[key] for shard in shards]

        # Clean up FSDP-specific key naming
        # Remove "_fsdp_wrapped_module" if present
        clean_key = key.replace("_fsdp_wrapped_module.", "")

        # Check if parameter is replicated or sharded
        if all(torch.equal(tensors[0], t) for t in tensors[1:]):
            # Replicated parameter - identical across all ranks
            full_state_dict[clean_key] = tensors[0]
            replicated_count += 1
        else:
            # Sharded parameter - concatenate along dim 0
            # FSDP typically shards along the first dimension
            full_state_dict[clean_key] = torch.cat(tensors, dim=0)
            sharded_count += 1

    print(f"✓ Merged {len(full_state_dict)} parameters:")
    print(f"  - {replicated_count} replicated parameters")
    print(f"  - {sharded_count} sharded parameters")

    return full_state_dict


def save_merged_model(actor_dir, output_dir, full_state_dict, torch_dtype="bfloat16"):
    """Save merged checkpoint as HuggingFace model."""
    actor_dir = Path(actor_dir)
    output_dir = Path(output_dir)

    print(f"\nSaving merged model to {output_dir}")

    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    config = AutoConfig.from_pretrained(actor_dir)

    # Create model
    print("  Creating model from config...")
    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}

    from accelerate import init_empty_weights

    # Create empty model
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(
            config,
            torch_dtype=dtype_map.get(torch_dtype, torch.bfloat16),
        )

    # Move to CPU and load state dict
    model.to_empty(device="cpu")
    print("  Loading merged state dict into model...")
    model.load_state_dict(full_state_dict, assign=True)

    # Save model
    print(f"  Saving model with dtype={torch_dtype}...")
    model.save_pretrained(output_dir, max_shard_size="5GB", safe_serialization=True)

    # Copy tokenizer files
    print("  Copying tokenizer files...")
    tokenizer_files = [
        "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
        "vocab.json", "merges.txt", "added_tokens.json", "chat_template.jinja"
    ]

    for filename in tokenizer_files:
        src = actor_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    # Copy config and generation config
    for filename in ["config.json", "generation_config.json"]:
        src = actor_dir / filename
        if src.exists():
            shutil.copy2(src, output_dir / filename)

    print(f"✓ Model saved successfully")


def merge_extra_states(actor_dir, output_dir, world_size):
    """Merge extra_state files (lr_scheduler, rng states)."""
    actor_dir = Path(actor_dir)
    output_dir = Path(output_dir)

    print(f"\nMerging extra_state files...")

    extra_states = []
    for rank in range(world_size):
        extra_path = actor_dir / f"extra_state_world_size_{world_size}_rank_{rank}.pt"
        if extra_path.exists():
            extra_state = torch.load(extra_path, map_location="cpu", weights_only=False)
            extra_states.append(extra_state)

    if not extra_states:
        print("  No extra_state files found")
        return

    # Use rank 0's extra_state (should be identical across ranks)
    merged_extra_state = extra_states[0]

    # Save merged extra_state
    extra_output_path = output_dir / "extra_state_merged.pt"
    torch.save(merged_extra_state, extra_output_path)
    print(f"✓ Saved extra_state to {extra_output_path}")

    # Save lr_scheduler state as JSON for inspection
    if "lr_scheduler" in merged_extra_state:
        lr_state_json = output_dir / "lr_scheduler_state.json"
        with open(lr_state_json, "w") as f:
            json.dump(merged_extra_state["lr_scheduler"], f, indent=2, default=str)
        print(f"✓ Saved lr_scheduler state to {lr_state_json}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert FSDP sharded checkpoint to HuggingFace format (Simple CPU version)"
    )
    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to checkpoint directory (e.g., .../global_step_150)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: {checkpoint_path}_merged)",
    )
    parser.add_argument(
        "--actor-dir",
        type=str,
        default="actor",
        help="Subdirectory containing checkpoints (default: actor)",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["float32", "float16", "bfloat16"],
        help="Model dtype for saving (default: bfloat16)",
    )

    args = parser.parse_args()

    # Resolve paths
    checkpoint_path = Path(args.checkpoint_path).resolve()
    if not checkpoint_path.exists():
        print(f"❌ Error: Checkpoint path does not exist: {checkpoint_path}")
        sys.exit(1)

    actor_dir = checkpoint_path / args.actor_dir
    if not actor_dir.exists():
        print(f"❌ Error: Actor directory does not exist: {actor_dir}")
        sys.exit(1)

    # Determine output path
    if args.output is None:
        # Generate smart default output path
        # If checkpoint is like: verl_checkpoints/model-name/global_step_150
        # Output will be: verl_checkpoints/model-name-merged-step150
        checkpoint_parent = checkpoint_path.parent
        checkpoint_name = checkpoint_path.name  # e.g., "global_step_150"

        # Extract step number
        if "global_step_" in checkpoint_name:
            step_num = checkpoint_name.replace("global_step_", "")
            model_name = checkpoint_parent.name
            output_name = f"{model_name}-merged-step{step_num}"
            output_path = checkpoint_parent.parent / output_name
        else:
            # Fallback to simple _merged suffix
            output_path = Path(str(checkpoint_path) + "_merged")
    else:
        output_path = Path(args.output).resolve()

    # Print configuration
    print("=" * 70)
    print("FSDP Checkpoint Converter (Simple Version)")
    print("=" * 70)
    print(f"Checkpoint:    {checkpoint_path}")
    print(f"Actor dir:     {actor_dir}")
    print(f"Output:        {output_path}")
    print(f"Dtype:         {args.dtype}")
    print("=" * 70)

    # Step 1: Detect world size
    world_size = get_world_size_from_checkpoint(actor_dir)

    # Step 2: Merge sharded checkpoints
    full_state_dict = merge_fsdp_shards(actor_dir, world_size)

    # Step 3: Save merged model
    save_merged_model(actor_dir, output_path, full_state_dict, args.dtype)

    # Step 4: Merge extra states
    merge_extra_states(actor_dir, output_path, world_size)

    # Done
    print("\n" + "=" * 70)
    print("✅ Conversion complete!")
    print("=" * 70)
    print(f"\nMerged model saved to: {output_path}")
    print(f"\nYou can now load this model with:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{output_path}')")
    print("=" * 70)


if __name__ == "__main__":
    main()
