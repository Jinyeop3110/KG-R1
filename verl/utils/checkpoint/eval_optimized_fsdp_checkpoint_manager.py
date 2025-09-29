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

import os
import warnings
from typing import Optional, Union

import torch
import torch.distributed
from torch.distributed.fsdp import ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.device import is_cuda_available
from verl.utils.fs import copy_to_local, is_non_local
from verl.utils.fsdp_utils import get_fsdp_state_ctx

from .checkpoint_manager import BaseCheckpointManager


class EvalOptimizedFSDPCheckpointManager(BaseCheckpointManager):
    """
    Memory-optimized FSDP checkpoint manager for evaluation that SKIPS optimizer states.
    
    This reduces memory usage from ~36GB to ~6.6GB for evaluation by only loading:
    - Model weights (required for inference)
    - Extra state (scheduler + RNG, minimal memory)
    
    SKIPS:
    - Optimizer states (~30GB) - not needed for evaluation
    
    Use this for evaluation scripts to avoid OOM in limited memory environments.
    """

    def __init__(
        self,
        model: FSDP,
        optimizer: torch.optim.Optimizer = None,  # Can be None for eval
        lr_scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        processing_class: Union[PreTrainedTokenizer, ProcessorMixin] = None,
        checkpoint_contents: Optional[list] = None,
        **kwargs,
    ):
        # For evaluation, we only need model and minimal extra state
        if checkpoint_contents is None:
            checkpoint_contents = ["model", "extra"]  # Skip optimizer
        
        if processing_class is None:
            assert "tokenizer" in kwargs, "tokenizer or processor must be provided"
            warnings.warn("`tokenizer` is deprecated. use `processing_class` instead.", DeprecationWarning, stacklevel=2)
            processing_class = kwargs.pop("tokenizer")

        # Only require model for evaluation
        assert "model" in checkpoint_contents, f"EvalOptimizedFSDPCheckpointManager must include 'model', got {checkpoint_contents}"

        super().__init__(
            model,
            optimizer,
            lr_scheduler=lr_scheduler,
            processing_class=processing_class,
            checkpoint_contents=checkpoint_contents,
        )

    def load_checkpoint(self, local_path: str, hdfs_path: str = None, del_local_after_load=False):
        """
        Load FSDP checkpoint optimized for evaluation - SKIPS optimizer states.
        
        Only loads:
          - model shards (required for inference)
          - extra state dict (scheduler + RNG, minimal memory)
        
        SKIPS:
          - optimizer shards (~30GB saved!)
        
        Args:
            local_path: Directory with per-rank checkpoint files.
            hdfs_path: Unused (for API compatibility).
            del_local_after_load: Remove local files after loading.
        """
        if local_path is None:
            return

        print(f"[EVAL-OPTIMIZED] Loading model-only checkpoint from {local_path}")
        print(f"[EVAL-OPTIMIZED] Skipping optimizer states to save ~30GB memory")

        # Load only model and extra state (skip optimizer)
        remote_model_path = os.path.join(local_path, f"model_world_size_{self.world_size}_rank_{self.rank}.pt")
        remote_extra_state_path = os.path.join(local_path, f"extra_state_world_size_{self.world_size}_rank_{self.rank}.pt")
        
        print(f"[rank-{self.rank}]: Loading model from {remote_model_path}")
        print(f"[rank-{self.rank}]: Loading extra_state from {remote_extra_state_path}")
        print(f"[rank-{self.rank}]: SKIPPING optimizer (saves ~30GB memory)")
        
        local_model_path = copy_to_local(remote_model_path)
        local_extra_state_path = copy_to_local(remote_extra_state_path)

        # Load tensors
        model_state_dict = torch.load(local_model_path, weights_only=False)
        extra_state_dict = torch.load(local_extra_state_path, weights_only=False)

        if del_local_after_load:
            try:
                os.remove(local_model_path) if is_non_local(local_model_path) else None
                os.remove(local_extra_state_path) if is_non_local(local_extra_state_path) else None
            except Exception as e:
                print(f"[rank-{self.rank}]: remove local resume ckpt file after loading failed, exception {e} will be ignored")

        # Load model state dict
        state_dict_cfg = ShardedStateDictConfig(offload_to_cpu=True if is_cuda_available else False)
        with get_fsdp_state_ctx(self.model, StateDictType.SHARDED_STATE_DICT, state_dict_cfg, None):  # No optimizer config
            self.model.load_state_dict(model_state_dict)
            print(f"[EVAL-OPTIMIZED] Model weights loaded successfully")

        # Load scheduler state if available (for compatibility)
        if "lr_scheduler" in extra_state_dict and self.lr_scheduler is not None:
            lr_scheduler_state_dict = extra_state_dict["lr_scheduler"]
            self.lr_scheduler.load_state_dict(lr_scheduler_state_dict)

        # Recover random state
        if "rng" in extra_state_dict:
            self.load_rng_state(extra_state_dict["rng"])

        print(f"[EVAL-OPTIMIZED] Checkpoint loaded with reduced memory footprint")
        print(f"[EVAL-OPTIMIZED] Memory saved by skipping optimizer: ~30GB")

    def save_checkpoint(self, local_path: str, hdfs_path: str = None, global_step: int = 0, max_ckpt_to_keep=None):
        """Not implemented - this manager is only for loading during evaluation."""
        raise NotImplementedError("EvalOptimizedFSDPCheckpointManager is read-only (evaluation only)")