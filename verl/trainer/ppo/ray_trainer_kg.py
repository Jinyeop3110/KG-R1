# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface

SAMPLE_ID HANDLING FIX:
======================
This trainer has been fixed to correctly handle sample_id from the dataset through
the entire pipeline to the LLMGenerationManager. The key issue was that the pop
operation was removing sample_id and dataset_name from the batch when creating
gen_batch for generation, causing the LLMGenerationManager to fall back to
generating "fallback_sample_000000" IDs instead of using authentic sample IDs
like "WebQTrn-643" from the dataset.

DataProto.pop() removes keys from the original DataProto and returns a NEW DataProto
with ONLY the popped keys. The fix is to ADD sample_id and dataset_name TO
the non_tensor_batch_keys_to_pop list AND meta_info keys to meta_info_keys_to_pop
so they get included in the gen_batch and preserved through the pipeline.

CLEAN TRAINING LOOP:
===================
The trainer handles KG-augmented training with minimal logging to avoid spam.
This helps monitor the quality of knowledge graph queries during training.
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from tensordict import TensorDict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import BaseCheckpointManager, find_latest_ckpt_path
from verl.utils.debug.performance import _timer
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean, pad_sequence_to_length
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager

# Import Role enum from the original trainer to ensure compatibility
from verl.trainer.ppo.ray_trainer import Role

WorkerType = Type[Worker]


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False, config=None, tokenizer=None):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    # Handle both structured rewards (kg_format_multiturn) and token-level rewards (legacy)
    if "token_level_scores" in data.batch:
        token_level_scores = data.batch["token_level_scores"]
    elif "dummy_reward_tensor" in data.batch:
        # For structured rewards, we need to convert them to token-level scores for KL penalty
        if "structured_rewards" in data.non_tensor_batch:
            print("[KL-PENALTY] Converting structured rewards to token-level scores for KL penalty calculation")
            
            # Import our conversion function
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
            from convert_structured_rewards import convert_structured_to_token_level
            
            # Get response mask for conversion
            if multi_turn:
                loss_mask = data.batch["loss_mask"]
                response_mask = loss_mask[:, -response_length:]
            else:
                attention_mask = data.batch["attention_mask"]
                response_mask = attention_mask[:, -response_length:]
            
            structured_rewards = data.non_tensor_batch["structured_rewards"]
            turn_sequence_tensor = data.batch.get("turn_sequence_tensor")
            
            if turn_sequence_tensor is not None:
                # Convert with turn-proportional strategy
                token_level_scores = convert_structured_to_token_level(
                    structured_rewards=structured_rewards,
                    response_mask=response_mask,
                    turn_sequence_tensor=turn_sequence_tensor,
                    distribution_strategy="turn_proportional"
                )
                print(f"[KL-PENALTY] Converted {len(structured_rewards)} structured rewards to token-level using turn_proportional strategy")
            else:
                # Fallback: uniform distribution across valid tokens
                print(f"[KL-PENALTY] Warning: No turn_sequence_tensor available, using uniform distribution fallback")
                batch_size, seq_len = response_mask.shape
                token_level_scores = torch.zeros_like(response_mask, dtype=torch.float32)
                
                for i, reward_dict in enumerate(structured_rewards):
                    # Sum all reward components properly
                    turn_reward_sum = sum(reward_dict.get("turn_rewards", {}).values())
                    global_reward_sum = sum(v for k, v in reward_dict.get("global_rewards", {}).items() 
                                          if not k.startswith('_raw_'))
                    total_reward = turn_reward_sum + global_reward_sum
                    
                    # Distribute across all valid response tokens
                    valid_positions = (response_mask[i] == 1).nonzero(as_tuple=True)[0]
                    if len(valid_positions) > 0:
                        reward_per_token = total_reward / len(valid_positions)
                        token_level_scores[i, valid_positions] = reward_per_token
        else:
            # Use dummy_reward_tensor as fallback (should be zeros)
            token_level_scores = data.batch["dummy_reward_tensor"]
    else:
        raise KeyError("Neither 'token_level_scores' nor 'dummy_reward_tensor' found in batch")
    batch_size = data.batch.batch_size[0]

    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    
    # Apply KG token masking to reduce KL penalty for knowledge graph tokens
    if config is not None:
        kg_config = config.get('algorithm', {}).get('kg_token_masking', {})
        if kg_config.get('enable', False):
            if tokenizer is None:
                raise ValueError("KG token masking is enabled but tokenizer is None. Please provide tokenizer to apply_kl_penalty function.")
            kg_mask = core_algos.create_kg_token_mask(responses, tokenizer, kg_config)
            reduction_factor = kg_config.get('reduction_factor', 0.1)
            kld[kg_mask] *= reduction_factor
    
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    
    # Handle potential size mismatch after padding
    if attention_mask.size(1) < response_length:
        # If attention_mask is shorter than responses, pad it
        pad_length = response_length - attention_mask.size(1)
        padding = torch.zeros(attention_mask.size(0), pad_length, dtype=attention_mask.dtype, device=attention_mask.device)
        attention_mask = torch.cat([attention_mask, padding], dim=1)
    
    response_mask = attention_mask[:, -response_length:]
    return response_mask


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, tokenizer=None, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        # Check if token_level_rewards already exists (from apply_kl_penalty)
        if "token_level_rewards" not in data.batch:
            # Handle both structured rewards (kg_format_multiturn) and token-level rewards (legacy)
            if "structured_rewards" in data.non_tensor_batch:
                print("[GAE-STRUCTURED] Converting structured rewards to token-level rewards for GAE (no KL penalty applied)")
                # Convert structured rewards to token-level rewards for GAE
                structured_rewards = data.non_tensor_batch["structured_rewards"]
                response_mask = data.batch["response_mask"]
                turn_sequence_tensor = data.batch.get("turn_sequence_tensor")
                
                # Use turn-aware distribution if turn information is available
                if turn_sequence_tensor is not None:
                    # Import our conversion function
                    import sys
                    import os
                    sys.path.append(os.path.join(os.path.dirname(__file__), '../../..'))
                    from convert_structured_rewards import convert_structured_to_token_level
                    
                    # PPO: Convert with final-token placement for consistent reward scaling
                    # Note: GRPO uses different structured advantage computation and is unaffected
                    token_level_rewards = convert_structured_to_token_level(
                        structured_rewards=structured_rewards,
                        response_mask=response_mask,
                        turn_sequence_tensor=turn_sequence_tensor,
                        distribution_strategy="final_token_only"
                    )
                    
                    print(f"[GAE-PPO] Converted {len(structured_rewards)} structured rewards to token-level using final_token_only strategy")
                else:
                    # PPO-specific fallback: final token placement (matching FormatRewardManager)
                    # Note: GRPO framework uses different reward processing and is unaffected
                    print(f"[GAE-PPO] Warning: No turn_sequence_tensor available, using final-token placement (PPO-only fix)")
                    batch_size, seq_len = response_mask.shape
                    token_level_rewards = torch.zeros_like(response_mask, dtype=torch.float32)
                    
                    for i, reward_dict in enumerate(structured_rewards):
                        # Normalize turn-wise rewards to avoid length bias in PPO
                        turn_rewards_dict = reward_dict.get("turn_rewards", {})
                        if turn_rewards_dict:
                            turn_reward_mean = sum(turn_rewards_dict.values()) / len(turn_rewards_dict)
                        else:
                            turn_reward_mean = 0.0
                            
                        global_reward_sum = sum(v for k, v in reward_dict.get("global_rewards", {}).items() 
                                              if not k.startswith('_raw_'))
                        # PPO normalization: global + mean(turn-wise) to prevent length bias
                        total_reward = turn_reward_mean + global_reward_sum
                        
                        # PPO: Place full reward on final valid token (FormatRewardManager style)
                        # This provides consistent reward scaling for GAE unlike distributed placement
                        valid_positions = (response_mask[i] == 1).nonzero(as_tuple=True)[0]
                        if len(valid_positions) > 0:
                            final_token_pos = valid_positions[-1]  # Last valid position
                            token_level_rewards[i, final_token_pos] = total_reward
                
                data.batch["token_level_rewards"] = token_level_rewards
            elif "token_level_scores" in data.batch:
                # Legacy format: copy token_level_scores to token_level_rewards
                data.batch["token_level_rewards"] = data.batch["token_level_scores"]
            else:
                raise ValueError("No structured rewards or token_level_scores found for GAE computation")
        else:
            print("[GAE-STRUCTURED] Using existing token_level_rewards (already processed by KL penalty)")
        
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Check if multi-turn advantage calculation is enabled
        if "turn_sequence_tensor" in data.batch and kwargs.get("enable_multiturn_advantage", False):
            # Multi-turn GRPO advantage calculation
            grpo_calculation_mask = data.batch["response_mask"]
            if multi_turn:
                # If multi-turn, replace the mask with the relevant part of loss_mask
                response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
                grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
                
            # Check if we have structured rewards (new format) or token_level_rewards (old format)
            if "structured_rewards" in data.non_tensor_batch:
                structured_rewards = data.non_tensor_batch["structured_rewards"]
            else:
                raise ValueError("No structured rewards found in batch.")

            # Choose between multiturn and uniform advantage calculation
            if kwargs.get("uniform_reward_mode", False):
                advantages, returns = core_algos.compute_grpo_uniform_advantage(
                    structured_rewards=structured_rewards,
                    response_mask=grpo_calculation_mask,
                    turn_sequence_tensor=data.batch["turn_sequence_tensor"],
                    index=data.non_tensor_batch["uid"],
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )
            else:
                advantages, returns = core_algos.compute_grpo_multiturn_advantage(
                    structured_rewards=structured_rewards,
                    response_mask=grpo_calculation_mask,
                    turn_sequence_tensor=data.batch["turn_sequence_tensor"],
                    index=data.non_tensor_batch["uid"],
                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                )
            
            # Test1: Debug alignment between turn_sequence_tensor and response_mask
            turn_sequence_tensor = data.batch["turn_sequence_tensor"]
            batch_size = turn_sequence_tensor.shape[0]
            
            debug_verbose = False  # Set to True to enable debug output
            if debug_verbose:
                print(f"\n[DEBUG TEST1] Turn-Response Mask Alignment Check:")
                print("=" * 60)
                
                for debug_idx in [0, batch_size - 1]:  # First and last samples
                    if debug_idx < batch_size:
                        sample_turn_tensor = turn_sequence_tensor[debug_idx]
                        sample_response_mask = grpo_calculation_mask[debug_idx]
                        
                        # Extract response portion of turn tensor to match response mask dimensions
                        response_length = sample_response_mask.shape[0]
                        sample_turn_response = sample_turn_tensor[-response_length:]
                        
                        # Check if action tokens (turn > 0) match response tokens (mask == 1)
                        action_mask = (sample_turn_response > 0)
                        response_bool_mask = (sample_response_mask == 1)
                        
                        # Check alignment
                        matches = (action_mask == response_bool_mask)
                        mismatch_count = (~matches).sum().item()
                        total_tokens = matches.numel()
                        
                        print(f"\n🔍 SAMPLE {debug_idx} (batch_size={batch_size}):")
                        print(f"  Full turn tensor length: {sample_turn_tensor.shape[0]}")
                        print(f"  Response portion:        {sample_turn_response.tolist()}")
                        print(f"  Response mask:           {sample_response_mask.tolist()}")
                        print(f"  Action mask:             {action_mask.int().tolist()}")
                        print(f"  Response bool:           {response_bool_mask.int().tolist()}")
                        print(f"  Alignment:       {matches.int().tolist()}")
                        print(f"  Mismatches:      {mismatch_count}/{total_tokens} tokens")
                        
                        if mismatch_count > 0:
                            print(f"  ❌ MISMATCH DETECTED!")
                            # Show positions of mismatches
                            mismatch_positions = torch.where(~matches)[0].tolist()
                            print(f"  Mismatch positions: {mismatch_positions}")
                            for pos in mismatch_positions[:5]:  # Show first 5 mismatches
                                print(f"    Position {pos}: turn={sample_turn_response[pos].item()}, response={sample_response_mask[pos].item()}")
                        else:
                            print(f"  ✅ PERFECT ALIGNMENT!")
                
                print("=" * 60)
            
            # Test2: Decode and print actual token sequences for each turn
            if debug_verbose:
                print(f"\n[DEBUG TEST2] Token Sequence Decoding:")
                print("=" * 60)
                if tokenizer is not None:
                    print("📝 Decoding token IDs to actual text content")
                else:
                    print("📝 NOTE: Tokenizer not available, showing raw token IDs only")
            
                for debug_idx in [0, batch_size - 1]:  # First and last samples
                    if debug_idx < batch_size:
                        sample_turn_tensor = turn_sequence_tensor[debug_idx]
                        sample_response_mask = grpo_calculation_mask[debug_idx]
                        
                        # Extract response portion of turn tensor to match response mask dimensions
                        response_length = sample_response_mask.shape[0]
                        sample_turn_response = sample_turn_tensor[-response_length:]
                        
                        # Get input_ids and responses if available
                        if "input_ids" in data.batch:
                            input_ids = data.batch["input_ids"][debug_idx]
                            print(f"\n🔍 SAMPLE {debug_idx} Token Sequences:")
                            print(f"  Input IDs:       {input_ids.tolist()}")
                            
                            # Decode input text if tokenizer available
                            if tokenizer is not None:
                                try:
                                    input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
                                    print(f"  Input Text:      '{input_text}'")
                                except Exception as e:
                                    print(f"  Input Text:      [Decode error: {e}]")
                        
                        if "responses" in data.batch:
                            responses = data.batch["responses"][debug_idx]
                            print(f"  Response IDs:    {responses.tolist()}")
                            
                            # Decode response text if tokenizer available
                            if tokenizer is not None:
                                try:
                                    response_text = tokenizer.decode(responses, skip_special_tokens=True)
                                    print(f"  Response Text:   '{response_text}'")
                                except Exception as e:
                                    print(f"  Response Text:   [Decode error: {e}]")
                        
                        # Show turn-wise token grouping for response portion
                        unique_turns = torch.unique(sample_turn_response[sample_turn_response > 0])
                        print(f"  Turn breakdown (response portion):")
                        for turn_idx in unique_turns:
                            turn_mask = (sample_turn_response == turn_idx)
                            turn_positions = torch.where(turn_mask)[0].tolist()
                            
                            # Get the actual tokens for this turn from responses
                            if "responses" in data.batch:
                                turn_token_ids = data.batch["responses"][debug_idx][turn_mask]
                                print(f"    Turn {turn_idx.item()}: positions {turn_positions}")
                                print(f"      Token IDs: {turn_token_ids.tolist()}")
                                
                                # Decode turn text if tokenizer available  
                                if tokenizer is not None:
                                    try:
                                        turn_text = tokenizer.decode(turn_token_ids, skip_special_tokens=True)
                                        print(f"      Turn Text: '{turn_text}'")
                                    except Exception as e:
                                        print(f"      Turn Text: [Decode error: {e}]")
                            else:
                                print(f"    Turn {turn_idx.item()}: positions {turn_positions}")
                
                print("=" * 60)
            
            # Test3: Check advantages != 0 corresponds to response mask
            if debug_verbose:
                print(f"\n[DEBUG TEST3] Advantages-Response Mask Correspondence:")
                print("=" * 60)
            
                for debug_idx in [0, batch_size - 1]:  # First and last samples
                    if debug_idx < batch_size:
                        sample_advantages = advantages[debug_idx]
                        sample_response_mask = grpo_calculation_mask[debug_idx]
                        
                        # Check where advantages are non-zero
                        nonzero_adv_mask = (sample_advantages != 0)
                        response_bool_mask = (sample_response_mask == 1)
                        
                        # Check correspondence
                        corresponds = (nonzero_adv_mask == response_bool_mask)
                        mismatch_count = (~corresponds).sum().item()
                        total_tokens = corresponds.numel()
                        
                        print(f"\n🔍 SAMPLE {debug_idx} Advantages-Response Correspondence:")
                        print(f"  Advantages:      {sample_advantages.tolist()}")
                        print(f"  Response mask:   {sample_response_mask.tolist()}")
                        print(f"  Nonzero adv:     {nonzero_adv_mask.int().tolist()}")
                        print(f"  Response bool:   {response_bool_mask.int().tolist()}")
                        print(f"  Correspondence:  {corresponds.int().tolist()}")
                        print(f"  Mismatches:      {mismatch_count}/{total_tokens} tokens")
                        
                        if mismatch_count > 0:
                            print(f"  ❌ MISMATCH DETECTED!")
                            # Show positions of mismatches
                            mismatch_positions = torch.where(~corresponds)[0].tolist()
                            print(f"  Mismatch positions: {mismatch_positions}")
                            for pos in mismatch_positions[:5]:  # Show first 5 mismatches
                                print(f"    Position {pos}: adv={sample_advantages[pos].item():.4f}, response={sample_response_mask[pos].item()}")
                        else:
                            print(f"  ✅ PERFECT CORRESPONDENCE!")
                
                print("=" * 60)
        else:
            # Standard single-turn GRPO advantage calculation
            grpo_calculation_mask = data.batch["response_mask"]
            if multi_turn:
                # If multi-turn, replace the mask with the relevant part of loss_mask
                response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
                grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
            # Call compute_grpo_outcome_advantage with parameters matching its definition
            advantages, returns = core_algos.compute_grpo_outcome_advantage(
                token_level_rewards=data.batch["token_level_rewards"],
                response_mask=grpo_calculation_mask,
                index=data.non_tensor_batch["uid"],
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )

        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.OPO:
        advantages, returns = core_algos.compute_opo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        raise NotImplementedError
    return data


class RayPPOTrainer:
    """
    PPO Trainer with Knowledge Graph capabilities and external prompt augmentation control.
    
    This trainer extends the base PPO functionality with:
    1. External control of RL dataset's prompt augmentation
    2. Search-augmented generation for KG reasoning
    3. State masking for multi-turn conversations
    
    Key Design for Prompt Augmentation:
    -----------------------------------
    The trainer does NOT perform prompt augmentation internally. Instead, it controls
    the RL dataset's prompt augmentation by calling `set_current_step()` on the dataset.
    
    The RL dataset is responsible for:
    1. Applying prompt augmentation based on its configuration
    2. Re-tokenizing augmented prompts to update input_ids, attention_mask, and position_ids
    3. Managing hint scheduling based on the current training step
    
    This ensures clean separation of concerns where the trainer orchestrates training
    and the dataset handles data preprocessing including prompt augmentation.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
    ):
        """Initialize distributed PPO trainer with Ray backend."""

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn



        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0
        
        # Check if KG generation is enabled (following same pattern as search trainer)
        search_config = getattr(config.actor_rollout_ref.rollout, 'search', None)
        self.use_search_generation = search_config is not None and search_config.get('enable', False)
        
        if self.use_search_generation:
            self.search_config = search_config
            
            # For KG generation with GRPO, we use trainer-side batch expansion
            grpo_rollout_n = getattr(config.actor_rollout_ref.rollout, 'grpo_rollout_n', None)
            original_rollout_n = config.actor_rollout_ref.rollout.n
            
            # Determine effective rollout_n (fallback to original rollout.n if grpo_rollout_n doesn't exist)
            effective_rollout_n = grpo_rollout_n if grpo_rollout_n is not None else original_rollout_n
            
            if effective_rollout_n > 1:
                pass  # GRPO batch expansion will be handled in training loop
            else:
                pass  # No batch expansion needed

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.OPO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            assert n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0, f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            megatron_dp = n_gpus // (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size)
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        # Use grpo_rollout_n for batch size calculation since that's what determines the actual batch expansion
        grpo_rollout_n = getattr(config.actor_rollout_ref.rollout, 'grpo_rollout_n', config.actor_rollout_ref.rollout.n)
        real_train_batch_size = config.data.train_batch_size * grpo_rollout_n
        assert real_train_batch_size % minimal_bsz == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size ({minimal_bsz})"

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor": "micro_batch_size",
                "critic": "micro_batch_size",
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size", 
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )

            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
            "unbiased-fixed-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"

        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"

        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)


    def _validate(self):
        # Update the validation dataset's current step for prompt augmentation control
        if hasattr(self.val_dataset, 'set_current_step'):
            self.val_dataset.set_current_step(self.global_steps)

        data_source_lst = []
        reward_extra_infos_dict = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            # Extract meta_info for KG queries
            meta_info = {}
            if "sample_id" in test_data:
                if isinstance(test_data["sample_id"], np.ndarray):
                    meta_info["sample_ids"] = test_data["sample_id"].tolist()
                else:
                    meta_info["sample_ids"] = [test_data["sample_id"]]
            if "dataset_name" in test_data:
                if isinstance(test_data["dataset_name"], np.ndarray):
                    dataset_names = test_data["dataset_name"].tolist()
                    # Fix lowercase cwq to uppercase CWQ
                    meta_info["dataset_names"] = ["CWQ" if ds == "cwq" else ds for ds in dataset_names]
                else:
                    dataset_name = test_data["dataset_name"]
                    # Fix lowercase cwq to uppercase CWQ
                    meta_info["dataset_names"] = ["CWQ" if dataset_name == "cwq" else dataset_name]
            
            test_batch = DataProto.from_single_dict(test_data, meta_info=meta_info)
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # Control prompt augmentation in the dataset - no trainer-side augmentation needed
            # The dataset handles augmentation based on its configuration

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
                
            # CRITICAL: Add sample_id and dataset_name to the pop list so they get included in test_gen_batch
            # The pop() method returns a NEW DataProto with ONLY the popped keys
            if "sample_id" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("sample_id")
            if "dataset_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("dataset_name")
                
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            # Extract per-sample information for KG queries AFTER repeat operation
            # Note: test_batch was already repeated, so we extract from the repeated batch
            sample_ids = []
            dataset_names = []
            
            # Now extract from test_gen_batch since that's where the fields are preserved after pop
            if "sample_id" in test_gen_batch.non_tensor_batch:
                sample_ids = test_gen_batch.non_tensor_batch["sample_id"].tolist()
            else:
                sample_ids = []
                
            if "dataset_name" in test_gen_batch.non_tensor_batch:
                # Use explicit dataset_name from extra_info
                dataset_names = test_gen_batch.non_tensor_batch["dataset_name"].tolist()
            elif "data_source" in test_gen_batch.non_tensor_batch:
                # Fallback: map data_source to dataset_name
                data_sources = test_gen_batch.non_tensor_batch["data_source"]
                for ds in data_sources:
                    if ds in ["webqsp_kg", "webqsp"]:
                        dataset_names.append("webqsp")
                    elif ds in ["cwq_kg", "cwq"]:
                        dataset_names.append("CWQ")
                    else:
                        dataset_names.append("webqsp")  # fallback
            else:
                # Use webqsp as default fallback
                dataset_names = ["webqsp"] * len(test_gen_batch.batch)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": True,
                "validate": True,
            }
            
            # Add per-sample information if available
            if test_gen_batch.meta_info is None:
                test_gen_batch.meta_info = {}
            if sample_ids:
                test_gen_batch.meta_info["sample_ids"] = sample_ids
            if dataset_names:
                test_gen_batch.meta_info["dataset_names"] = dataset_names
                
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            
            # If we padded the batch, we need to pad the sample_ids and dataset_names too
            if pad_size > 0 and sample_ids:
                # Pad sample_ids by repeating the last element
                sample_ids.extend([sample_ids[-1]] * pad_size)
                print(f"[INFO] Padded sample_ids from {len(sample_ids)-pad_size} to {len(sample_ids)} samples")
            
            if pad_size > 0 and dataset_names:
                # Pad dataset_names by repeating the last element
                dataset_names.extend([dataset_names[-1]] * pad_size)
            
            # Update the meta_info with padded lists
            if test_gen_batch_padded.meta_info is None:
                test_gen_batch_padded.meta_info = {}
            if sample_ids:
                test_gen_batch_padded.meta_info["sample_ids"] = sample_ids
            if dataset_names:
                test_gen_batch_padded.meta_info["dataset_names"] = dataset_names
            
            if not self.use_search_generation:
                # Use standard generation for validation
                if not self.async_rollout_mode:
                    test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                else:
                    self.async_rollout_manager.wake_up()
                    test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
                    self.async_rollout_manager.sleep()
            else:
                # Use search-augmented generation for validation - direct integration approach
                first_input_ids = test_gen_batch_padded.batch['input_ids'][:, -self.generation_manager.config.max_start_length:].clone().long()
                test_output_gen_batch_padded = self.generation_manager.run_llm_loop(
                    gen_batch=test_gen_batch_padded,
                    initial_input_ids=first_input_ids,
                )

            # DEBUG: Print meta_info during validation to understand truncation behavior
            if hasattr(test_output_gen_batch_padded, 'meta_info') and test_output_gen_batch_padded.meta_info:
                meta_info = test_output_gen_batch_padded.meta_info
                print(f"\n=== DEBUG VALIDATION META_INFO ===")
                
                # Print basic stats
                if 'turns_stats' in meta_info:
                    print(f"Turns stats: {meta_info['turns_stats']}")
                if 'valid_action_stats' in meta_info:
                    print(f"Valid action stats: {meta_info['valid_action_stats']}")
                
                # Print interaction history (the key debug information)
                if 'interaction_history' in meta_info:
                    history = meta_info['interaction_history']
                    
                    # Print actions per turn
                    if 'actions' in history:
                        print(f"Actions per turn: {history['actions']}")
                    
                    # Print search results (truncated for readability)
                    if 'search_results' in history:
                        for turn_idx, turn_results in enumerate(history['search_results']):
                            print(f"Turn {turn_idx} search results count: {len(turn_results)}")
                            for sample_idx, result in enumerate(turn_results[:3]):  # Show first 3 samples
                                result_preview = result[:100] + "..." if len(result) > 100 else result
                                print(f"  Sample {sample_idx}: {repr(result_preview)}")
                
                print(f"=== END DEBUG META_INFO ===\n")

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            # Convert to long integers if they're floats (can happen with search generation)
            if output_ids.dtype != torch.long:
                output_ids = output_ids.long()
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            
            # Handle missing reward_tensor by parsing structured_rewards
            if "reward_tensor" in result:
                reward_tensor = result["reward_tensor"]
                scores = reward_tensor.sum(-1).cpu().tolist() if reward_tensor.dim() > 1 else reward_tensor.cpu().tolist()
            elif "structured_rewards" in result:
                # Extract total_scores directly from structured_rewards
                scores = [reward_dict.get("total_score", 0.0) for reward_dict in result["structured_rewards"]]
            else:
                raise ValueError(f"Neither 'reward_tensor' nor 'structured_rewards' found in result: {list(result.keys())}")
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(scores)))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        
        # Define core metrics that should have their means logged to val-core
        core_metrics = {
            'exact_match', 'retrieval_quality', 'total_score',
            'turn_format_score', 'turn_kg_query_validity',
            'exact_match_binary', 'f1', 'precision', 'recall',
            'entity_f1', 'entity_precision', 'entity_recall'
        }
        
        for data_source, var2metric2val in data_src2var2metric2val.items():
            for var_name, metric2val in var2metric2val.items():
                # Find the maximum sample size (the overall mean across all samples)
                max_n = 0
                for metric_name in metric2val.keys():
                    if metric_name.startswith('mean@'):
                        n = int(metric_name.split('@')[1])
                        max_n = max(max_n, n)
                
                for metric_name, metric_val in metric2val.items():
                    should_log = False
                    
                    if var_name == 'exact_match':
                        # exact_match only needs mean@1
                        if metric_name == 'mean@1':
                            should_log = True
                    elif var_name == 'retrieval_quality':
                        # retrieval_quality only needs mean (not mean@4)
                        if metric_name == 'mean@1':
                            should_log = True
                    else:
                        # Other metrics: only log the overall mean across all samples
                        if metric_name == f'mean@{max_n}':
                            should_log = True
                    
                    if should_log:
                        # Core metrics go to val-core, others to val-aux
                        if var_name in core_metrics:
                            metric_sec = "val-core"
                        else:
                            metric_sec = "val-aux"
                        
                        # Rename mean@1 to mean for cleaner logging
                        display_metric_name = metric_name
                        if metric_name == 'mean@1':
                            display_metric_name = 'mean'
                        
                        pfx = f"{metric_sec}/{data_source}/{var_name}/{display_metric_name}"
                        metric_dict[pfx] = metric_val

        return metric_dict

    def _process_rollout_metrics(self, reward_extra_infos_dict: dict[str, list]) -> dict[str, float]:
        """
        Process rollout metrics from reward_extra_infos_dict into rl-core and rl-aux metrics.
        
        Args:
            reward_extra_infos_dict: Dict mapping metric names to lists of values from rollout
            
        Returns:
            Dict mapping metric names to aggregated values for WandB logging
        """
        from collections import defaultdict
        import numpy as np
        
        # Define core metrics that should be logged to rl-core
        core_metrics = {
            'exact_match', 'retrieval_quality', 'total_score',
            'turn_format_score', 'turn_kg_query_validity',
            'exact_match_binary', 'f1', 'precision', 'recall',
            'entity_f1', 'entity_precision', 'entity_recall'
        }
        
        rollout_metrics = {}
        
        for var_name, values in reward_extra_infos_dict.items():
            if not values:
                continue
                
            # Skip the generic "reward" key since we have more specific metrics
            if var_name == "reward":
                continue
                
            # Compute simple statistics
            mean_val = np.mean(values)
            
            # For exact_match and retrieval_quality: compute simple mean
            if var_name in ['exact_match', 'retrieval_quality'] and len(values) >= 1:
                values_array = np.array(values)
                mean_val = np.mean(values_array)
                
                # Determine metric section
                if var_name in core_metrics:
                    metric_sec = "rl-core"
                else:
                    metric_sec = "rl-aux"
                
                # Log mean for both metrics
                rollout_metrics[f"{metric_sec}/rollout/{var_name}/mean"] = mean_val
            else:
                # Other metrics: just overall mean
                if var_name in core_metrics:
                    metric_sec = "rl-core"
                else:
                    metric_sec = "rl-aux"
                
                rollout_metrics[f"{metric_sec}/rollout/{var_name}/mean"] = mean_val
        
        return rollout_metrics

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # Initialize generation manager for KG search if enabled (following search trainer pattern)
        if self.use_search_generation:
            from kg_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
            gen_config = GenerationConfig(
                max_turns=self.search_config.get('max_turns', 6),  # KG default: 6 turns
                max_start_length=self.config.data.get('max_start_length', 1000),  # KG default: 1000
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.get('max_obs_length', 650),  # KG default: 650
                num_gpus=self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes,
                no_think_rl=self.config.algorithm.get('no_think_rl', False),
                search_url=self.search_config.get('search_url', 'http://127.0.0.1:8001/retrieve'),  # KG server default
                topk=self.search_config.get('topk', 5),  # KG default: 5
            )
            
            self.generation_manager = LLMGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=self.actor_rollout_wg,
                config=gen_config,
            )

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1

        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)

        # save dataloader
        BaseCheckpointManager.local_mkdir(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _create_loss_mask(self, batch, metrics):
        """Create loss mask for state tokens.
        
        This method creates a loss mask based on the info_mask to ensure that
        only certain tokens (typically state tokens) contribute to the loss calculation.
        This is useful for multi-turn conversations where we want to mask out
        information tokens that shouldn't contribute to the policy gradient loss.
        
        Args:
            batch (DataProto): The batch containing the data
            metrics (dict): Dictionary to store metrics
            
        Returns:
            tuple: (batch, metrics) with loss_mask added to batch and metrics updated
        """
        # Validation: ensure info_mask exists when state_masking is enabled
        if 'info_mask' not in batch.batch:
            raise ValueError(
                "state_masking=true requires 'info_mask' in batch, but it's missing. "
                "Make sure your generation process creates info_mask for search-augmented responses."
            )
        
        response_length = batch.batch['responses'].shape[-1]
        response_mask = batch.batch['attention_mask'][:, -response_length:]
        
        # Use info_mask to create loss_mask for response tokens only
        loss_mask = batch.batch['info_mask'][:, -response_length:]
        batch.batch['loss_mask'] = loss_mask
        

        # Calculate coverage metrics
        total_state_tokens = loss_mask.sum().item()
        total_response_tokens = response_mask.sum().item()
        coverage = total_state_tokens / total_response_tokens if total_response_tokens > 0 else 0.0

        metrics.update({
            'state_tokens/total': total_state_tokens,
            'state_tokens/coverage': coverage,
        })
        

        
        return batch, metrics

    def fit(self):
        """
        The training loop of PPO with Knowledge Graph capabilities.
        
        SAMPLE_ID HANDLING:
        ===================
        The sample_id should flow through the pipeline as follows:
        1. JSON data (e.g., {"id": "WebQTest-0", "question": "...", ...})
        2. Data processing script extracts 'id' field and stores as 'sample_id' in extra_info
        3. Dataset extracts sample_id from extra_info and stores as top-level field
        4. Collate function converts sample_id to numpy array for batching
        5. DataProto.from_single_dict categorizes numpy arrays as non-tensors
        6. Trainer should find sample_id in batch.non_tensor_batch["sample_id"]
        7. **CRITICAL**: Pop operation must preserve sample_id and dataset_name in gen_batch
        8. DataProto.repeat() will properly repeat these fields for GRPO
        9. LLMGenerationManager receives sample_ids in meta_info for KG queries
        
        RECENT FIX:
        ===========
        The key issue was that the pop operation was removing sample_id and dataset_name 
        from the batch when creating gen_batch for generation. This caused the 
        LLMGenerationManager to fall back to generating "fallback_sample_000000" IDs
        instead of using the authentic sample IDs like "WebQTrn-643" from the dataset.
        
        The fix ensures that sample_id and dataset_name are preserved in gen_batch
        by NOT including them in the batch_keys_to_pop or non_tensor_batch_keys_to_pop lists.
        
        CORRECTED UNDERSTANDING:
        =======================
        DataProto.pop() removes keys from the original DataProto and returns a NEW DataProto
        with ONLY the popped keys. The gen_batch only contains keys that were explicitly
        included in batch_keys_to_pop and non_tensor_batch_keys_to_pop lists.
        
        The actual fix is to ADD sample_id and dataset_name TO the non_tensor_batch_keys_to_pop
        list so they get included in the gen_batch that's passed to generation.
        
        CLEAN TRAINING LOOP:
===================
The trainer handles KG-augmented training with minimal logging to avoid spam.
        
        This helps monitor the quality of knowledge graph queries during training.
        
        The trainer loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                
                # Extract meta_info for KG queries
                meta_info = {}
                if "sample_id" in batch_dict:
                    if isinstance(batch_dict["sample_id"], np.ndarray):
                        meta_info["sample_ids"] = batch_dict["sample_id"].tolist()
                    else:
                        meta_info["sample_ids"] = [batch_dict["sample_id"]]
                if "dataset_name" in batch_dict:
                    if isinstance(batch_dict["dataset_name"], np.ndarray):
                        dataset_names = batch_dict["dataset_name"].tolist()
                        # Fix lowercase cwq to uppercase CWQ
                        meta_info["dataset_names"] = ["CWQ" if ds == "cwq" else ds for ds in dataset_names]
                    else:
                        dataset_name = batch_dict["dataset_name"]
                        # Fix lowercase cwq to uppercase CWQ
                        meta_info["dataset_names"] = ["CWQ" if dataset_name == "cwq" else dataset_name]
                
                batch: DataProto = DataProto.from_single_dict(batch_dict, meta_info=meta_info)

                # Update the dataset's current step for prompt augmentation control
                if hasattr(self.train_dataset, 'set_current_step'):
                    self.train_dataset.set_current_step(self.global_steps)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                    
                # CRITICAL: Add sample_id and dataset_name to the pop list so they get included in gen_batch
                # The pop() method returns a NEW DataProto with ONLY the popped keys
                if "sample_id" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("sample_id")
                if "dataset_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("dataset_name")
                    
                # Pop keys needed for generation, including meta_info
                meta_info_keys_to_pop = []
                if "sample_ids" in batch.meta_info:
                    meta_info_keys_to_pop.append("sample_ids")
                if "dataset_names" in batch.meta_info:
                    meta_info_keys_to_pop.append("dataset_names")
                    
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                    meta_info_keys=meta_info_keys_to_pop,
                )

                # Set basic meta_info for generation (preserve existing meta_info)
                if self.use_search_generation:
                    if gen_batch.meta_info is None:
                        gen_batch.meta_info = {}
                    gen_batch.meta_info.update({
                        "eos_token_id": self.tokenizer.eos_token_id,
                        "pad_token_id": self.tokenizer.pad_token_id,
                        "recompute_log_prob": False,
                        "do_sample": True,  # Training uses sampling
                    })

                is_last_step = self.global_steps >= self.total_training_steps

                with _timer("step", timing_raw):
                    # Set up UIDs for GRPO grouping and expand gen_batch before generation
                    # Use separate parameter for trainer-side rollout expansion (for GRPO grouping)
                    # This is decoupled from VLLM's rollout.n parameter
                    grpo_rollout_n = getattr(self.config.actor_rollout_ref.rollout, 'grpo_rollout_n', 
                                            self.config.actor_rollout_ref.rollout.n)
                    
                    # Generate UIDs for GRPO grouping - same UID for all responses from same question
                    base_uids = [str(uuid.uuid4()) for _ in range(len(gen_batch.batch))]
                    gen_batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
                    
                    # Expand gen_batch before generation for GRPO
                    expanded_gen_batch = gen_batch.repeat(repeat_times=grpo_rollout_n, interleave=True)

                    # Initialize meta_info if it's None
                    if expanded_gen_batch.meta_info is None:
                        expanded_gen_batch.meta_info = {}

                    # Extract per-sample information for KG queries AFTER repeat operation
                    # since DataProto.repeat() doesn't repeat meta_info
                    if self.use_search_generation:
                        sample_ids = []
                        dataset_names = []
                        
                        # DataProto.repeat() should properly handle sample_id in non_tensor_batch
                        # Now that we preserve sample_id in gen_batch, it should be available after repeat()
                        if "sample_id" in expanded_gen_batch.non_tensor_batch:
                            sample_ids = expanded_gen_batch.non_tensor_batch["sample_id"].tolist()
                        else:
                            raise ValueError(
                                "sample_id is missing in gen_batch after repeat(). "
                                "Ensure that sample_id is included in the non_tensor_batch_keys_to_pop list during pop()."
                            )
                        
                        if "dataset_name" in expanded_gen_batch.non_tensor_batch:
                            # Use explicit dataset_name from extra_info
                            dataset_names = expanded_gen_batch.non_tensor_batch["dataset_name"].tolist()
                        elif "data_source" in expanded_gen_batch.non_tensor_batch:
                            # Fallback: map data_source to dataset_name
                            data_sources = expanded_gen_batch.non_tensor_batch["data_source"]
                            for ds in data_sources:
                                if ds in ["webqsp_kg", "webqsp"]:
                                    dataset_names.append("webqsp")
                                elif ds in ["cwq_kg", "cwq"]:
                                    dataset_names.append("CWQ")
                                else:
                                    dataset_names.append("webqsp")  # fallback
                        else:
                            # Use webqsp as default fallback
                            dataset_names = ["webqsp"] * len(expanded_gen_batch.batch)
                        
                        # Add per-sample information to meta_info AFTER repeat
                        if sample_ids:
                            expanded_gen_batch.meta_info["sample_ids"] = sample_ids
                        if dataset_names:
                            expanded_gen_batch.meta_info["dataset_names"] = dataset_names
                    
                    # generate a batch
                    with _timer("gen", timing_raw):
                        if not self.use_search_generation:
                            if not self.async_rollout_mode:
                                gen_batch_output = self.actor_rollout_wg.generate_sequences(expanded_gen_batch)
                            else:
                                self.async_rollout_manager.wake_up()
                                gen_batch_output = self.async_rollout_manager.generate_sequences(expanded_gen_batch)
                                self.async_rollout_manager.sleep()
                        else:
                            # Use search-augmented generation with LLMGenerationManager
                            first_input_ids = expanded_gen_batch.batch['input_ids'][:, -self.generation_manager.config.max_start_length:].clone().long()
                            self.generation_manager.timing_raw = timing_raw
                            gen_batch_output = self.generation_manager.run_llm_loop(
                                gen_batch=expanded_gen_batch,
                                initial_input_ids=first_input_ids,
                            )

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            if gen_baseline_batch.meta_info is None:
                                gen_baseline_batch.meta_info = {}
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    # Expand the original batch to match the expanded gen_batch
                    # Use the same UIDs that were used for generation
                    batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
                    batch = batch.repeat(repeat_times=grpo_rollout_n, interleave=True)
                    
                    batch = batch.union(gen_batch_output)

                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    #if self.config.trainer.balance_batch:
                    #    self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    if batch.meta_info is None:
                        batch.meta_info = {}
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # Create loss mask for state tokens if state_masking is enabled
                    # This needs to be done early so that entropy loss calculation can use it
                    if (self.use_search_generation and 
                        self.config.actor_rollout_ref.actor.get('state_masking', False)):
                        batch, metrics = self._create_loss_mask(batch, metrics)

                    with _timer("reward", timing_raw):
                        # compute reward model score
                        if self.use_rm:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                        else:
                            reward_data_proto, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch = batch.union(reward_data_proto)

                    # recompute old_log_probs
                    with _timer("old_log_prob", timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        
                        # Use loss_mask if state_masking is enabled, otherwise use response_mask
                        # Note: entropys is computed on response tokens only, so we need to ensure the mask matches
                        if (self.use_search_generation and 
                            self.config.actor_rollout_ref.actor.get('state_masking', False) and
                            'loss_mask' in batch.batch):
                            # loss_mask is for the full sequence, but entropys is for response tokens only
                            # We need to extract the response part of loss_mask
                            response_length = batch.batch["responses"].shape[-1]
                            entropy_masks = batch.batch["loss_mask"][:, -response_length:]
                        else:
                            entropy_masks = batch.batch["response_mask"]
                        
                        # Ensure tensor sizes match for entropy calculation
                        if entropys.shape != entropy_masks.shape:
                            # If there's still a size mismatch, truncate or pad to match
                            min_length = min(entropys.shape[-1], entropy_masks.shape[-1])
                            entropys = entropys[:, :min_length]
                            entropy_masks = entropy_masks[:, :min_length]
                        
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_loss = agg_loss(loss_mat=entropys, loss_mask=entropy_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy_loss": entropy_loss.detach().item()}
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer("ref", timing_raw):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer("values", timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer("adv", timing_raw):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_data_proto, reward_extra_infos_dict = ray.get(future_reward)
                            batch = batch.union(reward_data_proto)
                        # Note: reward tensor is now in batch.batch["token_level_scores"] via union

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            # Use multi_turn=True if state_masking is enabled to properly handle loss_mask
                            multi_turn = (self.use_search_generation and 
                                        self.config.actor_rollout_ref.actor.get('state_masking', False))
                            batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, 
                                                                kl_penalty=self.config.algorithm.kl_penalty,
                                                                multi_turn=multi_turn,
                                                                config=self.config,
                                                                tokenizer=self.tokenizer)
                            metrics.update(kl_metrics)
                        else:
                            # Handle both structured rewards (new format) and token_level_scores (legacy format)
                            if "token_level_scores" in batch.batch:
                                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                            # If using structured rewards, token_level_rewards will be created in compute_advantage

                        # compute advantages, executed on the driver process

                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor

                        # Use multi_turn=True if state_masking is enabled to properly handle loss_mask
                        multi_turn_for_advantage = (self.use_search_generation and 
                                                  self.config.actor_rollout_ref.actor.get('state_masking', False))
                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=grpo_rollout_n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            multi_turn=multi_turn_for_advantage,
                            enable_multiturn_advantage=self.config.algorithm.get("enable_multiturn_advantage", False),
                            uniform_reward_mode=self.config.algorithm.get("uniform_reward_mode", False),
                            use_pf_ppo=self.config.algorithm.use_pf_ppo,
                            pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                            tokenizer=self.tokenizer,
                            pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                        )

                    # update critic
                    if self.use_critic:
                        with _timer("update_critic", timing_raw):
                            # Set multi_turn flag to enable state_masking support in critic
                            if batch.meta_info is None:
                                batch.meta_info = {}
                            batch.meta_info["multi_turn"] = (
                                self.use_search_generation and 
                                self.config.actor_rollout_ref.actor.get('state_masking', False)
                            )
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer("update_actor", timing_raw):
                            if batch.meta_info is None:
                                batch.meta_info = {}
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with _timer("dump_rollout_generations", timing_raw):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer("testing", timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and (is_last_step or (self.global_steps > 1 and self.global_steps % self.config.trainer.save_freq == 0)):
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # Add reward metrics from multi-turn reward manager
                if reward_extra_infos_dict:
                    rollout_metrics = self._process_rollout_metrics(reward_extra_infos_dict)
                    metrics.update(rollout_metrics)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return


