# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO
"""

import re
from collections import defaultdict

import numpy as np
import torch

import verl.utils.torch_functional as verl_F


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        pass


def get_kl_controller(kl_ctrl):
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            nextvalues = values[:, t + 1] if t < gen_len - 1 else 0.0
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam = delta + gamma * lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: str = True,
):
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        norm_adv_by_std_in_grpo: (bool)
            whether to scale the GRPO advantage.
            If True, the advantage is scaled by the std, as in the original GRPO.
            If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                # For single rollouts, use the actual reward as advantage
                # instead of setting mean=0 which doesn't make sense
                id2mean[idx] = torch.tensor(0.0)  # Keep mean=0 so advantage = reward
                id2std[idx] = torch.tensor(1.0)   # Keep std=1 so no scaling
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
                id2std[idx] = torch.std(torch.tensor([id2score[idx]]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]

    scores = scores.unsqueeze(-1) * response_mask
    return scores, scores



def compute_grpo_multiturn_advantage(
    structured_rewards: list,
    response_mask: torch.Tensor,
    turn_sequence_tensor: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute multi-turn advantage for GRPO with separate normalization for turn-specific and global rewards.
    
    This function takes structured rewards and assigns them to tokens, then normalizes them independently:
    - Turn-specific advantages: Normalized per turn across rollouts
    - Global advantages: Normalized across all action tokens across rollouts
    
    Args:
        structured_rewards: (list) length bs
            List of reward dictionaries with keys: turn_rewards, global_rewards
        response_mask: (torch.Tensor) shape (bs, seq_len)
            Mask for response tokens
        turn_sequence_tensor: (torch.Tensor) shape (bs, seq_len)
            Turn assignments: -1 for non-action, 1,2,3... for turn numbers
        index: (np.ndarray) shape (bs,)
            Sample indices for GRPO grouping
        epsilon: (float)
            Small value to avoid division by zero
        norm_adv_by_std_in_grpo: (bool)
            Whether to normalize by standard deviation
            
    Returns:
        advantages: (torch.Tensor) shape (bs, seq_len)
            Multi-turn advantages with separate normalization
        returns: (torch.Tensor) shape (bs, seq_len)
            Same as advantages for compatibility
    """
    device = turn_sequence_tensor.device
    batch_size, seq_len = turn_sequence_tensor.shape
    
    # Initialize advantage tensors
    turn_specific_advantages = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    global_advantages = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    
    # Get unique turn numbers (exclude -1)
    unique_turns = torch.unique(turn_sequence_tensor)
    unique_turns = unique_turns[unique_turns > 0]  # Only positive turn numbers
    
    # 1. Collect turn-specific scalar rewards for GRPO normalization
    id2turn_rewards = defaultdict(list)
    id2turn_info = defaultdict(list)  # Track which sample and turn each reward belongs to
    
    with torch.no_grad():
        # Extract scalar turn rewards directly from structured_rewards
        for i in range(batch_size):
            sample_structured_rewards = structured_rewards[i]
            turn_rewards = sample_structured_rewards['turn_rewards']
            
            for turn_num, turn_reward in turn_rewards.items():
                # Collect scalar turn reward for GRPO normalization
                id2turn_rewards[index[i]].append(turn_reward)
                id2turn_info[index[i]].append((i, turn_num))
        
        # Calculate group statistics for turn rewards
        id2mean_turn = {}
        id2std_turn = {}
        
        for idx in id2turn_rewards:
            if len(id2turn_rewards[idx]) == 1:
                id2mean_turn[idx] = torch.tensor(0.0, device=device)
                id2std_turn[idx] = torch.tensor(1.0, device=device)
            elif len(id2turn_rewards[idx]) > 1:
                turn_scores = torch.tensor(id2turn_rewards[idx], device=device)
                id2mean_turn[idx] = turn_scores.mean()
                id2std_turn[idx] = turn_scores.std()
        
        # Normalize and assign turn advantages to all tokens in each turn
        for idx in id2turn_rewards:
            for reward_idx, (sample_idx, turn_num) in enumerate(id2turn_info[idx]):
                turn_reward = id2turn_rewards[idx][reward_idx]
                
                if norm_adv_by_std_in_grpo:
                    turn_advantage = (turn_reward - id2mean_turn[idx]) / (id2std_turn[idx] + epsilon)
                else:
                    turn_advantage = turn_reward - id2mean_turn[idx]
                
                # Apply this advantage to ALL tokens in this turn
                turn_mask = (turn_sequence_tensor[sample_idx] == turn_num)
                turn_specific_advantages[sample_idx, turn_mask] = turn_advantage
    
    # 2. Process global advantages - extract scalar global rewards and normalize
    id2global_rewards = defaultdict(list)
    id2global_info = defaultdict(list)
    
    with torch.no_grad():
        # Extract scalar global rewards directly from structured_rewards
        for i in range(batch_size):
            sample_structured_rewards = structured_rewards[i]
            global_rewards_dict = sample_structured_rewards['global_rewards']
            
            # Sum only the actual weighted reward components (exclude raw scores)
            total_global_reward = sum(v for k, v in global_rewards_dict.items() 
                                    if not k.startswith('_raw_'))
            
            # Collect scalar global reward for GRPO normalization
            id2global_rewards[index[i]].append(total_global_reward)
            id2global_info[index[i]].append(i)
        
        # Calculate group statistics for global rewards
        id2mean_global = {}
        id2std_global = {}
        
        for idx in id2global_rewards:
            if len(id2global_rewards[idx]) == 1:
                id2mean_global[idx] = torch.tensor(0.0, device=device)
                id2std_global[idx] = torch.tensor(1.0, device=device)
            elif len(id2global_rewards[idx]) > 1:
                global_scores = torch.tensor(id2global_rewards[idx], device=device)
                id2mean_global[idx] = global_scores.mean()
                id2std_global[idx] = global_scores.std()
        
        # Normalize and assign global advantages to all action tokens
        for idx in id2global_rewards:
            for reward_idx, sample_idx in enumerate(id2global_info[idx]):
                global_reward = id2global_rewards[idx][reward_idx]
                
                if norm_adv_by_std_in_grpo:
                    global_advantage = (global_reward - id2mean_global[idx]) / (id2std_global[idx] + epsilon)
                else:
                    global_advantage = global_reward - id2mean_global[idx]
                
                # Apply this advantage to ALL action tokens in this sample
                action_mask = turn_sequence_tensor[sample_idx] > 0
                global_advantages[sample_idx, action_mask] = global_advantage
    
    # Combine turn-specific and global advantages
    total_advantages = 0.5*turn_specific_advantages + global_advantages
    
    # Extract response portion of advantages to match response_mask dimensions
    # response_mask covers only the response tokens, but total_advantages covers full sequence
    response_length = response_mask.shape[1]
    response_advantages = total_advantages[:, -response_length:]
    
    # Apply response mask to final advantages
    # response_advantages = response_advantages * response_mask
    
    # Debug printout for first few samples
    if batch_size > 0:
        max_debug_samples = min(2, batch_size)
        print(f"\n[DEBUG] Multi-Turn Advantage Details (showing {max_debug_samples}/{batch_size} samples):")
        print("=" * 80)
        
        for i in range(max_debug_samples):
            sample_turn_tensor = turn_sequence_tensor[i]
            sample_turn_advantages = turn_specific_advantages[i]
            sample_global_advantages = global_advantages[i]
            sample_total_advantages = total_advantages[i]
            
            print(f"\n🧮 SAMPLE {i} ADVANTAGE CALCULATION:")
            print(f"Turn Sequence: {sample_turn_tensor}")
            
            # Show advantage breakdown by component
            action_mask = sample_turn_tensor > 0
            info_mask = sample_turn_tensor == -1
            
            if action_mask.any():
                action_positions = torch.where(action_mask)[0]
                print(f"\n📊 Action Token Analysis:")
                print(f"  Positions: {action_positions[:15].tolist()}")
                print(f"  Turn-specific advantages: {sample_turn_advantages[action_positions][:15].tolist()}")
                print(f"  Global advantages: {sample_global_advantages[action_positions][:15].tolist()}")
                print(f"  Total advantages: {sample_total_advantages[action_positions][:15].tolist()}")
            
            if info_mask.any():
                info_positions = torch.where(info_mask)[0]
                info_advantages = sample_total_advantages[info_positions]
                print(f"\n📝 Info Token Check:")
                print(f"  Positions: {info_positions[:10].tolist()}")
                print(f"  Advantages (should be 0): {info_advantages[:10].tolist()}")
                print(f"  Non-zero info advantages: {(info_advantages != 0).sum().item()}")
            
            # Turn-by-turn breakdown
            unique_turns = torch.unique(sample_turn_tensor)
            unique_turns = unique_turns[unique_turns > 0]
            
            print(f"\n🔄 Turn-by-Turn Advantage Summary:")
            for turn_num in unique_turns:
                turn_mask = (sample_turn_tensor == turn_num)
                turn_positions = torch.where(turn_mask)[0]
                turn_advantages = sample_total_advantages[turn_mask]
                
                print(f"  Turn {turn_num.item()}: {len(turn_positions)} tokens, "
                      f"avg_adv={turn_advantages.mean():.4f}, "
                      f"std_adv={turn_advantages.std():.4f}")
            
            print(f"\n📈 Sample Summary:")
            print(f"  Total advantage sum: {sample_total_advantages.sum():.4f}")
            print(f"  Non-zero advantages: {(sample_total_advantages != 0).sum().item()}")
            print("-" * 40)
    
    return response_advantages, response_advantages


def compute_grpo_uniform_advantage(
    structured_rewards: list,
    response_mask: torch.Tensor,
    turn_sequence_tensor: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute uniform advantage for GRPO by summing all turn-wise and global rewards into a single reward.
    
    This function takes structured rewards, sums all components (turn + global), and applies
    the unified reward uniformly to all action tokens, losing turn-specific credit assignment.
    
    Args:
        structured_rewards: (list) length bs
            List of reward dictionaries with keys: turn_rewards, global_rewards
        response_mask: (torch.Tensor) shape (bs, seq_len)
            Mask for response tokens
        turn_sequence_tensor: (torch.Tensor) shape (bs, seq_len)
            Turn assignments: -1 for non-action, 1,2,3... for turn numbers
        index: (np.ndarray) shape (bs,)
            Sample indices for GRPO grouping
        epsilon: (float)
            Small value to avoid division by zero
        norm_adv_by_std_in_grpo: (bool)
            Whether to normalize by standard deviation
            
    Returns:
        advantages: (torch.Tensor) shape (bs, seq_len)
            Uniform advantages applied to all action tokens
        returns: (torch.Tensor) shape (bs, seq_len)
            Same as advantages for compatibility
    """
    device = turn_sequence_tensor.device
    batch_size, seq_len = turn_sequence_tensor.shape
    
    # Initialize unified advantage tensor
    unified_advantages = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
    
    # Collect unified rewards for GRPO normalization
    id2unified_rewards = defaultdict(list)
    id2unified_info = defaultdict(list)
    
    with torch.no_grad():
        # Extract and sum all rewards per sample
        for i in range(batch_size):
            sample_structured_rewards = structured_rewards[i]
            turn_rewards = sample_structured_rewards['turn_rewards']
            global_rewards_dict = sample_structured_rewards['global_rewards']
            
            # Sum all turn rewards and normalize by number of turns
            total_turn_reward = sum(turn_rewards.values())
            num_turns = len(turn_rewards) if turn_rewards else 1  # Avoid division by zero
            normalized_turn_reward = total_turn_reward / num_turns
            
            # Sum all global rewards (exclude raw scores)
            total_global_reward = sum(v for k, v in global_rewards_dict.items() 
                                    if not k.startswith('_raw_'))
            
            # Unified reward is normalized turn rewards + global rewards
            unified_reward = normalized_turn_reward + total_global_reward
            
            # Collect for GRPO normalization
            id2unified_rewards[index[i]].append(unified_reward)
            id2unified_info[index[i]].append(i)
        
        # Calculate group statistics for unified rewards
        id2mean_unified = {}
        id2std_unified = {}
        
        for idx in id2unified_rewards:
            if len(id2unified_rewards[idx]) == 1:
                id2mean_unified[idx] = torch.tensor(0.0, device=device)
                id2std_unified[idx] = torch.tensor(1.0, device=device)
            elif len(id2unified_rewards[idx]) > 1:
                unified_scores = torch.tensor(id2unified_rewards[idx], device=device)
                id2mean_unified[idx] = unified_scores.mean()
                id2std_unified[idx] = unified_scores.std()
        
        # Apply unified advantage to all action tokens
        for idx in id2unified_rewards:
            for reward_idx, sample_idx in enumerate(id2unified_info[idx]):
                unified_reward = id2unified_rewards[idx][reward_idx]
                
                if norm_adv_by_std_in_grpo:
                    unified_advantage = (unified_reward - id2mean_unified[idx]) / (id2std_unified[idx] + epsilon)
                else:
                    unified_advantage = unified_reward - id2mean_unified[idx]
                
                # Apply this advantage to ALL action tokens in this sample
                action_mask = turn_sequence_tensor[sample_idx] > 0
                unified_advantages[sample_idx, action_mask] = unified_advantage
    
    # Extract response portion of advantages to match response_mask dimensions
    response_length = response_mask.shape[1]
    response_advantages = unified_advantages[:, -response_length:]
    
    # Debug printout for first few samples
    if batch_size > 0:
        max_debug_samples = min(2, batch_size)
        print(f"\n[DEBUG] Uniform Advantage Details (showing {max_debug_samples}/{batch_size} samples):")
        print("=" * 80)
        
        for i in range(max_debug_samples):
            sample_unified_advantages = unified_advantages[i]
            
            print(f"\n🧮 SAMPLE {i} UNIFORM ADVANTAGE:")
            unique_turns = torch.unique(turn_sequence_tensor[i])
            unique_turns = unique_turns[unique_turns > 0]
            
            for turn_num in unique_turns:
                turn_mask = turn_sequence_tensor[i] == turn_num
                turn_positions = torch.where(turn_mask)[0]
                turn_advantages = sample_unified_advantages[turn_mask]
                
                print(f"  Turn {turn_num.item()}: {len(turn_positions)} tokens, "
                      f"uniform_adv={turn_advantages.mean():.4f}")
            
            print(f"📈 Total advantage sum: {sample_unified_advantages.sum():.4f}")
            print("-" * 40)
    
    return response_advantages, response_advantages


def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
):
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        norm_adv_by_std_in_grpo: if True, normalize advantage by std within group

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}.")
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


def compute_reinforce_plus_plus_baseline_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: torch.Tensor, epsilon: float = 1e-6):
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


def compute_rloo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.tensor(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (response_num - 1)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_opo_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, index: np.ndarray, epsilon: float = 1e-6):
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.tensor(id2score[idx])
                len_tensor = torch.tensor(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


def compute_reinforce_plus_plus_outcome_advantage(token_level_rewards: torch.Tensor, response_mask: torch.Tensor, gamma: torch.Tensor):
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


def compute_remax_outcome_advantage(token_level_rewards: torch.Tensor, reward_baselines: torch.Tensor, response_mask: torch.Tensor):
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    kl = old_log_prob - ref_log_prob
    kl = torch.clamp(kl, min=-10, max=10)
    return token_level_scores - kl * kl_ratio


def agg_loss(loss_mat: torch.Tensor, loss_mask: torch.Tensor, loss_agg_mode: str):
    """
    Aggregate the loss matrix into a scalar.

    Args:
        loss_mat: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_mask: `(torch.Tensor)`:
            shape: (bs, response_length)
        loss_agg_mode: (str) choices:
            method to aggregate the loss matrix into a scalar.
    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        loss = verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / torch.sum(loss_mask, dim=-1)  # token-mean
        loss = torch.mean(seq_losses)  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        loss = torch.sum(seq_losses) / loss_mask.shape[-1]  # The divisor
        # (loss_mask.shape[-1]) should ideally be constant
        # throughout training to well-replicate the DrGRPO paper.
        # TODO: Perhaps add user-defined normalizer argument to
        # agg_loss to ensure divisor stays constant throughout.
    elif loss_agg_mode == "unbiased-fixed-norm":
        # Unbiased loss normalization using fixed MAX_TOKENS denominator
        # This prevents bias toward lengthy responses
        import os
        max_length = int(os.environ.get('MAX_LENGTH', 2200))  # Default to 2200 if not set
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum per sequence
        loss = torch.sum(seq_losses) / max_length  # Fixed normalization
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0," + f" but get the value: {clip_ratio_c}."

    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-10, max=10)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)
        loss_agg_mode (str): Loss aggregation mode

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(vpreds: torch.Tensor, returns: torch.Tensor, values: torch.Tensor, response_mask: torch.Tensor, cliprange_value: float, loss_agg_mode: str = "token-mean"):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:

    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        kl = torch.clamp(kl, min=-5, max=5)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return kld

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def create_kg_token_mask(tokens: torch.Tensor, tokenizer, config=None):
    """
    Create a mask to identify KG-related tokens for reduced KL penalty.
    
    This function identifies tokens that are part of knowledge graph queries
    and other KG-specific patterns, allowing for reduced KL penalty on these
    tokens during training to prevent instability from out-of-vocabulary patterns.
    
    Args:
        tokens: `(torch.Tensor)`
            Token IDs tensor of shape (batch_size, sequence_length)
        tokenizer: Tokenizer instance to decode token IDs
        config: Configuration object with kg_token_masking settings
    
    Returns:
        kg_mask: `(torch.Tensor)`
            Boolean mask of shape (batch_size, sequence_length) where True indicates
            KG-related tokens that should receive reduced KL penalty
    """
    # Check if KG token masking is enabled
    if config is None or not config.get('enable', False):
        # Return all-False mask if masking is disabled
        return torch.zeros_like(tokens, dtype=torch.bool)
    
    # Check tokenizer is provided when masking is enabled
    if tokenizer is None:
        raise ValueError("KG token masking is enabled but tokenizer is None. Please provide tokenizer to create_kg_token_mask function.")
    
    # Get patterns from config - only use what's explicitly specified
    kg_patterns = list(config.get('patterns', [
        '<kg-query>', '</kg-query>',
        '<search>', '</search>',
        '<think>', '</think>',
        'get_tail_relations', 'get_head_relations',
        'get_tail_entities', 'get_head_entities', 
        'get_conditional_relations'
    ]))
    
    batch_size, seq_len = tokens.shape
    kg_mask = torch.zeros_like(tokens, dtype=torch.bool)
    
    # Process each sequence in the batch
    for batch_idx in range(batch_size):
        # Decode tokens to text for pattern matching
        try:
            # Decode the sequence, handling potential decoding errors
            sequence_text = tokenizer.decode(tokens[batch_idx], skip_special_tokens=False)
        except:
            # If decoding fails, skip this sequence
            continue
            
        # Find positions of KG patterns in the decoded text
        for pattern in kg_patterns:
            start_idx = 0
            while True:
                # Find next occurrence of pattern
                pattern_start = sequence_text.find(pattern, start_idx)
                if pattern_start == -1:
                    break
                    
                pattern_end = pattern_start + len(pattern)
                
                # Find corresponding token positions
                # This is approximate - we'll mark tokens that likely contain the pattern
                try:
                    # Encode just the pattern to see how many tokens it spans
                    pattern_tokens = tokenizer.encode(pattern, add_special_tokens=False)
                    pattern_token_count = len(pattern_tokens)
                    
                    # Estimate token position in sequence (rough approximation)
                    # This could be improved with more precise token-to-char mapping
                    char_to_token_ratio = seq_len / max(len(sequence_text), 1)
                    approx_token_start = int(pattern_start * char_to_token_ratio)
                    approx_token_end = min(seq_len, approx_token_start + pattern_token_count + 2)  # +2 for safety margin
                    
                    # Mark tokens in this range as KG tokens
                    kg_mask[batch_idx, approx_token_start:approx_token_end] = True
                    
                except:
                    # If tokenization fails, skip this pattern
                    pass
                    
                start_idx = pattern_end
        
    # Debug logging
    if config.get('debug_logging', False):
        total_masked = kg_mask.sum().item()
        total_tokens = kg_mask.numel()
        print(f"[KG_MASK_DEBUG] Masked {total_masked}/{total_tokens} tokens ({100*total_masked/total_tokens:.2f}%)")
    
    return kg_mask


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data
