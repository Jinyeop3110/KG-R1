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
Metrics related to the PPO trainer.
"""

from collections import defaultdict
from functools import partial
from typing import Any, Callable, Dict, List, Union

import numpy as np
import torch

from verl import DataProto
from verl.utils.import_utils import deprecated


def compute_uid_grouped_stats(values: List[Union[float, int]], uids: List[str], prefix: str) -> Dict[str, float]:
    """
    Compute statistics for values grouped by UID (same prompt).
    
    This function groups values by their corresponding UIDs and computes both:
    1. Statistics of group means (how do different prompts perform on average)
    2. Statistics of group standard deviations (how much variance within each prompt group)
    
    Args:
        values: List of metric values (one per sample)
        uids: List of UIDs corresponding to each value (same length as values)
        prefix: Prefix for the output metric names (e.g., "rl-aux/reward-per-group")
        
    Returns:
        Dict with grouped statistics:
        - "{prefix}/mean": Mean of all group means
        - "{prefix}/max": Max of all group means  
        - "{prefix}/min": Min of all group means
        - "{prefix}/std": Std of all group means
        - "{prefix}-std/mean": Mean of group standard deviations
        - "{prefix}-std/max": Max of group standard deviations
        - "{prefix}-std/min": Min of group standard deviations
        - "{prefix}-std/std": Std of group standard deviations
    """
    # Convert numpy arrays to lists if needed
    if hasattr(values, 'tolist'):
        values = values.tolist()
    if hasattr(uids, 'tolist'):
        uids = uids.tolist()
    
    if len(values) == 0 or len(uids) == 0 or len(values) != len(uids):
        return {}
    
    # Group values by UID
    uid_to_values = defaultdict(list)
    for value, uid in zip(values, uids):
        uid_to_values[uid].append(value)
    
    # Compute statistics for each group
    group_means, group_stds, group_mins, group_maxs = [], [], [], []
    for uid, group_values in uid_to_values.items():
        if len(group_values) > 0:
            group_tensor = torch.tensor(group_values, dtype=torch.float32)
            group_means.append(torch.mean(group_tensor).item())
            group_mins.append(torch.min(group_tensor).item())
            group_maxs.append(torch.max(group_tensor).item())
            
            if len(group_values) > 1:
                group_stds.append(torch.std(group_tensor).item())
            else:
                group_stds.append(0.0)  # Single item has no std
    
    if not group_means:
        return {}
    
    # Compute overall statistics
    result = {
        f"{prefix}/mean": float(np.mean(group_means)),
        f"{prefix}/max": float(np.max(group_maxs)),
        f"{prefix}/min": float(np.min(group_mins)),
        f"{prefix}/std": float(np.std(group_means))
    }
    
    # Add group std statistics if we have multiple groups
    if group_stds and len([s for s in group_stds if s > 0]) > 0:
        result.update({
            f"{prefix}-std/mean": float(np.mean(group_stds)),
            f"{prefix}-std/max": float(np.max(group_stds)),
            f"{prefix}-std/min": float(np.min(group_stds)),
            f"{prefix}-std/std": float(np.std(group_stds))
        })
    
    return result


def group_by_uid(values: List[Any], uids: List[str]) -> Dict[str, List[int]]:
    """
    Group indices by UID for reusable UID-based operations.
    
    Args:
        values: List of values (used only for length validation)
        uids: List of UIDs corresponding to each value
        
    Returns:
        Dict mapping each UID to list of indices that have that UID
    """
    if len(values) == 0 or len(uids) == 0 or len(values) != len(uids):
        return {}
    
    uid_to_indices = defaultdict(list)
    
    for i, uid in enumerate(uids):
        uid_to_indices[uid].append(i)
    
    return dict(uid_to_indices)


@deprecated("verl.utils.metric.reduce_metrics")
def reduce_metrics(metrics: Dict[str, List[Any]]) -> Dict[str, Any]:
    """
    Reduces a dictionary of metric lists by computing the mean of each list.

    Args:
        metrics: A dictionary mapping metric names to lists of metric values.

    Returns:
        A dictionary with the same keys but with each list replaced by its mean value.

    Example:
        >>> metrics = {"loss": [1.0, 2.0, 3.0], "accuracy": [0.8, 0.9, 0.7]}
        >>> reduce_metrics(metrics)
        {"loss": 2.0, "accuracy": 0.8}
    """
    from verl.utils.metric import reduce_metrics

    return reduce_metrics(metrics)


def _compute_response_info(batch: DataProto) -> Dict[str, Any]:
    """
    Computes information about prompts and responses from a batch.

    This is an internal helper function that extracts masks and lengths for prompts and responses.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.

    Returns:
        A dictionary containing:
            - response_mask: Attention mask for the response tokens
            - prompt_length: Tensor of prompt lengths for each item in the batch
            - response_length: Tensor of response lengths for each item in the batch
    """
    response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-response_length]
    response_mask = batch.batch["attention_mask"][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch: DataProto, use_critic: bool = True) -> Dict[str, Any]:
    """
    Computes various metrics from a batch of data for PPO training.

    This function calculates metrics related to scores, rewards, advantages, returns, values,
    and sequence lengths from a batch of data. It provides statistical information (mean, max, min)
    for each metric category.

    Args:
        batch: A DataProto object containing batch data with token-level scores, rewards, advantages, etc.
        use_critic: Whether to include critic-specific metrics. Defaults to True.

    Returns:
        A dictionary of metrics including:
            - critic/score/mean, max, min: Statistics about sequence scores
            - critic/rewards/mean, max, min: Statistics about sequence rewards
            - rl-aux/reward-per-group/mean, max, min, std: Per-prompt-group reward statistics
            - critic/advantages/mean, max, min: Statistics about advantages
            - critic/returns/mean, max, min: Statistics about returns
            - critic/values/mean, max, min: Statistics about critic values (if use_critic=True)
            - critic/vf_explained_var: Explained variance of the value function (if use_critic=True)
            - response_length/mean, max, min, clip_ratio: Statistics about response lengths
            - prompt_length/mean, max, min, clip_ratio: Statistics about prompt lengths
    """
    # Handle both structured rewards (new format) and token_level_scores (legacy format)
    if "token_level_scores" in batch.batch:
        sequence_score = batch.batch["token_level_scores"].sum(-1)
        sequence_reward = batch.batch["token_level_rewards"].sum(-1)
    else:
        # For structured rewards format, compute sequence-level scores from advantages
        # Use advantages as a proxy for token-level rewards for metrics
        sequence_score = batch.batch["advantages"].sum(-1)
        sequence_reward = batch.batch["advantages"].sum(-1)
    
    # Compute per-prompt-group reward statistics using reusable function
    prompt_group_rewards = {}
    prompt_group_std_stats = {}
    if "uid" in batch.non_tensor_batch:
        reward_values = [sequence_reward[i].item() for i in range(len(sequence_reward))]
        uid_info = batch.non_tensor_batch["uid"]
        
        # Convert to lists if they are numpy arrays
        if hasattr(uid_info, 'tolist'):
            uid_info = uid_info.tolist()
        
        # Use the reusable UID grouping function
        grouped_stats = compute_uid_grouped_stats(reward_values, uid_info, "rl-aux/reward-per-group")
        
        # Extract the stats in the old format for backward compatibility
        if grouped_stats:
            prompt_group_rewards = {
                "mean": grouped_stats.get("rl-aux/reward-per-group/mean", 0.0),
                "min": grouped_stats.get("rl-aux/reward-per-group/min", 0.0),
                "max": grouped_stats.get("rl-aux/reward-per-group/max", 0.0),
                "std": grouped_stats.get("rl-aux/reward-per-group/std", 0.0),
            }
            
            # Extract std stats if available
            if "rl-aux/reward-per-group-std/mean" in grouped_stats:
                prompt_group_std_stats = {
                    "mean": grouped_stats.get("rl-aux/reward-per-group-std/mean", 0.0),
                    "std": grouped_stats.get("rl-aux/reward-per-group-std/std", 0.0),
                    "min": grouped_stats.get("rl-aux/reward-per-group-std/min", 0.0),
                    "max": grouped_stats.get("rl-aux/reward-per-group-std/max", 0.0),
                }

    advantages = batch.batch["advantages"]
    returns = batch.batch["returns"]

    max_response_length = batch.batch["responses"].shape[-1]

    prompt_mask = batch.batch["attention_mask"][:, :-max_response_length].bool()
    response_mask = batch.batch["attention_mask"][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info["prompt_length"]
    response_length = response_info["response_length"]

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch["values"]
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        "critic/score/mean": torch.mean(sequence_score).detach().item(),
        "critic/score/max": torch.max(sequence_score).detach().item(),
        "critic/score/min": torch.min(sequence_score).detach().item(),
        # reward
        "critic/rewards/mean": torch.mean(sequence_reward).detach().item(),
        "critic/rewards/max": torch.max(sequence_reward).detach().item(),
        "critic/rewards/min": torch.min(sequence_reward).detach().item(),
        # RL core metrics - essential reward metrics for monitoring training progress (like val-core)
        "rl-core/reward/mean": torch.mean(sequence_reward).detach().item(),
        # RL auxiliary metrics - detailed statistics (like val-aux)
        "rl-aux/reward/max": torch.max(sequence_reward).detach().item(),
        "rl-aux/reward/min": torch.min(sequence_reward).detach().item(),
        "rl-aux/reward/std": torch.std(sequence_reward).detach().item(),
        "rl-aux/score/mean": torch.mean(sequence_score).detach().item(),
        "rl-aux/score/max": torch.max(sequence_score).detach().item(),
        "rl-aux/score/min": torch.min(sequence_score).detach().item(),
        "rl-aux/score/std": torch.std(sequence_score).detach().item(),
        # per-prompt-group reward statistics
        **(
            {
                "rl-aux/reward-per-group/mean": prompt_group_rewards["mean"],
                "rl-aux/reward-per-group/min": prompt_group_rewards["min"],
                "rl-aux/reward-per-group/max": prompt_group_rewards["max"],
                "rl-aux/reward-per-group/std": prompt_group_rewards["std"],
            }
            if prompt_group_rewards
            else {}
        ),
        # statistics of per-group standard deviations
        **(
            {
                "rl-aux/reward-per-group-std/mean": prompt_group_std_stats["mean"],
                "rl-aux/reward-per-group-std/std": prompt_group_std_stats["std"],
                "rl-aux/reward-per-group-std/min": prompt_group_std_stats["min"],
                "rl-aux/reward-per-group-std/max": prompt_group_std_stats["max"],
            }
            if prompt_group_std_stats
            else {}
        ),
        # adv
        "critic/advantages/mean": torch.mean(valid_adv).detach().item(),
        "critic/advantages/max": torch.max(valid_adv).detach().item(),
        "critic/advantages/min": torch.min(valid_adv).detach().item(),
        # returns
        "critic/returns/mean": torch.mean(valid_returns).detach().item(),
        "critic/returns/max": torch.max(valid_returns).detach().item(),
        "critic/returns/min": torch.min(valid_returns).detach().item(),
        **(
            {
                # values
                "critic/values/mean": torch.mean(valid_values).detach().item(),
                "critic/values/max": torch.max(valid_values).detach().item(),
                "critic/values/min": torch.min(valid_values).detach().item(),
                # vf explained var
                "critic/vf_explained_var": (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
            }
            if use_critic
            else {}
        ),
        # response length
        "response_length/mean": torch.mean(response_length).detach().item(),
        "response_length/max": torch.max(response_length).detach().item(),
        "response_length/min": torch.min(response_length).detach().item(),
        "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        "prompt_length/mean": torch.mean(prompt_length).detach().item(),
        "prompt_length/max": torch.max(prompt_length).detach().item(),
        "prompt_length/min": torch.min(prompt_length).detach().item(),
        "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    
    # Add reward component metrics if available
    reward_component_metrics = compute_reward_component_metrics(batch)
    metrics.update(reward_component_metrics)
    
    # Add Pass@K metrics if UID info is available
    pass_at_k_metrics = compute_pass_at_k_metrics(batch)
    metrics.update(pass_at_k_metrics)
    
    return metrics


def compute_pass_at_k_metrics(batch: DataProto, k: int = 3, num_bootstrap: int = 100) -> Dict[str, Any]:
    """
    Calculate Pass@K metrics using bootstrap sampling for GRPO training.
    
    For GRPO with multiple responses per prompt, this measures the probability
    that at least 1 out of K randomly sampled responses has exact match or retrieval success.
    
    Args:
        batch: DataProto object containing batch data with reward components and UID info
        k: Number of samples to consider (default 3 for Pass@3)
        num_bootstrap: Number of bootstrap iterations for stable estimates
        
    Returns:
        Dict containing Pass@K metrics for WandB logging
    """
    import random
    
    # Check if we have the necessary data
    if not hasattr(batch, 'non_tensor_batch') or batch.non_tensor_batch is None:
        return {}
    
    non_tensor_batch = batch.non_tensor_batch
    if 'uid' not in non_tensor_batch:
        return {}
    
    # Get UID info and reward components
    uid_info = non_tensor_batch['uid']
    batch_size = len(uid_info)
    
    # Check if we have exact_match and retrieval_quality scores
    if 'em_score' not in non_tensor_batch and 'exact_match' not in non_tensor_batch:
        return {}
    
    # Use em_score or exact_match, whichever is available
    em_scores = non_tensor_batch.get('em_score', non_tensor_batch.get('exact_match', [0.0] * batch_size))
    retrieval_scores = non_tensor_batch.get('retrieval_score', non_tensor_batch.get('retrieval_quality', [0.0] * batch_size))
    
    # Initialize result arrays with zeros
    pass_metrics = {
        f'exact_match_pass@{k}': np.zeros(batch_size),
        f'retrieval_quality_pass@{k}': np.zeros(batch_size)
    }
    
    # Group responses by UID using reusable function
    uid_to_indices = group_by_uid(list(range(batch_size)), uid_info)
    
    # Process each UID group
    for uid, indices in uid_to_indices.items():
        if len(indices) < k:
            # Not enough responses for Pass@K, keep zero scores
            continue
        
        # Get scores for this group
        group_em_scores = [em_scores[i] for i in indices]
        group_retrieval_scores = [retrieval_scores[i] for i in indices]
        
        # Bootstrap sampling
        em_successes = 0
        retrieval_successes = 0
        
        for _ in range(num_bootstrap):
            # Randomly sample k responses
            sampled_indices = random.sample(range(len(indices)), k)
            
            # Check if any sampled response has exact match
            has_em = any(group_em_scores[idx] > 0 for idx in sampled_indices)
            if has_em:
                em_successes += 1
            
            # Check if any sampled response has retrieval success
            has_retrieval = any(group_retrieval_scores[idx] > 0 for idx in sampled_indices)
            if has_retrieval:
                retrieval_successes += 1
        
        # Calculate Pass@K probabilities
        em_prob = em_successes / num_bootstrap
        retrieval_prob = retrieval_successes / num_bootstrap
        
        # Assign to all indices in this group
        for idx in indices:
            pass_metrics[f'exact_match_pass@{k}'][idx] = em_prob
            pass_metrics[f'retrieval_quality_pass@{k}'][idx] = retrieval_prob
    
    # Convert to the format expected by the metrics system
    result_metrics = {}
    for metric_name, values in pass_metrics.items():
        result_metrics[f"rl-core/{metric_name}/mean"] = float(np.mean(values))
        result_metrics[f"rl-aux/{metric_name}/max"] = float(np.max(values))
        result_metrics[f"rl-aux/{metric_name}/min"] = float(np.min(values))
        result_metrics[f"rl-aux/{metric_name}/std"] = float(np.std(values))
    
    return result_metrics


def compute_timing_metrics(batch: DataProto, timing_raw: Dict[str, float]) -> Dict[str, Any]:
    """
    Computes timing metrics for different processing stages in PPO training.

    This function calculates both raw timing metrics (in seconds) and per-token timing in milliseconds for various processing stages like generation, reference computation,
    value computation, advantage computation, and model updates.

    Args:
        batch: A DataProto object containing batch data with responses and attention masks.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.

    Returns:
        A dictionary containing:
            - timing_s/{name}: Raw timing in seconds for each stage
            - timing_per_token_ms/{name}: Per-token timing in milliseconds for each stage

    Note:
        Different stages use different token counts for normalization:
        - "gen" uses only response tokens
        - Other stages ("ref", "values", "adv", "update_critic", "update_actor") use all tokens
          (prompt + response)
    """
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info["prompt_length"]).item()
    num_response_tokens = torch.sum(response_info["response_length"]).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        "gen": num_response_tokens,
        **{name: num_overall_tokens for name in ["ref", "values", "adv", "update_critic", "update_actor"]},
    }

    return {
        **{f"timing_s/{name}": value for name, value in timing_raw.items()},
        **{f"timing_per_token_ms/{name}": timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys()) & set(timing_raw.keys())},
    }


def compute_throughout_metrics(batch: DataProto, timing_raw: Dict[str, float], n_gpus: int) -> Dict[str, Any]:
    """
    Computes throughput metrics for PPO training.

    This function calculates performance metrics related to token processing speed,
    including the total number of tokens processed, time per step, and throughput
    (tokens per second per GPU).

    Args:
        batch: A DataProto object containing batch data with meta information about token counts.
        timing_raw: A dictionary mapping stage names to their execution times in seconds.
                   Must contain a "step" key with the total step time.
        n_gpus: Number of GPUs used for training.

    Returns:
        A dictionary containing:
            - perf/total_num_tokens: Total number of tokens processed in the batch
            - perf/time_per_step: Time taken for the step in seconds
            - perf/throughput: Tokens processed per second per GPU

    Note:
        The throughput is calculated as total_tokens / (time * n_gpus) to normalize
        across different GPU counts.
    """
    total_num_tokens = sum(batch.meta_info["global_token_num"])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        "perf/total_num_tokens": total_num_tokens,
        "perf/time_per_step": time,
        "perf/throughput": total_num_tokens / (time * n_gpus),
    }


def bootstrap_metric(
    data: list[Any],
    subset_size: int,
    reduce_fns: list[Callable[[np.ndarray], float]],
    n_bootstrap: int = 100,
    seed: int = 42,
) -> list[tuple[float, float]]:
    """
    Performs bootstrap resampling to estimate statistics of metrics.

    This function uses bootstrap resampling to estimate the mean and standard deviation
    of metrics computed by the provided reduction functions on random subsets of the data.

    Args:
        data: List of data points to bootstrap from.
        subset_size: Size of each bootstrap sample.
        reduce_fns: List of functions that compute a metric from a subset of data.
        n_bootstrap: Number of bootstrap iterations. Defaults to 1000.
        seed: Random seed for reproducibility. Defaults to 42.

    Returns:
        A list of tuples, where each tuple contains (mean, std) for a metric
        corresponding to each reduction function in reduce_fns.

    Example:
        >>> data = [1, 2, 3, 4, 5]
        >>> reduce_fns = [np.mean, np.max]
        >>> bootstrap_metric(data, 3, reduce_fns)
        [(3.0, 0.5), (4.5, 0.3)]  # Example values
    """
    np.random.seed(seed)

    bootstrap_metric_lsts = [[] for _ in range(len(reduce_fns))]
    for _ in range(n_bootstrap):
        bootstrap_idxs = np.random.choice(len(data), size=subset_size, replace=True)
        bootstrap_data = [data[i] for i in bootstrap_idxs]
        for i, reduce_fn in enumerate(reduce_fns):
            bootstrap_metric_lsts[i].append(reduce_fn(bootstrap_data))
    return [(np.mean(lst), np.std(lst)) for lst in bootstrap_metric_lsts]


def calc_maj_val(data: list[dict[str, Any]], vote_key: str, val_key: str) -> float:
    """
    Calculate a value based on majority voting.

    This function identifies the most common value for a specified vote key
    in the data, then returns the corresponding value for that majority vote.

    Args:
        data: List of dictionaries, where each dictionary contains both vote_key and val_key.
        vote_key: The key in each dictionary used for voting/counting.
        val_key: The key in each dictionary whose value will be returned for the majority vote.

    Returns:
        The value associated with the most common vote.

    Example:
        >>> data = [
        ...     {"pred": "A", "val": 0.9},
        ...     {"pred": "B", "val": 0.8},
        ...     {"pred": "A", "val": 0.7}
        ... ]
        >>> calc_maj_val(data, vote_key="pred", val_key="val")
        0.9  # Returns the first "val" for the majority vote "A"
    """
    vote2vals = defaultdict(list)
    for d in data:
        vote2vals[d[vote_key]].append(d[val_key])

    vote2cnt = {k: len(v) for k, v in vote2vals.items()}
    maj_vote = max(vote2cnt, key=vote2cnt.get)

    maj_val = vote2vals[maj_vote][0]

    return maj_val


def process_validation_metrics(data_sources: list[str], sample_inputs: list[str], infos_dict: dict[str, list[Any]], seed: int = 42) -> dict[str, dict[str, dict[str, float]]]:
    """
    Process validation metrics into a structured format with statistical analysis.

    This function organizes validation metrics by data source and prompt, then computes
    various statistical measures including means, standard deviations, best/worst values,
    and majority voting results. It also performs bootstrap sampling to estimate statistics
    for different sample sizes.

    Args:
        data_sources: List of data source identifiers for each sample.
        sample_inputs: List of input prompts corresponding to each sample.
        infos_dict: Dictionary mapping variable names to lists of values for each sample.
        seed: Random seed for bootstrap sampling. Defaults to 42.

    Returns:
        A nested dictionary with the structure:
        {
            data_source: {
                variable_name: {
                    metric_name: value
                }
            }
        }

        Where metric_name includes:
        - "mean@N": Mean value across N samples
        - "std@N": Standard deviation across N samples
        - "best@N/mean": Mean of the best values in bootstrap samples of size N
        - "best@N/std": Standard deviation of the best values in bootstrap samples
        - "worst@N/mean": Mean of the worst values in bootstrap samples
        - "worst@N/std": Standard deviation of the worst values in bootstrap samples
        - "maj@N/mean": Mean of majority voting results in bootstrap samples (if "pred" exists)
        - "maj@N/std": Standard deviation of majority voting results (if "pred" exists)

    Example:
        >>> data_sources = ["source1", "source1", "source2"]
        >>> sample_inputs = ["prompt1", "prompt1", "prompt2"]
        >>> infos_dict = {"score": [0.8, 0.9, 0.7], "pred": ["A", "A", "B"]}
        >>> result = process_validation_metrics(data_sources, sample_inputs, infos_dict)
        >>> # result will contain statistics for each data source and variable
    """
    # Group metrics by data source, prompt and variable
    data_src2prompt2var2vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for sample_idx, data_source in enumerate(data_sources):
        prompt = sample_inputs[sample_idx]
        var2vals = data_src2prompt2var2vals[data_source][prompt]
        for var_name, var_vals in infos_dict.items():
            var2vals[var_name].append(var_vals[sample_idx])

    # Calculate metrics for each group
    data_src2prompt2var2metric = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for data_source, prompt2var2vals in data_src2prompt2var2vals.items():
        for prompt, var2vals in prompt2var2vals.items():
            for var_name, var_vals in var2vals.items():
                if isinstance(var_vals[0], str):
                    continue

                metric = {}
                n_resps = len(var_vals)
                metric[f"mean@{n_resps}"] = np.mean(var_vals)

                if n_resps > 1:
                    metric[f"std@{n_resps}"] = np.std(var_vals)

                    ns = []
                    n = 2
                    while n < n_resps:
                        ns.append(n)
                        n *= 2
                    ns.append(n_resps)

                    for n in ns:
                        [(bon_mean, bon_std), (won_mean, won_std)] = bootstrap_metric(data=var_vals, subset_size=n, reduce_fns=[np.max, np.min], seed=seed)
                        metric[f"best@{n}/mean"], metric[f"best@{n}/std"] = bon_mean, bon_std
                        metric[f"worst@{n}/mean"], metric[f"worst@{n}/std"] = won_mean, won_std
                        if var2vals.get("pred", None) is not None:
                            vote_data = [{"val": val, "pred": pred} for val, pred in zip(var_vals, var2vals["pred"])]
                            [(maj_n_mean, maj_n_std)] = bootstrap_metric(
                                data=vote_data,
                                subset_size=n,
                                reduce_fns=[partial(calc_maj_val, vote_key="pred", val_key="val")],
                                seed=seed,
                            )
                            metric[f"maj@{n}/mean"], metric[f"maj@{n}/std"] = maj_n_mean, maj_n_std

                data_src2prompt2var2metric[data_source][prompt][var_name] = metric

    # Aggregate metrics across prompts
    data_src2var2metric2prompt_vals = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for data_source, prompt2var2metric in data_src2prompt2var2metric.items():
        for prompt, var2metric in prompt2var2metric.items():
            for var_name, metric in var2metric.items():
                for metric_name, metric_val in metric.items():
                    data_src2var2metric2prompt_vals[data_source][var_name][metric_name].append(metric_val)

    data_src2var2metric2val = defaultdict(lambda: defaultdict(lambda: defaultdict(float)))
    for data_source, var2metric2prompt_vals in data_src2var2metric2prompt_vals.items():
        for var_name, metric2prompt_vals in var2metric2prompt_vals.items():
            for metric_name, prompt_vals in metric2prompt_vals.items():
                data_src2var2metric2val[data_source][var_name][metric_name] = np.mean(prompt_vals)

    return data_src2var2metric2val


def compute_reward_component_metrics(batch: DataProto) -> Dict[str, Any]:
    """
    Compute metrics for individual reward components (if available).
    This provides detailed breakdown of reward components during training.
    Core metrics (em_score, final_format_score, reward) are categorized as rl-core,
    while other components are rl-aux for auxiliary detailed analysis.
    """
    metrics = {}
    
    # Check if reward component data is available in non_tensor_batch
    if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch:
        # Core reward components (most important for monitoring training)
        core_component_keys = ['score', 'em_score', 'retrieval_score', 'format_score', 'reward']
        
        # Auxiliary reward components (detailed breakdown)
        aux_component_keys = [
            'answer_match_score', 'structure_format_score', 
            'final_format_score', 'query_validity_score',
            'error_penalty_score'
        ]
        
        # Query count metrics (important for KG task monitoring)
        query_count_keys = [
            'valid_queries', 'invalid_queries', 'total_queries',
            # NEW: Unique query metrics for reward hacking prevention
            'unique_valid_queries', 'unique_invalid_queries', 
            'duplicate_valid_queries', 'duplicate_invalid_queries',
            'query_repetition_rate', 'unique_query_efficiency',
            'reward_hacking_prevented', 'reward_inflation_prevented',
            #'kg_server_error_count', 'kg_not_found_count', 
            #'kg_format_error_count', 'kg_no_data_count'
        ]
        
        # Process core components
        for key in core_component_keys:
            if key in batch.non_tensor_batch:
                values = batch.non_tensor_batch[key]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    metrics[f"rl-core/{key}/mean"] = float(np.mean(values))
                    # Additional stats for core metrics go to aux
                    metrics[f"rl-aux/{key}/max"] = float(np.max(values))
                    metrics[f"rl-aux/{key}/min"] = float(np.min(values))
                    metrics[f"rl-aux/{key}/std"] = float(np.std(values))
        
        # Process auxiliary components
        for key in aux_component_keys:
            if key in batch.non_tensor_batch:
                values = batch.non_tensor_batch[key]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    metrics[f"rl-aux/components/{key}/mean"] = float(np.mean(values))
                    metrics[f"rl-aux/components/{key}/max"] = float(np.max(values))
                    metrics[f"rl-aux/components/{key}/min"] = float(np.min(values))
                    metrics[f"rl-aux/components/{key}/std"] = float(np.std(values))
        
        # Process query count metrics
        for key in query_count_keys:
            if key in batch.non_tensor_batch:
                values = batch.non_tensor_batch[key]
                if isinstance(values, np.ndarray) and len(values) > 0:
                    metrics[f"rl-aux/query_counts/{key}/mean"] = float(np.mean(values))
                    metrics[f"rl-aux/query_counts/{key}/max"] = float(np.max(values))
                    metrics[f"rl-aux/query_counts/{key}/min"] = float(np.min(values))
                    metrics[f"rl-aux/query_counts/{key}/std"] = float(np.std(values))
                    # Also add total sum for query counts (useful for monitoring)
                    metrics[f"rl-aux/query_counts/{key}/sum"] = float(np.sum(values))
    
    return metrics
