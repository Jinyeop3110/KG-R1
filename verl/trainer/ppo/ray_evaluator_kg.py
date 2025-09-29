# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
Ray-based KG Evaluator for Pass@K evaluation with efficient response caching.

This evaluator generates n_rollout_eval responses per prompt using DataProto.repeat()
and computes pass@k metrics for multiple k values (1,3,5,8) from the same response set.

Key features:
- Evaluation-only mode (no training)
- Efficient batch generation with DataProto.repeat()
- Pass@k caching for multiple k values
- KG search integration
- Memory efficient processing
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Optional, Type, List, Dict, Any
import random

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
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
    group_by_uid,
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

# Import from ray_trainer_kg to reuse components
from verl.trainer.ppo.ray_trainer_kg import (
    RayPPOTrainer,
    AdvantageEstimator, 
    Role,
    WorkerType,
    compute_response_mask
)

class RayKGEvaluator(RayPPOTrainer):
    """
    KG Evaluator based on RayPPOTrainer but focused on evaluation only.
    
    Key differences from trainer:
    - No policy updates or training loops
    - Dedicated evaluation with n_rollout_eval parameter
    - Pass@k metric caching for k=1,3,5,8
    - Memory efficient batch processing
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping,
        resource_pool_manager,
        device_name: str = "cuda",
        n_rollout_eval: int = 8,
        k_values: List[int] = None,
        val_reward_fn=None,
        eval_samples: int = 0,
        save_detailed_results: bool = False,
        ray_worker_group_cls=None,
        **kwargs
    ):
        # Initialize parent with all required parameters
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            device_name=device_name,
            reward_fn=None,  # Not needed for evaluation
            val_reward_fn=val_reward_fn,
            **kwargs
        )
        
        # Evaluation-specific parameters
        self.n_rollout_eval = n_rollout_eval
        self.k_values = k_values or [1, 3, 5, 8]
        self.eval_samples = eval_samples  # 0 means evaluate all samples
        self.save_detailed_results = save_detailed_results
        
        # Validate k_values
        max_k = max(self.k_values)
        if max_k > self.n_rollout_eval:
            raise ValueError(f"Max k value ({max_k}) cannot exceed n_rollout_eval ({self.n_rollout_eval})")
        
        # Additional initialization needed for KG search (missing from parent call)
        self._initialize_search_components()
        
        # Initialize metrics tracking
        self.last_length_metrics = {}  # Store length metrics for FLOP calculations
        
        print(f"KG Evaluator initialized with n_rollout_eval={self.n_rollout_eval}, k_values={self.k_values}")
        print(f"KG Search enabled: {self.use_search_generation}")
    
    def _initialize_search_components(self):
        """Initialize KG search components that may not be set by parent."""
        
        # Initialize use_search_generation if not already set by parent
        if not hasattr(self, 'use_search_generation'):
            search_config = getattr(self.config.actor_rollout_ref.rollout, 'search', None)
            self.use_search_generation = search_config is not None and search_config.get('enable', False)
            
            if self.use_search_generation:
                self.search_config = search_config
        
        # Initialize async_rollout_mode if not set (evaluation typically doesn't use async mode)
        if not hasattr(self, 'async_rollout_mode'):
            self.async_rollout_mode = False
        
        # Initialize generation manager for KG search
        if self.use_search_generation and not hasattr(self, 'generation_manager'):
            from kg_r1.llm_agent.generation import LLMGenerationManager, GenerationConfig
            
            # Create generation manager with KG search configuration
            generation_config = GenerationConfig(
                max_turns=self.search_config.get('max_turns', 7),
                max_start_length=self.config.data.max_prompt_length,
                max_prompt_length=self.config.data.max_prompt_length,
                max_response_length=self.config.data.max_response_length,
                max_obs_length=self.config.data.max_obs_length,
                num_gpus=self.config.trainer.n_gpus_per_node,
                no_think_rl=False,
                search_url=self.search_config.get('search_url', 'http://127.0.0.1:8001/retrieve'),
                topk=self.search_config.get('topk', 3)
            )
            
            self.generation_manager = LLMGenerationManager(
                tokenizer=self.tokenizer,
                actor_rollout_wg=None,  # Will be set later when workers are initialized
                config=generation_config,
                is_validation=True
            )
    
    def evaluate_dataset(self, dataset_name: str = "test") -> Dict[str, Any]:
        """
        Evaluate the dataset with pass@k metrics.
        
        Args:
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary containing all evaluation metrics
        """
        print(f"Starting KG evaluation for {dataset_name} dataset")
        print(f"Generating {self.n_rollout_eval} responses per prompt")
        print(f"Computing pass@k for k values: {self.k_values}")
        
        # Initialize detailed results file if enabled
        self.detailed_results_file = None
        if self.save_detailed_results:
            # Create detailed results file path
            output_dir = self.config.trainer.get('default_local_dir', 'evaluation_results')
            os.makedirs(output_dir, exist_ok=True)
            detailed_file_path = os.path.join(output_dir, f'{dataset_name}_detailed_results.jsonl')
            self.detailed_results_file = open(detailed_file_path, 'w', encoding='utf-8')
            print(f"Saving detailed results to: {detailed_file_path}")
        
        # Use validation dataset for evaluation
        dataloader = self.val_dataloader
        
        # Limit samples if eval_samples is specified (> 0)
        if self.eval_samples > 0:
            print(f"Limiting evaluation to first {self.eval_samples} samples")
            # Create a limited dataloader by taking only the first N batches
            # Calculate how many batches we need
            samples_per_batch = dataloader.batch_size if hasattr(dataloader, 'batch_size') else 64
            num_batches_needed = (self.eval_samples + samples_per_batch - 1) // samples_per_batch  # Ceiling division
            print(f"Processing {num_batches_needed} batches to get ~{self.eval_samples} samples")
            
            # Create iterator and limit batches
            dataloader = list(dataloader)[:num_batches_needed]
        
        all_metrics = []
        all_inputs = []
        all_outputs = []
        all_scores = []
        all_reward_extra_infos = []
        
        for batch_idx, test_data in enumerate(tqdm(dataloader, desc="Evaluating batches")):
            print(f"\nProcessing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Process batch and generate responses
            batch_metrics = self._evaluate_batch(test_data)
            all_metrics.append(batch_metrics)
            
            # Save detailed results for this batch if enabled
            if self.save_detailed_results and self.detailed_results_file:
                self._save_batch_details(batch_metrics, batch_idx, dataset_name)
            
            # Store results for logging
            if 'inputs' in batch_metrics:
                all_inputs.extend(batch_metrics['inputs'])
            if 'outputs' in batch_metrics:
                all_outputs.extend(batch_metrics['outputs']) 
            if 'scores' in batch_metrics:
                all_scores.extend(batch_metrics['scores'])
            if 'reward_extra_infos' in batch_metrics:
                all_reward_extra_infos.extend(batch_metrics['reward_extra_infos'])
        
        # Aggregate metrics across all batches
        final_metrics = self._aggregate_metrics(all_metrics)
        
        # Log generations if configured
        if all_inputs and all_outputs:
            self._maybe_log_val_generations(
                inputs=all_inputs,
                outputs=all_outputs, 
                scores=all_scores
            )
        
        # Print final results
        print(f"\n{'='*60}")
        print(f"FINAL EVALUATION RESULTS for {dataset_name}")
        print(f"{'='*60}")
        
        # Print pass@k results
        for k in self.k_values:
            em_key = f"exact_match_pass@{k}/mean"
            rq_key = f"retrieval_quality_pass@{k}/mean"
            if em_key in final_metrics:
                print(f"Pass@{k} (Exact Match): {final_metrics[em_key]:.4f}")
            if rq_key in final_metrics:
                print(f"Pass@{k} (Retrieval Quality): {final_metrics[rq_key]:.4f}")
        
        # Print core metrics
        core_metrics = ['exact_match', 'retrieval_quality', 'total_score']
        for metric in core_metrics:
            mean_key = f"{metric}/mean"
            if mean_key in final_metrics:
                print(f"{metric.title().replace('_', ' ')}: {final_metrics[mean_key]:.4f}")
        
        # Close detailed results file if it was opened
        if self.detailed_results_file:
            self.detailed_results_file.close()
            print(f"Detailed results saved and file closed.")
        
        return final_metrics
    
    def _evaluate_batch(self, test_data: dict) -> Dict[str, Any]:
        """
        Evaluate a single batch with n_rollout_eval responses per prompt.
        
        Args:
            test_data: Dictionary containing batch data from dataloader
            
        Returns:
            Dictionary containing batch evaluation metrics
        """
        # Extract meta_info for KG queries first
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
        
        # Convert dictionary to DataProto format
        test_batch = DataProto.from_single_dict(test_data, meta_info=meta_info)
        
        # Handle KG search meta info extraction for backward compatibility
        if self.use_search_generation:
            sample_ids = []
            dataset_names = []
            
            # Since we converted to DataProto, we now have a single DataProto object, not a list
            if hasattr(test_batch, 'non_tensor_batch') and test_batch.non_tensor_batch is not None:
                if "sample_id" in test_batch.non_tensor_batch:
                    sample_ids = test_batch.non_tensor_batch["sample_id"].tolist()
                if "dataset_name" in test_batch.non_tensor_batch:
                    dataset_names = test_batch.non_tensor_batch["dataset_name"].tolist()
                    # Fix lowercase cwq to uppercase CWQ
                    dataset_names = ["CWQ" if ds == "cwq" else ds for ds in dataset_names]
            
            # Update meta_info with extracted values if they exist
            if sample_ids:
                meta_info["sample_ids"] = sample_ids
            if dataset_names:
                meta_info["dataset_names"] = dataset_names
        
        # Prepare generation batch - define pop keys similar to parent trainer
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
        if hasattr(test_batch, 'non_tensor_batch') and test_batch.non_tensor_batch is not None:
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
        
        # Add KG-specific keys for GRPO evaluation
        if self.use_search_generation:
            if hasattr(test_batch, 'non_tensor_batch') and test_batch.non_tensor_batch is not None:
                if "sample_id" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("sample_id")
                if "dataset_name" in test_batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("dataset_name")
        
        meta_info_keys_to_pop = []
        if self.use_search_generation:
            meta_info_keys_to_pop.extend(["sample_ids", "dataset_names"])
        
        test_gen_batch = test_batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            meta_info_keys=meta_info_keys_to_pop
        )
        
        # Add meta_info for KG search
        if meta_info:
            if test_gen_batch.meta_info is None:
                test_gen_batch.meta_info = {}
            test_gen_batch.meta_info.update(meta_info)
        
        # Expand batch for n_rollout_eval responses per prompt
        base_uids = [str(uuid.uuid4()) for _ in range(len(test_gen_batch.batch))]
        test_gen_batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
        
        # Use DataProto.repeat() to expand for multiple responses
        expanded_gen_batch = test_gen_batch.repeat(repeat_times=self.n_rollout_eval, interleave=True)
        
        # Initialize meta_info if needed
        if expanded_gen_batch.meta_info is None:
            expanded_gen_batch.meta_info = {}
        
        # Extract per-sample information for KG queries AFTER repeat operation
        # following the trainer's pattern: use expanded non_tensor_batch instead of manual expansion
        if self.use_search_generation:
            sample_ids = []
            dataset_names = []
            
            # DataProto.repeat() should have expanded sample_id in non_tensor_batch
            if "sample_id" in expanded_gen_batch.non_tensor_batch:
                sample_ids = expanded_gen_batch.non_tensor_batch["sample_id"].tolist()
            else:
                raise ValueError(
                    "sample_id is missing in expanded_gen_batch after repeat(). "
                    "Ensure that sample_id is included in the non_tensor_batch_keys_to_pop list during pop()."
                )
            
            if "dataset_name" in expanded_gen_batch.non_tensor_batch:
                # Use explicit dataset_name from non_tensor_batch
                dataset_names = expanded_gen_batch.non_tensor_batch["dataset_name"].tolist()
            elif "data_source" in expanded_gen_batch.non_tensor_batch:
                # Fallback: map data_source to dataset_name if needed
                data_sources = expanded_gen_batch.non_tensor_batch["data_source"].tolist()
                dataset_names = ["CWQ" if ds == "cwq" else ds for ds in data_sources]
            
            # Store in meta_info for generation manager
            expanded_gen_batch.meta_info.update({
                "sample_ids": sample_ids,
                "dataset_names": dataset_names
            })
        
        # Pad the expanded batch
        test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(expanded_gen_batch, self.actor_rollout_wg.world_size)
        
        # Generate responses
        if not self.use_search_generation:
            # Standard generation
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
        else:
            # KG search generation
            first_input_ids = test_gen_batch_padded.batch['input_ids'][:, -self.generation_manager.config.max_start_length:].clone().long()
            test_output_gen_batch_padded = self.generation_manager.run_llm_loop(
                gen_batch=test_gen_batch_padded,
                initial_input_ids=first_input_ids,
            )
            
            # Log detailed sample information similar to trainer
            self._log_sample_details(test_gen_batch_padded, test_output_gen_batch_padded)
        
        # Unpad results
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        
        # Expand original batch to match generated responses
        test_batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
        
        # Expand test_batch to match the expanded generation
        expanded_test_batch = test_batch.repeat(repeat_times=self.n_rollout_eval, interleave=True)
        
        # Union with generated responses
        final_test_batch = expanded_test_batch.union(test_output_gen_batch)
        
        # CRITICAL FIX: Ensure meta_info with interaction_history is preserved
        # The union operation may not properly handle meta_info, so we manually ensure it's transferred
        if hasattr(test_output_gen_batch, 'meta_info') and test_output_gen_batch.meta_info:
            if final_test_batch.meta_info is None:
                final_test_batch.meta_info = {}
            
            # Transfer critical KG evaluation data
            for key in ['interaction_history', 'turns_stats', 'valid_action_stats']:
                if key in test_output_gen_batch.meta_info:
                    final_test_batch.meta_info[key] = test_output_gen_batch.meta_info[key]
                    print(f"[DEBUG] Transferred {key} to final_test_batch.meta_info")
        
        # Debug: Check if interaction_history is available for reward computation
        if hasattr(final_test_batch, 'meta_info') and final_test_batch.meta_info:
            has_interaction_history = 'interaction_history' in final_test_batch.meta_info
            print(f"[DEBUG] final_test_batch has interaction_history: {has_interaction_history}")
            if has_interaction_history:
                history_length = len(final_test_batch.meta_info['interaction_history'])
                print(f"[DEBUG] interaction_history length: {history_length}")
        else:
            print(f"[DEBUG] WARNING: final_test_batch has no meta_info for reward computation!")
        
        # Compute rewards
        reward_results = self.val_reward_fn(final_test_batch, return_dict=True)
        
        # Extract metrics and compute pass@k for all k values
        batch_metrics = self._compute_batch_passatk_metrics([final_test_batch], reward_results)
        
        # Log detailed batch information for first few samples
        print(f"\n=== BATCH EVALUATION SUMMARY ===")
        print(f"Batch size: {len(final_test_batch.batch.get('input_ids', []))}")
        print(f"Generated {self.n_rollout_eval} responses per prompt")
        
        # Show brief preview of first few input/output pairs
        if hasattr(final_test_batch, 'batch'):
            max_preview = min(3, len(final_test_batch.batch.get('input_ids', [])))
            for i in range(max_preview):
                print(f"\nSample {i+1} preview:")
                
                # Show input
                if 'input_ids' in final_test_batch.batch:
                    try:
                        input_text = self.tokenizer.decode(final_test_batch.batch['input_ids'][i], skip_special_tokens=True)
                        print(f"  Input: {repr(input_text[:150])}{'...' if len(input_text) > 150 else ''}")
                    except Exception as e:
                        print(f"  Input: [Decode error: {e}]")
                
                # Show response
                if 'responses' in final_test_batch.batch:
                    try:
                        response_text = self.tokenizer.decode(final_test_batch.batch['responses'][i], skip_special_tokens=True)
                        print(f"  Response: {repr(response_text[:200])}{'...' if len(response_text) > 200 else ''}")
                    except Exception as e:
                        print(f"  Response: [Decode error: {e}]")
                
                # Show reward if available
                if 'reward_tensor' in reward_results and len(reward_results['reward_tensor']) > i:
                    print(f"  Reward: {reward_results['reward_tensor'][i]:.4f}")
        
        print(f"=== END BATCH SUMMARY ===\n")
        
        # Store additional info for logging
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # Extract samples for logging (similar to trainer approach)
        # Store original inputs from input_ids
        if hasattr(final_test_batch, 'batch') and 'input_ids' in final_test_batch.batch:
            input_ids = final_test_batch.batch["input_ids"]  # Save all samples
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
        
        # Store generated outputs from responses
        if hasattr(final_test_batch, 'batch') and 'responses' in final_test_batch.batch:
            output_ids = final_test_batch.batch["responses"]  # Save all samples
            # Convert to long integers if they're floats (can happen with search generation)
            if output_ids.dtype != torch.long:
                output_ids = output_ids.long()
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
        
        if 'reward_tensor' in reward_results and len(reward_results['reward_tensor']) > 0:
            scores = reward_results['reward_tensor'].tolist()
            sample_scores.extend(scores)
        
        batch_metrics.update({
            'inputs': sample_inputs,
            'outputs': sample_outputs,
            'scores': sample_scores,
            'reward_extra_infos': reward_results.get('extra_info', [])
        })
        
        return batch_metrics
    
    def _save_batch_details(self, batch_metrics: Dict[str, Any], batch_idx: int, dataset_name: str):
        """
        Save detailed results for a batch to JSONL file (continuous streaming).
        
        Args:
            batch_metrics: Dictionary containing inputs, outputs, scores, and reward_extra_infos
            batch_idx: Index of the current batch
            dataset_name: Name of the dataset being evaluated
        """
        import json
        
        # Extract batch data
        inputs = batch_metrics.get('inputs', [])
        outputs = batch_metrics.get('outputs', [])
        scores = batch_metrics.get('scores', [])
        reward_extra_infos = batch_metrics.get('reward_extra_infos', [])
        
        # Ensure all lists have the same length
        max_len = max(len(inputs), len(outputs), len(scores), len(reward_extra_infos))
        
        for i in range(max_len):
            # Get data for this sample (with safe fallbacks)
            prompt = inputs[i] if i < len(inputs) else ""
            response = outputs[i] if i < len(outputs) else ""
            score = scores[i] if i < len(scores) else 0.0
            extra_info = reward_extra_infos[i] if i < len(reward_extra_infos) else {}
            
            # Create detailed record
            record = {
                "uid": f"{dataset_name}_batch{batch_idx:03d}_sample{i:03d}",
                "dataset": dataset_name,
                "batch_idx": batch_idx,
                "sample_idx": i,
                "prompt": prompt,
                "response": response,
                "score": score,
                "metrics": {},
                "metadata": {
                    "response_length": len(response) if response else 0,
                    "prompt_length": len(prompt) if prompt else 0
                }
            }
            
            # Extract individual metrics from extra_info if available
            if isinstance(extra_info, dict):
                # Extract core metrics
                for metric_key in ['f1', 'precision', 'recall', 'em_score', 'retrieval_score', 
                                 'exact_match', 'retrieval_quality', 'format_score']:
                    if metric_key in extra_info:
                        record["metrics"][metric_key] = extra_info[metric_key]
                
                # Add any other available metrics
                for key, value in extra_info.items():
                    if key not in record["metrics"] and isinstance(value, (int, float, bool)):
                        record["metrics"][key] = value
            
            # Write to JSONL file
            try:
                json.dump(record, self.detailed_results_file, ensure_ascii=False)
                self.detailed_results_file.write('\n')
                self.detailed_results_file.flush()  # Ensure immediate write to disk
            except Exception as e:
                print(f"[WARNING] Failed to save detailed result for sample {i}: {e}")
    
    def _log_sample_details(self, input_batch: DataProto, output_batch: DataProto, max_samples: int = 3):
        """
        Log detailed sample information including inputs, responses, and search history.
        Similar to the trainer's debugging output.
        
        Args:
            input_batch: Input DataProto with prompts
            output_batch: Output DataProto with generated responses
            max_samples: Maximum number of samples to log detailed info for
        """
        # Log meta info debugging like in trainer
        if hasattr(output_batch, 'meta_info') and output_batch.meta_info:
            meta_info = output_batch.meta_info
            print(f"\n=== DEBUG EVALUATION META_INFO ===")
            
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
        
        # Log sample inputs and responses (first 3 samples per batch for debugging)
        if hasattr(input_batch, 'batch') and hasattr(output_batch, 'batch'):
            # Only log first 3 samples per batch for debugging
            batch_size = min(3, len(input_batch.batch.get('input_ids', [])))
            
            for i in range(batch_size):
                print(f"\n{'='*80}")
                print(f"BATCH SAMPLE {i} DEBUGGING")
                print(f"{'='*80}")
                
                # Log full conversation like in training
                if 'input_ids' in input_batch.batch and 'responses' in output_batch.batch:
                    input_ids = input_batch.batch['input_ids'][i]
                    response_ids = output_batch.batch['responses'][i]
                    try:
                        # Decode prompt and response
                        input_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        
                        # Combine like training logs - show complete conversation
                        full_conversation = input_text + response_text
                        
                        print(f"\n[COMPLETE CONVERSATION]:")
                        print(full_conversation)
                        print(f"\n[CONVERSATION LENGTH]: {len(full_conversation)} chars")
                        
                    except Exception as e:
                        print(f"[CONVERSATION DECODE ERROR]: {e}")
                
                # Log scores if available
                if hasattr(output_batch, 'meta_info') and 'scores' in output_batch.meta_info:
                    if i < len(output_batch.meta_info['scores']):
                        print(f"\n[SCORE]: {output_batch.meta_info['scores'][i]}")
                
                print(f"{'='*80}\n")
    
    def _compute_batch_passatk_metrics(self, batch: List[DataProto], reward_results: Dict) -> Dict[str, Any]:
        """
        Compute pass@k metrics for all k values from the batch results.
        
        Args:
            batch: List of expanded DataProto objects with UID grouping (should contain single object for evaluator)
            reward_results: Results from reward function
            
        Returns:
            Dictionary containing pass@k metrics for all k values
        """
        # Get the single DataProto from the batch
        if len(batch) == 1:
            combined_batch = batch[0]
        else:
            # Combine batch into single DataProto for metric computation (fallback)
            combined_batch = batch[0]
            for data_proto in batch[1:]:
                combined_batch = combined_batch.union(data_proto)
        
        # Add reward results to batch
        if 'reward_tensor' in reward_results:
            combined_batch.batch['reward'] = reward_results['reward_tensor']
        elif 'structured_rewards' in reward_results:
            # Handle new structured reward format
            structured_rewards = reward_results['structured_rewards']
            if hasattr(structured_rewards, 'sum'):
                combined_batch.batch['reward'] = structured_rewards.sum(dim=-1)  # Sum across components
            else:
                print(f"[DEBUG] structured_rewards format: {type(structured_rewards)}")
        
        # Extract reward component info from reward_extra_info (the correct format)
        if 'reward_extra_info' in reward_results and reward_results['reward_extra_info']:
            reward_extra_info = reward_results['reward_extra_info']
            print(f"[DEBUG] Processing reward_extra_info: {type(reward_extra_info)}")
            
            if isinstance(reward_extra_info, dict):
                # Extract all metrics from wandb metrics
                metric_keys = ['exact_match', 'retrieval_quality', 'f1', 'precision', 'recall']
                for key in metric_keys:
                    if key in reward_extra_info:
                        values = reward_extra_info[key]
                        if isinstance(values, list):
                            combined_batch.non_tensor_batch[key] = np.array([float(v) for v in values])
                            print(f"[DEBUG] Extracted {key}: {len(values)} values, mean={np.mean(values):.4f}")
                        else:
                            print(f"[DEBUG] WARNING: {key} is not a list: {type(values)}")
                    else:
                        print(f"[DEBUG] WARNING: {key} not found in reward_extra_info keys: {list(reward_extra_info.keys())}")
                
                print(f"[DEBUG] reward_extra_info keys: {list(reward_extra_info.keys())}")
            else:
                print(f"[DEBUG] WARNING: reward_extra_info is not a dict: {type(reward_extra_info)}")
        else:
            print(f"[DEBUG] WARNING: No reward_extra_info in reward_results keys: {reward_results.keys()}")
        
        # Compute pass@k metrics for all k values
        all_metrics = {}
        
        for k in self.k_values:
            passatk_metrics = self._compute_passatk_for_k(combined_batch, k)
            all_metrics.update(passatk_metrics)
        
        # Add single-value metrics (F1, precision, recall) - no @K needed
        if hasattr(combined_batch, 'non_tensor_batch') and combined_batch.non_tensor_batch:
            non_tensor_batch = combined_batch.non_tensor_batch
            
            # Add single metrics
            single_metrics = ['f1', 'precision', 'recall']
            for metric_name in single_metrics:
                if metric_name in non_tensor_batch:
                    values = non_tensor_batch[metric_name]
                    all_metrics[f"{metric_name}/mean"] = float(np.mean(values))
                    all_metrics[f"{metric_name}/std"] = float(np.std(values))
                    print(f"[DEBUG] Added {metric_name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")
        
        # Add length metrics (only mean, no std needed for lengths)
        length_metrics = self._compute_length_metrics(combined_batch)
        all_metrics.update(length_metrics)
        
        # Store length metrics for FLOP calculations
        self.last_length_metrics = length_metrics
        
        # Add FLOP metrics
        flop_metrics = self._compute_flops_metrics(combined_batch)
        all_metrics.update(flop_metrics)
        
        return all_metrics
    
    def _compute_passatk_for_k(self, batch: DataProto, k: int, num_bootstrap: int = 100) -> Dict[str, Any]:
        """
        Compute pass@k metrics for a specific k value.
        
        Args:
            batch: DataProto with UID grouping and reward components
            k: The k value for pass@k computation
            num_bootstrap: Number of bootstrap samples
            
        Returns:
            Dictionary with pass@k metrics for this k value
        """
        # Check if we have necessary data
        if not hasattr(batch, 'non_tensor_batch') or batch.non_tensor_batch is None:
            return {}
        
        non_tensor_batch = batch.non_tensor_batch
        if 'uid' not in non_tensor_batch:
            return {}
        
        # Get UID info and scores
        uid_info = non_tensor_batch['uid']
        batch_size = len(uid_info)
        
        # Get exact match and retrieval quality scores
        em_scores = non_tensor_batch.get('em_score', non_tensor_batch.get('exact_match', np.zeros(batch_size)))
        retrieval_scores = non_tensor_batch.get('retrieval_score', non_tensor_batch.get('retrieval_quality', np.zeros(batch_size)))
        
        # Get F1, precision, recall scores for F1@K metrics
        f1_scores = non_tensor_batch.get('f1', np.zeros(batch_size))
        precision_scores = non_tensor_batch.get('precision', np.zeros(batch_size))
        recall_scores = non_tensor_batch.get('recall', np.zeros(batch_size))
        
        print(f"[DEBUG] Pass@{k} computation:")
        print(f"[DEBUG] - batch_size: {batch_size}")
        print(f"[DEBUG] - non_tensor_batch keys: {list(non_tensor_batch.keys())}")
        print(f"[DEBUG] - em_scores: {type(em_scores)}, shape: {getattr(em_scores, 'shape', 'N/A')}, mean: {np.mean(em_scores):.4f}")
        print(f"[DEBUG] - retrieval_scores: {type(retrieval_scores)}, shape: {getattr(retrieval_scores, 'shape', 'N/A')}, mean: {np.mean(retrieval_scores):.4f}")
        print(f"[DEBUG] - f1_scores: {type(f1_scores)}, shape: {getattr(f1_scores, 'shape', 'N/A')}, mean: {np.mean(f1_scores):.4f}")
        
        # Initialize result arrays
        pass_metrics = {
            f'exact_match_pass@{k}': np.zeros(batch_size),
            f'retrieval_quality_pass@{k}': np.zeros(batch_size),
            f'f1_pass@{k}': np.zeros(batch_size),
            f'precision_pass@{k}': np.zeros(batch_size),
            f'recall_pass@{k}': np.zeros(batch_size)
        }
        
        # Group by UID
        uid_to_indices = group_by_uid(list(range(batch_size)), uid_info)
        
        # Compute pass@k for each group
        for uid, indices in uid_to_indices.items():
            if len(indices) < k:
                continue  # Not enough responses
            
            # Get scores for this group
            group_em_scores = [float(em_scores[i]) for i in indices]
            group_retrieval_scores = [float(retrieval_scores[i]) for i in indices]
            group_f1_scores = [float(f1_scores[i]) for i in indices]
            group_precision_scores = [float(precision_scores[i]) for i in indices]
            group_recall_scores = [float(recall_scores[i]) for i in indices]
            
            # Bootstrap sampling for pass@k
            em_successes = 0
            retrieval_successes = 0
            f1_max_sum = 0.0
            precision_max_sum = 0.0
            recall_max_sum = 0.0
            
            for _ in range(num_bootstrap):
                # Sample k responses without replacement
                sampled_indices = random.sample(range(len(indices)), k)
                
                # Check if any has exact match
                has_em = any(group_em_scores[idx] > 0 for idx in sampled_indices)
                if has_em:
                    em_successes += 1
                
                # Check if any has retrieval success  
                has_retrieval = any(group_retrieval_scores[idx] > 0 for idx in sampled_indices)
                if has_retrieval:
                    retrieval_successes += 1
                
                # Calculate max F1, precision, recall among k samples
                sampled_f1 = [group_f1_scores[idx] for idx in sampled_indices]
                sampled_precision = [group_precision_scores[idx] for idx in sampled_indices]
                sampled_recall = [group_recall_scores[idx] for idx in sampled_indices]
                
                f1_max_sum += max(sampled_f1) if sampled_f1 else 0.0
                precision_max_sum += max(sampled_precision) if sampled_precision else 0.0
                recall_max_sum += max(sampled_recall) if sampled_recall else 0.0
            
            # Calculate probabilities and averages
            em_prob = em_successes / num_bootstrap
            retrieval_prob = retrieval_successes / num_bootstrap
            f1_avg_max = f1_max_sum / num_bootstrap
            precision_avg_max = precision_max_sum / num_bootstrap
            recall_avg_max = recall_max_sum / num_bootstrap
            
            # Assign to all indices in group
            for idx in indices:
                pass_metrics[f'exact_match_pass@{k}'][idx] = em_prob
                pass_metrics[f'retrieval_quality_pass@{k}'][idx] = retrieval_prob
                pass_metrics[f'f1_pass@{k}'][idx] = f1_avg_max
                pass_metrics[f'precision_pass@{k}'][idx] = precision_avg_max
                pass_metrics[f'recall_pass@{k}'][idx] = recall_avg_max
        
        # Convert to metric format (only mean and std, no max/min)
        result_metrics = {}
        for metric_name, values in pass_metrics.items():
            result_metrics[f"{metric_name}/mean"] = float(np.mean(values))
            result_metrics[f"{metric_name}/std"] = float(np.std(values))
        
        print(f"[DEBUG] Pass@{k} final metrics:")
        for metric_name, value in result_metrics.items():
            if 'mean' in metric_name:
                print(f"[DEBUG]   {metric_name}: {value:.4f}")
        
        return result_metrics
    
    def _compute_length_metrics(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute length metrics in tokens for responses.
        
        Three types of lengths:
        1. total_length: prompt + llm_responses + observations (everything)
        2. response_length: llm_responses + observations (without prompt)  
        3. generation_length: llm_responses only (pure generation)
        
        Args:
            batch: DataProto with tokenized sequences
            
        Returns:
            Dict with length metrics (mean only)
        """
        if not hasattr(batch, 'batch') or batch.batch is None:
            return {}
        
        batch_data = batch.batch
        length_metrics = {}
        
        # 1. Total length: everything (input_ids)
        if 'input_ids' in batch_data:
            input_ids = batch_data['input_ids']
            if hasattr(input_ids, 'shape'):
                total_lengths = []
                for seq in input_ids:
                    # Count non-padding tokens (assuming pad_token_id or 0 for padding)
                    if hasattr(seq, 'ne'):  # torch tensor
                        non_pad_length = (seq != 0).sum().item()  # Assuming 0 is pad token
                    else:
                        non_pad_length = len([t for t in seq if t != 0])
                    total_lengths.append(non_pad_length)
                
                if total_lengths:
                    length_metrics['total_length/mean'] = float(np.mean(total_lengths))
                    print(f"[DEBUG] total_length: mean={np.mean(total_lengths):.1f} tokens")
        
        # 2. Response length: responses + observations (if available)
        if 'responses' in batch_data:
            responses = batch_data['responses']  
            if hasattr(responses, 'shape'):
                response_lengths = []
                for seq in responses:
                    if hasattr(seq, 'ne'):  # torch tensor
                        non_pad_length = (seq != 0).sum().item()
                    else:
                        non_pad_length = len([t for t in seq if t != 0])
                    response_lengths.append(non_pad_length)
                
                if response_lengths:
                    length_metrics['response_length/mean'] = float(np.mean(response_lengths))
                    print(f"[DEBUG] response_length: mean={np.mean(response_lengths):.1f} tokens")
        
        # 3. Generation length: Pure LLM responses (excluding KG observations)
        # Need to parse the actual text to separate LLM responses from <information>...</information>
        if 'responses' in batch_data and hasattr(batch, 'batch'):
            responses = batch_data['responses']
            if hasattr(responses, 'shape'):
                generation_lengths = []
                for i, seq in enumerate(responses):
                    try:
                        # Decode the response to text to analyze content
                        if hasattr(seq, 'tolist'):
                            token_ids = seq.tolist()
                        else:
                            token_ids = seq
                        
                        # Decode tokens to text
                        response_text = self.tokenizer.decode(token_ids, skip_special_tokens=True)
                        
                        # Count tokens in LLM-generated parts (exclude <information>...</information>)
                        llm_text = self._extract_llm_only_text(response_text)
                        llm_tokens = self.tokenizer.encode(llm_text, add_special_tokens=False)
                        generation_lengths.append(len(llm_tokens))
                        
                    except Exception as e:
                        # Fallback: use full response length if parsing fails
                        if hasattr(seq, 'ne'):
                            fallback_length = (seq != 0).sum().item()
                        else:
                            fallback_length = len([t for t in seq if t != 0])
                        generation_lengths.append(fallback_length)
                
                if generation_lengths:
                    length_metrics['generation_length/mean'] = float(np.mean(generation_lengths))
                    print(f"[DEBUG] generation_length: mean={np.mean(generation_lengths):.1f} tokens (LLM only)")
        
        # Debug: Show the differences
        if 'total_length/mean' in length_metrics and 'response_length/mean' in length_metrics and 'generation_length/mean' in length_metrics:
            total = length_metrics['total_length/mean']
            response = length_metrics['response_length/mean'] 
            generation = length_metrics['generation_length/mean']
            prompt_approx = total - response
            kg_info_approx = response - generation
            print(f"[DEBUG] Length breakdown: prompt≈{prompt_approx:.0f}, generation≈{generation:.0f}, kg_info≈{kg_info_approx:.0f}, total={total:.0f}")
        
        return length_metrics
    
    def _compute_flops_metrics(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute FLOPs (Floating Point Operations) used during KG-R1 generation.
        
        KG-R1 Generation Pattern (different from pure autoregressive):
        Turn 1: prompt → generate search/action → observe KG info
        Turn 2: prompt + action1 + info1 → generate action2 → observe info2
        Turn N: prompt + history → generate final answer
        
        FLOP calculation:
        - Each turn: 2 * params * context_length * tokens_generated_in_turn
        - Context grows after each turn with: action_tokens + KG_info_tokens
        - NOT pure autoregressive (each turn processes full context independently)
        
        Args:
            batch: DataProto with tokenized sequences
            
        Returns:
            Dict with FLOP metrics including KG-R1 specific calculations
        """
        if not hasattr(batch, 'batch') or batch.batch is None:
            return {}
        
        flops_metrics = {}
        batch_data = batch.batch
        
        # Get model configuration parameters
        model_config = self._get_model_config()
        if not model_config:
            print("[DEBUG] Could not retrieve model config for FLOP calculation")
            return flops_metrics
        
        # Extract model parameters
        hidden_size = model_config.get('hidden_size', 0)
        num_layers = model_config.get('num_hidden_layers', 0) or model_config.get('num_layers', 0)
        vocab_size = model_config.get('vocab_size', 0)
        
        if not all([hidden_size, num_layers, vocab_size]):
            print(f"[DEBUG] Missing model config: hidden_size={hidden_size}, num_layers={num_layers}, vocab_size={vocab_size}")
            return flops_metrics
        
        # Estimate model parameters (simplified calculation)
        # Transformer parameters ≈ 12 * num_layers * hidden_size^2 + vocab_size * hidden_size
        transformer_params = 12 * num_layers * (hidden_size ** 2)  # Main transformer blocks
        embedding_params = vocab_size * hidden_size  # Embedding layer
        total_params = transformer_params + embedding_params
        
        print(f"[DEBUG] Model config: layers={num_layers}, hidden={hidden_size}, vocab={vocab_size}")
        print(f"[DEBUG] Estimated parameters: {total_params/1e9:.2f}B ({transformer_params/1e9:.2f}B transformer + {embedding_params/1e9:.2f}B embedding)")
        
        # Calculate FLOPs for different length types
        flop_calculations = []
        
        # 1. Generation FLOPs: KG-R1 multi-turn generation cost using interaction history
        if 'responses' in batch_data and hasattr(batch, 'meta_info') and 'interaction_history' in batch.meta_info:
            responses = batch_data['responses']
            interaction_history = batch.meta_info['interaction_history']
            
            if hasattr(responses, 'shape') and interaction_history:
                generation_flops_list = []
                for i, response_seq in enumerate(responses):
                    if i >= len(interaction_history):
                        # Fallback for missing interaction history
                        gen_length = (response_seq != 0).sum().item() if hasattr(response_seq, 'ne') else len([t for t in response_seq if t != 0])
                        fallback_flops = 2 * total_params * 512 * gen_length  # Simple fallback
                        generation_flops_list.append(fallback_flops)
                        continue
                    
                    sample_history = interaction_history[i]
                    actions = sample_history.get('actions', [])
                    search_results = sample_history.get('search_results', [])
                    responses_str = sample_history.get('responses_str', [])
                    
                    if not actions:
                        # No interaction history available, use fallback
                        gen_length = (response_seq != 0).sum().item() if hasattr(response_seq, 'ne') else len([t for t in response_seq if t != 0])
                        fallback_flops = 2 * total_params * 512 * gen_length
                        generation_flops_list.append(fallback_flops)
                        continue
                    
                    # Calculate FLOPs based on actual interaction history
                    total_flops = 0
                    current_context_length = 512  # Initial prompt length
                    
                    for turn_idx, (action, search_result, response_str) in enumerate(zip(actions, search_results, responses_str)):
                        if not response_str:  # Skip empty turns
                            continue
                            
                        # Tokenize the actual generated response for this turn
                        action_tokens = self.tokenizer.encode(response_str, add_special_tokens=False) if response_str else []
                        action_token_count = len(action_tokens)
                        
                        # Forward pass FLOPs for this turn's generation
                        turn_flops = 2 * total_params * current_context_length * action_token_count
                        total_flops += turn_flops
                        
                        # Update context length with generated action
                        current_context_length += action_token_count
                        
                        # Add KG observation tokens to context if search was performed
                        if search_result:
                            kg_tokens = self.tokenizer.encode(search_result, add_special_tokens=False) if search_result else []
                            current_context_length += len(kg_tokens)
                    
                    generation_flops_list.append(total_flops)
                
                if generation_flops_list:
                    mean_gen_flops = float(np.mean(generation_flops_list))
                    flops_metrics['generation_flops/mean'] = mean_gen_flops
                    flops_metrics['generation_gflops/mean'] = mean_gen_flops / 1e9  # GFLOPs
                    print(f"[DEBUG] generation_flops (KG-R1 interaction-based): mean={mean_gen_flops/1e9:.2f} GFLOPs")
                    
                    # Add detailed turn-level debug info from interaction history
                    if interaction_history and len(interaction_history) > 0:
                        sample_history = interaction_history[0]
                        actions = sample_history.get('actions', [])
                        search_results = sample_history.get('search_results', [])
                        responses_str = sample_history.get('responses_str', [])
                        
                        print(f"[DEBUG] Sample interaction breakdown: {len(actions)} turns")
                        context_len = 512
                        total_action_tokens = 0
                        total_kg_tokens = 0
                        
                        for turn_idx, (action, search_result, response_str) in enumerate(zip(actions[:3], search_results[:3], responses_str[:3])):  # Show first 3 turns
                            if response_str:
                                action_tokens = len(self.tokenizer.encode(response_str, add_special_tokens=False))
                                kg_tokens = len(self.tokenizer.encode(search_result, add_special_tokens=False)) if search_result else 0
                                total_action_tokens += action_tokens
                                total_kg_tokens += kg_tokens
                                print(f"[DEBUG]   Turn {turn_idx+1}: context={context_len}, action={action_tokens}t, kg={kg_tokens}t")
                                context_len += action_tokens + kg_tokens
                        
                        if len(actions) > 3:
                            print(f"[DEBUG]   ... ({len(actions)-3} more turns)")
                        print(f"[DEBUG] Total tokens: {total_action_tokens} action + {total_kg_tokens} KG = {total_action_tokens + total_kg_tokens}")
                    
                    flop_calculations.append(('generation', mean_gen_flops))
        
        # Fallback: If no interaction history available, use estimation method
        elif 'responses' in batch_data and 'input_ids' in batch_data:
            responses = batch_data['responses']
            input_ids = batch_data['input_ids']
            if hasattr(responses, 'shape') and hasattr(input_ids, 'shape'):
                print("[DEBUG] No interaction history available, using estimation method")
                generation_flops_list = []
                for i, (response_seq, input_seq) in enumerate(zip(responses, input_ids)):
                    if hasattr(response_seq, 'ne'):  # torch tensor
                        gen_length = (response_seq != 0).sum().item()
                        total_length = (input_seq != 0).sum().item()
                    else:
                        gen_length = len([t for t in response_seq if t != 0])
                        total_length = len([t for t in input_seq if t != 0])
                    
                    prompt_length = total_length - gen_length
                    
                    # Estimated multi-turn calculation
                    estimated_turns = min(7, max(1, gen_length // 50))  # ~50 tokens per turn
                    avg_tokens_per_turn = gen_length / estimated_turns if estimated_turns > 0 else gen_length
                    kg_info_length = prompt_length - 512  # Assume ~512 initial prompt
                    kg_info_per_turn = max(0, kg_info_length / max(1, estimated_turns - 1))
                    
                    turn_flops = 0
                    current_context = 512  # Initial prompt
                    for turn in range(estimated_turns):
                        turn_flops += 2 * total_params * current_context * avg_tokens_per_turn
                        current_context += avg_tokens_per_turn + kg_info_per_turn
                    
                    generation_flops_list.append(turn_flops)
                
                if generation_flops_list:
                    mean_gen_flops = float(np.mean(generation_flops_list))
                    flops_metrics['generation_flops/mean'] = mean_gen_flops
                    flops_metrics['generation_gflops/mean'] = mean_gen_flops / 1e9  # GFLOPs
                    print(f"[DEBUG] generation_flops (KG-R1 estimated): mean={mean_gen_flops/1e9:.2f} GFLOPs")
                    flop_calculations.append(('generation', mean_gen_flops))
        
        # 2. Total FLOPs: Forward pass for entire sequence (prompt + generation)
        if 'input_ids' in batch_data:
            input_ids = batch_data['input_ids']
            if hasattr(input_ids, 'shape'):
                total_flops_list = []
                for seq in input_ids:
                    if hasattr(seq, 'ne'):  # torch tensor
                        total_length = (seq != 0).sum().item()
                    else:
                        total_length = len([t for t in seq if t != 0])
                    
                    # Total forward pass FLOPs
                    total_flops = 2 * total_params * total_length
                    total_flops_list.append(total_flops)
                
                if total_flops_list:
                    mean_total_flops = float(np.mean(total_flops_list))
                    flops_metrics['total_flops/mean'] = mean_total_flops
                    flops_metrics['total_gflops/mean'] = mean_total_flops / 1e9  # GFLOPs
                    print(f"[DEBUG] total_flops: mean={mean_total_flops/1e9:.2f} GFLOPs")
                    flop_calculations.append(('total', mean_total_flops))
        
        # 3. Compute FLOPs per token ratios
        if 'generation_flops/mean' in flops_metrics and 'generation_length/mean' in self.last_length_metrics:
            gen_flops = flops_metrics['generation_flops/mean']
            gen_length = self.last_length_metrics.get('generation_length/mean', 1)
            if gen_length > 0:
                flops_per_token = gen_flops / gen_length
                flops_metrics['flops_per_gen_token/mean'] = flops_per_token
                flops_metrics['gflops_per_gen_token/mean'] = flops_per_token / 1e9
                print(f"[DEBUG] flops_per_gen_token: {flops_per_token/1e9:.4f} GFLOPs/token")
        
        # Add model parameter info to metrics for reference
        flops_metrics['model_parameters/total'] = float(total_params)
        flops_metrics['model_parameters/billions'] = float(total_params / 1e9)
        
        # Debug: Show FLOP breakdown
        if len(flop_calculations) >= 2:
            gen_flops = next((f for name, f in flop_calculations if name == 'generation'), 0)
            total_flops = next((f for name, f in flop_calculations if name == 'total'), 0)
            prompt_flops = total_flops - gen_flops
            print(f"[DEBUG] FLOP breakdown: prompt≈{prompt_flops/1e9:.1f}G, generation≈{gen_flops/1e9:.1f}G, total={total_flops/1e9:.1f}G")
        
        return flops_metrics
    
    def _get_model_config(self) -> Optional[Dict[str, Any]]:
        """
        Retrieve model configuration for FLOP calculation.
        
        Returns:
            Dict with model config parameters or None if unavailable
        """
        try:
            # Try to get model config from the config
            model_path = self.config.actor_rollout_ref.model.path
            
            # Handle common model paths
            if 'Qwen2.5-3B' in model_path:
                # Qwen2.5-3B-Instruct configuration
                return {
                    'hidden_size': 2048,
                    'num_hidden_layers': 36,
                    'num_attention_heads': 16,
                    'vocab_size': 151936,
                    'model_type': 'qwen2_5'
                }
            elif 'Qwen2.5-7B' in model_path:
                # Qwen2.5-7B-Instruct configuration  
                return {
                    'hidden_size': 4096,
                    'num_hidden_layers': 32,
                    'num_attention_heads': 32,
                    'vocab_size': 151936,
                    'model_type': 'qwen2_5'
                }
            elif 'Llama-2-7b' in model_path or 'llama2-7b' in model_path:
                # Llama-2-7B configuration
                return {
                    'hidden_size': 4096,
                    'num_hidden_layers': 32,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'model_type': 'llama'
                }
            else:
                # Generic fallback - try to infer from model name
                if '3B' in model_path or '3b' in model_path:
                    return {
                        'hidden_size': 2048,
                        'num_hidden_layers': 36,
                        'num_attention_heads': 16,
                        'vocab_size': 50000,  # Generic fallback
                        'model_type': 'unknown'
                    }
                elif '7B' in model_path or '7b' in model_path:
                    return {
                        'hidden_size': 4096,
                        'num_hidden_layers': 32,
                        'num_attention_heads': 32,
                        'vocab_size': 50000,  # Generic fallback
                        'model_type': 'unknown'
                    }
                else:
                    print(f"[DEBUG] Unknown model path for FLOP calculation: {model_path}")
                    return None
        
        except Exception as e:
            print(f"[DEBUG] Error getting model config: {e}")
            return None
    
    def _extract_llm_only_text(self, response_text: str) -> str:
        """
        Extract only LLM-generated text, excluding KG server responses.
        
        Args:
            response_text: Full response text containing LLM responses and KG observations
            
        Returns:
            Text with only LLM-generated content (no <information>...</information> blocks)
        """
        import re
        
        # Remove <information>...</information> blocks (KG server responses)
        llm_text = re.sub(r'<information>.*?</information>', '', response_text, flags=re.DOTALL)
        
        # Also remove any other KG-related tags that might be included
        llm_text = re.sub(r'<kg-query>.*?</kg-query>', '', llm_text, flags=re.DOTALL)
        
        # Clean up extra whitespace
        llm_text = ' '.join(llm_text.split())
        
        return llm_text
    
    def _aggregate_metrics(self, batch_metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate metrics across all batches.
        
        Args:
            batch_metrics_list: List of metric dictionaries from each batch
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not batch_metrics_list:
            return {}
        
        # Collect all metric keys
        all_keys = set()
        for metrics in batch_metrics_list:
            all_keys.update(metrics.keys())
        
        # Remove non-numeric keys
        non_numeric_keys = {'inputs', 'outputs', 'scores', 'reward_extra_infos'}
        numeric_keys = all_keys - non_numeric_keys
        
        # Aggregate numeric metrics
        aggregated = {}
        for key in numeric_keys:
            values = []
            for metrics in batch_metrics_list:
                if key in metrics and isinstance(metrics[key], (int, float)):
                    values.append(metrics[key])
            
            if values:
                aggregated[key] = np.mean(values)
        
        return aggregated


def create_kg_evaluator(
    config,
    tokenizer,
    processor, 
    device_name: str = "cuda",
    n_rollout_eval: int = 8,
    k_values: List[int] = None,
    val_reward_fn=None,
    eval_samples: int = 0,
    **kwargs
) -> RayKGEvaluator:
    """
    Factory function to create KG evaluator.
    
    Args:
        config: Training configuration
        tokenizer: Tokenizer for text processing
        processor: Data processor
        device_name: Device to use for computation
        n_rollout_eval: Number of responses to generate per prompt
        k_values: List of k values for pass@k computation
        val_reward_fn: Validation reward function
        eval_samples: Number of samples to evaluate (0 for all)
        **kwargs: Additional arguments
        
    Returns:
        Initialized RayKGEvaluator instance
    """
    import ray
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    
    # Set up worker classes based on strategy (copied from main_ppo.py)
    # Default to fsdp strategy if not specified
    actor_strategy = config.get("actor_rollout_ref", {}).get("actor", {}).get("strategy", "fsdp")
    rollout_mode = config.get("actor_rollout_ref", {}).get("rollout", {}).get("mode", "sync")
    
    if actor_strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        
        actor_rollout_cls = AsyncActorRolloutRefWorker if rollout_mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
        
    elif actor_strategy == "megatron":
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        from verl.workers.megatron_workers import ActorRolloutRefWorker, AsyncActorRolloutRefWorker, CriticWorker
        
        actor_rollout_cls = AsyncActorRolloutRefWorker if rollout_mode == "async" else ActorRolloutRefWorker
        ray_worker_group_cls = NVMegatronRayWorkerGroup
        
    else:
        raise NotImplementedError(f"Strategy {actor_strategy} not implemented")
    
    # Set up role worker mapping
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
        Role.Critic: ray.remote(CriticWorker),
    }
    
    # Set up resource pool manager
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
    }
    
    # Add reward model worker if enabled
    reward_model_enabled = config.get("reward_model", {}).get("enable", False)
    if reward_model_enabled:
        reward_model_strategy = config.get("reward_model", {}).get("strategy", "fsdp")
        if reward_model_strategy in ["fsdp", "fsdp2"]:
            from verl.workers.fsdp_workers import RewardModelWorker
        elif reward_model_strategy == "megatron":
            from verl.workers.megatron_workers import RewardModelWorker
        else:
            raise NotImplementedError(f"Reward model strategy {reward_model_strategy} not implemented")
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id
    
    # Add reference model if needed
    use_kl_in_reward = config.get("algorithm", {}).get("use_kl_in_reward", False)
    use_kl_loss = config.get("actor_rollout_ref", {}).get("actor", {}).get("use_kl_loss", False)
    if use_kl_in_reward or use_kl_loss:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    return RayKGEvaluator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        device_name=device_name,
        n_rollout_eval=n_rollout_eval,
        k_values=k_values,
        val_reward_fn=val_reward_fn,
        eval_samples=eval_samples,
        **kwargs
    )