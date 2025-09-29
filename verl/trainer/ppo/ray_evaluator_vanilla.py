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
Pure Vanilla Evaluator for KGQA benchmarks without KG augmentation.

This evaluator provides a clean benchmark evaluation using VERL framework:
- No KG server integration
- No special formatting requirements  
- Simple prompt -> response evaluation
- Standard NLP metrics (exact match, F1)
- Pass@K evaluation
"""

import json
import os
from collections import defaultdict
from typing import Dict, Any, List
import numpy as np
import torch
import ray
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayWorkerGroup
from verl.trainer.ppo.ray_trainer_kg import RayPPOTrainer
from verl.utils.metric import reduce_metrics




def compute_f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth."""
    def normalize_text(text: str) -> str:
        return text.lower().strip()
    
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    common_tokens = set(pred_tokens) & set(gt_tokens)
    
    if not common_tokens:
        return 0.0
        
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    return 2 * (precision * recall) / (precision + recall)


def compute_exact_match(prediction: str, ground_truth: str) -> bool:
    """Compute exact match between prediction and ground truth."""
    def normalize_text(text: str) -> str:
        return text.lower().strip()
    
    return normalize_text(prediction) == normalize_text(ground_truth)


def compute_pass_at_k(results: List[bool], k: int) -> float:
    """Compute Pass@K metric from boolean results."""
    if len(results) < k:
        return 0.0
    return float(any(results[:k]))


class RayVanillaEvaluator(RayPPOTrainer):
    """
    Pure vanilla evaluator based on RayPPOTrainer but without KG integration.
    
    This evaluator:
    - Uses VERL's efficient actor_rollout_wg.generate_sequences() for batched generation  
    - Applies vanilla prompt augmentation
    - Generates responses without KG feedback
    - Computes standard NLP metrics
    - No special formatting requirements
    """
    
    def __init__(
        self,
        config,
        tokenizer,
        processor,
        role_worker_mapping,
        resource_pool_manager,
        ray_worker_group_cls,
        device_name: str = "cuda", 
        n_rollout_eval: int = 4,
        k_values: List[int] = [1, 2, 3, 4],
        eval_samples: int = 0
    ):
        """Initialize vanilla evaluator using VERL infrastructure."""
        
        # Initialize parent with all required parameters (same as ray_evaluator_kg)
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=ray_worker_group_cls,
            device_name=device_name,
            reward_fn=None,  # Not needed for evaluation
            val_reward_fn=None,  # Skip reward manager for vanilla
            train_dataset=None,  # Not needed for evaluation
            val_dataset=None,    # Not needed for evaluation 
            collate_fn=None,     # Not needed for evaluation
            train_sampler=None   # Not needed for evaluation
        )
        
        # Vanilla evaluation parameters
        self.n_rollout_eval = n_rollout_eval
        self.k_values = k_values
        
        # Cache for ground truth data
        self._ground_truth_cache = None
        self._load_ground_truth_data()
        self.eval_samples = eval_samples  # 0 means evaluate all samples
        
        # Validate k_values
        max_k = max(self.k_values)
        if max_k > self.n_rollout_eval:
            raise ValueError(f"Max k value ({max_k}) cannot exceed n_rollout_eval ({self.n_rollout_eval})")
        
        # CRITICAL: Override parent class settings for vanilla evaluation
        self.use_search_generation = False  # Disable KG search completely
        self.use_critic = False             # No critic needed for evaluation
        self.use_reference_policy = False   # No reference policy needed
        self.use_rm = False                 # No reward model needed for simple evaluation
        
        # Set async rollout mode to False for vanilla evaluation (simpler synchronous mode)
        self.async_rollout_mode = False
        
        # LLM-as-Judge configuration
        self.use_llm_judge = True  # Enable/disable LLM-based evaluation
        
        # Fallback rate tracking
        self.llm_judge_total_calls = 0
        self.llm_judge_successful_calls = 0
        self.llm_judge_fallbacks = 0
        self.vector_extraction_failures = 0
        
        # Placeholder for worker group - will be set by init_workers()
        self.actor_rollout_wg = None
        
        print(f"Vanilla Evaluator initialized with n_rollout_eval={self.n_rollout_eval}, k_values={self.k_values}")
        print(f"eval_samples={self.eval_samples} (0 = evaluate all samples)")
        print(f"Using efficient VERL batched generation (search disabled)")
        print(f"Note: actor_rollout_wg will be initialized by init_workers() call")
    
    def _load_ground_truth_data(self):
        """Load ground truth data directly from the dataset file."""
        try:
            import pandas as pd
            
            # Get the val dataset path from config
            val_files = self.config.data.get('val_files', self.config.data.get('train_files'))
            if isinstance(val_files, list):
                val_files = val_files[0]  # Use first file
            
            print(f"[LLM-JUDGE] Loading ground truth from: {val_files}")
            
            # Load parquet directly
            dataset = pd.read_parquet(val_files)
            
            # Cache the ground truth data
            self._ground_truth_cache = []
            for idx, row in dataset.iterrows():
                # Handle different data formats
                reward_model = row.get('reward_model', {})
                if isinstance(reward_model, dict):
                    ground_truth = reward_model.get('ground_truth', [])
                else:
                    ground_truth = []
                
                ground_truth_data = {
                    'ground_truth': ground_truth,
                    'prompt': row.get('prompt', ''),
                    'question': row.get('question', ''),
                    'target_text': ground_truth.get('target_text', []) if isinstance(ground_truth, dict) else []
                }
                self._ground_truth_cache.append(ground_truth_data)
            
            print(f"[LLM-JUDGE] Loaded {len(self._ground_truth_cache)} ground truth samples")
            
            # Debug first sample structure
            if self._ground_truth_cache:
                sample0 = self._ground_truth_cache[0]
                print(f"[LLM-JUDGE DEBUG] Sample 0 ground truth structure:")
                print(f"  ground_truth type: {type(sample0['ground_truth'])}")
                print(f"  target_text: {sample0['target_text']}")
                if isinstance(sample0['ground_truth'], dict):
                    gt = sample0['ground_truth']
                    print(f"  ground_truth keys: {list(gt.keys())}")
                    if 'target_text' in gt:
                        print(f"  target_text in gt: {gt['target_text']}")
            
        except Exception as e:
            print(f"[LLM-JUDGE] Warning: Could not load ground truth data: {e}")
            self._ground_truth_cache = []
    
    def init_workers(self):
        """Initialize workers with debugging for vanilla evaluation."""
        print("=== VANILLA EVALUATOR: Starting init_workers() ===")
        print(f"self.hybrid_engine = {getattr(self, 'hybrid_engine', 'NOT SET')}")
        print(f"self.use_critic = {getattr(self, 'use_critic', 'NOT SET')}")
        print(f"self.use_reference_policy = {getattr(self, 'use_reference_policy', 'NOT SET')}")
        print(f"self.use_rm = {getattr(self, 'use_rm', 'NOT SET')}")
        
        # Call parent class method
        super().init_workers()
        
        print(f"=== VANILLA EVALUATOR: After parent init_workers() ===")
        print(f"self.actor_rollout_wg = {getattr(self, 'actor_rollout_wg', 'NOT SET')}")
        if hasattr(self, 'actor_rollout_wg') and self.actor_rollout_wg is not None:
            print(f"SUCCESS: actor_rollout_wg is properly initialized")
        else:
            print(f"ERROR: actor_rollout_wg is still None or missing")
        print("=== VANILLA EVALUATOR: init_workers() completed ===")
        
        return self
    
    def evaluate_dataset(self, dataset_name: str = "test") -> Dict[str, Any]:
        """
        Run vanilla evaluation using efficient VERL batched generation.
        
        Args:
            dataset_name: Name of the dataset being evaluated
            
        Returns:
            Dictionary with Pass@K metrics and other statistics
        """
        print(f"Starting vanilla evaluation for {dataset_name} dataset")
        print(f"Generating {self.n_rollout_eval} responses per prompt")
        print(f"Computing pass@k for k values: {self.k_values}")
        
        # Use validation dataset for evaluation (same as ray_evaluator_kg)
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
        
        for batch_idx, test_data in enumerate(tqdm(dataloader, desc="Evaluating batches")):
            print(f"\nProcessing batch {batch_idx + 1}/{len(dataloader)}")
            
            # Use efficient batched evaluation (same as ray_evaluator_kg)
            batch_metrics = self._evaluate_batch_efficient(test_data)
            all_metrics.append(batch_metrics)
            
            # Collect data for analysis
            if 'inputs' in batch_metrics:
                all_inputs.extend(batch_metrics['inputs'])
            if 'outputs' in batch_metrics:
                all_outputs.extend(batch_metrics['outputs'])
            if 'scores' in batch_metrics:
                all_scores.extend(batch_metrics['scores'])
            
            # For now, process all batches (sample limiting can be added later)
        
        # Compute final metrics using the same approach as ray_evaluator_kg
        return self._compute_final_metrics_efficient(all_metrics)
    
    def _evaluate_batch_efficient(self, test_data) -> Dict[str, Any]:
        """
        Efficient batch evaluation using VERL's actor_rollout_wg.generate_sequences().
        Based on the pattern from ray_trainer_kg.py but without KG integration.
        """
        import uuid
        import numpy as np
        import torch
        
        # Step 1: Prepare generation batch (same as ray_trainer_kg)
        # Handle both dict and DataProto input formats
        if isinstance(test_data, dict):
            # Convert dict to DataProto format expected by VERL
            from verl import DataProto
            test_gen_batch = DataProto.from_dict(test_data)
        else:
            test_gen_batch = test_data
        
        # Add UIDs for tracking responses
        batch_size = len(test_gen_batch.batch) if hasattr(test_gen_batch, 'batch') else len(test_gen_batch)
        base_uids = [str(uuid.uuid4()) for _ in range(batch_size)]
        
        # Ensure non_tensor_batch exists
        if not hasattr(test_gen_batch, 'non_tensor_batch') or test_gen_batch.non_tensor_batch is None:
            test_gen_batch.non_tensor_batch = {}
        test_gen_batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
        
        # Step 2: Expand batch for n_rollout_eval responses per prompt
        expanded_gen_batch = test_gen_batch.repeat(repeat_times=self.n_rollout_eval, interleave=True)
        
        # Initialize meta_info if needed
        if expanded_gen_batch.meta_info is None:
            expanded_gen_batch.meta_info = {}
            
        # Store original dataset information for LLM-as-Judge evaluation
        if self.use_llm_judge:
            # The test_data contains the original dataset information
            # We need to pass this through so we can access ground truth and prompts
            expanded_gen_batch.meta_info['original_test_data'] = test_data
        
        # Step 2.5: Drop samples to make batch divisible by number of GPU chunks (4)
        original_batch_size = len(expanded_gen_batch.batch) if hasattr(expanded_gen_batch, 'batch') else len(expanded_gen_batch)
        num_gpu_chunks = 4  # VERL uses 4 GPUs
        
        if original_batch_size % num_gpu_chunks != 0:
            # Calculate how many samples to drop
            samples_to_drop = original_batch_size % num_gpu_chunks
            new_batch_size = original_batch_size - samples_to_drop
            print(f"[BATCH-SIZE] Dropping {samples_to_drop} samples to make batch size {original_batch_size} -> {new_batch_size} (divisible by {num_gpu_chunks})")
            
            if new_batch_size == 0:
                print(f"[BATCH-SIZE] Warning: Batch size would become 0 after dropping samples. Skipping this batch.")
                # Return empty results for this batch
                return {
                    'exact_match/mean': 0.0,
                    'exact_match/std': 0.0,
                    'f1/mean': 0.0,
                    'f1/std': 0.0,
                    'generation_length/mean': 0.0,
                    'num_samples': 0
                }
            
            # Create a new DataProto with the correct batch size by slicing
            from verl import DataProto
            
            # Create truncated batch data
            truncated_batch = {}
            if hasattr(expanded_gen_batch, 'batch') and expanded_gen_batch.batch is not None:
                for key, value in expanded_gen_batch.batch.items():
                    truncated_batch[key] = value[:new_batch_size]
            
            # Create truncated non_tensor_batch data
            truncated_non_tensor = {}
            if hasattr(expanded_gen_batch, 'non_tensor_batch') and expanded_gen_batch.non_tensor_batch is not None:
                for key, value in expanded_gen_batch.non_tensor_batch.items():
                    truncated_non_tensor[key] = value[:new_batch_size]
            
            # Create new DataProto with correct batch size
            expanded_gen_batch = DataProto.from_dict(truncated_batch)
            if truncated_non_tensor:
                expanded_gen_batch.non_tensor_batch = truncated_non_tensor
            
            # Verify the new batch size is correct
            new_len = len(expanded_gen_batch)
            print(f"[BATCH-SIZE] Verification: New DataProto length = {new_len}, expected = {new_batch_size}")
            if new_len != new_batch_size:
                print(f"[BATCH-SIZE] ERROR: Length mismatch after truncation! Expected {new_batch_size}, got {new_len}")
                raise RuntimeError(f"DataProto length mismatch: expected {new_batch_size}, got {new_len}")
            
            # Restore meta_info
            if not hasattr(expanded_gen_batch, 'meta_info') or expanded_gen_batch.meta_info is None:
                expanded_gen_batch.meta_info = {}
            
            # Copy over original meta_info if it existed
            if hasattr(test_data, 'meta_info') and test_data.meta_info:
                expanded_gen_batch.meta_info.update(test_data.meta_info)
            
            # Store original test data for LLM judge
            if self.use_llm_judge:
                expanded_gen_batch.meta_info['original_test_data'] = test_data
            
            # Store the original batch size for metric calculation
            expanded_gen_batch.meta_info['original_batch_size'] = original_batch_size
            expanded_gen_batch.meta_info['dropped_batch_size'] = new_batch_size
        else:
            print(f"[BATCH-SIZE] Batch size {original_batch_size} is already divisible by {num_gpu_chunks}, no changes needed")
            expanded_gen_batch.meta_info['original_batch_size'] = original_batch_size
            expanded_gen_batch.meta_info['dropped_batch_size'] = original_batch_size
        
        # Safe batch size logging
        gen_batch_size = len(test_gen_batch.batch) if hasattr(test_gen_batch, 'batch') else len(test_gen_batch)
        expanded_batch_size = len(expanded_gen_batch.batch) if hasattr(expanded_gen_batch, 'batch') else len(expanded_gen_batch)
        print(f"Processing batch: {gen_batch_size} prompts -> {expanded_batch_size} total generations")
        
        # Step 3: Generate responses using efficient VERL batched generation with timing
        # CRITICAL: Use the same pattern as ray_trainer_kg but with use_search_generation=False
        
        import time
        generation_start_time = time.time()
        
        if not self.use_search_generation:
            # Ensure worker group is initialized
            if self.actor_rollout_wg is None:
                raise RuntimeError("actor_rollout_wg not initialized. Make sure init_workers() was called before evaluation.")
            
            # Use standard generation (efficient VLLM batching)
            if not self.async_rollout_mode:
                test_output_gen_batch = self.actor_rollout_wg.generate_sequences(expanded_gen_batch)
            else:
                self.async_rollout_manager.wake_up()
                test_output_gen_batch = self.async_rollout_manager.generate_sequences(expanded_gen_batch)
                self.async_rollout_manager.sleep()
        else:
            # Should never reach here for vanilla evaluation
            raise ValueError("Vanilla evaluation should have use_search_generation=False")
        
        generation_end_time = time.time()
        generation_time = generation_end_time - generation_start_time
        
        # Compute token counts for throughput metrics
        batch_size = len(test_output_gen_batch.batch) if hasattr(test_output_gen_batch, 'batch') else len(test_output_gen_batch)
        
        # Count input tokens (prompts)
        input_token_count = 0
        if 'input_ids' in expanded_gen_batch.batch:
            input_ids = expanded_gen_batch.batch['input_ids']
            if hasattr(input_ids, 'numel'):
                input_token_count = input_ids.numel()
            else:
                input_token_count = sum(len(seq) for seq in input_ids)
        
        # Count output tokens (responses)
        output_token_count = 0
        if 'responses' in test_output_gen_batch.batch:
            responses = test_output_gen_batch.batch['responses']
            if hasattr(responses, 'numel'):
                output_token_count = responses.numel()
            else:
                output_token_count = sum(len(seq) for seq in responses)
        
        total_token_count = input_token_count + output_token_count
        
        # Compute throughput metrics
        if generation_time > 0:
            samples_per_sec = batch_size / generation_time
            tokens_per_sec = total_token_count / generation_time
            output_tokens_per_sec = output_token_count / generation_time
        else:
            samples_per_sec = tokens_per_sec = output_tokens_per_sec = 0.0
        
        # Store timing metrics for later use
        timing_metrics = {
            'generation_time': generation_time,
            'samples_per_sec': samples_per_sec,
            'tokens_per_sec': tokens_per_sec,
            'output_tokens_per_sec': output_tokens_per_sec,
            'total_tokens': total_token_count,
            'input_tokens': input_token_count,
            'output_tokens': output_token_count,
            'batch_size': batch_size
        }
        
        print(f"[VANILLA-TIMING] Generation: {generation_time:.2f}s, {samples_per_sec:.1f} samples/sec, {tokens_per_sec:.0f} tokens/sec")
        
        # Step 4: Fix tensor mismatches before union
        # The generate_sequences function may modify tensors in the output
        # Copy all input tensors from the original batch to ensure they match
        input_tensors = ['input_ids', 'attention_mask', 'token_type_ids', 'position_ids']
        for tensor_name in input_tensors:
            if tensor_name in expanded_gen_batch.batch and tensor_name in test_output_gen_batch.batch:
                test_output_gen_batch.batch[tensor_name] = expanded_gen_batch.batch[tensor_name]
        
        # Union generated responses with input batch
        final_test_batch = expanded_gen_batch.union(test_output_gen_batch)
        
        # Step 5: Compute rewards using parallel LLM-as-Judge evaluation
        print("[VANILLA-EVAL] Using parallel LLM judge evaluation...")
        reward_results = self.compute_vanilla_rewards_with_llm_judge(final_test_batch)
        
        # Step 6: Compute Pass@K metrics
        all_metrics = self._compute_batch_passatk_metrics([final_test_batch], reward_results)
        
        # Safe batch size calculation - use actual processed batch size
        processed_batch_size = len(final_test_batch.batch) if hasattr(final_test_batch, 'batch') else len(final_test_batch)
        
        # Return metrics directly (not wrapped in passatk_metrics)
        result = {
            'inputs': [final_test_batch],
            'outputs': [test_output_gen_batch], 
            'scores': reward_results.get('scores', []),
            'batch_size': processed_batch_size  # Use actual processed size, not original
        }
        
        # Add all computed metrics to the result
        result.update(all_metrics)
        
        # Add timing and throughput metrics to match KG-R1 evaluation output
        result.update({
            'generation_time/mean': timing_metrics['generation_time'],
            'samples_per_sec/mean': timing_metrics['samples_per_sec'],
            'tokens_per_sec/mean': timing_metrics['tokens_per_sec'],
            'output_tokens_per_sec/mean': timing_metrics['output_tokens_per_sec'],
            'total_tokens/mean': float(timing_metrics['total_tokens']) / max(1, processed_batch_size),
            'input_tokens/mean': float(timing_metrics['input_tokens']) / max(1, processed_batch_size),
            'output_tokens/mean': float(timing_metrics['output_tokens']) / max(1, processed_batch_size),
            'batch_processing_time': timing_metrics['generation_time']
        })
        
        return result
    
    def _compute_final_metrics(self, all_batch_metrics: List[Dict], 
                             all_responses: List[str], all_questions: List[str], 
                             all_ground_truths: List[str]) -> Dict[str, Any]:
        """Compute final Pass@K and other metrics."""
        
        # Collect all results
        all_exact_matches = []
        all_f1_scores = []
        
        for batch_metrics in all_batch_metrics:
            all_exact_matches.extend(batch_metrics.get('exact_matches', []))
            all_f1_scores.extend(batch_metrics.get('f1_scores', []))
        
        # Group results by question for Pass@K computation
        total_questions = sum(batch['batch_size'] for batch in all_batch_metrics)
        
        pass_at_k_results = {}
        f1_results = []
        
        # Compute Pass@K for each question
        for q_idx in range(total_questions):
            start_idx = q_idx * self.n_rollout_eval
            end_idx = start_idx + self.n_rollout_eval
            
            question_exact_matches = all_exact_matches[start_idx:end_idx]
            question_f1_scores = all_f1_scores[start_idx:end_idx]
            
            # Compute Pass@K for this question
            for k in self.k_values:
                if f'pass@{k}' not in pass_at_k_results:
                    pass_at_k_results[f'pass@{k}'] = []
                
                pass_at_k = compute_pass_at_k(question_exact_matches, k)
                pass_at_k_results[f'pass@{k}'].append(pass_at_k)
            
            # Best F1 score for this question
            if question_f1_scores:
                best_f1 = max(question_f1_scores)
                f1_results.append(best_f1)
        
        # Compute final statistics
        final_metrics = {}
        
        # Pass@K metrics
        for k_metric, values in pass_at_k_results.items():
            if values:
                final_metrics[f'exact_match_{k_metric}/mean'] = float(np.mean(values))
                final_metrics[f'exact_match_{k_metric}/std'] = float(np.std(values))
            else:
                final_metrics[f'exact_match_{k_metric}/mean'] = 0.0
                final_metrics[f'exact_match_{k_metric}/std'] = 0.0
        
        # F1 metrics
        if f1_results:
            final_metrics['f1/mean'] = float(np.mean(f1_results))
            final_metrics['f1/std'] = float(np.std(f1_results))
        else:
            final_metrics['f1/mean'] = 0.0
            final_metrics['f1/std'] = 0.0
        
        # Length statistics
        if all_responses:
            response_lengths = [len(r.split()) for r in all_responses]
            final_metrics['response_length/mean'] = float(np.mean(response_lengths))
        else:
            final_metrics['response_length/mean'] = 0.0
        
        # Print results
        print(f"\n{'='*50}")
        print("VANILLA EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Total questions evaluated: {total_questions}")
        print(f"Responses per question: {self.n_rollout_eval}")
        print()
        
        for k in self.k_values:
            pass_k_mean = final_metrics.get(f'exact_match_pass@{k}/mean', 0.0)
            print(f"Pass@{k} (Exact Match): {pass_k_mean:.4f} ({pass_k_mean*100:.1f}%)")
        
        print()
        f1_mean = final_metrics.get('f1/mean', 0.0)
        print(f"F1 Score: {f1_mean:.4f}")
        
        resp_length = final_metrics.get('response_length/mean', 0.0)
        print(f"Avg Response Length: {resp_length:.1f} words")
        
        # Display LLM Judge Statistics
        print()
        print("🤖 LLM JUDGE STATISTICS:")
        print(f"{'='*30}")
        if self.llm_judge_total_calls > 0:
            success_rate = (self.llm_judge_successful_calls / self.llm_judge_total_calls) * 100
            fallback_rate = (self.llm_judge_fallbacks / self.llm_judge_total_calls) * 100
            vector_failure_rate = (self.vector_extraction_failures / self.llm_judge_total_calls) * 100
            
            print(f"Total LLM Judge calls: {self.llm_judge_total_calls}")
            print(f"Successful extractions: {self.llm_judge_successful_calls} ({success_rate:.1f}%)")
            print(f"Vector extraction failures: {self.vector_extraction_failures} ({vector_failure_rate:.1f}%)")
            print(f"Heuristic fallbacks: {self.llm_judge_fallbacks} ({fallback_rate:.1f}%)")
            
            # Quality assessment
            if fallback_rate < 5:
                print("🟢 Excellent: Very low fallback rate")
            elif fallback_rate < 15:
                print("🟡 Good: Moderate fallback rate") 
            else:
                print("🔴 Attention: High fallback rate - consider improving prompt")
        else:
            print("No LLM Judge calls made")
        
        return final_metrics
    
    # ========================================================================================
    # LLM-as-Judge Evaluation Functions
    # ========================================================================================
    
    def parse_ground_truth_entities(self, reward_model_data: dict) -> List[str]:
        """
        Parse ground truth entities from the reward_model data structure.
        
        Args:
            reward_model_data: Dictionary containing ground_truth with target_text array
            
        Returns:
            List of ground truth entity strings
        """
        try:
            ground_truth = reward_model_data.get('ground_truth', {})
            target_texts = ground_truth.get('target_text', [])
            
            # Convert numpy array to list if needed
            if hasattr(target_texts, 'tolist'):
                target_texts = target_texts.tolist()
            
            # Ensure we have a list of strings
            if isinstance(target_texts, str):
                return [target_texts]
            elif isinstance(target_texts, (list, tuple)):
                return [str(text) for text in target_texts]
            else:
                return []
                
        except Exception as e:
            print(f"Warning: Failed to parse ground truth entities: {e}")
            return []
    
    def extract_question_from_prompt(self, prompt_data) -> str:
        """
        Extract clean question from the prompt data structure.
        
        Args:
            prompt_data: Either list of message dictionaries or decoded string from tokenizer
            
        Returns:
            Clean question string
        """
        try:
            if not prompt_data:
                return ""
            
            # Handle string input (from tokenizer.decode)
            if isinstance(prompt_data, str):
                return self._extract_question_from_string(prompt_data)
            
            # Handle list input (from dataset structure)
            elif isinstance(prompt_data, list):
                if len(prompt_data) == 0:
                    return ""
                
                # Get the user message content
                user_message = prompt_data[0].get('content', '')
                return self._extract_question_from_string(user_message)
            
            else:
                print(f"Warning: Unexpected prompt_data type: {type(prompt_data)}")
                return ""
            
        except Exception as e:
            print(f"Warning: Failed to extract question from prompt: {e}")
            return ""
    
    def _extract_question_from_string(self, text: str) -> str:
        """
        Extract question from a text string using pattern matching.
        
        Args:
            text: Input text containing the question
            
        Returns:
            Extracted question string
        """
        if not isinstance(text, str):
            return ""
        
        # Extract question part - look for "Question:" pattern
        if 'Question:' in text:
            question_part = text.split('Question:')[1].strip()
            
            # Remove initial entities hint: "question text?? (Initial entities: ...)"
            if '(Initial entities:' in question_part:
                clean_question = question_part.split('(Initial entities:')[0].strip()
            else:
                # Remove "Reasoning:" and everything after it (for COT prompts)
                if 'Reasoning:' in question_part:
                    clean_question = question_part.split('Reasoning:')[0].strip()
                # Remove "Answers:" and everything after it
                elif 'Answers:' in question_part:
                    clean_question = question_part.split('Answers:')[0].strip()
                # Remove "Answer:" and everything after it  
                elif 'Answer:' in question_part:
                    clean_question = question_part.split('Answer:')[0].strip()
                else:
                    clean_question = question_part.split('\n')[0].strip()
                
            return clean_question
        
        # If no "Question:" pattern, try to extract from common prompt formats
        if 'Answer the given question' in text:
            # Look for question after instruction
            parts = text.split('Answer the given question')[1].strip()
            if parts.startswith('directly and concisely'):
                # Skip instruction part and find actual question
                remaining = parts.split('Question:')
                if len(remaining) > 1:
                    extracted = remaining[1].strip()
                    # Clean up the extracted question
                    if 'Answers:' in extracted:
                        extracted = extracted.split('Answers:')[0].strip()
                    elif 'Answer:' in extracted:
                        extracted = extracted.split('Answer:')[0].strip()
                    return extracted.split('\n')[0].strip()
        
        # Fallback: return empty string to avoid confusion
        return ""
    
    def create_llm_judge_prompt(self, question: str, predicted_answer: str, ground_truth_entities: List[str]) -> str:
        """
        Create simplified evaluation prompt for LLM-as-Judge.
        
        Args:
            question: The original question (not used in new simple format)
            predicted_answer: Model's predicted answer
            ground_truth_entities: List of correct entities
            
        Returns:
            Simplified prompt for LLM judge evaluation focusing on semantic equivalence
        """
        entities_str = ", ".join([f"'{entity}'" for entity in ground_truth_entities])
        
        prompt = f"""For each gold entity in order, does the prediction refer to the same real-world entity with the same level of specificity? Respond only with a binary vector in format [0,1,0,1]. 
1 = same entity with adequate specificity, 0 = different entity or insufficient specificity.

Rules:
- Exact matches or clear equivalents: 1 (e.g., "Apple Inc." = "Apple")
- Too general when specifics required: 0 (e.g., "Islam" when ["Shia Islam", "Sunni Islam"] needed)
- Partial but incomplete: 0 (e.g., "Islam" covers both but misses the distinction)

Example: gold: ['Apple Inc.'] predicted: Apple
binary vector: [1]

Example: gold: ['New York', 'California'] predicted: NYC and LA  
binary vector: [1,0]

Example: gold: ['United States', 'Barack Obama'] predicted: America; Obama
binary vector: [1,1]

Example: gold: ['Shia Islam', 'Sunni Islam'] predicted: Islam
binary vector: [0,0]

Example: gold: ['Buddhism'] predicted: religion
binary vector: [0]

gold: [{entities_str}]
predicted: {predicted_answer}
binary vector:"""
        
        return prompt
    
    def extract_answer_from_response(self, raw_response: str) -> str:
        """
        Extract clean answer from model response, handling JSON array format.
        
        Args:
            raw_response: Raw model response that may include prompt echoing
            
        Returns:
            Clean answer string (semicolon-separated for compatibility)
        """
        import re
        import json
        
        # Remove common prompt artifacts
        clean_response = raw_response.strip()
        
        # First try to extract JSON array format: Answers: [ "answer1", "answer2" ] or standalone [ "answer" ]
        # Pattern 1: Find the LAST occurrence of "Answers: [ ... ]" to avoid mid-reasoning extraction
        answers_matches = list(re.finditer(r'Answers:\s*\[(.*?)\]', clean_response, re.DOTALL | re.IGNORECASE))
        if answers_matches:
            # Use the last match to get final answer (not mid-reasoning)
            answers_match = answers_matches[-1]
        else:
            # Pattern 2: Standalone [ ... ] at the beginning
            answers_match = re.search(r'^\s*\[(.*?)\]', clean_response, re.DOTALL)
        
        if answers_match:
            try:
                # Extract the content inside brackets
                bracket_content = answers_match.group(1).strip()
                
                # Try to parse as JSON array
                if bracket_content:
                    # Ensure proper JSON formatting
                    json_content = '[' + bracket_content + ']'
                    
                    try:
                        answers_list = json.loads(json_content)
                        if isinstance(answers_list, list) and answers_list:
                            # Convert to semicolon-separated format for compatibility
                            return '; '.join(str(ans).strip('"\'') for ans in answers_list if ans)
                    except json.JSONDecodeError:
                        # Fallback: manually parse comma-separated quoted strings
                        items = []
                        for item in bracket_content.split(','):
                            item = item.strip().strip('"\'').strip()
                            if item:
                                items.append(item)
                        if items:
                            return '; '.join(items)
            except Exception:
                pass
        
        # Fallback to legacy format: "Answer:" pattern
        if "Answer:" in clean_response:
            clean_response = clean_response.split("Answer:", 1)[1].strip()
        elif "Answers:" in clean_response:
            clean_response = clean_response.split("Answers:", 1)[1].strip()
        
        # For vanilla responses, stop at the first newline or question pattern to avoid continuation
        lines = clean_response.split('\n')
        first_line = lines[0].strip()
        
        # Remove brackets if present but not in proper JSON format
        first_line = re.sub(r'^\s*\[([^\]]+)\]\s*$', r'\1', first_line)
        
        # If first line looks like a complete answer (not ending with incomplete sentence), use it
        if first_line and not first_line.endswith(('and', 'or', 'but', 'which', 'that', 'the', 'a', 'an')):
            # Check if there are continuation patterns that suggest we should stop
            if len(lines) > 1:
                second_line = lines[1].strip()
                # Stop if next line starts with "Question:" or explanation patterns
                if (second_line.startswith(('Question:', 'Q:', 'Example', 'Note:', 'Therefore', 'This', 'Based on')) or
                    'Question:' in clean_response or len(first_line) > 200):
                    clean_response = first_line
        
        # Remove other common artifacts from the beginning
        for prefix in ["Question:", "Q:", "A:", "Response:"]:
            if clean_response.startswith(prefix):
                clean_response = clean_response[len(prefix):].strip()
        
        # Clean up brackets and quotes that might remain
        clean_response = re.sub(r'^\s*[\[\("\']*([^,]+)[\]\)"\']*.*$', r'\1', clean_response)
        
        # Truncate at reasonable length to avoid verbose explanations
        if len(clean_response) > 300:
            # Try to find a natural stopping point
            sentences = clean_response.split('. ')
            if len(sentences) > 1 and len(sentences[0]) < 200:
                clean_response = sentences[0]
            else:
                clean_response = clean_response[:300].rsplit(' ', 1)[0] + '...'
        
        # Remove extra whitespace
        clean_response = " ".join(clean_response.split())
        
        return clean_response
    
    def evaluate_with_llm_judge(self, question: str, predicted_answer: str, ground_truth_entities: List[str], detailed_log: bool = False) -> List[int]:
        """
        Use LLM judge (configured in KEYS.py) to evaluate if predicted answer covers ground truth entities.
        
        Args:
            question: Original question
            predicted_answer: Model's predicted answer  
            ground_truth_entities: List of correct entities
            
        Returns:
            Binary vector indicating which entities are covered
        """
        import openai
        import re
        import json
        import os
        
        try:
            # Track total calls
            self.llm_judge_total_calls += 1
            
            # Handle empty cases
            if not predicted_answer.strip():
                return [0] * len(ground_truth_entities)
            
            if not ground_truth_entities:
                return []
            
            # Create evaluation prompt
            prompt = self.create_llm_judge_prompt(question, predicted_answer, ground_truth_entities)
            
            if detailed_log:
                print(f"[LLM-JUDGE DETAILED] === LLM Judge Prompt ===")
                print(f"{prompt}")
                print(f"[LLM-JUDGE DETAILED] === End Prompt ===")
            
            # Set up OpenAI API client with key and model from KEYS.py
            try:
                from KEYS import JUDGE_MODELS, DEFAULT_JUDGE_MODEL
                judge_config = JUDGE_MODELS[DEFAULT_JUDGE_MODEL]
                api_key = judge_config["api_key"]
                judge_model = judge_config["model_name"]
            except ImportError:
                # Fallback to environment variable if KEYS.py not available
                api_key = os.getenv("OPENAI_API_KEY", "your-api-key-here")
                judge_model = "gpt-3.5-turbo"
                judge_config = {
                    "api_key": api_key,
                    "model_name": judge_model,
                    "max_tokens": 200,
                    "timeout": 15
                }
            
            client = openai.OpenAI(api_key=api_key)
            
            try:
                # Make API call using configured judge model
                # GPT-5 models use different parameter names
                api_params = {
                    "model": judge_model,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "timeout": judge_config.get("timeout", 15.0)
                }
                
                # Use correct parameter name based on model and judge_config
                if judge_model.startswith("gpt-5"):
                    api_params["max_completion_tokens"] = judge_config.get("max_completion_tokens", 300)
                    # GPT-5 doesn't support custom temperature, uses default
                else:
                    api_params["max_tokens"] = judge_config.get("max_tokens", 200)
                    api_params["temperature"] = judge_config.get("temperature", 0.0)
                
                response = client.chat.completions.create(**api_params)
                
                # Parse binary vector from response
                response_text = response.choices[0].message.content.strip()
                
                if detailed_log:
                    print(f"[LLM-JUDGE DETAILED] === {judge_model} Response ===")
                    print(f"[LLM-JUDGE DETAILED] Question: {question}")
                    print(f"[LLM-JUDGE DETAILED] Raw response: {repr(response_text)}")
                    print(f"[LLM-JUDGE DETAILED] Finish reason: {response.choices[0].finish_reason}")
                    print(f"[LLM-JUDGE DETAILED] === End Response ===")
                
                # Try to extract binary vector [0,1,0,1,0] from response
                print(f"[LLM-JUDGE] Raw response: '{response_text}'")
                
                vector_match = re.search(r'\[([0,1,\s]+)\]', response_text)
                if vector_match:
                    vector_str = vector_match.group(1)
                    print(f"[LLM-JUDGE] Found vector string: '{vector_str}'")
                    
                    try:
                        binary_vector = [int(x.strip()) for x in vector_str.split(',') if x.strip().isdigit()]
                        
                        print(f"[LLM-JUDGE] Parsed binary vector: {binary_vector}")
                        print(f"[LLM-JUDGE] Expected length: {len(ground_truth_entities)}, actual: {len(binary_vector)}")
                        
                        if detailed_log:
                            print(f"[LLM-JUDGE DETAILED] Extracted binary vector: {binary_vector}")
                            print(f"[LLM-JUDGE DETAILED] Expected length: {len(ground_truth_entities)}")
                            print(f"[LLM-JUDGE DETAILED] Actual length: {len(binary_vector)}")
                        
                        # Ensure vector length matches ground truth entities
                        if len(binary_vector) == len(ground_truth_entities):
                            self.llm_judge_successful_calls += 1
                            print(f"[LLM-JUDGE] ✅ Binary vector extraction SUCCESS")
                            
                            if detailed_log:
                                print(f"[LLM-JUDGE DETAILED] Binary vector validation: SUCCESS")
                                for idx, (entity, score) in enumerate(zip(ground_truth_entities, binary_vector)):
                                    print(f"[LLM-JUDGE DETAILED]   Entity {idx}: '{entity}' -> {score}")
                            return binary_vector
                        else:
                            self.vector_extraction_failures += 1
                            print(f"[LLM-JUDGE] ❌ Vector length mismatch. Expected {len(ground_truth_entities)}, got {len(binary_vector)}")
                            
                    except (ValueError, IndexError) as e:
                        self.vector_extraction_failures += 1
                        print(f"[LLM-JUDGE] ❌ Failed to parse binary vector: {e}")
                else:
                    self.vector_extraction_failures += 1
                    print(f"[LLM-JUDGE] ❌ No vector pattern found in response")
                
                # Fallback: if API call succeeded but parsing failed, use heuristic
                self.llm_judge_fallbacks += 1
                print(f"[LLM-JUDGE] 🔄 Fallback: Using heuristic evaluation")
                print(f"[LLM-JUDGE] Question: {question}")
                print(f"[LLM-JUDGE] GPT Response: {response_text}")
                
                binary_vector = []
                for entity in ground_truth_entities:
                    # Simple contains check as fallback
                    entity_lower = entity.lower()
                    predicted_lower = predicted_answer.lower()
                    
                    # Check for entity or key parts of entity in prediction
                    is_present = (
                        entity_lower in predicted_lower or
                        any(word.lower() in predicted_lower for word in entity.split() if len(word) > 2)
                    )
                    binary_vector.append(1 if is_present else 0)
                
                return binary_vector
            except openai.APITimeoutError:
                self.llm_judge_fallbacks += 1
                print(f"[LLM-JUDGE] ⏰ API timeout, using heuristic fallback")
                # Fallback to heuristic
                binary_vector = []
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    predicted_lower = predicted_answer.lower()
                    is_present = entity_lower in predicted_lower
                    binary_vector.append(1 if is_present else 0)
                return binary_vector
                
            except openai.APIError as e:
                self.llm_judge_fallbacks += 1
                print(f"[LLM-JUDGE] 🚫 OpenAI API error with {judge_model}: {e}, using heuristic fallback")
                # Fallback to heuristic
                binary_vector = []
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    predicted_lower = predicted_answer.lower()
                    is_present = entity_lower in predicted_lower
                    binary_vector.append(1 if is_present else 0)
                return binary_vector
                
            except Exception as e:
                self.llm_judge_fallbacks += 1
                print(f"[LLM-JUDGE] ❌ Unexpected error: {e}, using heuristic fallback")
                import traceback
                traceback.print_exc()
                # Fallback: return all zeros
                return [0] * len(ground_truth_entities)
            finally:
                # Properly close the OpenAI client to prevent event loop errors
                client.close()
        
        except Exception as e:
            print(f"[LLM-JUDGE] Critical error in evaluate_with_llm_judge: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback result
            return [0] * len(ground_truth_entities)
    
    def compute_metrics_from_binary_vector(self, binary_vector: List[int], ground_truth_entities: List[str]) -> Dict[str, float]:
        """
        Calculate exact match and F1 score from binary vector.
        
        Args:
            binary_vector: Binary vector indicating entity coverage
            ground_truth_entities: List of ground truth entities
            
        Returns:
            Dictionary with exact_match and f1_score
        """
        if not binary_vector or not ground_truth_entities:
            return {'exact_match': 0.0, 'f1_score': 0.0}
        
        # Hit@1: at least one entity must be covered (changed from strict exact match)
        exact_match = 1.0 if any(binary_vector) else 0.0
        
        # F1 Score calculation
        true_positives = sum(binary_vector)  # Number of correctly identified entities
        total_ground_truth = len(ground_truth_entities)  # Total entities that should be found
        total_predicted = sum(binary_vector)  # Number of entities claimed to be found
        
        if total_ground_truth == 0:
            precision = recall = f1_score = 0.0
        else:
            # For entity-level evaluation:
            # Precision = TP / (TP + FP) = correctly_found / total_claimed
            # Recall = TP / (TP + FN) = correctly_found / total_should_find
            
            precision = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
            recall = true_positives / total_ground_truth if total_ground_truth > 0 else 0.0
            
            if precision + recall > 0:
                f1_score = 2 * (precision * recall) / (precision + recall)
            else:
                f1_score = 0.0
        
        return {
            'exact_match': exact_match,
            'f1_score': f1_score,
            'precision': precision,
            'recall': recall,
            'entities_found': true_positives,
            'entities_total': total_ground_truth
        }
    
    def compute_vanilla_rewards_with_llm_judge(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute vanilla rewards using parallel LLM judge processing.
        
        This function uses ThreadPoolExecutor to process multiple LLM judge calls
        in parallel, reducing evaluation time from 5-10x to near real-time.
        
        Args:
            batch: DataProto containing responses and metadata
            
        Returns:
            Dictionary with scores, exact_match_scores, and f1_scores
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import time
        
        scores = []
        exact_match_scores = []
        f1_scores = []
        detailed_results = []
        
        # Get correct batch size from responses tensor
        if 'responses' in batch.batch:
            responses_shape = batch.batch['responses'].shape
            num_responses = responses_shape[0]
            
            print(f"[LLM-JUDGE DEBUG] Responses tensor shape: {responses_shape}")
            print(f"[LLM-JUDGE DEBUG] Using num_responses: {num_responses}")
        else:
            num_responses = 0
            print(f"[LLM-JUDGE DEBUG] No responses in batch")
        
        # Calculate effective number of samples to evaluate (after dropping)
        original_batch_size = batch.meta_info.get('original_batch_size', num_responses) if hasattr(batch, 'meta_info') and batch.meta_info else num_responses
        dropped_batch_size = batch.meta_info.get('dropped_batch_size', num_responses) if hasattr(batch, 'meta_info') and batch.meta_info else num_responses
        
        # Use the actual batch size after dropping, not the original
        effective_samples = min(dropped_batch_size, num_responses)  # Use dropped batch size
        if self.eval_samples > 0:
            # Respect eval_samples limit - don't evaluate beyond the requested count
            # Note: eval_samples applies before dropping, so use original_batch_size for this check
            original_eval_limit = min(original_batch_size, self.eval_samples)
            # But then apply dropping to get the final effective samples
            effective_samples = min(dropped_batch_size, original_eval_limit)
            
        print(f"[LLM-JUDGE] Batch has {num_responses} total responses ({original_batch_size} original -> {dropped_batch_size} after dropping)")
        print(f"[LLM-JUDGE] Will evaluate {effective_samples} samples, skipping {num_responses - effective_samples} unused samples")
            
        print(f"[LLM-JUDGE] Evaluating {effective_samples} responses with parallel processing...")
        
        # Prepare evaluation tasks for parallel processing
        evaluation_tasks = []
        start_time = time.time()
        
        if effective_samples < num_responses:
            print(f"[LLM-JUDGE] Limiting batch evaluation to first {effective_samples} samples (eval_samples={self.eval_samples})")
        
        for i in range(effective_samples):
            try:
                # Extract response - clean it properly
                if 'responses' in batch.batch:
                    response_ids = batch.batch['responses'][i]
                    raw_predicted_answer = self.tokenizer.decode(response_ids, skip_special_tokens=True).strip()
                    predicted_answer = self.extract_answer_from_response(raw_predicted_answer)
                    
                    # Log Qwen model responses for all samples
                    print(f"\n[QWEN-RESPONSE] === Sample {i+1} ===")
                    print(f"[QWEN-RESPONSE] Raw model output: {repr(raw_predicted_answer)}")
                    print(f"[QWEN-RESPONSE] Extracted answer: {repr(predicted_answer)}")
                    
                    # Detailed logging for first 2 samples in each batch
                    if i < 2:
                        print(f"\n[LLM-JUDGE DETAILED] === Sample {i} Analysis ===")
                        # Note: question will be extracted and shown below after ground truth extraction
                        print(f"[LLM-JUDGE DETAILED] Raw response: {repr(raw_predicted_answer[:200])}{'...' if len(raw_predicted_answer) > 200 else ''}")
                        print(f"[LLM-JUDGE DETAILED] Extracted answer: {repr(predicted_answer)}")
                else:
                    predicted_answer = ""
                    print(f"\n[QWEN-RESPONSE] === Sample {i+1} ===")
                    print(f"[QWEN-RESPONSE] Raw model output: (empty - no responses in batch)")
                    print(f"[QWEN-RESPONSE] Extracted answer: (empty)")
                
                # Extract ground truth and question from batch's original test data
                question = "Unknown question"
                ground_truth_entities = []
                
                # Get ground truth from batch's original_test_data (accounting for rollout expansion)
                original_sample_idx = i // self.n_rollout_eval
                
                try:
                    # First try to get from batch's original_test_data which has correct ordering
                    if (hasattr(batch, 'meta_info') and batch.meta_info and 
                        'original_test_data' in batch.meta_info):
                        
                        original_test_data = batch.meta_info['original_test_data']
                        
                        # Debug logging for first few samples
                        if i < 3:
                            print(f"[QWEN-RESPONSE DEBUG] Sample {i+1}: Using batch original_test_data")
                            print(f"[QWEN-RESPONSE DEBUG] Original sample idx: {original_sample_idx}")
                            print(f"[QWEN-RESPONSE DEBUG] original_test_data type: {type(original_test_data)}")
                            
                            if isinstance(original_test_data, dict):
                                print(f"[QWEN-RESPONSE DEBUG] original_test_data keys: {list(original_test_data.keys())}")
                                # Check if it's a dataset-like dict with lists/tensors
                                for key in ['prompt', 'prompts', 'question', 'questions', 'input', 'inputs', 'ground_truth']:
                                    if key in original_test_data:
                                        print(f"[QWEN-RESPONSE DEBUG] Found {key} with type: {type(original_test_data[key])}")
                                        if hasattr(original_test_data[key], '__len__'):
                                            print(f"[QWEN-RESPONSE DEBUG] {key} length: {len(original_test_data[key])}")
                            
                            if hasattr(original_test_data, 'batch'):
                                print(f"[QWEN-RESPONSE DEBUG] original_test_data.batch type: {type(original_test_data.batch)}")
                                if hasattr(original_test_data.batch, 'keys'):
                                    print(f"[QWEN-RESPONSE DEBUG] original_test_data.batch keys: {list(original_test_data.batch.keys())}")
                                    # Check for any field containing prompts or questions
                                    for key in ['prompt', 'prompts', 'question', 'questions', 'input', 'inputs', 'ground_truth']:
                                        if key in original_test_data.batch:
                                            print(f"[QWEN-RESPONSE DEBUG] Found {key} in batch with shape/type: {type(original_test_data.batch[key])}")
                                else:
                                    print(f"[QWEN-RESPONSE DEBUG] original_test_data.batch has no keys() method")
                            else:
                                print(f"[QWEN-RESPONSE DEBUG] original_test_data has no 'batch' attribute")
                        
                        # Extract ground truth from the batch's test data
                        # First check if original_test_data is a simple dict with lists
                        if isinstance(original_test_data, dict) and not hasattr(original_test_data, 'batch'):
                            if i < 3:
                                print(f"[QWEN-RESPONSE DEBUG] Attempting dict-based extraction")
                            
                            # Try to extract from dict format - check reward_model field
                            if 'reward_model' in original_test_data:
                                rm_data = original_test_data['reward_model']
                                if hasattr(rm_data, '__getitem__') and original_sample_idx < len(rm_data):
                                    reward_model_sample = rm_data[original_sample_idx]
                                    if isinstance(reward_model_sample, dict) and 'ground_truth' in reward_model_sample:
                                        ground_truth_raw = reward_model_sample['ground_truth']
                                        ground_truth_entities = self.parse_ground_truth_entities({'ground_truth': ground_truth_raw})
                                        if i < 3:
                                            print(f"[QWEN-RESPONSE DEBUG] Extracted GT from reward_model: {ground_truth_entities}")
                            
                            # Also check direct ground_truth field (fallback)
                            elif 'ground_truth' in original_test_data:
                                gt_data = original_test_data['ground_truth']
                                if hasattr(gt_data, '__getitem__') and original_sample_idx < len(gt_data):
                                    ground_truth_raw = gt_data[original_sample_idx]
                                    ground_truth_entities = self.parse_ground_truth_entities({'ground_truth': ground_truth_raw})
                                    if i < 3:
                                        print(f"[QWEN-RESPONSE DEBUG] Extracted GT from dict: {ground_truth_entities}")
                            
                            # Try to extract question from extra_info field (most likely location)
                            if 'extra_info' in original_test_data:
                                ei_data = original_test_data['extra_info']
                                if hasattr(ei_data, '__getitem__') and original_sample_idx < len(ei_data):
                                    extra_info_sample = ei_data[original_sample_idx]
                                    if isinstance(extra_info_sample, dict):
                                        # Check for various question fields
                                        for q_field in ['question', 'prompt', 'input', 'text']:
                                            if q_field in extra_info_sample:
                                                question = extra_info_sample[q_field]
                                                if i < 3:
                                                    print(f"[QWEN-RESPONSE DEBUG] Extracted question from extra_info.{q_field}: {question[:100]}...")
                                                break
                            
                            # Fallback: decode the input_ids to get the prompt
                            if question == "Unknown question" and 'input_ids' in original_test_data:
                                input_ids_data = original_test_data['input_ids']
                                if hasattr(input_ids_data, '__getitem__') and original_sample_idx < len(input_ids_data):
                                    input_ids = input_ids_data[original_sample_idx]
                                    try:
                                        # Decode the input to get the actual prompt
                                        decoded_prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                                        # Extract question from the decoded prompt
                                        question = self.extract_question_from_prompt(decoded_prompt)
                                        if i < 3:
                                            print(f"[QWEN-RESPONSE DEBUG] Extracted question from decoded input_ids: {question[:100]}...")
                                    except Exception as e:
                                        if i < 3:
                                            print(f"[QWEN-RESPONSE DEBUG] Failed to decode input_ids: {e}")
                            
                            # Last resort: use data_source as fallback identifier  
                            if question == "Unknown question" and 'data_source' in original_test_data:
                                ds_data = original_test_data['data_source']
                                if hasattr(ds_data, '__getitem__') and original_sample_idx < len(ds_data):
                                    data_source_sample = ds_data[original_sample_idx]
                                    if isinstance(data_source_sample, str):
                                        question = f"[Dataset: {data_source_sample}]"
                                        if i < 3:
                                            print(f"[QWEN-RESPONSE DEBUG] Using data_source as fallback: {question}")
                        
                        elif hasattr(original_test_data, 'batch'):
                            # Check if we can access the data at this index
                            batch_size = None
                            if hasattr(original_test_data.batch, '__len__'):
                                batch_size = len(original_test_data.batch)
                            elif isinstance(original_test_data.batch, dict):
                                # Get batch size from first tensor in the dict
                                for key, value in original_test_data.batch.items():
                                    if hasattr(value, '__len__'):
                                        batch_size = len(value)
                                        break
                            
                            if batch_size and original_sample_idx < batch_size:
                                # Get the specific sample's data from original test batch
                                sample_data = {}
                                for key in original_test_data.batch:
                                    if hasattr(original_test_data.batch[key], '__getitem__'):
                                        try:
                                            sample_data[key] = original_test_data.batch[key][original_sample_idx]
                                        except (IndexError, KeyError) as e:
                                            if i < 3:
                                                print(f"[QWEN-RESPONSE DEBUG] Could not access {key}[{original_sample_idx}]: {e}")
                                
                                # Extract ground truth
                                if 'ground_truth' in sample_data:
                                    ground_truth_raw = sample_data['ground_truth']
                                    ground_truth_entities = self.parse_ground_truth_entities({'ground_truth': ground_truth_raw})
                                
                                # Extract question
                                if 'question' in sample_data:
                                    question = sample_data['question']
                                elif 'prompt' in sample_data:
                                    question = self.extract_question_from_prompt(sample_data['prompt'])
                        
                    # Fallback to cached dataset if batch data not available
                    elif (self._ground_truth_cache and 
                          original_sample_idx < len(self._ground_truth_cache)):
                        
                        cached_data = self._ground_truth_cache[original_sample_idx]
                        ground_truth_raw = cached_data['ground_truth']
                        ground_truth_entities = self.parse_ground_truth_entities({'ground_truth': ground_truth_raw})
                        
                        # Extract question
                        if cached_data.get('question'):
                            question = cached_data['question']
                        elif cached_data.get('prompt'):
                            question = self.extract_question_from_prompt(cached_data['prompt'])
                    
                    # Add question to Qwen response logging for context
                    print(f"[QWEN-RESPONSE] Question asked: {question}")
                    print(f"[QWEN-RESPONSE] Ground truth: {ground_truth_entities}")
                    
                    # Detailed logging for first 2 samples
                    if i < 2:
                        print(f"[LLM-JUDGE DETAILED] Original sample idx: {original_sample_idx}")
                        print(f"[LLM-JUDGE DETAILED] Question: {question}")
                        print(f"[LLM-JUDGE DETAILED] Ground truth entities: {ground_truth_entities}")
                        print(f"[LLM-JUDGE DETAILED] Ground truth count: {len(ground_truth_entities)}")
                    
                    # Debug on first sample
                    elif i == 0:
                        print(f"[LLM-JUDGE DEBUG] Original sample idx: {original_sample_idx}")
                        print(f"[LLM-JUDGE DEBUG] Found ground truth entities: {len(ground_truth_entities)}")
                        print(f"[LLM-JUDGE DEBUG] Question: {question[:100]}...")
                    
                    if not ground_truth_entities:
                        if i < 5:  # Only show first 5 warnings to avoid spam
                            print(f"[LLM-JUDGE] Warning: Could not extract ground truth for sample {i} (original: {original_sample_idx})")
                        # Add placeholder task for failed samples
                        evaluation_tasks.append({
                            'index': i,
                            'question': question,
                            'predicted_answer': predicted_answer,
                            'ground_truth_entities': [],
                            'original_sample_idx': original_sample_idx,
                            'failed': True
                        })
                        continue
                        
                except Exception as e:
                    print(f"[LLM-JUDGE] Warning: Error extracting data for sample {i}: {e}")
                    evaluation_tasks.append({
                        'index': i,
                        'question': 'Unknown',
                        'predicted_answer': predicted_answer if 'predicted_answer' in locals() else '',
                        'ground_truth_entities': [],
                        'original_sample_idx': i // self.n_rollout_eval,
                        'failed': True
                    })
                    continue
                
                # Add task for parallel processing
                evaluation_tasks.append({
                    'index': i,
                    'question': question,
                    'predicted_answer': predicted_answer,
                    'ground_truth_entities': ground_truth_entities,
                    'original_sample_idx': original_sample_idx,
                    'failed': False
                })
                    
            except Exception as e:
                print(f"[LLM-JUDGE] Error preparing sample {i}: {e}")
                evaluation_tasks.append({
                    'index': i,
                    'question': 'Error',
                    'predicted_answer': '',
                    'ground_truth_entities': [],
                    'original_sample_idx': i // self.n_rollout_eval,
                    'failed': True
                })
        
        # Parallel processing with ThreadPoolExecutor
        def evaluate_single_task(task):
            """Evaluate a single task with LLM judge."""
            if task['failed']:
                return {
                    'index': task['index'],
                    'binary_vector': [],
                    'metrics': {
                        'exact_match': 0.0,
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'entities_found': 0,
                        'entities_total': 0
                    },
                    'question': task['question'],
                    'predicted_answer': task['predicted_answer'],
                    'ground_truth_entities': task['ground_truth_entities'],
                    'original_sample_idx': task['original_sample_idx'],
                    'failed': True
                }
            
            try:
                # Call LLM judge
                binary_vector = self.evaluate_with_llm_judge(
                    task['question'], 
                    task['predicted_answer'], 
                    task['ground_truth_entities'], 
                    detailed_log=(task['index'] < 2)
                )
                
                # Compute metrics
                metrics = self.compute_metrics_from_binary_vector(binary_vector, task['ground_truth_entities'])
                
                return {
                    'index': task['index'],
                    'binary_vector': binary_vector,
                    'metrics': metrics,
                    'question': task['question'],
                    'predicted_answer': task['predicted_answer'],
                    'ground_truth_entities': task['ground_truth_entities'],
                    'original_sample_idx': task['original_sample_idx'],
                    'failed': False
                }
                
            except Exception as e:
                print(f"[LLM-JUDGE] Error in parallel task {task['index']}: {e}")
                return {
                    'index': task['index'],
                    'binary_vector': [],
                    'metrics': {
                        'exact_match': 0.0,
                        'f1_score': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'entities_found': 0,
                        'entities_total': 0
                    },
                    'question': task['question'],
                    'predicted_answer': task['predicted_answer'],
                    'ground_truth_entities': task['ground_truth_entities'],
                    'original_sample_idx': task['original_sample_idx'],
                    'failed': True
                }
        
        # Execute parallel evaluation
        max_workers = min(32, len(evaluation_tasks))  # Increased parallelism for faster evaluation
        print(f"[LLM-JUDGE] Starting parallel evaluation with {max_workers} workers...")
        
        results = [None] * len(evaluation_tasks)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(evaluate_single_task, task): task for task in evaluation_tasks}
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_task):
                result = future.result()
                results[result['index']] = result
                completed_count += 1
                
                # Progress update every 10 completions
                if completed_count % 10 == 0 or completed_count <= 5:
                    print(f"[LLM-JUDGE] Completed {completed_count}/{len(evaluation_tasks)} evaluations")
        
        # Process results and build final arrays
        for result in results:
            if result is None:
                # Should not happen, but handle gracefully
                scores.append(0.0)
                exact_match_scores.append(0.0)
                f1_scores.append(0.0)
                continue
            
            # Store metrics
            metrics = result['metrics']
            exact_match_scores.append(metrics['exact_match'])
            f1_scores.append(metrics['f1_score'])
            scores.append(metrics['f1_score'])  # Use F1 as overall score
            
            # Store detailed results
            detailed_results.append({
                'question': result['question'],
                'predicted_answer': result['predicted_answer'],
                'ground_truth_entities': result['ground_truth_entities'],
                'binary_vector': result['binary_vector'],
                'metrics': metrics,
                'original_sample_idx': result['original_sample_idx'],
                'response_idx_within_sample': result['index'] % self.n_rollout_eval
            })
            
            # Detailed logging for first 2 samples
            if result['index'] < 2 and not result['failed']:
                print(f"[LLM-JUDGE DETAILED] === Reward Calculation Sample {result['index']} ===")
                print(f"[LLM-JUDGE DETAILED] Binary vector: {result['binary_vector']}")
                print(f"[LLM-JUDGE DETAILED] Exact Match: {metrics['exact_match']}")
                print(f"[LLM-JUDGE DETAILED] F1 Score: {metrics['f1_score']}")
                print(f"[LLM-JUDGE DETAILED] Precision: {metrics['precision']}")
                print(f"[LLM-JUDGE DETAILED] Recall: {metrics['recall']}")
                print(f"[LLM-JUDGE DETAILED] Entities found: {metrics['entities_found']}/{metrics['entities_total']}")
                print(f"[LLM-JUDGE DETAILED] === End Sample {result['index']} Analysis ===\n")
            
            # Log first few for debugging
            if result['index'] < 5:
                print(f"[LLM-JUDGE] Sample {result['index']+1} (orig_idx={result['original_sample_idx']}):")
                print(f"  Question: {result['question'][:100]}...")
                print(f"  Ground Truth: {result['ground_truth_entities']}")
                print(f"  Predicted: {result['predicted_answer'][:100]}...")
                print(f"  Binary Vector: {result['binary_vector']}")
                print(f"  Exact Match: {metrics['exact_match']:.3f}, F1: {metrics['f1_score']:.3f}")
        
        # Performance summary
        elapsed_time = time.time() - start_time
        print(f"[LLM-JUDGE] Parallel evaluation completed in {elapsed_time:.1f}s ({len(evaluation_tasks)/elapsed_time:.1f} samples/sec)")
        
        print(f"[LLM-JUDGE] Completed evaluation:")
        print(f"  Average Exact Match: {np.mean(exact_match_scores):.3f}")
        print(f"  Average F1 Score: {np.mean(f1_scores):.3f}")
        print(f"  Average Overall Score: {np.mean(scores):.3f}")
        
        return {
            'scores': scores,
            'exact_match_scores': exact_match_scores, 
            'f1_scores': f1_scores,
            'detailed_results': detailed_results
        }
    
    def _compute_batch_passatk_metrics(self, batch: List[DataProto], reward_results: Dict) -> Dict[str, Any]:
        """
        Compute pass@k metrics from reward results.
        Enhanced version matching ray_evaluator_kg functionality.
        """
        # Get the single DataProto from the batch
        if len(batch) == 1:
            combined_batch = batch[0]
        else:
            # Combine batch into single DataProto for metric computation (fallback)
            combined_batch = batch[0]
            for data_proto in batch[1:]:
                combined_batch = combined_batch.union(data_proto)
        
        # Extract reward components
        scores = reward_results.get('scores', [])
        exact_match_scores = reward_results.get('exact_match_scores', scores)  # Fallback to scores
        f1_scores = reward_results.get('f1_scores', scores)  # Fallback to scores
        
        all_metrics = {}
        
        # Compute Pass@K metrics for each k value and each metric type
        metric_types = {
            'exact_match': exact_match_scores,
            'f1': f1_scores,
            'precision': exact_match_scores,  # For vanilla, precision = exact_match
            'recall': f1_scores,  # For vanilla, recall ≈ f1
            'retrieval_quality': f1_scores  # For vanilla, retrieval_quality = f1 (no KG retrieval)
        }
        
        for metric_name, metric_scores in metric_types.items():
            if not metric_scores:
                continue
                
            # Overall metric (not Pass@K)
            all_metrics[f'{metric_name}/mean'] = float(np.mean(metric_scores))
            all_metrics[f'{metric_name}/std'] = float(np.std(metric_scores))
            
            # Group scores by original prompt (n_rollout_eval scores per prompt)
            grouped_scores = []
            for i in range(0, len(metric_scores), self.n_rollout_eval):
                group = metric_scores[i:i+self.n_rollout_eval]
                grouped_scores.append(group)
            
            # Compute Pass@K for each k value
            for k in self.k_values:
                passatk_values = []
                for group in grouped_scores:
                    # Pass@K: true if any of the top-k responses meets the criterion
                    if len(group) >= k:
                        top_k_scores = sorted(group, reverse=True)[:k]
                        # For exact match: any score > 0.5
                        # For F1: any score > 0.3 (lower threshold)
                        threshold = 0.5 if metric_name == 'exact_match' else 0.3
                        pass_k = any(score > threshold for score in top_k_scores)
                    else:
                        # Not enough responses for this k
                        pass_k = any(score > (0.5 if metric_name == 'exact_match' else 0.3) for score in group)
                    
                    passatk_values.append(float(pass_k))
                
                if passatk_values:
                    all_metrics[f'{metric_name}_pass@{k}/mean'] = float(np.mean(passatk_values))
                    all_metrics[f'{metric_name}_pass@{k}/std'] = float(np.std(passatk_values))
        
        # Compute length metrics
        length_metrics = self._compute_length_metrics(combined_batch)
        all_metrics.update(length_metrics)
        
        # Store length metrics for FLOP calculations
        self.last_length_metrics = length_metrics
        
        # Add FLOP metrics
        flop_metrics = self._compute_flops_metrics(combined_batch)
        all_metrics.update(flop_metrics)
        
        return all_metrics
    
    def _compute_length_metrics(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute length metrics in tokens for responses.
        
        For vanilla mode:
        1. total_length: prompt + response (everything)
        2. response_length: response only (without prompt)  
        3. generation_length: same as response_length (no KG info to separate)
        
        Args:
            batch: DataProto with tokenized sequences
            
        Returns:
            Dict with length metrics (mean only)
        """
        if not hasattr(batch, 'batch') or batch.batch is None:
            return {}
        
        batch_data = batch.batch
        length_metrics = {}
        
        # Get tokenizer padding token with robust fallback logic
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)
        
        if pad_token_id is None:
            # Llama2/Llama3 and GPT models don't have dedicated pad tokens
            # They use eos_token_id for padding during training/inference
            pad_token_id = getattr(self.tokenizer, 'eos_token_id', None)
            
            if pad_token_id is None:
                # Final fallback: use common default
                pad_token_id = 0
                
        # Optional: Log model-specific padding detection (uncomment for debugging)
        # model_name = getattr(self.tokenizer, 'name_or_path', 'unknown')
        # if 'llama' in model_name.lower():
        #     print(f"[LENGTH] Detected Llama model: {model_name}, using pad_token_id={pad_token_id} (eos_token)")
        # elif 'qwen' in model_name.lower():
        #     print(f"[LENGTH] Detected Qwen model: {model_name}, using pad_token_id={pad_token_id}")
        # else:
        #     print(f"[LENGTH] Model: {model_name}, using pad_token_id={pad_token_id}")
        
        # Helper function to count actual tokens (handles attention masks too)
        def count_actual_tokens(sequences, use_attention_mask=False):
            """Count actual non-padding tokens in sequences."""
            lengths = []
            
            # Try using attention_mask first (more reliable for Llama models)
            if use_attention_mask and 'attention_mask' in batch_data:
                attention_mask = batch_data['attention_mask']
                if hasattr(attention_mask, 'shape') and len(attention_mask) == len(sequences):
                    for mask in attention_mask:
                        if hasattr(mask, 'sum'):
                            length = mask.sum().item()
                        else:
                            length = sum(mask)
                        lengths.append(length)
                    return lengths
            
            # Fallback to padding token counting
            for seq in sequences:
                if hasattr(seq, 'ne'):  # torch tensor
                    non_pad_length = (seq != pad_token_id).sum().item()
                else:
                    non_pad_length = len([t for t in seq if t != pad_token_id])
                lengths.append(non_pad_length)
            
            return lengths

        # 1. Total length: everything (input_ids)
        if 'input_ids' in batch_data:
            input_ids = batch_data['input_ids']
            if hasattr(input_ids, 'shape'):
                total_lengths = count_actual_tokens(input_ids, use_attention_mask=True)
                
                if total_lengths:
                    length_metrics['total_length/mean'] = float(np.mean(total_lengths))
        
        # 2. Response length: responses only (if available)
        if 'responses' in batch_data:
            responses = batch_data['responses']  
            if hasattr(responses, 'shape'):
                # For responses, don't use attention_mask (it's for input_ids)
                response_lengths = count_actual_tokens(responses, use_attention_mask=False)
                
                if response_lengths:
                    length_metrics['response_length/mean'] = float(np.mean(response_lengths))
                    # For vanilla mode, generation_length = response_length (no KG info)
                    length_metrics['generation_length/mean'] = float(np.mean(response_lengths))
        
        return length_metrics
    
    def _compute_flops_metrics(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute FLOPs (Floating Point Operations) used during vanilla generation.
        
        Vanilla Generation Pattern (pure autoregressive):
        - Single forward pass: prompt → generate response tokens
        
        FLOP calculation:
        - 2 * params * context_length * tokens_generated
        - Context = prompt_length, generation = response_length
        
        Args:
            batch: DataProto with tokenized sequences
            
        Returns:
            Dict with FLOP metrics
        """
        if not hasattr(batch, 'batch') or batch.batch is None:
            return {}
        
        flops_metrics = {}
        batch_data = batch.batch
        
        # Get model configuration parameters
        model_config = self._get_model_config()
        if not model_config:
            return flops_metrics
        
        # Extract model parameters
        hidden_size = model_config.get('hidden_size', 0)
        num_layers = model_config.get('num_hidden_layers', 0) or model_config.get('num_layers', 0)
        vocab_size = model_config.get('vocab_size', 0)
        
        if not all([hidden_size, num_layers, vocab_size]):
            return flops_metrics
        
        # Estimate model parameters (simplified calculation)
        # Transformer parameters ≈ 12 * num_layers * hidden_size^2 + vocab_size * hidden_size
        transformer_params = 12 * num_layers * (hidden_size ** 2)  # Main transformer blocks
        embedding_params = vocab_size * hidden_size  # Embedding layer
        total_params = transformer_params + embedding_params
        
        flops_metrics['model_parameters/total'] = float(total_params)
        flops_metrics['model_parameters/billions'] = float(total_params / 1e9)
        
        # Calculate FLOPs if length metrics are available
        if hasattr(self, 'last_length_metrics') and self.last_length_metrics:
            generation_length = self.last_length_metrics.get('generation_length/mean', 0)
            total_length = self.last_length_metrics.get('total_length/mean', 0)
            
            if generation_length > 0 and total_length > 0:
                # Estimate context length (prompt) = total - generation
                context_length = max(total_length - generation_length, generation_length)
                
                # Vanilla generation FLOPs: 2 * params * context * gen_tokens
                generation_flops = 2.0 * total_params * context_length * generation_length
                
                flops_metrics['generation_flops/mean'] = float(generation_flops)
                flops_metrics['generation_gflops/mean'] = float(generation_flops / 1e9)
                
                if generation_length > 0:
                    flops_metrics['flops_per_gen_token/mean'] = float(generation_flops / generation_length)
                    flops_metrics['gflops_per_gen_token/mean'] = float(generation_flops / generation_length / 1e9)
                
                # Total FLOPs (same as generation for vanilla mode)
                flops_metrics['total_flops/mean'] = float(generation_flops)
                flops_metrics['total_gflops/mean'] = float(generation_flops / 1e9)
        
        return flops_metrics
    
    def _get_model_config(self) -> Dict[str, Any]:
        """
        Retrieve model configuration for FLOP calculation.
        
        Returns:
            Dict with model config parameters or empty dict if unavailable
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
            elif 'Llama-2-7b' in model_path or 'Llama2-7B' in model_path:
                # Llama2-7B configuration
                return {
                    'hidden_size': 4096,
                    'num_hidden_layers': 32,
                    'num_attention_heads': 32,
                    'vocab_size': 32000,
                    'model_type': 'llama'
                }
            else:
                # Generic fallback - try to detect model size from path
                if '3B' in model_path or '3b' in model_path:
                    return {
                        'hidden_size': 2048,
                        'num_hidden_layers': 32,
                        'vocab_size': 50000,
                        'model_type': 'generic_3b'
                    }
                elif '7B' in model_path or '7b' in model_path:
                    return {
                        'hidden_size': 4096,
                        'num_hidden_layers': 32,
                        'vocab_size': 50000,
                        'model_type': 'generic_7b'
                    }
                else:
                    return {}
        
        except Exception as e:
            print(f"[DEBUG] Error retrieving model config: {e}")
            return {}
    
    def _compute_final_metrics_efficient(self, all_batch_metrics: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate metrics from all batches to match KG evaluator output format.
        """
        # Collect all metrics (not just passatk_metrics anymore)
        all_metrics = {}
        
        for batch_metrics in all_batch_metrics:
            # Handle both old format (passatk_metrics) and new format (direct metrics)
            if 'passatk_metrics' in batch_metrics:
                # Old format - extract from passatk_metrics
                metrics = batch_metrics['passatk_metrics']
            else:
                # New format - metrics are direct in batch_metrics
                metrics = batch_metrics
            
            for key, value in metrics.items():
                if key not in all_metrics:
                    all_metrics[key] = []
                all_metrics[key].append(value)
        
        # Compute final aggregated metrics
        final_metrics = {}
        for key, values in all_metrics.items():
            if values:
                # Skip non-numeric keys (inputs, outputs, scores, etc.)
                if key in ['inputs', 'outputs', 'scores', 'batch_size']:
                    continue
                
                # Only process numeric metrics
                try:
                    numeric_values = []
                    for val in values:
                        if isinstance(val, (int, float, np.number)):
                            numeric_values.append(float(val))
                        elif hasattr(val, 'item'):  # numpy scalar
                            numeric_values.append(float(val.item()))
                    
                    if numeric_values:
                        if 'mean' in key:
                            # Average of means across batches
                            final_metrics[key] = float(np.mean(numeric_values))
                        elif 'std' in key:
                            # Average of stds across batches (approximation)
                            final_metrics[key] = float(np.mean(numeric_values))
                        else:
                            final_metrics[key] = float(np.mean(numeric_values))
                    else:
                        final_metrics[key] = 0.0
                        
                except (TypeError, ValueError) as e:
                    print(f"[DEBUG] Skipping non-numeric metric {key}: {e}")
                    continue
            else:
                final_metrics[key] = 0.0
        
        # Print comprehensive results matching KG evaluator format
        print(f"\n{'='*60}")
        print(f"FINAL VANILLA EVALUATION RESULTS")
        print(f"{'='*60}")
        
        # Print pass@k results for all metrics
        metric_names = ['exact_match', 'f1', 'precision', 'recall', 'retrieval_quality']
        for metric_name in metric_names:
            print(f"\n{metric_name.upper()} METRICS:")
            
            # Overall metric
            mean_key = f"{metric_name}/mean"
            std_key = f"{metric_name}/std"
            if mean_key in final_metrics:
                mean_val = final_metrics[mean_key]
                std_val = final_metrics.get(std_key, 0.0)
                print(f"  Overall {metric_name}: {mean_val:.4f} ± {std_val:.4f}")
            
            # Pass@K metrics
            for k in self.k_values:
                pass_mean_key = f"{metric_name}_pass@{k}/mean"
                pass_std_key = f"{metric_name}_pass@{k}/std"
                if pass_mean_key in final_metrics:
                    pass_mean = final_metrics[pass_mean_key]
                    pass_std = final_metrics.get(pass_std_key, 0.0)
                    print(f"  Pass@{k} ({metric_name}): {pass_mean:.4f} ± {pass_std:.4f} ({pass_mean*100:.1f}%)")
        
        # Print length metrics
        length_metrics = ['generation_length', 'response_length', 'total_length']
        print(f"\nLENGTH METRICS:")
        for length_metric in length_metrics:
            mean_key = f"{length_metric}/mean"
            if mean_key in final_metrics:
                print(f"  {length_metric.replace('_', ' ').title()}: {final_metrics[mean_key]:.1f} tokens")
        
        # Print FLOP metrics
        flop_metrics = ['generation_gflops', 'gflops_per_gen_token', 'model_parameters/billions']
        print(f"\nCOMPUTATIONAL METRICS:")
        for flop_metric in flop_metrics:
            mean_key = f"{flop_metric}/mean" if '/mean' not in flop_metric else flop_metric
            if mean_key in final_metrics:
                if 'gflops' in flop_metric:
                    print(f"  {flop_metric.replace('_', ' ').replace('/', ' ').title()}: {final_metrics[mean_key]:.2f}")
                elif 'billions' in flop_metric:
                    print(f"  Model Parameters: {final_metrics[mean_key]:.3f}B")
        
        # Print timing and throughput metrics to match KG-R1 evaluation
        timing_metrics = ['generation_time', 'samples_per_sec', 'tokens_per_sec', 'output_tokens_per_sec', 'batch_processing_time']
        print(f"\nTIMING AND THROUGHPUT METRICS:")
        for timing_metric in timing_metrics:
            mean_key = f"{timing_metric}/mean"
            if mean_key in final_metrics:
                if 'time' in timing_metric:
                    print(f"  {timing_metric.replace('_', ' ').title()}: {final_metrics[mean_key]:.2f}s")
                elif 'per_sec' in timing_metric:
                    if 'samples' in timing_metric:
                        print(f"  {timing_metric.replace('_', ' ').title()}: {final_metrics[mean_key]:.1f}")
                    else:
                        print(f"  {timing_metric.replace('_', ' ').title()}: {final_metrics[mean_key]:.0f}")
        
        # Print token count metrics  
        token_metrics = ['total_tokens', 'input_tokens', 'output_tokens']
        print(f"\nTOKEN COUNT METRICS:")
        for token_metric in token_metrics:
            mean_key = f"{token_metric}/mean"
            if mean_key in final_metrics:
                print(f"  {token_metric.replace('_', ' ').title()} (avg per sample): {final_metrics[mean_key]:.1f}")
        
        print(f"\n{'='*60}")
        
        return final_metrics


def create_vanilla_evaluator_from_config(config) -> RayVanillaEvaluator:
    """Create a vanilla evaluator from configuration."""
    from verl.trainer.fsdp_trainer import make_fsdp_tokenizer, make_llama_2_processor
    from verl.trainer.ppo.ray_trainer_kg import make_rollout_ref_workers, make_actor_workers
    from verl.single_controller.ray.resource_pool import ColoResourcePool
    
    # Create tokenizer and processor
    tokenizer = make_fsdp_tokenizer(config=config.actor_rollout_ref.model)
    processor = make_llama_2_processor(config=config.actor_rollout_ref.model) 
    
    # Create resource pool
    resource_pool = ColoResourcePool(world_size=config.trainer.n_gpus_per_node)
    
    # Create worker mapping
    role_worker_mapping = {}
    
    # Create rollout workers
    rollout_worker_group = make_rollout_ref_workers(
        config=config,
        role_worker_mapping=role_worker_mapping,
        resource_pool=resource_pool
    )
    
    return RayVanillaEvaluator(
        config=config,
        tokenizer=tokenizer, 
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool,
        n_rollout_eval=getattr(config, 'n_rollout_eval', 4),
        k_values=getattr(config, 'k_values', [1, 2, 3, 4])
    )