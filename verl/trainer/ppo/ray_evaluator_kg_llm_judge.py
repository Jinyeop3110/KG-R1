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
KG Evaluator with LLM Judge
Combines KG generation framework (ray_evaluator_kg.py) with LLM judge evaluation (ray_evaluator_vanilla.py)

This evaluator:
- Uses KG search and multi-turn reasoning for generation 
- Replaces reward-based scoring with parallel LLM-as-judge evaluation
- Supports MultiTQ and other temporal KG datasets
"""

import os
import re
import json
import uuid
import openai
import asyncio
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List
import numpy as np
import ray
import torch
from tqdm import tqdm

from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.ray_evaluator_kg import RayKGEvaluator
from verl.protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto

# Import LLM judge configuration
try:
    from KEYS import DEFAULT_JUDGE_MODEL, JUDGE_MODELS
except ImportError:
    print("Warning: KEYS.py not found, using default LLM judge configuration")
    DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
    JUDGE_MODELS = {
        "gpt-4o-mini": {
            "api_key": "your-api-key-here",
            "model_name": "gpt-4o-mini", 
            "provider": "openai",
            "max_tokens": 500,
            "temperature": 0.1,
            "timeout": 15
        }
    }


class RayKGLLMJudgeEvaluator(RayKGEvaluator):
    """
    KG Evaluator with LLM Judge.
    
    Inherits KG generation framework from RayKGEvaluator and replaces
    reward computation with parallel LLM judge evaluation.
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
        eval_samples: int = 0,
        save_detailed_results: bool = False,
        ray_worker_group_cls=None,
        **kwargs
    ):
        # Initialize parent KG evaluator (inherits generation framework)
        super().__init__(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            device_name=device_name,
            n_rollout_eval=n_rollout_eval,
            k_values=k_values,
            val_reward_fn=None,  # We replace this with LLM judge
            eval_samples=eval_samples,
            save_detailed_results=save_detailed_results,
            ray_worker_group_cls=ray_worker_group_cls,
            **kwargs
        )
        
        # LLM-as-Judge configuration (copied from ray_evaluator_vanilla.py)
        self.use_llm_judge = True
        
        # Fallback rate tracking
        self.llm_judge_total_calls = 0
        self.llm_judge_successful_calls = 0
        self.llm_judge_fallbacks = 0
        self.vector_extraction_failures = 0
        
        print(f"[KG-LLM-JUDGE] Initialized evaluator with:")
        print(f"  - KG generation: {self.use_search_generation}")
        print(f"  - LLM judge evaluation: {self.use_llm_judge}")
        print(f"  - Judge model: {DEFAULT_JUDGE_MODEL}")
        print(f"  - N rollout eval: {n_rollout_eval}")
        print(f"  - K values: {k_values}")
        print(f"  - Save detailed results: {self.save_detailed_results}")
        
        # Note: detailed_results_file will be initialized by parent class's evaluate_dataset method
        # Do not override it here to avoid breaking the parent's file opening logic
    
    def _evaluate_batch(self, test_data: dict) -> Dict[str, Any]:
        """
        Evaluate a single batch with KG generation + LLM judge evaluation.
        
        This method combines:
        1. KG generation framework from RayKGEvaluator._evaluate_batch()
        2. LLM judge evaluation from RayVanillaEvaluator.compute_vanilla_rewards_with_llm_judge()
        
        Args:
            test_data: Dictionary containing batch data from dataloader
            
        Returns:
            Dictionary containing batch evaluation metrics
        """
        
        # ==================== GENERATION PHASE ====================
        # Use parent class generation logic (KG search + multi-turn reasoning)
        
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
        
        # Store original test data for LLM judge evaluation BEFORE DataProto conversion
        if self.use_llm_judge:
            # Store pre-extracted data for proper alignment in LLM judge
            original_questions = []
            original_ground_truths = []
            
            # CRITICAL: Extract from ORIGINAL test_data dictionary BEFORE DataProto operations
            batch_size = len(test_data.get("input_ids", []))
            print(f"[KG-LLM-JUDGE WORKFLOW] PRE-EXTRACTION: Found {batch_size} samples in original test_data")
            
            for i in range(batch_size):
                # Extract question from original test_data input_ids
                if "input_ids" in test_data and len(test_data["input_ids"]) > i:
                    input_ids = test_data["input_ids"][i]
                    if hasattr(input_ids, 'dtype') and input_ids.dtype != torch.long:
                        input_ids = input_ids.long()
                    prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    question = self.extract_question_from_kg_prompt(prompt_text)
                else:
                    question = f"Question {i+1}"
                    
                # Extract ground truth from original test_data reward_model
                ground_truth_entities = []
                if "reward_model" in test_data and len(test_data["reward_model"]) > i:
                    reward_info = test_data["reward_model"][i]
                    if isinstance(reward_info, dict) and 'ground_truth' in reward_info:
                        gt = reward_info['ground_truth']
                        if isinstance(gt, dict) and 'target_text' in gt:
                            ground_truth_entities = gt['target_text']
                        elif isinstance(gt, (str, list)):
                            ground_truth_entities = [gt] if isinstance(gt, str) else gt
                
                print(f"[KG-LLM-JUDGE WORKFLOW] PRE-EXTRACTION Sample {i}: Q='{question[:50]}...' GT={ground_truth_entities}")
                                
                original_questions.append(question)
                original_ground_truths.append(ground_truth_entities)
        
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
        
        # Generate responses using KG search
        if not self.use_search_generation:
            # Standard generation
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)
        else:
            # KG search generation (main path for MultiTQ)
            print(f"[KG-LLM-JUDGE] Starting KG search generation for {len(expanded_gen_batch)} samples...")
            first_input_ids = test_gen_batch_padded.batch['input_ids'][:, -self.generation_manager.config.max_start_length:].clone().long()
            test_output_gen_batch_padded = self.generation_manager.run_llm_loop(
                gen_batch=test_gen_batch_padded,
                initial_input_ids=first_input_ids,
            )
            
            # Log detailed sample information similar to trainer
            self._log_sample_details(test_gen_batch_padded, test_output_gen_batch_padded)
            
        print(f"[KG-LLM-JUDGE] KG generation completed successfully")
        
        # Unpad results
        test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
        
        # Expand original batch to match generated responses
        test_batch.non_tensor_batch["uid"] = np.array(base_uids, dtype=object)
        
        # Expand test_batch to match the expanded generation
        expanded_test_batch = test_batch.repeat(repeat_times=self.n_rollout_eval, interleave=True)
        
        # Merge generation outputs with test batch (same as parent)
        final_test_batch = expanded_test_batch.union(test_output_gen_batch)
        
        # Connect pre-extracted data to final batch
        if self.use_llm_judge:
            if final_test_batch.meta_info is None:
                final_test_batch.meta_info = {}
                
            final_test_batch.meta_info['original_questions'] = original_questions
            final_test_batch.meta_info['original_ground_truths'] = original_ground_truths
            final_test_batch.meta_info['original_batch_size'] = batch_size
            
            print(f"[KG-LLM-JUDGE WORKFLOW] STORED: {len(original_questions)} questions, {len(original_ground_truths)} ground truths")
        
        # ==================== EVALUATION PHASE ====================
        # Replace reward computation with LLM judge evaluation
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== STARTING LLM JUDGE EVALUATION ==========")
        print(f"[KG-LLM-JUDGE WORKFLOW] Final batch size: {len(final_test_batch.batch.get('input_ids', []))}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Original batch size: {batch_size}")
        print(f"[KG-LLM-JUDGE WORKFLOW] N rollout eval: {self.n_rollout_eval}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Expected total responses: {batch_size * self.n_rollout_eval}")
        
        # Use LLM judge evaluation instead of reward computation
        reward_results = self.compute_kg_rewards_with_llm_judge(final_test_batch)
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== LLM JUDGE EVALUATION COMPLETED ==========")
        
        # Extract metrics and compute pass@k for all k values
        batch_metrics = self._compute_batch_passatk_metrics([final_test_batch], reward_results)
        
        # CRITICAL: Add individual sample data for detailed results saving
        if self.save_detailed_results:
            print(f"[KG-LLM-JUDGE] Collecting individual sample data for detailed results...")
            inputs = []
            outputs = []
            scores = []
            reward_extra_infos = []
            
            # Extract data from final_test_batch for each sample
            for i in range(len(final_test_batch.batch.get('input_ids', []))):
                try:
                    # Extract prompt/input
                    if 'input_ids' in final_test_batch.batch and len(final_test_batch.batch['input_ids']) > i:
                        input_ids = final_test_batch.batch['input_ids'][i]
                        if input_ids.dtype != torch.long:
                            input_ids = input_ids.long()
                        prompt = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                    else:
                        prompt = f"Sample {i+1}"
                    
                    # Extract response/output
                    if 'responses' in final_test_batch.batch and len(final_test_batch.batch['responses']) > i:
                        response_ids = final_test_batch.batch['responses'][i]
                        if response_ids.dtype != torch.long:
                            response_ids = response_ids.long()
                        response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                        # Extract assistant response for consistency
                        from verl.utils.reward_score.qa_em_format_kg import extract_assistant_response
                        response = extract_assistant_response(response)
                    else:
                        response = ""
                    
                    # Extract score
                    score = reward_results['reward_tensor'][i].item() if i < len(reward_results['reward_tensor']) else 0.0
                    
                    # Extract extra info
                    extra_info = {}
                    if 'reward_extra_info' in reward_results:
                        for key, values in reward_results['reward_extra_info'].items():
                            if i < len(values):
                                extra_info[key] = values[i]
                    
                    inputs.append(prompt)
                    outputs.append(response)
                    scores.append(score)
                    reward_extra_infos.append(extra_info)
                    
                except Exception as e:
                    print(f"[KG-LLM-JUDGE WARNING] Failed to extract data for sample {i}: {e}")
                    inputs.append(f"Sample {i+1}")
                    outputs.append("")
                    scores.append(0.0)
                    reward_extra_infos.append({})
            
            # Add to batch_metrics for detailed results saving
            batch_metrics['inputs'] = inputs
            batch_metrics['outputs'] = outputs  
            batch_metrics['scores'] = scores
            batch_metrics['reward_extra_infos'] = reward_extra_infos
            
            print(f"[KG-LLM-JUDGE] Collected data for {len(inputs)} samples for detailed results")
        
        # Log detailed batch information for first few samples
        print(f"\n=== KG-LLM-JUDGE BATCH SUMMARY ===")
        print(f"Batch size: {len(final_test_batch.batch.get('input_ids', []))}")
        print(f"Generated {self.n_rollout_eval} responses per prompt")
        print(f"Used KG search: {self.use_search_generation}")
        print(f"Used LLM judge: {self.use_llm_judge}")
        
        # Show LLM judge statistics
        if self.llm_judge_total_calls > 0:
            success_rate = (self.llm_judge_successful_calls / self.llm_judge_total_calls) * 100
            fallback_rate = (self.llm_judge_fallbacks / self.llm_judge_total_calls) * 100
            print(f"LLM Judge Success Rate: {success_rate:.1f}%")
            print(f"LLM Judge Fallback Rate: {fallback_rate:.1f}%")
        
        print(f"=== END KG-LLM-JUDGE SUMMARY ===\n")
        
        # Save detailed results to JSONL file if enabled
        if self.save_detailed_results and self.detailed_results_file:
            # Need to determine dataset name and batch index for saving
            # Extract dataset name from test_data if available
            dataset_name = test_data.get('dataset_name', 'test')
            if isinstance(dataset_name, (list, np.ndarray)) and len(dataset_name) > 0:
                dataset_name = dataset_name[0] if hasattr(dataset_name[0], 'lower') else str(dataset_name[0])
            batch_idx = getattr(self, '_current_batch_idx', 0)  # Use counter if available
            
            print(f"[KG-LLM-JUDGE] Saving detailed results for batch {batch_idx} to JSONL file...")
            self._save_batch_details(batch_metrics, batch_idx, dataset_name)
        
        return batch_metrics
    
    def compute_kg_rewards_with_llm_judge(self, batch: DataProto) -> Dict[str, Any]:
        """
        Compute KG rewards using parallel LLM judge processing.
        
        This replaces the reward computation from ray_evaluator_kg.py with
        LLM judge evaluation from ray_evaluator_vanilla.py.
        
        Args:
            batch: DataProto containing responses and metadata
            
        Returns:
            Dictionary with reward_tensor and reward_extra_info in same format as reward manager
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
            
            print(f"[KG-LLM-JUDGE DEBUG] Responses tensor shape: {responses_shape}")
            print(f"[KG-LLM-JUDGE DEBUG] Using num_responses: {num_responses}")
        else:
            num_responses = 0
            print(f"[KG-LLM-JUDGE DEBUG] No responses in batch")
        
        # Calculate effective number of samples to evaluate
        original_batch_size = batch.meta_info.get('original_batch_size', num_responses) if hasattr(batch, 'meta_info') and batch.meta_info else num_responses
        dropped_batch_size = batch.meta_info.get('dropped_batch_size', num_responses) if hasattr(batch, 'meta_info') and batch.meta_info else num_responses
        
        # Use the actual batch size after dropping, not the original
        effective_samples = min(dropped_batch_size, num_responses)
        if self.eval_samples > 0:
            # Respect eval_samples limit
            original_eval_limit = min(original_batch_size, self.eval_samples)
            effective_samples = min(dropped_batch_size, original_eval_limit)
            
        # WORKFLOW DEBUG: Print comprehensive batch information
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== BATCH ANALYSIS ==========")
        print(f"[KG-LLM-JUDGE WORKFLOW] Responses tensor shape: {responses_shape if 'responses' in batch.batch else 'NO_RESPONSES'}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Num responses from tensor: {num_responses}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Original batch size: {original_batch_size}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Dropped batch size: {dropped_batch_size}")
        print(f"[KG-LLM-JUDGE WORKFLOW] N rollout eval: {self.n_rollout_eval}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Effective samples to evaluate: {effective_samples}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Samples to skip: {num_responses - effective_samples}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Eval samples limit: {self.eval_samples}")
        
        # Debug meta_info contents
        if hasattr(batch, 'meta_info') and batch.meta_info:
            print(f"[KG-LLM-JUDGE WORKFLOW] Meta info keys: {list(batch.meta_info.keys())}")
            if 'original_questions' in batch.meta_info:
                print(f"[KG-LLM-JUDGE WORKFLOW] Pre-extracted questions count: {len(batch.meta_info['original_questions'])}")
            if 'original_ground_truths' in batch.meta_info:
                print(f"[KG-LLM-JUDGE WORKFLOW] Pre-extracted ground truths count: {len(batch.meta_info['original_ground_truths'])}")
        else:
            print(f"[KG-LLM-JUDGE WORKFLOW] WARNING: No meta_info found in batch")
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== STARTING PARALLEL EVALUATION ==========")
        
        # Prepare evaluation tasks
        evaluation_tasks = []
        
        for i in range(effective_samples):
            try:
                # WORKFLOW DEBUG: Track processing for each sample
                original_idx = i // self.n_rollout_eval  # Account for n_rollout_eval repeats
                
                # SAFETY CHECK: Ensure original_idx is within bounds for pre-extracted data
                if (hasattr(batch, 'meta_info') and batch.meta_info and 
                    'original_questions' in batch.meta_info and
                    original_idx >= len(batch.meta_info['original_questions'])):
                    print(f"[KG-LLM-JUDGE WORKFLOW] ⚠️ ALIGNMENT ERROR: original_idx={original_idx} >= len(original_questions)={len(batch.meta_info['original_questions'])}")
                    original_idx = min(original_idx, len(batch.meta_info['original_questions']) - 1)
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ FIXED: Using original_idx={original_idx} instead")
                
                print(f"[KG-LLM-JUDGE WORKFLOW] ========== PROCESSING SAMPLE {i} (original_idx={original_idx}) ==========")
                
                # QUESTION EXTRACTION DEBUG
                
                if (hasattr(batch, 'meta_info') and batch.meta_info and 
                    'original_questions' in batch.meta_info and
                    original_idx < len(batch.meta_info['original_questions'])):
                    question = batch.meta_info['original_questions'][original_idx]
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Question from pre-extracted: '{question[:100]}{'...' if len(question) > 100 else ''}'") 
                else:
                    # Fallback: extract from batch (less reliable due to expansion)
                    print(f"[KG-LLM-JUDGE WORKFLOW] ⚠️ Using fallback question extraction for sample {i}")
                    if 'input_ids' in batch.batch and len(batch.batch['input_ids']) > i:
                        input_ids = batch.batch['input_ids'][i]
                        if input_ids.dtype != torch.long:
                            input_ids = input_ids.long()
                        prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
                        question = self.extract_question_from_kg_prompt(prompt_text)
                        print(f"[KG-LLM-JUDGE WORKFLOW] Extracted from prompt: '{question[:100]}{'...' if len(question) > 100 else ''}'") 
                    else:
                        question = f"Question {i+1}"
                        print(f"[KG-LLM-JUDGE WORKFLOW] ❌ Using fallback question: '{question}'")
                    
                # RESPONSE EXTRACTION DEBUG
                if 'responses' in batch.batch and len(batch.batch['responses']) > i:
                    response_ids = batch.batch['responses'][i]
                    if response_ids.dtype != torch.long:
                        response_ids = response_ids.long()
                    full_response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                    
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Raw response length: {len(full_response)} chars")
                    
                    # Use the same assistant response extraction as standard KG evaluator
                    from verl.utils.reward_score.qa_em_format_kg import extract_assistant_response
                    assistant_response = extract_assistant_response(full_response)
                    
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Assistant response length: {len(assistant_response)} chars")
                    
                    # Show full responses for first 5 samples
                    if i < 5:
                        print(f"[KG-LLM-JUDGE WORKFLOW] === FULL RESPONSE SAMPLE {i} ===")
                        print(f"Full response: {full_response[:500]}{'...' if len(full_response) > 500 else ''}")
                        print(f"Assistant response: {assistant_response[:300]}{'...' if len(assistant_response) > 300 else ''}")
                        print(f"[KG-LLM-JUDGE WORKFLOW] === END RESPONSE SAMPLE {i} ===")
                        
                    # Additional cleaning for KG multi-turn format  
                    predicted_answer = self.extract_answer_from_response(assistant_response)
                    
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Predicted answer: '{predicted_answer}'")
                else:
                    predicted_answer = ""
                    print(f"[KG-LLM-JUDGE WORKFLOW] ❌ No response found for sample {i}")
                    
                # GROUND TRUTH EXTRACTION DEBUG
                if (hasattr(batch, 'meta_info') and batch.meta_info and 
                    'original_ground_truths' in batch.meta_info and
                    original_idx < len(batch.meta_info['original_ground_truths'])):
                    ground_truth_entities = batch.meta_info['original_ground_truths'][original_idx]
                    
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Ground truth from pre-extracted: {ground_truth_entities}")
                else:
                    # Fallback: extract from batch (less reliable due to expansion)
                    print(f"[KG-LLM-JUDGE WORKFLOW] ⚠️ Using fallback ground truth extraction for sample {i}")
                    ground_truth_entities = []
                    if hasattr(batch, 'non_tensor_batch') and batch.non_tensor_batch is not None:
                        print(f"[KG-LLM-JUDGE WORKFLOW] Checking non_tensor_batch for reward_model...")
                        if 'reward_model' in batch.non_tensor_batch and len(batch.non_tensor_batch['reward_model']) > original_idx:
                            reward_info = batch.non_tensor_batch['reward_model'][original_idx]
                            print(f"[KG-LLM-JUDGE WORKFLOW] Found reward_info type: {type(reward_info)}")
                            
                            if isinstance(reward_info, dict) and 'ground_truth' in reward_info:
                                gt = reward_info['ground_truth']
                                print(f"[KG-LLM-JUDGE WORKFLOW] Ground truth type: {type(gt)}, content: {gt}")
                                # Handle MultiTQ format: {'target_kb_id': [...], 'target_text': [...]}
                                if isinstance(gt, dict) and 'target_text' in gt:
                                    ground_truth_entities = gt['target_text']
                                elif isinstance(gt, str):
                                    ground_truth_entities = [gt]
                                elif isinstance(gt, list):
                                    ground_truth_entities = gt
                                else:
                                    ground_truth_entities = [str(gt)]
                            else:
                                print(f"[KG-LLM-JUDGE WORKFLOW] No ground_truth in reward_info: {reward_info}")
                        else:
                            print(f"[KG-LLM-JUDGE WORKFLOW] No reward_model or insufficient length in non_tensor_batch")
                    else:
                        print(f"[KG-LLM-JUDGE WORKFLOW] No non_tensor_batch available")
                    
                    if not ground_truth_entities:
                        print(f"[KG-LLM-JUDGE WORKFLOW] ❌ No ground truth found for sample {i}, using empty list")
                        ground_truth_entities = []
                    else:
                        print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Fallback ground truth: {ground_truth_entities}")
                
                # WORKFLOW ALIGNMENT DEBUG: Show complete picture for each sample
                print(f"[KG-LLM-JUDGE WORKFLOW] === SAMPLE {i} FINAL ALIGNMENT ===")
                print(f"[KG-LLM-JUDGE WORKFLOW] Question: {question}")
                print(f"[KG-LLM-JUDGE WORKFLOW] Predicted: '{predicted_answer}'")
                print(f"[KG-LLM-JUDGE WORKFLOW] Ground truth: {ground_truth_entities}")
                print(f"[KG-LLM-JUDGE WORKFLOW] Sample index: {i}, Original index: {original_idx}")
                
                # Sanity check for alignment
                if question and ground_truth_entities and predicted_answer:
                    print(f"[KG-LLM-JUDGE WORKFLOW] ✅ Complete data - ready for LLM judge")
                else:
                    missing = []
                    if not question: missing.append("question")
                    if not ground_truth_entities: missing.append("ground_truth")
                    if not predicted_answer: missing.append("predicted_answer")
                    print(f"[KG-LLM-JUDGE WORKFLOW] ⚠️ Missing data: {missing}")
                
                print(f"[KG-LLM-JUDGE WORKFLOW] === END SAMPLE {i} ALIGNMENT ===")
                
                evaluation_tasks.append({
                    'index': i,
                    'question': question,
                    'predicted_answer': predicted_answer, 
                    'ground_truth_entities': ground_truth_entities,
                    'failed': False
                })
                
            except Exception as e:
                print(f"[KG-LLM-JUDGE ERROR] Failed to prepare task {i}: {e}")
                evaluation_tasks.append({
                    'index': i,
                    'question': f"Question {i+1}",
                    'predicted_answer': "",
                    'ground_truth_entities': [],
                    'failed': True
                })
        
        # Parallel processing with ThreadPoolExecutor (same as vanilla evaluator)
        def evaluate_single_task(task):
            """Evaluate a single task with LLM judge."""
            if task['failed']:
                return {
                    'index': task['index'],
                    'score': 0.0,
                    'exact_match': 0.0, 
                    'f1': 0.0,
                    'binary_vector': [],
                    'failed': True
                }
                
            try:
                # WORKFLOW DEBUG: Call LLM judge with detailed logging for first few samples
                detailed_log = task['index'] < 3  # Show detailed logs for first 3 samples
                
                if detailed_log:
                    print(f"[KG-LLM-JUDGE WORKFLOW] === CALLING LLM JUDGE FOR TASK {task['index']} ===")
                    
                binary_vector = self.evaluate_with_llm_judge(
                    task['question'],
                    task['predicted_answer'],
                    task['ground_truth_entities'],
                    detailed_log=detailed_log
                )
                
                if detailed_log:
                    print(f"[KG-LLM-JUDGE WORKFLOW] === LLM JUDGE RESPONSE FOR TASK {task['index']}: {binary_vector} ===")
                
                # Compute metrics from binary vector
                if binary_vector and len(binary_vector) > 0:
                    exact_match_score = 1.0 if all(binary_vector) else 0.0
                    f1_score = sum(binary_vector) / len(binary_vector) if binary_vector else 0.0
                    overall_score = f1_score  # Use F1 as overall score
                else:
                    exact_match_score = 0.0
                    f1_score = 0.0
                    overall_score = 0.0
                    binary_vector = []
                
                return {
                    'index': task['index'],
                    'score': overall_score,
                    'exact_match': exact_match_score,
                    'f1': f1_score,
                    'binary_vector': binary_vector,
                    'failed': False
                }
                
            except Exception as e:
                print(f"[KG-LLM-JUDGE ERROR] Failed to evaluate task {task['index']}: {e}")
                return {
                    'index': task['index'],
                    'score': 0.0,
                    'exact_match': 0.0,
                    'f1': 0.0,
                    'binary_vector': [],
                    'failed': True
                }
        
        # WORKFLOW DEBUG: Parallel evaluation setup
        max_workers = min(32, len(evaluation_tasks)) if evaluation_tasks else 1  # Increased from 12 to 32
        start_time = time.time()
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== PARALLEL EVALUATION SETUP ===========")
        print(f"[KG-LLM-JUDGE WORKFLOW] Total evaluation tasks: {len(evaluation_tasks)}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Max workers: {max_workers}")
        print(f"[KG-LLM-JUDGE WORKFLOW] LLM Judge model: {DEFAULT_JUDGE_MODEL}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Starting parallel evaluation...")
        
        # Initialize result arrays
        scores = [0.0] * effective_samples
        exact_match_scores = [0.0] * effective_samples
        f1_scores = [0.0] * effective_samples
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_task = {executor.submit(evaluate_single_task, task): task for task in evaluation_tasks}
            
            # Collect results as they complete
            completed_count = 0
            for future in as_completed(future_to_task):
                result = future.result()
                idx = result['index']
                
                if idx < effective_samples:  # Safety check
                    scores[idx] = result['score']
                    exact_match_scores[idx] = result['exact_match']
                    f1_scores[idx] = result['f1']
                
                completed_count += 1
                
                # WORKFLOW DEBUG: Show progress and individual results
                print(f"[KG-LLM-JUDGE WORKFLOW] Completed task {idx}: score={result['score']:.3f}, f1={result['f1']:.3f}, binary_vector={result.get('binary_vector', [])}")
                
                if completed_count % 10 == 0 or completed_count == len(evaluation_tasks):
                    print(f"[KG-LLM-JUDGE WORKFLOW] Progress: {completed_count}/{len(evaluation_tasks)} evaluations completed")
        
        end_time = time.time()
        evaluation_duration = end_time - start_time
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== PARALLEL EVALUATION COMPLETED ===========")
        print(f"[KG-LLM-JUDGE WORKFLOW] Total duration: {evaluation_duration:.1f}s")
        if len(evaluation_tasks) > 0:
            print(f"[KG-LLM-JUDGE WORKFLOW] Average per evaluation: {evaluation_duration/len(evaluation_tasks):.2f}s")
            print(f"[KG-LLM-JUDGE WORKFLOW] Throughput: {len(evaluation_tasks)/evaluation_duration:.2f} evaluations/sec")
        else:
            print(f"[KG-LLM-JUDGE WORKFLOW] ❌ No evaluation tasks processed - check batch size calculation")
        
        # Pad results to match full batch size if needed
        while len(scores) < num_responses:
            scores.append(0.0)
            exact_match_scores.append(0.0)
            f1_scores.append(0.0)
        
        # Convert to tensors for compatibility with existing metrics computation
        reward_tensor = torch.tensor(scores[:num_responses], dtype=torch.float32)
        
        # Create reward_extra_info in same format as reward managers
        reward_extra_info = {
            'exact_match': exact_match_scores[:num_responses],
            'f1': f1_scores[:num_responses],
            'precision': f1_scores[:num_responses],  # Use F1 as approximation
            'recall': f1_scores[:num_responses],     # Use F1 as approximation
            'scores': scores[:num_responses]
        }
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== FINAL RESULTS SUMMARY ===========")
        print(f"[KG-LLM-JUDGE WORKFLOW] Total scores generated: {len(scores)}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Effective samples evaluated: {effective_samples}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Mean score: {np.mean(scores[:effective_samples]):.4f}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Mean exact match: {np.mean(exact_match_scores[:effective_samples]):.4f}")
        print(f"[KG-LLM-JUDGE WORKFLOW] Mean F1: {np.mean(f1_scores[:effective_samples]):.4f}")
        
        # Show distribution of scores
        if effective_samples > 0:
            score_dist = np.bincount([int(s * 10) for s in scores[:effective_samples] if 0 <= s <= 1], minlength=11)
            print(f"[KG-LLM-JUDGE WORKFLOW] Score distribution (0.0-1.0 in 0.1 bins): {score_dist.tolist()}")
        
        print(f"[KG-LLM-JUDGE WORKFLOW] ========== END RESULTS SUMMARY =========")
        
        return {
            'reward_tensor': reward_tensor,
            'reward_extra_info': reward_extra_info
        }
    
    def extract_question_from_kg_prompt(self, prompt_text: str) -> str:
        """
        Extract question from KG prompt format.
        
        KG prompts typically have format:
        System: [instructions]
        User: [question]
        [KG search results]
        """
        try:
            # Look for common patterns in KG prompts
            if "User:" in prompt_text:
                # Extract user question part
                user_part = prompt_text.split("User:")[-1].strip()
                # Remove any KG search results that might be appended
                lines = user_part.split('\n')
                # Take first non-empty line as the question
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("[") and not line.startswith("Search"):
                        return line
            
            # Fallback patterns
            if "Question:" in prompt_text:
                return prompt_text.split("Question:")[-1].strip().split('\n')[0].strip()
                
            # If no clear pattern, return first substantial line
            lines = prompt_text.split('\n')
            for line in lines:
                line = line.strip()
                if len(line) > 10 and not line.startswith("System:") and not line.startswith("User:"):
                    return line
                    
            return prompt_text[:200] + "..." if len(prompt_text) > 200 else prompt_text
            
        except Exception as e:
            print(f"[KG-LLM-JUDGE WARNING] Failed to extract question: {e}")
            return prompt_text[:100] + "..." if len(prompt_text) > 100 else prompt_text
    
    def extract_answer_from_response(self, raw_response: str) -> str:
        """
        Extract answer from model response using only <answer> tags.
        
        This uses the same simple extraction method as ray_evaluator_kg.py
        for consistency across KG evaluation modes.
        
        Args:
            raw_response: Raw model response containing <answer> tags
            
        Returns:
            Extracted answer string or empty string if no answer found
        """
        try:
            import re
            
            # Step 1: Remove information blocks that might contain error messages
            cleaned_str = re.sub(r'<information>.*?</information>', '', raw_response, flags=re.DOTALL)
            
            # Step 2: Find all answer tags
            answer_pattern = r'<answer>(.*?)</answer>'
            matches = re.findall(answer_pattern, cleaned_str, re.DOTALL)
            
            # Step 3: Return first answer or empty string
            if len(matches) == 0:
                print(f"[KG-LLM-JUDGE] No <answer> tags found in response")
                return ""
            
            answer_content = matches[0].strip()
            print(f"[KG-LLM-JUDGE] Extracted from <answer> tags: '{answer_content}'")
            return answer_content
                        
        except Exception as e:
            print(f"[KG-LLM-JUDGE WARNING] Failed to extract answer: {e}")
            return ""
    
    def _is_multitq_evaluation(self, question: str, ground_truth_entities: List[str]) -> bool:
        """
        Detect if this is MultiTQ evaluation based on question or ground truth format.
        
        Args:
            question: The question text
            ground_truth_entities: List of ground truth entities
            
        Returns:
            True if this appears to be MultiTQ evaluation
        """
        # Check for temporal patterns in ground truth entities
        temporal_patterns = [
            r'^\d{4}-\d{2}$',  # YYYY-MM format
            r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD format
            r'^\d{4}$',  # Year only
        ]
        
        # Check if any ground truth entities match temporal patterns
        for entity in ground_truth_entities:
            if isinstance(entity, str):
                for pattern in temporal_patterns:
                    if re.match(pattern, entity.strip()):
                        return True
        
        # Check for MultiTQ-specific question patterns
        multitq_indicators = [
            'when did', 'what year', 'what month', 'what day',
            'during which', 'in which year', 'in which month',
            'before', 'after', 'between', 'from', 'until',
            'start', 'end', 'begin', 'finish'
        ]
        
        question_lower = question.lower()
        for indicator in multitq_indicators:
            if indicator in question_lower:
                return True
                
        return False
    
    def create_llm_judge_prompt(self, question: str, predicted_answer: str, ground_truth_entities: List[str]) -> str:
        """
        Create evaluation prompt for LLM-as-Judge adapted for KG/MultiTQ evaluation.
        
        Args:
            question: The original question
            predicted_answer: Model's predicted answer
            ground_truth_entities: List of correct entities/answers
            
        Returns:
            Prompt for LLM judge evaluation focusing on semantic equivalence
        """
        entities_str = ", ".join([f"'{entity}'" for entity in ground_truth_entities])
        
        # Make the expected vector length very clear
        num_entities = len(ground_truth_entities)
        vector_example = "[" + ",".join(["0"] * num_entities) + "]"
        
        # Use MultiTQ-specific prompt if detected
        if self._is_multitq_evaluation(question, ground_truth_entities):
            return self._create_multitq_judge_prompt(question, predicted_answer, ground_truth_entities, entities_str, num_entities, vector_example)
        else:
            return self._create_general_judge_prompt(question, predicted_answer, ground_truth_entities, entities_str, num_entities, vector_example)
    
    def _create_multitq_judge_prompt(self, question: str, predicted_answer: str, ground_truth_entities: List[str], entities_str: str, num_entities: int, vector_example: str) -> str:
        """
        Create MultiTQ-specific LLM judge prompt with temporal format handling.
        """
        prompt = f"""For each of the {num_entities} gold temporal entities in order, does the prediction refer to the same time period with appropriate temporal precision?

Respond with EXACTLY {num_entities} numbers in format {vector_example}.
1 = correct temporal match with adequate precision, 0 = wrong time period or inadequate precision.

MultiTQ Temporal Matching Rules:
- Exact format matches: 1 (e.g., "2018-02" = "February 2018" = "Feb 2018")
- Year equivalence: 1 (e.g., "2019" = "year 2019" = "in 2019")  
- Date equivalence: 1 (e.g., "2021-05-15" = "May 15, 2021")
- Wrong precision: 0 (e.g., "2018" when specific month "2018-07" required)
- Wrong time period: 0 (e.g., "2017" when "2018-03" required)

Examples:
gold: ['2018-02'] predicted: February 2018
binary vector: [1]

gold: ['2020-03', '2020-04'] predicted: March and April 2020
binary vector: [1,1]

gold: ['2019'] predicted: The year 2019
binary vector: [1]

gold: ['2021-05-15'] predicted: May 15th, 2021
binary vector: [1]

gold: ['2018-03'] predicted: 2018
binary vector: [0]

gold: [{entities_str}] ({num_entities} temporal entities)
predicted: {predicted_answer}
binary vector:"""
        
        return prompt
    
    def _create_general_judge_prompt(self, question: str, predicted_answer: str, ground_truth_entities: List[str], entities_str: str, num_entities: int, vector_example: str) -> str:
        """
        Create general KG evaluation prompt for non-MultiTQ datasets.
        """
        prompt = f"""For each of the {num_entities} gold entities in order, does the prediction refer to the same real-world entity with the same level of specificity? 

Respond with EXACTLY {num_entities} numbers in format {vector_example}.
1 = same entity with adequate specificity, 0 = different entity or insufficient specificity.

Rules for temporal/KG questions:
- Exact matches or clear equivalents: 1 (e.g., "Apple Inc." = "Apple")
- Temporally correct answers: 1 (e.g., "Obama was president 2009-2017" matches "Barack Obama" for presidential questions)  
- Too general when specifics required: 0 (e.g., "Islam" when ["Shia Islam", "Sunni Islam"] needed)
- Partial but incomplete: 0 (e.g., "Islam" covers both but misses the distinction)
- Wrong time period: 0 (e.g., "current president" when asking about historical president)

Example: gold: ['Apple Inc.'] predicted: Apple
binary vector: [1]

Example: gold: ['New York', 'California'] predicted: NYC and LA  
binary vector: [1,0]

Example: gold: ['Barack Obama'] predicted: Obama was the 44th president
binary vector: [1]

Example: gold: ['2009', '2017'] predicted: Obama served from 2009 to 2017
binary vector: [1,1]

gold: [{entities_str}] ({num_entities} entities)
predicted: {predicted_answer}
binary vector:"""
        
        return prompt
    
    def evaluate_with_llm_judge(self, question: str, predicted_answer: str, ground_truth_entities: List[str], detailed_log: bool = False) -> List[int]:
        """
        Use LLM judge to evaluate if predicted answer covers ground truth entities.
        
        Args:
            question: The original question
            predicted_answer: Model's predicted answer
            ground_truth_entities: List of correct entities
            detailed_log: Whether to log detailed information
            
        Returns:
            Binary vector indicating which ground truth entities are covered
        """
        # Use same logic as vanilla evaluator with KG-specific adaptations
        if not ground_truth_entities:
            print(f"[KG-LLM-JUDGE] Warning: No ground truth entities provided")
            return []
        
        try:
            # Track total calls
            self.llm_judge_total_calls += 1
            
            # Handle empty cases
            if not predicted_answer.strip():
                print(f"[KG-LLM-JUDGE] Empty prediction, returning zeros")
                self.llm_judge_fallbacks += 1
                return [0] * len(ground_truth_entities)
            
            if not ground_truth_entities:
                print(f"[KG-LLM-JUDGE] Empty ground truth, returning empty vector")
                return []
            
            # Create evaluation prompt
            prompt = self.create_llm_judge_prompt(question, predicted_answer, ground_truth_entities)
            
            if detailed_log:
                print(f"[KG-LLM-JUDGE DETAILED] === LLM Judge Prompt ===")
                print(f"Question: {question}")
                print(f"Predicted: {predicted_answer}")
                print(f"Ground Truth: {ground_truth_entities}")
                print(f"Prompt: {prompt[:500]}{'...' if len(prompt) > 500 else ''}")
                print("=" * 50)
            else:
                # Always show basic info for debugging workflow
                print(f"[KG-LLM-JUDGE] Evaluating: Q='{question[:50]}...', P='{predicted_answer[:30]}...', GT={ground_truth_entities}")
            
            try:
                # Get judge model configuration
                judge_model = DEFAULT_JUDGE_MODEL
                judge_config = JUDGE_MODELS.get(judge_model, JUDGE_MODELS["gpt-4o-mini"])
                
                client = openai.OpenAI(api_key=judge_config["api_key"])
                
                # Handle different model configurations
                request_params = {
                    "model": judge_config["model_name"],
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "timeout": judge_config.get("timeout", 15)
                }
                
                # Handle model-specific parameters (same as vanilla evaluator)
                if judge_model.startswith("gpt-5"):
                    # GPT-5 doesn't support custom temperature, uses default
                    pass
                else:
                    request_params["temperature"] = judge_config.get("temperature", 0.1)
                
                # Use max_completion_tokens for GPT-5 models
                if "max_completion_tokens" in judge_config:
                    request_params["max_completion_tokens"] = judge_config["max_completion_tokens"]
                else:
                    request_params["max_tokens"] = judge_config.get("max_tokens", 500)
                
                response = client.chat.completions.create(**request_params)
                response_text = response.choices[0].message.content.strip()
                
                if detailed_log:
                    print(f"[KG-LLM-JUDGE DETAILED] GPT Response: {response_text}")
                else:
                    # Always show response for workflow debugging
                    print(f"[KG-LLM-JUDGE] GPT Response: {response_text[:100]}{'...' if len(response_text) > 100 else ''}")
                
                # Extract binary vector using regex
                vector_pattern = r'\[([0-9,\s]+)\]'
                vector_matches = re.findall(vector_pattern, response_text)
                
                if vector_matches:
                    vector_str = vector_matches[-1]  # Take the last match
                    try:
                        binary_vector = [int(x.strip()) for x in vector_str.split(',')]
                        
                        # Ensure vector length matches ground truth entities
                        if len(binary_vector) == len(ground_truth_entities):
                            self.llm_judge_successful_calls += 1
                            
                            # Show actual results in one line
                            vector_summary = [f"'{gt}'→{score}" for gt, score in zip(ground_truth_entities, binary_vector)]
                            print(f"[KG-LLM-JUDGE] ✅ SUCCESS: [{', '.join(vector_summary)}] | Predicted: '{predicted_answer[:200]}{'...' if len(predicted_answer) > 200 else ''}'")
                            
                            if detailed_log:
                                print(f"[KG-LLM-JUDGE DETAILED] Full vector: {binary_vector}")
                                for i, (entity, score) in enumerate(zip(ground_truth_entities, binary_vector)):
                                    status = "✅ MATCH" if score == 1 else "❌ NO MATCH"
                                    print(f"  {i}: '{entity}' → {status}")
                            
                            return binary_vector
                        else:
                            self.vector_extraction_failures += 1
                            print(f"[KG-LLM-JUDGE] ❌ Vector length mismatch: got {len(binary_vector)}, expected {len(ground_truth_entities)}")
                    except (ValueError, IndexError) as e:
                        self.vector_extraction_failures += 1
                        print(f"[KG-LLM-JUDGE] ❌ Vector parsing error: {e}")
                else:
                    print(f"[KG-LLM-JUDGE] ❌ No vector pattern found in response")
                
                # Fallback: if API call succeeded but parsing failed, use heuristic
                self.llm_judge_fallbacks += 1
                print(f"[KG-LLM-JUDGE] 🔄 Fallback: Using heuristic evaluation")
                print(f"[KG-LLM-JUDGE] Question: {question}")
                print(f"[KG-LLM-JUDGE] GPT Response: {response_text}")
                
                # Heuristic fallback
                binary_vector = []
                predicted_lower = predicted_answer.lower()
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    # Simple string matching
                    if entity_lower in predicted_lower or any(word in predicted_lower for word in entity_lower.split()):
                        binary_vector.append(1)
                    else:
                        binary_vector.append(0)
                
                return binary_vector
                
            except openai.APITimeoutError:
                self.llm_judge_fallbacks += 1
                print(f"[KG-LLM-JUDGE] ⏰ API timeout, using heuristic fallback")
                # Fallback to heuristic
                binary_vector = []
                predicted_lower = predicted_answer.lower()
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    if entity_lower in predicted_lower:
                        binary_vector.append(1)
                    else:
                        binary_vector.append(0)
                
                return binary_vector
                
            except openai.APIError as e:
                self.llm_judge_fallbacks += 1
                print(f"[KG-LLM-JUDGE] 🚫 OpenAI API error with {judge_model}: {e}, using heuristic fallback")
                # Fallback to heuristic
                binary_vector = []
                predicted_lower = predicted_answer.lower()
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    if entity_lower in predicted_lower:
                        binary_vector.append(1)
                    else:
                        binary_vector.append(0)
                        
                return binary_vector
                
            except Exception as e:
                self.llm_judge_fallbacks += 1
                print(f"[KG-LLM-JUDGE] ❌ Unexpected error: {e}, using heuristic fallback")
                import traceback
                traceback.print_exc()
                
                # Heuristic fallback
                binary_vector = []
                predicted_lower = predicted_answer.lower()
                for entity in ground_truth_entities:
                    entity_lower = entity.lower()
                    if entity_lower in predicted_lower:
                        binary_vector.append(1)
                    else:
                        binary_vector.append(0)
                
                return binary_vector
            finally:
                client.close()
        
        except Exception as e:
            print(f"[KG-LLM-JUDGE] Critical error in evaluate_with_llm_judge: {e}")
            import traceback
            traceback.print_exc()
            # Return fallback result
            return [0] * len(ground_truth_entities)
    
    def _save_batch_details(self, batch_metrics: Dict[str, Any], batch_idx: int, dataset_name: str):
        """
        Save detailed results for a batch to JSONL file (continuous streaming).
        Adapted from the original KG evaluator for LLM judge results.
        
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
            
            # Create detailed record for LLM judge evaluation
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
                    "prompt_length": len(prompt) if prompt else 0,
                    "evaluation_mode": "kg-llm-judge"
                }
            }
            
            # Extract LLM judge specific metrics from extra_info if available
            if isinstance(extra_info, dict):
                # Extract LLM judge metrics
                for metric_key in ['f1', 'exact_match', 'binary_vector', 'llm_judge_score',
                                 'ground_truth_entities', 'predicted_answer', 'question']:
                    if metric_key in extra_info:
                        record["metrics"][metric_key] = extra_info[metric_key]
                
                # Extract KG-specific information if available
                for kg_key in ['kg_turns', 'search_results', 'generation_time']:
                    if kg_key in extra_info:
                        record["metadata"][kg_key] = extra_info[kg_key]
            
            # Write record to JSONL file
            try:
                json_line = json.dumps(record, ensure_ascii=False, separators=(',', ':'))
                self.detailed_results_file.write(json_line + '\n')
                self.detailed_results_file.flush()  # Ensure immediate write
            except Exception as e:
                print(f"[KG-LLM-JUDGE WARNING] Failed to write detailed result for sample {i}: {e}")


def create_kg_llm_judge_evaluator(
    config,
    tokenizer,
    processor, 
    device_name: str = "cuda",
    n_rollout_eval: int = 8,
    k_values: List[int] = None,
    eval_samples: int = 0,
    save_detailed_results: bool = False,
    **kwargs
) -> RayKGLLMJudgeEvaluator:
    """
    Factory function to create KG LLM Judge evaluator.
    
    Args:
        config: Configuration object with actor, reward, and trainer settings
        tokenizer: Tokenizer for text processing
        processor: Data processor
        device_name: Device to use for computation
        n_rollout_eval: Number of responses to generate per prompt
        k_values: List of k values for pass@k computation
        eval_samples: Number of samples to evaluate (0 for all)
        save_detailed_results: Whether to save detailed evaluation results
        **kwargs: Additional arguments
        
    Returns:
        Initialized RayKGLLMJudgeEvaluator instance
    """
    
    if k_values is None:
        k_values = [1, 2, 3, 4]
    
    print(f"Creating KG LLM Judge evaluator with:")
    print(f"  - Device: {device_name}")
    print(f"  - N rollout eval: {n_rollout_eval}")
    print(f"  - K values: {k_values}")
    print(f"  - Eval samples: {eval_samples}")
    print(f"  - Save detailed results: {save_detailed_results}")
    
    # Define worker classes based on strategy
    if config.actor_rollout_ref.actor.strategy in ["fsdp", "fsdp2"]:
        from verl.single_controller.ray import RayWorkerGroup
        from verl.workers.fsdp_workers import ActorRolloutRefWorker
        
        actor_rollout_cls = ActorRolloutRefWorker
        ray_worker_group_cls = RayWorkerGroup
    else:
        raise NotImplementedError("Only FSDP strategy supported for KG LLM Judge evaluation")
    
    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role
    
    # Setup resource pool (same pattern as KG evaluator)
    global_pool_id = "global_pool"
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    
    role_worker_mapping = {
        Role.ActorRollout: ray.remote(actor_rollout_cls),
    }
    
    mapping = {
        Role.ActorRollout: global_pool_id,
    }
    
    # Add RefPolicy if needed
    if hasattr(config.actor_rollout_ref, 'ref_policy') and config.actor_rollout_ref.ref_policy.enable:
        role_worker_mapping[Role.RefPolicy] = ray.remote(ActorRolloutRefWorker)
        mapping[Role.RefPolicy] = global_pool_id
    
    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)
    
    return RayKGLLMJudgeEvaluator(
        config=config,
        tokenizer=tokenizer,
        processor=processor,
        role_worker_mapping=role_worker_mapping,
        resource_pool_manager=resource_pool_manager,
        ray_worker_group_cls=ray_worker_group_cls,
        device_name=device_name,
        n_rollout_eval=n_rollout_eval,
        k_values=k_values,
        eval_samples=eval_samples,
        save_detailed_results=save_detailed_results,
        **kwargs
    )