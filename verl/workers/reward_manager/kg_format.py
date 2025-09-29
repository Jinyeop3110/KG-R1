# Copyright 2024 PRIME team and/or its affiliates
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

from collections import defaultdict
import os
import datetime

import torch

from verl import DataProto
from verl.utils.reward_score import qa_em_format_kg


def _select_kg_rm_score_fn(data_source):
    """Select the appropriate KG reward scoring function based on data source."""
    if data_source in ['webqsp_kg', 'cwq_kg', 'kgR1_webqsp', 'kgR1_cwq', 'kgR1_multitq']:
        return qa_em_format_kg.compute_score_em_kg_refactored
    else:
        raise NotImplementedError(f"Data source '{data_source}' not supported for KG format reward manager. Supported: webqsp_kg, cwq_kg, kgR1_webqsp, kgR1_cwq, kgR1_multitq")


class KGFormatRewardManager:
    """The KG format-aware reward manager with support for knowledge graph reasoning."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", 
                 answer_match_score=1.0, format_score=0.0, valid_query_reward=0.0, final_format_score=0., 
                 structure_format_score=0.2, retrieval_score=0.1, query_weight=0.,
                 kg_server_error_penalty=-0.05, kg_not_found_penalty=-0.1, kg_format_error_penalty=-0.2, kg_no_data_penalty=-0.02,
                 debug_long_responses=False, response_length_threshold=1000, debug_log_dir="debug_logs",
                 **kwargs) -> None:
        
        print("Initialized KGFormatRewardManager with:",
              f"answer_match_score={answer_match_score}",
              f"format_score={format_score}",
              f"valid_query_reward={valid_query_reward}",
              f"structure_format_score={structure_format_score}",
              f"final_format_score={final_format_score}",
              f"retrieval_score={retrieval_score}",
              f"kg_server_error_penalty={kg_server_error_penalty}",
              f"kg_not_found_penalty={kg_not_found_penalty}",
              f"kg_format_error_penalty={kg_format_error_penalty}",
              f"kg_no_data_penalty={kg_no_data_penalty}",
              f"compute_score={compute_score is not None}",
              f"debug_long_responses={debug_long_responses}",
              f"response_length_threshold={response_length_threshold}",)
        
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score
        self.reward_fn_key = reward_fn_key
        # New reward parameters
        self.answer_match_score = answer_match_score
        self.format_score = format_score
        self.valid_query_reward = valid_query_reward
        self.structure_format_score = structure_format_score
        self.final_format_score = final_format_score
        self.retrieval_score = retrieval_score
        # Differentiated error penalties
        self.kg_server_error_penalty = kg_server_error_penalty
        self.kg_not_found_penalty = kg_not_found_penalty
        self.kg_format_error_penalty = kg_format_error_penalty
        self.kg_no_data_penalty = kg_no_data_penalty
        # Legacy/compatibility
        self.query_weight = query_weight
        
        # Debug parameters for long response logging
        self.debug_long_responses = debug_long_responses
        self.response_length_threshold = response_length_threshold
        self.debug_log_dir = debug_log_dir
        self.debug_step_counter = 0  # Track global step counter for debug logging
        
        # Initialize debug logging if enabled
        if self.debug_long_responses:
            # Check if debug_log_dir ends with .log - if so, use it as a file path
            if self.debug_log_dir.endswith('.log'):
                self.debug_log_file = self.debug_log_dir
                # Create parent directory if it doesn't exist
                parent_dir = os.path.dirname(self.debug_log_file)
                if parent_dir:
                    os.makedirs(parent_dir, exist_ok=True)
            else:
                # Original behavior - treat as directory
                os.makedirs(self.debug_log_dir, exist_ok=True)
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                self.debug_log_file = os.path.join(self.debug_log_dir, f"long_responses_{timestamp}.log")
            print(f"Debug mode enabled: logging long responses (>{response_length_threshold} tokens) to {self.debug_log_file}")

    def _log_long_response_debug(self, sequences_str, sample_interaction_history, ground_truth, score, data_source, response_length, sample_idx, step):
        """Log detailed information for long responses to debug file."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        separator = "=" * 80
        log_content = f"""
{separator}
[LONG RESPONSE DEBUG LOG] - {timestamp}
[Step] {step}
[Sample Index] {sample_idx}
[Data Source] {data_source}
[Response Length] {response_length} tokens (threshold: {self.response_length_threshold})
{separator}

[KG Format Reward Manager - {data_source}]
[sequences] {sequences_str}

sample_interaction_history: {sample_interaction_history}

[ground_truth] {ground_truth}"""

        # Add score information
        if isinstance(score, dict):
            for key, value in score.items():
                log_content += f"\n[{key}] {value}"
        else:
            log_content += f"\n[score] {score}"
        
        log_content += f"\n{separator}\n"
        
        # Write to debug log file
        try:
            with open(self.debug_log_file, 'a', encoding='utf-8') as f:
                f.write(log_content)
        except Exception as e:
            print(f"Warning: Failed to write to debug log file {self.debug_log_file}: {e}")

    def __call__(self, data: DataProto, return_dict=False):
        """Compute rewards with KG format-aware scoring."""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        
        # Increment step counter for debug logging
        self.debug_step_counter += 1

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            
            # Debug logging to understand prompt extraction
            if i == 0 and already_print_data_sources.get(data_source, 0) == 0:
                print(f"[DEBUG] Prompt extraction diagnostics:")
                print(f"  - prompt_length (shape): {prompt_length}")
                print(f"  - valid_prompt_length (sum of mask): {valid_prompt_length}")
                print(f"  - prompt_ids shape: {prompt_ids.shape}")
                print(f"  - First 10 attention mask values: {data_item.batch['attention_mask'][:10].tolist()}")
                print(f"  - Taking last {valid_prompt_length} tokens from prompt_ids")
            
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            # Convert to long integers if they're floats (can happen with search generation)
            if sequences.dtype != torch.long:
                sequences = sequences.long()
            sequences_str = self.tokenizer.decode(sequences)
            
            # Ensure the sequence has the expected conversation format for validation
            # The KG scoring function expects "<|im_start|>assistant" marker
            if "<|im_start|>assistant" not in sequences_str:
                # Find where the response part starts (after the prompt)
                prompt_str = self.tokenizer.decode(valid_prompt_ids)
                response_str = self.tokenizer.decode(valid_response_ids)
                
                # Reconstruct with proper assistant marker
                if not response_str.startswith("<|im_start|>assistant"):
                    # Add the assistant marker if it's missing
                    sequences_str = prompt_str + "<|im_start|>assistant\n" + response_str
                else:
                    sequences_str = prompt_str + response_str

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]

            # Extract interaction_history from meta_info - CRITICAL for KG scoring
            batch_interaction_history = data.meta_info.get("interaction_history", [])
            
            # The new interaction_history is a list where each element corresponds to a sample.
            # We just need to pick the one for the current sample index `i`.
            sample_interaction_history = {}
            if batch_interaction_history and i < len(batch_interaction_history):
                sample_interaction_history = batch_interaction_history[i]
            else:
                print(f"Warning: interaction_history is missing or index {i} is out of range. Defaulting to empty dict.")
                sample_interaction_history = {}

            # select rm_score function based on data source
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            
            # Add data_source to interaction_history for logging purposes
            sample_interaction_history['data_source'] = data_source
            
            # Add logging for all KG evaluation examples
            # Only show first few examples to avoid log spam
            if i < 3:  # Show first 3 examples per batch
                dataset_type = 'UNKNOWN'
                if 'multitq' in data_source.lower():
                    dataset_type = 'MultiTQ'
                elif 'cwq' in data_source.lower():
                    dataset_type = 'CWQ'
                elif 'webqsp' in data_source.lower():
                    dataset_type = 'WebQSP'
                elif 'simpleqa' in data_source.lower():
                    dataset_type = 'SimpleQA'
                elif 'trex' in data_source.lower():
                    dataset_type = 'T-REx'
                elif 'zero_shot_re' in data_source.lower():
                    dataset_type = 'ZeroShotRE'
                
                print(f"[{dataset_type}-EVAL-KG] Data source: {data_source}")
                print(f"[{dataset_type}-EVAL-KG] Sample {i+1} being processed...")
            
            # Use custom compute_score function if provided, otherwise use KG format-specific scoring
            if self.compute_score is not None:
                raise NotImplementedError("Custom compute_score function is not supported for KG format reward manager")
            else:
                # Use the KG format-specific scoring function
                compute_score_fn = _select_kg_rm_score_fn(data_source)
                
                # Use refactored scoring function with cleaner parameter structure
                score = compute_score_fn(
                    solution_str=sequences_str, 
                    ground_truth=ground_truth,
                    interaction_history=sample_interaction_history,  # <-- CRITICAL FIX: Pass interaction history
                    valid_query_reward=self.valid_query_reward,
                    answer_match_score=self.answer_match_score,
                    structure_format_score=self.structure_format_score,
                    final_format_score=self.final_format_score,
                    retrieval_score=self.retrieval_score,
                    kg_server_error_penalty=self.kg_server_error_penalty,
                    kg_not_found_penalty=self.kg_not_found_penalty,
                    kg_format_error_penalty=self.kg_format_error_penalty,
                    kg_no_data_penalty=self.kg_no_data_penalty,
                    verbose=True  # Enable verbose logging for unique query tracking
                )

            if isinstance(score, dict):
                reward = score["score"]
                # Store only numeric/simple values in reward_extra_info to avoid type errors during metric processing
                for key, value in score.items():
                    if isinstance(value, (int, float, bool)) and key != "score":
                        reward_extra_info[key].append(value)
            else:
                reward = score

            reward_tensor[i, valid_response_length - 1] = reward

            # Debug logging for long responses
            if self.debug_long_responses and valid_response_length > self.response_length_threshold:
                # Convert tensor to int for logging
                response_length_int = int(valid_response_length.item()) if hasattr(valid_response_length, 'item') else int(valid_response_length)
                self._log_long_response_debug(
                    sequences_str=sequences_str,
                    sample_interaction_history=sample_interaction_history,
                    ground_truth=ground_truth,
                    score=score,
                    data_source=data_source,
                    response_length=response_length_int,
                    sample_idx=i,
                    step=self.debug_step_counter
                )
                print(f"[DEBUG] Step {self.debug_step_counter}, Sample {i}: Long response detected ({response_length_int} tokens) - logged to {self.debug_log_file}")

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print(f"[KG Format Reward Manager - {data_source}]")
                print("[sequences]", sequences_str)
                print("\n")
                print("sample_interaction_history:", sample_interaction_history)
                print("\n")
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor
