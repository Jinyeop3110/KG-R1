import torch
import re
from collections import defaultdict
import os
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from .tensor_helper import TensorHelper, TensorConfig
from verl import DataProto
from verl.utils.tracking import Tracking
import shutil
import requests
import json # Added for parsing JSON queries and formatting fallback
from functools import lru_cache
from kg_r1.search.error_types import KGErrorType

@dataclass
class GenerationConfig:
    max_turns: int
    max_start_length: int
    max_prompt_length: int 
    max_response_length: int
    max_obs_length: int
    num_gpus: int
    no_think_rl: bool=False
    search_url: str = None
    topk: int = 3 # Note: topk is not directly used by the KG server's current actions
    # NEW: Dataset-specific server URLs (optional, fallback to environment variables or defaults)
    simpleqa_server_url: str = None  # Default: FB2M server on port 9001
    cwq_server_url: str = None 
    webqsp_server_url: str = None
    metaqa_server_url: str = None  # Default: MetaQA server on port 9002

class LLMGenerationManager:
    def __init__(
        self,
        tokenizer,
        actor_rollout_wg,
        config: GenerationConfig,
        is_validation: bool = False,
    ):
        self.tokenizer = tokenizer
        
        # Handle tokenizer-specific configurations
        self._setup_tokenizer_compatibility()
        
        self.actor_rollout_wg = actor_rollout_wg
        self.config = config
        self.is_validation = is_validation
        
        # Track if we've already logged routing for each dataset
        self._routing_logged = set()

        self.tensor_fn = TensorHelper(TensorConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            max_prompt_length=config.max_prompt_length,
            max_obs_length=config.max_obs_length,
            max_start_length=config.max_start_length
        ))
        
        # Pre-compile regex patterns for optimization
        self._init_regex_patterns()
        
        # Initialize server routing configuration
        self._init_server_routing()

    def _setup_tokenizer_compatibility(self):
        """
        Setup tokenizer compatibility for different model families.
        Handles Llama2, Llama3, and Qwen2.5 tokenizers.
        """
        # Get tokenizer class name and model name if available
        tokenizer_class = self.tokenizer.__class__.__name__
        model_name_or_path = getattr(self.tokenizer, 'name_or_path', '')
        
        # Detect tokenizer type based on class name or model path
        is_llama = False
        is_qwen = False
        
        # Check for Llama tokenizers (Llama2 or Llama3)
        if 'llama' in tokenizer_class.lower() or 'llama' in model_name_or_path.lower():
            is_llama = True
            print(f"[TOKENIZER] Detected Llama tokenizer: {tokenizer_class} (model: {model_name_or_path})")
        # Check for Qwen tokenizers
        elif 'qwen' in tokenizer_class.lower() or 'qwen' in model_name_or_path.lower():
            is_qwen = True
            print(f"[TOKENIZER] Detected Qwen tokenizer: {tokenizer_class} (model: {model_name_or_path})")
        else:
            # Raise error for unsupported tokenizers
            raise ValueError(
                f"Unsupported tokenizer: {tokenizer_class} (model: {model_name_or_path}). "
                f"Only Llama2, Llama3, and Qwen2.5 tokenizers are supported."
            )
        
        # Handle Llama-specific setup
        if is_llama:
            # Llama tokenizers don't have a pad_token by default
            if self.tokenizer.pad_token is None:
                print("[TOKENIZER] Llama tokenizer has no pad_token, setting pad_token = eos_token")
                self.tokenizer.pad_token = self.tokenizer.eos_token
                self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            else:
                print(f"[TOKENIZER] Llama tokenizer already has pad_token: {repr(self.tokenizer.pad_token)}")
        
        # Qwen tokenizers already have proper pad_token setup, no changes needed
        elif is_qwen:
            print(f"[TOKENIZER] Qwen tokenizer pad_token: {repr(self.tokenizer.pad_token)} (id: {self.tokenizer.pad_token_id})")
            # No modifications needed for Qwen - preserve existing behavior
        
        # Verify pad_token is now set
        if self.tokenizer.pad_token_id is None:
            raise ValueError(f"Failed to set pad_token_id for {tokenizer_class}")

    def _init_server_routing(self):
        """Initialize server URL routing configuration for different datasets."""
        # Priority: Config > Environment Variables > Defaults
        self.server_routing = {
            "simpleqa": os.environ.get("SIMPLEQA_SERVER_URL", "http://127.0.0.1:9001/retrieve"),
            "cwq": os.environ.get("CWQ_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "webqsp": os.environ.get("WEBQSP_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "multitq": os.environ.get("MULTITQ_SERVER_URL", "http://127.0.0.1:8001/retrieve"),
            "metaqa": os.environ.get("METAQA_SERVER_URL", "http://127.0.0.1:9002/retrieve"),
            "grailqa": os.environ.get("GRAILQA_SERVER_URL", "http://127.0.0.1:9000/retrieve"),
            "trex": os.environ.get("TREX_SERVER_URL", "http://127.0.0.1:9011/retrieve"),
            "qald10en": os.environ.get("QALD10EN_SERVER_URL", "http://127.0.0.1:9010/retrieve"),
            "zero_shot_re": os.environ.get("ZERO_SHOT_RE_SERVER_URL", "http://127.0.0.1:9012/retrieve"),
        }
        
        # Fallback server URL (original behavior)
        self.fallback_server_url = self.config.search_url or "http://127.0.0.1:8001/retrieve"
        
        print(f"[KG_ROUTING] Server routing configuration:")
        for dataset, url in self.server_routing.items():
            print(f"  {dataset} -> {url}")
        print(f"  fallback -> {self.fallback_server_url}")

    def _get_server_url_for_dataset(self, dataset_name: str) -> str:
        """
        Get the appropriate server URL for a given dataset.
        
        Args:
            dataset_name: Name of the dataset (simpleqa, cwq, webqsp, etc.)
            
        Returns:
            Server URL for the dataset
        """
        if not dataset_name:
            return self.fallback_server_url
            
        # Normalize dataset name to lowercase
        dataset_key = dataset_name.lower().strip()
        
        # Return specific server URL or fallback
        server_url = self.server_routing.get(dataset_key, self.fallback_server_url)
        
        # Log routing decision for debugging (only once per dataset)
        if dataset_key not in self._routing_logged:
            self._routing_logged.add(dataset_key)
            if dataset_key in self.server_routing:
                print(f"[KG_ROUTING] {dataset_name} -> {server_url}")
            else:
                print(f"[KG_ROUTING] Unknown dataset '{dataset_name}', using fallback -> {server_url}")
            
        return server_url

    def _init_regex_patterns(self):
        """Pre-compile all regex patterns for performance optimization."""
        # Patterns for postprocess_predictions
        self.search_pattern = re.compile(r'<search>(.*?)</search>', re.DOTALL)
        self.kg_query_pattern1 = re.compile(r'<kg-query>(.*?)</kg-query>', re.DOTALL)  
        self.kg_query_pattern2 = re.compile(r'<kg-query\s+([^>]+)\s*/>', re.DOTALL)
        self.answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL)
        
        # Patterns for _parse_kg_query
        self.prefix_patterns = [
            re.compile(r'^kg-query\s+execute\s*["\']?', re.IGNORECASE),
            re.compile(r'^kg-query\s+', re.IGNORECASE), 
            re.compile(r'^function_name\s*\(\s*', re.IGNORECASE),
            re.compile(r'^query\s*[:\s]+', re.IGNORECASE),
        ]
        self.nested_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*\(\s*(get_[a-zA-Z_]+\([^)]+\))\s*\)$')
        # Strict patterns for single-entity functions (new: get_relations_in/out, legacy: get_head/tail_relations)
        self.get_relations_quoted_pattern = re.compile(r'^(get_relations|get_relations_in|get_relations_out|get_head_relations|get_tail_relations)\s*\(\s*"([^"]+)"\s*\)$')
        self.get_relations_unquoted_pattern = re.compile(r'^(get_relations|get_relations_in|get_relations_out|get_head_relations|get_tail_relations)\s*\(\s*([^,)]+)\s*\)$')
        
        # Original patterns for other functions that accept 2 arguments  
        self.quoted_function_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*"([^"]+)"\s*(?:,\s*"([^"]+)")?\s*\)$')
        self.function_pattern = re.compile(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\(\s*([^,)]+)(?:\s*,\s*([^)]+))?\s*\)$')

    def _batch_tokenize(self, responses: List[str]) -> torch.Tensor:
        """Tokenize a batch of responses."""
        if not responses:
            # Handle empty list case
            return torch.empty((0, 0), dtype=torch.long)
        
        return self.tokenizer(
            responses, 
            add_special_tokens=False, 
            return_tensors='pt', 
            padding="longest"
        )['input_ids']

    def _postprocess_responses(self, responses: torch.Tensor) -> Tuple[torch.Tensor, List[str]]:
        """Process responses to stop at search operation or answer operation."""
        responses_str = self.tokenizer.batch_decode(
            responses, 
            skip_special_tokens=True
        )


        def clean_response(resp: str) -> str:
            # First, remove any conversation markers if present
            if '<|im_start|>assistant' in resp:
                assistant_idx = resp.find('<|im_start|>assistant')
                if assistant_idx >= 0:
                    after_assistant = resp[assistant_idx:].find('\n')
                    if after_assistant >= 0:
                        resp = resp[assistant_idx + after_assistant + 1:]
                    else:
                        resp = resp[assistant_idx + len('<|im_start|>assistant'):]
            
            # --- MODIFIED LOGIC START ---

            # ORIGINAL LOGIC - Find first occurrence of closing tags
            try:
                answer_idx = resp.index('</answer>')
            except ValueError:
                answer_idx = -1

            try:
                kg_query_idx = resp.index('</kg-query>')
            except ValueError:
                kg_query_idx = -1

            # Determine which tag comes first, if any
            if answer_idx != -1 and (answer_idx < kg_query_idx or kg_query_idx == -1):
                # '</answer>' is present and comes before '</kg-query>' or '</kg-query>' is not present
                return resp[:answer_idx + len('</answer>')]
            elif kg_query_idx != -1:
                # '</kg-query>' is present and comes before '</answer>' or '</answer>' is not present
                return resp[:kg_query_idx + len('</kg-query>')]
            else:
                # No complete tags found, return as-is
                return resp
        
        responses_str = [clean_response(resp) for resp in responses_str]

        if self.config.no_think_rl:
            raise ValueError('stop')
            # if no_think_rl is enabled, only keep action in the str
            actions, _ = self.env.postprocess_predictions(responses_str)
            responses_str=[f"<answer>{envs[idx].ACTION_LOOKUP[action]}</answer>" for idx, action in enumerate(actions)]
        
        responses = self._batch_tokenize(responses_str)
        return responses, responses_str

    def _process_next_obs(self, next_obs: List[str]) -> torch.Tensor:
        """Process next observations from environment."""
        
        next_obs_ids = self.tokenizer(
            next_obs, 
            padding='longest',
            return_tensors='pt',
            add_special_tokens=False,  # Prevents adding special tokens
        )['input_ids']

        if next_obs_ids.shape[1] > self.config.max_obs_length:
            next_obs_ids = next_obs_ids[:, :self.config.max_obs_length]

        # --- Added safeguard to preserve closing </information> tag after truncation ---
        # Decode each truncated observation string and ensure that any <information> block
        # still ends with a proper closing tag. If the tag is missing (likely due to
        # truncation) we append an ellipsis and the closing tag to avoid malformed
        # markup that confuses the LLM.
        obs_fixed = []
        needs_fix = False
        for obs_ids in next_obs_ids:
            text = self.tokenizer.decode(obs_ids, skip_special_tokens=True)
            if "<information>" in text and "</information>" not in text:
                needs_fix = True
                obs_fixed.append(text.rstrip() + " …</information>")
            else:
                obs_fixed.append(text)
        if needs_fix:
            # Re-tokenize the fixed observations so shapes stay consistent
            next_obs_ids = self.tokenizer(
                obs_fixed,
                padding='longest',
                return_tensors='pt',
                add_special_tokens=False,
            )['input_ids']
        # --- End safeguard ---

        return next_obs_ids

    def _update_rolling_state(self, rollings: DataProto, cur_responses: torch.Tensor, 
                            next_obs_ids: torch.Tensor) -> Dict:
        """Update rolling state with new responses and observations."""
        # Ensure all tensors are on the same device before concatenation
        device = rollings.batch['input_ids'].device # Assume initial device is the target
        
        current_input_ids = rollings.batch['input_ids'].to(device)
        current_responses = cur_responses.to(device)
        current_next_obs_ids = next_obs_ids.to(device)

        # Concatenate and handle padding        
        new_input_ids = self.tensor_fn.concatenate_with_padding([
            current_input_ids,
            current_responses,
            current_next_obs_ids
        ])
        
        # Create attention mask and position ids
        new_attention_mask = self.tensor_fn.create_attention_mask(new_input_ids)
        new_position_ids = self.tensor_fn.create_position_ids(new_attention_mask)

        # Cut to appropriate length
        effective_len = new_attention_mask.sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)

        new_rollings = DataProto.from_dict({
            'input_ids': new_input_ids[:, -max_len:],
            'position_ids': new_position_ids[:, -max_len:],
            'attention_mask': new_attention_mask[:, -max_len:]
        })
        new_rollings.meta_info.update(rollings.meta_info)
        
        return new_rollings

    def _info_masked_concatenate_with_padding(self, 
                prompt: torch.Tensor, 
                prompt_with_mask: torch.Tensor, 
                response: torch.Tensor, 
                info: torch.Tensor = None,
                pad_to_left: bool = True
            ) -> torch.Tensor:
        """Concatenate tensors and handle padding. Additionally, create a mask (info_mask) to cover the information block if it exists."""
        pad_id = self.tokenizer.pad_token_id
        tensors = [prompt, response]
        tensors_with_mask = [prompt_with_mask, response]
        
        if info is not None:
            tensors.append(info)
            info_mask = torch.full(info.size(), pad_id, dtype=info.dtype, device=info.device) # information mask
            tensors_with_mask.append(info_mask)
        
        concatenated = torch.cat(tensors, dim=1)
        concatenated_with_info = torch.cat(tensors_with_mask, dim=1)
        
        mask = concatenated != pad_id if pad_to_left else concatenated == pad_id
        sorted_indices = mask.to(torch.int64).argsort(dim=1, stable=True)
        padded_tensor = concatenated.gather(1, sorted_indices)
        padded_tensor_with_info = concatenated_with_info.gather(1, sorted_indices)

        return padded_tensor, padded_tensor_with_info, sorted_indices

    def _update_right_side(self, right_side: Dict, 
                          cur_responses: torch.Tensor,
                          next_obs_ids: torch.Tensor = None,
                          turn_idx: int = None,
                          active_mask: torch.Tensor = None) -> Dict:
        """Update right side state."""
        device = right_side['responses'].device # Assume initial device is the target
        
        # Debug storage removed - no longer needed after successful fix
        
        current_right_side_responses = right_side['responses'].to(device)
        current_right_side_responses_with_info_mask = right_side['responses_with_info_mask'].to(device)
        current_right_side_turn_tokens = right_side['turn_tokens'].to(device)
        current_cur_responses = cur_responses.to(device)

        if next_obs_ids is not None:
            current_next_obs_ids = next_obs_ids.to(device)
            responses, responses_with_info_mask, sorted_indices = self._info_masked_concatenate_with_padding(
                    current_right_side_responses,
                    current_right_side_responses_with_info_mask,
                    current_cur_responses,
                    current_next_obs_ids, 
                    pad_to_left=False
                )
        else:
            responses, responses_with_info_mask, sorted_indices = self._info_masked_concatenate_with_padding(
                    current_right_side_responses,
                    current_right_side_responses_with_info_mask,
                    current_cur_responses,
                    pad_to_left=False
                )
        # Handle turn token generation
        if turn_idx is not None:
            # Start with -1 for all tokens (no advantage)
            action_turn_tokens = torch.full_like(current_cur_responses, -1, dtype=torch.long, device=device)
            
            # Only assign turn numbers to non-pad tokens within active examples
            if active_mask is not None:
                # Expand active_mask to match response dimensions
                if active_mask.dim() == 1:
                    expanded_active_mask = active_mask.unsqueeze(1).expand_as(current_cur_responses)
                else:
                    expanded_active_mask = active_mask
                
                # Combine active mask with non-pad mask: only assign to real response tokens
                valid_response_mask = expanded_active_mask & (current_cur_responses != self.tokenizer.pad_token_id)
                action_turn_tokens[valid_response_mask] = turn_idx
            else:
                # Fallback: assign turn numbers to all response tokens (old behavior)
                action_turn_tokens.fill_(turn_idx)
            
            if next_obs_ids is not None:
                # Info tokens always get -1 (no advantage)
                info_turn_tokens = torch.full_like(current_next_obs_ids, -1, dtype=torch.long, device=device)
                # Concatenate turn tokens: [action_tokens, info_tokens]
                current_turn_tokens = torch.cat([action_turn_tokens, info_turn_tokens], dim=1)
            else:
                current_turn_tokens = action_turn_tokens
                
            # Update turn token sequence
            turn_tokens = torch.cat([current_right_side_turn_tokens, current_turn_tokens], dim=1)
            
            # CRITICAL FIX: Apply the same reordering to turn tokens as was applied to the actual tokens
            # This ensures turn IDs stay aligned with their corresponding tokens after padding reordering
            turn_tokens = turn_tokens.gather(1, sorted_indices)
        else:
            # Fallback: use existing turn tokens
            turn_tokens = current_right_side_turn_tokens
        
        effective_len = self.tensor_fn.create_attention_mask(responses).sum(dim=1).max()
        max_len = min(self.config.max_prompt_length, effective_len)
        
        return {
            'responses': responses[:, :max_len], 
            'responses_with_info_mask': responses_with_info_mask[:, :max_len],
            'turn_tokens': turn_tokens[:, :max_len]
        }

    def _generate_with_gpu_padding(self, active_batch: DataProto) -> DataProto:
        """
            Wrapper for generation that handles multi-GPU padding requirements.
            if num_gpus <= 1, return self.actor_rollout_wg.generate_sequences(active_batch)
            if active_batch size is not divisible by num_gpus, pad with first sequence
            then remove padding from output
        """
        num_gpus = self.config.num_gpus
        if num_gpus <= 1:
            return self.actor_rollout_wg.generate_sequences(active_batch)
            
        batch_size = active_batch.batch['input_ids'].shape[0]
        remainder = batch_size % num_gpus
        
        for key in active_batch.batch.keys():
            active_batch.batch[key] = active_batch.batch[key].long()
        if remainder == 0:
            return self.actor_rollout_wg.generate_sequences(active_batch)
        
        # Add padding sequences
        padding_size = num_gpus - remainder
        padded_batch = {}
        
        for k, v in active_batch.batch.items():
            # Use first sequence as padding template
            pad_sequence = v[0:1].repeat(padding_size, *[1] * (len(v.shape) - 1))
            padded_batch[k] = torch.cat([v, pad_sequence], dim=0)

        padded_active_batch = DataProto.from_dict(padded_batch)
        for key in padded_active_batch.batch.keys():
            padded_active_batch.batch[key] = padded_active_batch.batch[key].long()

        # Generate with padded batch
        padded_output = self.actor_rollout_wg.generate_sequences(padded_active_batch)

        # Remove padding from output
        trimmed_batch = {k: v[:-padding_size] if padding_size > 0 else v for k, v in padded_output.batch.items()}
        
        # Handle meta_info if present
        if hasattr(padded_output, 'meta_info') and padded_output.meta_info:
            trimmed_meta = {}
            for k, v in padded_output.meta_info.items():
                if isinstance(v, torch.Tensor):
                    trimmed_meta[k] = v[:-padding_size] if padding_size > 0 else v
                else:
                    trimmed_meta[k] = v
            padded_output.meta_info = trimmed_meta
            
        padded_output.batch = trimmed_batch
        return padded_output

    def run_llm_loop(self, gen_batch, initial_input_ids: torch.Tensor) -> Tuple[Dict, Dict]:
        """Run main LLM generation loop."""
        
        # CRITICAL FIX: Use the actual expanded batch size, not the original
        # When using rollouts (e.g., grpo_rollout_n=8), the batch is already expanded
        actual_batch_size = gen_batch.batch['input_ids'].shape[0]
        
        original_left_side = {'input_ids': initial_input_ids[:, -self.config.max_start_length:]}
        original_right_side = {
            'responses': initial_input_ids[:, []], 
            'responses_with_info_mask': initial_input_ids[:, []],
            'turn_tokens': initial_input_ids[:, []]
        }
        
        active_mask = torch.ones(actual_batch_size, dtype=torch.bool)
        turns_stats = torch.ones(actual_batch_size, dtype=torch.int)
        valid_action_stats = torch.zeros(actual_batch_size, dtype=torch.int)
        valid_search_stats = torch.zeros(actual_batch_size, dtype=torch.int)
        active_num_list = [active_mask.sum().item()]
        rollings = gen_batch
        
        # Store detailed interaction information for reward calculation
        # CRITICAL FIX: Use actual_batch_size to match the expanded batch
        # Extract dataset information from gen_batch to include in interaction_history
        dataset_names = None
        if hasattr(gen_batch, 'meta_info') and 'dataset_names' in gen_batch.meta_info:
            dataset_names = gen_batch.meta_info['dataset_names']
        elif hasattr(gen_batch, 'non_tensor_batch') and 'data_source' in gen_batch.non_tensor_batch:
            # Fallback: extract from non_tensor_batch
            dataset_names = gen_batch.non_tensor_batch['data_source']
            if hasattr(dataset_names, 'tolist'):
                dataset_names = dataset_names.tolist()
        
        interaction_history = []
        for i in range(actual_batch_size):
            # Include data_source in each sample's interaction history for MultiTQ detection
            sample_data_source = ''
            if dataset_names and i < len(dataset_names):
                sample_data_source = str(dataset_names[i])
            
            interaction_history.append({
                "data_source": sample_data_source,  # ADD: Include dataset info for reward calculation
                "actions": [],
                "search_results": [],
                "valid_actions": [],
                "is_search_actions": [],
                "raw_server_responses": [],
                "responses_str": [],
            })
        
        print(f"[DEBUG-INTERACTION] Created interaction_history with data_source info: {[h['data_source'] for h in interaction_history[:3]]}...")

        # Main generation loop
        for step in range(self.config.max_turns):
            if not active_mask.sum():
                break
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )
            
            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)
            
            # Execute in environment and process observations
            # Note: The batch is already expanded on the trainer side for multiple rollouts
            next_obs, dones, valid_action, is_search, raw_server_responses = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, gen_batch.meta_info, active_mask
            )
            
            # Store interaction details for this turn
            cur_actions, _ = self.postprocess_predictions(responses_str)
            for i in range(actual_batch_size):
                if active_mask[i]:  # Active samples: record actual data
                    interaction_history[i]["actions"].append(cur_actions[i])
                    interaction_history[i]["search_results"].append(next_obs[i])
                    interaction_history[i]["valid_actions"].append(valid_action[i])
                    interaction_history[i]["is_search_actions"].append(is_search[i])
                    interaction_history[i]["responses_str"].append(responses_str[i])
                else:  # Inactive samples: record placeholder data for consistency
                    interaction_history[i]["actions"].append("")  # No action
                    interaction_history[i]["search_results"].append("")  # No search result
                    interaction_history[i]["valid_actions"].append(0)  # Invalid action
                    interaction_history[i]["is_search_actions"].append(0)  # Not search
                    interaction_history[i]["responses_str"].append("")  # No response
                
                # ALL samples get raw_server_responses recorded for batch consistency
                interaction_history[i]["raw_server_responses"].append(raw_server_responses[i])
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            
            # With expanded batch from trainer side, active_mask should match the expanded size
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            
            turns_stats[curr_active_mask] += 1
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)

            next_obs_ids = self._process_next_obs(next_obs)
            
            # Update states
            rollings = self._update_rolling_state(
                rollings,
                responses_ids,
                next_obs_ids
            )
            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                next_obs_ids,
                turn_idx=step + 1,
                active_mask=active_mask
            )
            
        # final LLM rollout
        if active_mask.sum():
            rollings.batch = self.tensor_fn.cut_to_effective_len(
                rollings.batch,
                keys=['input_ids', 'attention_mask', 'position_ids']
            )

            # gen_output = self.actor_rollout_wg.generate_sequences(rollings)
            rollings_active = DataProto.from_dict({
                k: v[active_mask] for k, v in rollings.batch.items()
            })
            
            gen_output = self._generate_with_gpu_padding(rollings_active)

            meta_info = gen_output.meta_info            
            responses_ids, responses_str = self._postprocess_responses(gen_output.batch['responses'])
            responses_ids, responses_str = self.tensor_fn._example_level_pad(responses_ids, responses_str, active_mask)

            # Execute in environment and process observations
            _, dones, valid_action, is_search, raw_server_responses = self.execute_predictions(
                responses_str, self.tokenizer.pad_token, gen_batch.meta_info, active_mask, do_search=False
            )

            # Store final turn interaction details
            cur_actions, _ = self.postprocess_predictions(responses_str)
            query_response_idx = 0
            for i in range(actual_batch_size):
                if active_mask[i]:  # Active samples: record actual final turn data
                    action = cur_actions[i]
                    interaction_history[i]["actions"].append(action)
                    interaction_history[i]["search_results"].append('')  # No search in final turn
                    interaction_history[i]["valid_actions"].append(valid_action[i])
                    interaction_history[i]["is_search_actions"].append(is_search[i])
                    interaction_history[i]["responses_str"].append(responses_str[i])
                else:  # Inactive samples: record placeholder data for consistency
                    interaction_history[i]["actions"].append("")  # No action
                    interaction_history[i]["search_results"].append('')  # No search in final turn
                    interaction_history[i]["valid_actions"].append(0)  # Invalid action
                    interaction_history[i]["is_search_actions"].append(0)  # Not search
                    interaction_history[i]["responses_str"].append("")  # No response

                # Final turn: Use actual raw_server_responses from execute_predictions
                # This ensures KG queries in final turn get proper error responses instead of empty dicts
                if i < len(raw_server_responses):
                    interaction_history[i]["raw_server_responses"].append(raw_server_responses[i])
                else:
                    # Fallback for cases where raw_server_responses is shorter than expected
                    interaction_history[i]["raw_server_responses"].append({})

            # Store original active_mask before updating with done status
            final_turn_active_mask = active_mask.clone()
            
            curr_active_mask = torch.tensor([not done for done in dones], dtype=torch.bool)
            active_mask = active_mask * curr_active_mask
            active_num_list.append(active_mask.sum().item())
            valid_action_stats += torch.tensor(valid_action, dtype=torch.int)
            valid_search_stats += torch.tensor(is_search, dtype=torch.int)
            

            original_right_side = self._update_right_side(
                original_right_side,
                responses_ids,
                turn_idx=self.config.max_turns + 1,
                active_mask=final_turn_active_mask  # Use original active_mask for final turn
            )
        
        meta_info['turns_stats'] = turns_stats.tolist()
        meta_info['active_mask'] = active_mask.tolist()
        meta_info['valid_action_stats'] = valid_action_stats.tolist()
        meta_info['valid_search_stats'] = valid_search_stats.tolist()
        
        # Add detailed interaction information for reward calculation
        meta_info['interaction_history'] = interaction_history
        
        # Ensure the final output maintains the original batch size
        # This is crucial for GRPO which expands the batch size
        assert original_left_side['input_ids'].shape[0] == actual_batch_size, \
            f"Left side batch size mismatch: {original_left_side['input_ids'].shape[0]} != {actual_batch_size}"
        assert original_right_side['responses'].shape[0] == actual_batch_size, \
            f"Right side batch size mismatch: {original_right_side['responses'].shape[0]} != {actual_batch_size}"
        
        final_output = self._compose_final_output(original_left_side, original_right_side, meta_info)
        
        # Final check to ensure the output has the correct batch size
        assert final_output.batch['responses'].shape[0] == actual_batch_size, \
            f"Final output batch size mismatch: {final_output.batch['responses'].shape[0]} != {actual_batch_size}"
        
        return final_output

    def _compose_final_output(self, left_side: Dict,
                            right_side: Dict,
                            meta_info: Dict) -> Tuple[Dict, Dict]:
        """Compose final generation output."""
        final_output = right_side.copy()
        final_output['prompts'] = left_side['input_ids']
        
        # Combine input IDs
        # Ensure all tensors are on the same device before concatenation
        device = left_side['input_ids'].device
        
        final_output['input_ids'] = torch.cat([
            left_side['input_ids'].to(device).long(),
            right_side['responses'].to(device).long()
        ], dim=1)
        
        # Ensure responses tensors are long type for tokenizer compatibility
        final_output['responses'] = final_output['responses'].to(device).long()
        final_output['responses_with_info_mask'] = final_output['responses_with_info_mask'].to(device).long()
        
        # Create attention mask and position ids
        final_output['attention_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'].to(device)),
            self.tensor_fn.create_attention_mask(final_output['responses'])
        ], dim=1)
        final_output['info_mask'] = torch.cat([
            self.tensor_fn.create_attention_mask(left_side['input_ids'].to(device)),
            self.tensor_fn.create_attention_mask(final_output['responses_with_info_mask'])
        ], dim=1)
        
        final_output['position_ids'] = self.tensor_fn.create_position_ids(
            final_output['attention_mask']
        )
        
        # Generate turn sequence tensor for multi-turn advantage calculation
        # Create turn tokens for prompt (left_side) - all get -1 (no advantage)
        device = left_side['input_ids'].device
        prompt_turn_tokens = torch.full_like(left_side['input_ids'], -1, dtype=torch.long, device=device)
        
        # Combine prompt and response turn tokens
        final_output['turn_sequence_tensor'] = torch.cat([
            prompt_turn_tokens,
            right_side['turn_tokens']
        ], dim=1)
        
        # DEBUG: LLM token validation (can be enabled for debugging)
        # Note: Successfully fixed turn token alignment - 0 errors achieved!
        # Uncomment below for debugging if needed in the future
        
        # turn_tokens = final_output['turn_sequence_tensor']
        # info_mask = final_output['info_mask']
        # 
        # if turn_tokens.size(0) > 0:
        #     sample_turn = turn_tokens[0]
        #     sample_info = info_mask[0]
        #     llm_generated_positions = (sample_turn > 0)
        #     valid_token_positions = (sample_info == 1)
        #     llm_tokens_with_invalid_mask = llm_generated_positions & ~valid_token_positions
        #     
        #     if llm_tokens_with_invalid_mask.sum() > 0:
        #         print(f"[DEBUG] LLM Token Validation Issues: {llm_tokens_with_invalid_mask.sum().item()} errors found")
        #     else:
        #         print(f"[DEBUG] LLM Token Validation: Perfect alignment achieved!")
        
        # Store the full concatenated sequence for debugging/logging if needed
        # This is what the test script expects as 'full_interaction_ids'
        # It should be a list of tensors, one per batch item.
        # In this case, batch size is implicitly handled by the structure of left_side and right_side.
        # final_output['input_ids'] here is the full sequence for all items in the batch.
        # The test script expects a list containing this tensor.
        meta_info['full_interaction_ids'] = [final_output['input_ids'].clone()] 

        # Ensure meta_info from gen_batch is preserved if it was updated
        # The meta_info passed here is from the last gen_output, might not be the original one.
        # However, sample_id and dataset_name should be stable.
        # For safety, if we need the original gen_batch.meta_info, it should be explicitly carried.
        # For now, assume meta_info contains what's needed or what was last updated.

        final_output_proto = DataProto.from_dict(final_output)
        final_output_proto.meta_info.update(meta_info)
        
        return final_output_proto # Return DataProto object

    def execute_predictions(self, predictions: List[str], pad_token: str, meta_info: Dict, active_mask=None, do_search=True) -> Tuple[List[str], List[bool], List[int], List[int], List[Dict]]:
        """
        Execute predictions across multiple environments.
        NOTE: the function is the actual `step` function in the environment
        NOTE penalty_for_invalid is not included in observation shown to the LLM
        
        Args:
            envs: List of environment instances
            predictions: List of action predictions
            pad_token: Token to use for padding
            meta_info: Dict containing batch-level metadata OR per-sample metadata
            
        Returns:
            Tuple of (next_obs, dones, valid_action, is_search, raw_server_responses)
        """
        cur_actions, contents = self.postprocess_predictions(predictions)
        
        # Check for responses that exceed max_prompt_length and mark them as invalid
        response_length_exceeded = []
        for i, prediction in enumerate(predictions):
            # Tokenize the response to check its length
            response_tokens = self.tokenizer(
                prediction, 
                add_special_tokens=False, 
                return_tensors='pt'
            )['input_ids']
            response_length = response_tokens.shape[1]
            
            if response_length > self.config.max_response_length:
                response_length_exceeded.append(True)
                # Override action to be invalid if response is too long
                cur_actions[i] = 'error'
                contents[i] = f'Response too long ({response_length} > {self.config.max_response_length} tokens)'
            else:
                response_length_exceeded.append(False)
        
        next_obs, dones, valid_action, is_search = [], [], [], []
        raw_server_responses = []
        
        # Handle kg-query actions (send to KG server)
        # CRITICAL FIX: Track which sample index each kg-query belongs to
        kg_queries_with_indices = [(i, content) for i, (action, content) in enumerate(zip(cur_actions, contents)) if action == 'kg-query']
        kg_queries_contents = [content for _, content in kg_queries_with_indices]
        kg_query_indices = [i for i, _ in kg_queries_with_indices]
        
        if do_search and kg_queries_contents:
            # Create per-sample meta_info for KG queries
            # Pass the batch indices of queries to properly extract meta_info
            kg_query_meta_infos = self._extract_kg_meta_info_for_queries(cur_actions, meta_info, active_mask, kg_query_indices)
            kg_results, raw_kg_responses = self.batch_search(kg_queries_contents, kg_query_meta_infos) # Pass per-sample meta_info
            
            # VALIDATION: Ensure we got the right number of responses
            if len(kg_results) != len(kg_queries_contents):
                raise ValueError(
                    f"KG response count mismatch: got {len(kg_results)} results for "
                    f"{len(kg_queries_contents)} queries"
                )
            if len(raw_kg_responses) != len(kg_queries_contents):
                raise ValueError(
                    f"Raw KG response count mismatch: got {len(raw_kg_responses)} responses for "
                    f"{len(kg_queries_contents)} queries"
                )
        else:
            kg_results = [''] * len(kg_queries_contents)
            raw_kg_responses = [{} for _ in kg_queries_contents] # Return empty dicts for no search
        
        # Create a mapping from sample index to kg response
        kg_response_map = {}
        for idx, (sample_idx, _) in enumerate(kg_queries_with_indices):
            kg_response_map[sample_idx] = (kg_results[idx], raw_kg_responses[idx])
        
        # Handle search actions (return "not implemented" message)
        search_queries_count = sum([1 for action in cur_actions if action == 'search'])
        search_results = ["The search server has not been implemented. Please use <kg-query> with function calls instead."] * search_queries_count
        for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
            
            if not active:
                next_obs.append('')
                dones.append(1)
                valid_action.append(0)
                is_search.append(0)
                raw_server_responses.append({})
            else:
                if action == 'answer':
                    next_obs.append('')
                    dones.append(1)
                    valid_action.append(1)
                    is_search.append(0)
                    raw_server_responses.append({})
                elif action == 'kg-query':
                    # CRITICAL FIX: Use the mapping to get the correct response for this sample
                    if i in kg_response_map:
                        kg_result, raw_kg_response = kg_response_map[i]
                        next_obs.append(f'\n\n<information>{kg_result.strip()}</information>\n\n')
                        raw_server_responses.append(raw_kg_response)
                    else:
                        # Fallback if something went wrong
                        next_obs.append(f'\n\n<information>Error: No response found for this query</information>\n\n')
                        raw_server_responses.append({"error": "No response mapped"})
                    dones.append(0)
                    # Fix: KG queries are invalid if search is disabled (final turn)
                    if do_search:
                        valid_action.append(1)  # Valid when search is enabled
                    else:
                        valid_action.append(0)  # Invalid in final turn when search is disabled
                    is_search.append(1)
                elif action == 'search':
                    next_obs.append(f'\n\n<information>{search_results.pop(0).strip()}</information>\n\n')
                    dones.append(0)
                    valid_action.append(1)
                    is_search.append(1)
                    raw_server_responses.append({"success": False, "action": "search", "kg_metadata": {"success": False, "error_type": KGErrorType.FORMAT_ERROR}})  # Mock response for search
                else:
                    # Check if this is a length-exceeded response
                    if response_length_exceeded[i]:
                        next_obs.append(f'\n\n<information>Your previous response was too long ({contents[i]}). Please provide shorter responses within the token limit.</information>\n\n')
                    else:
                        next_obs.append(f'\n\n<information>Your previous action is invalid. You should put the query between <kg-query> and </kg-query> if you want to search, or put the answer between <answer> and </answer> if you want to give the final answer.</information>\n\n')
                    dones.append(0)
                    valid_action.append(0)
                    is_search.append(0)
                    raw_server_responses.append({"success": False, "action": "invalid", "kg_metadata": {"success": False, "error_type": KGErrorType.FORMAT_ERROR}})
            
        # CRITICAL FIX: Remove assertion since we're using mapping now
        assert len(search_results) == 0
            
        return next_obs, dones, valid_action, is_search, raw_server_responses

    def postprocess_predictions(self, predictions: List[Any]) -> Tuple[List[str], List[str]]: # Changed return type for contents
        """
        Process (text-based) predictions from llm into actions and their contents.
        
        Args:
            predictions: List of raw predictions
            
        Returns:
            Tuple of (actions list, contents list) 
            Content for search is the string "action_type, entity_id [, relation_name]"
            Content for answer is the answer string.
        """
        actions = []
        contents = []
                
        for prediction in predictions:
            if isinstance(prediction, str): # for llm output
                # Use simpler, more robust pattern matching like the search version
                kg_pattern = r'<(kg-query)>(.*?)</\1>'
                search_pattern = r'<(search)>(.*?)</\1>'
                answer_pattern = r'<(answer)>(.*?)</\1>'
                
                # Try patterns in order - answer first, then kg-query, then search
                match = re.search(answer_pattern, prediction, re.DOTALL)
                if match:
                    content = match.group(2).strip()
                    action = 'answer'
                else:
                    match = re.search(kg_pattern, prediction, re.DOTALL)
                    if match:
                        content = match.group(2).strip()
                        action = 'kg-query'
                    else:
                        match = re.search(search_pattern, prediction, re.DOTALL)
                        if match:
                            content = match.group(2).strip()
                            action = 'search'
                        else:
                            content = ''
                            action = 'error'
            else:
                raise ValueError(f"Invalid prediction type: {type(prediction)}")
            
            actions.append(action)
            contents.append(content) # content is now the string to be parsed by _batch_search
            
        return actions, contents

    def batch_search(self, search_query_contents: List[str] = None, meta_info_list = None) -> Tuple[List[str], List[Dict[str, Any]]]: # Changed to return both
        """
        Batchified search for queries using the KG retrieval server.
        Args:
            search_query_contents: List of strings, each being function call queries or error messages.
            meta_info_list: List of meta_info dicts (one per query) OR single dict for backward compatibility
        Returns:
            A tuple of (formatted_string_results, raw_server_responses) from the KG server.
        """
        if not search_query_contents:
            return [], []

        # Handle error messages directly (like deprecated <search> tag errors)
        results = []
        valid_queries = []
        valid_meta_infos = []
        error_indices = []
        
        for i, content in enumerate(search_query_contents):
            if content.startswith("ERROR:"):
                # This is an error message, return it directly
                results.append(content)
                error_indices.append(i)
            else:
                # This is a valid query to process
                valid_queries.append(content)
                # Get corresponding meta_info
                if isinstance(meta_info_list, list) and i < len(meta_info_list):
                    valid_meta_infos.append(meta_info_list[i])
                elif isinstance(meta_info_list, dict):
                    # Backward compatibility: single meta_info for all queries
                    valid_meta_infos.append(meta_info_list)
                else:
                    # Fallback
                    valid_meta_infos.append({"sample_id": "unknown", "dataset_name": "unknown"})
        
        raw_server_responses = []
        if valid_queries:
            kg_server_responses, kg_results = self._batch_search(valid_queries, valid_meta_infos) # Pass list of meta_infos
            
            # Merge error messages and valid results in correct order
            final_results = []
            final_raw_responses = []
            valid_idx = 0
            for i in range(len(search_query_contents)):
                if i in error_indices:
                    final_results.append(results[error_indices.index(i)])
                    # Create a mock error response for consistency
                    final_raw_responses.append({
                        "success": False,
                        "choices": [{"message": {"content": results[error_indices.index(i)]}}]
                    })
                else:
                    final_results.append(kg_results[valid_idx])
                    final_raw_responses.append(kg_server_responses[valid_idx])
                    valid_idx += 1
            return final_results, final_raw_responses
        else:
            # Create mock error responses for all error messages
            error_responses = []
            for result in results:
                error_responses.append({
                    "success": False,
                    "choices": [{"message": {"content": result}}]
                })
            return results, error_responses

    def _batch_search(self, search_query_contents: List[str], meta_info_list) -> Tuple[List[Dict[str, Any]], List[str]]:
        """
        Sends a batch of requests to the KG retrieval server.
        Args:
            search_query_contents: List of strings, each being "action_type, entity_id [, relation_name]".
            meta_info_list: List of meta_info dicts, one per query.
        Returns:
            A tuple of (raw_server_responses, formatted_string_responses) from the KG server.
        """
        parsed_requests = []

        for i, query_content_str in enumerate(search_query_contents):
            
            # Get meta_info for this specific query
            if isinstance(meta_info_list, list) and i < len(meta_info_list):
                meta_info = meta_info_list[i]
            elif isinstance(meta_info_list, dict):
                # Backward compatibility
                meta_info = meta_info_list
            else:
                meta_info = {"sample_id": "unknown", "dataset_name": "webqsp"}
            
            # Retrieve sample_id and dataset_name from this query's meta_info
            sample_id = meta_info.get("sample_id", "unknown_sample_id")
            dataset_name = meta_info.get("dataset_name", "webqsp")  # Use webqsp as safe fallback
            
            # If we still have unknown_sample_id, generate a better fallback
            if sample_id == "unknown_sample_id":
                sample_id = f"generated_sample_{i:06d}"
            
            try:
                parsed = self._parse_kg_query(query_content_str)
                action_type = parsed["action_type"]
                entity_id = parsed["entity_id"]
                relation_name = parsed["relation_name"]
                
                # Client-side validation: Check if action_type is valid
                valid_action_types = [
                    "get_relations_in", "get_relations_out", "get_entities_in", "get_entities_out",  # New names
                    "get_relations", "get_head_relations", "get_tail_relations", "get_head_entities", "get_tail_entities"  # Legacy
                ]
                if action_type not in valid_action_types:
                    # Create descriptive error message for invalid action type
                    error_msg = (
                        f"Invalid action type '{action_type}'. "
                        f"Use one of: get_relations_in, get_relations_out, get_entities_in, get_entities_out. "
                        f"Example: get_relations_in(\"entity\") or get_entities_out(\"entity\", \"relation\")"
                    )
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue

                # Client-side validation: Check if entity_id is not empty
                if not entity_id or not entity_id.strip():
                    error_msg = "Entity name cannot be empty. Provide a valid entity name in quotes."
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue

                # Client-side validation: Check relation requirements
                if action_type in ["get_entities_in", "get_entities_out", "get_head_entities", "get_tail_entities"] and (not relation_name or not relation_name.strip()):
                    error_msg = f"Relation name is required for {action_type}. Use format: {action_type}(\"entity\", \"relation\")"
                    parsed_requests.append({
                        "_is_client_error": True,
                        "error_message": error_msg,
                    })
                    continue
                
                request_payload = {
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "action_type": action_type,
                    "entity_id": entity_id.strip(),  # Clean up entity_id
                }
                # Add batch_index for debugging if available
                if "batch_index" in meta_info:
                    request_payload["_debug_batch_idx"] = meta_info["batch_index"]
                # Only add relation to payload if it's not None and not an empty string.
                # The KG server expects the key to be absent if no relation is applicable or provided.
                # If relation_name is an empty string here, it means the LLM provided 3 parts but the 3rd was empty.
                # For actions like get_tail_entities, an empty relation is invalid, and the server should handle that.
                if relation_name and relation_name.strip(): # This will be true if relation_name is a non-empty string
                    request_payload["relation"] = relation_name.strip()
                
                parsed_requests.append(request_payload)

            except Exception as e:
                
                # Create LLM-friendly error message
                error_msg = self._create_query_format_error(query_content_str, str(e))
                
                parsed_requests.append({
                    "_is_client_error": True, 
                    "error_message": error_msg,
                })

        if not parsed_requests:
            return [], []

        # Filter out client-side errors before sending to server
        valid_requests = [req for req in parsed_requests if "_is_client_error" not in req]
        
        server_responses = []
        if valid_requests:
            # NEW: Group requests by dataset for dynamic routing
            requests_by_server = defaultdict(list)
            request_to_server_mapping = {}
            
            for idx, req in enumerate(valid_requests):
                dataset_name = req.get("dataset_name", "")
                server_url = self._get_server_url_for_dataset(dataset_name)
                requests_by_server[server_url].append(req)
                request_to_server_mapping[idx] = server_url
            
            # Send requests to appropriate servers
            server_url_to_responses = {}
            
            for server_url, server_requests in requests_by_server.items():
                try:
                    print(f"[KG_ROUTING] Sending {len(server_requests)} requests to {server_url}")
                    
                    # Add timeout and connection settings for training stability
                    response = requests.post(
                        server_url, 
                        json=server_requests, 
                        timeout=120,  # Increased timeout for complex Freebase queries (GrailQA)
                        headers={'Content-Type': 'application/json'}
                    )
                    response.raise_for_status()  # Raise an exception for HTTP errors
                    server_url_to_responses[server_url] = response.json()
                    
                    print(f"[KG_ROUTING] Received {len(server_url_to_responses[server_url])} responses from {server_url}")
                    
                except requests.exceptions.Timeout as e:
                    # Handle timeout errors specifically  
                    print(f"[KG_TIMEOUT_ERROR] Request timed out to {server_url}")
                    error_responses = []
                    for req in server_requests:
                        error_msg = f"KG server timeout ({server_url})"
                        error_responses.append({
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except requests.exceptions.ConnectionError as e:
                    # Handle connection errors specifically
                    print(f"[KG_CONNECTION_ERROR] Connection failed to {server_url}: {str(e)}")
                    error_responses = []
                    for req in server_requests:
                        error_msg = f"KG server connection failed ({server_url})"
                        error_responses.append({
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except requests.exceptions.RequestException as e:
                    # Handle other HTTP errors with detailed logging
                    print(f"[KG_REQUEST_ERROR] HTTP error to {server_url}: {str(e)}")
                    # Add debug info about the request that failed
                    
                    error_responses = []
                    for req in server_requests:
                        error_msg = self._format_http_error(e, req)
                        error_responses.append({
                            "error": error_msg, 
                            "query_time": 0, 
                            "total_results": 0,
                            "request_payload": req,  # Include request for unique query tracking
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        })
                    server_url_to_responses[server_url] = error_responses
                    
                except json.JSONDecodeError as e:
                    # Handle invalid JSON responses
                    print(f"[KG_JSON_ERROR] Invalid JSON response from {server_url}: {str(e)}")
                    error_msg = f"KG server response error ({server_url})"
                    error_responses = [
                        {
                            "error": error_msg,
                            "query_time": 0,
                            "total_results": 0,
                            "request_payload": req,
                            "kg_metadata": {"success": False, "error_type": KGErrorType.SERVER_ERROR},
                        }
                        for req in server_requests
                    ]
                    server_url_to_responses[server_url] = error_responses
            
            # Reconstruct responses in original order
            server_responses = []
            server_response_idx = defaultdict(int)  # Track response index for each server
            
            for idx, req in enumerate(valid_requests):
                server_url = request_to_server_mapping[idx]
                response_idx = server_response_idx[server_url]
                server_responses.append(server_url_to_responses[server_url][response_idx])
                server_response_idx[server_url] += 1

        # Merge client-side errors with server responses in correct order
        final_responses = []
        server_idx = 0
        for req in parsed_requests:
            if "_is_client_error" in req:
                final_responses.append({
                    "error": req["error_message"], 
                    "query_time": 0, 
                    "total_results": 0,
                    "request_payload": req,  # Include original request for unique query tracking
                    "kg_metadata": {"success": False, "error_type": KGErrorType.FORMAT_ERROR},
                })
            else:
                response = server_responses[server_idx].copy()  # Make a copy to avoid modifying original
                response["request_payload"] = req  # Attach the original request payload
                if "kg_metadata" not in response:
                    response["kg_metadata"] = {
                        "success": response.get("success", True),
                        "error_type": response.get("error_type", KGErrorType.SUCCESS)
                    }
                final_responses.append(response)
                server_idx += 1

        # VALIDATION: Ensure response count matches query count
        if len(final_responses) != len(search_query_contents):
            raise ValueError(
                f"Final response count mismatch in _batch_search: "
                f"got {len(final_responses)} responses for {len(search_query_contents)} queries"
            )
        
        formatted_responses = [self._passages2string(item) for item in final_responses]
        
        # VALIDATION: Ensure formatted response count matches
        if len(formatted_responses) != len(search_query_contents):
            raise ValueError(
                f"Formatted response count mismatch in _batch_search: "
                f"got {len(formatted_responses)} formatted responses for {len(search_query_contents)} queries"
            )
        
        return final_responses, formatted_responses

    def _passages2string(self, kg_server_response_item: Dict[str, Any]) -> str:
        """
        Formats a single KG server response item into a string for the LLM.
        Updated to handle the new kg_retrieval response format.
        """
        if not isinstance(kg_server_response_item, dict):
            return f"Error: Unexpected KG server response format: {type(kg_server_response_item)}"

        # Handle the new kg_retrieval format
        if "object" in kg_server_response_item and kg_server_response_item.get("object") == "kg_retrieval":
            # New format: kg_retrieval response
            if not kg_server_response_item.get("success", False):
                # Error case - return the error message directly without prefixing
                if "choices" in kg_server_response_item and kg_server_response_item["choices"]:
                    error_content = kg_server_response_item["choices"][0].get("message", {}).get("content", "Unknown error")
                    return error_content
                return "Unknown error occurred. Please check your search format."
            
            # Success case - extract content from choices
            if "choices" in kg_server_response_item and kg_server_response_item["choices"]:
                content = kg_server_response_item["choices"][0].get("message", {}).get("content", "No content")
                return content
            
            return "No content in KG server response"
        
        # Handle old format for backward compatibility
        if "error" in kg_server_response_item:
            return kg_server_response_item['error']  # Return error message directly

        action_actual_results = kg_server_response_item.get("results")
        if not action_actual_results or not isinstance(action_actual_results, list) or len(action_actual_results) == 0:
            # This case might also indicate an error handled by the server and put into the 'results' list.
            # e.g. {"results": [{"error": "subgraph not found"}]}
            if isinstance(action_actual_results, list) and len(action_actual_results) > 0 and "error" in action_actual_results[0]:
                 return action_actual_results[0]['error']  # Return error message directly
            return "No results found for your search query."

        data_payload = action_actual_results[0] # The actual data is wrapped in a list

        if "error" in data_payload: # Error specific to the KG operation for that item
            return data_payload['error']  # Return error message directly

        if "relations" in data_payload:
            relations = data_payload['relations']
            if relations:
                return f"Found relations: {', '.join(relations)}."
            else:
                return "No relations found."
        elif "head_entities" in data_payload:
            head_entities = data_payload['head_entities']
            if head_entities:
                return f"Found head entities: {', '.join(head_entities)}."
            else:
                return "No head entities found."
        elif "tail_entities" in data_payload:
            tail_entities = data_payload['tail_entities']
            if tail_entities:
                return f"Found tail entities: {', '.join(tail_entities)}."
            else:
                return "No tail entities found."
        else:
            return f"Retrieved data: {json.dumps(data_payload)}" # Fallback for other structures

    def _get_available_actions_from_server(self) -> List[str]:
        """Get available actions from the server endpoint."""
        try:
            response = requests.get(f"{self.config.search_url.rstrip('/retrieve')}/actions")
            if response.status_code == 200:
                return response.json().get("actions", [])
        except Exception:
            pass
        # Fallback to default actions (new naming scheme)
        return ["get_relations_in", "get_relations_out", "get_entities_in", "get_entities_out"]

    def _create_query_format_error(self, query_content: str, original_error: str) -> str:
        """Create an LLM-friendly error message for query format issues."""
        
        # Get available actions dynamically from server
        available_actions = self._get_available_actions_from_server()
        functions_list = ", ".join(available_actions)
        
        return f"Query format error. Available functions: {functions_list}"

    def _parse_kg_query(self, query_content_str: str) -> Dict[str, Any]:
        """
        Parse KG query in function-call format.
        
        Format: "action_type("entity_name"[, "relation_name"])" (quoted format - preferred)
        Also accepts: "action_type(entity_name[, relation_name])" (unquoted format for backward compatibility)
        
        Returns:
            Dictionary with parsed components: action_type, entity_id, relation_name
        """
        
        query_content_str = query_content_str.strip()
        
        # Try to strip known problematic prefixes using pre-compiled patterns
        cleaned_query = query_content_str
        for prefix_pattern in self.prefix_patterns:
            cleaned_query = prefix_pattern.sub('', cleaned_query).strip()
            
        # Also handle cases like 'kg-query execute "get_head_entities(...)"' 
        # where the function call is wrapped in extra quotes
        if cleaned_query.startswith('"') and cleaned_query.endswith('"'):
            cleaned_query = cleaned_query[1:-1].strip()
        elif cleaned_query.startswith("'") and cleaned_query.endswith("'"):
            cleaned_query = cleaned_query[1:-1].strip()
            
        # Special case: handle malformed quotes like 'get_head_entities(...))"'
        if cleaned_query.endswith(')"') and not cleaned_query.startswith('"'):
            cleaned_query = cleaned_query[:-1]  # Remove trailing quote
        elif cleaned_query.endswith(")'") and not cleaned_query.startswith("'"):
            cleaned_query = cleaned_query[:-1]  # Remove trailing quote
            
        # Handle nested function calls using pre-compiled pattern
        nested_match = self.nested_pattern.match(cleaned_query)
        if nested_match:
            cleaned_query = nested_match.group(1).strip()
        
        # Check for get_relations, get_head_relations, get_tail_relations first - strict single entity only
        get_relations_quoted_match = self.get_relations_quoted_pattern.match(cleaned_query)
        if get_relations_quoted_match:
            action_type = get_relations_quoted_match.group(1)
            entity_id = get_relations_quoted_match.group(2)
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": None
            }
            
        get_relations_unquoted_match = self.get_relations_unquoted_pattern.match(cleaned_query)
        if get_relations_unquoted_match:
            action_type = get_relations_unquoted_match.group(1)
            entity_id = get_relations_unquoted_match.group(2).strip()
            # Strip quotes if present
            if entity_id.startswith('"') and entity_id.endswith('"'):
                entity_id = entity_id[1:-1]
            elif entity_id.startswith("'") and entity_id.endswith("'"):
                entity_id = entity_id[1:-1]
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": None
            }
        
        # Try function-call format with quotes for other functions (get_head_entities, get_tail_entities)
        quoted_match = self.quoted_function_pattern.match(cleaned_query)
        
        if quoted_match:
            action_type = quoted_match.group(1)
            
            # Reject single-entity functions with 2+ arguments - should have been caught above
            if action_type in ["get_relations", "get_relations_in", "get_relations_out", "get_head_relations", "get_tail_relations"]:
                error_msg = (
                    f"Invalid {action_type} format: '{query_content_str}'. "
                    f"{action_type} accepts only one entity argument: {action_type}(\"entity\"). "
                    f"For relation-specific queries, use get_entities_out(\"entity\", \"relation\") or get_entities_in(\"entity\", \"relation\")."
                )
                raise ValueError(error_msg)
            
            entity_id = quoted_match.group(2)  # Quotes are automatically removed by the regex groups
            relation_name = quoted_match.group(3) if quoted_match.group(3) else None
            
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": relation_name
            }
        
        # Try function-call format without quotes for other functions (get_head_entities, get_tail_entities)
        match = self.function_pattern.match(cleaned_query)
        
        if match:
            action_type = match.group(1).strip()
            
            # Reject single-entity functions with 2+ arguments - should have been caught above
            if action_type in ["get_relations", "get_relations_in", "get_relations_out", "get_head_relations", "get_tail_relations"]:
                error_msg = (
                    f"Invalid {action_type} format: '{query_content_str}'. "
                    f"{action_type} accepts only one entity argument: {action_type}(\"entity\") or {action_type}(entity). "
                    f"For relation-specific queries, use get_entities_out(\"entity\", \"relation\") or get_entities_in(\"entity\", \"relation\")."
                )
                raise ValueError(error_msg)
            
            entity_id = match.group(2).strip()
            relation_name = match.group(3).strip() if match.group(3) else None
            
            # CRITICAL FIX: Strip quotes from entity_id and relation_name when using unquoted pattern
            # This handles mixed quoting cases like: get_tail_entities(m.046vpjr, "relation_name")
            if entity_id.startswith('"') and entity_id.endswith('"'):
                entity_id = entity_id[1:-1]
            elif entity_id.startswith("'") and entity_id.endswith("'"):
                entity_id = entity_id[1:-1]
                
            if relation_name:
                if relation_name.startswith('"') and relation_name.endswith('"'):
                    relation_name = relation_name[1:-1]
                elif relation_name.startswith("'") and relation_name.endswith("'"):
                    relation_name = relation_name[1:-1]
            
            return {
                "action_type": action_type,
                "entity_id": entity_id,
                "relation_name": relation_name
            }
        
        # No valid format found - provide clear error message with quoted format examples
        error_msg = (
            f"Invalid query format: '{query_content_str}'. "
            f"Use function call format like: get_relations_in(\"entity\") or get_entities_out(\"entity\", \"relation\"). "
            f"Available functions: get_relations_in(\"entity\"), get_relations_out(\"entity\"), get_entities_out(\"entity\", \"relation\"), get_entities_in(\"entity\", \"relation\"). "
            f"Example: get_relations_in(\"Natalie Portman\")"
        )
        raise ValueError(error_msg)

    def _extract_kg_meta_info_for_queries(self, cur_actions: List[str], meta_info: Dict, active_mask, kg_query_batch_indices: List[int] = None) -> List[Dict]:
        """
        Extract per-sample meta_info for KG queries.
        
        Args:
            cur_actions: List of actions for each sample
            meta_info: Batch meta_info that may contain per-sample data
            active_mask: Mask indicating which samples are active
            
        Returns:
            List of meta_info dicts, one for each KG query
        """
        kg_query_meta_infos = []
        
        # Check if we have per-sample information available
        batch_sample_ids = meta_info.get("sample_ids", [])
        batch_dataset_names = meta_info.get("dataset_names", [])
        fallback_sample_id = meta_info.get("sample_id", "unknown_sample_id")
        
        
        # VALIDATION: The trainer must have already expanded sample_ids after DataProto.repeat()
        # This is critical for correct query-response mapping with GRPO rollouts
        if batch_sample_ids:
            if len(batch_sample_ids) != len(cur_actions):
                raise ValueError(
                    f"Sample IDs length mismatch: len(batch_sample_ids)={len(batch_sample_ids)} != "
                    f"len(cur_actions)={len(cur_actions)}. The trainer should have expanded sample_ids "
                    f"after DataProto.repeat() for GRPO rollouts."
                )
        
        # Generate fallback sample IDs only if we don't have per-sample info at all
        if not batch_sample_ids:
            if fallback_sample_id == "unknown_sample_id":
                batch_sample_ids = [f"fallback_sample_{i:06d}" for i in range(len(cur_actions))]
            else:
                # If we have a single fallback_sample_id, it means we're in a non-expanded batch scenario
                batch_sample_ids = [fallback_sample_id] * len(cur_actions)
        
        # Better fallback logic for dataset_name
        fallback_dataset_name = meta_info.get("dataset_name", None)
        if fallback_dataset_name is None:
            # Try to extract from data_source if available
            data_source = meta_info.get("data_source", None)
            if data_source:
                if isinstance(data_source, list) and len(data_source) > 0:
                    data_source = data_source[0]
                
                if data_source in ["webqsp_kg", "webqsp"]:
                    fallback_dataset_name = "webqsp"
                elif data_source in ["cwq_kg", "cwq"]:
                    fallback_dataset_name = "CWQ"
                elif data_source in ["kgR1_multitq", "multitq"]:
                    fallback_dataset_name = "multitq"
                else:
                    fallback_dataset_name = "webqsp"  # Safe fallback
            else:
                fallback_dataset_name = "webqsp"  # Safe fallback instead of "unknown_dataset_name"
        
        # Generate fallback dataset names if we don't have per-sample info
        if not batch_dataset_names:
            batch_dataset_names = [fallback_dataset_name] * len(cur_actions)
        
        # If kg_query_batch_indices is provided, use it directly
        if kg_query_batch_indices:
            for kg_query_idx, batch_idx in enumerate(kg_query_batch_indices):
                # Get per-sample info for this specific batch index
                if batch_idx < len(batch_sample_ids):
                    sample_id = batch_sample_ids[batch_idx]
                else:
                    if fallback_sample_id == "unknown_sample_id":
                        sample_id = f"query_sample_{batch_idx:06d}"  # Generate unique sample ID
                    else:
                        sample_id = fallback_sample_id
                    
                if batch_idx < len(batch_dataset_names):
                    dataset_name = batch_dataset_names[batch_idx]
                else:
                    dataset_name = fallback_dataset_name
                
                kg_query_meta_infos.append({
                    "sample_id": sample_id,
                    "dataset_name": dataset_name,
                    "batch_index": batch_idx  # Add absolute batch index for debugging
                })
        else:
            # Fallback to old behavior if kg_query_batch_indices not provided
            kg_query_idx = 0
            for i, (action, active) in enumerate(zip(cur_actions, active_mask)):
                if active and action == 'kg-query':
                    # Get per-sample info if available, otherwise use fallback
                    if i < len(batch_sample_ids):
                        sample_id = batch_sample_ids[i]
                    else:
                        if fallback_sample_id == "unknown_sample_id":
                            sample_id = f"query_sample_{i:06d}"  # Generate unique sample ID
                        else:
                            sample_id = fallback_sample_id
                        
                    if i < len(batch_dataset_names):
                        dataset_name = batch_dataset_names[i]
                    else:
                        dataset_name = fallback_dataset_name
                    
                    kg_query_meta_infos.append({
                        "sample_id": sample_id,
                        "dataset_name": dataset_name,
                        "batch_index": i  # Add absolute batch index for debugging
                    })
                    kg_query_idx += 1
        
        return kg_query_meta_infos

    def _format_http_error(self, error: Exception, request_data: Dict = None) -> str:
        """
        Format HTTP transport errors into concise messages.
        Note: Server-side errors are now handled by the KG server with detailed messages.
        This only handles HTTP transport failures.
        
        Args:
            error: The HTTP exception
            request_data: The original request data (used for context in fallback cases)
            
        Returns:
            Concise error message for HTTP transport failures
        """
        error_str = str(error).lower()
        
        # Extract entity name from request if available for fallback
        entity_name = "query"
        if request_data and "entity_id" in request_data:
            entity_name = f"'{request_data['entity_id']}'"
        
        # For actual server errors that contain detailed error messages, try to extract them
        if hasattr(error, 'response') and error.response is not None:
            try:
                # Try to get the detailed server error message
                response_data = error.response.json()
                
                # Handle list response (batch requests)
                if isinstance(response_data, list) and len(response_data) > 0:
                    server_response = response_data[0]
                else:
                    server_response = response_data
                
                # Try multiple extraction strategies
                detailed_error = None
                
                # Strategy 1: kg_retrieval format with choices
                if ("choices" in server_response and 
                    len(server_response["choices"]) > 0 and
                    "message" in server_response["choices"][0] and
                    "content" in server_response["choices"][0]["message"]):
                    detailed_error = server_response["choices"][0]["message"]["content"]
                
                # Strategy 2: Direct error field
                elif "error" in server_response:
                    detailed_error = server_response["error"]
                
                # Strategy 3: Detail field (FastAPI validation errors)
                elif "detail" in server_response:
                    if isinstance(server_response["detail"], list) and len(server_response["detail"]) > 0:
                        # FastAPI validation error format
                        detail_item = server_response["detail"][0]
                        if "msg" in detail_item:
                            detailed_error = detail_item["msg"]
                        else:
                            detailed_error = str(server_response["detail"])
                    else:
                        detailed_error = str(server_response["detail"])
                
                # Strategy 4: Success=False format  
                elif (server_response.get("success") is False and 
                      "message" in server_response):
                    detailed_error = server_response["message"]
                
                if detailed_error:
                    return detailed_error
                
                # If we got a response but couldn't extract error, log it for debugging
                
            except (json.JSONDecodeError, KeyError, IndexError, AttributeError, TypeError) as parse_error:
                print(f"[KG_PARSE_ERROR] Failed to parse server response: {parse_error}")
                # Try to get raw response text for debugging
                try:
                    response_text = error.response.text[:200] if hasattr(error.response, 'text') else "No response text"
                    print(f"[KG_RAW_RESPONSE] First 200 chars: {response_text}")
                except:
                    pass
        
        # Handle HTTP transport errors with simple, generic messages
        if "timeout" in error_str or "timed out" in error_str:
            return "KG server request timed out"
        elif "connection" in error_str or "cannot connect" in error_str:
            return "Cannot connect to KG server"
        elif "404" in error_str or "not found" in error_str:
            return "KG server endpoint not found"
        elif "500" in error_str or "internal server error" in error_str:
            return "KG server internal error"
        elif "503" in error_str or "service unavailable" in error_str:
            return "KG server unavailable"
        else:
            # Generic fallback for any other HTTP transport issues
            return f"KG server request failed for {entity_name}"


