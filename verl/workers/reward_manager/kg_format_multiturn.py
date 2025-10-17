"""
Multi-turn KG Format Reward Manager.

This module extends the standard KG format reward manager to support 
turn-wise reward calculation for multi-turn KG-augmented reasoning.

Reward Structure:
- Turn-specific: KG valid query + retrieval quality (per turn)
- Global: Format scoring + exact match (entire sequence)
- Pass@K: Bootstrap sampling metrics for GRPO multiple responses
"""

import torch
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from collections import defaultdict

from .kg_format import KGFormatRewardManager
from verl.utils.reward_score.qa_em_format_kg import normalize_answer, is_retrieval_correct_kg, extract_answer_kg, em_check_kg, extract_assistant_response
from kg_r1.search.error_types import KGErrorType

# Removed enhanced_metrics imports - using entity-level calculations only


class KGFormatMultiTurnRewardManager(KGFormatRewardManager):
    """
    Multi-turn extension of KG Format Reward Manager.
    
    Provides turn-wise reward calculation where:
    - Turn-specific rewards: KG query validity + retrieval quality
    - Global rewards: Format compliance + exact match
    """
    
    def __init__(self, tokenizer, num_examine, **reward_kwargs):
        super().__init__(tokenizer, num_examine, **reward_kwargs)
        
        # Multi-turn specific configuration
        self.turn_specific_weights = {
            'kg_query_validity': reward_kwargs.get('turn_kg_query_validity', 0.1),
            'is_answer_score': reward_kwargs.get('turn_is_answer_score', 0.1),
            'format_score': reward_kwargs.get('turn_format_score', 0.15),
        }
        
        self.global_weights = {
            'exact_match': reward_kwargs.get('global_exact_match', 0.3),
            'retrieval_quality': reward_kwargs.get('global_retrieval_quality', 0.4),
        }
        
        self.verbose = reward_kwargs.get('verbose', False)
        
        # Answer score mode: 'binary' (default) or 'f1' 
        self.answer_score_mode = reward_kwargs.get('answer_score_mode', 'binary')
        
        # Clean logging control - set this to True to see full assistant responses
        self.show_full_responses = reward_kwargs.get('show_full_responses', False)
        
        # OTC (Optimal Turn Count) scaling configuration
        self.otc_scaling = reward_kwargs.get('otc_scaling', False)
        self.max_turns = reward_kwargs.get('max_turns', 7)  # Default max turns
        
        if self.verbose:
            print(f"[MultiTurnReward] Turn weights: {self.turn_specific_weights}")
            print(f"[MultiTurnReward] Global weights: {self.global_weights}")
            print(f"[MultiTurnReward] Answer score mode: {self.answer_score_mode}")
            print(f"[MultiTurnReward] OTC scaling: {self.otc_scaling} (max_turns: {self.max_turns})")
            if self.show_full_responses:
                print(f"[MultiTurnReward] Full response logging enabled")
        
        # Track which samples we've printed for num_examine functionality
        self.printed_samples = 0
    
    def __call__(self, data, return_dict=False):
        """
        Calculate multi-turn rewards and return structured reward dictionaries.
        
        Args:
            data: Batch data containing input/output sequences and metadata
            return_dict: Whether to return dict with reward_extra_info for WandB logging
        
        Returns:
            List[Dict] or dict: Structured rewards per sample, optionally with extra info dict
        """
        # Check if turn_sequence_tensor is available
        turn_sequence_tensor = data.batch.get('turn_sequence_tensor')
        if turn_sequence_tensor is None:
            # Fallback to single-turn reward calculation
            if self.verbose:
                print("[MultiTurnReward] No turn_sequence_tensor found, falling back to single-turn")
            return super().__call__(data)
        
        batch_size = len(data)
        
        # Store detailed reward info for analysis
        detailed_rewards = []
        
        # Process each sample in the batch
        for i in range(batch_size):
            # Calculate multi-turn rewards for this sample
            reward_dict = self._calculate_multiturn_rewards(data, i)
            detailed_rewards.append(reward_dict)
        
        # Store detailed rewards in meta_info for analysis
        if not hasattr(data, 'meta_info'):
            data.meta_info = {}
        data.meta_info['detailed_multiturn_rewards'] = detailed_rewards
        
        # Log sample responses in kg_format.py style
        if True:
            # Extract data source for proper header
            data_source = "webqsp_kg"  # Conservative default (original)
            if hasattr(data, 'meta_info') and data.meta_info:
                data_source = data.meta_info.get('data_source', 'webqsp_kg')
            elif len(data) > 0 and hasattr(data[0], 'meta_info') and data[0].meta_info:
                # Try to extract from individual sample meta_info
                data_source = data[0].meta_info.get('data_source', 'webqsp_kg')
            
            self._log_sample_detailed(data, detailed_rewards, data_source)
            # Note: _log_multiturn_stats will be called from compute_grpo_multiturn_advantage
        
        # Create reward_tensor for validation compatibility
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        for i, reward_dict in enumerate(detailed_rewards):
            # Get response length for this sample
            response_ids = data[i].batch["responses"]
            valid_response_length = data[i].batch["attention_mask"][data[i].batch["prompts"].shape[-1]:].sum()
            # Place total score at the last token position (following parent class pattern)
            reward_tensor[i, valid_response_length - 1] = reward_dict["total_score"]
        
        # Prepare reward_extra_info for WandB logging
        if return_dict:
            # Pass UID information for proper GRPO-style grouping in Pass@K
            uid_info = data.non_tensor_batch.get('uid', None) if hasattr(data, 'non_tensor_batch') else None
            reward_extra_info = self._compute_wandb_metrics(detailed_rewards, uid_info)
            return {
                "structured_rewards": detailed_rewards,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return detailed_rewards
    
    def _calculate_multiturn_rewards(self, data, sample_idx: int) -> Dict[str, Any]:
        """
        Calculate turn-specific and global rewards for a single sample.
        
        Args:
            data: Batch data
            sample_idx: Index of the sample in the batch
            
        Returns:
            Dict containing turn_rewards and global_rewards
        """
        # Extract sample data
        data_item = data[sample_idx]
        
        # Get interaction history for this sample
        batch_interaction_history = data.meta_info.get("interaction_history", [])
        sample_interaction_history = {}
        if batch_interaction_history and sample_idx < len(batch_interaction_history):
            sample_interaction_history = batch_interaction_history[sample_idx]
        
        # Parse turns from interaction history
        turn_data = self._parse_turns_from_interaction_history(sample_interaction_history, sample_idx)
        
        # Track unique queries for this sample to prevent reward hacking
        seen_query_ids = set()
        
        # Extract full assistant response text for format checking (following kg_format.py pattern)
        prompt_ids = data_item.batch["prompts"]
        response_ids = data_item.batch["responses"]
        attention_mask = data_item.batch["attention_mask"]
        
        prompt_length = prompt_ids.shape[-1]
        
        # Fix: attention_mask is 2D (batch_size, seq_len), need to index correctly
        sample_attention_mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
        
        valid_prompt_length = sample_attention_mask[:prompt_length].sum()
        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
        
        valid_response_length = sample_attention_mask[prompt_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]
        
        # Decode full sequence
        sequences = torch.cat((valid_prompt_ids, valid_response_ids))
        if sequences.dtype != torch.long:
            sequences = sequences.long()
        full_sequence_str = self.tokenizer.decode(sequences)
        
        # For multi-turn, we don't need to extract assistant response from the full sequence
        # We use the turn-by-turn responses_str from server interaction history instead
        assistant_response = full_sequence_str  # Keep full sequence for any legacy compatibility
        
        # Calculate turn-specific rewards with component breakdown
        turn_rewards = {}
        turn_components = {}  # Store individual components for WandB logging
        for turn_num, turn_info in turn_data.items():
            # Pass current verbose state to turn calculation
            turn_reward, components = self._calculate_turn_reward_with_components(turn_info, turn_num, assistant_response, seen_query_ids, data_item, sample_idx, sample_interaction_history, verbose=self.verbose)
            turn_rewards[turn_num] = turn_reward
            turn_components[turn_num] = components
        
        # Debug: Compare responses_str with tensor-based extraction (commented out to reduce log verbosity)
        #        #     self._debug_compare_extraction_methods(data_item, sample_interaction_history, turn_data)
        
        # Calculate global rewards
        global_rewards = self._calculate_global_rewards(data_item, sample_interaction_history)
        
        # Calculate total score: avg(turn_rewards) + sum(global_rewards)
        # Only sum actual weighted components for global rewards (exclude raw logging values)
        total_turn_score = sum(turn_rewards.values()) / len(turn_rewards) if turn_rewards else 0.0
        total_global_score = sum(v for k, v in global_rewards.items() if not k.startswith('_'))
        total_score = total_turn_score + total_global_score
        
        # Extract answer metrics if available
        answer_metrics = getattr(data_item, 'answer_metrics', {})
        
        return {
            "turn_rewards": turn_rewards,
            "turn_components": turn_components,  # Store for WandB logging
            "global_rewards": global_rewards,
            "total_score": total_score,
            "sample_idx": sample_idx,
            "raw_interaction_history": sample_interaction_history,  # Store for debugging
            "answer_metrics": answer_metrics  # Entity-level metrics only
        }
    
    def _parse_turns_from_interaction_history(self, interaction_history: Dict, sample_idx: int) -> Dict[int, Dict]:
        """
        Parse interaction history into turn-specific data for a single sample.
        
        Args:
            interaction_history: Single sample's interaction history
            sample_idx: Sample index (for debugging/logging)
            
        Returns:
            Dict mapping turn_num -> turn_data
        """
        actions = interaction_history.get('actions', [])
        search_results = interaction_history.get('search_results', [])
        valid_actions = interaction_history.get('valid_actions', [])
        is_search_actions = interaction_history.get('is_search_actions', [])
        raw_server_responses = interaction_history.get('raw_server_responses', [])
        
        turn_data = {}
        
        
        for turn_idx, (action, search_result, valid_action, is_search, raw_response) in enumerate(
            zip(actions, search_results, valid_actions, is_search_actions, raw_server_responses)
        ):
            turn_num = turn_idx + 1  # 1-indexed turns
            
            turn_data[turn_num] = {
                'action': action,
                'search_result': search_result,
                'valid_action': valid_action,
                'is_search': is_search,
                'raw_server_response': raw_response,
                'turn_idx': turn_idx
            }
        
        return turn_data
    
    def _calculate_turn_reward_with_components(self, turn_info: Dict, turn_num: int, full_response: str, seen_query_ids: set, data_item, sample_idx: int, sample_interaction_history: Dict, verbose: bool = False) -> tuple[float, Dict[str, float]]:
        """
        Calculate reward for a specific turn and return individual components.
        
        Args:
            turn_info: Turn-specific information
            turn_num: Turn number (1-indexed)
            full_response: Full assistant response text for format checking
            seen_query_ids: Set of already seen query IDs for uniqueness tracking
            
        Returns:
            tuple: (total_reward, components_dict)
        """
        action = turn_info['action']
        components = {
            'kg_query_validity': 0.0,
            'is_answer_score': 0.0,
            'format_score': 0.0,
            'action': action  # Store action so it shows up in logging instead of "(unknown)"
        }
        
        # Step 1: Calculate format score (applicable to both kg-query and answer actions)
        format_reward, extracted_content = self._calculate_turn_format_score_reward(turn_info, full_response, data_item, sample_idx, sample_interaction_history)
        components['extracted_content'] = extracted_content
        
        components['format_score'] = format_reward
        format_score = format_reward * self.turn_specific_weights['format_score']
        
        # Step 3: Calculate action-specific rewards
        if action == 'kg-query':
            # kg-query action: format_score (if valid format) + kg_query_validity (unique only)
            kg_query_reward = self._calculate_kg_query_validity_reward(turn_info, seen_query_ids)
            components['kg_query_validity'] = kg_query_reward
            kg_query_score = kg_query_reward * self.turn_specific_weights['kg_query_validity']
            total_reward = format_score + kg_query_score
            
        elif action == 'answer':
            # answer action: is_answer_score + format_score (if valid format)
            is_answer_reward = self._calculate_is_answer_score_reward(turn_info, data_item)
            components['is_answer_score'] = is_answer_reward
            is_answer_score = is_answer_reward * self.turn_specific_weights['is_answer_score']
            total_reward = is_answer_score + format_score
            
        else:
            # Other actions (shouldn't happen in KG reasoning, but handle gracefully)
            total_reward = 0.0
        
        return total_reward, components
    
    
    
    def _extract_query_identifier(self, raw_response: Dict) -> str:
        """
        Extract a unique identifier for the query from the server response.
        
        Uses the same logic as the original kg_format system for consistency.
        
        Args:
            raw_response: KG server response containing query information
            
        Returns:
            Unique string identifier for the query
        """
        # Import the existing function to maintain consistency
        from verl.utils.reward_score.qa_em_format_kg import extract_query_identifier
        return extract_query_identifier(raw_response)
    
    def _calculate_kg_query_validity_reward(self, turn_info: Dict, seen_query_ids: set) -> float:
        """
        Calculate KG query validity reward for a turn with uniqueness tracking.
        
        Only unique, valid queries get reward. Duplicate queries get 0 reward
        to prevent reward hacking.
        
        Args:
            turn_info: Turn information
            seen_query_ids: Set of already seen query IDs for uniqueness tracking
            
        Returns:
            float: KG query validity reward (0-1, -0.1 for invalid)
        """
        # Only process KG query actions
        if turn_info['action'] != 'kg-query':
            return 0.0  # No KG query in this turn
        
        # Check if query was syntactically valid
        if not turn_info['valid_action']:
            return 0.0
        
        # Check if query actually succeeded (not just syntactically valid)
        raw_response = turn_info.get('raw_server_response', {})
        if isinstance(raw_response, dict):
            # Check for KG-specific errors
            if 'kg_metadata' in raw_response:
                kg_metadata = raw_response['kg_metadata']
                success = kg_metadata.get('success', False)
                error_type = kg_metadata.get('error_type')
                
                # If the KG query failed, return 0 (no reward but no penalty)
                if not success or error_type != 'KG_SUCCESS':
                    return 0.0
            
            # Extract unique identifier for this query
            query_id = self._extract_query_identifier(raw_response)
            
            # Check if we've seen this query before
            if query_id in seen_query_ids:
                # Duplicate query - no reward
                return 0.0
            else:
                # Unique successful query - add to seen set and give reward
                seen_query_ids.add(query_id)
                return 1.0
        
        # Fallback: if we can't extract query ID, check valid_action only
        return 1.0 if turn_info['valid_action'] else 0.0
    
    def _calculate_is_answer_score_reward(self, turn_info: Dict, data_item=None) -> float:
        """
        Calculate is answer score reward for a turn.
        
        This is the turn-wise equivalent of final_format_score from the original system.
        Simply returns 1.0 if this turn is an answer action, 0.0 otherwise.
        
        Args:
            turn_info: Turn information
            data_item: Data item for accessing ground truth (optional)
            
        Returns:
            float: Is answer score reward (0-1)
        """
        # Check if this turn is an answer action
        if turn_info['action'] != 'answer':
            # This turn did not produce an answer
            return 0.0
            
        # This turn produced an answer - return 1.0 (binary reward)
        return 1.0
    
    def _calculate_turn_format_score_reward(self, turn_info: Dict, full_response: str, data_item=None, sample_idx: int = 0, sample_interaction_history: Dict = None) -> float:
        """
        Calculate turn-wise format score reward using server-based indexing.
        
        This now uses the reliable server interaction history instead of error-prone
        string parsing to determine format compliance.
        
        Args:
            turn_info: Turn information containing action type, server response, and turn index  
            full_response: The full assistant response text to analyze
            
        Returns:
            float: Format score reward (0-1)
        """
        import re
        
        # Only check format for turns that have actions (kg-query or answer)
        if turn_info['action'] not in ['kg-query', 'answer']:
            return (0.0, "")
        
        action = turn_info['action']
        turn_idx = turn_info.get('turn_idx', 0)
        raw_server_response = turn_info.get('raw_server_response', {})
        
        # Data item should always be provided in normal operation
        if data_item is None:
            raise ValueError(f"data_item is None for turn {turn_idx}. This indicates a problem in the data pipeline.")
        
        # Use the sample_interaction_history passed directly from _calculate_multiturn_rewards
        # This avoids re-extracting and potential indexing issues
        if sample_interaction_history is None:
            sample_interaction_history = {}
        
        turn_content = self._extract_turn_content_server_interaction(turn_info, sample_interaction_history)
        
        # If server interaction extraction fails, fall back to tensor-based extraction
        if not turn_content:
            return (0.0, "")
                
        # Validate format based on action type
        if action == 'kg-query':
            format_valid = self._has_proper_kg_query_format(turn_content)
        elif action == 'answer':
            format_valid = self._has_proper_answer_format(turn_content)
        else:
            format_valid = False
            
                
        return (1.0 if format_valid else 0.0, turn_content)
    
    def _extract_turn_content_from_tokens(self, data_item, turn_positions: torch.Tensor) -> str:
        """
        Extract turn content using actual token positions from turn sequence tensor.
        
        Args:
            data_item: Data item containing response tokens
            turn_positions: Token positions for this turn from turn sequence tensor
            
        Returns:
            str: Decoded content from the actual turn tokens
        """
        try:
            if len(turn_positions) == 0:
                return ""
            
            # Get full sequence tokens (input_ids = prompt + response) to match turn_sequence_tensor indexing
            full_tokens = data_item.batch.get("input_ids")
            if full_tokens is None:
                return ""
            
            # Extract tokens at turn positions
            if len(full_tokens.shape) > 1:
                full_tokens = full_tokens.squeeze(0)  # Remove batch dimension
            
            # Ensure positions are within bounds
            valid_positions = turn_positions[turn_positions < len(full_tokens)]
            
            # DEBUG: Check position filtering (reduced verbosity)
            #            #     print(f"[DEBUG_POSITIONS] turn_positions: {len(turn_positions)} total, {len(valid_positions)} valid")
            #     if len(turn_positions) != len(valid_positions):
            #         filtered_out = turn_positions[turn_positions >= len(full_tokens)]
            #         print(f"[DEBUG_POSITIONS] Filtered out {len(filtered_out)} positions >= {len(full_tokens)}")
            #         print(f"[DEBUG_POSITIONS] Max turn_position: {turn_positions.max().item()}, sequence length: {len(full_tokens)}")
            
            if len(valid_positions) == 0:
                return ""
            
            turn_tokens = full_tokens[valid_positions]
            
            # Debug logging for token extraction details (reduced verbosity)
            # if self.verbose and len(valid_positions) < 50:  # Only for reasonable sized extractions
            #     has_input_ids = "input_ids" in data_item.batch
            #     has_responses = "responses" in data_item.batch
            #     print(f"[DEBUG] Token source analysis:")
            #     print(f"  Has input_ids: {has_input_ids}, Has responses: {has_responses}")
            #     print(f"  Using: {'input_ids' if has_input_ids else 'responses'}")
            #     print(f"  Full tokens length: {len(full_tokens)}")
            #     print(f"  Total response tokens: {len(full_tokens)}")
            #     print(f"  Valid positions: {valid_positions[:10].tolist()}..." if len(valid_positions) > 10 else f"  Valid positions: {valid_positions.tolist()}")
            #     print(f"  Turn tokens shape: {turn_tokens.shape}")
            
            # Decode the tokens
            turn_content = self.tokenizer.decode(turn_tokens, skip_special_tokens=True)
            
            # Debug logging for extracted content (reduced verbosity)
            #            #     print(f"\n[DEBUG] EXTRACTED TURN CONTENT:")
            #     print(f"  Content length: {len(turn_content)} chars")
            #     print(f"  Content preview: {turn_content[:500]}...")
            #     
            #     # Test format checking on extracted content
            #     import re
            #     think_match = re.search(r'<think>(.*?)</think>', turn_content, re.DOTALL)
            #     kg_match = re.search(r'<kg-query>(.*?)</kg-query>', turn_content, re.DOTALL)
            #     print(f"\n[DEBUG] FORMAT CHECK ON EXTRACTED CONTENT:")
            #     print(f"  Has <think>...</think>: {think_match is not None}")
            #     print(f"  Has <kg-query>...</kg-query>: {kg_match is not None}")
            #     print(f"  Both present (good format): {think_match is not None and kg_match is not None}")
            
            return turn_content.strip()
            
        except Exception as e:
            print(f"[ERROR] Failed to extract turn content from tokens: {e}")
            return ""
    
    def _extract_turn_content_server_interaction(self, turn_info: Dict, sample_interaction_history: Dict) -> str:
        """
        Extract turn content using responses_str from server interaction history.
        This is the primary and most reliable method for both kg-query and answer actions.
        
        Args:
            turn_info: Turn information containing turn index
            sample_interaction_history: Single sample's interaction history
            
        Returns:
            str: Extracted turn content, or empty string if extraction fails
        """
        try:
            turn_idx = turn_info.get('turn_idx', 0)
            responses_str_list = sample_interaction_history.get('responses_str', [])
            
            return responses_str_list[turn_idx]
            
        except (IndexError, KeyError, TypeError):
            return ""
    
    def _extract_turn_content_tensor_based(self, turn_info: Dict, data_item, full_response: str) -> str:
        """
        Extract turn content using turn_sequence_tensor and token positions.
        This is the fallback method when server interaction extraction fails.
        
        Args:
            turn_info: Turn information containing turn index
            data_item: Data item with tensor information
            full_response: Full response text for context
            sample_idx: Sample index in the batch
            
        Returns:
            str: Extracted turn content, or empty string if extraction fails
        """
        try:
            turn_idx = turn_info.get('turn_idx', 0)
            
            
            # Get turn_sequence_tensor from batch (tensors are stored in batch, not meta_info)
            turn_sequence_tensor = data_item.batch.get('turn_sequence_tensor')
            if turn_sequence_tensor is None:
                return ""
            
            # Handle batch dimension - get the tensor for this sample (assuming single sample processing)
            if len(turn_sequence_tensor.shape) > 1:
                sample_turn_tensor = turn_sequence_tensor[0]  # Single sample processing
            else:
                sample_turn_tensor = turn_sequence_tensor
            
            # Find all token positions that belong to this turn (turn_idx + 1 since tensor uses 1-based indexing)
            target_turn_id = turn_idx + 1
            turn_token_positions = torch.where(sample_turn_tensor == target_turn_id)[0]
            
            if len(turn_token_positions) == 0:
                return ""
            
            # Extract turn content using the token positions
            turn_content = self._extract_turn_content_from_tokens(data_item, turn_token_positions)
            
            
            return turn_content
            
        except Exception as e:
            print(f"[ERROR] Tensor-based extraction failed for turn {turn_idx}: {e}")
            return ""
    
    
    def _has_proper_kg_query_format(self, turn_content: str) -> bool:
        """
        Check if the turn content has proper MDP-style format for a kg-query.
        
        Strict format: ENTIRE content must be <think>content</think> -> (only whitespace/newlines) -> <kg-query>content</kg-query>
        No extra text before or after is allowed.
        
        Args:
            turn_content: Content for this turn
            
        Returns:
            bool: True if format follows strict MDP pattern for the ENTIRE content
        """
        import re
        
        # Strip leading/trailing whitespace for validation
        content = turn_content.strip()
        
        # Strict pattern: entire content must match exactly this pattern
        # ^<think>.*?</think>\s*<kg-query>.*?</kg-query>$
        strict_pattern = r'^<think>.*?</think>\s*<kg-query>.*?</kg-query>$'
        
        match = re.match(strict_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return False
        
        # Additional validation: ensure no malformed/duplicate tags
        # Count opening and closing tags - should be exactly 1 of each
        think_open_count = len(re.findall(r'<think>', content, re.IGNORECASE))
        think_close_count = len(re.findall(r'</think>', content, re.IGNORECASE))
        kg_open_count = len(re.findall(r'<kg-query>', content, re.IGNORECASE))
        kg_close_count = len(re.findall(r'</kg-query>', content, re.IGNORECASE))
        
        # Must have exactly 1 of each tag
        if not (think_open_count == 1 and think_close_count == 1 and 
                kg_open_count == 1 and kg_close_count == 1):
            return False
        
        # Extract text between </think> and <kg-query> to ensure only whitespace
        think_end_match = re.search(r'</think>', content, re.IGNORECASE)
        kg_start_match = re.search(r'<kg-query>', content, re.IGNORECASE)
        
        if not think_end_match or not kg_start_match:
            return False
        
        think_end_pos = think_end_match.end()
        kg_start_pos = kg_start_match.start()
        
        between_text = content[think_end_pos:kg_start_pos]
        
        # Check that between_text contains only whitespace (spaces, tabs, newlines)
        return re.match(r'^\s*$', between_text) is not None
    
    def _has_proper_answer_format(self, turn_content: str) -> bool:
        """
        Check if the turn content has proper MDP-style format for an answer.
        
        Strict format: ENTIRE content must be <think>content</think> -> (only whitespace/newlines) -> <answer>content</answer>
        No extra text before or after is allowed.
        
        Args:
            turn_content: Content for this turn
            
        Returns:
            bool: True if format follows strict MDP pattern for the ENTIRE content
        """
        import re
        
        # Strip leading/trailing whitespace for validation
        content = turn_content.strip()
        
        # Strict pattern: entire content must match exactly this pattern
        # ^<think>.*?</think>\s*<answer>.*?</answer>$
        strict_pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>$'
        
        match = re.match(strict_pattern, content, re.DOTALL | re.IGNORECASE)
        
        if not match:
            return False
        
        # Additional validation: ensure no malformed/duplicate tags
        # Count opening and closing tags - should be exactly 1 of each
        think_open_count = len(re.findall(r'<think>', content, re.IGNORECASE))
        think_close_count = len(re.findall(r'</think>', content, re.IGNORECASE))
        answer_open_count = len(re.findall(r'<answer>', content, re.IGNORECASE))
        answer_close_count = len(re.findall(r'</answer>', content, re.IGNORECASE))
        
        # Must have exactly 1 of each tag
        if not (think_open_count == 1 and think_close_count == 1 and 
                answer_open_count == 1 and answer_close_count == 1):
            return False
        
        # Extract text between </think> and <answer> to ensure only whitespace
        think_end_match = re.search(r'</think>', content, re.IGNORECASE)
        answer_start_match = re.search(r'<answer>', content, re.IGNORECASE)
        
        if not think_end_match or not answer_start_match:
            return False
        
        think_end_pos = think_end_match.end()
        answer_start_pos = answer_start_match.start()
        
        between_text = content[think_end_pos:answer_start_pos]
        
        # Check that between_text contains only whitespace (spaces, tabs, newlines)
        return re.match(r'^\s*$', between_text) is not None
    
    def _calculate_retrieval_quality_reward(self, turn_info: Dict) -> float:
        """
        Calculate retrieval quality reward for a single turn.
        
        NOTE: This method is no longer used for reward calculation since retrieval
        quality has been moved to global rewards. Kept for potential future use.
        
        Args:
            turn_info: Turn information containing search results and raw responses
            
        Returns:
            float: Retrieval quality reward (0-1)
        """
        # Only calculate if this turn involved KG search
        if not turn_info['is_search'] or turn_info['action'] != 'kg-query':
            return 0.0
        
        # Check if this turn had valid action (successful query)
        if not turn_info['valid_action']:
            return 0.0
        
        search_result = turn_info.get('search_result', '')
        raw_response = turn_info.get('raw_server_response', {})
        
        # We need ground truth to compare against - this should be passed from higher level
        # For now, we'll implement a simpler quality check based on successful retrieval
        
        # 1. Check if the KG server returned successful results
        success_score = self._evaluate_kg_server_success(raw_response)
        
        # 2. Check if the search result is non-empty and meaningful
        content_quality_score = self._evaluate_search_content_quality(search_result)
        
        # 3. Check for uniqueness (avoid repetitive queries)
        # This would need to be implemented with turn history comparison
        uniqueness_score = 1.0  # Placeholder - implement if needed
        
        # Combine scores (weighted average)
        total_score = (
            success_score * 0.5 +           # Server success is important
            content_quality_score * 0.4 +   # Content quality matters
            uniqueness_score * 0.1           # Uniqueness is bonus
        )
        
        if self.verbose:
            print(f"[RetrievalQuality] success={success_score:.2f}, "
                  f"content={content_quality_score:.2f}, "
                  f"unique={uniqueness_score:.2f}, total={total_score:.2f}")
        
        return total_score
    
    def _evaluate_kg_server_success(self, raw_response: Dict) -> float:
        """
        Evaluate if the KG server returned successful results.
        
        Args:
            raw_response: Raw server response dictionary
            
        Returns:
            float: Success score (0-1)
        """
        if not raw_response:
            return 0.0
        
        # Check kg_metadata for success
        kg_metadata = raw_response.get('kg_metadata', {})
        success = kg_metadata.get('success', False)
        error_type = kg_metadata.get('error_type', KGErrorType.SERVER_ERROR)
        
        if success and error_type == KGErrorType.SUCCESS:
            return 1.0
        elif error_type == KGErrorType.SERVER_ERROR:
            return 0.0  # Server issues, no penalty but no reward
        elif error_type in [KGErrorType.ENTITY_NOT_FOUND, KGErrorType.SAMPLE_NOT_FOUND, KGErrorType.RELATION_NOT_FOUND, KGErrorType.NO_RESULTS]:
            return 0.2  # Valid query but no results found
        elif error_type == KGErrorType.FORMAT_ERROR:
            return 0.0  # Invalid query format
        else:
            return 0.1  # Unknown error, small reward for trying
    
    def _evaluate_search_content_quality(self, search_result: str) -> float:
        """
        Evaluate the quality of the search result content.
        
        Args:
            search_result: The formatted search result shown to the LLM
            
        Returns:
            float: Content quality score (0-1)
        """
        if not search_result or not search_result.strip():
            return 0.0
        
        # Remove information tags and whitespace
        content = search_result.replace('<information>', '').replace('</information>', '').strip()
        
        if not content:
            return 0.0
        
        # Check for meaningful content patterns
        quality_indicators = [
            'Found' in content,              # Standard success message
            ':' in content,                  # Structured data (e.g., "Found relations: ...")
            len(content.split()) > 2,        # Non-trivial length
            not content.lower().startswith('error'),  # Not an error message
            not content.lower().startswith('invalid'), # Not invalid message
        ]
        
        # Calculate quality score based on indicators
        quality_score = sum(quality_indicators) / len(quality_indicators)
        
        # Bonus for longer, more informative content
        length_bonus = min(len(content.split()) / 10.0, 0.2)  # Up to 0.2 bonus for 10+ words
        
        return min(quality_score + length_bonus, 1.0)
    
    def _check_retrieval_contains_answer(self, turn_info: Dict, ground_truth_answers: List[str]) -> bool:
        """
        Check if the retrieval for this turn contains the target answer.
        
        This is the most sophisticated check - whether the retrieved information
        actually contains the correct answer we're looking for.
        
        Args:
            turn_info: Turn information
            ground_truth_answers: List of correct answers
            
        Returns:
            bool: True if retrieval contains the target answer
        """
        if not ground_truth_answers:
            return False
        
        search_result = turn_info.get('search_result', '')
        raw_response = turn_info.get('raw_server_response', {})
        
        # Create a mini interaction history for this turn
        turn_interaction_history = {
            'search_results': [search_result] if search_result else [],
            'raw_server_responses': [raw_response] if raw_response else []
        }
        
        # Use the existing retrieval correctness check
        return is_retrieval_correct_kg("", ground_truth_answers, turn_interaction_history)
    
    def _calculate_global_rewards(self, data_item, interaction_history: Dict) -> Dict[str, float]:
        """
        Calculate global rewards for the entire sequence.
        
        Global components:
        - Exact match (final answer correctness)
        - Retrieval quality (overall retrieval effectiveness)
        
        Note: Format scoring has been moved to turn-wise rewards.
        
        Args:
            data_item: Single sample data
            interaction_history: Sample interaction history
            
        Returns:
            Dict of global reward components
        """
        global_rewards = {}
        
        # Calculate raw scores
        exact_match_score_raw = self._calculate_exact_match_reward(data_item, interaction_history)
        retrieval_quality_score_raw = self._calculate_global_retrieval_quality_reward(data_item, interaction_history)
        
        # Apply OTC scaling if enabled
        otc_scaling_factor = 1.0  # Default: no scaling
        kg_turns_used = 0
        
        if self.otc_scaling:
            kg_turns_used = self._count_kg_query_turns(interaction_history)
            # Exponential OTC scaling: e at 0 turns, 1 at max_turns
            # Formula: e * exp(-1 * turns / max_turns) = e^(1 - turns/max_turns)
            import math
            if kg_turns_used >= self.max_turns:
                otc_scaling_factor = 1.0  # Minimum at max turns
            else:
                # Exponential decay from e to 1
                ratio = kg_turns_used / self.max_turns
                otc_scaling_factor = math.e ** (1 - ratio)
            
            # Apply OTC scaling to raw scores before weighting
            exact_match_score_scaled = exact_match_score_raw * otc_scaling_factor
            retrieval_quality_score_scaled = retrieval_quality_score_raw * otc_scaling_factor
        else:
            # No scaling
            exact_match_score_scaled = exact_match_score_raw
            retrieval_quality_score_scaled = retrieval_quality_score_raw
        
        # Apply weights to scaled scores
        global_rewards['exact_match'] = exact_match_score_scaled * self.global_weights['exact_match']
        global_rewards['retrieval_quality'] = retrieval_quality_score_scaled * self.global_weights['retrieval_quality']
        
        # Store raw scores and OTC scaling info for correct logging
        global_rewards['_raw_exact_match'] = exact_match_score_raw
        global_rewards['_raw_retrieval_quality'] = retrieval_quality_score_raw
        global_rewards['_otc_scaling_factor'] = otc_scaling_factor
        global_rewards['_kg_turns_used'] = kg_turns_used
        
        # Clean logging for global scores  
        if self.verbose and (exact_match_score_raw > 0 or retrieval_quality_score_raw > 0):
            print(f"[global_scores] [exact_match] {exact_match_score_raw:.3f} [retrieval_quality] {retrieval_quality_score_raw:.3f}")
        
        return global_rewards
    
    
    def _count_kg_query_turns(self, interaction_history: Dict) -> int:
        """
        Count the number of KG query turns used in the interaction.
        
        Args:
            interaction_history: Sample interaction history containing actions
            
        Returns:
            int: Number of KG query turns (count of 'kg-query' actions)
        """
        if not interaction_history:
            return 0
            
        actions = interaction_history.get('actions', [])
        kg_query_count = sum(1 for action in actions if action == 'kg-query')
        
        return kg_query_count
    
    
    def _extract_full_solution_text(self, data_item) -> str:
        """Extract full solution text (prompt + response) like the original kg_format.py."""
        try:
            # Get both prompt and response IDs
            prompt_ids = data_item.batch["prompts"].squeeze(0)
            response_ids = data_item.batch["responses"].squeeze(0)
            
            # Concatenate prompt and response
            full_sequence = torch.cat([prompt_ids, response_ids])
            
            # Decode the full sequence without skipping special tokens to preserve chat format
            full_solution_text = self.tokenizer.decode(full_sequence, skip_special_tokens=False)
            
            return full_solution_text
        except Exception as e:
            # Only log errors, not debug info                print(f"⚠️  Error extracting solution text: {e}")
            return ""
    
    
    
    
    
    
    def _calculate_exact_match_reward(self, data_item, interaction_history: Dict) -> float:
        """
        Calculate exact match reward for final answer with enhanced metrics.
        
        Args:
            data_item: Single sample data
            interaction_history: Sample interaction history
            
        Returns:
            float: Exact match reward (0-1)
        """
        # Extract full solution string like the original kg_format.py
        full_solution_text = self._extract_full_solution_text(data_item)
        if not full_solution_text:
            return 0.0
        
        # Extract assistant response like the original system
        assistant_response = extract_assistant_response(full_solution_text)
        if not assistant_response:
            return 0.0
        
        # Get ground truth answer directly like kg_format.py
        ground_truth_raw = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        if not ground_truth_raw:
            return 0.0
            
        # Parse ground truth structure - it can be dict with target_text or direct string/list
        ground_truth = self._parse_ground_truth(ground_truth_raw)
        if not ground_truth:
            return 0.0
        
        # Handle both single answer and list of answers
        if isinstance(ground_truth, str):
            ground_truth_answers = [ground_truth]
        elif isinstance(ground_truth, list):
            ground_truth_answers = ground_truth
        else:
            ground_truth_answers = [str(ground_truth)]
        
        # Extract the final answer from assistant response
        predicted_answer = extract_answer_kg(assistant_response)
        
        # Add logging for all KG evaluation examples
        # Determine dataset type from interaction history
        dataset_type = 'UNKNOWN'
        data_source = str(interaction_history.get('data_source', '')).lower()
        
        # Determine dataset type from data_source
        extra_info = interaction_history.get('extra_info', {})
        
        if 'multitq' in data_source:
            dataset_type = 'MultiTQ'
        elif 'cwq' in data_source:
            dataset_type = 'CWQ'
        elif 'webqsp' in data_source:
            dataset_type = 'WebQSP'
        elif 'simpleqa' in data_source:
            dataset_type = 'SimpleQA'
        elif 'trex' in data_source:
            dataset_type = 'T-REx'
        elif 'zero_shot_re' in data_source:
            dataset_type = 'ZeroShotRE'
        # Fallback: check extra_info if data_source didn't match
        elif extra_info and 'dataset_name' in extra_info and extra_info['dataset_name'] == 'multitq':
            dataset_type = 'MultiTQ'
            data_source = 'multitq'  # Update for consistency
        
        # Calculate EM score for display (no logging)
        # Pass interaction_history to enable MultiTQ temporal granularity handling
        em_match_result = self._check_exact_match(predicted_answer, ground_truth_answers, interaction_history=interaction_history, dataset_name=data_source)
        em_score = 1.0 if em_match_result else 0.0
        
        # Token-level enhanced metrics removed - using entity-level calculations only
        
        # Always calculate entity-level metrics for both exact match and F1
        # Pass interaction_history and dataset info to enable MultiTQ temporal handling
        entity_level_exact_match = self._calculate_entity_level_match(predicted_answer, ground_truth_answers, interaction_history=interaction_history, dataset_name=data_source)
        
        # Calculate proper entity-level F1, precision, and recall
        # Pass raw ground truth to properly handle multi-entity cases
        entity_metrics = self._calculate_entity_level_f1_precision_recall(predicted_answer, ground_truth_raw)
        entity_level_f1 = entity_metrics['f1']
        entity_level_precision = entity_metrics['precision']  
        entity_level_recall = entity_metrics['recall']
        
        # Store metrics for WandB logging - all entity-level
        if not hasattr(data_item, 'answer_metrics'):
            data_item.answer_metrics = {}
        
        data_item.answer_metrics.update({
            'exact_match_binary': float(entity_level_exact_match),
            'f1': float(entity_level_f1),
            'precision': float(entity_level_precision),
            'recall': float(entity_level_recall)
        })
        
        # Choose score based on answer score mode - both use entity-level calculation
        if self.answer_score_mode == 'binary':
            # Binary mode: use entity-level exact match score
            final_score = entity_level_exact_match
        elif self.answer_score_mode == 'f1':
            # F1 mode: use entity-level F1 score 
            final_score = entity_level_f1
        else:
            # Unknown mode, fallback to binary
            if self.verbose:
                print(f"[exact_match] Warning: unknown answer_score_mode '{self.answer_score_mode}', falling back to binary")
            final_score = entity_level_exact_match
        
        # Show enhanced metrics in verbose mode
        if self.verbose and (entity_level_exact_match > 0 or final_score > 0 or self.show_full_responses):
            print(f"\n[enhanced_exact_match_details] mode={self.answer_score_mode}")
            print(f"  [entity_exact_match] {entity_level_exact_match:.1f}")
            reward_source = 'entity_exact_match' if self.answer_score_mode == 'binary' else 'entity_f1'
            print(f"  [final_score] {final_score:.1f} (from {reward_source})")
            print(f"  [predicted] '{predicted_answer}'")
            print(f"  [expected] {ground_truth_answers[:2]}...")  # Show first 2
            if self.show_full_responses:
                print(f"\n[full_assistant_response]")
                print(assistant_response)
                print("=" * 80)
        
        return float(final_score)
    
    def _extract_final_answer(self, response_text: str) -> str:
        """
        Extract the final answer from the response text.
        
        Looks for content within <answer>...</answer> tags.
        
        Args:
            response_text: Full response text
            
        Returns:
            str: Extracted answer or empty string if not found
        """
        # Use the existing extract_answer_kg function
        extracted = extract_answer_kg(response_text)
        
        if extracted:
            return extracted.strip()
        
        # Fallback: look for answer patterns manually
        import re
        answer_pattern = re.compile(r'<answer>(.*?)</answer>', re.DOTALL | re.IGNORECASE)
        matches = answer_pattern.findall(response_text)
        
        if matches:
            # Take the last answer if multiple exist
            return matches[-1].strip()
        
        return ""
    
    def _check_exact_match(self, predicted_answer: str, ground_truth_answers: List[str], interaction_history: Dict = None, dataset_name: str = None) -> bool:
        """
        Check if predicted answer matches any ground truth answer.
        
        Uses the same normalization as the existing KG scoring system.
        
        Args:
            predicted_answer: The extracted predicted answer
            ground_truth_answers: List of acceptable ground truth answers
            interaction_history: Dictionary containing interaction data for dataset detection
            dataset_name: Optional explicit dataset name
            
        Returns:
            bool: True if there's an exact match
        """
        if not predicted_answer or not ground_truth_answers:
            return False
        
        # Use the existing em_check_kg function for consistency
        # Pass interaction_history and dataset_name to enable MultiTQ temporal handling
        return em_check_kg(predicted_answer, ground_truth_answers, dataset_name=dataset_name, verbose=False, interaction_history=interaction_history)
    
    def _normalize_answer_for_comparison(self, answer: str) -> str:
        """
        Normalize answer for comparison using the same logic as existing system.
        
        Args:
            answer: Raw answer string
            
        Returns:
            str: Normalized answer
        """
        return normalize_answer(answer)
    
    def _calculate_entity_level_match(self, predicted_answer: str, ground_truth_answers: List[str], interaction_history: Dict = None, dataset_name: str = None) -> float:
        """
        Calculate entity-level exact match score using entity segmentation.
        
        This approach uses the enhanced em_check_kg function which includes
        MultiTQ temporal granularity handling for year-only vs year-month questions.
        
        Args:
            predicted_answer: The extracted predicted answer
            ground_truth_answers: List of acceptable ground truth answers
            interaction_history: Dictionary containing interaction data for dataset detection
            dataset_name: Optional explicit dataset name
            
        Returns:
            float: 1.0 if match found (including MultiTQ temporal handling), 0.0 otherwise
        """
        if not predicted_answer or not ground_truth_answers:
            return 0.0
        
        # Use the enhanced em_check_kg function which includes MultiTQ temporal handling
        # This handles year-only vs year-month questions automatically for MultiTQ
        return float(em_check_kg(predicted_answer, ground_truth_answers, dataset_name=dataset_name, verbose=False, interaction_history=interaction_history))
    
    def _calculate_entity_level_f1_precision_recall(self, predicted_answer: str, ground_truth_raw) -> Dict[str, float]:
        """
        Calculate entity-level F1, precision, and recall for multi-entity answers.

        This method properly handles cases where ground truth contains multiple distinct entities,
        using set-based comparison after parsing comma-separated predictions.

        Args:
            predicted_answer: The extracted predicted answer (possibly comma-separated)
            ground_truth_raw: Raw ground truth data (can be dict with target_text and target_kb_id lists)

        Returns:
            Dict with f1, precision, and recall scores
        """
        if not predicted_answer:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Parse predicted answer into individual entities (split by comma)
        predicted_entities = [e.strip() for e in predicted_answer.split(',')]
        predicted_normalized_set = set()
        for pred_entity in predicted_entities:
            if pred_entity:
                normalized = normalize_answer(pred_entity)
                if normalized:
                    predicted_normalized_set.add(normalized)

        if not predicted_normalized_set:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Parse ground truth structure
        if isinstance(ground_truth_raw, dict):
            target_texts = ground_truth_raw.get("target_text", [])
            target_kb_ids = ground_truth_raw.get("target_kb_id", [])

            # Ensure both are lists
            if not isinstance(target_texts, list):
                target_texts = [target_texts] if target_texts else []
            if not isinstance(target_kb_ids, list):
                target_kb_ids = [target_kb_ids] if target_kb_ids else []
        else:
            # Fallback to original method for non-dict ground truth
            if isinstance(ground_truth_raw, str):
                ground_truth_answers = [ground_truth_raw]
            elif isinstance(ground_truth_raw, list):
                ground_truth_answers = ground_truth_raw
            else:
                ground_truth_answers = [str(ground_truth_raw)]

            # Parse ground truth into set
            normalized_gt_entities = set()
            for gt_answer in ground_truth_answers:
                if gt_answer:
                    normalized_gt = normalize_answer(gt_answer)
                    if normalized_gt:
                        normalized_gt_entities.add(normalized_gt)

            if not predicted_normalized_set and not normalized_gt_entities:
                return {'f1': 1.0, 'precision': 1.0, 'recall': 1.0}
            elif not predicted_normalized_set:
                return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}
            elif not normalized_gt_entities:
                return {'f1': 0.0, 'precision': 0.0, 'recall': 1.0}

            # Set-based comparison
            correct_entities = predicted_normalized_set & normalized_gt_entities
            num_correct = len(correct_entities)
            num_predicted = len(predicted_normalized_set)
            num_ground_truth = len(normalized_gt_entities)

            precision = num_correct / num_predicted if num_predicted > 0 else 0.0
            recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0

            if precision + recall == 0:
                f1 = 0.0
            else:
                f1 = 2 * (precision * recall) / (precision + recall)

            return {'f1': f1, 'precision': precision, 'recall': recall}

        # Build normalized ground truth entity set
        # For dict-based ground truth, each index is a distinct entity with possible text/kb_id variants
        num_entities = max(len(target_texts), len(target_kb_ids))
        if num_entities == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 1.0}

        # Build a set of all valid ground truth representations
        ground_truth_normalized_set = set()
        for i in range(num_entities):
            # Add text form
            if i < len(target_texts) and target_texts[i]:
                normalized_text = normalize_answer(str(target_texts[i]))
                if normalized_text:
                    ground_truth_normalized_set.add(normalized_text)

            # Add kb_id form
            if i < len(target_kb_ids) and target_kb_ids[i]:
                normalized_kb = normalize_answer(str(target_kb_ids[i]))
                if normalized_kb:
                    ground_truth_normalized_set.add(normalized_kb)

        # Set-based comparison: find intersection
        correct_entities = predicted_normalized_set & ground_truth_normalized_set
        num_correct = len(correct_entities)
        num_predicted = len(predicted_normalized_set)
        num_ground_truth = len(ground_truth_normalized_set)

        # Calculate proper set-based metrics
        precision = num_correct / num_predicted if num_predicted > 0 else 0.0
        recall = num_correct / num_ground_truth if num_ground_truth > 0 else 0.0

        # Calculate F1
        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        return {
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def _calculate_global_retrieval_quality_reward(self, data_item, interaction_history: Dict) -> float:
        """
        Calculate global retrieval quality reward across all turns.
        
        This evaluates the overall effectiveness of retrieval across the entire interaction,
        considering whether the target answer appears in any of the retrieved information.
        
        Args:
            data_item: Single sample data
            interaction_history: Sample interaction history
            
        Returns:
            float: Global retrieval quality reward (0-1)
        """
        # Get ground truth answer
        reward_model_data = data_item.non_tensor_batch.get("reward_model", {})
        ground_truth_raw = reward_model_data.get("ground_truth", "")
        
        if not ground_truth_raw:
            return 0.0
            
        # Parse ground truth structure
        ground_truth = self._parse_ground_truth(ground_truth_raw)
        
        if not ground_truth:
            return 0.0
        
        # Handle both single answer and list of answers
        if isinstance(ground_truth, str):
            ground_truth_answers = [ground_truth]
        elif isinstance(ground_truth, list):
            ground_truth_answers = ground_truth
        else:
            ground_truth_answers = [str(ground_truth)]
        
        # Check if any retrieval contains the correct answer
        retrieval_contains_answer = self._check_global_retrieval_contains_answer(
            interaction_history, ground_truth_answers
        )
        
        if retrieval_contains_answer:
            # Perfect score if answer was retrieved
            retrieval_score = 1.0
        else:
            # Binary scoring: no partial credit if answer not found
            retrieval_score = 0.0
        
        
        return retrieval_score
    
    def _parse_ground_truth(self, ground_truth_raw):
        """
        Parse ground truth data structure using the same logic as kg_format.py.
        Based on qa_em_format_kg.compute_score_em_kg_refactored function.
        
        Args:
            ground_truth_raw: Raw ground truth data (can be string, dict, or list)
            
        Returns:
            List of ground truth answer strings
        """
        try:
            # Handle stringified dictionaries first
            if isinstance(ground_truth_raw, str):
                # Try to parse as JSON in case it's a stringified dict
                if ground_truth_raw.startswith('{') and ground_truth_raw.endswith('}'):
                    import ast
                    try:
                        ground_truth_raw = ast.literal_eval(ground_truth_raw)
                    except:
                        # If parsing fails, treat as plain string
                        return [ground_truth_raw]
                else:
                    # Plain string
                    return [ground_truth_raw]
            
            # Now handle the parsed structure using the same logic as kg_format.py
            if isinstance(ground_truth_raw, dict):
                target_texts = ground_truth_raw.get("target_text", [])
                target_kb_ids = ground_truth_raw.get("target_kb_id", [])
                
                # Ensure both are lists
                if not isinstance(target_texts, list):
                    target_texts = [target_texts] if target_texts else []
                if not isinstance(target_kb_ids, list):
                    target_kb_ids = [target_kb_ids] if target_kb_ids else []
                
                # Combine both target_text and target_kb_id as potential answers
                ground_truth_answers = []
                ground_truth_answers.extend([str(text) for text in target_texts if text])
                ground_truth_answers.extend([str(kb_id) for kb_id in target_kb_ids if kb_id])
                
                # Remove duplicates while preserving order
                seen = set()
                ground_truth_answers = [x for x in ground_truth_answers if not (x in seen or seen.add(x))]
                
                # If no valid answers found, use first target_text as fallback
                if not ground_truth_answers and target_texts:
                    ground_truth_answers = [str(target_texts[0])]
                    
                return ground_truth_answers
                
            elif isinstance(ground_truth_raw, list):
                return [str(item) for item in ground_truth_raw if item]
            else:
                return [str(ground_truth_raw)]
                
        except Exception as e:
            return None
    
    def _check_global_retrieval_contains_answer(self, interaction_history: Dict, ground_truth_answers: List[str]) -> bool:
        """
        Check if any retrieval across all turns contains the target answer.
        
        Args:
            interaction_history: Sample interaction history
            ground_truth_answers: List of correct answers
            
        Returns:
            bool: True if any retrieval contains the target answer
        """
        # Use the existing retrieval correctness check from the utilities
        from verl.utils.reward_score.qa_em_format_kg import is_retrieval_correct_kg
        
        return is_retrieval_correct_kg("", ground_truth_answers, interaction_history)
    
    def _calculate_retrieval_effort_score(self, interaction_history: Dict) -> float:
        """
        Calculate a partial score based on retrieval effort and quality.
        
        This provides some reward even when the target answer wasn't retrieved,
        based on the quality and success rate of the retrieval attempts.
        
        Args:
            interaction_history: Sample interaction history
            
        Returns:
            float: Retrieval effort score (0-0.5)
        """
        raw_responses = interaction_history.get('raw_server_responses', [])
        search_results = interaction_history.get('search_results', [])
        
        if not raw_responses:
            return 0.0
        
        successful_retrievals = 0
        total_retrievals = 0
        quality_score = 0.0
        
        for i, response in enumerate(raw_responses):
            if isinstance(response, dict) and 'kg_metadata' in response:
                total_retrievals += 1
                kg_metadata = response.get('kg_metadata', {})
                success = kg_metadata.get('success', False)
                
                if success:
                    successful_retrievals += 1
                    # Add quality score based on search result content
                    if i < len(search_results):
                        search_result = search_results[i]
                        quality_score += self._evaluate_search_content_quality(search_result)
        
        if total_retrievals == 0:
            return 0.0
        
        # Calculate success rate
        success_rate = successful_retrievals / total_retrievals
        
        # Calculate average quality
        avg_quality = quality_score / max(successful_retrievals, 1)
        
        # Combine success rate and quality (max 0.5 to leave room for actual answer retrieval)
        effort_score = min((success_rate * 0.3 + avg_quality * 0.2), 0.5)
        
        return effort_score
    
    def _log_sample_detailed(self, data, detailed_rewards, data_source: str = "webqsp_kg"):
        """
        Log detailed sample information in kg_format.py style for debugging.
        Only logs the FIRST sample to avoid intermingled logging of multiple samples.
        """
        if len(data) > 0:
            sample_idx = 1
            if sample_idx < len(detailed_rewards):
                reward_dict = detailed_rewards[sample_idx]
            else:
                raise ValueError("No detailed rewards available for sample index 0")
            
            # Add sample separator for clarity
            print(f"\n{'='*80}")
            print(f"[KG Multi-Turn Format Reward Manager - {data_source}] - Sample {sample_idx}")
            print(f"{'='*80}")
            
            # Extract basic information
            data_item = data[sample_idx]
            sequences_str = self._extract_sequences_string(data_item)
            ground_truth = self._extract_ground_truth(data_item)
            
            print(f"[sequences] {sequences_str}")
            print()
            print(f"[ground_truth] {ground_truth}")

            # Print turn-wise rewards
            self._log_turn_rewards(reward_dict)

            # Print global rewards
            self._log_global_rewards(reward_dict, data_item)

            # Print final score calculation
            self._log_final_score_calculation(reward_dict)
            print(f"{'='*80}\n")

    def _log_turn_rewards(self, reward_dict: Dict):
        """Log turn-wise rewards with complete calculation flow demonstration."""
        turn_rewards = reward_dict.get('turn_rewards', {})
        turn_components = reward_dict.get('turn_components', {})

        if not turn_rewards:
            return

        print("\n[turn_rewards]")
        for turn_num in sorted(turn_rewards.keys()):
            turn_reward = turn_rewards[turn_num]
            components = turn_components.get(turn_num, {})

            # Extract raw component scores (before weight application)
            kg_validity_raw = components.get('kg_query_validity', 0.0)
            answer_validity_raw = components.get('is_answer_score', 0.0)
            format_score_raw = components.get('format_score', 0.0)
            action_type = components.get('action', 'unknown')
            extracted_content = components.get('extracted_content', '')

            # Show the complete calculation flow with weights
            print(f"  Turn {turn_num} ({action_type}):")

            # Step 1: Show raw component scores
            print(f"    [raw_components] format={format_score_raw:.1f}, kg_validity={kg_validity_raw:.1f}, answer_validity={answer_validity_raw:.1f}")

            # Step 2: Show weight application
            format_weight = self.turn_specific_weights['format_score']
            kg_weight = self.turn_specific_weights['kg_query_validity']
            answer_weight = self.turn_specific_weights['is_answer_score']

            format_weighted = format_score_raw * format_weight
            kg_weighted = kg_validity_raw * kg_weight
            answer_weighted = answer_validity_raw * answer_weight

            print(f"    [weights] format={format_weight}, kg_validity={kg_weight}, answer_validity={answer_weight}")
            print(f"    [weighted_scores] format={format_weighted:.3f}, kg_validity={kg_weighted:.3f}, answer_validity={answer_weighted:.3f}")

            # Step 3: Show final calculation based on action type
            if action_type == 'kg-query':
                calculated_total = format_weighted + kg_weighted
                print(f"    [calculation] {format_weighted:.3f} + {kg_weighted:.3f} = {calculated_total:.3f}")
            elif action_type == 'answer':
                calculated_total = format_weighted + answer_weighted
                print(f"    [calculation] {format_weighted:.3f} + {answer_weighted:.3f} = {calculated_total:.3f}")
            else:
                calculated_total = 0.0
                print(f"    [calculation] unknown_action = {calculated_total:.3f}")

            # Step 4: Validation
            calculation_correct = abs(turn_reward - calculated_total) < 0.001
            status = "✅ CORRECT" if calculation_correct else "❌ INCORRECT"
            print(f"    [final] expected={calculated_total:.3f}, actual={turn_reward:.3f} {status}")

            # Step 5: Format validation for debugging
            if extracted_content:
                import re
                if action_type == 'kg-query':
                    has_think = re.search(r'<think>.*?</think>', extracted_content, re.DOTALL) is not None
                    has_kg = re.search(r'<kg-query>.*?</kg-query>', extracted_content, re.DOTALL) is not None
                    expected_format_raw = 1.0 if (has_think and has_kg) else 0.0
                    format_validation = "✅" if abs(format_score_raw - expected_format_raw) < 0.001 else "❌"
                    print(f"    [format_check] has_think={has_think}, has_kg={has_kg}, expected={expected_format_raw:.1f}, actual={format_score_raw:.1f} {format_validation}")
                elif action_type == 'answer':
                    has_think = re.search(r'<think>.*?</think>', extracted_content, re.DOTALL) is not None
                    has_answer = re.search(r'<answer>.*?</answer>', extracted_content, re.DOTALL) is not None
                    expected_format_raw = 1.0 if (has_think and has_answer) else 0.0
                    format_validation = "✅" if abs(format_score_raw - expected_format_raw) < 0.001 else "❌"
                    print(f"    [format_check] has_think={has_think}, has_answer={has_answer}, expected={expected_format_raw:.1f}, actual={format_score_raw:.1f} {format_validation}")

            # Step 6: Show content for debugging if needed
            should_show_content = (
                not calculation_correct or
                format_score_raw == 0.0 or
                self.show_full_responses
            )

            if should_show_content and extracted_content:
                content_preview = extracted_content[:500] + ('...' if len(extracted_content) > 500 else '')
                print(f"    [extracted_content] {content_preview}")

            print()  # Add spacing between turns

    def _log_global_rewards(self, reward_dict: Dict, data_item=None):
        """Log global rewards with complete calculation flow demonstration."""
        global_rewards = reward_dict.get('global_rewards', {})

        if not global_rewards:
            return

        print("\n[global_rewards]")

        # Show the calculation flow for global rewards - use actual raw scores
        exact_match_raw = global_rewards.get('_raw_exact_match', 0.0)
        retrieval_quality_raw = global_rewards.get('_raw_retrieval_quality', 0.0)

        # Show both exact match binary and F1 scores if available
        if data_item and hasattr(data_item, 'answer_metrics'):
            exact_match_binary = data_item.answer_metrics.get('exact_match_binary', 0.0)
            f1_score = data_item.answer_metrics.get('f1', 0.0)
            print(f"  [raw_components] exact_match={exact_match_raw:.3f} (binary={exact_match_binary:.3f}, f1={f1_score:.3f}), retrieval_quality={retrieval_quality_raw:.3f}")
        else:
            print(f"  [raw_components] exact_match={exact_match_raw:.3f}, retrieval_quality={retrieval_quality_raw:.3f}")

        # Show OTC scaling information if enabled
        otc_scaling_factor = global_rewards.get('_otc_scaling_factor', 1.0)
        kg_turns_used = global_rewards.get('_kg_turns_used', 0)

        if self.otc_scaling:
            print(f"  [otc_scaling] kg_turns_used={kg_turns_used}, max_turns={self.max_turns}, scaling_factor={otc_scaling_factor:.3f}")
            print(f"  [scaled_components] exact_match={exact_match_raw * otc_scaling_factor:.3f}, retrieval_quality={retrieval_quality_raw * otc_scaling_factor:.3f}")

        # Show weights
        em_weight = self.global_weights['exact_match']
        rq_weight = self.global_weights['retrieval_quality']

        # Use actual weighted scores from the reward calculation
        em_weighted = global_rewards.get('exact_match', 0.0)
        rq_weighted = global_rewards.get('retrieval_quality', 0.0)
        total_global = em_weighted + rq_weighted

        # Include OTC scaling in weights display
        otc_weight = otc_scaling_factor if self.otc_scaling else 1.0
        print(f"  [weights] exact_match={em_weight}, retrieval_quality={rq_weight}, otc_scaling={otc_weight:.3f}")
        print(f"  [weighted_scores] exact_match={em_weighted:.3f}, retrieval_quality={rq_weighted:.3f}")

        # Show calculation with OTC scaling notation if enabled
        if self.otc_scaling and otc_scaling_factor != 1.0:
            # Calculate what the base weighted values would be without OTC scaling
            em_base = exact_match_raw * em_weight
            rq_base = retrieval_quality_raw * rq_weight
            base_total = em_base + rq_base
            print(f"  [calculation_with_otc] ({em_base:.3f} + {rq_base:.3f}) * {otc_scaling_factor:.3f} = {total_global:.3f}")
        else:
            print(f"  [calculation] {em_weighted:.3f} + {rq_weighted:.3f} = {total_global:.3f}")
        print(f"  [total_global] {total_global:.3f}")

    def _log_final_score_calculation(self, reward_dict: Dict):
        """Log the final total score calculation with validation."""
        turn_rewards = reward_dict.get('turn_rewards', {})
        global_rewards = reward_dict.get('global_rewards', {})
        total_score = reward_dict.get('total_score', 0.0)

        # Calculate expected totals (must match the actual calculation method)
        total_turn_score = sum(turn_rewards.values()) / len(turn_rewards) if turn_rewards else 0.0
        total_global_score = sum(v for k, v in global_rewards.items() if not k.startswith('_'))
        expected_total = total_turn_score + total_global_score

        # Validation
        calculation_correct = abs(total_score - expected_total) < 0.001
        status = "✅ CORRECT" if calculation_correct else "❌ INCORRECT"

        print(f"\n[final_calculation]")
        print(f"  [components] turn_total={total_turn_score:.3f}, global_total={total_global_score:.3f}")
        print(f"  [calculation] {total_turn_score:.3f} + {total_global_score:.3f} = {expected_total:.3f}")
        print(f"  [final] expected={expected_total:.3f}, actual={total_score:.3f} {status}")
    
    def _extract_sequences_string(self, data_item) -> str:
        """Extract sequences string like kg_format.py."""
        try:
            prompt_ids = data_item.batch["prompts"]
            response_ids = data_item.batch["responses"]
            attention_mask = data_item.batch["attention_mask"]
            
            prompt_length = prompt_ids.shape[-1]
            sample_attention_mask = attention_mask[0] if attention_mask.dim() > 1 else attention_mask
            
            valid_prompt_length = int(sample_attention_mask[:prompt_length].sum())
            valid_response_length = int(sample_attention_mask[prompt_length:].sum())
            
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            valid_response_ids = response_ids[:valid_response_length]
            
            sequences = torch.cat((valid_prompt_ids, valid_response_ids))
            if sequences.dtype != torch.long:
                sequences = sequences.long()
                
            sequences_str = self.tokenizer.decode(sequences, skip_special_tokens=False)
                
            return sequences_str
        except Exception as e:
            return f"[Error extracting sequences: {e}]"
    
    def _extract_ground_truth(self, data_item) -> str:
        """Extract ground truth like kg_format.py."""
        try:
            # Get ground truth directly from reward_model like kg_format.py
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            
            # Handle different ground truth formats
            if isinstance(ground_truth, (list, tuple)):
                if len(ground_truth) > 2:
                    return f"{ground_truth[:2]}..."
                return str(ground_truth)
            
            return str(ground_truth)
        except Exception as e:
            return f"[Error extracting ground_truth: {e}]"
    
    def _compute_wandb_metrics(self, detailed_rewards: List[Dict], uid_info=None) -> Dict[str, List]:
        """
        Compute metrics for WandB logging from detailed reward breakdowns.
        
        Args:
            detailed_rewards: List of reward dictionaries for each sample in batch
            
        Returns:
            Dict mapping metric names to lists of values for WandB logging
        """
        from collections import defaultdict
        
        wandb_metrics = defaultdict(list)
        
        for reward_dict in detailed_rewards:
            turn_components = reward_dict.get('turn_components', {})
            global_rewards = reward_dict.get('global_rewards', {})
            
            # Global rewards (already aggregated)
            exact_match = global_rewards.get('exact_match', 0.0)
            retrieval_quality = global_rewards.get('retrieval_quality', 0.0)
            
            # Use raw scores for logging instead of weighted versions
            exact_match_raw = global_rewards.get('_raw_exact_match', 0.0)
            retrieval_quality_raw = global_rewards.get('_raw_retrieval_quality', 0.0)
            
            # Log the traditional exact_match (binary) for compatibility
            wandb_metrics['exact_match'].append(exact_match_raw)
            wandb_metrics['retrieval_quality'].append(retrieval_quality_raw)
            
            # Log entity-level answer metrics if available
            answer_metrics = reward_dict.get('answer_metrics', {})
            if answer_metrics:
                wandb_metrics['exact_match_binary'].append(answer_metrics.get('exact_match_binary', 0.0))
                wandb_metrics['f1'].append(answer_metrics.get('f1', 0.0))
                wandb_metrics['precision'].append(answer_metrics.get('precision', 0.0))
                wandb_metrics['recall'].append(answer_metrics.get('recall', 0.0))
            else:
                # Fill with zeros if no answer metrics available
                for metric in ['exact_match_binary', 'f1', 'precision', 'recall']:
                    wandb_metrics[metric].append(0.0)
            
            # Turn-wise metrics using stored components
            if turn_components:
                # Extract components for each turn
                kg_query_validity_values = []
                is_answer_values = []
                format_score_values = []
                
                for turn_num, components in turn_components.items():
                    kg_query_validity_values.append(components.get('kg_query_validity', 0.0))
                    is_answer_values.append(components.get('is_answer_score', 0.0))
                    format_score_values.append(components.get('format_score', 0.0))
                
                # Turn-wise averages
                wandb_metrics['turn_kg_query_validity'].append(
                    sum(kg_query_validity_values) / len(kg_query_validity_values) if kg_query_validity_values else 0.0
                )
                
                wandb_metrics['turn_format_score'].append(
                    sum(format_score_values) / len(format_score_values) if format_score_values else 0.0
                )
                
                # Binary: 1 if any turn produced an answer, 0 otherwise
                wandb_metrics['turn_is_answer_score'].append(
                    1.0 if any(val > 0 for val in is_answer_values) else 0.0
                )
                
                # Additional aggregate metrics
                num_turns = len(turn_components)
                wandb_metrics['num_turns'].append(num_turns)
                
            else:
                # No turn components available
                wandb_metrics['turn_kg_query_validity'].append(0.0)
                wandb_metrics['turn_format_score'].append(0.0)
                wandb_metrics['turn_is_answer_score'].append(0.0)
                wandb_metrics['num_turns'].append(0)
            
            # Always add total score
            wandb_metrics['total_score'].append(reward_dict.get('total_score', 0.0))
        
        # Note: Pass@K metrics are now calculated in metric_utils.py as part of the main metrics pipeline
        
        return dict(wandb_metrics)
    
    
    def _debug_compare_extraction_methods(self, data_item, sample_interaction_history: Dict, turn_data: Dict):
        """
        Debug function to compare responses_str from interaction history with tensor-based extraction.
        
        Args:
            data_item: Data item with tensor information and response tokens
            sample_interaction_history: Single sample's interaction history
            turn_data: Parsed turn data for this sample
        """
        print(f"\n{'='*60}")
        print(f"{'='*60}")
        
        # Check if we have responses_str in interaction history
        if 'responses_str' not in sample_interaction_history:
            return
        
        responses_str_list = sample_interaction_history['responses_str']
        # Get full response text from tokens for comparison
        try:
            response_ids = data_item.batch["responses"]
            if response_ids.dim() > 1:
                response_ids = response_ids[0]  # Single sample processing
            full_response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        except Exception as e:
            return
        
        # Compare each turn
        for turn_num, turn_info in turn_data.items():
            turn_idx = turn_info['turn_idx']
            action = turn_info['action']
            
            print(f"\n--- Turn {turn_num} (idx={turn_idx}, action={action}) ---")
            
            # Method 1: Get from responses_str (server interaction)
            server_content = ""
            if turn_idx < len(responses_str_list):
                server_content = responses_str_list[turn_idx]
                print(f"[SERVER_INTERACTION] Length: {len(server_content)} chars")
                if server_content:
                    print(f"[SERVER_INTERACTION] Preview: {server_content[:10000]}...")
                else:
                    print(f"[SERVER_INTERACTION] ❌ Empty content")
            else:
                print(f"[SERVER_INTERACTION] ❌ Index {turn_idx} out of bounds (list length: {len(responses_str_list)})")
            
            # Method 2: Get from tensor-based extraction
            tensor_content = ""
            try:
                tensor_content = self._extract_turn_content_tensor_based(turn_info, data_item, full_response_text)
                print(f"[TENSOR_BASED] Length: {len(tensor_content)} chars")
                if tensor_content:
                    print(f"[TENSOR_BASED] Preview: {tensor_content[:10000]}...")
                else:
                    print(f"[TENSOR_BASED] ❌ Empty content")
            except Exception as e:
                print(f"[TENSOR_BASED] ❌ Extraction failed: {e}")
            
            # Compare the two methods
            if server_content and tensor_content:
                # Normalize for comparison (strip whitespace)
                server_normalized = server_content.strip()
                tensor_normalized = tensor_content.strip()
                
                if server_normalized == tensor_normalized:
                    print(f"[COMPARISON] ✅ EXACT MATCH")
                else:
                    print(f"[COMPARISON] ❌ MISMATCH DETECTED")
                    print(f"  Server length: {len(server_normalized)}")
                    print(f"  Tensor length: {len(tensor_normalized)}")
                    
                    # Show first difference
                    min_len = min(len(server_normalized), len(tensor_normalized))
                    for i in range(min_len):
                        if server_normalized[i] != tensor_normalized[i]:
                            print(f"  First diff at position {i}:")
                            print(f"    Server: '{server_normalized[max(0,i-10):i+10]}'")
                            print(f"    Tensor: '{tensor_normalized[max(0,i-10):i+10]}'")
                            break
                    
                    if len(server_normalized) != len(tensor_normalized):
                        print(f"  Length difference: {abs(len(server_normalized) - len(tensor_normalized))} chars")
            elif server_content and not tensor_content:
                print(f"[COMPARISON] ⚠️  Server has content, tensor extraction failed")
            elif not server_content and tensor_content:
                print(f"[COMPARISON] ⚠️  Tensor has content, server extraction failed") 
            else:
                print(f"[COMPARISON] ❌ Both methods failed to extract content")
        
        print(f"{'='*60}")
        print(f"{'='*60}\n")