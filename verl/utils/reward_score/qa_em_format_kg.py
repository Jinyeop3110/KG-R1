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

"""
KG-aware QA exact match with format scoring
Provides structured evaluation for KG-augmented generation
"""

import re
import string
import random
from typing import Dict, Any, Union, List, Tuple
from collections import defaultdict
import sys
import os
import hashlib
import json

# Add path to error types module
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../kg_r1/search'))
from error_types import KGErrorType


def normalize_temporal_multitq(text):
    """Normalize temporal expressions for MultiTQ, handling comma-separated lists"""
    import re
    from datetime import datetime
    
    # Check if this looks like a comma-separated list of temporal expressions
    if ',' in text and any(re.search(r'\d{4}', item.strip()) for item in text.split(',')):
        # Process each item in the comma-separated list
        items = [item.strip() for item in text.split(',')]
        normalized_items = []
        
        for item in items:
            if not item:
                continue
                
            # Skip normalization if already in YYYY-MM format
            if re.match(r'^\d{4}-\d{2}$', item):
                normalized_items.append(item)
                continue
            
            # Apply month name normalization to this item
            normalized_item = _normalize_single_temporal_item(item)
            normalized_items.append(normalized_item)
        
        return ', '.join(normalized_items)
    else:
        # Single item - use existing logic
        return _normalize_single_temporal_item(text)


def _normalize_single_temporal_item(text):
    """Normalize a single temporal expression"""
    import re
    
    # Skip normalization if already in YYYY-MM format
    if re.match(r'^\d{4}-\d{2}$', text):
        return text
    
    # Handle "July 2008" -> "2008-07"
    month_patterns = {
        'january': '01', 'february': '02', 'march': '03', 'april': '04',
        'may': '05', 'june': '06', 'july': '07', 'august': '08',
        'september': '09', 'october': '10', 'november': '11', 'december': '12'
    }
    
    for month_name, month_num in month_patterns.items():
        # "july 2008" -> "2008-07"
        pattern = rf'\b{month_name}\s+(\d{{4}})\b'
        text = re.sub(pattern, rf'\1-{month_num}', text, flags=re.IGNORECASE)
    
    return text


def _remove_temporal_annotations_multitq(text):
    """Remove temporal annotations like [YYYY-MM] from entity names for MultiTQ"""
    import re
    
    # Remove temporal brackets like [2008-11], [2012-07]
    # Pattern: [YYYY-MM] or [YYYY]
    text = re.sub(r'\s*\[\d{4}(?:-\d{2})?\]', '', text)
    
    # Clean up extra spaces and commas
    text = re.sub(r'\s*,\s*', ', ', text)  # Normalize comma spacing
    text = re.sub(r',\s*$', '', text)      # Remove trailing comma
    text = text.strip()
    
    return text


def normalize_answer(s, dataset_name=None):
    """Normalize answer for comparison"""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    # Check for MultiTQ temporal normalization before standard processing
    if dataset_name and dataset_name.lower() in ['multitq', 'kgr1_multitq']:
        # First remove temporal annotations like [2008-07] from entity names
        text_without_annotations = _remove_temporal_annotations_multitq(s)
        
        # Then apply temporal normalization on the cleaned text
        temporal_normalized = normalize_temporal_multitq(lower(text_without_annotations))
        
        # If temporal normalization produced YYYY-MM format(s), use as-is (don't remove hyphens)
        # Check for single YYYY-MM or comma-separated YYYY-MM, YYYY-MM pattern
        if (re.match(r'^\d{4}-\d{2}$', temporal_normalized) or  # Single: "2008-07"
            re.match(r'^\d{4}-\d{2}(?:,\s*\d{4}-\d{2})*$', temporal_normalized)):  # List: "2008-07, 2009-03"
            return temporal_normalized
        else:
            # No temporal conversion happened, proceed with standard normalization on cleaned text
            return white_space_fix(remove_articles(remove_punc(lower(text_without_annotations))))
    else:
        # Standard normalization for non-MultiTQ datasets
        return white_space_fix(remove_articles(remove_punc(lower(s))))


def extract_query_identifier(response: Dict) -> str:
    """Extract a unique identifier for the query from the response.
    
    Args:
        response: KG server response containing query information
        
    Returns:
        Unique string identifier for the query
    """
    # Option 1: Use the request_payload if available (most reliable)
    request_payload = response.get('request_payload', {})
    if request_payload:
        # Create a normalized identifier from the request parameters
        action_type = request_payload.get('action_type', '')
        entity_id = request_payload.get('entity_id', '')
        relation = request_payload.get('relation', '')
        sample_id = request_payload.get('sample_id', '')
        dataset_name = request_payload.get('dataset_name', '')
        
        # Create a consistent identifier
        identifier_parts = [
            action_type.lower().strip(),
            entity_id.lower().strip(),
            relation.lower().strip() if relation else '',
            sample_id.lower().strip(),
            dataset_name.lower().strip()
        ]
        
        # Join parts with a delimiter, removing empty parts
        identifier = '|'.join(part for part in identifier_parts if part)
        return identifier
    
    # Option 2: Use query text if available (fallback)
    query_text = response.get('query_text', response.get('query', ''))
    if query_text:
        return normalize_query_text(query_text)
    
    # Option 3: Use content from the response (last resort)
    content = response.get('content', '')
    if content:
        # Hash the content to create a unique identifier
        content_hash = hashlib.md5(str(content).encode()).hexdigest()
        return f"content_hash:{content_hash}"
    
    # Option 4: Use entire response as fallback
    response_str = json.dumps(response, sort_keys=True)
    response_hash = hashlib.md5(response_str.encode()).hexdigest()
    return f"response_hash:{response_hash}"


def normalize_query_text(query_text: str) -> str:
    """Normalize query text to handle minor variations.
    
    Args:
        query_text: Raw query text
        
    Returns:
        Normalized query text
    """
    # Remove extra whitespace, convert to lowercase
    normalized = ' '.join(query_text.lower().split())
    # Remove common punctuation variations
    normalized = re.sub(r'[^\w\s]', '', normalized)
    normalized = re.sub(r'\s+', ' ', normalized)
    return normalized.strip()


def extract_kg_query_stats(solution_str: str, interaction_history: Dict) -> Dict[str, Any]:
    """Extract KG query statistics with unique query counting to prevent reward hacking.
    
    Uses actual server response data from interaction history. Only counts responses that
    contain 'kg_metadata' field, which indicates an actual KG query was made. This filters
    out empty dicts from answer actions and other non-KG responses.
    
    Args:
        solution_str: The solution text (unused, kept for compatibility)
        interaction_history: Dict containing actual query results and error information (required)
    
    Returns:
        Dict containing query statistics: total_queries, valid_queries, invalid_queries, total_errors, error_counts, unique_valid_queries
    """
    # Initialize error counts with all possible error types
    error_counts = {error_type: 0 for error_type in KGErrorType.get_all_types()}
    
    # Track unique queries to prevent reward hacking
    unique_valid_queries = set()
    unique_invalid_queries = set()
    all_query_identifiers = []  # For debugging
    
    stats = {
        'total_queries': 0,
        'valid_queries': 0,
        'invalid_queries': 0,
        'total_errors': 0,
        'error_counts': error_counts,
        'unique_valid_queries': 0,
        'unique_invalid_queries': 0,
        'duplicate_valid_queries': 0,
        'duplicate_invalid_queries': 0
    }
    
    # Use actual server responses with error types
    raw_responses = interaction_history.get('raw_server_responses', [])
    
    total_queries = 0
    valid_queries = 0
    invalid_queries = 0
    
    for response in raw_responses:
        # Only count responses that are dicts with kg_metadata (actual KG queries)
        if isinstance(response, dict) and 'kg_metadata' in response:
            kg_metadata = response['kg_metadata']
            error_type = kg_metadata.get('error_type')
            success = kg_metadata.get('success', False)
            
            # Extract unique identifier for this query
            query_id = extract_query_identifier(response)
            all_query_identifiers.append(query_id)
            
            total_queries += 1
            
            if success and error_type == KGErrorType.SUCCESS:
                valid_queries += 1
                unique_valid_queries.add(query_id)
                stats['error_counts'][KGErrorType.SUCCESS] += 1
            else:
                invalid_queries += 1
                unique_invalid_queries.add(query_id)
                if error_type:
                    stats['error_counts'][error_type] += 1
                    stats['total_errors'] += 1
    
    # Calculate unique and duplicate counts
    unique_valid_count = len(unique_valid_queries)
    unique_invalid_count = len(unique_invalid_queries)
    
    stats['total_queries'] = total_queries
    stats['valid_queries'] = valid_queries
    stats['invalid_queries'] = invalid_queries
    stats['unique_valid_queries'] = unique_valid_count
    stats['unique_invalid_queries'] = unique_invalid_count
    stats['duplicate_valid_queries'] = valid_queries - unique_valid_count
    stats['duplicate_invalid_queries'] = invalid_queries - unique_invalid_count
    
    return stats


def is_year_only_question(ground_truth_answers: List[str]) -> bool:
    """Check if this is a year-only question (vs year-month) for MultiTQ temporal granularity."""
    if not ground_truth_answers:
        return False
    
    # Check if all ground truth answers are just years (YYYY format)
    year_pattern = re.compile(r'^\d{4}$')
    return all(year_pattern.match(str(answer).strip()) for answer in ground_truth_answers)

def extract_year_from_timestamps(extracted_answer: str) -> str:
    """Extract unique years from timestamp list for year-only questions."""
    if not extracted_answer:
        return extracted_answer
    
    # Find all year patterns (YYYY) in the answer
    year_pattern = re.compile(r'\b(\d{4})\b')
    years = year_pattern.findall(extracted_answer)
    
    if not years:
        return extracted_answer
    
    # Get unique years and sort them
    unique_years = sorted(set(years))
    
    # Return comma-separated if multiple years, single year if one
    if len(unique_years) == 1:
        return unique_years[0]
    else:
        return ', '.join(unique_years)

def em_check_kg(prediction: str, golden_answers: Union[str, List[str]], dataset_name: str = None, verbose: bool = False, interaction_history: Dict = None) -> float:
    """KG-aware exact match check with enhanced answer extraction and list handling.
    
    This function expects pre-filtered assistant response content.
    
    Args:
        prediction: The prediction text from the model
        golden_answers: Ground truth answer(s) 
        dataset_name: Dataset name for dataset-specific normalization (e.g., "multitq")
        verbose: If True, print detailed normalization steps
    """
    if isinstance(golden_answers, str):
        golden_answers = [golden_answers]
    
    # Extract answer from prediction using KG-aware extraction
    # Note: prediction should already be filtered to assistant response only
    extracted_answer = extract_answer_kg(prediction)
    if extracted_answer is None:
        if verbose:
            print(f"[EM-DEBUG] No answer tags found in prediction")
        return 0
    
    # Normalize all ground truth answers
    normalized_golden_answers = [normalize_answer(ga, dataset_name) for ga in golden_answers if ga]
    
    # First try exact match with the full extracted answer
    normalized_prediction = normalize_answer(extracted_answer, dataset_name)
    
    # MultiTQ-specific temporal granularity handling
    multitq_year_processed = False
    is_multitq = (
        (dataset_name and 'multitq' in dataset_name.lower()) or
        (interaction_history and 'multitq' in str(interaction_history.get('data_source', '')).lower())
    )
    
    # Debug logging for MultiTQ detection
    if verbose or True:  # Always show for debugging
        print(f"[DEBUG-MultiTQ] dataset_name: '{dataset_name}'")
        print(f"[DEBUG-MultiTQ] interaction_history data_source: '{interaction_history.get('data_source', '') if interaction_history else 'NO_HISTORY'}'")
        print(f"[DEBUG-MultiTQ] is_multitq: {is_multitq}")
        print(f"[DEBUG-MultiTQ] ground_truth: {golden_answers}")
    
    if is_multitq:
        is_year_only = is_year_only_question(golden_answers)
        if is_year_only:
            # For year-only questions, extract years from timestamps
            year_extracted = extract_year_from_timestamps(extracted_answer)
            year_normalized = normalize_answer(year_extracted, dataset_name)
            multitq_year_processed = True
            
            if verbose:
                print(f"[MultiTQ-TEMPORAL] ===== YEAR-ONLY QUESTION DETECTED =====")
                print(f"[MultiTQ-TEMPORAL] Original extracted: '{extracted_answer}'")
                print(f"[MultiTQ-TEMPORAL] Year-extracted: '{year_extracted}'")
                print(f"[MultiTQ-TEMPORAL] Year-normalized: '{year_normalized}'")
                print(f"[MultiTQ-TEMPORAL] Ground truth (year): {golden_answers}")
            
            # Check year-based match first
            if year_normalized in normalized_golden_answers:
                if verbose:
                    print(f"[MultiTQ-TEMPORAL] ✅ YEAR-BASED MATCH: '{year_normalized}' in {normalized_golden_answers}")
                return 1.0
            
            # Update the normalized prediction for further processing
            normalized_prediction = year_normalized
    
    if verbose:
        print(f"[EM-DEBUG] ===== DETAILED EM CALCULATION =====")
        print(f"[EM-DEBUG] Raw extracted answer: '{extracted_answer}'")
        print(f"[EM-DEBUG] Normalized extracted: '{normalized_prediction}'")
        print(f"[EM-DEBUG] Raw ground truth: {golden_answers}")
        print(f"[EM-DEBUG] Normalized ground truth: {normalized_golden_answers}")
        print(f"[EM-DEBUG] Dataset name: {dataset_name}")
        if multitq_year_processed:
            print(f"[EM-DEBUG] MultiTQ year processing: APPLIED")
    
    # Regular exact match check (this may be redundant if year processing found match)
    if normalized_prediction in normalized_golden_answers:
        if verbose:
            print(f"[EM-DEBUG] ✅ EXACT MATCH FOUND: '{normalized_prediction}' in {normalized_golden_answers}")
        return 1.0
    
    # If no exact match, check if the extracted answer is a comma-separated list
    # Split by common delimiters and check for partial matches
    delimiters = [',', ';', ' and ', ' & ', '\n', '|']
    predicted_items = [extracted_answer]  # Start with the full answer
    
    # Split by each delimiter
    for delimiter in delimiters:
        if delimiter in extracted_answer:
            predicted_items = []
            for item in extracted_answer.split(delimiter):
                cleaned_item = item.strip()
                if cleaned_item:
                    predicted_items.append(cleaned_item)
            break  # Use the first delimiter that produces a split
    
    # Check for partial matches - if any predicted item matches ground truth, return 1.0
    if len(predicted_items) > 1:  # Only do partial matching if we have multiple items
        if verbose:
            print(f"[EM-DEBUG] Checking list items: {predicted_items}")
        
        for predicted_item in predicted_items:
            normalized_predicted_item = normalize_answer(predicted_item, dataset_name)
            if verbose:
                print(f"[EM-DEBUG] Item '{predicted_item}' -> '{normalized_predicted_item}'")
            
            if normalized_predicted_item in normalized_golden_answers:
                if verbose:
                    print(f"[EM-DEBUG] ✅ PARTIAL MATCH FOUND: '{normalized_predicted_item}' in {normalized_golden_answers}")
                return 1.0  # Any match in a list gets full credit
    
    if verbose:
        print(f"[EM-DEBUG] ❌ NO MATCH: '{normalized_prediction}' not in {normalized_golden_answers}")
        print(f"[EM-DEBUG] ========================================")
    
    return 0


def is_valid_kg_sequence(text: str) -> Tuple[bool, str]:
    """Check if the sequence contains valid KG reasoning structure.
    
    This function expects pre-filtered assistant response content.
    Focuses purely on tag structure validation.
    """
    # Since the text is already filtered to assistant response, use it directly
    content = text
    
    # Remove hint section before validation
    hint_section_pattern = r'\[Hint\]:.*?ERROR HANDLING:.*?(?=\n|$)'
    content = re.sub(hint_section_pattern, '', content, flags=re.DOTALL | re.IGNORECASE)
    
    # Check for balanced tags
    tags_to_check = ["think", "kg-query", "information", "answer"]
    for tag in tags_to_check:
        opening_count = len(re.findall(f"<{tag}>", content))
        closing_count = len(re.findall(f"</{tag}>", content))
        if opening_count != closing_count:
            return False, f"Mismatch in {tag} tags: {opening_count} opening vs {closing_count} closing tags"
    
    # Now check for proper sequence pattern and no extraneous content
    
    # 1. First split the content by any tags we recognize
    split_pattern = r"(</?(?:think|kg-query|information|answer)>)"
    parts = re.split(split_pattern, content)
    
    # 2. Keep track of the current position in the expected sequence
    state = "start"  # start -> think -> kg-query -> information -> think -> ... -> answer -> end
    seen_answer = False
    
    # 3. Check each part
    for i, part in enumerate(parts):
        # Skip empty parts
        if not part.strip():
            continue
            
        # Check if this is a tag
        if re.match(r"</?(?:think|kg-query|information|answer)>", part):
            # This is a tag, check if it's valid in the current state
            if part == "<think>" and state in ["start", "information"]:
                state = "in_think"
            elif part == "</think>" and state == "in_think":
                state = "after_think"
            elif part == "<kg-query>" and state == "after_think":
                state = "in_kg_query"
            elif part == "</kg-query>" and state == "in_kg_query":
                state = "after_kg_query"
            elif part == "<information>" and state == "after_kg_query":
                state = "in_information"
            elif part == "</information>" and state == "in_information":
                state = "information"
            elif part == "<answer>" and state in ["after_think"]:
                state = "in_answer"
                seen_answer = True
            elif part == "</answer>" and state == "in_answer":
                state = "end"
            else:
                return False, f"Unexpected tag {part} in state {state}"
        else:
            # This is content, check if it's valid in the current state
            if state in ["in_think", "in_kg_query", "in_information", "in_answer"]:
                # Content is allowed inside tags
                pass
            elif state in ["start", "after_think", "after_kg_query", "information"]:
                # Only whitespace is allowed between tags
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' between tags (state: {state})"
            elif state == "end":
                # After answer tag, only whitespace is allowed
                if part.strip():
                    return False, f"Unexpected content '{part.strip()}' after answer tag"
            else:
                return False, f"Unexpected content in state {state}"
    
    # Check final state - we need at least an answer tag to be valid
    if not seen_answer:
        return False, "No answer tags found"
        
    if state != "end":
        return False, f"Incomplete sequence, ended in state {state}"
        
    return True, "Valid sequence format"

def extract_answer_kg(solution_str: str) -> Union[str, None]:
    """Extract the answer from the solution string.
    
    Unlike the general case that requires 2+ answer tags, 
    KG responses should have exactly one answer tag.
    This function expects pre-filtered assistant response content.
    """
    # First, remove all <information>...</information> content
    # This prevents error messages from contaminating answer extraction
    cleaned_str = re.sub(r'<information>.*?</information>', '', solution_str, flags=re.DOTALL)
    
    # Now search for answer tags in the cleaned string
    answer_pattern = r'<answer>(.*?)</answer>'
    matches = re.findall(answer_pattern, cleaned_str, re.DOTALL)
    
    if len(matches) == 0:
        return None
    
    # For KG responses, we expect exactly one answer tag
    # If there are multiple, take the first one
    return matches[0]

def is_retrieval_correct_kg(text: str, golden_answers: List[str], interaction_history: Dict, dataset_name: str = None) -> bool:
    """Check if the retrieval contains the correct answer using actual retrieval results.
    
    Args:
        text: The assistant response text (unused, kept for compatibility)
        golden_answers: List of correct answers to check for
        interaction_history: Dict containing actual retrieval results (required)
        dataset_name: Dataset name for dataset-specific normalization (e.g., "multitq")
    """
    # Check search_results (formatted observations shown to LLM)
    if 'search_results' in interaction_history:
        for result in interaction_history['search_results']:
            if result and result.strip():
                # Only check the part after ":" in the retrieval result
                if ':' in result:
                    answer_part = result.split(':', 1)[1].strip()
                else:
                    answer_part = result.strip()
                
                # Check if any golden answer appears in the answer part
                normalized_result = normalize_answer(answer_part, dataset_name)
                for golden_answer in golden_answers:
                    if normalize_answer(golden_answer, dataset_name) in normalized_result:
                        return True
    
    # Also check raw_server_responses for successful retrievals
    if 'raw_server_responses' in interaction_history:
        for response in interaction_history['raw_server_responses']:
            if isinstance(response, dict):
                kg_metadata = response.get('kg_metadata', {})
                success = kg_metadata.get('success', False)
                error_type = kg_metadata.get('error_type')
                
                # Only check successful retrievals
                if success and error_type == KGErrorType.SUCCESS:
                    choices = response.get('choices', [])
                    if choices and choices[0].get('message', {}).get('content'):
                        content = choices[0]['message']['content']
                        
                        # Only check the part after ":" in the response content
                        if ':' in content:
                            answer_part = content.split(':', 1)[1].strip()
                        else:
                            answer_part = content.strip()
                        
                        normalized_content = normalize_answer(answer_part, dataset_name)
                        for golden_answer in golden_answers:
                            if normalize_answer(golden_answer, dataset_name) in normalized_content:
                                return True
    return False


def extract_assistant_response(solution_str: str) -> str:
    """Extract only the assistant's response from the full solution string.
    
    This filters out the prompt to avoid contamination from example answer tags.
    Used consistently across all scoring functions.
    Supports both ChatML format (<|im_start|>assistant) and LLaMA format ([INST]...[/INST]).
    """
    # Try ChatML format first (Qwen, GPT, etc.)
    assistant_markers = []
    start_pos = 0
    while True:
        pos = solution_str.find('<|im_start|>assistant', start_pos)
        if pos == -1:
            break
        assistant_markers.append(pos)
        start_pos = pos + 1
    
    if assistant_markers:
        # Use the last assistant marker (most recent response)
        last_assistant_pos = assistant_markers[-1]
        assistant_pattern = r"<\|im_start\|>assistant\s*"
        assistant_match = re.search(assistant_pattern, solution_str[last_assistant_pos:])
        
        if assistant_match:
            # Extract content after the assistant marker
            start_pos = last_assistant_pos + assistant_match.end()
            assistant_content = solution_str[start_pos:]
            
            # Find the end of this assistant response (next <|im_end|> or end of string)
            end_match = re.search(r'<\|im_end\|>', assistant_content)
            if end_match:
                assistant_content = assistant_content[:end_match.start()]
            
            return assistant_content.strip()
    
    # Try LLaMA format ([INST]...[/INST] response)
    llama_pattern = r'\[/INST\]\s*(.*?)(?:\s*</s>|$)'
    llama_matches = re.findall(llama_pattern, solution_str, re.DOTALL)
    
    if llama_matches:
        # Use the last match (most recent response)
        return llama_matches[-1].strip()
    
    # Fallback: if no format markers found, return the full string
    # This handles cases where the response might be pre-filtered
    return solution_str.strip()


# --- REFACTORED SCORING LOGIC ---

def _calculate_base_score(
    em_score: float, 
    is_valid_format: bool, 
    has_answer_tags: bool,
    rewards: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculates the base score from answer correctness and format validity.
    
    Args:
        em_score: 1.0 if answer is correct, 0.0 if incorrect
        is_valid_format: Whether the response has proper <think>...<answer> structure
        has_answer_tags: Whether the response has <answer> tags (even if wrong structure)
        rewards: Dictionary of reward values
    
    Returns:
        Dictionary with base_score and component breakdown
    """
    if em_score > 0:
        # Correct answer gets the main reward. Add a bonus for perfect format.
        base_score = rewards['answer_correct'] + (rewards['format_bonus'] if is_valid_format else 0.0)
        return {
            'base_score': base_score,
            'answer_component': rewards['answer_correct'],
            'format_component': rewards['format_bonus'] if is_valid_format else 0.0,
            'answer_tag_component': 0.0
        }
    
    if is_valid_format:
        # Wrong answer but perfect format gets the format bonus.
        return {
            'base_score': rewards['format_bonus'],
            'answer_component': 0.0,
            'format_component': rewards['format_bonus'],
            'answer_tag_component': 0.0
        }
        
    if has_answer_tags:
        # Wrong answer and invalid structure, but at least it tried with an <answer> tag.
        return {
            'base_score': rewards['answer_tag_bonus'],
            'answer_component': 0.0,
            'format_component': 0.0,
            'answer_tag_component': rewards['answer_tag_bonus']
        }
        
    # Failed on all fronts.
    return {
        'base_score': 0.0,
        'answer_component': 0.0,
        'format_component': 0.0,
        'answer_tag_component': 0.0
    }


def _calculate_kg_interaction_score(
    kg_stats: Dict[str, Any], 
    rewards: Dict[str, float]
) -> Dict[str, Any]:
    """
    Calculates the score adjustment based on KG query performance.
    
    Args:
        kg_stats: Dictionary with KG query statistics including unique query metrics
        rewards: Dictionary of reward/penalty values
    
    Returns:
        Dictionary with kg_interaction_score and component breakdown
    """
    score = 0.0
    error_counts = kg_stats['error_counts']
    
    # Positive reward for unique, successful queries (prevents reward hacking)
    unique_valid_reward = kg_stats['unique_valid_queries'] * rewards['valid_query_reward']
    score += unique_valid_reward
    
    # Penalties for different categories of errors
    major_error_penalty = error_counts.get(KGErrorType.FORMAT_ERROR, 0) * rewards['major_error_penalty']
    score += major_error_penalty
    
    minor_error_penalty = (
        error_counts.get(KGErrorType.ENTITY_NOT_FOUND, 0) +
        error_counts.get(KGErrorType.RELATION_NOT_FOUND, 0) +
        error_counts.get(KGErrorType.SAMPLE_NOT_FOUND, 0)
    ) * rewards['minor_error_penalty']
    score += minor_error_penalty
    
    # Small penalty for queries that are valid but yield no useful data
    no_data_penalty = error_counts.get(KGErrorType.NO_RESULTS, 0) * rewards['no_data_penalty']
    score += no_data_penalty
    
    # Penalty for transient server errors
    server_error_penalty = error_counts.get(KGErrorType.SERVER_ERROR, 0) * rewards['server_error_penalty']
    score += server_error_penalty

    return {
        'kg_interaction_score': score,
        'unique_valid_reward': unique_valid_reward,
        'major_error_penalty': major_error_penalty,
        'minor_error_penalty': minor_error_penalty,
        'no_data_penalty': no_data_penalty,
        'server_error_penalty': server_error_penalty
    }


def compute_score_em_kg_refactored(
    solution_str: str, 
    ground_truth: Union[str, Dict[str, Any]], 
    interaction_history: Dict,
    # Backward compatibility parameters
    valid_query_reward: float = 0.05,
    answer_match_score: float = 0.8,
    structure_format_score: float = 0.2,
    final_format_score: float = 0.1,
    retrieval_score: float = 0.1,
    kg_server_error_penalty: float = -0.05,
    kg_not_found_penalty: float = -0.1,
    kg_format_error_penalty: float = -0.2,
    kg_no_data_penalty: float = -0.02,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Compute KG-aware exact match score with a simplified and clearer reward structure.
    
    Args:
        solution_str: The model's solution string.
        ground_truth: Ground truth answer(s) or dict with target info.
        interaction_history: Dict containing actual server interaction details.
        valid_query_reward: Reward for each *unique* successful KG query
        answer_match_score: Main reward for getting the answer right
        structure_format_score: Bonus for perfect <think>...<answer> structure
        final_format_score: Consolation prize for using <answer> tags correctly
        retrieval_score: Bonus if the right answer was in the KG results
        kg_server_error_penalty: Penalty for transient KG server issues
        kg_not_found_penalty: Penalty for "not found" errors
        kg_format_error_penalty: Penalty for syntax/format errors in a query
        kg_no_data_penalty: Penalty for valid queries that return nothing
        verbose: If True, prints debugging information.
    
    Returns:
        Dictionary with detailed scoring information.
    """
    # Define rewards and penalties in a single, manageable dictionary.
    # This makes tuning much easier and maintains backward compatibility.
    rewards = {
        'answer_correct': answer_match_score,        # Main reward for getting the answer right
        'format_bonus': structure_format_score,      # Bonus for perfect <think>...<answer> structure
        'retrieval_bonus': retrieval_score,          # Bonus if the right answer was in the KG results
        'answer_tag_bonus': final_format_score,      # Consolation prize for using <answer> tags correctly
        'valid_query_reward': valid_query_reward,    # Reward for each *unique* successful KG query
        'major_error_penalty': kg_format_error_penalty,   # Penalty for syntax/format errors in a query
        'minor_error_penalty': kg_not_found_penalty,      # Penalty for "not found" errors
        'no_data_penalty': kg_no_data_penalty,            # Penalty for valid queries that return nothing
        'server_error_penalty': kg_server_error_penalty,  # Penalty for transient KG server issues
    }
    
    # 1. Pre-processing: Isolate the assistant's response and parse ground truth
    assistant_response = extract_assistant_response(solution_str)
    
    # Extract dataset information for temporal normalization
    dataset_name = None
    if isinstance(ground_truth, dict):
        # Try to get dataset_name from extra_info field first
        extra_info = ground_truth.get("extra_info", {})
        if isinstance(extra_info, dict):
            dataset_name = extra_info.get("dataset_name")
        # Fallback to direct dataset_name field if not in extra_info
        if not dataset_name:
            dataset_name = ground_truth.get("dataset_name")
    
    # Parse ground truth (preserve existing logic)
    if isinstance(ground_truth, dict):
        target_texts = ground_truth.get("target_text", [])
        target_kb_ids = ground_truth.get("target_kb_id", [])
        
        # Ensure both are lists
        if not isinstance(target_texts, list):
            target_texts = [target_texts] if target_texts else []
        if not isinstance(target_kb_ids, list):
            target_kb_ids = [target_kb_ids] if target_kb_ids else []
        
        # Combine both target_text and target_kb_id as potential answers
        ground_truth_answers = []
        
        # Process target_texts with parsing for malformed dict strings
        for text in target_texts:
            if text:
                text_str = str(text)
                # Check if it's a malformed dict string like "{'text': '2010', 'kb_id': '2010'}"
                if text_str.startswith("{'") and text_str.endswith("'}") and 'text' in text_str:
                    try:
                        import ast
                        parsed_dict = ast.literal_eval(text_str)
                        if isinstance(parsed_dict, dict) and 'text' in parsed_dict:
                            # Extract the actual answer from the malformed format
                            ground_truth_answers.append(str(parsed_dict['text']))
                            print(f"[FIXED-GROUND-TRUTH] Parsed malformed dict: {text_str} -> {parsed_dict['text']}")
                        else:
                            ground_truth_answers.append(text_str)
                    except Exception as e:
                        print(f"[PARSE-ERROR] Failed to parse dict string: {text_str}, error: {e}")
                        ground_truth_answers.append(text_str)
                else:
                    ground_truth_answers.append(text_str)
        
        ground_truth_answers.extend([str(kb_id) for kb_id in target_kb_ids if kb_id])
        
        # Remove duplicates while preserving order
        seen = set()
        ground_truth_answers = [x for x in ground_truth_answers if not (x in seen or seen.add(x))]
        
        # If no valid answers found, use first target_text as fallback
        if not ground_truth_answers and target_texts:
            ground_truth_answers = [str(target_texts[0])]
    elif isinstance(ground_truth, list):
        ground_truth_answers = [str(item) for item in ground_truth if item]
    else:
        ground_truth_answers = [str(ground_truth)]
    
    # 2. Evaluate different aspects of the response
    em_score = em_check_kg(assistant_response, ground_truth_answers, dataset_name, verbose=True, interaction_history=interaction_history)
    is_valid_format, format_msg = is_valid_kg_sequence(assistant_response)
    extracted_answer = extract_answer_kg(assistant_response)
    has_answer_tags = extracted_answer is not None
    
    # Add logging for all KG evaluation examples
    # Determine dataset name from available data
    dataset_type = 'UNKNOWN'
    if dataset_name and 'multitq' in dataset_name.lower():
        dataset_type = 'MultiTQ'
    elif dataset_name and 'cwq' in dataset_name.lower():
        dataset_type = 'CWQ'
    elif dataset_name and 'webqsp' in dataset_name.lower():
        dataset_type = 'WebQSP'
    elif 'multitq' in str(interaction_history.get('data_source', '')).lower():
        dataset_type = 'MultiTQ'
    elif 'cwq' in str(interaction_history.get('data_source', '')).lower():
        dataset_type = 'CWQ'
    elif 'webqsp' in str(interaction_history.get('data_source', '')).lower():
        dataset_type = 'WebQSP'
    
    # Always show logging for KG evaluation examples (regardless of dataset)
    print(f"[{dataset_type}-EVAL] ===== ANSWER EXTRACTION EXAMPLE =====")
    print(f"[{dataset_type}-EVAL] Question asked: {interaction_history.get('original_query', 'N/A')}")
    print(f"[{dataset_type}-EVAL] Raw extracted answer: '{extracted_answer}'")
    print(f"[{dataset_type}-EVAL] Raw ground truth answers: {ground_truth_answers}")
    if extracted_answer:
        normalized_extracted = normalize_answer(extracted_answer, dataset_name)
        normalized_ground_truth = [normalize_answer(ga, dataset_name) for ga in ground_truth_answers if ga]
        print(f"[{dataset_type}-EVAL] Normalized extracted: '{normalized_extracted}'")
        print(f"[{dataset_type}-EVAL] Normalized ground truth: {normalized_ground_truth}")
    print(f"[{dataset_type}-EVAL] Final EM Score: {em_score}")
    print(f"[{dataset_type}-EVAL] Has answer tags: {has_answer_tags}")
    print(f"[{dataset_type}-EVAL] Dataset name for normalization: {dataset_name}")
    print(f"[{dataset_type}-EVAL] ===============================================")
    
    retrieval_correct = is_retrieval_correct_kg(assistant_response, ground_truth_answers, interaction_history, dataset_name)
    kg_stats = extract_kg_query_stats(assistant_response, interaction_history)

    # 3. Calculate score components using helper functions
    base_score_result = _calculate_base_score(em_score, is_valid_format, has_answer_tags, rewards)
    base_score = base_score_result['base_score']
    
    retrieval_score_value = rewards['retrieval_bonus'] if retrieval_correct else 0.0
    
    kg_interaction_result = _calculate_kg_interaction_score(kg_stats, rewards)
    
    kg_interaction_score = kg_interaction_result['kg_interaction_score']
    
    # 4. Final Score: A simple, additive combination
    if retrieval_score_value>0:
        total_score = base_score + retrieval_score_value
    else:
        total_score = base_score + kg_interaction_score
    
    # 5. Compile a streamlined results dictionary that maintains backward compatibility
    return {
        # Main scores
        "score": total_score,
        "total_score": total_score,
        "base_score": base_score,
        "kg_interaction_score": kg_interaction_score,
        "retrieval_score": retrieval_score_value,
        "em_score": em_score,  # 1.0 for correct, 0.0 for incorrect
        
        # Legacy compatibility fields
        "format_score": rewards['format_bonus'] if is_valid_format else 0.0,
        "structure_format_score": rewards['format_bonus'] if is_valid_format else 0.0,
        "final_format_score": rewards['answer_tag_bonus'] if has_answer_tags else 0.0,
        "valid_query_score": kg_interaction_result['unique_valid_reward'],
        
        # Component breakdowns
        "answer_component": base_score_result['answer_component'],
        "format_component": base_score_result['format_component'],
        "answer_tag_component": base_score_result['answer_tag_component'],
        "unique_valid_reward": kg_interaction_result['unique_valid_reward'],
        "major_error_penalty": kg_interaction_result['major_error_penalty'],
        "minor_error_penalty": kg_interaction_result['minor_error_penalty'],
        "no_data_penalty": kg_interaction_result['no_data_penalty'],
        "server_error_penalty": kg_interaction_result['server_error_penalty'],
        
        # Format and validation flags
        "has_valid_format": is_valid_format,
        "is_valid_structure": is_valid_format,
        "has_answer_tags": has_answer_tags,
        "retrieval_correct": retrieval_correct,
        "format_message": format_msg,
        "extracted_answer": extracted_answer,
        "ground_truth_answers": ground_truth_answers,
        
        # KG statistics (including unique query metrics)
        "kg_stats": kg_stats,
        "kg_error_counts": dict(kg_stats['error_counts']),
        "total_kg_errors": kg_stats['total_errors'],
        "total_queries": kg_stats['total_queries'],
        "valid_queries": kg_stats['valid_queries'],
        "invalid_queries": kg_stats['invalid_queries'],
        
        # Unique query metrics for reward hacking prevention
        "unique_valid_queries": kg_stats['unique_valid_queries'],
        "unique_invalid_queries": kg_stats['unique_invalid_queries'],
        "duplicate_valid_queries": kg_stats['duplicate_valid_queries'],
        "duplicate_invalid_queries": kg_stats['duplicate_invalid_queries'],
        "query_repetition_rate": kg_stats['duplicate_valid_queries'] / max(kg_stats['valid_queries'], 1),
        "unique_query_efficiency": kg_stats['unique_valid_queries'] / max(kg_stats['total_queries'], 1),
        "reward_hacking_prevented": kg_stats['duplicate_valid_queries'] > 0,
        "reward_inflation_prevented": kg_stats['duplicate_valid_queries'] * valid_query_reward,
        
        # Specific error counts
        "kg_server_error_count": kg_stats['error_counts'].get(KGErrorType.SERVER_ERROR, 0),
        "kg_not_found_count": (
            kg_stats['error_counts'].get(KGErrorType.ENTITY_NOT_FOUND, 0) + 
            kg_stats['error_counts'].get(KGErrorType.SAMPLE_NOT_FOUND, 0) + 
            kg_stats['error_counts'].get(KGErrorType.RELATION_NOT_FOUND, 0)
        ),
        "kg_format_error_count": kg_stats['error_counts'].get(KGErrorType.FORMAT_ERROR, 0),
        "kg_no_data_count": kg_stats['error_counts'].get(KGErrorType.NO_RESULTS, 0),
        
        # Penalty amounts applied
        "kg_server_error_penalty_applied": kg_stats['error_counts'].get(KGErrorType.SERVER_ERROR, 0) * kg_server_error_penalty,
        "kg_not_found_penalty_applied": (
            kg_stats['error_counts'].get(KGErrorType.ENTITY_NOT_FOUND, 0) + 
            kg_stats['error_counts'].get(KGErrorType.SAMPLE_NOT_FOUND, 0) + 
            kg_stats['error_counts'].get(KGErrorType.RELATION_NOT_FOUND, 0)
        ) * kg_not_found_penalty,
        "kg_format_error_penalty_applied": kg_stats['error_counts'].get(KGErrorType.FORMAT_ERROR, 0) * kg_format_error_penalty,
        "kg_no_data_penalty_applied": kg_stats['error_counts'].get(KGErrorType.NO_RESULTS, 0) * kg_no_data_penalty,
        
        # Configuration used (for debugging and reproducibility)
        "reward_config": rewards,
        
        # Verbose debugging information
        "verbose_info": {
            "assistant_response": assistant_response if verbose else None,
            "ground_truth": ground_truth if verbose else None,
            "method": "refactored_kg_aware"
        } if verbose else {}
    }

