"""
Factuality-focused LLM-as-Judge Implementation
Based on "Scaling Reasoning can Improve Factuality in Large Language Models" (arxiv:2505.11140)
"""

import re
import json
import time
import asyncio
import hashlib
from typing import Dict, Any, List, Union, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import sys
import os

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from KEYS import JUDGE_MODELS, DEFAULT_JUDGE_MODEL, JUDGE_RETRY_ATTEMPTS
except ImportError:
    print("Warning: KEYS.py not found. Using default configuration.")
    JUDGE_MODELS = {}
    DEFAULT_JUDGE_MODEL = "gpt-5-mini"
    JUDGE_RETRY_ATTEMPTS = 3

import openai
from openai import OpenAI


@dataclass
class FactualityJudgeResult:
    """Result from factuality-focused LLM judge"""
    is_correct: bool  # Binary: Does it refer to the same real-world entity?
    confidence: float  # Judge's confidence (0.0 to 1.0)
    explanation: str
    processing_time: float
    model_used: str


class FactualityLLMJudge:
    """
    Factuality-focused LLM judge based on arxiv:2505.11140 methodology
    Evaluates whether answers refer to the same real-world entity
    """
    
    def __init__(self, judge_model: str = None):
        """
        Initialize Factuality LLM Judge
        
        Args:
            judge_model: Model to use for judging (should be gpt-4o-mini or similar)
        """
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL
        
        # Initialize OpenAI client
        if self.judge_model in JUDGE_MODELS:
            config = JUDGE_MODELS[self.judge_model]
            if config["provider"] == "openai":
                self.client = OpenAI(api_key=config["api_key"])
                self.model_config = config
            else:
                raise NotImplementedError(f"Provider {config['provider']} not implemented yet")
        else:
            raise ValueError(f"Judge model {self.judge_model} not found in KEYS.py")
    
    def get_factuality_prompt(self, question: str, reference_answers: List[str], 
                            predicted_answer: str) -> str:
        """
        Create the factuality evaluation prompt based on arxiv:2505.11140 methodology
        
        Args:
            question: The original question
            reference_answers: List of correct reference answers
            predicted_answer: The model's predicted answer
            
        Returns:
            Formatted prompt for factuality evaluation
        """
        # Format reference answers
        if len(reference_answers) == 1:
            ref_text = f"Reference Answer: {reference_answers[0]}"
        else:
            ref_text = "Reference Answers (any of these would be correct):\\n"
            for i, ref in enumerate(reference_answers, 1):
                ref_text += f"{i}. {ref}\\n"
        
        prompt = f"""You are an expert evaluator for factual question-answering systems. Your task is to determine whether a predicted answer refers to the same real-world entity or concept as the reference answer(s).

**Question**: {question}

**{ref_text}**

**Predicted Answer**: {predicted_answer}

**Instructions**:
1. Evaluate whether the predicted answer refers to the SAME real-world entity, concept, or fact as the reference answer(s)
2. Consider semantic equivalence - different surface forms can refer to the same entity
3. Account for different levels of specificity (e.g., "English" vs "Jamaican English" - both refer to the same language family)
4. Consider alternative names, abbreviations, or descriptions of the same entity
5. Ignore minor formatting differences, articles, or punctuation

**Examples of SAME entity**:
- "Paris" and "Paris, France" (same city)
- "English" and "Jamaican English" (same language family)
- "2014" and "2014 World Series" (same time period/event)
- "UTC-10:00" and "Hawaii-Aleutian Time Zone" (same time zone)

**Examples of DIFFERENT entities**:
- "Berlin" and "Munich" (different cities)
- "2014" and "2018" (different years)
- "England" and "United Kingdom" (different political entities)

**Decision**: Does the predicted answer refer to the same real-world entity or concept as any of the reference answers?

**Output Format (JSON)**:
{{
    "is_correct": [true or false],
    "confidence": [0.0 to 1.0 - how confident you are in this judgment],
    "explanation": "[Brief explanation of your reasoning]"
}}"""
        
        return prompt
    
    def evaluate(self, question: str, reference_answers: Union[str, List[str]], 
                predicted_answer: str) -> FactualityJudgeResult:
        """
        Evaluate factuality using entity alignment approach
        
        Args:
            question: The original question
            reference_answers: Ground truth answer(s)
            predicted_answer: Model's predicted answer
            
        Returns:
            FactualityJudgeResult with binary correctness and metadata
        """
        # Normalize reference answers to list
        if isinstance(reference_answers, str):
            reference_answers = [reference_answers]
        
        start_time = time.time()
        
        # Generate prompt
        prompt = self.get_factuality_prompt(question, reference_answers, predicted_answer)
        
        # Call judge LLM with retries
        for attempt in range(JUDGE_RETRY_ATTEMPTS):
            try:
                response = self._call_judge_llm(prompt)
                judge_result = self._parse_factuality_response(response, time.time() - start_time)
                return judge_result
                
            except Exception as e:
                if attempt == JUDGE_RETRY_ATTEMPTS - 1:
                    # Final attempt failed, return default
                    return FactualityJudgeResult(
                        is_correct=False,
                        confidence=0.0,
                        explanation=f"Judge evaluation failed: {str(e)}",
                        processing_time=time.time() - start_time,
                        model_used=self.judge_model
                    )
                else:
                    # Wait before retry
                    time.sleep(2 ** attempt)
    
    def _call_judge_llm(self, prompt: str) -> str:
        """Call the judge LLM with the evaluation prompt"""
        try:
            # Handle different parameter formats for different models
            completion_params = {
                "model": self.model_config["model_name"],
                "messages": [
                    {"role": "system", "content": "You are an expert evaluator for factual question-answering systems."},
                    {"role": "user", "content": prompt}
                ],
                "timeout": self.model_config["timeout"]
            }
            
            # Add temperature if supported
            if "temperature" in self.model_config:
                completion_params["temperature"] = self.model_config["temperature"]
            
            # Use correct parameter name for max tokens
            if "max_completion_tokens" in self.model_config:
                completion_params["max_completion_tokens"] = self.model_config["max_completion_tokens"]
            elif "max_tokens" in self.model_config:
                completion_params["max_tokens"] = self.model_config["max_tokens"]
            
            response = self.client.chat.completions.create(**completion_params)
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Failed to call judge LLM: {str(e)}")
    
    def _parse_factuality_response(self, response: str, processing_time: float) -> FactualityJudgeResult:
        """
        Parse the factuality judge LLM response into structured result
        """
        try:
            # Find JSON in response using balanced brace parsing
            start = response.find('{')
            if start != -1:
                brace_count = 0
                end = start
                for i, char in enumerate(response[start:], start):
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            end = i + 1
                            break
                
                if brace_count == 0:
                    json_str = response[start:end]
                    result_dict = json.loads(json_str)
                    
                    is_correct = bool(result_dict.get("is_correct", False))
                    explanation = result_dict.get("explanation", "No explanation provided")
                    confidence = float(result_dict.get("confidence", 0.5))
                    
                    # Validate confidence range
                    confidence = max(0.0, min(1.0, confidence))
                    
                    return FactualityJudgeResult(
                        is_correct=is_correct,
                        explanation=explanation,
                        confidence=confidence,
                        processing_time=processing_time,
                        model_used=self.judge_model
                    )
            
            # Fallback parsing if JSON extraction fails
            response_lower = response.lower()
            if any(word in response_lower for word in ['true', 'correct', 'same', 'yes']):
                is_correct = True
            elif any(word in response_lower for word in ['false', 'incorrect', 'different', 'no']):
                is_correct = False
            else:
                is_correct = False  # Default to false if unclear
            
            return FactualityJudgeResult(
                is_correct=is_correct,
                explanation=response[:200] + "..." if len(response) > 200 else response,
                confidence=0.3,  # Low confidence for parsing issues
                processing_time=processing_time,
                model_used=self.judge_model
            )
                
        except Exception as e:
            # Complete parsing failure
            return FactualityJudgeResult(
                is_correct=False,
                explanation=f"Failed to parse judge response: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                model_used=self.judge_model
            )


def compute_score_factuality_judge(solution_str: str, ground_truth: Union[str, Dict[str, Any]], 
                                 question: str = "", judge_model: str = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for factuality LLM judge evaluation
    Compatible with existing reward system - returns binary scores for Pass@K
    
    Args:
        solution_str: Model's generated response
        ground_truth: Ground truth answer(s) or dict with target info
        question: Original question (if available)
        judge_model: Judge model to use
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with binary score (0.0 or 1.0) and metadata
    """
    # Initialize factuality judge
    judge = FactualityLLMJudge(judge_model=judge_model)
    
    # Parse ground truth (same logic as original)
    reference_answers = []
    if isinstance(ground_truth, dict):
        target_texts = ground_truth.get("target_text", [])
        if not isinstance(target_texts, list):
            target_texts = [target_texts] if target_texts else []
        reference_answers = [str(text) for text in target_texts if text]
        
        # Also include KB IDs as potential answers
        target_kb_ids = ground_truth.get("target_kb_id", [])
        if not isinstance(target_kb_ids, list):
            target_kb_ids = [target_kb_ids] if target_kb_ids else []
        reference_answers.extend([str(kb_id) for kb_id in target_kb_ids if kb_id])
        
    elif isinstance(ground_truth, (list, tuple)):
        reference_answers = [str(ans) for ans in ground_truth if ans]
    else:
        reference_answers = [str(ground_truth)] if ground_truth else []
    
    # Fallback if no reference answers
    if not reference_answers:
        return {"score": 0.0, "explanation": "No reference answers provided"}
    
    # Extract model's answer from solution string
    predicted_answer = solution_str
    # Try to extract answer from <answer> tags if present
    answer_match = re.search(r'<answer>([^<]*)</answer>', solution_str, re.IGNORECASE | re.DOTALL)
    if answer_match:
        predicted_answer = answer_match.group(1).strip()
    
    # Evaluate using factuality judge
    try:
        result = judge.evaluate(question, reference_answers, predicted_answer)
        
        # Return binary score for Pass@K compatibility
        binary_score = 1.0 if result.is_correct else 0.0
        
        return {
            "score": binary_score,  # Binary for Pass@K evaluation
            "is_correct": result.is_correct,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "judge_model": result.model_used,
            "reference_answers": reference_answers,
            "extracted_answer": predicted_answer
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "is_correct": False,
            "explanation": f"Factuality judge evaluation failed: {str(e)}",
            "confidence": 0.0,
            "processing_time": 0.0,
            "judge_model": judge_model or DEFAULT_JUDGE_MODEL
        }


# For backward compatibility
def compute_score_em_kg_factuality(**kwargs):
    """Wrapper for existing KG evaluation system"""
    return compute_score_factuality_judge(**kwargs)