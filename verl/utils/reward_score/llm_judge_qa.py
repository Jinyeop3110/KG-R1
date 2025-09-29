"""
LLM-as-Judge Evaluation for QA Tasks
Implements semantic evaluation using LLMs to judge factual correctness and answer quality
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
    DEFAULT_JUDGE_MODEL = "gpt-4o-mini"
    JUDGE_RETRY_ATTEMPTS = 3

import openai
from openai import OpenAI


@dataclass
class JudgeResult:
    """Result from LLM judge evaluation"""
    score: float  # 0.0 to 1.0
    explanation: str
    confidence: float  # How confident the judge is (0.0 to 1.0)
    processing_time: float
    model_used: str


class LLMJudgeEvaluator:
    """
    LLM-as-Judge evaluator for QA tasks using semantic understanding
    rather than strict exact match
    """
    
    def __init__(self, judge_model: str = None, cache_enabled: bool = True):
        """
        Initialize LLM Judge Evaluator
        
        Args:
            judge_model: Model to use for judging (from KEYS.py)
            cache_enabled: Whether to cache results for repeated evaluations
        """
        self.judge_model = judge_model or DEFAULT_JUDGE_MODEL
        self.cache_enabled = cache_enabled
        self.cache = {}
        
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
    
    def get_judge_prompt(self, question: str, reference_answers: List[str], model_answer: str) -> str:
        """
        Create the judge prompt for evaluation
        
        Args:
            question: The original question
            reference_answers: List of valid reference answers
            model_answer: The model's generated answer
            
        Returns:
            Formatted prompt for the judge LLM
        """
        # Format reference answers
        if len(reference_answers) == 1:
            ref_text = f"Reference Answer: {reference_answers[0]}"
        else:
            ref_text = "Reference Answers (any of these would be correct):\\n"
            for i, ref in enumerate(reference_answers, 1):
                ref_text += f"{i}. {ref}\\n"
        
        prompt = f"""You are an expert evaluator for question-answering systems. Your task is to assess whether a model's answer is factually correct and addresses the given question.

**Question**: {question}

**{ref_text}**

**Model's Answer**: {model_answer}

**Instructions:**
1. Evaluate if the model's answer is factually correct compared to the reference answer(s)
2. Consider semantic equivalence, not just exact text matching
3. Account for multiple valid ways to express the same fact
4. Give partial credit for partially correct answers
5. Consider if the answer appropriately addresses the question

**Scoring Scale:**
- 1.0: Completely correct and fully addresses the question
- 0.8: Mostly correct with minor inaccuracies or slightly incomplete
- 0.6: Partially correct, captures main idea but missing some details
- 0.4: Some correct elements but significant errors or gaps  
- 0.2: Largely incorrect but shows some understanding
- 0.0: Completely incorrect, irrelevant, or no answer

**Output Format (JSON):**
{{
    "score": [number between 0.0 and 1.0],
    "explanation": "[Brief justification for the score]",
    "confidence": [how confident you are in this assessment, 0.0 to 1.0]
}}"""
        
        return prompt
    
    def _get_cache_key(self, question: str, reference_answers: List[str], model_answer: str) -> str:
        """Generate cache key for evaluation"""
        content = f"{question}|{','.join(sorted(reference_answers))}|{model_answer}|{self.judge_model}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def evaluate_async(self, question: str, reference_answers: Union[str, List[str]], 
                           model_answer: str) -> JudgeResult:
        """
        Asynchronously evaluate a QA pair using LLM judge
        
        Args:
            question: The original question
            reference_answers: Ground truth answer(s) 
            model_answer: Model's generated answer
            
        Returns:
            JudgeResult with score, explanation, and metadata
        """
        # Normalize reference answers to list
        if isinstance(reference_answers, str):
            reference_answers = [reference_answers]
        
        # Check cache first
        cache_key = None
        if self.cache_enabled:
            cache_key = self._get_cache_key(question, reference_answers, model_answer)
            if cache_key in self.cache:
                return self.cache[cache_key]
        
        start_time = time.time()
        
        # Generate prompt
        prompt = self.get_judge_prompt(question, reference_answers, model_answer)
        
        # Call judge LLM with retries
        for attempt in range(JUDGE_RETRY_ATTEMPTS):
            try:
                response = await self._call_judge_llm(prompt)
                judge_result = self._parse_judge_response(response, time.time() - start_time)
                
                # Cache result
                if self.cache_enabled and cache_key:
                    self.cache[cache_key] = judge_result
                
                return judge_result
                
            except Exception as e:
                if attempt == JUDGE_RETRY_ATTEMPTS - 1:
                    # Final attempt failed, return default
                    return JudgeResult(
                        score=0.0,
                        explanation=f"Judge evaluation failed: {str(e)}",
                        confidence=0.0,
                        processing_time=time.time() - start_time,
                        model_used=self.judge_model
                    )
                else:
                    # Wait before retry
                    await asyncio.sleep(2 ** attempt)
    
    def evaluate(self, question: str, reference_answers: Union[str, List[str]], 
                model_answer: str) -> JudgeResult:
        """
        Synchronous wrapper for evaluate_async
        """
        return asyncio.run(self.evaluate_async(question, reference_answers, model_answer))
    
    async def _call_judge_llm(self, prompt: str) -> str:
        """
        Call the judge LLM with the evaluation prompt
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model_config["model_name"],
                messages=[
                    {"role": "system", "content": "You are an expert evaluator for question-answering systems."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=self.model_config["max_tokens"],
                temperature=self.model_config["temperature"],
                timeout=self.model_config["timeout"]
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            raise Exception(f"Failed to call judge LLM: {str(e)}")
    
    def _parse_judge_response(self, response: str, processing_time: float) -> JudgeResult:
        """
        Parse the judge LLM response into structured result
        """
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\\{[^{}]*(?:\\{[^{}]*\\}[^{}]*)*\\}', response)
            if json_match:
                json_str = json_match.group()
                result_dict = json.loads(json_str)
                
                score = float(result_dict.get("score", 0.0))
                explanation = result_dict.get("explanation", "No explanation provided")
                confidence = float(result_dict.get("confidence", 0.5))
                
                # Validate score range
                score = max(0.0, min(1.0, score))
                confidence = max(0.0, min(1.0, confidence))
                
                return JudgeResult(
                    score=score,
                    explanation=explanation,
                    confidence=confidence,
                    processing_time=processing_time,
                    model_used=self.judge_model
                )
            else:
                # Fallback: try to extract score from text
                score_match = re.search(r'(?:score|rating)[:\s]*([0-9]*\.?[0-9]+)', response.lower())
                if score_match:
                    score = float(score_match.group(1))
                    # Normalize to 0-1 range if needed
                    if score > 1.0:
                        score = score / 10.0 if score <= 10.0 else score / 100.0
                    score = max(0.0, min(1.0, score))
                else:
                    score = 0.0
                
                return JudgeResult(
                    score=score,
                    explanation=response[:200] + "..." if len(response) > 200 else response,
                    confidence=0.3,  # Low confidence for parsing issues
                    processing_time=processing_time,
                    model_used=self.judge_model
                )
                
        except Exception as e:
            # Complete parsing failure
            return JudgeResult(
                score=0.0,
                explanation=f"Failed to parse judge response: {str(e)}",
                confidence=0.0,
                processing_time=processing_time,
                model_used=self.judge_model
            )


def compute_score_llm_judge(solution_str: str, ground_truth: Union[str, Dict[str, Any]], 
                           question: str = "", judge_model: str = None, **kwargs) -> Dict[str, Any]:
    """
    Main entry point for LLM judge evaluation compatible with existing reward system
    
    Args:
        solution_str: Model's generated response
        ground_truth: Ground truth answer(s) or dict with target info
        question: Original question (if available)
        judge_model: Judge model to use
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with score and metadata compatible with existing system
    """
    # Initialize judge
    judge = LLMJudgeEvaluator(judge_model=judge_model)
    
    # Parse ground truth
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
    model_answer = solution_str
    # Try to extract answer from <answer> tags if present
    answer_match = re.search(r'<answer>([^<]*)</answer>', solution_str, re.IGNORECASE | re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip()
    
    # Evaluate using judge
    try:
        result = judge.evaluate(question, reference_answers, model_answer)
        
        return {
            "score": result.score,
            "explanation": result.explanation,
            "confidence": result.confidence,
            "processing_time": result.processing_time,
            "judge_model": result.model_used,
            "reference_answers": reference_answers,
            "extracted_answer": model_answer
        }
        
    except Exception as e:
        return {
            "score": 0.0,
            "explanation": f"Judge evaluation failed: {str(e)}",
            "confidence": 0.0,
            "processing_time": 0.0,
            "judge_model": judge_model or DEFAULT_JUDGE_MODEL
        }


# For backward compatibility with existing evaluation system
def compute_score_em_kg_judge(**kwargs):
    """Wrapper for existing KG evaluation system"""
    return compute_score_llm_judge(**kwargs)