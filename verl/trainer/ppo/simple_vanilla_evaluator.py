"""
Simple Vanilla Evaluator using VLLM directly.

This is a standalone evaluator that:
1. Uses VLLM for generation
2. Applies vanilla prompt augmentation
3. Computes standard NLP metrics
4. No KG dependencies or complex VERL setup
"""

import json
import os
from typing import Dict, Any, List
import numpy as np
from tqdm import tqdm

import pandas as pd
from transformers import AutoTokenizer

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


class SimpleVanillaEvaluator:
    """Simple vanilla evaluator using VLLM directly."""
    
    def __init__(self, model_path: str, n_rollout_eval: int = 4, k_values: List[int] = [1, 2, 3, 4]):
        self.model_path = model_path
        self.n_rollout_eval = n_rollout_eval
        self.k_values = k_values
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # Initialize VLLM engine
        self._setup_vllm()
        
    def _setup_vllm(self):
        """Setup generation backend - using transformers for compatibility."""
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading model with transformers: {self.model_path}")
            
            # Load model and tokenizer
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                trust_remote_code=True
            )
            
            # Use the tokenizer we already created
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            print(f"✅ Transformers model initialized with {self.model_path}")
            
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            raise
    
    def _create_vanilla_prompt(self, question: str) -> str:
        """Create vanilla prompt with examples."""
        return f"""You are Qwen, created by Alibaba Cloud. You are a helpful assistant.

Answer the given question directly and concisely based on your knowledge.
Provide only the factual answer without explanation or reasoning.

Examples:
Question: What is the capital of France?
Answer: Paris

Question: Who wrote Romeo and Juliet?
Answer: William Shakespeare

Question: What year did World War II end?
Answer: 1945

Question: What is the largest planet in our solar system?
Answer: Jupiter

{question}
Answer:"""
    
    def evaluate(self, data_file: str, eval_samples: int = 0) -> Dict[str, Any]:
        """Run vanilla evaluation."""
        print(f"Loading dataset: {data_file}")
        
        # Load data
        df = pd.read_parquet(data_file)
        
        if eval_samples > 0:
            df = df.head(eval_samples)
            print(f"Limited to {eval_samples} samples")
        
        print(f"Evaluating {len(df)} samples with {self.n_rollout_eval} responses each")
        
        # Extract questions and answers
        questions = []
        ground_truths = []
        
        for _, row in df.iterrows():
            # Extract question from prompt (assuming it contains the question)
            prompt = row.get('prompt', '')
            
            # Simple extraction - look for the actual question
            if 'Question:' in prompt:
                question = prompt.split('Question:')[-1].strip()
                if '(Initial entities:' in question:
                    question = question.split('(Initial entities:')[0].strip()
            else:
                question = prompt[-200:]  # Fallback: last 200 chars
                
            questions.append(question)
            
            # Get ground truth
            if 'ground_truth' in row:
                ground_truth = row['ground_truth']
            elif 'answer' in row:
                ground_truth = row['answer']  
            else:
                ground_truth = "unknown"
                
            ground_truths.append(str(ground_truth))
        
        # Generate responses
        all_results = []
        
        for q_idx, question in enumerate(tqdm(questions, desc="Generating responses")):
            # Create vanilla prompt
            vanilla_prompt = self._create_vanilla_prompt(question)
            
            # Generate multiple responses using transformers
            question_results = []
            for rollout_idx in range(self.n_rollout_eval):
                try:
                    # Tokenize input
                    inputs = self.tokenizer(vanilla_prompt, return_tensors="pt", padding=True)
                    inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                    
                    # Generate response
                    import torch
                    with torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            max_new_tokens=64,
                            temperature=0.7 if rollout_idx > 0 else 0.0,  # First response greedy, others sampled
                            do_sample=rollout_idx > 0,
                            pad_token_id=self.tokenizer.eos_token_id,
                            eos_token_id=self.tokenizer.eos_token_id
                        )
                    
                    # Decode response
                    response = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                    response = response.strip()
                    
                    # Clean response - remove any special formatting
                    response = response.replace('<answer>', '').replace('</answer>', '').strip()
                    
                    question_results.append(response)
                    
                except Exception as e:
                    print(f"Generation error for question {q_idx}, rollout {rollout_idx}: {e}")
                    question_results.append("")
            
            all_results.append(question_results)
        
        # Compute metrics
        return self._compute_metrics(questions, all_results, ground_truths)
    
    def _compute_metrics(self, questions: List[str], all_results: List[List[str]], 
                        ground_truths: List[str]) -> Dict[str, Any]:
        """Compute Pass@K and other metrics."""
        
        pass_at_k_results = {f'pass@{k}': [] for k in self.k_values}
        f1_results = []
        
        # Compute metrics for each question
        for q_idx, (question, responses, gt) in enumerate(zip(questions, all_results, ground_truths)):
            # Compute exact matches for this question
            exact_matches = [compute_exact_match(resp, gt) for resp in responses]
            f1_scores = [compute_f1_score(resp, gt) for resp in responses]
            
            # Pass@K for this question
            for k in self.k_values:
                pass_at_k = compute_pass_at_k(exact_matches, k)
                pass_at_k_results[f'pass@{k}'].append(pass_at_k)
            
            # Best F1 for this question
            best_f1 = max(f1_scores) if f1_scores else 0.0
            f1_results.append(best_f1)
            
            # Debug: print first few examples
            if q_idx < 3:
                print(f"\n=== SAMPLE {q_idx} DEBUG ===")
                print(f"Question: {question[:100]}...")
                print(f"Ground truth: {gt}")
                print(f"Responses: {responses}")
                print(f"Exact matches: {exact_matches}")
                print(f"F1 scores: {[f'{f:.3f}' for f in f1_scores]}")
        
        # Final statistics
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
        all_responses = [resp for responses in all_results for resp in responses]
        if all_responses:
            response_lengths = [len(r.split()) for r in all_responses]
            final_metrics['response_length/mean'] = float(np.mean(response_lengths))
        else:
            final_metrics['response_length/mean'] = 0.0
        
        # Print results
        print(f"\n{'='*50}")
        print("SIMPLE VANILLA EVALUATION RESULTS")
        print(f"{'='*50}")
        print(f"Total questions evaluated: {len(questions)}")
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
        
        return final_metrics