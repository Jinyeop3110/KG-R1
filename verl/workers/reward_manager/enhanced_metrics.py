"""
Enhanced evaluation metrics for KGQA with precision, recall, and F1 scores.
"""

import re
import string
from typing import Dict, List, Set, Tuple
from collections import Counter


def normalize_answer(text: str) -> str:
    """Normalize answer text for comparison."""
    if not text:
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


def extract_tokens(text: str) -> Set[str]:
    """Extract normalized tokens from text."""
    normalized = normalize_answer(text)
    if not normalized:
        return set()
    return set(normalized.split())


def extract_entities(text: str) -> Set[str]:
    """Extract potential entities from text using simple patterns."""
    if not text:
        return set()
    
    entities = set()
    
    # Capitalize words (potential proper nouns/entities)
    capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    entities.update([word.strip() for word in capitalized_words])
    
    # Numbers (years, quantities, etc.)
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    entities.update(numbers)
    
    # Common entity patterns (years, dates)
    year_patterns = re.findall(r'\b(19|20)\d{2}\b', text)
    entities.update(year_patterns)
    
    return entities


def compute_token_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute token-level F1, precision, and recall."""
    pred_tokens = extract_tokens(prediction)
    gt_tokens = extract_tokens(ground_truth)
    
    if not pred_tokens and not gt_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    elif not pred_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    elif not gt_tokens:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0}
    
    common_tokens = pred_tokens & gt_tokens
    
    precision = len(common_tokens) / len(pred_tokens)
    recall = len(common_tokens) / len(gt_tokens)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "precision": precision,
        "recall": recall, 
        "f1": f1
    }


def compute_entity_f1(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Compute entity-level F1, precision, and recall."""
    pred_entities = extract_entities(prediction)
    gt_entities = extract_entities(ground_truth)
    
    if not pred_entities and not gt_entities:
        return {"entity_precision": 1.0, "entity_recall": 1.0, "entity_f1": 1.0}
    elif not pred_entities:
        return {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0}
    elif not gt_entities:
        return {"entity_precision": 0.0, "entity_recall": 1.0, "entity_f1": 0.0}
    
    common_entities = pred_entities & gt_entities
    
    precision = len(common_entities) / len(pred_entities)
    recall = len(common_entities) / len(gt_entities)
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        "entity_precision": precision,
        "entity_recall": recall,
        "entity_f1": f1
    }


def compute_best_f1_against_set(prediction: str, ground_truth_list: List[str]) -> Dict[str, float]:
    """Compute best F1 score against a set of ground truth answers."""
    if not ground_truth_list:
        return {
            "token_precision": 0.0,
            "token_recall": 0.0, 
            "token_f1": 0.0,
            "entity_precision": 0.0,
            "entity_recall": 0.0,
            "entity_f1": 0.0
        }
    
    best_token_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_entity_metrics = {"entity_precision": 0.0, "entity_recall": 0.0, "entity_f1": 0.0}
    
    for gt in ground_truth_list:
        # Token-level metrics
        token_metrics = compute_token_f1(prediction, gt)
        if token_metrics["f1"] > best_token_metrics["f1"]:
            best_token_metrics = token_metrics
            
        # Entity-level metrics  
        entity_metrics = compute_entity_f1(prediction, gt)
        if entity_metrics["entity_f1"] > best_entity_metrics["entity_f1"]:
            best_entity_metrics = entity_metrics
    
    return {
        "precision": best_token_metrics["precision"],
        "recall": best_token_metrics["recall"],
        "f1": best_token_metrics["f1"],
        "entity_precision": best_entity_metrics["entity_precision"],
        "entity_recall": best_entity_metrics["entity_recall"],
        "entity_f1": best_entity_metrics["entity_f1"]
    }


def enhanced_answer_evaluation(prediction: str, ground_truth_list: List[str]) -> Dict[str, float]:
    """
    Comprehensive answer evaluation with multiple metrics.
    
    Args:
        prediction: Predicted answer string
        ground_truth_list: List of acceptable ground truth answers
        
    Returns:
        Dict containing various evaluation metrics
    """
    results = {}
    
    # Normalize inputs
    if not ground_truth_list:
        ground_truth_list = [""]
    
    # 1. Exact Match (original binary accuracy)
    normalized_pred = normalize_answer(prediction)
    exact_matches = [normalize_answer(gt) for gt in ground_truth_list]
    exact_match = any(normalized_pred == gt for gt in exact_matches)
    results["exact_match"] = exact_match
    results["em_score"] = 1 if exact_match else 0
    
    # 2. Token-level F1 metrics (best against all ground truths)
    f1_metrics = compute_best_f1_against_set(prediction, ground_truth_list)
    results.update(f1_metrics)
    
    # 3. String similarity metrics
    if ground_truth_list and ground_truth_list[0]:
        best_gt = ground_truth_list[0]  # Use first GT for additional metrics
        
        # Character-level overlap
        pred_chars = set(normalized_pred)
        gt_chars = set(normalize_answer(best_gt))
        if pred_chars or gt_chars:
            char_overlap = len(pred_chars & gt_chars) / max(len(pred_chars | gt_chars), 1)
            results["char_overlap"] = char_overlap
        else:
            results["char_overlap"] = 1.0
    else:
        results["char_overlap"] = 0.0
    
    return results


def log_enhanced_metrics_to_wandb(metrics: Dict[str, float], prefix: str = "eval"):
    """Log enhanced metrics to wandb."""
    try:
        import wandb
        if wandb.run is not None:
            wandb_metrics = {}
            for key, value in metrics.items():
                if isinstance(value, (int, float)):
                    wandb_metrics[f"{prefix}/{key}"] = value
            
            if wandb_metrics:
                wandb.log(wandb_metrics)
    except ImportError:
        pass  # wandb not available
    except Exception as e:
        print(f"Warning: Failed to log to wandb: {e}")


def aggregate_metrics_across_samples(all_metrics: List[Dict[str, float]]) -> Dict[str, float]:
    """Aggregate enhanced metrics across multiple samples."""
    if not all_metrics:
        return {}
    
    aggregated = {}
    
    # Get all unique metric keys
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    # Calculate averages for each metric
    for key in all_keys:
        values = [m.get(key, 0.0) for m in all_metrics if key in m]
        if values:
            aggregated[f"avg_{key}"] = sum(values) / len(values)
            aggregated[f"max_{key}"] = max(values)
            aggregated[f"min_{key}"] = min(values)
    
    return aggregated