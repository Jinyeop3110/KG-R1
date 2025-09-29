"""
KG Format Reward Manager

This module contains the reward manager for Knowledge Graph question answering tasks.
"""

from typing import Dict, List, Any
from verl.utils.torch_functional import TensorFunctional
from verl.utils.model import load_tokenizer
from verl.single_controller.base.rollout import RolloutWorkerGroup
from verl.protocol import DataProto

from .qa_em_format_kg import compute_score_em_kg


class KGFormatRewardManager:
    """
    Reward manager for Knowledge Graph question answering with fake information detection.
    """
    
    def __init__(self):
        """
        Initialize the KG Format Reward Manager.
        """
        pass
    
    def compute_score(self, data_batch: DataProto, meta_info_batch: Dict = None) -> Dict[str, Any]:
        """
        Compute reward scores for a batch of KG question answering samples.
        
        Args:
            data_batch: Batch of data containing model responses
            meta_info_batch: Batch metadata containing ground truth answers and other info
            
        Returns:
            Dictionary containing computed scores and metrics
        """
        # Extract generated responses (solutions)
        solutions = [sample for sample in data_batch.meta_info['samples']]
        
        # Get ground truth data and other metadata
        ground_truth_answers = meta_info_batch.get('ground_truth_answer', [])
        
        # Extract info_mask if available
        info_masks = []
        if hasattr(data_batch.batch, 'info_mask') and data_batch.batch['info_mask'] is not None:
            # Convert tensor mask to list for processing
            info_mask_tensor = data_batch.batch['info_mask']
            if info_mask_tensor.dim() == 2:  # [batch_size, seq_len]
                for i in range(info_mask_tensor.shape[0]):
                    info_masks.append(info_mask_tensor[i].cpu().tolist())
            else:
                # Handle other dimensions if needed
                for i in range(len(solutions)):
                    info_masks.append(None)
        else:
            info_masks = [None] * len(solutions)
        
        # Extract interaction_history if available from meta_info
        interaction_histories = []
        if hasattr(data_batch, 'meta_info') and 'interaction_history' in data_batch.meta_info:
            # The interaction_history is batch-level, but we need to distribute it per sample
            batch_interaction_history = data_batch.meta_info['interaction_history']
            # For now, assume all samples in the batch share the same interaction history
            # This might need to be adjusted based on actual batch structure
            for i in range(len(solutions)):
                interaction_histories.append(batch_interaction_history)
        else:
            interaction_histories = [None] * len(solutions)
        
        # Compute scores for each sample
        all_scores = []
        for i, (solution, gt_answer, info_mask, interaction_history) in enumerate(
            zip(solutions, ground_truth_answers, info_masks, interaction_histories)
        ):
            score_result = compute_score_em_kg(
                solution_str=solution,
                ground_truth_answer=gt_answer,
                info_mask=info_mask,
                interaction_history=interaction_history
            )
            all_scores.append(score_result)
        
        # Aggregate results
        return self._aggregate_scores(all_scores)
    
    def _aggregate_scores(self, all_scores: List[Dict]) -> Dict[str, Any]:
        """
        Aggregate individual sample scores into batch-level metrics.
        
        Args:
            all_scores: List of score dictionaries from individual samples
            
        Returns:
            Aggregated score dictionary
        """
        if not all_scores:
            return {'scores': [], 'total_score': 0.0}
        
        # Extract final scores for each sample
        final_scores = [score_dict['total_score'] for score_dict in all_scores]
        
        # Collect all metrics for aggregation
        aggregated = {
            'scores': final_scores,
            'total_score': sum(final_scores) / len(final_scores),  # Average score
        }
        
        # Aggregate other metrics if they exist
        if 'em_score' in all_scores[0]:
            aggregated['em_scores'] = [score_dict['em_score'] for score_dict in all_scores]
            aggregated['avg_em_score'] = sum(aggregated['em_scores']) / len(aggregated['em_scores'])
        
        if 'format_score' in all_scores[0]:
            aggregated['format_scores'] = [score_dict['format_score'] for score_dict in all_scores]
            aggregated['avg_format_score'] = sum(aggregated['format_scores']) / len(aggregated['format_scores'])
        
        if 'quality_score' in all_scores[0]:
            aggregated['quality_scores'] = [score_dict['quality_score'] for score_dict in all_scores]
            aggregated['avg_quality_score'] = sum(aggregated['quality_scores']) / len(aggregated['quality_scores'])
        
        if 'retrieval_success_score' in all_scores[0]:
            aggregated['retrieval_success_scores'] = [score_dict['retrieval_success_score'] for score_dict in all_scores]
            aggregated['avg_retrieval_success_score'] = sum(aggregated['retrieval_success_scores']) / len(aggregated['retrieval_success_scores'])
        

        
        return aggregated 