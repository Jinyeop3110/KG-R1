"""
Action definitions and handlers for the Knowledge Graph Retrieval Server.

This file defines the action types and their corresponding handler classes.
Each action class implements the logic for a specific KG operation.
"""

import json
import os
import logging
import time
import random
import uuid
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union, List
from enum import Enum
from pydantic import BaseModel
from .knowledge_graph import KnowledgeGraph
from .error_types import KGErrorType
from .relation_formatter import format_relations

logger = logging.getLogger(__name__)


def kg_retrieval_completion_response(content: str, action_type: str, is_error: bool = False, error_type: str = None) -> Dict[str, Any]:
    """Create a KG retrieval completion response in OpenAI-style format with explicit error type."""
    # Use SUCCESS as default error_type when not an error
    if not is_error and error_type is None:
        from .error_types import KGErrorType
        error_type = KGErrorType.SUCCESS
    
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
        "object": "kg_retrieval",
        "created": int(time.time()),
        "model": "kg-retrieval",
        "success": not is_error,
        "action": action_type,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": content
                },
                "logprobs": None,
                "finish_reason": "stop" if not is_error else "error"
            }
        ],
        "usage": {},
        "system_fingerprint": "",
        "kg_metadata": {
            "action_type": action_type,
            "success": not is_error,
            "error_type": error_type,
            "timestamp": time.time()
        }
    }


class ActionType(str, Enum):
    """Defines the type of action to be performed by the retriever."""
    # New intuitive naming scheme
    GET_RELATIONS_IN = "get_relations_in"  # Incoming: ? → rel → entity
    GET_RELATIONS_OUT = "get_relations_out"  # Outgoing: entity → rel → ?
    GET_ENTITIES_IN = "get_entities_in"  # Get entities incoming via relation
    GET_ENTITIES_OUT = "get_entities_out"  # Get entities outgoing via relation
    GET_CONDITIONAL_RELATIONS = "get_conditional_relations"

    # Legacy aliases for backward compatibility
    GET_RELATIONS = "get_relations"  # Deprecated
    GET_HEAD_RELATIONS = "get_relations_in"  # Backward compatibility
    GET_TAIL_RELATIONS = "get_relations_out"  # Backward compatibility
    GET_HEAD_ENTITIES = "get_entities_in"  # Backward compatibility
    GET_TAIL_ENTITIES = "get_entities_out"  # Backward compatibility


class SearchRequest(BaseModel):
    """Request model for KG search operations."""
    action_type: ActionType
    dataset_name: str
    sample_id: Optional[str] = None
    entity_id: Optional[str] = None
    relation: Optional[str] = None
    concept: Optional[str] = None


class ActionHandler(ABC):
    """Abstract base class for action handlers."""
    
    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg
        
    @abstractmethod
    def execute(self, sample_id: str, **kwargs) -> Dict[str, Any]:
        """Execute the action with given parameters and return OpenAI-style response."""
        pass


class GetRelationsAction(ActionHandler):
    """Handler for getting relations for entities. DEPRECATED - use get_head_relations or get_tail_relations."""
    
    def execute(self, sample_id: str, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Get all relations for a given entity in the subgraph (both head and tail relations). DEPRECATED."""
        from .error_types import KGErrorType
        
        # Strict validation: get_relations should not receive extra arguments
        if 'relation_name' in kwargs and kwargs['relation_name'] is not None:
            error_content = f"get_relations accepts only one entity argument"
            return kg_retrieval_completion_response(
                error_content, "get_relations", 
                is_error=True, error_type=KGErrorType.FORMAT_ERROR
            )
        
        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations", 
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )
        
        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])
        
        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations", 
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)
        
        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations", 
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        found_relation_indices = set()
        
        # Search for relations where entity is both head and tail (comprehensive search)
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if head_idx == target_entity_local_idx or tail_idx == target_entity_local_idx:
                    found_relation_indices.add(rel_idx)
        
        # Convert relation indices back to relation names
        found_relations = []
        for rel_idx in found_relation_indices:
            try:
                relation_name = self.kg.get_relation(rel_idx)
                found_relations.append(relation_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid relation index {rel_idx} in sample {sample_id}")
        
        relations_list = sorted(found_relations)
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        if relations_list:
            # Use hierarchical formatting for better token efficiency
            formatted_relations = format_relations(relations_list)
            # Add deprecation warning to content
            content = f'Relations for entity "{clean_entity_id}" [DEPRECATED: Use get_head_relations() or get_tail_relations()]:\n{formatted_relations}'
            return kg_retrieval_completion_response(
                content, "get_relations", 
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No relations found for entity "{clean_entity_id}" in knowledge graph [DEPRECATED: Use get_head_relations() or get_tail_relations()]'
            return kg_retrieval_completion_response(
                content, "get_relations", 
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )


class GetRelationsInAction(ActionHandler):
    """Handler for getting incoming relations (? → relation → entity)."""

    def execute(self, sample_id: str, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Get all incoming relations where the given entity is the tail/object (? → relation → entity)."""
        from .error_types import KGErrorType

        # Strict validation: get_relations_in should not receive extra arguments
        if 'relation' in kwargs and kwargs['relation'] is not None:
            error_content = f"get_relations_in accepts only one entity argument"
            return kg_retrieval_completion_response(
                error_content, "get_relations_in",
                is_error=True, error_type=KGErrorType.FORMAT_ERROR
            )

        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_in",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])

        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_in",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)

        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_in",
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        found_relation_indices = set()

        # Search for relations where entity is the tail (incoming relations)
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if tail_idx == target_entity_local_idx:
                    found_relation_indices.add(rel_idx)

        # Convert relation indices back to relation names
        found_relations = []
        for rel_idx in found_relation_indices:
            try:
                relation_name = self.kg.get_relation(rel_idx)
                found_relations.append(relation_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid relation index {rel_idx} in sample {sample_id}")

        relations_list = list(found_relations)
        random.shuffle(relations_list)  # Randomize order to handle truncation fairly
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        if relations_list:
            # Use hierarchical formatting for better token efficiency
            formatted_relations = format_relations(relations_list)
            content = f'Incoming relations for entity "{clean_entity_id}":\n{formatted_relations}'
            return kg_retrieval_completion_response(
                content, "get_relations_in",
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No incoming relations found for entity "{clean_entity_id}" in knowledge graph'
            return kg_retrieval_completion_response(
                content, "get_relations_in",
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )

# Backward compatibility alias
GetHeadRelationsAction = GetRelationsInAction


class GetRelationsOutAction(ActionHandler):
    """Handler for getting outgoing relations (entity → relation → ?)."""

    def execute(self, sample_id: str, entity_id: str, **kwargs) -> Dict[str, Any]:
        """Get all outgoing relations where the given entity is the head/subject (entity → relation → ?)."""
        from .error_types import KGErrorType

        # Strict validation: get_relations_out should not receive extra arguments
        if 'relation' in kwargs and kwargs['relation'] is not None:
            error_content = f"get_relations_out accepts only one entity argument"
            return kg_retrieval_completion_response(
                error_content, "get_relations_out",
                is_error=True, error_type=KGErrorType.FORMAT_ERROR
            )

        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_out",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])

        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_out",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)

        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_relations_out",
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        found_relation_indices = set()

        # Search for relations where entity is the head (outgoing relations)
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if head_idx == target_entity_local_idx:
                    found_relation_indices.add(rel_idx)

        # Convert relation indices back to relation names
        found_relations = []
        for rel_idx in found_relation_indices:
            try:
                relation_name = self.kg.get_relation(rel_idx)
                found_relations.append(relation_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid relation index {rel_idx} in sample {sample_id}")

        relations_list = list(found_relations)
        random.shuffle(relations_list)  # Randomize order to handle truncation fairly
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        if relations_list:
            # Use hierarchical formatting for better token efficiency
            formatted_relations = format_relations(relations_list)
            content = f'Outgoing relations for entity "{clean_entity_id}":\n{formatted_relations}'
            return kg_retrieval_completion_response(
                content, "get_relations_out",
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No outgoing relations found for entity "{clean_entity_id}" in knowledge graph'
            return kg_retrieval_completion_response(
                content, "get_relations_out",
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )

# Backward compatibility alias
GetTailRelationsAction = GetRelationsOutAction


class GetEntitiesInAction(ActionHandler):
    """Handler for getting entities via incoming relations (? → relation → entity)."""

    def execute(self, sample_id: str, entity_id: str, relation: str, **kwargs) -> Dict[str, Any]:
        """Get entities connected via incoming relation (? → relation → entity)."""
        from .error_types import KGErrorType

        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_in",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])

        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_in",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)

        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_in",
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        # Convert relation to index (this should work as relations are global)
        relation_idx = self.kg.get_id_from_relation(relation)
        if relation_idx is None:
            clean_relation = relation.strip('"').strip("'")
            error_content = f'Relation "{clean_relation}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_in",
                is_error=True, error_type=KGErrorType.RELATION_NOT_FOUND
            )

        found_entity_indices = set()

        # Search using all matching local entity indices (? → relation → entity)
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if rel_idx == relation_idx and tail_idx == target_entity_local_idx:
                    found_entity_indices.add(head_idx)

        # Convert entity indices back to entity names
        found_entities = []
        for entity_idx in found_entity_indices:
            try:
                entity_name = self.kg.get_entity(entity_idx)
                found_entities.append(entity_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid entity index {entity_idx} in sample {sample_id}")

        entities_list = sorted(found_entities)
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        if entities_list:
            content = f'Entities incoming via relation "{relation}" to "{clean_entity_id}": {", ".join(entities_list)}'
            return kg_retrieval_completion_response(
                content, "get_entities_in",
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No entities found incoming via relation "{relation}" to "{clean_entity_id}" in knowledge graph'
            return kg_retrieval_completion_response(
                content, "get_entities_in",
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )

# Backward compatibility alias
GetHeadEntitiesAction = GetEntitiesInAction


class GetEntitiesOutAction(ActionHandler):
    """Handler for getting entities via outgoing relations (entity → relation → ?)."""

    def execute(self, sample_id: str, entity_id: str, relation: str, **kwargs) -> Dict[str, Any]:
        """Get entities connected via outgoing relation (entity → relation → ?)."""
        from .error_types import KGErrorType

        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_out",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])

        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_out",
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)
        relation_idx = self.kg.get_id_from_relation(relation)

        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_out",
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        if relation_idx is None:
            clean_relation = relation.strip('"').strip("'")
            error_content = f'Relation "{clean_relation}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_entities_out",
                is_error=True, error_type=KGErrorType.RELATION_NOT_FOUND
            )

        found_entity_indices = set()

        # Search using all matching local entity indices (entity → relation → ?)
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if rel_idx == relation_idx and head_idx == target_entity_local_idx:
                    found_entity_indices.add(tail_idx)

        # Convert entity indices back to entity names
        found_entities = []
        for entity_idx in found_entity_indices:
            try:
                entity_name = self.kg.get_entity(entity_idx)
                found_entities.append(entity_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid entity index {entity_idx} in sample {sample_id}")

        entities_list = sorted(found_entities)
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        if entities_list:
            content = f'Entities outgoing via relation "{relation}" from "{clean_entity_id}": {", ".join(entities_list)}'
            return kg_retrieval_completion_response(
                content, "get_entities_out",
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No entities found outgoing via relation "{relation}" from "{clean_entity_id}" in knowledge graph'
            return kg_retrieval_completion_response(
                content, "get_entities_out",
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )

# Backward compatibility alias
GetTailEntitiesAction = GetEntitiesOutAction


class GetConditionalRelationsAction(ActionHandler):
    """Handler for getting relations filtered by concept (first part of relation name)."""
    
    def execute(self, sample_id: str, entity_id: str, concept: str, **kwargs) -> Dict[str, Any]:
        """Get relations for a given entity filtered by concept (e.g., 'people', 'location', 'organization')."""
        from .error_types import KGErrorType
        
        # Strict validation: get_conditional_relations should not receive extra arguments
        if 'relation' in kwargs and kwargs['relation'] is not None:
            error_content = f"get_conditional_relations accepts entity and concept arguments only"
            return kg_retrieval_completion_response(
                error_content, "get_conditional_relations", 
                is_error=True, error_type=KGErrorType.FORMAT_ERROR
            )
        
        if sample_id not in self.kg.subgraphs:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_conditional_relations", 
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )
        
        sample_data = self.kg.subgraphs[sample_id]
        triples = sample_data.get("subgraph_triples", [])
        
        if not triples:
            error_content = f'Sample "{sample_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_conditional_relations", 
                is_error=True, error_type=KGErrorType.SAMPLE_NOT_FOUND
            )

        # Use the enhanced get_id_from_entity with sample_id
        target_entity_indices = self.kg.get_id_from_entity(entity_id, sample_id)
        
        if not target_entity_indices:
            # Strip quotes from entity_id for cleaner error messages
            clean_entity_id = entity_id.strip('"').strip("'")
            error_content = f'Entity "{clean_entity_id}" not found in KG'
            return kg_retrieval_completion_response(
                error_content, "get_conditional_relations", 
                is_error=True, error_type=KGErrorType.ENTITY_NOT_FOUND
            )

        found_relation_indices = set()
        
        # Search using all matching local entity indices
        for target_entity_local_idx in target_entity_indices:
            for head_idx, rel_idx, tail_idx in triples:
                if head_idx == target_entity_local_idx:
                    found_relation_indices.add(rel_idx)
        
        # Convert relation indices back to relation names and filter by concept
        concept_relations = []
        concept_lower = concept.lower().strip()
        
        for rel_idx in found_relation_indices:
            try:
                relation_name = self.kg.get_relation(rel_idx)
                # Check if relation starts with the concept (case-insensitive)
                if relation_name and '.' in relation_name:
                    relation_concept = relation_name.split('.')[0].lower()
                    if relation_concept == concept_lower:
                        concept_relations.append(relation_name)
            except (IndexError, TypeError):
                logger.warning(f"Invalid relation index {rel_idx} in sample {sample_id}")
        
        relations_list = sorted(concept_relations)
        # Strip quotes from entity_id for cleaner messages
        clean_entity_id = entity_id.strip('"').strip("'")
        clean_concept = concept.strip('"').strip("'")
        
        if relations_list:
            content = f'Relations for entity "{clean_entity_id}" with concept "{clean_concept}": {", ".join(relations_list)}'
            return kg_retrieval_completion_response(
                content, "get_conditional_relations", 
                is_error=False, error_type=KGErrorType.SUCCESS
            )
        else:
            content = f'No relations found for entity "{clean_entity_id}" with concept "{clean_concept}" in knowledge graph'
            return kg_retrieval_completion_response(
                content, "get_conditional_relations", 
                is_error=True, error_type=KGErrorType.NO_RESULTS
            )


# Registry for action handlers
ACTION_REGISTRY = {
    # New naming scheme
    ActionType.GET_RELATIONS_IN: GetRelationsInAction,
    ActionType.GET_RELATIONS_OUT: GetRelationsOutAction,
    ActionType.GET_ENTITIES_IN: GetEntitiesInAction,
    ActionType.GET_ENTITIES_OUT: GetEntitiesOutAction,
    ActionType.GET_CONDITIONAL_RELATIONS: GetConditionalRelationsAction,
    # Legacy - deprecated
    ActionType.GET_RELATIONS: GetRelationsAction,
}

# Mapping of action names to ActionType for easy lookup
ACTION_NAME_MAPPING = {
    # New naming scheme
    "get_relations_in": ActionType.GET_RELATIONS_IN,
    "get_relations_out": ActionType.GET_RELATIONS_OUT,
    "get_entities_in": ActionType.GET_ENTITIES_IN,
    "get_entities_out": ActionType.GET_ENTITIES_OUT,
    "get_conditional_relations": ActionType.GET_CONDITIONAL_RELATIONS,
    # Legacy aliases for backward compatibility
    "get_relations": ActionType.GET_RELATIONS,
    "get_head_relations": ActionType.GET_RELATIONS_IN,
    "get_tail_relations": ActionType.GET_RELATIONS_OUT,
    "get_head_entities": ActionType.GET_ENTITIES_IN,
    "get_tail_entities": ActionType.GET_ENTITIES_OUT,
}

def get_filtered_action_registry(action_names: List[str] = None) -> Dict[ActionType, Any]:
    """
    Get a filtered action registry based on specified action names.
    
    Args:
        action_names: List of action names to include. If None, returns all actions.
        
    Returns:
        Filtered action registry
    """
    if action_names is None:
        return ACTION_REGISTRY.copy()
    
    filtered_registry = {}
    for action_name in action_names:
        if action_name in ACTION_NAME_MAPPING:
            action_type = ACTION_NAME_MAPPING[action_name]
            if action_type in ACTION_REGISTRY:
                filtered_registry[action_type] = ACTION_REGISTRY[action_type]
        else:
            logger.warning(f"Unknown action name: {action_name}")
    
    return filtered_registry
