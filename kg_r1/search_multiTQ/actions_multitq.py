"""
Action handlers for MultiTQ Knowledge Graph Retrieval.

Adapted for temporal reasoning with MultiTQ's split-wise KG structure.
"""

import json
import time
import uuid
import random
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from enum import Enum
from pydantic import BaseModel
try:
    from .knowledge_graph_multitq import KnowledgeGraphMultiTQ
except ImportError:
    from knowledge_graph_multitq import KnowledgeGraphMultiTQ

def kg_retrieval_completion_response(content: str, action_type: str, is_error: bool = False, error_type: str = None) -> Dict[str, Any]:
    """Create a KG retrieval completion response for MultiTQ."""
    return {
        "id": f"multitq-{uuid.uuid4().hex[:24]}",
        "object": "kg_retrieval",
        "created": int(time.time()),
        "model": "kg-retrieval-multitq",
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
            "error_type": error_type or ("KG_SUCCESS" if not is_error else "ERROR"),
            "timestamp": time.time()
        }
    }


class ActionType(str, Enum):
    """Action types for MultiTQ KG retrieval."""
    GET_HEAD_RELATIONS = "get_head_relations"
    GET_TAIL_RELATIONS = "get_tail_relations"
    GET_HEAD_ENTITIES = "get_head_entities"
    GET_TAIL_ENTITIES = "get_tail_entities"
    # Removed temporal-specific actions as requested


class SearchRequest(BaseModel):
    """Request model for MultiTQ KG search operations."""
    action_type: ActionType
    dataset_name: str  # Should be "multitq"
    sample_id: Optional[str] = None  # Question ID
    entity_id: Optional[str] = None
    relation: Optional[str] = None
    concept: Optional[str] = None
    timestamp: Optional[str] = None  # Filter by specific timestamp (e.g., "2015-06-20")


class ActionHandler(ABC):
    """Abstract base class for MultiTQ action handlers."""
    
    def __init__(self, kg: KnowledgeGraphMultiTQ):
        self.kg = kg
        
    @abstractmethod
    def execute(self, sample_id: str = None, **kwargs) -> Dict[str, Any]:
        """Execute the action and return response."""
        pass
    
    def format_temporal_fact(self, quad: Tuple[int, int, int, int]) -> str:
        """Format a temporal quadruple as a string."""
        head = self.kg.get_entity(quad[0])
        relation = self.kg.get_relation(quad[1])
        tail = self.kg.get_entity(quad[2])
        timestamp = self.kg.get_timestamp(quad[3])
        return f"[{timestamp}] ({head}, {relation}, {tail})"


class GetHeadRelationsAction(ActionHandler):
    """Get relations where entity appears as TAIL (object) - matches CWQ semantics."""
    
    def execute(self, sample_id: str = None, entity_id: str = None, **kwargs) -> Dict[str, Any]:
        if not entity_id:
            return kg_retrieval_completion_response(
                "Error: entity_id is required", 
                "get_head_relations",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        # Find entity IDs
        entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
        if not entity_ids:
            return kg_retrieval_completion_response(
                f'Entity "{entity_id}" not found in knowledge graph',
                "get_head_relations",
                is_error=True, error_type="KG_ENTITY_NOT_FOUND"
            )
        
        # Collect all unique relations (NO timestamps for efficiency)
        # CWQ semantics: head_relations = relations where entity is TAIL
        unique_relations = set()
        for eid in entity_ids:
            relations = self.kg.get_relations_for_entity(eid, as_head=False)  # Changed: entity as TAIL
            for rel_id, head_id, timestamp_id in relations:
                rel_name = self.kg.get_relation(rel_id)
                unique_relations.add(rel_name)
        
        # Format relations without timestamps - maximum token efficiency
        formatted_relations = sorted(list(unique_relations))
        
        # Limit results
        formatted_relations = formatted_relations[:50]
        
        if not formatted_relations:
            content = f'No relations found where "{entity_id}" appears as head'
            return kg_retrieval_completion_response(
                content, "get_head_relations", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = f'Head relations for entity "{entity_id}":\n'
            content += '\n'.join(formatted_relations)
            return kg_retrieval_completion_response(content, "get_head_relations")


class GetTailRelationsAction(ActionHandler):
    """Get relations where entity appears as HEAD (subject) - matches CWQ semantics."""
    
    def execute(self, sample_id: str = None, entity_id: str = None, **kwargs) -> Dict[str, Any]:
        if not entity_id:
            return kg_retrieval_completion_response(
                "Error: entity_id is required",
                "get_tail_relations", 
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        # Find entity IDs
        entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
        if not entity_ids:
            return kg_retrieval_completion_response(
                f'Entity "{entity_id}" not found in knowledge graph',
                "get_tail_relations",
                is_error=True, error_type="KG_ENTITY_NOT_FOUND"
            )
        
        # Collect all unique relations (NO timestamps for efficiency)  
        # CWQ semantics: tail_relations = relations where entity is HEAD
        unique_relations = set()
        for eid in entity_ids:
            relations = self.kg.get_relations_for_entity(eid, as_head=True)  # Changed: entity as HEAD
            for rel_id, tail_id, timestamp_id in relations:
                rel_name = self.kg.get_relation(rel_id)
                unique_relations.add(rel_name)
        
        # Format relations without timestamps - maximum token efficiency
        formatted_relations = sorted(list(unique_relations))
        
        # Limit results
        formatted_relations = formatted_relations[:50]
        
        if not formatted_relations:
            content = f'No relations found where "{entity_id}" appears as tail'
            return kg_retrieval_completion_response(
                content, "get_tail_relations", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = f'Tail relations for entity "{entity_id}":\n'
            content += '\n'.join(formatted_relations)
            return kg_retrieval_completion_response(content, "get_tail_relations")


class GetHeadEntitiesAction(ActionHandler):
    """Get head entities for a given TAIL entity and relation - matches CWQ semantics."""
    
    def execute(self, sample_id: str = None, entity_id: str = None, relation: str = None, timestamp: str = None, **kwargs) -> Dict[str, Any]:
        if not entity_id:
            return kg_retrieval_completion_response(
                "Error: entity_id is required",
                "get_head_entities",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        if not relation:
            return kg_retrieval_completion_response(
                "Error: relation is required",
                "get_head_entities",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        # Find entity IDs
        entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
        if not entity_ids:
            return kg_retrieval_completion_response(
                f'Entity "{entity_id}" not found in knowledge graph',
                "get_head_entities",
                is_error=True, error_type="KG_ENTITY_NOT_FOUND"
            )
        
        # CWQ semantics: find HEAD entities where given entity is TAIL
        # Query pattern: (?, relation, entity) - WHO did relation TO entity
        entity_timestamps = {}  # Dict to consolidate timestamps by entity
        for eid in entity_ids:
            # Get entities where target_entity appears as TAIL (as_head=False)
            entities = self.kg.get_entities_by_relation(eid, relation, as_head=False)
            for head_id, timestamp_id in entities:
                entity_name = self.kg.get_entity(head_id)
                fact_timestamp = self.kg.get_timestamp(timestamp_id)
                
                # Filter by timestamp if provided
                if timestamp:
                    # Support partial matching (e.g., "2015" matches "2015-06-20")
                    if not fact_timestamp.startswith(timestamp):
                        continue
                
                if entity_name not in entity_timestamps:
                    entity_timestamps[entity_name] = []
                entity_timestamps[entity_name].append(fact_timestamp)
        
        # Format consolidated entities with grouped timestamps (like relations format)
        all_entities = []
        for entity_name, timestamps in sorted(entity_timestamps.items()):
            # Group timestamps by entity and apply truncation like relations
            sorted_timestamps = sorted(list(set(timestamps)))  # Remove duplicates and sort
            
            # Show all deduplicated YYYY-MM timestamps (no truncation) - compact format
            ts_str = ";".join(sorted_timestamps)
            
            all_entities.append(f"{entity_name} [{ts_str}]")
        
        # Limit to 50 unique entities
        all_entities = all_entities[:50]
        
        # Create descriptive message
        time_filter = f" at {timestamp}" if timestamp else ""
        
        if not all_entities:
            content = f'No head entities found for relation "{relation}" with tail "{entity_id}"{time_filter}'
            return kg_retrieval_completion_response(
                content, "get_head_entities", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = f'Head entities for relation "{relation}" with tail "{entity_id}"{time_filter}: {", ".join(all_entities)}'
            return kg_retrieval_completion_response(content, "get_head_entities")


class GetTailEntitiesAction(ActionHandler):
    """Get tail entities for a given HEAD entity and relation - matches CWQ semantics."""
    
    def execute(self, sample_id: str = None, entity_id: str = None, relation: str = None, timestamp: str = None, **kwargs) -> Dict[str, Any]:
        if not entity_id:
            return kg_retrieval_completion_response(
                "Error: entity_id is required",
                "get_tail_entities",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        if not relation:
            return kg_retrieval_completion_response(
                "Error: relation is required",
                "get_tail_entities",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        # Find entity IDs
        entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
        if not entity_ids:
            return kg_retrieval_completion_response(
                f'Entity "{entity_id}" not found in knowledge graph',
                "get_tail_entities",
                is_error=True, error_type="KG_ENTITY_NOT_FOUND"
            )
        
        # CWQ semantics: find TAIL entities where given entity is HEAD  
        # Query pattern: (entity, relation, ?) - WHAT did entity do via relation
        entity_timestamps = {}  # Dict to consolidate timestamps by entity
        for eid in entity_ids:
            # Get entities where target_entity appears as HEAD (as_head=True)
            entities = self.kg.get_entities_by_relation(eid, relation, as_head=True)
            for tail_id, timestamp_id in entities:
                entity_name = self.kg.get_entity(tail_id)
                fact_timestamp = self.kg.get_timestamp(timestamp_id)
                
                # Filter by timestamp if provided
                if timestamp:
                    # Support partial matching (e.g., "2015" matches "2015-06-20")
                    if not fact_timestamp.startswith(timestamp):
                        continue
                
                if entity_name not in entity_timestamps:
                    entity_timestamps[entity_name] = []
                entity_timestamps[entity_name].append(fact_timestamp)
        
        # Format consolidated entities with grouped timestamps (like relations format)
        all_entities = []
        for entity_name, timestamps in sorted(entity_timestamps.items()):
            # Group timestamps by entity and apply truncation like relations
            sorted_timestamps = sorted(list(set(timestamps)))  # Remove duplicates and sort
            
            # Show all deduplicated YYYY-MM timestamps (no truncation) - compact format
            ts_str = ";".join(sorted_timestamps)
            
            all_entities.append(f"{entity_name} [{ts_str}]")
        
        # Limit to 50 unique entities
        all_entities = all_entities[:50]
        
        # Create descriptive message
        time_filter = f" at {timestamp}" if timestamp else ""
        
        if not all_entities:
            content = f'No tail entities found for relation "{relation}" with head "{entity_id}"{time_filter}'
            return kg_retrieval_completion_response(
                content, "get_tail_entities", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = f'Tail entities for relation "{relation}" with head "{entity_id}"{time_filter}: {", ".join(all_entities)}'
            return kg_retrieval_completion_response(content, "get_tail_entities")


class GetTemporalFactsAction(ActionHandler):
    """Get temporal facts within a date range."""
    
    def execute(self, sample_id: str = None, start_date: str = None, end_date: str = None, entity_id: str = None, **kwargs) -> Dict[str, Any]:
        # Get temporal facts
        facts = self.kg.get_temporal_facts(start_date, end_date)
        
        # Filter by entity if provided
        if entity_id:
            entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
            if entity_ids:
                filtered = []
                for fact in facts:
                    if fact[0] in entity_ids or fact[2] in entity_ids:
                        filtered.append(fact)
                facts = filtered
        
        # Limit results
        facts = facts[:50]
        
        if not facts:
            date_range = f" between {start_date} and {end_date}" if start_date or end_date else ""
            entity_filter = f' for entity "{entity_id}"' if entity_id else ""
            content = f"No temporal facts found{date_range}{entity_filter}"
            return kg_retrieval_completion_response(
                content, "get_temporal_facts", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = "Temporal facts:\n"
            content += '\n'.join(self.format_temporal_fact(f) for f in facts)
            return kg_retrieval_completion_response(content, "get_temporal_facts")


class GetEntityTimelineAction(ActionHandler):
    """Get timeline of events for an entity."""
    
    def execute(self, sample_id: str = None, entity_id: str = None, **kwargs) -> Dict[str, Any]:
        if not entity_id:
            return kg_retrieval_completion_response(
                "Error: entity_id is required",
                "get_entity_timeline",
                is_error=True, error_type="KG_FORMAT_ERROR"
            )
        
        # Find entity IDs
        entity_ids = self.kg.get_id_from_entity(entity_id, sample_id)
        if not entity_ids:
            return kg_retrieval_completion_response(
                f'Entity "{entity_id}" not found in knowledge graph',
                "get_entity_timeline",
                is_error=True, error_type="KG_ENTITY_NOT_FOUND"
            )
        
        # Collect all facts involving the entity
        timeline = []
        for eid in entity_ids:
            # As head
            for quad in self.kg.head_index.get(eid, []):
                timeline.append(quad)
            # As tail
            for quad in self.kg.tail_index.get(eid, []):
                timeline.append(quad)
        
        # Remove duplicates and sort by timestamp
        timeline = list(set(timeline))
        timeline.sort(key=lambda x: self.kg.get_timestamp(x[3]))
        
        # Limit results
        timeline = timeline[:50]
        
        if not timeline:
            content = f'No timeline found for entity "{entity_id}"'
            return kg_retrieval_completion_response(
                content, "get_entity_timeline", 
                is_error=True, error_type="KG_NO_RESULTS"
            )
        else:
            content = f'Timeline for "{entity_id}":\n'
            content += '\n'.join(self.format_temporal_fact(f) for f in timeline)
            return kg_retrieval_completion_response(content, "get_entity_timeline")


# Action registry - EXCLUDING temporal-specific actions as requested
ACTION_REGISTRY_MULTITQ = {
    ActionType.GET_HEAD_RELATIONS: GetHeadRelationsAction,
    ActionType.GET_TAIL_RELATIONS: GetTailRelationsAction,
    ActionType.GET_HEAD_ENTITIES: GetHeadEntitiesAction,
    ActionType.GET_TAIL_ENTITIES: GetTailEntitiesAction
    # Removed: GET_TEMPORAL_FACTS and GET_ENTITY_TIMELINE as requested
}