"""
MultiTQ Knowledge Graph implementation.

This module handles temporal knowledge graphs from MultiTQ dataset,
using split-wise KG files (train.txt, valid.txt, test.txt) instead of per-sample subgraphs.
"""

import os
import json
from typing import Dict, List, Optional, Any, Set, Tuple
import time
from collections import defaultdict

class KnowledgeGraphMultiTQ:
    """
    Knowledge Graph for MultiTQ temporal KGQA dataset.
    Uses split-wise temporal KG quadruples instead of per-sample subgraphs.
    """
    
    def __init__(self, data_path: str, split: str = "train"):
        """
        Initialize MultiTQ KG.
        
        Args:
            data_path: Path to MultiTQ data (e.g., /path/to/MultiTQ)
            split: Which split to use ("train", "valid", or "test")
        """
        self.data_path = data_path
        self.split = split
        self.kg_path = os.path.join(data_path, "kg")
        self.questions_path = os.path.join(data_path, "questions")
        
        # Load entity and relation mappings
        self.entities = self._load_entities()
        self.relations = self._load_relations()
        self.timestamps = self._load_timestamps()
        
        # Create lookup dictionaries
        print(f"Building MultiTQ lookup dictionaries for {len(self.entities)} entities, {len(self.relations)} relations...")
        self._entity_to_id = {entity: idx for idx, entity in enumerate(self.entities)}
        self._relation_to_id = {relation: idx for idx, relation in enumerate(self.relations)}
        self._timestamp_to_id = {ts: idx for idx, ts in enumerate(self.timestamps)}
        
        # Create normalized lookup for case-insensitive matching
        self._normalized_entity_to_id = {}
        for idx, entity in enumerate(self.entities):
            normalized = entity.strip().lower().replace("_", " ")
            if normalized not in self._normalized_entity_to_id:
                self._normalized_entity_to_id[normalized] = []
            self._normalized_entity_to_id[normalized].append(idx)
        
        # Load temporal KG quadruples for the specific split
        self.quadruples = self._load_temporal_kg()
        
        # Build inverted indices for efficient retrieval
        self._build_indices()
        
        # Load questions for sample ID mapping
        self.questions = self._load_questions()
        
        print(f"MultiTQ KG initialized for '{split}' split:")
        print(f"  - Entities: {len(self.entities)}")
        print(f"  - Relations: {len(self.relations)}")
        print(f"  - Timestamps: {len(self.timestamps)}")
        print(f"  - Quadruples: {len(self.quadruples)}")
        print(f"  - Questions: {len(self.questions)}")
    
    def _format_for_display(self, name: str) -> str:
        """Convert underscores to spaces for natural language display."""
        return name.replace('_', ' ') if name else name
    
    def _format_for_lookup(self, name: str) -> str:
        """Convert spaces to underscores for internal lookup."""
        return name.replace(' ', '_') if name else name
    
    def _load_entities(self) -> List[str]:
        """Load entity mappings from entity2id.json"""
        entities_file = os.path.join(self.kg_path, "entity2id.json")
        if not os.path.exists(entities_file):
            return []
        
        with open(entities_file, 'r', encoding='utf-8') as f:
            entity_dict = json.load(f)
        
        # Sort by ID to ensure consistent ordering
        sorted_entities = sorted(entity_dict.items(), key=lambda x: x[1])
        return [entity for entity, _ in sorted_entities]
    
    def _load_relations(self) -> List[str]:
        """Load relation mappings from relation2id.json"""
        relations_file = os.path.join(self.kg_path, "relation2id.json")
        if not os.path.exists(relations_file):
            return []
        
        with open(relations_file, 'r', encoding='utf-8') as f:
            relation_dict = json.load(f)
        
        # Sort by ID to ensure consistent ordering
        sorted_relations = sorted(relation_dict.items(), key=lambda x: x[1])
        return [relation for relation, _ in sorted_relations]
    
    def _load_timestamps(self) -> List[str]:
        """Load timestamp mappings from ts2id.json"""
        ts_file = os.path.join(self.kg_path, "ts2id.json")
        if not os.path.exists(ts_file):
            return []
        
        with open(ts_file, 'r', encoding='utf-8') as f:
            ts_dict = json.load(f)
        
        # Sort by ID to ensure consistent ordering
        sorted_ts = sorted(ts_dict.items(), key=lambda x: x[1])
        return [ts for ts, _ in sorted_ts]
    
    def _load_temporal_kg(self) -> List[Tuple[int, int, int, int]]:
        """Load temporal KG quadruples from split-specific file"""
        # Map split names to file names
        split_files = {
            "train": "train.txt",
            "valid": "valid.txt",
            "test": "test.txt"
        }
        
        kg_file = os.path.join(self.kg_path, split_files.get(self.split, "train.txt"))
        if not os.path.exists(kg_file):
            print(f"Warning: KG file not found: {kg_file}")
            return []
        
        quadruples = []
        with open(kg_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 4:
                        head, relation, tail, timestamp = parts
                        # Convert to indices
                        head_id = self._entity_to_id.get(head)
                        relation_id = self._relation_to_id.get(relation)
                        tail_id = self._entity_to_id.get(tail)
                        timestamp_id = self._timestamp_to_id.get(timestamp)
                        
                        if all(x is not None for x in [head_id, relation_id, tail_id, timestamp_id]):
                            quadruples.append((head_id, relation_id, tail_id, timestamp_id))
        
        return quadruples
    
    def _build_indices(self):
        """Build inverted indices for efficient retrieval"""
        # Index by head entity
        self.head_index = defaultdict(list)
        # Index by tail entity
        self.tail_index = defaultdict(list)
        # Index by relation
        self.relation_index = defaultdict(list)
        # Index by timestamp
        self.timestamp_index = defaultdict(list)
        
        for quad in self.quadruples:
            head, relation, tail, timestamp = quad
            self.head_index[head].append(quad)
            self.tail_index[tail].append(quad)
            self.relation_index[relation].append(quad)
            self.timestamp_index[timestamp].append(quad)
    
    def _load_questions(self) -> Dict[str, Dict]:
        """Load questions for the split"""
        split_files = {
            "train": "train.json",
            "valid": "dev.json",
            "test": "test.json"
        }
        
        questions_file = os.path.join(self.questions_path, split_files.get(self.split, "train.json"))
        if not os.path.exists(questions_file):
            return {}
        
        questions = {}
        with open(questions_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                quid = str(item.get("quid", ""))
                if quid:
                    questions[quid] = item
        
        return questions
    
    def get_entity(self, idx: int) -> str:
        """Get entity name by index with underscore-to-space conversion for display."""
        if 0 <= idx < len(self.entities):
            return self._format_for_display(self.entities[idx])
        return f"entity_{idx}"
    
    def get_relation(self, idx: int) -> str:
        """Get relation name by index with underscore-to-space conversion for display."""
        if 0 <= idx < len(self.relations):
            return self._format_for_display(self.relations[idx])
        return f"relation_{idx}"
    
    def get_timestamp(self, idx: int) -> str:
        """Get timestamp by index - returns YYYY-MM format for token efficiency"""
        if 0 <= idx < len(self.timestamps):
            full_timestamp = self.timestamps[idx]
            # Convert YYYY-MM-DD to YYYY-MM for token efficiency
            if len(full_timestamp) >= 7 and full_timestamp[4] == '-':
                return full_timestamp[:7]  # Take first 7 characters (YYYY-MM)
            return full_timestamp
        return f"timestamp_{idx}"
    
    def get_id_from_entity(self, entity: str, sample_id: str = None) -> Optional[List[int]]:
        """
        Get entity IDs that match the given entity name.
        For MultiTQ, we search across the entire split KG, not per-sample subgraphs.
        Handles both underscore and space formats flexibly.
        """
        # Try exact match first
        if entity in self._entity_to_id:
            return [self._entity_to_id[entity]]
        
        # Try with underscores (convert spaces to underscores for lookup)
        entity_with_underscores = self._format_for_lookup(entity)
        if entity_with_underscores in self._entity_to_id:
            return [self._entity_to_id[entity_with_underscores]]
        
        # Try normalized match (both directions)
        normalized = entity.strip().lower().replace("_", " ")
        if normalized in self._normalized_entity_to_id:
            return self._normalized_entity_to_id[normalized]
        
        # Try partial match
        matches = []
        entity_lower = entity.lower()
        for idx, ent in enumerate(self.entities):
            ent_lower = ent.lower()
            # Try both original and normalized versions for partial matching
            if (entity_lower in ent_lower or ent_lower in entity_lower or
                entity_lower.replace(' ', '_') in ent_lower or 
                entity_lower.replace('_', ' ') in ent_lower.replace('_', ' ')):
                matches.append(idx)
        
        return matches if matches else None
    
    def get_relations_for_entity(self, entity_id: int, as_head: bool = True) -> List[Tuple[int, int, int]]:
        """
        Get relations where entity appears as head or tail.
        Returns list of (relation_id, other_entity_id, timestamp_id) tuples.
        """
        if as_head:
            quads = self.head_index.get(entity_id, [])
            return [(q[1], q[2], q[3]) for q in quads]  # relation, tail, timestamp
        else:
            quads = self.tail_index.get(entity_id, [])
            return [(q[1], q[0], q[3]) for q in quads]  # relation, head, timestamp
    
    def get_entities_by_relation(self, entity_id: int, relation: str, as_head: bool = True) -> List[Tuple[int, int]]:
        """
        Get entities connected by a specific relation.
        Returns list of (entity_id, timestamp_id) tuples.
        """
        # Get relation ID
        relation_id = self._relation_to_id.get(relation)
        if relation_id is None:
            # Try with underscores (convert spaces to underscores for lookup)
            relation_with_underscores = self._format_for_lookup(relation)
            relation_id = self._relation_to_id.get(relation_with_underscores)
        
        if relation_id is None:
            # Try normalized matching (both directions)
            relation_lower = relation.lower()
            for rid, rel in enumerate(self.relations):
                rel_lower = rel.lower()
                if (relation_lower in rel_lower or rel_lower in relation_lower or
                    relation_lower.replace(' ', '_') in rel_lower or 
                    relation_lower.replace('_', ' ') in rel_lower.replace('_', ' ')):
                    relation_id = rid
                    break
        
        if relation_id is None:
            return []
        
        results = []
        if as_head:
            quads = self.head_index.get(entity_id, [])
            for q in quads:
                if q[1] == relation_id:
                    results.append((q[2], q[3]))  # tail, timestamp
        else:
            quads = self.tail_index.get(entity_id, [])
            for q in quads:
                if q[1] == relation_id:
                    results.append((q[0], q[3]))  # head, timestamp
        
        return results
    
    def get_question_info(self, sample_id: str) -> Optional[Dict]:
        """Get question information for a given sample ID"""
        return self.questions.get(sample_id)
    
    def get_temporal_facts(self, start_date: str = None, end_date: str = None) -> List[Tuple[int, int, int, int]]:
        """
        Get facts within a temporal range.
        """
        if not start_date and not end_date:
            return self.quadruples
        
        filtered = []
        for quad in self.quadruples:
            timestamp = self.get_timestamp(quad[3])
            if start_date and timestamp < start_date:
                continue
            if end_date and timestamp > end_date:
                continue
            filtered.append(quad)
        
        return filtered