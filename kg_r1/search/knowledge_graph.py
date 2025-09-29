import os
import json
from typing import Dict, List, Optional, Any, Union
import time

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

class KnowledgeGraph:
    def __init__(self, name: str, data_path: str, use_entities_text: bool = False):
        self.name = name
        self.data_path = data_path
        self.use_entities_text = use_entities_text
        self.entities = self._load_entities()
        self.relations = self._load_relations()
        
        # Create O(1) lookup dictionaries for performance
        print(f"Building lookup dictionaries for {len(self.entities)} entities and {len(self.relations)} relations...")
        self._entity_to_id = {entity: idx for idx, entity in enumerate(self.entities)}
        self._relation_to_id = {relation: idx for idx, relation in enumerate(self.relations)}
        
        # Create normalized lookup for case-insensitive matching (returns list of all matching indices)
        self._normalized_entity_to_id = {}
        for idx, entity in enumerate(self.entities):
            normalized_entity = entity.strip().lower()
            if normalized_entity not in self._normalized_entity_to_id:
                self._normalized_entity_to_id[normalized_entity] = []
            self._normalized_entity_to_id[normalized_entity].append(idx)
        print(f"Lookup dictionaries built successfully!")
        
        self.subgraphs = {}
        self.load_all_samples()

    def _load_entities(self):
        if self.use_entities_text:
            entities_file = os.path.join(self.data_path, 'entities_text.txt')
        else:
            entities_file = os.path.join(self.data_path, 'entities.txt')
            
        if not os.path.exists(entities_file):
            return []
        with open(entities_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]

    def _load_relations(self):
        relations_file = os.path.join(self.data_path, 'relations.txt')
        if not os.path.exists(relations_file):
            return []
        with open(relations_file, 'r') as f:
            return [line.strip() for line in f if line.strip()]


    def _load_json_dataset(self, filename):
        """Load JSONL dataset (one JSON object per line)"""
        path = os.path.join(self.data_path, filename)
        if not os.path.exists(path):
            return []
        
        data = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        json_obj = json.loads(line)
                        data.append(json_obj)
                    except json.JSONDecodeError as e:
                        print(f"Error parsing line in {filename}: {e}")
                        continue
                    except Exception as e:
                        print(f"Error processing line in {filename}: {e}")
                        continue
        return data

    def load_all_samples(self):
        """Load and parse all samples from train_simple.json and test_simple.json."""
        self.subgraphs = {}  # Dictionary mapping IDs to subgraphs
        
        start_time = time.time()
        
        # Sequential loading
        for split in ["train_simple.json", "test_simple.json"]:
            entries = self._load_json_dataset(split)
            print(f"Loaded {len(entries)} entries from {split}")
            for entry in entries:
                sample_id = entry.get("id")
                if sample_id:  # Only process if we have a valid sample_id
                    self.subgraphs[sample_id] = {
                        "answers": entry.get("answers", []),
                        "subgraph_triples": entry.get("subgraph", {}).get("tuples", []),
                        "subgraph_entities": entry.get("subgraph", {}).get("entities", []),
                        "subgraph": entry.get("subgraph", {}),
                        #"question": entry.get("question", "")
                    }
        
        end_time = time.time()
        print(f"Loaded {len(self.subgraphs)} subgraphs in {end_time - start_time:.2f} seconds")

    def get_subgraph_for_sample(self, sample_id: str) -> Optional[Dict[str, Any]]:
        """Get subgraph data for a given sample ID"""
        if sample_id in self.subgraphs:
            return self.subgraphs[sample_id].get("subgraph", {})
        return None
    
    def get_entity(self, idx: int) -> str:
        return self.entities[idx]
    
    def get_relation(self, idx: int) -> str:
        return self.relations[idx]
    
    def get_id_from_entity(self, entity: str, sample_id: str = None) -> Optional[List[int]]:
        """
        Get the index/indices of an entity that exist in a specific sample's subgraph.
        
        Args:
            entity: The entity name to search for
            sample_id: Required - the sample ID to search within
            
        Returns:
            List of local indices that match in the subgraph, or None if no matches or invalid input
        """
        if sample_id is None:
            # Requirement: return None if sample_id is not provided
            return None
        
        if sample_id not in self.subgraphs:
            return None
            
        sample_data = self.subgraphs[sample_id]
        subgraph_entities = sample_data.get("subgraph_entities", [])
        
        if not subgraph_entities:
            return None
        
        # Convert subgraph_entities to a set for O(1) lookup
        subgraph_entities_set = set(subgraph_entities)
        
        # Find all global indices that match the entity (exact and normalized)
        candidate_indices = []
        
        # Try exact match first
        if entity in self._entity_to_id:
            candidate_indices.append(self._entity_to_id[entity])
        
        # Try normalized match (case-insensitive) - get all matching indices
        normalized_query = entity.strip().lower()
        normalized_matches = self._normalized_entity_to_id.get(normalized_query, [])
        candidate_indices.extend(normalized_matches)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for idx in candidate_indices:
            if idx not in seen:
                seen.add(idx)
                unique_candidates.append(idx)
        
        # Filter candidates to only include those that exist in the subgraph
        matching_indices = [idx for idx in unique_candidates if idx in subgraph_entities_set]
        
        return matching_indices if matching_indices else None
        
    def get_id_from_relation(self, relation: str) -> Optional[int]:
        """Get the index of a relation using O(1) dictionary lookup."""
        return self._relation_to_id.get(relation)

    def get_memory_usage(self):
        """Get current memory usage in MB"""
        if PSUTIL_AVAILABLE:
            process = psutil.Process()
            memory_info = process.memory_info()
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
            }
        else:
            return {'error': 'psutil not available - install with: pip install psutil'}

    # Other methods to be implemented

def main():
    """Test function to demonstrate KnowledgeGraph functionality"""
    data_path = "/nobackup/users/yeopjin/workspace/RL_KG/data_kg/webqsp"
    
    print("=" * 60)
    print("Testing KnowledgeGraph with Sequential Loading")
    print("=" * 60)
    
    # Test with sequential loading (entities.txt)
    print("\n1. Testing with entities.txt:")
    kg = KnowledgeGraph(name="webqsp", data_path=data_path, use_entities_text=False)
    
    print(f"Knowledge Graph: {kg.name}")
    print(f"Data path: {kg.data_path}")
    print(f"Use entities text: {kg.use_entities_text}")
    print(f"Number of entities: {len(kg.entities)}")
    print(f"Number of relations: {len(kg.relations)}")
    print(f"Number of subgraphs loaded: {len(kg.subgraphs)}")
    
    # Show first few entities (IDs)
    if kg.entities:
        print(f"First 5 entities (IDs): {kg.entities[:5]}")
    
    # Test with entities_text.txt
    print("\n" + "=" * 60)
    print("2. Testing with entities_text.txt:")
    kg_text = KnowledgeGraph(name="webqsp_text", data_path=data_path, use_entities_text=True)
    
    print(f"Use entities text: {kg_text.use_entities_text}")
    print(f"Number of entities: {len(kg_text.entities)}")
    
    # Show first few entities (text)
    if kg_text.entities:
        print(f"First 5 entities (text): {kg_text.entities[:5]}")
    
    # Check memory usage
    memory_usage = kg.get_memory_usage()
    if 'error' not in memory_usage:
        print(f"Memory usage - RSS: {memory_usage['rss_mb']:.2f} MB, VMS: {memory_usage['vms_mb']:.2f} MB")
    else:
        print(f"Memory usage: {memory_usage['error']}")
    
    # Test sample data access
    sample_ids = list(kg.subgraphs.keys())
    if sample_ids:
        sample_id = sample_ids[0]
        print(f"\nTesting with sample ID: {sample_id}")
        
        if sample_id in kg.subgraphs:
            sample_data = kg.subgraphs[sample_id]
            print(f"Sample {sample_id}:")
            print(f"Subgraph entities count: {len(sample_data.get('subgraph_entities', []))}")
            print(f"Number of triples: {len(sample_data.get('subgraph_triples', []))}")
            print(f"Answers: {sample_data.get('answers', [])}")
            
            # Show sample data
            triples = sample_data.get('subgraph_triples', [])
            if triples:
                print(f"First few triples: {triples[:3]}")
            
            subgraph_entities = sample_data.get('subgraph_entities', [])
            if subgraph_entities:
                print(f"First few subgraph entities: {subgraph_entities[:5]}")
    
    # Print first few entities and relations if available
    if kg.entities:
        print(f"\nFirst 5 entities: {kg.entities[:5]}")
    if kg.relations:
        print(f"First 5 relations: {kg.relations[:5]}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"- Total subgraphs: {len(kg.subgraphs)}")
    print(f"- Total entities (IDs): {len(kg.entities)}")
    print(f"- Total entities (text): {len(kg_text.entities)}")
    print(f"- Total relations: {len(kg.relations)}")

if __name__ == "__main__":
    main()
