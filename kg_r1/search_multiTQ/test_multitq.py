#!/usr/bin/env python3
"""
Test script for MultiTQ KG server.
"""

import sys
import time
import requests
import json
from knowledge_graph_multitq import KnowledgeGraphMultiTQ


def test_local_kg():
    """Test the KG loading locally."""
    print("=" * 60)
    print("Testing MultiTQ Knowledge Graph Loading")
    print("=" * 60)
    
    data_path = "/nobackup/users/yeopjin/workspace/RL_KG/data_multitq_kg/MultiTQ"
    
    # Test each split
    for split in ["train", "valid", "test"]:
        print(f"\n📊 Testing {split} split...")
        try:
            kg = KnowledgeGraphMultiTQ(data_path, split)
            
            # Test entity lookup
            test_entity = "Nicos_Anastasiades"
            entity_ids = kg.get_id_from_entity(test_entity)
            if entity_ids:
                print(f"  ✅ Found entity '{test_entity}' with IDs: {entity_ids[:3]}")
                
                # Test getting relations
                eid = entity_ids[0]
                head_rels = kg.get_relations_for_entity(eid, as_head=True)
                tail_rels = kg.get_relations_for_entity(eid, as_head=False)
                print(f"  ✅ Entity has {len(head_rels)} head relations, {len(tail_rels)} tail relations")
            else:
                print(f"  ⚠️ Entity '{test_entity}' not found")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")


def test_server_api():
    """Test the server API endpoints."""
    print("\n" + "=" * 60)
    print("Testing MultiTQ KG Server API")
    print("=" * 60)
    
    base_url = "http://127.0.0.1:8001"
    
    # Test root endpoint
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Server running: {data['service']}")
            print(f"   Available splits: {data.get('available_splits', [])}")
            print(f"   Available actions: {data.get('available_actions', [])}")
    except requests.ConnectionError:
        print("❌ Server not running. Start it with:")
        print("   python -m kg_r1.search_multiTQ.server_multitq")
        return
    
    # Test retrieval endpoints
    test_cases = [
        {
            "name": "Get head relations for Nicos_Anastasiades",
            "request": {
                "action_type": "get_head_relations",
                "dataset_name": "train",
                "entity_id": "Nicos_Anastasiades"
            }
        },
        {
            "name": "Get entities with specific relation",
            "request": {
                "action_type": "get_head_entities",
                "dataset_name": "train",
                "entity_id": "Nicos_Anastasiades",
                "relation": "Make_an_appeal_or_request"
            }
        },
        {
            "name": "Get temporal facts in 2015",
            "request": {
                "action_type": "get_temporal_facts",
                "dataset_name": "train",
                "start_date": "2015-01-01",
                "end_date": "2015-12-31",
                "entity_id": "al-Shabaab"
            }
        },
        {
            "name": "Get entity timeline",
            "request": {
                "action_type": "get_entity_timeline",
                "dataset_name": "train",
                "entity_id": "John_Garang"
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n📝 Test: {test_case['name']}")
        try:
            response = requests.post(
                f"{base_url}/retrieve",
                json=test_case['request']
            )
            if response.status_code == 200:
                data = response.json()
                if data.get("success"):
                    content = data["choices"][0]["message"]["content"]
                    # Show first few lines of content
                    lines = content.split('\n')[:5]
                    for line in lines:
                        print(f"   {line}")
                    if len(content.split('\n')) > 5:
                        print(f"   ... ({len(content.split('\n'))} total lines)")
                else:
                    print(f"   ❌ Error: {data['choices'][0]['message']['content']}")
            else:
                print(f"   ❌ HTTP {response.status_code}: {response.text}")
        except Exception as e:
            print(f"   ❌ Exception: {e}")


def main():
    """Run all tests."""
    # Test local KG loading
    test_local_kg()
    
    # Test server API (if running)
    test_server_api()
    
    print("\n" + "=" * 60)
    print("Testing complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()