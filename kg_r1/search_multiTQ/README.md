# MultiTQ Knowledge Graph Server

A specialized KG server for MultiTQ temporal KGQA dataset, using split-wise temporal knowledge graphs.

## 🎯 Overview

This server handles temporal reasoning for MultiTQ questions by:
- **Using split-wise KGs**: Direct access to `train.txt`, `valid.txt`, `test.txt` files
- **Including temporal info**: Relations show timestamps like `Make_statement [2015-11-26, 2015-11-30]`
- **No preprocessing**: Uses raw MultiTQ data directly

## 📁 Architecture

```
kg_r1/search_multiTQ/
├── __init__.py                    # Module initialization
├── knowledge_graph_multitq.py    # MultiTQ KG handler
├── actions_multitq.py            # Action handlers with timestamps
├── server_multitq.py             # FastAPI server
├── test_multitq.py               # Test script
└── README.md                     # This file
```

## 🔧 Supported Actions

| Action | Description | Example Output |
|--------|-------------|---------------|
| `get_head_relations` | Relations where entity is subject | `Make_statement [2015-11-26, 2015-11-30]` |
| `get_tail_relations` | Relations where entity is object | `Praised_by [2012-01-15, 2013-05-20]` |
| `get_head_entities` | Entities connected by relation | `- Abdullah_Gül [2005-02-10]` |
| `get_tail_entities` | Entities connected by relation | `- European_Union [2012-02-20]` |

## 🚀 Usage

### Start Server
```bash
# Use the launch script
./scripts/launch_multitq_kg_server.sh

# Or manually
source /nobackup/users/yeopjin/init_general.sh
export MULTITQ_DATA_PATH="/nobackup/users/yeopjin/workspace/RL_KG/data_multitq_kg/MultiTQ"
python -m kg_r1.search_multiTQ.server_multitq
```

Server runs on: `http://127.0.0.1:8002`

### API Example
```bash
curl -X POST "http://127.0.0.1:8002/retrieve" \
     -H "Content-Type: application/json" \
     -d '{
       "action_type": "get_head_relations",
       "dataset_name": "train", 
       "entity_id": "Nicos_Anastasiades"
     }'
```

### Python Client Example
```python
import requests

response = requests.post("http://127.0.0.1:8002/retrieve", json={
    "action_type": "get_head_relations",
    "dataset_name": "train",
    "entity_id": "al-Shabaab"
})

result = response.json()
print(result["choices"][0]["message"]["content"])
```

## 📊 Data Statistics

| Split | Facts | Questions | Usage |
|-------|-------|-----------|--------|
| Train | 322,958 | 386,787 | Training temporal reasoning |
| Valid | 69,224 | 57,979 | Validation/development |
| Test | 69,147 | 54,584 | Final evaluation |

**Total**: 10,488 entities, 251 relations, 4,017 timestamps

## 🕰️ Temporal Features

### Timestamps in Relations
Relations include temporal information:
```
- Consult [2005-02-10, 2005-02-12, 2006-07-19, 2012-02-20]
- Make_an_appeal_or_request [2005-02-12]
- Express_intent_to_meet_or_negotiate [2005-02-11]
```

### Multi-granularity Time
Supports day/month/year level temporal reasoning:
- Day: `2015-06-20`
- Month: `2015-06` 
- Year: `2015`

## 🧪 Testing

```bash
# Test local KG loading
source /nobackup/users/yeopjin/init_general.sh
python test_temporal_relations.py

# Test server functionality
python kg_r1/search_multiTQ/test_multitq.py
```

## 🔍 Example Output

**Input**: `get_head_relations` for "al-Shabaab"
**Output**:
```
Relations where "al-Shabaab" appears as head (with timestamps):
- Abduct,_hijack,_or_take_hostage [2011-09-18, 2011-09-19, 2011-11-17, 2011-12-01, 2012-01-17]
- Accuse [2009-05-09, 2009-09-17, 2009-10-01, 2009-10-18, 2010-08-14, ... (11 total)]
- Use_unconventional_violence [2015-06-20]
```

## 🎯 Use Cases

Perfect for MultiTQ questions like:
- "When did al-Shabaab use unconventional violence against Muslims in the UK?"
- "Who did Nicos make his last appeal to before the Greek PM?"
- "What was the first thing John Garang did in 2005?"

## ⚡ Performance

- **Fast loading**: Split-wise indexing for efficient retrieval
- **Memory efficient**: Direct file access, no intermediate processing
- **Concurrent**: Supports multiple splits simultaneously

## 🔗 Integration

This server integrates with the KG-R1 training pipeline by providing temporal KG retrieval for MultiTQ dataset questions.