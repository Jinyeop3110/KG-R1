# KG-R1: Knowledge Graph Reasoning with Reinforcement Learning

[![Paper](https://img.shields.io/badge/Paper-ICLR_2026-blue.svg)](https://arxiv.org/abs/PLACEHOLDER)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)

KG-R1 is a **single-agent knowledge graph reasoning system** that enables language models to perform multi-turn reasoning over structured knowledge graphs through reinforcement learning. Using Group Relative Policy Optimization (GRPO), KG-R1 learns efficient knowledge graph exploration strategies with cross-KG transferability.

## 🚀 Key Features

- **🤖 Single-Agent Architecture**: Unified LLM agent replaces complex multi-module KG-RAG pipelines
- **🔄 Schema-Agnostic KG Server**: Works across different knowledge graphs (Freebase, Wikidata, Temporal KGs)  
- **📈 Cross-KG Transferability**: Train once, deploy on multiple knowledge graph schemas
- **⚡ Multi-turn Reasoning**: Up to 7 turns of iterative knowledge graph exploration
- **🎯 GRPO Training**: Stable reinforcement learning with group relative policy optimization
- **📊 Comprehensive Evaluation**: Pass@K metrics, FLOP analysis, LLM-as-judge evaluation

## 🏗️ System Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Language      │    │  KG Operations   │    │  Knowledge      │
│   Model         │◄──►│  Interface       │◄──►│  Graph Server   │
│   (Qwen2.5-3B)  │    │  (4 primitives)  │    │  (Any Schema)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   GRPO          │    │  Multi-turn      │    │  Structured     │
│   Training      │    │  Reasoning       │    │  Retrieval      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

## 🔧 Core KG Operations

KG-R1 uses four primitive operations that work across different knowledge graph schemas:

```python
# 1. Get outgoing relations from an entity
get_tail_relations("Barack_Obama")
# → ["profession", "birthplace", "spouse", ...]

# 2. Get entities connected via a specific relation
get_tail_entities("Barack_Obama", "profession") 
# → ["President", "Lawyer", "Author"]

# 3. Get incoming relations to an entity  
get_head_relations("Hawaii")
# → ["birthplace_of", "location_of", "part_of", ...]

# 4. Get entities that point to target via relation
get_head_entities("Hawaii", "birthplace_of")
# → ["Barack_Obama", "Jason_Momoa", ...]
```

## 📊 Multi-turn Reasoning Example

```
Question: "Who was the president when Barack Obama was born?"

Turn 1: <search>get_tail_relations("Barack_Obama")</search>
<information>Relations: ["birthdate", "birthplace", "profession", ...]</information>

Turn 2: <search>get_tail_entities("Barack_Obama", "birthdate")</search>
<information>Entities: ["August_4_1961"]</information>

Turn 3: <search>get_head_entities("1961", "president_during_year")</search>
<information>Entities: ["John F. Kennedy"]</information>

<answer>John F. Kennedy was the president when Barack Obama was born in 1961.</answer>
```

## 🛠️ Quick Start

### Installation

```bash
# Create environment
conda create -n kg_r1 python=3.9
conda activate kg_r1

# Install dependencies
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
pip install vllm==0.6.3
pip install -e .

# Additional dependencies
pip install flash-attn --no-build-isolation
pip install wandb transformers datasets uvicorn fastapi pandas pyarrow
```

### Basic Usage

```python
from kg_r1 import KGReasoningAgent

# Initialize agent with pre-trained model
agent = KGReasoningAgent(
    model_path="path/to/trained/model",
    kg_server_url="http://localhost:8001"
)

# Perform multi-turn reasoning
question = "Who was the president when Barack Obama was born?"
answer = agent.reason(question)
print(f"Answer: {answer}")
```

### Training Your Own Model

```bash
# 1. Start KG server
python kg_r1/search/kg_retrieval_server.py --base_data_path data_kg --port 8001

# 2. Train with GRPO
bash scripts/train_grpo_kg_qwen_3b_cwq.sh

# 3. Evaluate trained model  
bash scripts/eval_kg_r1_comprehensive.sh
```

## 📈 Performance Results

### Efficiency Analysis
- **Token Distribution**: ~83% KG information, ~13% reasoning generation, ~4% prompt
- **Computational Cost**: ~51,479 GFLOPs for complex multi-turn reasoning
- **Context Management**: Efficient context growth (512 → 550 tokens across turns)

### Cross-KG Transferability
- **Freebase → Wikidata**: Zero-shot transfer with minimal performance drop
- **Entity-focused → Temporal**: Adapts to time-aware reasoning tasks
- **Schema Agnostic**: Same operations work across different KG formats

## 🗃️ Supported KG Formats

- **Freebase**: Mid-based entities (`m.abc123`), dotted relations (`people.person.place_of_birth`)
- **Wikidata**: Q/P-based entities (`Q76`), property relations (`P19`)
- **Temporal KGs**: Date-annotated entities, temporal relations
- **Domain KGs**: Custom entity/relation vocabularies
- **Multilingual KGs**: Same structure, different languages

## 📊 Evaluation Framework

### Metrics Suite
```bash
# Run comprehensive evaluation
bash eval_scripts/eval_kg_r1_comprehensive.sh

# Key metrics generated:
# - Pass@K (K=1,2,3,4): Multi-attempt accuracy
# - F1/Precision/Recall: Answer quality scores  
# - FLOP Analysis: Computational efficiency
# - Turn Statistics: Multi-turn reasoning patterns
# - LLM-as-Judge: Semantic evaluation beyond exact match
```

### Cross-Dataset Evaluation
```bash
# Test transferability across different KG schemas
bash eval_scripts/run_cross_kg_evaluation.sh

# Evaluates on: CWQ, WebQSP, MultiTQ, GrailQA, SimpleQA, T-REx
```

## 🏗️ Training Your Own Models

### Data Format
```python
# Each training sample requires:
{
    "id": "sample_001",
    "question": "What is the capital of France?",
    "answers": ["Paris"],
    "data_source": "kgR1_dataset",
    "subgraph": {
        "entities": ["France", "Paris", "Europe", ...],
        "relations": ["capital_of", "located_in", "part_of", ...], 
        "tuples": [[0, 1, 1], [1, 2, 2], ...]  # (head_idx, rel_idx, tail_idx)
    }
}
```

### Training Pipeline
```bash
# 1. Convert your dataset to KG-R1 format
python scripts/data_conversion/convert_to_kg_format.py \
    --input_file your_dataset.json \
    --output_dir data_kg/your_dataset/

# 2. Set up KG server for your knowledge graph  
python scripts/kg_setup/setup_kg_server.py --kg_path /path/to/your/kg

# 3. Train with GRPO
bash scripts/train_grpo_kg_your_dataset.sh
```

## 📚 Citation

```bibtex
@inproceedings{kg-r1-2026,
  title={KG-R1: Efficient and Transferable Agentic KG-RAG via Reinforcement Learning},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

## 🤝 Contributing

We welcome contributions! Please see our [contributing guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

KG-R1 builds upon several excellent open-source projects:
- [veRL](https://github.com/volcengine/verl) - Reinforcement learning framework
- [vLLM](https://github.com/vllm-project/vllm) - Efficient LLM inference
- [DeepSeek-R1](https://github.com/deepseek-ai/DeepSeek-R1) - Multi-turn reasoning inspiration

---

For detailed documentation and advanced usage examples, visit our [documentation](docs/).