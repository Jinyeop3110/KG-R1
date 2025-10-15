# KG-R1 Essential Scripts Documentation

This document describes the three essential scripts needed to run the KG-R1 system, now included in this minimal implementation.

## 📁 Created Scripts

All scripts have been successfully created and made executable:

1. ✅ **Training Script**: `train_grpo_kg_qwen_3b_cwq_f1_turn7.sh`
2. ✅ **Main Evaluation Script**: `eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh`
3. ✅ **Cross-Benchmark Evaluation**: `eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh`
4. ✅ **KG Server Helper**: `eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh`

---

## 🚀 Quick Start Guide

### Prerequisites

Before running any scripts, ensure you have:

1. **Set up the environment**:
   ```bash
   conda create -n kgr1 python=3.9
   conda activate kgr1
   pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu121
   pip install vllm==0.6.3
   pip install -e .
   pip install flash-attn --no-build-isolation
   pip install wandb
   ```

2. **Downloaded the data**:
   ```bash
   python scripts/data_kg/setup_datasets.py --save_path data_kg
   ```

3. **Set up the KG server environment** (separate conda env):
   ```bash
   conda create -n kg_server python=3.10
   conda activate kg_server
   pip install fastapi uvicorn pydantic requests transformers datasets networkx pandas pyarrow
   ```

---

## 1️⃣ Training Script

**File**: `train_grpo_kg_qwen_3b_cwq_f1_turn7.sh`

**Purpose**: Train a KG-R1 agent on ComplexWebQuestions (CWQ) dataset using GRPO (Group Relative Policy Optimization).

### Usage

```bash
# Step 1: Start the KG server (in a separate terminal)
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg

# Step 2: Run training (in another terminal)
conda activate kgr1
bash train_grpo_kg_qwen_3b_cwq_f1_turn7.sh
```

### Key Configuration

- **Model**: Qwen/Qwen2.5-3B-Instruct
- **Dataset**: CWQ (ComplexWebQuestions)
- **Max turns**: 7
- **Training steps**: 400
- **Batch size**: 128
- **GRPO rollouts**: 16
- **Output**: `verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7/`

### Key Features

- Multi-turn KG reasoning with up to 7 turns
- GRPO algorithm with group relative advantage estimation
- F1-based reward scoring
- Automatic checkpoint saving every 50 steps
- WandB logging to 'KG-R1-main' project

---

## 2️⃣ Main Evaluation Script

**File**: `eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh`

**Purpose**: Evaluate a trained KG-R1 checkpoint on CWQ or WebQSP datasets with Pass@K metrics.

### Usage

```bash
# Step 1: Start the KG server
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg

# Step 2: Run evaluation (auto-detect latest checkpoint)
conda activate kgr1
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh

# Advanced usage:
# Specify checkpoint path
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh /path/to/checkpoint

# Specify checkpoint and dataset
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh /path/to/checkpoint webqsp

# Specify checkpoint, dataset, and step number
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh /path/to/checkpoint cwq 300
```

### Key Configuration

- **Default checkpoint**: `verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7`
- **Datasets**: CWQ or WebQSP
- **Pass@K evaluation**: K = 1, 2, 3, 4
- **Max turns**: 7
- **Output**: `eval_results/eval_kg-r1/<experiment_name>/`

### Output Files

- `evaluation.log` - Complete evaluation log
- Detailed results with Pass@K metrics
- F1 and Hit@1 scores per K value

---

## 3️⃣ Cross-Benchmark Evaluation Script

**File**: `eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh`

**Purpose**: Evaluate KG-R1 model on 5 different KG-QA benchmarks to test cross-KG transferability.

### Supported Benchmarks

1. **TRex** (Wikidata relations)
2. **Zero-Shot RE** (Relation extraction)
3. **GrailQA** (Large-scale Wikidata QA)
4. **SimpleQA** (Simple factoid questions)
5. **QALD-10en** (Multi-domain questions)

### Usage

```bash
# Basic usage (auto-detect latest checkpoint)
bash eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh

# Specify checkpoint
bash eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh /path/to/checkpoint

# Specify checkpoint and step
bash eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh /path/to/checkpoint 300
```

### How It Works

1. **Automatic server management**: Each benchmark gets its own KG server on a dedicated port
2. **Screen sessions**: Servers run in detached screen sessions (auto-cleanup on exit)
3. **Sequential evaluation**: Runs all 5 benchmarks one after another
4. **Comprehensive logging**: Session log + individual benchmark logs

### Port Mapping

- SimpleQA: 9001
- GrailQA: 9000
- TRex: 9011
- QALD-10en: 9010
- Zero-Shot RE: 9012

### Output Structure

```
eval_results/eval_kg-r1_other_benchmarks/cwq_turn7_all_benchmarks_<timestamp>/
├── session_complete.log      # Complete session log
├── session_summary.log        # Results summary
├── trex/
│   └── evaluation.log
├── zero_shot_re/
│   └── evaluation.log
├── grailqa/
│   └── evaluation.log
├── simpleqa/
│   └── evaluation.log
└── qald10en/
    └── evaluation.log
```

---

## 4️⃣ KG Server Helper Script

**File**: `eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh`

**Purpose**: Start a KG server for any of the supported benchmarks (used internally by cross-benchmark script).

### Usage

```bash
# Start server for a specific dataset
bash eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh simpleqa

# Start with custom number of workers
bash eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh grailqa 8
```

### Supported Datasets

- `simpleqa`, `grailqa`, `trex`, `qald10en`, `zero_shot_re`

---

## 📊 Expected Results

Based on the README and paper:

### Main Benchmarks (CWQ/WebQSP)

| Dataset | F1 (1 run) | Hit@1 (1 run) | F1 (3 runs) | Hit@1 (3 runs) |
|---------|------------|---------------|-------------|----------------|
| WebQSP  | 77.5       | 84.7          | 85.8        | 91.7           |
| CWQ     | 70.9       | 73.8          | 81.0        | 83.9           |

### Cross-KG Transferability (Zero-shot)

| Benchmark | F1 | Hit@1 |
|-----------|-----|-------|
| SimpleQA  | 64.6 | 64.7 |
| GrailQA   | 42.8 | 50.2 |
| TRex      | 81.3 | 85.6 |
| QALD-10en | 55.9 | 57.7 |

---

## 🔧 Troubleshooting

### Common Issues

1. **KG Server Not Starting**
   ```bash
   # Check if port is already in use
   lsof -ti:8001
   # Kill existing process
   kill -9 $(lsof -ti:8001)
   ```

2. **CUDA Out of Memory**
   - Reduce `VAL_BATCH_SIZE` in scripts
   - For GrailQA, batch size is already set to 64 (larger dataset)

3. **Test File Not Found**
   ```bash
   # Re-download datasets
   python scripts/data_kg/setup_datasets.py --save_path data_kg
   ```

4. **Screen Session Not Cleaning Up**
   ```bash
   # Manually kill all KG server screen sessions
   screen -ls | grep kg_server | cut -d. -f1 | xargs -I{} screen -S {} -X quit
   ```

---

## 📝 Customization

### Modify Training Parameters

Edit `train_grpo_kg_qwen_3b_cwq_f1_turn7.sh`:

- Change `MAX_LENGTH` for longer/shorter prompts
- Adjust `actor_rollout_ref.rollout.search.max_turns` for different turn counts
- Modify `trainer.total_training_steps` for longer training

### Change Evaluation K Values

Edit evaluation scripts:

```bash
K_VALUES="1 2 3 4 5"  # Add more K values
```

### Use Different Model

```bash
export BASE_MODEL='Qwen/Qwen2.5-7B-Instruct'
```

---

## ✅ Verification

To verify all scripts are properly set up:

```bash
# Check all scripts exist and are executable
ls -lh train_grpo_kg_qwen_3b_cwq_f1_turn7.sh
ls -lh eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh
ls -lh eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh
ls -lh eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh

# All should show -rwxr-xr-x (executable permissions)
```

---

## 🎯 Complete Workflow Example

```bash
# 1. Set up data
python scripts/data_kg/setup_datasets.py --save_path data_kg

# 2. Start KG server (Terminal 1)
conda activate kg_server
python kg_r1/search/server.py --port 8001 --data_dir data_kg

# 3. Run training (Terminal 2)
conda activate kgr1
bash train_grpo_kg_qwen_3b_cwq_f1_turn7.sh

# 4. After training, evaluate on main dataset
bash eval_scripts/kg_r1_eval_main/eval_qwen_3b_cwq_f1_turn7.sh

# 5. Test cross-KG transferability
bash eval_scripts/kg_r1_eval_otherbenchmarks/run_all_benchmarks_cwq_turn7.sh
```

---

## 📚 References

- **Original implementation**: `/nobackup/users/yeopjin/workspace/RL_KG`
- **Paper**: KG-R1: Efficient and Transferable Agentic KG-RAG via RL (ICLR 2026 submission)
- **Base framework**: veRL (https://github.com/volcengine/verl)
- **Related work**: Search-R1 (https://github.com/PeterGriffinJin/Search-R1)

---

**Status**: ✅ All essential scripts implemented and ready to use!
