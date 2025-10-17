# FSDP Checkpoint to HuggingFace Conversion Guide

## Quick Start

To convert your FSDP sharded checkpoint to a single HuggingFace model:

```bash
cd /home/yeopjin/orcd/pool/workspace/RL_KG

bash scripts/convert_checkpoint.sh \
    /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150
```

This will create: `global_step_150_merged/` with a single HuggingFace model.

---

## What Gets Converted

### Input (FSDP Sharded Checkpoint)
```
global_step_150/
└── actor/
    ├── model_world_size_4_rank_0.pt  (3.2GB - GPU 0's shard)
    ├── model_world_size_4_rank_1.pt  (3.2GB - GPU 1's shard)
    ├── model_world_size_4_rank_2.pt  (3.2GB - GPU 2's shard)
    ├── model_world_size_4_rank_3.pt  (3.2GB - GPU 3's shard)
    ├── extra_state_world_size_4_rank_0.pt
    ├── extra_state_world_size_4_rank_1.pt
    ├── extra_state_world_size_4_rank_2.pt
    ├── extra_state_world_size_4_rank_3.pt
    ├── config.json
    ├── tokenizer.json
    └── ... (other tokenizer files)
```

### Output (Merged HuggingFace Model)
```
global_step_150_merged/
├── model.safetensors              # Merged model weights (~12.8GB total)
├── config.json                     # Model configuration
├── generation_config.json          # Generation settings
├── tokenizer.json                  # Tokenizer
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
├── extra_state_merged.pt          # Merged training state
└── lr_scheduler_state.json        # Human-readable scheduler state
```

---

## Usage Examples

### Example 1: Default Output Path
```bash
bash scripts/convert_checkpoint.sh \
    verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150
```
Creates: `verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150_merged/`

### Example 2: Custom Output Path
```bash
bash scripts/convert_checkpoint.sh \
    verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150 \
    my_models/webqsp_merged
```
Creates: `my_models/webqsp_merged/`

### Example 3: Direct Python (with environment setup)
```bash
source /home/yeopjin/orcd/pool/init_general.sh

python3 scripts/convert_checkpoint_simple.py \
    /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150 \
    --output my_custom_output \
    --dtype bfloat16
```

---

## How It Works

### The 4 `.pt` Files Explained

When training with FSDP on 4 GPUs:
- Each GPU saves **1/4 of the model parameters** (different shards)
- **NOT identical** - each contains different layers/weights
- Total size: 4 × 3.2GB = 12.8GB (full model)

**Example parameter distribution:**
```
rank_0.pt: model.embed_tokens, layers.0-7.self_attn.q_proj (shard 0)
rank_1.pt: layers.8-15.self_attn.k_proj (shard 1)
rank_2.pt: layers.16-23.mlp.gate_proj (shard 2)
rank_3.pt: layers.24-31.lm_head (shard 3)
```

### Conversion Process

1. **Load all 4 shards** into CPU memory
2. **Merge sharded parameters**:
   - Replicated parameters (identical) → use any copy
   - Sharded parameters (different) → concatenate
3. **Save as single HuggingFace model**
4. **Copy tokenizer and configs**
5. **Merge extra states** (lr_scheduler, RNG)

---

## After Conversion: Using the Merged Model

### Load for Inference

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load model
model = AutoModelForCausalLM.from_pretrained(
    "verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150_merged",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(
    "verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150_merged"
)

# Generate
inputs = tokenizer("What is the capital of France?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_length=100)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Load for Further Training

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
import torch

# Load model
model = AutoModelForCausalLM.from_pretrained("path/to/merged")
tokenizer = AutoTokenizer.from_pretrained("path/to/merged")

# Setup optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Restore lr_scheduler state
extra_state = torch.load("path/to/merged/extra_state_merged.pt")
scheduler = get_scheduler("linear", optimizer=optimizer, num_training_steps=1000)
scheduler.load_state_dict(extra_state["lr_scheduler"])

# Restore RNG state (for reproducibility)
if "rng" in extra_state:
    torch.set_rng_state(extra_state["rng"]["cpu"])
    torch.cuda.set_rng_state_all(extra_state["rng"]["cuda"])

# Continue training...
```

### Use in Evaluation Script

You can now use the merged checkpoint path directly:

```bash
# Before: requires 4 GPUs
python verl/trainer/main_eval.py \
    actor_rollout_ref.model.path=verl_checkpoints/.../global_step_150/actor

# After: works on any GPU count
python verl/trainer/main_eval.py \
    actor_rollout_ref.model.path=verl_checkpoints/.../global_step_150_merged
```

---

## Advanced Options

### Change Output Data Type

```bash
python3 scripts/convert_checkpoint_simple.py \
    /path/to/checkpoint \
    --dtype float16  # Options: float32, float16, bfloat16
```

### Different Actor Directory

```bash
python3 scripts/convert_checkpoint_simple.py \
    /path/to/checkpoint \
    --actor-dir critic  # Default: actor
```

---

## Troubleshooting

### Issue: Out of Memory

**Solution 1:** The script runs on CPU, so you don't need GPU. But if you run out of RAM:
- Close other programs
- Or use a machine with more RAM (the model needs ~13GB+ RAM)

**Solution 2:** Use float16 instead of bfloat16:
```bash
python3 scripts/convert_checkpoint_simple.py /path/to/checkpoint --dtype float16
```

### Issue: Missing Checkpoint Files

Error: `Missing checkpoint for rank X`

**Check:**
```bash
ls global_step_150/actor/model_world_size_4_rank_*.pt
```

Should show: `rank_0.pt`, `rank_1.pt`, `rank_2.pt`, `rank_3.pt`

### Issue: Script Can't Find Modules

**Solution:** Make sure to source the init script:
```bash
source /home/yeopjin/orcd/pool/init_general.sh
python3 scripts/convert_checkpoint_simple.py ...
```

Or use the wrapper:
```bash
bash scripts/convert_checkpoint.sh ...
```

---

## Files Created

This conversion tool created the following files:

1. **Main conversion script (simple version)**:
   - `scripts/convert_checkpoint_simple.py` - CPU/single-GPU converter

2. **Advanced conversion script (multi-GPU)**:
   - `verl/utils/checkpoint/convert_fsdp_to_hf.py` - Uses torchrun (faster for large models)

3. **Wrapper script**:
   - `scripts/convert_checkpoint.sh` - Easy-to-use bash wrapper

4. **Documentation**:
   - `scripts/README_convert_checkpoint.md` - Detailed guide
   - `CHECKPOINT_CONVERSION_GUIDE.md` - This file

---

## Summary

| Question | Answer |
|----------|--------|
| **Are the 4 .pt files identical?** | No - each contains different parameter shards |
| **Do I need all 4 files?** | Yes - they combine to form the complete model |
| **Can I load on 1 GPU?** | Yes, after conversion with this script |
| **What gets preserved?** | Model weights, tokenizer, configs, lr_scheduler, RNG state |
| **How long does conversion take?** | ~2-5 minutes (depends on disk speed) |
| **Do I need GPUs to convert?** | No - runs on CPU |

---

## Contact

For issues with this conversion tool, check:
1. This guide
2. `scripts/README_convert_checkpoint.md`
3. Script help: `python3 scripts/convert_checkpoint_simple.py --help`
