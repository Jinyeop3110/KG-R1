# FSDP Checkpoint Conversion Guide

This guide explains how to convert FSDP sharded checkpoints to a single HuggingFace model.

## Overview

When training with FSDP (Fully Sharded Data Parallel) on multiple GPUs, the model is saved as multiple shard files:
- `model_world_size_4_rank_0.pt` (GPU 0's shard)
- `model_world_size_4_rank_1.pt` (GPU 1's shard)
- `model_world_size_4_rank_2.pt` (GPU 2's shard)
- `model_world_size_4_rank_3.pt` (GPU 3's shard)

These scripts merge these shards into a single HuggingFace model that can be loaded on any GPU configuration.

---

## Quick Start

### Simple Method (CPU/Single GPU)

Use the simple script that runs on CPU or single GPU:

```bash
python scripts/convert_checkpoint_simple.py \
    /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150
```

This creates: `global_step_150_merged/`

### Custom Output Path

```bash
python scripts/convert_checkpoint_simple.py \
    /path/to/checkpoint/global_step_150 \
    --output /path/to/my_merged_model
```

### Options

```bash
python scripts/convert_checkpoint_simple.py \
    /path/to/checkpoint/global_step_150 \
    --output /path/to/output \
    --actor-dir actor \              # Subdirectory with checkpoints (default: actor)
    --dtype bfloat16                 # Output dtype: float32, float16, bfloat16
```

---

## Advanced Method (Multi-GPU - Faster)

For large models, use the distributed converter with torchrun:

```bash
# Use same number of GPUs as training (e.g., 4 GPUs)
torchrun --nproc_per_node=4 verl/utils/checkpoint/convert_fsdp_to_hf.py \
    --checkpoint_path /path/to/checkpoint/global_step_150 \
    --output_path /path/to/output
```

**Advantages:**
- Faster for large models
- More memory efficient
- Uses FSDP's built-in merging logic

**Requirements:**
- Must run with same number of GPUs as training (world_size)
- Requires `torchrun`

---

## Output Structure

The merged checkpoint will contain:

```
global_step_150_merged/
├── config.json                      # Model configuration
├── generation_config.json           # Generation settings
├── model.safetensors               # Merged model weights (or multiple shards)
├── tokenizer.json                   # Tokenizer
├── tokenizer_config.json
├── vocab.json
├── merges.txt
├── special_tokens_map.json
├── extra_state_merged.pt           # Merged training state (lr_scheduler, rng)
└── lr_scheduler_state.json         # Human-readable lr_scheduler state
```

---

## Usage After Conversion

### Load the Merged Model

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "/path/to/global_step_150_merged",
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("/path/to/global_step_150_merged")

# Generate
inputs = tokenizer("Hello, how are you?", return_tensors="pt")
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0]))
```

### Resume Training (if needed)

```python
import torch

# Load extra state
extra_state = torch.load("/path/to/global_step_150_merged/extra_state_merged.pt")

# Restore lr_scheduler
optimizer = ...  # your optimizer
lr_scheduler = ...  # your scheduler
lr_scheduler.load_state_dict(extra_state["lr_scheduler"])

# Restore RNG state
if "rng" in extra_state:
    torch.set_rng_state(extra_state["rng"]["cpu"])
    torch.cuda.set_rng_state_all(extra_state["rng"]["cuda"])
```

---

## Comparison: Simple vs Advanced

| Feature | Simple Script | Advanced Script |
|---------|--------------|-----------------|
| **GPU Requirement** | None (runs on CPU) | Same as training (e.g., 4 GPUs) |
| **Speed** | Slower | Faster |
| **Memory** | High (loads all shards) | Low (distributed loading) |
| **Setup** | Easy (no torchrun) | Requires torchrun |
| **Use Case** | Small-medium models | Large models (>7B) |

---

## Troubleshooting

### Error: "Missing checkpoint for rank X"

Make sure all shard files exist:
```bash
ls global_step_150/actor/model_world_size_4_rank_*.pt
```

You should see all ranks (0, 1, 2, 3).

### Out of Memory Error

**Solution 1:** Use the advanced script with torchrun (distributes memory)

**Solution 2:** Reduce dtype:
```bash
python scripts/convert_checkpoint_simple.py \
    /path/to/checkpoint \
    --dtype float16  # or bfloat16
```

### Keys Don't Match

If you get state_dict key mismatch errors, the simple script may not handle your model's sharding correctly. Use the advanced script instead.

---

## Example

```bash
# Convert checkpoint
python scripts/convert_checkpoint_simple.py \
    /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150

# Output:
# ======================================================================
# FSDP Checkpoint Converter (Simple Version)
# ======================================================================
# Checkpoint:    .../global_step_150
# Actor dir:     .../global_step_150/actor
# Output:        .../global_step_150_merged
# Dtype:         bfloat16
# ======================================================================
# ✓ Detected world_size=4 from checkpoint
#
# Loading 4 checkpoint shards...
#   [1/4] Loading model_world_size_4_rank_0.pt... (XXX keys)
#   [2/4] Loading model_world_size_4_rank_1.pt... (XXX keys)
#   [3/4] Loading model_world_size_4_rank_2.pt... (XXX keys)
#   [4/4] Loading model_world_size_4_rank_3.pt... (XXX keys)
#
# Merging shards into full state dict...
# ✓ Merged XXX parameters:
#   - XXX replicated parameters
#   - XXX sharded parameters
#
# Saving merged model to .../global_step_150_merged
#   Creating model from config...
#   Saving model with dtype=bfloat16...
#   Copying tokenizer files...
# ✓ Model saved successfully
#
# Merging extra_state files...
# ✓ Saved extra_state to .../extra_state_merged.pt
# ✓ Saved lr_scheduler state to .../lr_scheduler_state.json
#
# ======================================================================
# ✅ Conversion complete!
# ======================================================================
```

---

## Notes

1. **The simple script uses a heuristic** to determine which parameters are sharded:
   - If a parameter is identical across all ranks → replicated (use any copy)
   - If a parameter differs → sharded (concatenate along dim 0)

2. **For production use on large models**, prefer the advanced script with torchrun for guaranteed correctness.

3. **The conversion preserves**:
   - ✅ Model weights
   - ✅ Tokenizer
   - ✅ Model config
   - ✅ Generation config
   - ✅ LR scheduler state
   - ✅ RNG state

4. **Not preserved** (only in advanced script with distributed):
   - ❌ Optimizer state (too large, usually not needed for inference)
