# FSDP Checkpoint to HuggingFace Conversion Tools

This directory contains tools to convert FSDP sharded checkpoints (4 `.pt` files) into a single merged HuggingFace model.

## Quick Start

```bash
cd /home/yeopjin/orcd/pool/workspace/RL_KG

bash conversion_model_weights/convert_checkpoint.sh \
    verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150
```

This creates: `global_step_150_merged/` with your merged model.

## Files in This Directory

- **convert_checkpoint_simple.py** - Main conversion script (CPU-based, no GPU needed)
- **convert_checkpoint.sh** - Bash wrapper for easy execution
- **convert_fsdp_to_hf.py** - Advanced multi-GPU converter (for very large models)
- **CHECKPOINT_CONVERSION_GUIDE.md** - Complete usage guide
- **README_convert_checkpoint.md** - Detailed technical documentation
- **README.md** - This file

## GPU Requirements

**The simple script (default) does NOT require GPUs:**
- ✅ Runs entirely on CPU
- ✅ Needs ~15-20GB RAM
- ✅ GPUs can stay busy with other work
- ✅ Takes ~3-5 minutes

## Example Usage

### Default output (creates `{checkpoint}_merged/`)
```bash
bash conversion_model_weights/convert_checkpoint.sh \
    verl_checkpoints/my-model/global_step_150
```

### Custom output path
```bash
bash conversion_model_weights/convert_checkpoint.sh \
    verl_checkpoints/my-model/global_step_150 \
    my_models/my_merged_model
```

### Direct Python usage
```bash
source /home/yeopjin/orcd/pool/init_general.sh

python3 conversion_model_weights/convert_checkpoint_simple.py \
    verl_checkpoints/my-model/global_step_150 \
    --output my_output \
    --dtype bfloat16
```

## What Gets Converted

**Input:** 4 sharded checkpoints (3.2GB each) + extra states
```
global_step_150/actor/
├── model_world_size_4_rank_0.pt
├── model_world_size_4_rank_1.pt
├── model_world_size_4_rank_2.pt
├── model_world_size_4_rank_3.pt
└── extra_state_world_size_4_rank_*.pt
```

**Output:** Single HuggingFace model (~12.8GB total)
```
global_step_150_merged/
├── model.safetensors
├── config.json
├── tokenizer files
└── extra_state_merged.pt
```

## Loading the Merged Model

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "verl_checkpoints/my-model/global_step_150_merged",
    torch_dtype="auto",
    device_map="auto"
)
```

## For More Information

See [CHECKPOINT_CONVERSION_GUIDE.md](./CHECKPOINT_CONVERSION_GUIDE.md) for complete documentation.
