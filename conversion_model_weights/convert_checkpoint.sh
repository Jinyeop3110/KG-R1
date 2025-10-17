#!/bin/bash
# Wrapper script to convert FSDP checkpoint to HuggingFace format
#
# Usage:
#   bash conversion_model_weights/convert_checkpoint.sh /path/to/checkpoint/global_step_150 [output_path]
#
# Example:
#   bash conversion_model_weights/convert_checkpoint.sh \
#       /home/yeopjin/orcd/pool/workspace/RL_KG/verl_checkpoints/webqsp-KG-r1-grpo-qwen2.5-3b-it_Aug11_f1_turn7_rep1/global_step_150

# Source the initialization script
source /home/yeopjin/orcd/pool/init_general.sh

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

# Check arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <checkpoint_path> [output_path]"
    echo ""
    echo "Example:"
    echo "  $0 /path/to/checkpoint/global_step_150"
    echo "  $0 /path/to/checkpoint/global_step_150 /path/to/output"
    exit 1
fi

CHECKPOINT_PATH="$1"
OUTPUT_PATH="${2:-}"

# Build command
CMD="python3 $SCRIPT_DIR/convert_checkpoint_simple.py \"$CHECKPOINT_PATH\""

if [ -n "$OUTPUT_PATH" ]; then
    CMD="$CMD --output \"$OUTPUT_PATH\""
fi

# Run conversion
echo "========================================"
echo "Running checkpoint conversion..."
echo "========================================"
echo "Command: $CMD"
echo ""

eval $CMD
