#!/bin/bash

# KG-R1 Main Evaluation Script for CWQ with Turn 7
# Usage:
#   ./eval_qwen_3b_cwq_f1_turn7.sh                    # Use all defaults
#   ./eval_qwen_3b_cwq_f1_turn7.sh /path/to/ckpt      # Specify checkpoint
#   ./eval_qwen_3b_cwq_f1_turn7.sh /path/to/ckpt DATASET  # Specify checkpoint and dataset
#   ./eval_qwen_3b_cwq_f1_turn7.sh /path/to/ckpt DATASET 300  # Specify everything

# ==================== CONFIGURATION ====================
# You can modify these variables or override them via command line arguments

# Default checkpoint to evaluate (set your checkpoint path here)
DEFAULT_CHECKPOINT="/nobackup/users/yeopjin/workspace/KG-R1/verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7"

# Default dataset to evaluate on
DEFAULT_DATASET="cwq"  # Options: "cwq" or "webqsp"

# Default checkpoint step (leave empty for auto-detection)
DEFAULT_STEP=""  # e.g., "300" to use global_step_300, or "" for latest

# Number of samples to evaluate (0 for all samples)
EVAL_SAMPLES=0

# K values for Pass@K evaluation
K_VALUES="1 2 3 4"

# WandB project name
WAND_PROJECT='KG-R1-Evaluation'

# Base model
BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'

# Validation batch size (adjust based on model size)
VAL_BATCH_SIZE=128

# Max turns for KG reasoning
MAX_TURNS=7

# ==================== END CONFIGURATION ====================

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data_kg'
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

rm -rf ~/.cache/torch/triton/

# Parse command line arguments (these override the defaults above)
CHECKPOINT_BASE="${1:-$DEFAULT_CHECKPOINT}"
DATASET_NAME="${2:-$DEFAULT_DATASET}"
CHECKPOINT_STEP="${3:-$DEFAULT_STEP}"  # Optional: specify which step to load

# Determine which checkpoint step to use
if [ -z "$CHECKPOINT_STEP" ]; then
    # Auto-detect latest checkpoint based on latest_checkpointed_iteration.txt
    if [ -f "$CHECKPOINT_BASE/latest_checkpointed_iteration.txt" ]; then
        LATEST_STEP=$(cat "$CHECKPOINT_BASE/latest_checkpointed_iteration.txt")
        CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$LATEST_STEP"
        echo "Auto-detected checkpoint: global_step_$LATEST_STEP"
    else
        # Fallback: find the highest numbered global_step directory
        LATEST_STEP=$(ls -d "$CHECKPOINT_BASE"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
        if [ -n "$LATEST_STEP" ]; then
            CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$LATEST_STEP"
            echo "Found latest checkpoint: global_step_$LATEST_STEP"
        else
            echo "Error: No checkpoint found in $CHECKPOINT_BASE"
            exit 1
        fi
    fi
else
    CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$CHECKPOINT_STEP"
    echo "Using specified checkpoint: global_step_$CHECKPOINT_STEP"
fi

# Calculate MAX_K for experiment naming
MAX_K=$(echo $K_VALUES | tr ' ' '\n' | sort -nr | head -1)
EXPERIMENT_NAME="${DATASET_NAME}-turn${MAX_TURNS}-eval-k${MAX_K}-n${EVAL_SAMPLES}-$(date +%m%d_%H%M)"

# Display configuration
echo "==================== EVALUATION CONFIGURATION ===================="
echo "Checkpoint base: $CHECKPOINT_BASE"
echo "Checkpoint path: $CHECKPOINT_PATH"
echo "Dataset: $DATASET_NAME"
echo "Eval samples: $EVAL_SAMPLES"
echo "K values: $K_VALUES"
echo "Max turns: $MAX_TURNS"
echo "Base model: $BASE_MODEL"
echo "Validation batch size: $VAL_BATCH_SIZE"
echo "Experiment name: $EXPERIMENT_NAME"
echo "=================================================================="

# Set dataset files
if [ "$DATASET_NAME" = "cwq" ]; then
    TEST_FILE="$DATA_DIR/cwq_search_augmented_initial_entities/test.parquet"
elif [ "$DATASET_NAME" = "webqsp" ]; then
    TEST_FILE="$DATA_DIR/webqsp_search_augmented_initial_entities/test.parquet"
else
    echo "Error: Unknown dataset $DATASET_NAME"
    exit 1
fi

# Verify test file exists
if [ ! -f "$TEST_FILE" ]; then
    echo "Error: Test file not found: $TEST_FILE"
    echo "Please run: python scripts/data_kg/setup_datasets.py --save_path data_kg"
    exit 1
fi

# Create output directory
OUTPUT_DIR="eval_results/eval_kg-r1/$EXPERIMENT_NAME"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Starting evaluation..."
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

PYTHONUNBUFFERED=1 python -m verl.trainer.main_eval \
    n_rollout_eval=$MAX_K \
    k_values="[$(echo $K_VALUES | sed 's/ /,/g')]" \
    eval_samples=$EVAL_SAMPLES \
    mode=kg-search \
    +save_detailed_results=true \
    actor_rollout_ref.rollout.search.max_turns=$MAX_TURNS \
    actor_rollout_ref.rollout.search.timeout=120 \
    actor_rollout_ref.rollout.search.search_url="http://127.0.0.1:8001/retrieve" \
    reward_model.reward_kwargs.max_turns=$MAX_TURNS \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.project_name=$WAND_PROJECT \
    trainer.default_local_dir="$OUTPUT_DIR" \
    data.train_files="$TEST_FILE" \
    data.val_files="$TEST_FILE" \
    data.val_batch_size=$VAL_BATCH_SIZE \
    data.prompt_augmentation.enable=true \
    data.prompt_augmentation.guideline_level=detailed_flat \
    data.prompt_augmentation.hint_steps=500 \
    actor_rollout_ref.model.path=$BASE_MODEL \
    trainer.resume_mode=resume_path \
    trainer.resume_from_path="$CHECKPOINT_PATH" \
    reward_model.reward_kwargs.debug_log_dir="${EXPERIMENT_NAME}_debug.log" \
    2>&1 | tee "$OUTPUT_DIR/evaluation.log"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "=================================================================="
    echo "✅ Evaluation completed successfully!"
    echo "=================================================================="
    echo "Results saved to: $OUTPUT_DIR"
    echo "Log file: $OUTPUT_DIR/evaluation.log"
    echo "=================================================================="
else
    echo ""
    echo "❌ Evaluation failed! Check the log for details."
    exit 1
fi
