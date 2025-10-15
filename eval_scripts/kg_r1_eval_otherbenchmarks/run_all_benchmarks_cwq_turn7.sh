#!/bin/bash

# Unified script to run all 5 benchmark tests for KG-R1 model evaluation
# Optimized for cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7 checkpoint
# CWQ Turn7 version

# ==================== CONFIGURATION ====================
DEFAULT_CHECKPOINT="/nobackup/users/yeopjin/workspace/KG-R1/verl_checkpoints/cwq-KG-r1-grpo-qwen2.5-3b-it_f1_turn7"
DEFAULT_STEP=""
EVAL_SAMPLES=0  # 0 = evaluate all samples
K_VALUES="1 2 3 4"
BASE_MODEL='Qwen/Qwen2.5-3B-Instruct'
VAL_BATCH_SIZE=128
MAX_TURNS=7  # Matches the turn7 checkpoint

# Session timestamp for unique experiment naming
SESSION_TIMESTAMP=$(date +%m%d_%H%M)
SESSION_DIR="cwq_turn7_all_benchmarks_${SESSION_TIMESTAMP}"

# Benchmark configurations: dataset_name:port:data_subdir
BENCHMARKS=(
    "trex:9011:trex_search_augmented_initial_entities"
    "zero_shot_re:9012:zero_shot_re_search_augmented_initial_entities"
    "grailqa:9000:grailqa_search_augmented_initial_entities"
    "simpleqa:9001:simpleqa_search_augmented_initial_entities"
    "qald10en:9010:qald10en_search_augmented_initial_entities"
)

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DATA_DIR='data_kg'
export VLLM_ATTENTION_BACKEND=XFORMERS
export HYDRA_FULL_ERROR=1
export PYTHONUNBUFFERED=1

# Unified server script path
UNIFIED_SERVER_SCRIPT="./eval_scripts/kg_r1_eval_otherbenchmarks/start_unified_kg_server.sh"

rm -rf ~/.cache/torch/triton/

# Function to log to both console and session log
log_both() {
    echo "$@" | tee -a "$SESSION_LOG"
}

# Function to log only to session log (for detailed output)
log_session() {
    echo "$@" >> "$SESSION_LOG"
}

log_both "=================================================================="
log_both "🚀 RUNNING ALL BENCHMARKS FOR KG-R1 CWQ TURN7 MODEL"
log_both "=================================================================="

# Parse command line arguments
CHECKPOINT_BASE="${1:-$DEFAULT_CHECKPOINT}"
CHECKPOINT_STEP="${2:-$DEFAULT_STEP}"

# Determine checkpoint path
if [ -z "$CHECKPOINT_STEP" ]; then
    if [ -f "$CHECKPOINT_BASE/latest_checkpointed_iteration.txt" ]; then
        LATEST_STEP=$(cat "$CHECKPOINT_BASE/latest_checkpointed_iteration.txt")
        CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$LATEST_STEP"
        log_both "Auto-detected checkpoint: global_step_$LATEST_STEP"
    else
        LATEST_STEP=$(ls -d "$CHECKPOINT_BASE"/global_step_* 2>/dev/null | sed 's/.*global_step_//' | sort -n | tail -1)
        if [ -n "$LATEST_STEP" ]; then
            CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$LATEST_STEP"
            log_both "Found latest checkpoint: global_step_$LATEST_STEP"
        else
            log_both "❌ Error: No checkpoint found in $CHECKPOINT_BASE"
            exit 1
        fi
    fi
else
    CHECKPOINT_PATH="$CHECKPOINT_BASE/global_step_$CHECKPOINT_STEP"
    log_both "Using specified checkpoint: global_step_$CHECKPOINT_STEP"
fi

# Create main output directory
MAIN_OUTPUT_DIR="eval_results/eval_kg-r1_other_benchmarks/$SESSION_DIR"
mkdir -p "$MAIN_OUTPUT_DIR"

# Set up comprehensive session logging
SESSION_LOG="$MAIN_OUTPUT_DIR/session_complete.log"
SUMMARY_LOG="$MAIN_OUTPUT_DIR/session_summary.log"

# Initialize session log with header
cat > "$SESSION_LOG" << EOF
================================================================
KG-R1 CWQ TURN7 BENCHMARKS - COMPLETE SESSION LOG
================================================================
Started: $(date)
Checkpoint: $CHECKPOINT_PATH
Session: $SESSION_DIR
Output Directory: $MAIN_OUTPUT_DIR
Max Turns: $MAX_TURNS
================================================================

EOF

log_both "Checkpoint: $CHECKPOINT_PATH"
log_both "Session: $SESSION_DIR"
log_both "Benchmarks: ${#BENCHMARKS[@]} total"
log_both "Max turns: $MAX_TURNS"
log_both "Eval samples: $EVAL_SAMPLES per benchmark"
log_both "=================================================================="

echo "Output directory: $MAIN_OUTPUT_DIR"
echo "Session log: $SESSION_LOG"
echo "Summary log: $SUMMARY_LOG"
echo ""

# Calculate MAX_K for evaluation
MAX_K=$(echo $K_VALUES | tr ' ' '\n' | sort -nr | head -1)

# Global cleanup function to kill all servers
cleanup_all_servers() {
    log_both ""
    log_both "🧹 Cleaning up all KG servers..."
    for benchmark_config in "${BENCHMARKS[@]}"; do
        IFS=':' read -r dataset_name port data_subdir <<< "$benchmark_config"
        session_name="${dataset_name}_kg_server"
        screen -S "$session_name" -X quit 2>/dev/null || true
        log_both "  ✅ Cleaned up $dataset_name server"
    done
    log_both "🎉 All servers cleaned up!"
}

# Set global cleanup trap
trap cleanup_all_servers EXIT INT TERM

# Function to run a single benchmark
run_benchmark() {
    local dataset_name="$1"
    local port="$2"
    local data_subdir="$3"
    local benchmark_num="$4"
    local total_benchmarks="$5"

    local experiment_name="${dataset_name}_cwq_turn7_${SESSION_TIMESTAMP}"
    local session_name="${dataset_name}_kg_server"
    local test_file="$DATA_DIR/$data_subdir/test.parquet"
    local search_url="http://127.0.0.1:$port/retrieve"
    local wand_project="KG-R1-Test-${dataset_name^}-CWQ-Turn7"

    # Set batch size: 64 for GrailQA, 128 for others
    local batch_size=$VAL_BATCH_SIZE
    if [ "$dataset_name" = "grailqa" ]; then
        batch_size=64
    fi

    local output_dir="$MAIN_OUTPUT_DIR/$dataset_name"
    mkdir -p "$output_dir"

    log_both ""
    log_both "=================================================================="
    log_both "📊 BENCHMARK [$benchmark_num/$total_benchmarks]: ${dataset_name^^}"
    log_both "=================================================================="
    log_both "Dataset: $dataset_name"
    log_both "Port: $port"
    log_both "Test file: $test_file"
    log_both "Output dir: $output_dir"
    log_both "Experiment: $experiment_name"
    log_both "=================================================================="

    # Verify test file exists
    if [ ! -f "$test_file" ]; then
        log_both "❌ Error: Test file not found: $test_file"
        log_both "Please ensure data is downloaded with: python scripts/data_kg/setup_datasets.py"
        return 1
    fi

    # Kill any existing screen session for this benchmark
    screen -S "$session_name" -X quit 2>/dev/null || true
    sleep 2

    log_both "🚀 Starting $dataset_name KG server in screen session: $session_name"
    screen -dmS "$session_name" bash -c "cd /nobackup/users/yeopjin/workspace/KG-R1 && $UNIFIED_SERVER_SCRIPT $dataset_name 4"

    # Wait for server to be ready (2 minutes)
    log_both "⏳ Waiting for $dataset_name server to be ready..."
    for i in {1..60}; do
        if curl -s -m 3 "http://127.0.0.1:$port/health" >/dev/null 2>&1; then
            log_both "✅ $dataset_name KG server is ready!"
            break
        fi
        if [ $i -eq 60 ]; then
            log_both "❌ $dataset_name server failed to start within 120s"
            screen -S "$session_name" -X quit 2>/dev/null || true
            return 1
        fi
        log_session "⏳ Waiting... ($((i*2))s/120s)"
        sleep 2
    done

    log_both "🎯 Starting $dataset_name evaluation..."

    # Run the evaluation
    log_session "Starting VERL evaluation for $dataset_name..."
    PYTHONUNBUFFERED=1 python -m verl.trainer.main_eval \
        n_rollout_eval=$MAX_K \
        k_values="[$(echo $K_VALUES | sed 's/ /,/g')]" \
        eval_samples=$EVAL_SAMPLES \
        mode=kg-search \
        +save_detailed_results=true \
        actor_rollout_ref.rollout.search.max_turns=$MAX_TURNS \
        actor_rollout_ref.rollout.search.timeout=120 \
        actor_rollout_ref.rollout.search.search_url="$search_url" \
        reward_model.reward_kwargs.max_turns=$MAX_TURNS \
        trainer.experiment_name="$experiment_name" \
        trainer.project_name="$wand_project" \
        trainer.default_local_dir="$output_dir" \
        data.train_files="$test_file" \
        data.val_files="$test_file" \
        data.val_batch_size=$batch_size \
        data.prompt_augmentation.enable=true \
        data.prompt_augmentation.guideline_level=detailed_flat \
        data.prompt_augmentation.hint_steps=500 \
        actor_rollout_ref.model.path="$BASE_MODEL" \
        trainer.resume_mode=resume_path \
        trainer.resume_from_path="$CHECKPOINT_PATH" \
        reward_model.reward_kwargs.debug_log_dir="${experiment_name}_debug.log" \
        2>&1 | tee "$output_dir/evaluation.log" | tee -a "$SESSION_LOG"

    local eval_status=$?

    # Clean up this benchmark's server
    log_both "🧹 Cleaning up $dataset_name server..."
    screen -S "$session_name" -X quit 2>/dev/null || true

    if [ $eval_status -eq 0 ]; then
        log_both "✅ $dataset_name evaluation completed successfully!"
        log_both "📁 Results saved to: $output_dir"
        log_both "📄 Log file: $output_dir/evaluation.log"
    else
        log_both "❌ $dataset_name evaluation failed!"
        return 1
    fi

    log_session "Completed $dataset_name evaluation at $(date)"
    log_both "=================================================================="
    log_both "✅ BENCHMARK [$benchmark_num/$total_benchmarks] COMPLETE: ${dataset_name^^}"
    log_both "=================================================================="

    return 0
}

# Main execution loop
log_both ""
log_both "🏁 Starting benchmark evaluation sequence..."
log_both ""
log_session "Starting main evaluation loop at $(date)..."

successful_benchmarks=0
failed_benchmarks=0
total_benchmarks=${#BENCHMARKS[@]}

start_time=$(date +%s)

for i in "${!BENCHMARKS[@]}"; do
    benchmark_config="${BENCHMARKS[$i]}"
    IFS=':' read -r dataset_name port data_subdir <<< "$benchmark_config"

    benchmark_num=$((i + 1))

    if run_benchmark "$dataset_name" "$port" "$data_subdir" "$benchmark_num" "$total_benchmarks"; then
        ((successful_benchmarks++))
        log_session "SUCCESS: $dataset_name completed successfully"
    else
        ((failed_benchmarks++))
        log_both "⚠️  Continuing with next benchmark despite failure..."
        log_session "FAILED: $dataset_name evaluation failed"
    fi

    # Small delay between benchmarks
    if [ $benchmark_num -lt $total_benchmarks ]; then
        log_both ""
        log_both "⏸️  Brief pause before next benchmark..."
        log_session "Brief 5-second pause before next benchmark"
        sleep 5
    fi
done

end_time=$(date +%s)
total_time=$((end_time - start_time))
total_time_formatted=$(printf "%02d:%02d:%02d" $((total_time/3600)) $((total_time%3600/60)) $((total_time%60)))

# Log comprehensive summary
log_both ""
log_both "=================================================================="
log_both "🎉 ALL BENCHMARKS COMPLETE!"
log_both "=================================================================="
log_both "✅ Successful: $successful_benchmarks/$total_benchmarks"
log_both "❌ Failed: $failed_benchmarks/$total_benchmarks"
log_both "⏱️  Total time: $total_time_formatted"
log_both "📁 All results saved to: $MAIN_OUTPUT_DIR"
log_both ""
log_both "📊 BENCHMARK SUMMARY:"
log_both "=================================================================="

# Create comprehensive summary log
cat > "$SUMMARY_LOG" << EOF
================================================================
KG-R1 CWQ TURN7 BENCHMARKS - SESSION SUMMARY
================================================================
Completed: $(date)
Session: $SESSION_DIR
Checkpoint: $CHECKPOINT_PATH
Total Runtime: $total_time_formatted
Max Turns: $MAX_TURNS

RESULTS SUMMARY:
✅ Successful: $successful_benchmarks/$total_benchmarks
❌ Failed: $failed_benchmarks/$total_benchmarks

BENCHMARK DETAILS:
EOF

for benchmark_config in "${BENCHMARKS[@]}"; do
    IFS=':' read -r dataset_name port data_subdir <<< "$benchmark_config"
    output_dir="$MAIN_OUTPUT_DIR/$dataset_name"

    if [ -f "$output_dir/evaluation.log" ]; then
        log_both "✅ $dataset_name: $output_dir"
        echo "✅ $dataset_name: $output_dir" >> "$SUMMARY_LOG"
    else
        log_both "❌ $dataset_name: FAILED or incomplete"
        echo "❌ $dataset_name: FAILED or incomplete" >> "$SUMMARY_LOG"
    fi
done

log_both "=================================================================="
log_both "🔍 To view detailed results:"
log_both "   Session log: $SESSION_LOG"
log_both "   Summary log: $SUMMARY_LOG"
log_both "   ls -la $MAIN_OUTPUT_DIR"
log_both "   cat $MAIN_OUTPUT_DIR/*/evaluation.log"
log_both "=================================================================="

# Complete summary log
cat >> "$SUMMARY_LOG" << EOF

FILES GENERATED:
- Complete session log: $SESSION_LOG
- Summary log: $SUMMARY_LOG
- Individual evaluation logs: $MAIN_OUTPUT_DIR/*/evaluation.log

TO VIEW RESULTS:
   ls -la $MAIN_OUTPUT_DIR
   cat $MAIN_OUTPUT_DIR/*/evaluation.log
================================================================
EOF

log_session "Session completed at $(date) with $successful_benchmarks/$total_benchmarks successful"

if [ $failed_benchmarks -eq 0 ]; then
    log_both "🎊 ALL BENCHMARKS SUCCESSFUL!"
    exit 0
else
    log_both "⚠️  Some benchmarks failed. Check logs above."
    exit 1
fi
