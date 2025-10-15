#!/bin/bash

# Unified KG Server Startup Script for Other KGQA Benchmarks
# Usage: ./start_unified_kg_server.sh <dataset> [workers]
# Datasets: simpleqa, grailqa, trex, qald10en, zero_shot_re

if [ $# -lt 1 ]; then
    echo "❌ Usage: $0 <dataset> [workers]"
    echo "📋 Available datasets: simpleqa, grailqa, trex, qald10en, zero_shot_re"
    echo "📋 Example: $0 simpleqa 16"
    echo "📋 Example: $0 grailqa 8"
    exit 1
fi

DATASET=$1
WORKERS=${2:-4}  # Default 4 workers

# Dataset to port mapping
declare -A DATASET_PORTS=(
    ["simpleqa"]="9001"
    ["grailqa"]="9000"
    ["trex"]="9011"
    ["qald10en"]="9010"
    ["zero_shot_re"]="9012"
)

# Get port for dataset
PORT=${DATASET_PORTS[$DATASET]}
if [ -z "$PORT" ]; then
    echo "❌ Error: Unknown dataset '$DATASET'"
    echo "📋 Supported datasets: ${!DATASET_PORTS[@]}"
    exit 1
fi

echo "🚀 Starting KG Server for $DATASET..."

# Set common environment variables
export KG_BASE_DATA_PATH="./data_kg"
export KG_SUPPORTED_KGS="$DATASET"
export KG_USE_ENTITIES_TEXT="true"  # All other benchmarks use entities_text.txt
export KG_RELATION_FORMAT="flat"
export KG_ENABLED_ACTIONS="get_head_relations,get_tail_relations,get_head_entities,get_tail_entities,get_conditional_relations"
export KG_THREAD_WORKERS="$WORKERS"

# Kill any existing server on this port
EXISTING_PID=$(lsof -ti:$PORT 2>/dev/null)
if [ -n "$EXISTING_PID" ]; then
    echo "🔄 Killing existing server on port $PORT (PID: $EXISTING_PID)"
    kill -9 $EXISTING_PID 2>/dev/null || true
    sleep 2
fi

echo "📋 Configuration:"
echo "   - Dataset: $DATASET"
echo "   - Port: $PORT"
echo "   - Entities: entities_text.txt (human-readable names)"
echo "   - Workers: $WORKERS threads"
echo "   - Data Path: ./data_kg"
echo "   - Usage: $0 <dataset> [workers] (default: 4)"

# Increase system limits for high concurrency
ulimit -n 65536   # File descriptors
ulimit -s 262144  # Stack size (256MB per thread)

# Start the server
echo "🔧 Launching uvicorn server with optimized limits..."
echo "   - File descriptors: $(ulimit -n)"
echo "   - Stack size: $(ulimit -s)KB"

uvicorn kg_r1.search.server:app --host "0.0.0.0" --port $PORT --workers 4 \
  --backlog 2048 \
  --limit-max-requests 50000 \
  --timeout-keep-alive 60 \
  --timeout-graceful-shutdown 60
