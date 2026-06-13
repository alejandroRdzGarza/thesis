#!/bin/bash
# Start the OpenVLA server on a RunPod pod.
# Run this inside a tmux session so it survives disconnect.
#
# Usage:
#   tmux new -s vla
#   bash ~/VLA-Benchmark/runpod/run_server.sh
#
# Override GPU:
#   GPU=1 bash ~/VLA-Benchmark/runpod/run_server.sh

set -e

REPO_DIR="/workspace/VLA-Benchmark"
MODEL_PATH="${OPENVLA_MODEL_PATH:-/workspace/vla_model}"
SERVER_SCRIPT="${REPO_DIR}/VLA-Model/openvla/openvla_server.py"
PORT="${OPENVLA_PORT:-8000}"

export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="

# --- Dependencies -----------------------------------------------------------
# RunPod PyTorch images ship with torch/CUDA pre-installed.
# Install any missing Python deps before starting.
pip install -q fastapi uvicorn[standard] bitsandbytes Pillow transformers accelerate

# --- Checks -----------------------------------------------------------------
if [[ ! -f "$SERVER_SCRIPT" ]]; then
    echo "ERROR: server script not found at $SERVER_SCRIPT"
    echo "  Clone the repo first: git clone https://github.com/YOUR_USERNAME/VLA-Benchmark.git $REPO_DIR"
    exit 1
fi

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
    echo "ERROR: model weights not found at $MODEL_PATH"
    echo "  Run: bash $REPO_DIR/runpod/download_weights.sh"
    exit 1
fi

echo "=== Server starting on port $PORT ==="
echo "=== Model: $MODEL_PATH ==="

OPENVLA_MODEL_PATH="$MODEL_PATH" \
OPENVLA_PORT="$PORT" \
python -u "$SERVER_SCRIPT" 2>&1
