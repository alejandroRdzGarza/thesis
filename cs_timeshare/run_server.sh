#!/bin/bash
# Start the OpenVLA server on a CS timeshare machine.
# Run this inside a tmux session so it survives logout.
#
# Usage (on vanilla / cream):
#   tmux new -s vla
#   bash ~/thesis/cs_timeshare/run_server.sh
#
# Override the GPU at the command line if needed:
#   GPU=0 bash ~/thesis/cs_timeshare/run_server.sh

set -e

# UCL CS machines require CS_OS; not always set on vanilla's login shell.
export CS_OS="${CS_OS:-linux}"

THESIS_DIR="/cs/student/project_msc/2025/rai/jesusr01/thesis"
VENV_DIR="${THESIS_DIR}/venv"
MODEL_PATH="${OPENVLA_MODEL_PATH:-${THESIS_DIR}/vla_model}"
SERVER_SCRIPT="${THESIS_DIR}/VLA-Model/openvla/openvla_server.py"
PORT="${OPENVLA_PORT:-8000}"

# --- GPU visibility ---------------------------------------------------------
# Default GPU=0. Override with: GPU=1 bash run_server.sh
# Run `nvidia-smi` first to find a GPU with free memory.
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
echo "=== CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES} ==="

# --- Python -----------------------------------------------------------------
if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "ERROR: venv not found at $VENV_DIR"
    echo "Run: bash ${THESIS_DIR}/cs_timeshare/setup_env.sh"
    exit 1
fi
source "$VENV_DIR/bin/activate"

# --- Model ------------------------------------------------------------------
if [[ ! -d "$MODEL_PATH" ]]; then
    echo "ERROR: model weights not found at $MODEL_PATH"
    echo "Upload them from your Mac: bash cs_timeshare/upload_weights.sh"
    exit 1
fi

echo "=== Server starting on $(hostname):$PORT ==="
echo "$(hostname)" > "$HOME/openvla_node.txt"

OPENVLA_MODEL_PATH="$MODEL_PATH" python -u "$SERVER_SCRIPT" 2>&1
