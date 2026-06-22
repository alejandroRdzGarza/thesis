#!/bin/bash
# setup_oft.sh — Install OpenVLA-OFT repo and download the LIBERO-Spatial checkpoint.
#
# Run ONCE on the RunPod pod after cloning the thesis repo.
# Use tmux so the download survives a disconnect:
#   tmux new -s setup
#   bash /workspace/thesis/runpod/setup_oft.sh
#
# Prerequisites:
#   - huggingface-cli must be logged in:
#       huggingface-cli login        (paste your read token when prompted)
#   - OR set HF_TOKEN env var:
#       HF_TOKEN=hf_xxx bash runpod/setup_oft.sh

set -e

OFT_REPO="/workspace/openvla_oft_repo"
MODEL_DEST="/workspace/vla_model"
HF_MODEL="moojink/openvla-7b-oft-finetuned-libero-spatial"

# ── 1. HuggingFace login (optional — model is public) ────────────────────────
if [[ -n "${HF_TOKEN}" ]]; then
    echo "=== Logging in to HuggingFace with provided token ==="
    huggingface-cli login --token "${HF_TOKEN}"
else
    echo "=== Skipping HuggingFace login (model is public) ==="
fi

# ── 2. Clone and install openvla-oft repo ────────────────────────────────────
if [[ -d "${OFT_REPO}" ]]; then
    echo "=== openvla-oft repo already at ${OFT_REPO} — pulling latest ==="
    git -C "${OFT_REPO}" pull --ff-only
else
    echo "=== Cloning moojink/openvla-oft ==="
    git clone https://github.com/moojink/openvla-oft.git "${OFT_REPO}"
fi

echo "=== Installing openvla-oft package ==="
pip install -e "${OFT_REPO}" --quiet

# Verify key imports work
echo "=== Verifying OFT imports ==="
python - <<'PYEOF'
from experiments.robot.libero.run_libero_eval import GenerateConfig
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
print(f"  OK: PROPRIO_DIM={PROPRIO_DIM}  NUM_ACTIONS_CHUNK={NUM_ACTIONS_CHUNK}")
PYEOF

# ── 3. Download model weights ────────────────────────────────────────────────
if [[ -d "${MODEL_DEST}" && -f "${MODEL_DEST}/config.json" ]]; then
    EXISTING=$(python -c "import json; d=json.load(open('${MODEL_DEST}/config.json')); print(d.get('_name_or_path','unknown'))" 2>/dev/null || echo "unknown")
    if echo "${EXISTING}" | grep -q "oft"; then
        echo "=== OFT weights already at ${MODEL_DEST} (${EXISTING}) — skipping download ==="
    else
        echo "=== Found non-OFT model at ${MODEL_DEST} (${EXISTING}) — removing and re-downloading ==="
        rm -rf "${MODEL_DEST}"
    fi
fi

if [[ ! -d "${MODEL_DEST}" ]]; then
    echo "=== Downloading ${HF_MODEL} to ${MODEL_DEST} ==="
    echo "    (~14 GB — this will take several minutes)"
    pip install -q "huggingface_hub[cli]"
    huggingface-cli download "${HF_MODEL}" \
        --local-dir "${MODEL_DEST}"
    echo "=== Download complete ==="
    du -sh "${MODEL_DEST}"
fi

# ── 4. Done ──────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete.  Start the server with:"
echo ""
echo "  OPENVLA_MODEL_PATH=${MODEL_DEST} \\"
echo "  OPENVLA_OFT_REPO=${OFT_REPO} \\"
echo "  python /workspace/thesis/VLA-Model/openvla/openvla_server.py"
echo "========================================================"
