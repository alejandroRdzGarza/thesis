#!/bin/bash
# Download OpenVLA-7B weights from HuggingFace to /workspace/vla_model.
# Run this ONCE on the RunPod pod after cloning the repo.
#
# Usage (inside the pod terminal or via SSH):
#   bash ~/VLA-Benchmark/runpod/download_weights.sh
#
# The download is ~14 GB. Use tmux so it survives a disconnect:
#   tmux new -s download
#   bash ~/VLA-Benchmark/runpod/download_weights.sh

set -e

DEST="/workspace/vla_model"

if [[ -d "$DEST" && -f "$DEST/model.safetensors.index.json" ]]; then
    echo "Weights already present at $DEST — nothing to do."
    exit 0
fi

echo "=== Installing huggingface_hub CLI ==="
pip install -q "huggingface_hub[cli]"

echo "=== Downloading openvla/openvla-7b to $DEST ==="
echo "    (~14 GB — this will take several minutes)"
echo ""

huggingface-cli download openvla/openvla-7b \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False

echo ""
echo "=== Download complete ==="
echo "    Weights at: $DEST"
du -sh "$DEST"
