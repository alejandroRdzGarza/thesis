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

# Available LIBERO-Spatial checkpoints:
#
#   openvla/openvla-7b-finetuned-libero-spatial   (base model, ~14 GB)
#     Single action per inference, ~1-5s/call on GPU.
#     Gets ~84.7% TSR on LIBERO-Spatial when called synchronously,
#     but degrades badly over HTTP due to action repetition.
#
#   openvla/openvla-oft-libero-spatial             (OFT model, ~14 GB) ← RECOMMENDED
#     Parallel decoding + action chunking, 25-50x faster (~0.1-0.5s/call).
#     This is the model AEGIS uses as their simulation baseline (50.9% TSR
#     on SafeLIBERO WITH obstacles). Compatible with the same /act server endpoint.
#     Set UNNORM_KEY=libero_spatial (same as base).
#
# Override via env var:
#   HF_MODEL=openvla/openvla-7b-finetuned-libero-spatial bash download_weights.sh
HF_MODEL="${HF_MODEL:-openvla/openvla-oft-libero-spatial}"
DEST="/workspace/vla_model"

if [[ -d "$DEST" && -f "$DEST/model.safetensors.index.json" ]]; then
    echo "Weights already present at $DEST — nothing to do."
    exit 0
fi

echo "=== Installing huggingface_hub CLI ==="
pip install -q "huggingface_hub[cli]"

echo "=== Downloading ${HF_MODEL} to $DEST ==="
echo "    (~14 GB — this will take several minutes)"
echo ""

huggingface-cli download "${HF_MODEL}" \
    --local-dir "$DEST" \
    --local-dir-use-symlinks False

echo ""
echo "=== Download complete ==="
echo "    Weights at: $DEST"
du -sh "$DEST"
