#!/bin/bash
# tunnel_ucl.sh — Forward localhost:8000 → $GPU_HOST:8000 via the CS gateway
#
# Usage: bash runpod/tunnel_ucl.sh
# Keep running in a terminal while benchmarking from Mac.

UCL_USER="${UCL_USER:?set UCL_USER to your UCL CS username}"
GATEWAY="${UCL_GATEWAY:?set UCL_GATEWAY, e.g. gateway.cs.ucl.ac.uk}"
GPU_HOST="${GPU_HOST:?set GPU_HOST to the lab GPU machine}"

echo "Tunnelling localhost:8000 -> ${GPU_HOST}:8000 via ${GATEWAY}"
ssh -N \
  -L 8000:${GPU_HOST}:8000 \
  ${UCL_USER}@${GATEWAY}
