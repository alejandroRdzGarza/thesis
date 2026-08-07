#!/bin/bash
# tunnel_ucl.sh — Forward localhost:8000 → gadwall-l:8000 via knuckles gateway
#
# Usage: bash runpod/tunnel_ucl.sh
# Keep running in a terminal while benchmarking from Mac.

UCL_USER=${UCL_USER:-jesusr01}
GATEWAY=knuckles.cs.ucl.ac.uk
GPU_HOST=${GPU_HOST:-shoveler-l}

echo "Tunnelling localhost:8000 -> ${GPU_HOST}:8000 via ${GATEWAY}"
ssh -N \
  -L 8000:${GPU_HOST}:8000 \
  ${UCL_USER}@${GATEWAY}
