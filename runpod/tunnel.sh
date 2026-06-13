#!/bin/bash
# Forward the RunPod server port to localhost:8000.
# Run this on your LOCAL Mac.
#
# Usage:
#   bash runpod/tunnel.sh
#
# Find your pod's SSH details in the RunPod dashboard:
#   Pod → Connect → SSH over exposed TCP
# It will look like:
#   ssh root@HOSTNAME.runpod.io -p PORT
# or:
#   ssh root@connect.runpod.io -p PORT
#
# Set those values below, or pass them as env vars:
#   POD_HOST=abc123.runpod.io POD_PORT=12345 bash runpod/tunnel.sh

POD_HOST="${POD_HOST:-}"
POD_PORT="${POD_PORT:-}"
LOCAL_PORT=8000
REMOTE_PORT=8000

if [[ -z "$POD_HOST" || -z "$POD_PORT" ]]; then
    echo "ERROR: set POD_HOST and POD_PORT before running."
    echo ""
    echo "  Find them in the RunPod dashboard → Pod → Connect → SSH over exposed TCP"
    echo "  Then run:"
    echo "    POD_HOST=abc123.runpod.io POD_PORT=12345 bash runpod/tunnel.sh"
    exit 1
fi

echo "Tunnelling localhost:${LOCAL_PORT} -> ${POD_HOST}:${REMOTE_PORT}"
echo "  via SSH port ${POD_PORT}"
echo "Press Ctrl+C to close."
echo ""
echo "Once connected, reach the server at: http://127.0.0.1:${LOCAL_PORT}/act"
echo ""

ssh -N \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    -p "${POD_PORT}" \
    root@"${POD_HOST}"
