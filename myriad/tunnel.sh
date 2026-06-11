#!/bin/bash
# Run this on your LOCAL Mac to forward the VLA server port to localhost:8000.
# Usage: bash myriad/tunnel.sh <ucl-username>
#
# Prereq: job must be running and openvla_node.txt must be written on Myriad.

UCL_USER="${1:?Usage: $0 <ucl-username>}"
MYRIAD="myriad.rc.ucl.ac.uk"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "Fetching compute node name from Myriad..."
NODE=$(ssh "${UCL_USER}@${MYRIAD}" "cat ~/openvla_node.txt 2>/dev/null")

if [[ -z "$NODE" ]]; then
    echo "ERROR: ~/openvla_node.txt not found on Myriad."
    echo "Make sure the job is running (check with: qstat)"
    exit 1
fi

echo "Tunnelling localhost:${LOCAL_PORT} -> ${NODE}:${REMOTE_PORT} via ${MYRIAD}"
echo "Press Ctrl+C to close the tunnel."
echo ""
echo "Once connected, your local scripts can reach the server at:"
echo "  http://127.0.0.1:${LOCAL_PORT}/act"

ssh -N -L "${LOCAL_PORT}:${NODE}:${REMOTE_PORT}" "${UCL_USER}@${MYRIAD}"
