#!/bin/bash
# Forward the compute host's VLA server port to localhost:8000.
# Run this on your LOCAL Mac.
#
# Usage:
#   bash cs_timeshare/tunnel.sh

CS_USER="${CS_USER:?set CS_USER to your UCL CS username}"
GATEWAY="${CS_GATEWAY:?set CS_GATEWAY, e.g. gateway.cs.ucl.ac.uk}"
MACHINE="${CS_MACHINE:?set CS_MACHINE to the compute host}"
LOCAL_PORT=8000
REMOTE_PORT=8000

echo "Tunnelling localhost:${LOCAL_PORT} -> ${MACHINE}:${REMOTE_PORT}"
echo "  via ${CS_USER}@${GATEWAY}"
echo "Press Ctrl+C to close."
echo ""
echo "Once connected, reach the server at: http://127.0.0.1:${LOCAL_PORT}/act"
echo ""

ssh -N \
    -J "${CS_USER}@${GATEWAY}" \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    "${CS_USER}@${MACHINE}"
