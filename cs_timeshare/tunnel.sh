#!/bin/bash
# Forward cream's VLA server port to localhost:8000.
# Run this on your LOCAL Mac.
#
# Usage:
#   bash cs_timeshare/tunnel.sh

CS_USER="jesusr01"
GATEWAY="knuckles.cs.ucl.ac.uk"
MACHINE="cream.cs.ucl.ac.uk"
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
