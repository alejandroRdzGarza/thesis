#!/bin/bash
# Upload all project files (excluding large model weights) to cream.
# Run this on your LOCAL Mac from the project root.
#
# Usage:
#   bash cs_timeshare/upload_project.sh

CS_USER="${CS_USER:?set CS_USER to your UCL CS username}"
GATEWAY="${CS_GATEWAY:?set CS_GATEWAY, e.g. gateway.cs.ucl.ac.uk}"
MACHINE="${CS_MACHINE:?set CS_MACHINE to the compute host}"
REMOTE_THESIS="${CS_BASE:?set CS_BASE to your project directory on the host}/thesis"

echo "=== Creating thesis directory on cream (if needed) ==="
ssh -J "${CS_USER}@${GATEWAY}" "${CS_USER}@${MACHINE}" "mkdir -p ${REMOTE_THESIS}"

echo ""
echo "=== Uploading project files -> ${REMOTE_THESIS}/ on cream ==="
rsync -avz --progress \
    -e "ssh -J ${CS_USER}@${GATEWAY}" \
    --exclude=".git" \
    --exclude=".history" \
    --exclude="VLA-Model/openvla/openvla-7b/" \
    --exclude="*.pyc" \
    --exclude="__pycache__" \
    --exclude="*.DS_Store" \
    . \
    "${CS_USER}@${MACHINE}:${REMOTE_THESIS}/"

echo ""
echo "Done. Project is at ${REMOTE_THESIS}/ on cream."
echo ""
echo "Next: upload model weights:"
echo "  bash cs_timeshare/upload_weights.sh"
