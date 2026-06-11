#!/bin/bash
# Upload OpenVLA model weights from your Mac to cream.
# Run this on your LOCAL Mac from the project root.
#
# Usage:
#   bash cs_timeshare/upload_weights.sh

CS_USER="jesusr01"
GATEWAY="knuckles.cs.ucl.ac.uk"
MACHINE="cream.cs.ucl.ac.uk"
LOCAL_WEIGHTS="VLA-Model/openvla/openvla-7b/"
REMOTE_THESIS="/cs/student/project_msc/2025/rai/jesusr01/thesis"

echo "=== Creating vla_model directory on cream (if needed) ==="
ssh -J "${CS_USER}@${GATEWAY}" "${CS_USER}@${MACHINE}" "mkdir -p ${REMOTE_THESIS}/vla_model"

echo ""
echo "=== Uploading model weights -> ${REMOTE_THESIS}/vla_model/ on cream ==="
rsync -avz --progress \
    -e "ssh -J ${CS_USER}@${GATEWAY}" \
    "${LOCAL_WEIGHTS}" \
    "${CS_USER}@${MACHINE}:${REMOTE_THESIS}/vla_model/"

echo ""
echo "Done. Model weights are at ${REMOTE_THESIS}/vla_model/ on cream."
