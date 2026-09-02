#!/bin/bash
# Upload OpenVLA model weights from your Mac to cream.
# Run this on your LOCAL Mac from the project root.
#
# Usage:
#   bash cs_timeshare/upload_weights.sh

CS_USER="${CS_USER:?set CS_USER to your UCL CS username}"
GATEWAY="${CS_GATEWAY:?set CS_GATEWAY, e.g. gateway.cs.ucl.ac.uk}"
MACHINE="${CS_MACHINE:?set CS_MACHINE to the compute host}"
LOCAL_WEIGHTS="VLA-Model/openvla/openvla-7b/"
REMOTE_THESIS="${CS_BASE:?set CS_BASE to your project directory on the host}/thesis"

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
