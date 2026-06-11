#!/bin/bash
# Run this ONCE on a Myriad login node to create the Python environment.
# Usage: bash setup_env.sh

set -e

VENV_DIR="$HOME/venvs/openvla"

echo "=== Loading Myriad modules ==="
module purge
module load gcc-libs/10.2.0
module load python3/3.9-gnu-10.2.0
module load cuda/11.2.0/gnu-10.2.0
module load cudnn/8.1.0.77/cuda-11.2

echo "=== Creating virtual environment at $VENV_DIR ==="
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

pip install --upgrade pip

echo "=== Installing VLA server dependencies ==="
pip install torch==2.0.1+cu118 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.40.2
pip install fastapi==0.111.0
pip install uvicorn==0.30.1
pip install Pillow==10.3.0
pip install numpy==1.26.4
pip install pydantic==2.7.1
pip install accelerate==0.30.1
pip install timm==0.9.16

echo ""
echo "=== Setup complete ==="
echo "Python env: $VENV_DIR"
echo ""
echo "Next steps:"
echo "  1. Upload model weights to Myriad (from your Mac):"
echo "     rsync -avz --progress VLA-Model/openvla/openvla-7b/ <ucl-username>@myriad.rc.ucl.ac.uk:~/openvla-7b/"
echo "  2. Submit the server job:"
echo "     qsub myriad/submit_vla_server.sh"
