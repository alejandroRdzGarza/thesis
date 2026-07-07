#!/usr/bin/env bash
# One-shot setup for the π0.5 + LIBERO benchmark environment on RunPod.
# Run once after a fresh pod start (or after losing /workspace state).
#
# Usage:
#   bash /workspace/thesis/runpod/setup_pi05_env.sh

set -e

VENV=/workspace/pi_env
LIBERO_DIR=/workspace/LIBERO
THESIS=/workspace/thesis

echo "=== System packages ==="
apt-get update -qq
apt-get install -y libegl1 libgles2 libosmesa6 libgl1 libglu1-mesa -qq

echo "=== Python venv ==="
python3.11 -m venv "$VENV" 2>/dev/null || true
source "$VENV/bin/activate"

echo "=== Core Python deps ==="
pip install -q --upgrade pip

pip install -q \
    mujoco==3.2.3 \
    robosuite==1.4.1 \
    bddl==1.0.1 \
    numpy==1.26.4 \
    scipy \
    opencv-python \
    matplotlib \
    Pillow \
    imageio \
    imageio-ffmpeg \
    h5py \
    PyYAML \
    tqdm \
    easydict \
    cloudpickle \
    lxml

echo "=== openpi-client ==="
pip install -q "$THESIS/openpi/packages/openpi-client/"

echo "=== LIBERO from source ==="
if [ ! -d "$LIBERO_DIR" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
pip install -q -e "$LIBERO_DIR"

echo "=== Environment variables ==="
cat >> "$VENV/bin/activate" << 'EOF'

# π0.5 + LIBERO rendering and path setup
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export PYTHONPATH=/workspace/LIBERO:$PYTHONPATH
EOF

echo ""
echo "=== Done! Activate with: ==="
echo "  source $VENV/bin/activate"
echo ""
echo "=== Then start the π0.5 server (terminal 1): ==="
echo "  source \$HOME/.local/bin/env"
echo "  cd $THESIS/openpi && export OPENPI_DATA_HOME=/workspace/openpi-cache"
echo "  uv run scripts/serve_policy.py --env libero"
echo ""
echo "=== And run the benchmark (terminal 2): ==="
echo "  cd $THESIS"
echo "  python run_libero_benchmark.py --suite libero_spatial --task 0 --mode plain --episodes 2 --vla pi05 --horizon 200"
