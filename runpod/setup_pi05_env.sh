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
OPENPI_DIR=/workspace/thesis/openpi

echo "=== System packages ==="
apt-get update -qq
apt-get install -y libegl1 libgles2 libosmesa6 libgl1 libglu1-mesa git-lfs -qq

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

echo "=== uv (openpi package manager) ==="
if ! command -v uv &>/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi

# Point uv cache to the network volume — container root filesystem is small
export UV_CACHE_DIR=/workspace/uv-cache
grep -q "UV_CACHE_DIR" "$HOME/.bashrc" 2>/dev/null || \
    echo 'export UV_CACHE_DIR=/workspace/uv-cache' >> "$HOME/.bashrc"

echo "=== openpi repo ==="
if [ ! -d "$OPENPI_DIR" ]; then
    GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI_DIR"
fi

echo "=== openpi deps (uv sync) ==="
cd "$OPENPI_DIR"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

echo "=== LIBERO + extras into openpi .venv (no separate pi_env needed) ==="
if [ ! -d "$LIBERO_DIR" ]; then
    git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
fi
cd "$OPENPI_DIR"
uv pip install -e "$LIBERO_DIR"
uv pip install bddl easydict cloudpickle lxml h5py imageio imageio-ffmpeg PyYAML

echo ""
echo "=== Done! No separate venv needed — use openpi's .venv for everything ==="
echo ""
echo "=== Terminal 1 — start the π0.5 server (checkpoint auto-downloads ~10GB): ==="
echo "  export UV_CACHE_DIR=/workspace/uv-cache"
echo "  cd $OPENPI_DIR"
echo "  export OPENPI_DATA_HOME=/workspace/openpi-cache"
echo "  uv run scripts/serve_policy.py --env libero"
echo ""
echo "=== Terminal 2 — run the benchmark: ==="
echo "  cd $THESIS"
echo "  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl PYTHONPATH=/workspace/LIBERO"
echo "  $OPENPI_DIR/.venv/bin/python run_libero_benchmark.py --vla pi05 --suite libero_spatial --task 0 --episodes 5"
