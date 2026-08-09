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
# The SIM stack must go in the SAME venv as openpi. An earlier version installed mujoco/
# robosuite/bddl into $VENV but left this line with only the small extras, so importing
# openpi and robosuite together failed. `future` is required by bddl==1.0.1 and was in
# neither list.
uv pip install mujoco==3.2.3 robosuite==1.4.1 bddl==1.0.1 numpy==1.26.4 \
    scipy opencv-python matplotlib Pillow imageio imageio-ffmpeg h5py PyYAML tqdm \
    easydict cloudpickle lxml future

echo ""
echo "=== openpi thesis patches (pi05_libero_cbf config, PolicyWithLogprob, flow_sde) ==="
# The nested openpi checkout is git-ignored and NOT backed up, so a fresh clone is stock upstream
# and has none of the thesis hooks. Without this there is no pi05_libero_cbf config and no
# _build_flow_velocity_fn, and every experiment script fails on import.
if ! grep -q "pi05_libero_cbf" "$OPENPI_DIR/src/openpi/training/config.py" 2>/dev/null; then
    (cd "$OPENPI_DIR" && git apply "$THESIS/openpi_patches/flow_sde_grpo.patch") \
        && echo "  patches applied" \
        || echo "  WARNING: patch failed — apply openpi_patches/ by hand (see its README)"
else
    echo "  already patched"
fi

echo "=== SafeLIBERO ==="
# SafeLIBERO is a FORK of LIBERO carrying the safelibero_* suites. Stock LIBERO does not have
# them, so PYTHONPATH must point here or make_libero_env raises KeyError: 'safelibero_spatial'.
if [ -d /workspace/vlsa-aegis/safelibero ]; then
    echo "  found at /workspace/vlsa-aegis/safelibero"
    echo "  export PYTHONPATH=/workspace/vlsa-aegis/safelibero   # NOT /workspace/LIBERO"
else
    echo "  NOT PRESENT. Copy it from a pod that has it (it is not on GitHub):"
    echo "    rsync -avz -e 'ssh -p PORT -i KEY' root@OLD_POD:/workspace/vlsa-aegis /tmp/"
    echo "    rsync -avz -e 'ssh -p PORT -i KEY' /tmp/vlsa-aegis root@NEW_POD:/workspace/"
fi

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
