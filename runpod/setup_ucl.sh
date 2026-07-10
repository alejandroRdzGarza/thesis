#!/bin/bash
# setup_ucl.sh — Install OpenVLA-OFT on UCL CS Lab machines (gadwall-l etc.)
#
# Run ONCE on the lab machine from the NFS home dir.
# Use tmux so it survives disconnects:
#   tmux new -s setup
#   bash ~/thesis/runpod/setup_ucl.sh
#
# Prerequisites:
#   - ~/thesis/ synced from Mac (rsync or git)
#   - ~/thesis/checkpoint_step006500/ present (fine-tuned weights)

set -e

HOME_DIR="/cs/student/project_msc/2025/rai/jesusr01"
VENV="$HOME_DIR/venv_vla"
OFT_REPO="$HOME_DIR/openvla_oft_repo"
PYTHON310="/opt/Python/Python-3.10.14/bin/python3.10"
THESIS="$HOME_DIR/thesis"
CHECKPOINT="$THESIS/checkpoint_step006500"

# Redirect pip cache off the tiny home quota onto the 100GB project space
export PIP_CACHE_DIR="$HOME_DIR/.cache/pip"
export XDG_CACHE_HOME="$HOME_DIR/.cache"
mkdir -p "$PIP_CACHE_DIR"

echo "========================================================"
echo "  UCL VLA Setup"
echo "  home     : $HOME_DIR"
echo "  venv     : $VENV"
echo "  oft repo : $OFT_REPO"
echo "  model    : $CHECKPOINT"
echo "========================================================"

# ── 1. Create venv ────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    echo "=== Creating Python 3.10 venv ==="
    "$PYTHON310" -m venv "$VENV"
fi
source "$VENV/bin/activate"
pip install --upgrade pip --quiet

# ── 2. PyTorch (CUDA 12.1 — compatible with driver 13.1) ─────────────────────
echo "=== Installing PyTorch ==="
pip install torch==2.3.0 torchvision==0.18.0 \
    --index-url https://download.pytorch.org/whl/cu121 --quiet

# ── 3. Core inference deps ────────────────────────────────────────────────────
echo "=== Installing inference dependencies ==="
pip install --quiet \
    fastapi==0.136.3 \
    uvicorn==0.49.0 \
    pillow \
    numpy \
    accelerate==0.27.2 \
    transformers==4.40.1 \
    timm==0.9.16 \
    safetensors \
    bitsandbytes==0.42.0 \
    pydantic \
    opencv-python \
    json-numpy \
    draccus \
    huggingface_hub

# ── 4. LIBERO + MuJoCo (for benchmark) ───────────────────────────────────────
echo "=== Installing MuJoCo and LIBERO ==="
pip install --quiet mujoco==3.2.0 PyOpenGL glfw

# Clone and install LIBERO if not already installed
if ! python -c "import libero" 2>/dev/null; then
    echo "  Installing LIBERO..."
    LIBERO_DIR="$HOME_DIR/libero_repo"
    if [[ ! -d "$LIBERO_DIR" ]]; then
        git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$LIBERO_DIR"
    fi
    pip install -e "$LIBERO_DIR" --quiet
fi

# ── 5. Clone and install openvla-oft ─────────────────────────────────────────
if [[ -d "$OFT_REPO" ]]; then
    echo "=== openvla-oft already at $OFT_REPO — pulling ==="
    git -C "$OFT_REPO" pull --ff-only
else
    echo "=== Cloning moojink/openvla-oft ==="
    git clone https://github.com/moojink/openvla-oft.git "$OFT_REPO"
fi

echo "=== Installing openvla-oft package ==="
pip install -e "$OFT_REPO" --quiet

# ── 6. Apply patches (same as RunPod setup_oft.sh) ───────────────────────────
echo "=== Patching prismatic __init__.py files ==="
for _init in \
    "$OFT_REPO/prismatic/__init__.py" \
    "$OFT_REPO/prismatic/training/__init__.py" \
    "$OFT_REPO/prismatic/models/__init__.py" \
    "$OFT_REPO/prismatic/models/vlas/__init__.py" \
    "$OFT_REPO/prismatic/vla/__init__.py" \
    "$OFT_REPO/prismatic/vla/datasets/__init__.py" \
    "$OFT_REPO/prismatic/vla/datasets/rlds/__init__.py"; do
    echo "# Emptied for inference-only use." > "$_init"
    echo "  patched: $_init"
done

pip install -q "protobuf>=4.21.0"

echo "=== Patching data_utils.py ==="
python - <<'PYEOF'
import os; home = os.environ.get("HOME", "/cs/student/msc/rai/2025/jesusr01")
path = home + "/openvla_oft_repo/prismatic/vla/datasets/rlds/utils/data_utils.py"
with open(path) as f:
    content = f.read()
if "_DlimpStub" in content:
    print(f"  Already patched: {path}")
else:
    lines = content.split("\n"); out = []
    for line in lines:
        stripped = line.lstrip()
        if stripped == "import dlimp as dl":
            ind = line[: len(line) - len(stripped)]
            out += [f"{ind}try:", f"{ind}    import dlimp as dl",
                    f"{ind}except Exception:",
                    f"{ind}    class _DlimpStub:",
                    f"{ind}        class DLataset: pass",
                    f"{ind}    dl = _DlimpStub()"]
        else:
            out.append(line)
    with open(path, "w") as f: f.write("\n".join(out))
    print(f"  Patched: {path}")
PYEOF

echo "=== Patching update_auto_map ==="
python - <<'PYEOF'
import os; home = os.environ.get("HOME", "/cs/student/msc/rai/2025/jesusr01")
path = home + "/openvla_oft_repo/experiments/robot/openvla_utils.py"
with open(path) as f:
    content = f.read()
if "# PATCHED: skip backup" in content:
    print(f"  Already patched: {path}")
elif "shutil.copy2(config_path, backup_path)" in content:
    patched = content.replace("shutil.copy2(config_path, backup_path)",
                              "pass  # PATCHED: skip backup")
    with open(path, "w") as f: f.write(patched)
    print(f"  Patched: {path}")
else:
    print(f"  WARNING: target line not found in {path}")
PYEOF

# ── 7. Verify imports ─────────────────────────────────────────────────────────
echo "=== Verifying OFT imports ==="
OPENVLA_OFT_REPO="$OFT_REPO" python - <<'PYEOF'
import sys, os
sys.path.insert(0, os.environ["OPENVLA_OFT_REPO"])
from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla, get_vla_action
print("  OK: openvla_utils imports passed")
PYEOF

# ── 8. Done ───────────────────────────────────────────────────────────────────
echo ""
echo "========================================================"
echo "  Setup complete.  Start the server with:"
echo ""
echo "  source $VENV/bin/activate"
echo "  export OPENVLA_OFT_REPO=$OFT_REPO"
echo "  export OPENVLA_MODEL_PATH=$CHECKPOINT"
echo "  cd $THESIS"
echo "  python VLA-Model/openvla/openvla_server.py"
echo "========================================================"
