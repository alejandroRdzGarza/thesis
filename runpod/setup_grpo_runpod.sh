#!/usr/bin/env bash
# setup_grpo_runpod.sh — ONE-SHOT, idempotent setup for the flow-SDE GRPO pipeline on a
# fresh RunPod pod (or after /workspace was wiped). Safe to re-run: every step is guarded.
#
# Why a script: the pipeline needs THREE repos, and two of them are NOT in the thesis remote
# (openpi is git-ignored; the SafeLIBERO fork lives in a separate repo). Cloning only the
# thesis leaves you with `KeyError: safelibero_object` / `No module named jax`. This wires
# all three together on /workspace (the persistent network volume) so a restart keeps it.
#
#   Bootstrap (only the thesis clone is manual):
#     bash                                  # RunPod login shell is often tcsh
#     cd /workspace
#     git clone https://github.com/alejandroRdzGarza/thesis.git
#     tmux new -s setup                     # ~20-40 min (uv sync is the long pole); survive drops
#     bash /workspace/thesis/runpod/setup_grpo_runpod.sh
#
# After it prints "Setup complete", start training with the commands it echoes.
set -euo pipefail

BASE=/workspace
THESIS=$BASE/thesis
OPENPI=$BASE/openpi
AEGIS=$BASE/vlsa-aegis
LIBERO_ROOT=$AEGIS/safelibero            # PYTHONPATH root: contains libero/libero/benchmark (safelibero_* suites)
OPENPI_BASE_COMMIT=15a9616               # upstream openpi commit our flow-SDE GRPO patch applies onto

export HOME=$BASE/.home; mkdir -p "$HOME"
export UV_CACHE_DIR=$BASE/uv-cache       # keep uv's cache on the volume, not the small container root
export GIT_LFS_SKIP_SMUDGE=1             # openpi has LFS pointers we don't need for the RL pipeline
export PATH="$HOME/.local/bin:$PATH"

echo "### [1/6] system packages (EGL headless rendering + git-lfs) ###"
apt-get update -qq
apt-get install -y -qq libegl1 libgles2 libosmesa6 libgl1 libglu1-mesa git-lfs rsync curl

echo "### [2/6] clone the three repos ###"
[ -d "$AEGIS/.git" ]  || git clone https://github.com/THU-RCSCT/vlsa-aegis.git "$AEGIS"
[ -d "$OPENPI/.git" ] || git clone https://github.com/Physical-Intelligence/openpi.git "$OPENPI"
# rl_env_runpod.sh expects the fork at $BASE/libero_repo; symlink it to the vlsa-aegis subdir.
ln -sfn "$LIBERO_ROOT" "$BASE/libero_repo"

echo "### [3/6] openpi -> base commit + flow-SDE GRPO patch (HOOK A/B/C, LoRA config) ###"
cd "$OPENPI"
if [ ! -f src/openpi/models/flow_sde.py ]; then
    git checkout -q "$OPENPI_BASE_COMMIT"
    git apply "$THESIS/openpi_patches/flow_sde_grpo.patch"
    echo "  patch applied."
else
    echo "  openpi already patched (flow_sde.py present) — skipping checkout+apply."
fi

echo "### [4/6] uv + openpi deps (jax[cuda12], flax, orbax, ...) — the long pole ###"
command -v uv >/dev/null 2>&1 || { curl -LsSf https://astral.sh/uv/install.sh | sh; }
export PATH="$HOME/.local/bin:$PATH"
cd "$OPENPI"
uv sync
uv pip install -e .

echo "### [5/6] LIBERO runtime deps into openpi's .venv (fork itself runs from PYTHONPATH) ###"
# numpy pinned <2.0: robosuite 1.4.1 + the working UCL/RunPod env need 1.26.x. cvxpy = CBF QP.
uv pip install \
    robosuite==1.4.1 bddl==1.0.1 mujoco==3.2.3 \
    easydict cloudpickle lxml h5py imageio imageio-ffmpeg PyYAML \
    "numpy==1.26.4" cvxpy

echo "### [6/6] verify the whole stack imports + SafeLIBERO suites register ###"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$THESIS:$LIBERO_ROOT"
"$OPENPI/.venv/bin/python" - <<'PY'
import jax; print("  jax devices:", jax.devices())
import openpi.models.flow_sde, openpi.policies.policy_logprob, openpi.training.flow_grpo
print("  openpi flow-SDE GRPO modules OK")
import cvxpy; print("  cvxpy", cvxpy.__version__)
from libero.libero import benchmark
d = benchmark.get_benchmark_dict()
suites = sorted(k for k in d if k.startswith("safelibero_"))
assert "safelibero_object" in d, "SafeLIBERO suites NOT registered — check the vlsa-aegis clone / PYTHONPATH"
print("  SafeLIBERO suites OK:", suites)
PY

echo
echo "=================================================================="
echo " Setup complete — everything on /workspace (survives pod restart)."
echo
echo " Next (in tmux):"
echo "   source $THESIS/experiments/rl_env_runpod.sh"
echo "   cd $THESIS"
echo "   # The base pi05_libero checkpoint (~10GB) auto-downloads on round 0."
echo "   BASE_CKPT=\$CKPT N_ROUNDS=6 OUT=results_grpo_v2 ./run_grpo_training.sh 2>&1 | tee grpo_v2.log"
echo "=================================================================="
