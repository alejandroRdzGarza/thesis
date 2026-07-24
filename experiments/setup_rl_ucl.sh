#!/usr/bin/env bash
# setup_rl_ucl.sh — idempotent setup for the co-located flow-SDE GRPO pipeline on UCL.
# Safe to re-run; skips work already done. Run on any UCL GPU node.
#
#   bash $BASE/thesis/experiments/setup_rl_ucl.sh
#
# ASSUMES these already live on the shared $BASE (they persist across nodes / node changes):
#   - $BASE/openpi         openpi repo + its uv .venv (jax+CUDA working)
#   - $BASE/libero_repo    the SafeLIBERO fork (rsync'd from the Mac; see NOTE below)
#   - $BASE/openpi_cache/openpi-assets/checkpoints/pi05_libero   base π0.5 checkpoint
#   - $BASE/thesis         this repo (experiments/, openpi patch)
#
# NOTE — two things come from the MAC and can't be reproduced on UCL alone. If they're
# missing (truly fresh setup), run these FROM THE MAC first, then re-run this script:
#   1) the openpi flow-SDE GRPO patch:
#        rsync -avz openpi_patches/flow_sde_grpo.patch \
#          jesusr01@knuckles.cs.ucl.ac.uk:$BASE/openpi_patches/    # then: cd $BASE/openpi && git apply ../openpi_patches/flow_sde_grpo.patch
#      (or rsync the changed openpi/src files directly)
#   2) the SafeLIBERO fork into libero_repo (benchmark, bddl, init, obstacle objects, assets):
#        MACLIB=/Users/.../miniforge3/envs/libero/lib/python3.10/site-packages/libero/libero/
#        rsync -avz --exclude='datasets' --exclude='__pycache__' "$MACLIB" \
#          jesusr01@knuckles.cs.ucl.ac.uk:$BASE/libero_repo/libero/libero/

set -euo pipefail
export BASE=/cs/student/project_msc/2025/rai/jesusr01

echo "== 1/3  libero runtime deps into openpi/.venv (pinned; keep numpy<2) =="
cd "$BASE/openpi"
uv pip install cvxpy opencv-python
uv pip install "numpy<2" "mujoco==2.3.7" "robosuite==1.4.1" "bddl==3.6.0" "gym==0.25.2" easydict
uv pip install -e ../libero_repo

echo "== 2/3  create LIBERO config under \$BASE/.home (auto-answer N to dataset prompt) =="
mkdir -p "$BASE/.home"
HOME="$BASE/.home" MUJOCO_GL=egl PYTHONPATH="$BASE/libero_repo" \
  "$BASE/openpi/.venv/bin/python" -c "from libero.libero import benchmark; print('LIBERO config OK')" <<< "N" || true

echo "== 3/3  verify the unified env (jax GPU + libero stack + SafeLIBERO suites) =="
# shellcheck source=/dev/null
source "$BASE/thesis/experiments/rl_env.sh"
"$PY" -c "
import jax; print('jax:', jax.devices())
import numpy, robosuite, mujoco, cvxpy
print('numpy', numpy.__version__, '| robosuite', robosuite.__version__, '| mujoco', mujoco.__version__)
from libero.libero import benchmark
suites = sorted(k for k in benchmark.get_benchmark_dict() if 'safe' in k)
print('safelibero suites:', suites)
assert suites, 'SafeLIBERO not registered — rsync the fork into libero_repo (see NOTE)'
import experiments.libero_runner, experiments.load_policy
print('SETUP COMPLETE — jax on GPU, libero+SafeLIBERO ready')
"
