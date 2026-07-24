# rl_env.sh — source this in EVERY shell before running the flow-SDE GRPO pipeline on UCL.
#   source $BASE/thesis/experiments/rl_env.sh      (or wherever the thesis repo lives)
#
# Everything routes to project_msc ($BASE); NOTHING touches the (quota-full) home dir.
# HOME is pointed into $BASE so LIBERO's ~/.libero and jax's cache land on project_msc.

export BASE=/cs/student/project_msc/2025/rai/jesusr01
mkdir -p "$BASE/.home"

export MUJOCO_GL=egl                     # headless rendering on compute nodes
export PYTHONUNBUFFERED=1                 # flush prints immediately (live progress under | tee)
export HOME="$BASE/.home"                # keep LIBERO/jax caches off the full home mount
export WANDB_MODE=disabled
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export UV_CACHE_DIR="$BASE/.uv-cache"
export HF_HOME="$BASE/.hf_cache_new"

# libero_repo (SafeLIBERO fork) is on PYTHONPATH because its editable install doesn't
# expose the package; thesis is on PYTHONPATH for `experiments.*`.
export PYTHONPATH="$BASE/thesis:$BASE/libero_repo"

# The unified interpreter: openpi's uv venv (has jax+CUDA AND the libero stack installed).
export PY="$BASE/openpi/.venv/bin/python"

# Base π0.5 checkpoint (params/ + assets/).
export CKPT="$BASE/openpi_cache/openpi-assets/checkpoints/pi05_libero"

# GPU memory allocator: JAX defaults (preallocate on, 75%) are the balance — a high
# MEM_FRACTION starves CUDA module loading ("Failed to get module function"), while
# PREALLOCATE=false fragments. Leave both UNSET unless tuning. If you hit an OOM, first
# check `nvidia-smi` for a leftover process holding VRAM (kill it) before touching these.
unset XLA_PYTHON_CLIENT_MEM_FRACTION XLA_PYTHON_CLIENT_PREALLOCATE

echo "rl_env: PY=$PY  BASE=$BASE  (MUJOCO_GL=$MUJOCO_GL, HOME=$HOME)"
