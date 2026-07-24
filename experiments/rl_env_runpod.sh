# rl_env_runpod.sh — source this in every shell on the RunPod pod before running the pipeline.
#   source /workspace/thesis/experiments/rl_env_runpod.sh
# RunPod is root + everything on /workspace (persistent network volume). No home quota issue,
# but HOME is pointed into /workspace so LIBERO's ~/.libero + jax cache persist across restarts.

export BASE=/workspace
mkdir -p "$BASE/.home"

export MUJOCO_GL=egl
export HOME="$BASE/.home"
export WANDB_MODE=disabled
export OPENPI_DATA_HOME="$BASE/openpi_cache"          # where openpi downloads the checkpoint
export JAX_COMPILATION_CACHE_DIR="$BASE/.jax_cache"
export HF_HOME="$BASE/.hf_cache"

# libero_repo holds the SafeLIBERO fork (imported via PYTHONPATH — no editable install needed).
export PYTHONPATH="$BASE/thesis:$BASE/libero_repo"
export PY="$BASE/openpi/.venv/bin/python"
export CKPT="$BASE/openpi_cache/openpi-assets/checkpoints/pi05_libero"

echo "rl_env(runpod): PY=$PY  BASE=$BASE  MUJOCO_GL=$MUJOCO_GL"
