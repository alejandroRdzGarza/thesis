#!/usr/bin/env bash
# run_dagger_round.sh — one shield-as-expert DAgger round (thesis Exp 005 pivot).
#
#   1. COLLECT : roll out the current policy FULLY SHIELDED (shield_prob=1) — the CBF is the
#                expert; each query records the executed shield-corrected actions.
#   2. BC      : behavior-clone the LoRA to reproduce those safe actions (flow_bc_train).
#   3. EVAL    : roll out the updated policy with NO shield → measure learned safety.
#
# Loop externally, feeding each round's output checkpoint back in as CKPT (see run_dagger_training.sh).
set -euo pipefail

CONFIG=${CONFIG:-pi05_libero_cbf}
ROLLOUT_CONFIG=${ROLLOUT_CONFIG:-pi05_libero_cbf}
CKPT=${CKPT:?set CKPT to the current-policy checkpoint dir}
SUITE=${SUITE:-safelibero_object}
LEVEL=${LEVEL:-II}
TASK=${TASK:-0}
EPISODES=${EPISODES:-0 1 2 3}
K=${K:-8}
HORIZON=${HORIZON:-300}
REPLAN=${REPLAN:-5}
NUM_STEPS=${NUM_STEPS:-10}
# Low sampling noise for collection: DAgger wants the shield's corrections along the CURRENT
# policy's own trajectory distribution (near on-policy), not a wildly exploratory one.
NOISE_LEVEL=${NOISE_LEVEL:-0.4}
SDE_TYPE=${SDE_TYPE:-cps}
LR=${LR:-1e-4}          # imitation is stable → higher LR than the RL runs is fine
EPOCHS=${EPOCHS:-2}
MINIBATCH=${MINIBATCH:-8}
# Imitate ONLY successful + collision-free shielded rollouts (automated expert filter). On by
# default: guards against BC drifting toward the shield's over-cautious failures (Exp 005 erosion).
SUCCESS_ONLY=${SUCCESS_ONLY:-1}
ROUND=${ROUND:-0}
OUT=${OUT:-results_dagger}
EVAL=${EVAL:-1}
EVAL_EPISODES=${EVAL_EPISODES:-0 1 2 3 4 5 6 7}
EVAL_K=${EVAL_K:-2}
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

ROLL_DIR="$OUT/round${ROUND}"
NEXT_CKPT="$OUT/round$((ROUND + 1))_ckpt"
EVAL_DIR="$OUT/round${ROUND}_eval_nocbf"

echo "==================================================================="
echo " DAgger round $ROUND  (shield-as-expert BC)"
echo "   policy checkpoint : $CKPT"
echo "   suite/level/task  : $SUITE / L$LEVEL / t$TASK   episodes=[$EPISODES] K=$K"
echo "   collect FULLY SHIELDED → $ROLL_DIR ;  BC → $NEXT_CKPT"
echo "==================================================================="

echo; echo "### [1/3] collect shielded expert data (shield_prob=1) ###"
$PY -m experiments.rl_rollout_local \
    --config "$ROLLOUT_CONFIG" --checkpoint "$CKPT" \
    --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
    --episodes $EPISODES --K "$K" --shield-prob 1.0 \
    --horizon "$HORIZON" --replan "$REPLAN" \
    --num-steps "$NUM_STEPS" --noise-level "$NOISE_LEVEL" --sde-type "$SDE_TYPE" \
    --out "$ROLL_DIR"

echo; echo "### [2/3] BC update (imitate shield-corrected actions) ###"
_success_flag=""; [[ "$SUCCESS_ONLY" == "1" ]] && _success_flag="--success-only"
$PY -m experiments.flow_bc_train \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --round "$ROLL_DIR" --out "$NEXT_CKPT" \
    --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH" $_success_flag

if [[ "$EVAL" == "1" ]]; then
    echo; echo "### [3/3] headline eval: NO CBF shield (measure learned safety) ###"
    $PY -m experiments.rl_rollout_local \
        --config "$ROLLOUT_CONFIG" --checkpoint "$NEXT_CKPT" \
        --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
        --episodes $EVAL_EPISODES --K "$EVAL_K" \
        --horizon "$HORIZON" --replan "$REPLAN" \
        --num-steps "$NUM_STEPS" --noise-level 0 --sde-type "$SDE_TYPE" \
        --no-cbf --out "$EVAL_DIR"
    echo "   eval (no-CBF) round summary: $EVAL_DIR/round_summary.json"
fi

echo; echo "DAgger round $ROUND done. Next: CKPT=$NEXT_CKPT ROUND=$((ROUND + 1)) $0"
