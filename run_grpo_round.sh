#!/usr/bin/env bash
# run_grpo_round.sh — one flow-SDE GRPO round: rollout -> train -> (optional) eval.
#
# Chains the two entry points so a UCL round is a single command:
#   1. rl_rollout_local  : K CBF-shielded stochastic rollouts per episode, scored traces
#   2. flow_grpo_train   : GRPO update over the LoRA action head -> next-round checkpoint
#   3. (EVAL=1) rl_rollout_local --no-cbf : measure CAR-without-CBF (learned safety)
#
# Loop it externally over rounds, feeding each round's output checkpoint back in as CKPT:
#   CKPT=/path/pi05_libero   ROUND=0 ./run_grpo_round.sh
#   CKPT=out/round1_ckpt     ROUND=1 ./run_grpo_round.sh
#   ...
#
# Config via env vars (all have defaults):
set -euo pipefail

CONFIG=${CONFIG:-pi05_libero_cbf}        # TrainConfig (LoRA freeze filter) for training
ROLLOUT_CONFIG=${ROLLOUT_CONFIG:-pi05_libero_cbf}  # LoRA config; partial load handles base ckpt at round 0
CKPT=${CKPT:?set CKPT to the current-policy checkpoint dir (params/ + assets/)}
SUITE=${SUITE:-safelibero_object}
LEVEL=${LEVEL:-II}
TASK=${TASK:-0}
EPISODES=${EPISODES:-0 1 2 3}
K=${K:-8}
HORIZON=${HORIZON:-300}
REPLAN=${REPLAN:-5}
NUM_STEPS=${NUM_STEPS:-10}
NOISE_LEVEL=${NOISE_LEVEL:-0.7}
SDE_TYPE=${SDE_TYPE:-cps}
CLIP=${CLIP:-0.2}
LR=${LR:-1e-5}          # RL fine-tuning is fragile — high LR collapses the policy in one round
EPOCHS=${EPOCHS:-1}
MINIBATCH=${MINIBATCH:-8}
ROUND=${ROUND:-0}
OUT=${OUT:-results_grpo}
EVAL=${EVAL:-1}                          # 1 = run the no-CBF headline eval after training
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

ROLL_DIR="$OUT/round${ROUND}"
NEXT_CKPT="$OUT/round$((ROUND + 1))_ckpt"
EVAL_DIR="$OUT/round${ROUND}_eval_nocbf"

echo "==================================================================="
echo " GRPO round $ROUND"
echo "   policy checkpoint : $CKPT"
echo "   suite/level/task  : $SUITE / L$LEVEL / t$TASK   episodes=[$EPISODES] K=$K"
echo "   flow-SDE          : steps=$NUM_STEPS noise=$NOISE_LEVEL type=$SDE_TYPE"
echo "   rollouts -> $ROLL_DIR ;  next ckpt -> $NEXT_CKPT"
echo "==================================================================="

echo; echo "### [1/3] rollout collection (CBF-shielded) ###"
$PY -m experiments.rl_rollout_local \
    --config "$ROLLOUT_CONFIG" --checkpoint "$CKPT" \
    --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
    --episodes $EPISODES --K "$K" \
    --horizon "$HORIZON" --replan "$REPLAN" \
    --num-steps "$NUM_STEPS" --noise-level "$NOISE_LEVEL" --sde-type "$SDE_TYPE" \
    --out "$ROLL_DIR"

echo; echo "### [2/3] GRPO update (LoRA action head) ###"
$PY -m experiments.flow_grpo_train \
    --config "$CONFIG" --checkpoint "$CKPT" \
    --round "$ROLL_DIR" --out "$NEXT_CKPT" \
    --noise-level "$NOISE_LEVEL" --sde-type "$SDE_TYPE" --clip "$CLIP" \
    --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"

if [[ "$EVAL" == "1" ]]; then
    echo; echo "### [3/3] headline eval: NO CBF shield (measure learned safety) ###"
    # ODE (deterministic) eval of the UPDATED policy — noise_level 0.
    $PY -m experiments.rl_rollout_local \
        --config "$ROLLOUT_CONFIG" --checkpoint "$NEXT_CKPT" \
        --suite "$SUITE" --level "$LEVEL" --task "$TASK" \
        --episodes $EPISODES --K 1 \
        --horizon "$HORIZON" --replan "$REPLAN" \
        --num-steps "$NUM_STEPS" --noise-level 0 --sde-type "$SDE_TYPE" \
        --no-cbf --out "$EVAL_DIR"
    echo "   eval (no-CBF) round summary: $EVAL_DIR/round_summary.json"
fi

echo; echo "GRPO round $ROUND done. Next: CKPT=$NEXT_CKPT ROUND=$((ROUND + 1)) $0"
