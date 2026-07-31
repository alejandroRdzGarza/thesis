#!/usr/bin/env bash
# run_distill_round0.sh — Exp 007 round 0: OFFLINE BC distillation of the classical MPC-CBF expert
# into π0.5. Collect optimal-safe classical demos → BC the LoRA on the clean ones → no-CBF eval.
#
# This is the offline half of BC-then-DAgger. The DAgger rounds (roll out the student VLA, label
# its states with the same classical expert) build on this checkpoint and reuse the trainer.
#
#   source experiments/rl_env_runpod.sh   (or rl_env.sh on UCL)
#   CKPT=$CKPT ./run_distill_round0.sh 2>&1 | tee distill_round0.log
set -euo pipefail

CKPT=${CKPT:?set CKPT to the base pi05_libero checkpoint dir}
SUITE=${SUITE:-safelibero_spatial}
LEVEL=${LEVEL:-II}
TASKS=${TASKS:-0 1 2 3}
COLLECT_EPISODES=${COLLECT_EPISODES:-0 1 2 3 4 5 6 7}   # more episodes → more CLEAN demos after filtering
EVAL_TASK=${EVAL_TASK:-0}
EVAL_EPISODES=${EVAL_EPISODES:-0 1 2 3 4 5 6 7}
OUT=${OUT:-results_distill}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-2}
MINIBATCH=${MINIBATCH:-8}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

DEMOS="$OUT/round0_demos"
CKPT_OUT="$OUT/round0_ckpt"
EVAL_DIR="$OUT/round0_eval_nocbf"

echo "==================================================================="
echo " DISTILL round 0 (offline BC)  |  $SUITE / L$LEVEL / tasks [$TASKS]"
echo "   collect episodes=[$COLLECT_EPISODES]  → BC → no-CBF eval (t$EVAL_TASK)"
echo "==================================================================="

echo; echo "### [1/3] collect classical expert demos (optimal-safe, offline) ###"
$PY -m experiments.collect_classical_demos --suite "$SUITE" --level "$LEVEL" \
    --tasks $TASKS --episodes $COLLECT_EPISODES --horizon "$HORIZON" --out "$DEMOS"

echo; echo "### [2/3] BC-distill the LoRA on the CLEAN demos (--success-only) ###"
$PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$CKPT" \
    --round "$DEMOS" --out "$CKPT_OUT" --success-only \
    --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"

echo; echo "### [3/3] headline eval: NO CBF shield (did the VLA internalize the expert's safety?) ###"
$PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$CKPT_OUT" \
    --suite "$SUITE" --level "$LEVEL" --task "$EVAL_TASK" --episodes $EVAL_EPISODES --K 2 \
    --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 --no-cbf --out "$EVAL_DIR"
echo "   eval (no-CBF) summary: $EVAL_DIR/round_summary.json"
echo; echo "DISTILL round 0 done → BC checkpoint: $CKPT_OUT"
