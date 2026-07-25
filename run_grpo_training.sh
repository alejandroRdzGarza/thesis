#!/usr/bin/env bash
# run_grpo_training.sh — run N flow-SDE GRPO rounds back-to-back, chaining checkpoints.
#
# Round N takes round${N}_ckpt as input (round 0 takes $BASE_CKPT) and produces
# round$((N+1))_ckpt, via run_grpo_round.sh (rollout → GRPO update → no-CBF eval).
# RESUMABLE: a round whose output checkpoint already exists is skipped, so re-running
# this script continues where it left off (e.g. after an interruption, or if round 0 was
# already done manually).
#
# Source your env first, then run inside tmux:
#   source experiments/rl_env_runpod.sh        (or rl_env.sh on UCL)
#   BASE_CKPT=$CKPT N_ROUNDS=4 ./run_grpo_training.sh 2>&1 | tee grpo_training.log
#
# Config via env vars (defaults shown); the task knobs are forwarded to run_grpo_round.sh.
set -euo pipefail

BASE_CKPT=${BASE_CKPT:?set BASE_CKPT to the base pi05_libero checkpoint (round 0 input)}
N_ROUNDS=${N_ROUNDS:-4}
export OUT=${OUT:-results_grpo}
export SUITE=${SUITE:-safelibero_object}
export LEVEL=${LEVEL:-II}
export TASK=${TASK:-0}
export EPISODES=${EPISODES:-0 1 2 3}
export K=${K:-8}
export HORIZON=${HORIZON:-300}
export NUM_STEPS=${NUM_STEPS:-10}

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo " GRPO training: $N_ROUNDS rounds  |  $SUITE / L$LEVEL / t$TASK"
echo "   episodes=[$EPISODES]  K=$K  horizon=$HORIZON  num_steps=$NUM_STEPS"
echo "   base checkpoint: $BASE_CKPT   →   output under $OUT/"
echo "==================================================================="

for (( N=0; N<N_ROUNDS; N++ )); do
    out_ckpt="$OUT/round$((N + 1))_ckpt"
    if [[ -d "$out_ckpt/lora_params" || -d "$out_ckpt/params" ]]; then
        echo ">>> round $N already done ($out_ckpt exists) — skipping"
        continue
    fi
    if (( N == 0 )); then in_ckpt="$BASE_CKPT"; else in_ckpt="$OUT/round${N}_ckpt"; fi
    if [[ ! -d "$in_ckpt" ]]; then
        echo "!!! round $N input checkpoint missing: $in_ckpt — stopping"; exit 1
    fi

    echo; echo ">>> ROUND $N   input=$in_ckpt   →   output=$out_ckpt"; echo
    CKPT="$in_ckpt" ROUND="$N" bash "$HERE/run_grpo_round.sh" 2>&1 | tee "round${N}.log"
done

echo; echo "===================== TRAINING COMPLETE ============================"
echo " Round-over-round trend (watch mean_cbf_penalty → 0, success_rate holding):"
for (( N=0; N<N_ROUNDS; N++ )); do
    echo "  round $N  with-CBF : $(cat "$OUT/round${N}/round_summary.json" 2>/dev/null || echo '(missing)')"
    echo "  round $N  no-CBF   : $(cat "$OUT/round${N}_eval_nocbf/round_summary.json" 2>/dev/null || echo '(missing)')"
done
echo "===================================================================="
