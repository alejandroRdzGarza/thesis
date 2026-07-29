#!/usr/bin/env bash
# run_dagger_training.sh — run N shield-as-expert DAgger rounds back-to-back (thesis Exp 005).
#
# Round N takes round${N}_ckpt as input (round 0 takes $BASE_CKPT) and produces round$((N+1))_ckpt
# via run_dagger_round.sh (collect fully-shielded → BC → no-CBF eval). RESUMABLE: a round whose
# output checkpoint exists is skipped.
#
#   source experiments/rl_env_runpod.sh        (or rl_env.sh on UCL)
#   BASE_CKPT=$CKPT N_ROUNDS=6 OUT=results_dagger ./run_dagger_training.sh 2>&1 | tee dagger.log
set -euo pipefail

BASE_CKPT=${BASE_CKPT:?set BASE_CKPT to the base pi05_libero checkpoint (round 0 input)}
N_ROUNDS=${N_ROUNDS:-6}
export OUT=${OUT:-results_dagger}
export SUITE=${SUITE:-safelibero_object}
export LEVEL=${LEVEL:-II}
export TASK=${TASK:-0}
export EPISODES=${EPISODES:-0 1 2 3}
export K=${K:-8}
export HORIZON=${HORIZON:-300}
export NUM_STEPS=${NUM_STEPS:-10}

HERE="$(cd "$(dirname "$0")" && pwd)"

echo "==================================================================="
echo " DAgger training (shield-as-expert): $N_ROUNDS rounds  |  $SUITE / L$LEVEL / t$TASK"
echo "   episodes=[$EPISODES]  K=$K  horizon=$HORIZON   →   output under $OUT/"
echo "==================================================================="

for (( N=0; N<N_ROUNDS; N++ )); do
    out_ckpt="$OUT/round$((N + 1))_ckpt"
    if [[ -d "$out_ckpt/lora_params" || -d "$out_ckpt/params" ]]; then
        echo ">>> round $N already done ($out_ckpt exists) — skipping"; continue
    fi
    if (( N == 0 )); then in_ckpt="$BASE_CKPT"; else in_ckpt="$OUT/round${N}_ckpt"; fi
    if [[ ! -d "$in_ckpt" ]]; then
        echo "!!! round $N input checkpoint missing: $in_ckpt — stopping"; exit 1
    fi
    echo; echo ">>> DAGGER ROUND $N   input=$in_ckpt   →   output=$out_ckpt"; echo
    CKPT="$in_ckpt" ROUND="$N" bash "$HERE/run_dagger_round.sh" 2>&1 | tee "dagger_round${N}.log"
done

echo; echo "===================== DAGGER COMPLETE =============================="
echo " Round-over-round no-CBF trend (want collision ↓, success holding):"
for (( N=0; N<N_ROUNDS; N++ )); do
    echo "  round $N  no-CBF : $(cat "$OUT/round${N}_eval_nocbf/round_summary.json" 2>/dev/null || echo '(missing)')"
done
echo "===================================================================="
