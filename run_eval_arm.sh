#!/usr/bin/env bash
# run_eval_arm.sh — evaluate ONE checkpoint over the SafeLIBERO grid, exactly as the six-arm
# comparison did.
#
# WHY THIS EXISTS. The other eval scripts (run_eval.sh, run_dagger_eval.sh, run_baseline_sweep.sh,
# run_safelibero_all.sh) all drive the pi0.5 WEBSOCKET SERVER via run_libero_benchmark.py. The
# six-arm table (base/r1/r2 x shield on/off) was produced by rl_rollout_local loading a checkpoint
# IN-PROCESS — a different inference path. Mixing the two would put a path difference inside the
# comparison, so any new arm that is to sit in that table has to go through this.
#
# The body is eval_arm() from run_shielded_distill.sh, extracted verbatim so there is one
# definition rather than a copy that can drift.
#
#   ./run_eval_arm.sh planner_A_nocbf results_distill/planner_A_ckpt --no-cbf
#   ./run_eval_arm.sh planner_A_cbf   results_distill/planner_A_ckpt ""
#
# Resumable: a scene is skipped only when its manifest.csv exists. The original tested for the
# DIRECTORY, which treats a scene that crashed part-way as complete and silently folds partial
# data into the results.
set -euo pipefail

NAME=${1:?usage: run_eval_arm.sh <arm-name> <checkpoint-dir> [cbf-flag, "" or --no-cbf]}
CKPT=${2:?}
FLAG=${3:---no-cbf}

PY=${PY:-/workspace/openpi/.venv/bin/python}
OUT=${OUT:-results_distill}
# Defaults copied from run_shielded_distill.sh — change them and this arm stops being comparable.
SUITES=${SUITES:-"safelibero_spatial safelibero_object safelibero_goal"}
LEVELS=${LEVELS:-"I II"}
TASKS=${TASKS:-"0 1 2 3"}
EVAL_INITS=${EVAL_INITS:-$(seq 35 39)}     # held out: training used 0-34
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
CONFIG=${CONFIG:-pi05_libero_cbf}

n_total=$(( $(wc -w <<<"$SUITES") * $(wc -w <<<"$LEVELS") * $(wc -w <<<"$TASKS") ))
echo "=== eval $NAME  ckpt=$CKPT  flag='${FLAG}' ==="
echo "    $n_total scenes x $(wc -w <<<"$EVAL_INITS") held-out inits ($EVAL_INITS)"
echo "    horizon=$HORIZON num_steps=$NUM_STEPS noise_level=0"
echo

i=0
t0=$(date +%s)
for s in $SUITES; do for l in $LEVELS; do for t in $TASKS; do
    i=$((i+1))
    od="$OUT/eval_${NAME}/${s}_L${l}_t${t}"
    if [ -f "$od/manifest.csv" ]; then
        echo "  [$i/$n_total] $s L$l t$t — already done, skipping"
        continue
    fi
    el=$(( $(date +%s) - t0 ))
    echo "  [$i/$n_total] $s L$l t$t   (${el}s elapsed)"
    $PY -m experiments.rl_rollout_local --config "$CONFIG" --checkpoint "$CKPT" \
        --suite "$s" --level "$l" --task "$t" --episodes $EVAL_INITS --K 1 \
        --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 $FLAG --out "$od" \
        || echo "     [warn] failed — continuing (re-run to retry this scene)"
done; done; done

echo
echo "  ✔ $NAME done in $(( ($(date +%s) - t0) / 60 )) min -> $OUT/eval_${NAME}/"
echo "  compare with:"
echo "    \$PY -m experiments.sweep_eval_stats \\"
echo "      \"base+noCBF=results_shielded/eval_base_nocbf\" \\"
echo "      \"r1+noCBF=results_shielded/eval_r1_nocbf\" \\"
echo "      \"${NAME}=$OUT/eval_${NAME}\""
