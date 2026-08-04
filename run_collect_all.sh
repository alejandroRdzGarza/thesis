#!/usr/bin/env bash
# run_collect_all.sh — collect classical MPC-CBF expert demos across ALL suites × levels × tasks.
# Keeps every rollout (the manifest marks success + collision); flow_bc_train --success-only then
# trains on the clean+safe ones. This is the "all scenes with successful samples" training set.
#
#   PY=$PY ./run_collect_all.sh 2>&1 | tee collect_all.log
#   then: flow_bc_train --config pi05_libero_cbf --checkpoint $CKPT --success-only \
#             --round demos_all/*_* --out results_distill_all/ckpt --epochs 20
set -uo pipefail
SUITES=${SUITES:-"safelibero_spatial safelibero_object safelibero_goal"}
LEVELS=${LEVELS:-"I II"}
TASKS=${TASKS:-"0 1 2 3"}
EPISODES=${EPISODES:-$(seq 0 29)}     # inits per task to attempt (more → more clean demos)
HORIZON=${HORIZON:-300}
OUT=${OUT:-demos_all}
PY=${PY:-python}
export PYTHONPATH=${PYTHONPATH:-.}
mkdir -p "$OUT"
for suite in $SUITES; do
  for level in $LEVELS; do
    d="$OUT/${suite}_L${level}"
    if [ -f "$d/manifest.csv" ]; then echo "[skip] $d exists"; continue; fi
    echo "=== collect $suite L$level tasks[$TASKS] ==="
    $PY -m experiments.collect_classical_demos --suite "$suite" --level "$level" \
        --tasks $TASKS --episodes $EPISODES --horizon "$HORIZON" --out "$d"
    grep -h "CLEAN" "$d"/*.log 2>/dev/null | tail -1
    [ -f "$d/manifest.csv" ] && echo "  → $(tail -n +2 "$d/manifest.csv" | awk -F, '$2>0 && $3==0' | wc -l | tr -d ' ') CLEAN demos in $d"
  done
done
echo; echo "collected → $OUT/  (per suite/level manifests). Clean-demo counts above."
echo "distil:  flow_bc_train --round $OUT/*_L* --success-only --checkpoint \$CKPT --out results_distill_all/ckpt --epochs 20"
