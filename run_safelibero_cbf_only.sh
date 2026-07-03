#!/usr/bin/env bash
# run_safelibero_cbf_only.sh — full SafeLIBERO benchmark, CBF mode only.
#
# Runs all 4 SafeLIBERO suites (spatial, object, goal, long) x 2 safety
# levels (I, II) x 4 tasks/suite, ellipsoid-CBF mode only (no plain baseline).
#
# Requires: conda activate libero, VLA server reachable at localhost:8000.
#
# Usage:
#   bash run_safelibero_cbf_only.sh [EPISODES] [--save-video] [--show-every N]
#
# Examples:
#   bash run_safelibero_cbf_only.sh 10
#   bash run_safelibero_cbf_only.sh 50 --save-video

set -euo pipefail

EPISODES=10
SAVE_VIDEO=""
SHOW_EVERY=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-video)   SAVE_VIDEO="--save-video"; shift ;;
        --show-every)   SHOW_EVERY="--show-every $2"; shift 2 ;;
        --show-every=*) SHOW_EVERY="--show-every ${1#*=}"; shift ;;
        [0-9]*)         EPISODES="$1"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="results_safelibero_cbf"
mkdir -p "$RESULTS_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

SUITES=(safelibero_spatial safelibero_object safelibero_goal safelibero_long)
LEVELS=(I II)

echo "============================================================"
echo "  SafeLIBERO Full Benchmark — CBF only"
echo "  episodes/task : ${EPISODES}"
echo "  suites        : ${SUITES[*]}"
echo "  levels        : ${LEVELS[*]}"
echo "  save video    : ${SAVE_VIDEO:-off}"
echo "  results dir   : ${RESULTS_DIR}/"
echo "  started       : $(timestamp)"
echo "============================================================"

i=0
total=$(( ${#SUITES[@]} * ${#LEVELS[@]} ))
for suite in "${SUITES[@]}"; do
    for level in "${LEVELS[@]}"; do
        i=$((i+1))
        echo ""
        echo ">>> [$i/$total] ${suite} L${level} — CBF  started: $(timestamp)"
        python run_libero_benchmark.py \
            --suite "$suite" \
            --safety-level "$level" \
            --all \
            --episodes "$EPISODES" \
            --mode cbf \
            --horizon 500 \
            --replan-steps 8 \
            --results-dir "$RESULTS_DIR" \
            $SAVE_VIDEO $SHOW_EVERY \
            2>&1 | tee "$RESULTS_DIR/${suite}_L${level}_cbf.log"
        echo ">>> [$i/$total] done: $(timestamp)"
    done
done

echo ""
echo "============================================================"
echo "  Benchmark complete: $(timestamp)"
echo "  Results: ${RESULTS_DIR}/"
if [[ -n "$SAVE_VIDEO" ]]; then
echo "  Videos:  ${RESULTS_DIR}/videos/"
fi
echo "============================================================"
