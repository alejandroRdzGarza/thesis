#!/usr/bin/env bash
# run_eval.sh — SafeLIBERO evaluation for comparing VLA model versions
#
# Runs all 4 SafeLIBERO suites × 2 levels × 2 modes (plain + cbf) = 16 combos.
# Results are saved under --out tagged by --model-tag so runs can be compared.
#
# PREREQUISITES:
#   - π0.5 server running on UCL GPU machine (base or fine-tuned checkpoint)
#   - Tunnel open: bash runpod/tunnel_ucl.sh
#
# USAGE:
#   # Evaluate base π0.5
#   bash run_eval.sh --model-tag base_pi05 --episodes 50
#
#   # Evaluate fine-tuned (DAgger R0) π0.5
#   bash run_eval.sh --model-tag ft_dagger_r0 --episodes 50
#
#   # Quick sanity check (5 episodes)
#   bash run_eval.sh --model-tag base_pi05 --episodes 5
#
#   # Single suite only
#   bash run_eval.sh --model-tag ft_dagger_r0 --suite safelibero_spatial --episodes 50
#
# Resume-safe: existing CSVs are NOT overwritten (rename --out to rerun).

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EPISODES=50
MODEL_TAG="pi05"
OUT_BASE="results_eval"
PI05_HOST="127.0.0.1"
PI05_PORT=8000
SAVE_VIDEO=""
SHOW_EVERY=""
SUITE_ARG=""
LEVELS=(I II)
MODES=(plain cbf)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model-tag)    MODEL_TAG="$2";   shift 2 ;;
        --episodes)     EPISODES="$2";    shift 2 ;;
        --out)          OUT_BASE="$2";    shift 2 ;;
        --pi05-host)    PI05_HOST="$2";   shift 2 ;;
        --pi05-port)    PI05_PORT="$2";   shift 2 ;;
        --suite)        SUITE_ARG="$2";   shift 2 ;;
        --save-video)   SAVE_VIDEO="--save-video"; shift ;;
        --show-every)   SHOW_EVERY="--show-every $2"; shift 2 ;;
        --cbf-only)     MODES=(cbf);      shift ;;
        --plain-only)   MODES=(plain);    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$SUITE_ARG" ]]; then
    SUITES=("$SUITE_ARG")
else
    SUITES=(safelibero_spatial safelibero_object safelibero_goal safelibero_long)
fi

OUT="${OUT_BASE}/${MODEL_TAG}"
LOG_DIR="${OUT}/logs"
mkdir -p "$OUT" "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

total_combos=$(( ${#SUITES[@]} * ${#LEVELS[@]} * ${#MODES[@]} ))
echo "============================================================"
echo "  SafeLIBERO Evaluation"
echo "  model tag      : ${MODEL_TAG}"
echo "  suites         : ${SUITES[*]}"
echo "  levels         : ${LEVELS[*]}"
echo "  modes          : ${MODES[*]}"
echo "  episodes/combo : ${EPISODES}"
echo "  total combos   : ${total_combos}"
echo "  output         : ${OUT}/"
echo "  π0.5 server    : ws://${PI05_HOST}:${PI05_PORT}"
echo "  started        : $(timestamp)"
echo "============================================================"

# Server reachability check
echo -n "  Checking π0.5 server ... "
if ! python -c "
import socket, sys
s = socket.socket(); s.settimeout(5)
try:
    s.connect(('${PI05_HOST}', ${PI05_PORT})); s.close(); print('OK')
except Exception as e:
    print(f'UNREACHABLE: {e}'); sys.exit(1)
"; then
    echo ""
    echo "  ERROR: Cannot reach π0.5 server at ${PI05_HOST}:${PI05_PORT}"
    echo "  Make sure the tunnel is running: bash runpod/tunnel_ucl.sh"
    exit 1
fi

combo=0
for suite in "${SUITES[@]}"; do
    for level in "${LEVELS[@]}"; do
        for mode in "${MODES[@]}"; do
            combo=$((combo + 1))
            tag="${suite}_L${level}_${mode}"
            agg_csv="${OUT}/${MODEL_TAG}_${tag}_agg.csv"

            # Skip if already done
            if ls "${OUT}"/*"${tag}"*_agg.csv 2>/dev/null | grep -q .; then
                echo ""
                echo "  [${combo}/${total_combos}] ${tag}: already done, skipping"
                continue
            fi

            log="${LOG_DIR}/${tag}_$(date +%Y%m%d_%H%M%S).log"
            echo ""
            echo "============================================================"
            echo "  COMBO [${combo}/${total_combos}]: ${tag}"
            echo "  started: $(timestamp)"
            echo "============================================================"

            python run_libero_benchmark.py \
                --suite        "$suite"     \
                --safety-level "$level"     \
                --all                       \
                --episodes     "$EPISODES"  \
                --mode         "$mode"      \
                --horizon      300          \
                --replan-steps 8            \
                --vla          pi05         \
                --pi05-host    "$PI05_HOST" \
                --pi05-port    "$PI05_PORT" \
                --results-dir  "$OUT"       \
                $SAVE_VIDEO $SHOW_EVERY     \
                2>&1 | tee "$log" || {
                echo "  [ERROR] combo ${tag} failed — see ${log}"
                echo "  Continuing..."
            }

            echo "  COMBO [${combo}/${total_combos}]: ${tag}  done: $(timestamp)"
        done
    done
done

echo ""
echo "============================================================"
echo "  EVALUATION COMPLETE: $(timestamp)"
echo "  Model: ${MODEL_TAG}"
echo "  Results: ${OUT}/"
echo ""
echo "  Aggregate CSVs:"
find "$OUT" -name "*_agg.csv" | sort | while read f; do
    echo "    $f"
done
echo ""
echo "  To compare models:"
echo "    python experiments/compare_eval.py --dirs results_eval/base_pi05 results_eval/ft_dagger_r0"
echo "============================================================"
