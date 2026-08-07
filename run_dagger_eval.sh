#!/usr/bin/env bash
# run_dagger_eval.sh — Evaluate a single DAgger checkpoint on SafeLIBERO
#
# Wraps run_eval.sh with DAgger-specific conventions:
#   - Tags results by round (e.g. dagger_r0, dagger_r1, base_pi05)
#   - Runs both plain and CBF modes (plain = raw policy improvement, cbf = safety ceiling)
#   - After evaluation, prints the behavioral shift report via dagger_analysis.py
#
# PREREQUISITES:
#   1. Fine-tuned checkpoint served on UCL GPU machine:
#        source $BASE/env.sh
#        uv run python -m openpi.serve.websocket_server \
#            --policy.config pi05_libero_cbf \
#            --policy.dir $BASE/thesis/openpi/checkpoints/pi05_libero_cbf/dagger_r0/9999
#   2. SSH tunnel open:
#        bash runpod/tunnel_ucl.sh
#
# USAGE:
#   # Evaluate DAgger round 0 (fine-tuned checkpoint):
#   bash run_dagger_eval.sh --round dagger_r0 --episodes 50
#
#   # Evaluate base π0.5 (no fine-tuning) as reference:
#   bash run_dagger_eval.sh --round base_pi05 --episodes 50
#
#   # Quick sanity check (5 episodes, spatial suite only):
#   bash run_dagger_eval.sh --round dagger_r0 --episodes 5 --suite safelibero_spatial
#
#   # After collecting both rounds, compare:
#   python experiments/dagger_analysis.py \
#       --rounds base_pi05 dagger_r0 \
#       --dirs   results_dagger/base_pi05 results_dagger/dagger_r0 \
#       --plot

set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
ROUND=""
EPISODES=50
PI05_HOST="127.0.0.1"
PI05_PORT=8000
OUT_BASE="results_dagger"
SUITE_ARG=""
MODES=(plain cbf)    # both modes: plain shows raw policy improvement, cbf shows safety ceiling
SAVE_VIDEO=""
SHOW_EVERY=""
SAFETY_PROMPT=""
COMPARE_ROUNDS=()
COMPARE_DIRS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --round)        ROUND="$2";         shift 2 ;;
        --episodes)     EPISODES="$2";      shift 2 ;;
        --pi05-host)    PI05_HOST="$2";     shift 2 ;;
        --pi05-port)    PI05_PORT="$2";     shift 2 ;;
        --out)          OUT_BASE="$2";      shift 2 ;;
        --suite)        SUITE_ARG="$2";     shift 2 ;;
        --cbf-only)     MODES=(cbf);        shift ;;
        --plain-only)   MODES=(plain);      shift ;;
        --save-video)    SAVE_VIDEO="--save-video"; shift ;;
        --show-every)    SHOW_EVERY="--show-every $2"; shift 2 ;;
        --safety-prompt) SAFETY_PROMPT="--safety-prompt"; shift ;;
        --compare)
            # --compare round1:dir1 round2:dir2  (for post-eval analysis)
            shift
            while [[ $# -gt 0 && "$1" != --* ]]; do
                label="${1%%:*}"
                dir="${1#*:}"
                COMPARE_ROUNDS+=("$label")
                COMPARE_DIRS+=("$dir")
                shift
            done
            ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -z "$ROUND" ]]; then
    echo "ERROR: --round is required (e.g. --round dagger_r0)"
    echo "Usage: bash run_dagger_eval.sh --round <tag> [--episodes N] [--suite SUITE]"
    exit 1
fi

OUT="${OUT_BASE}/${ROUND}"
LOG_DIR="${OUT}/logs"
mkdir -p "$OUT" "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

if [[ -n "$SUITE_ARG" ]]; then
    SUITES=("$SUITE_ARG")
else
    SUITES=(safelibero_spatial safelibero_object safelibero_goal)
fi
LEVELS=(I II)
N_TASKS=4   # tasks per suite (all SafeLIBERO suites have 4)

total_combos=$(( ${#SUITES[@]} * ${#LEVELS[@]} * N_TASKS * ${#MODES[@]} ))

echo "============================================================"
echo "  DAgger Evaluation"
echo "  round          : ${ROUND}"
echo "  suites         : ${SUITES[*]}"
echo "  levels         : ${LEVELS[*]}"
echo "  modes          : ${MODES[*]}"
echo "  tasks/suite    : ${N_TASKS}"
echo "  episodes/task  : ${EPISODES}"
echo "  total combos   : ${total_combos}"
echo "  output         : ${OUT}/"
echo "  π0.5 server    : ws://${PI05_HOST}:${PI05_PORT}"
echo "  started        : $(timestamp)"
echo "============================================================"

# Server reachability check
echo -n "  Checking π0.5 server ... "
python -c "
import socket, sys
s = socket.socket(); s.settimeout(5)
try:
    s.connect(('${PI05_HOST}', ${PI05_PORT})); s.close(); print('OK')
except Exception as e:
    print(f'UNREACHABLE: {e}'); sys.exit(1)
" || {
    echo ""
    echo "  ERROR: Cannot reach π0.5 server at ${PI05_HOST}:${PI05_PORT}"
    echo "  Make sure the tunnel is running: bash runpod/tunnel_ucl.sh"
    exit 1
}

# ── Run evaluation — paired: suite → level → task → (plain, cbf) ─────────────
combo=0
n_failed=0

for suite in "${SUITES[@]}"; do
    for level in "${LEVELS[@]}"; do
        for (( task=0; task<N_TASKS; task++ )); do
            for mode in "${MODES[@]}"; do
                combo=$((combo + 1))
                tag="${suite}_t$(printf '%02d' $task)_L${level}_${mode}"

                # Resume check: skip if agg CSV already exists for this task/mode
                if ls "${OUT}"/*"$(printf 't%02d' $task)"*"L${level}"*"${mode}"*_agg.csv 2>/dev/null | grep -q .; then
                    echo ""
                    echo "  [${combo}/${total_combos}] ${ROUND}/${tag}: already done, skipping"
                    continue
                fi

                log="${LOG_DIR}/${tag}_$(date +%Y%m%d_%H%M%S).log"
                echo ""
                echo "============================================================"
                echo "  [${combo}/${total_combos}] ${ROUND} / ${tag}"
                echo "  started: $(timestamp)"
                echo "============================================================"

                python run_libero_benchmark.py \
                    --suite        "$suite"     \
                    --safety-level "$level"     \
                    --task         "$task"      \
                    --episodes     "$EPISODES"  \
                    --mode         "$mode"      \
                    --horizon      600          \
                    --replan-steps 5            \
                    --safety-radius 0.18        \
                    --vla          pi05         \
                    --pi05-host    "$PI05_HOST" \
                    --pi05-port    "$PI05_PORT" \
                    --results-dir  "$OUT"       \
                    $SAVE_VIDEO $SHOW_EVERY $SAFETY_PROMPT \
                    2>&1 | tee "$log" || {
                    echo "  [ERROR] ${ROUND}/${tag} failed — see ${log}"
                    n_failed=$((n_failed + 1))
                }

                echo "  [${combo}/${total_combos}] done: $(timestamp)"
            done
        done
    done
done

echo ""
echo "============================================================"
echo "  EVALUATION COMPLETE: ${ROUND}  $(timestamp)"
echo "  Results: ${OUT}/"
echo ""

if [[ $n_failed -gt 0 ]]; then
    echo "  WARNING: ${n_failed} combo(s) failed"
fi

echo "  Aggregate CSVs:"
find "$OUT" -name "*_agg.csv" | sort | while read f; do echo "    $f"; done
echo ""

# ── Optional cross-round comparison ──────────────────────────────────────────
if [[ ${#COMPARE_ROUNDS[@]} -gt 0 ]]; then
    echo "============================================================"
    echo "  CROSS-ROUND COMPARISON"
    echo "============================================================"

    # Build --rounds and --dirs args for dagger_analysis.py
    ANALYSIS_ROUNDS=()
    ANALYSIS_DIRS=()
    for i in "${!COMPARE_ROUNDS[@]}"; do
        ANALYSIS_ROUNDS+=("${COMPARE_ROUNDS[$i]}")
        ANALYSIS_DIRS+=("${COMPARE_DIRS[$i]}")
    done
    # Also include this round
    ANALYSIS_ROUNDS+=("$ROUND")
    ANALYSIS_DIRS+=("$OUT")

    python experiments/dagger_analysis.py \
        --rounds "${ANALYSIS_ROUNDS[@]}" \
        --dirs   "${ANALYSIS_DIRS[@]}"   \
        ${SUITE_ARG:+--suite "$SUITE_ARG"}
fi

echo ""
echo "  To compare against other rounds:"
echo "    python experiments/dagger_analysis.py \\"
echo "        --rounds base_pi05 ${ROUND} \\"
echo "        --dirs   results_dagger/base_pi05 ${OUT} \\"
echo "        --plot"
echo "============================================================"

exit $(( n_failed > 0 ? 1 : 0 ))
