#!/usr/bin/env bash
# run_safelibero_all.sh — Full SafeLIBERO benchmark: plain baseline + CBF with data collection
#
# For every suite × level combination this script runs TWO passes:
#   1. plain — no CBF, records baseline CAR/TSR/collision metrics
#   2. cbf   — CBF active, saves HDF5 per episode for OpenVLA-OFT fine-tuning
#
# Suites (default): safelibero_spatial  safelibero_object  safelibero_goal
# Levels (default): I  II
# Tasks per suite:  4  (--all flag)
#
# Estimated wall time (RunPod A100, default 25 eps):
#   3 suites × 2 levels × 2 modes × 4 tasks × 25 eps × ~45s ≈ 15 hours
#
# The script is RESUMABLE — each suite/level/mode block writes a .done_* sentinel
# on success and is silently skipped on re-run.  Failures leave no sentinel so the
# block will be retried automatically.
#
# Usage:
#   bash run_safelibero_all.sh                             # defaults
#   bash run_safelibero_all.sh 10                          # quick test (10 eps)
#   bash run_safelibero_all.sh --episodes 50               # 50 eps
#   bash run_safelibero_all.sh --level I                   # level I only
#   bash run_safelibero_all.sh --include-long              # also run safelibero_long
#   bash run_safelibero_all.sh --results-dir my_results    # custom output dir
#
# After completion, merge all CBF HDF5 files for fine-tuning:
#   python merge_cbf_dataset.py \
#       --input <RESULTS_DIR>/cbf_dataset \
#       --output cbf_finetune_data.h5

set -uo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────────
EPISODES=25
HORIZON=600
REPLAN=8
VLA=pi05
OPENVLA_PORT=8000
PI05_HOST=127.0.0.1
PI05_PORT=8000
RESULTS_DIR=results_safelibero_all
MODEL_TAG=""
INCLUDE_LONG=0
SAVE_VIDEO=""
SHOW_EVERY=""
LEVELS=(I II)

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)     EPISODES="$2";      shift 2 ;;
        --horizon)      HORIZON="$2";       shift 2 ;;
        --replan)       REPLAN="$2";        shift 2 ;;
        --results-dir)  RESULTS_DIR="$2";   shift 2 ;;
        --model-tag)    MODEL_TAG="$2";     shift 2 ;;
        --vla)          VLA="$2";           shift 2 ;;
        --openvla-port) OPENVLA_PORT="$2";  shift 2 ;;
        --pi05-host)    PI05_HOST="$2";     shift 2 ;;
        --pi05-port)    PI05_PORT="$2";     shift 2 ;;
        --level)        LEVELS=("$2");      shift 2 ;;  # single level: I or II
        --include-long) INCLUDE_LONG=1;   shift ;;
        --save-video)   SAVE_VIDEO="--save-video"; shift ;;
        --show-every)   SHOW_EVERY="--show-every $2"; shift 2 ;;
        [0-9]*)         EPISODES="$1";    shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

SUITES=(safelibero_spatial safelibero_object safelibero_goal)
if [[ $INCLUDE_LONG -eq 1 ]]; then
    SUITES+=(safelibero_long)
fi

if [[ -n "$MODEL_TAG" ]]; then
    RESULTS_DIR="${RESULTS_DIR}/${MODEL_TAG}"
fi
mkdir -p "$RESULTS_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

TOTAL_RUNS=$(( ${#SUITES[@]} * ${#LEVELS[@]} * 2 ))
EST_EPS=$(( ${#SUITES[@]} * ${#LEVELS[@]} * 2 * 4 * EPISODES ))

echo "========================================================================"
echo "  SafeLIBERO All-Suites Benchmark (plain + CBF + data collection)"
echo "  suites        : ${SUITES[*]}"
echo "  levels        : ${LEVELS[*]}"
echo "  episodes/task : $EPISODES    horizon: $HORIZON    replan: $REPLAN"
echo "  vla           : $VLA    openvla-port: $OPENVLA_PORT"
echo "  results dir   : $RESULTS_DIR/"
echo "  cbf hdf5 dir  : $RESULTS_DIR/cbf_dataset/"
echo "  total runs    : $TOTAL_RUNS"
echo "  total episodes: ~$EST_EPS  (~$(( EST_EPS * 45 / 3600 ))h estimated)"
echo "  started       : $(timestamp)"
echo "========================================================================"

run_idx=0
n_failed=0
failed_runs=()

for suite in "${SUITES[@]}"; do
    for level in "${LEVELS[@]}"; do
        for mode in plain cbf; do
            run_idx=$(( run_idx + 1 ))
            label="${suite}_L${level}_${mode}"
            sentinel="$RESULTS_DIR/.done_${label}"

            echo ""
            if [[ -f "$sentinel" ]]; then
                echo ">>> [$run_idx/$TOTAL_RUNS] SKIP (done): $suite  level=$level  mode=$mode"
                continue
            fi

            echo ">>> [$run_idx/$TOTAL_RUNS] START: $suite  level=$level  mode=$mode  $(timestamp)"

            # --collect-cbf-data only for CBF runs
            extra_flags=""
            if [[ "$mode" == "cbf" ]]; then
                extra_flags="--collect-cbf-data"
            fi

            log_file="$RESULTS_DIR/${label}.log"

            # Run and capture exit status without aborting the script on failure
            # (no sentinel written on failure → block will be retried on re-run)
            if python run_libero_benchmark.py \
                    --suite "$suite" \
                    --safety-level "$level" \
                    --all \
                    --mode "$mode" \
                    --episodes "$EPISODES" \
                    --horizon "$HORIZON" \
                    --replan-steps "$REPLAN" \
                    --vla "$VLA" \
                    --openvla-port "$OPENVLA_PORT" \
                    --pi05-host "$PI05_HOST" \
                    --pi05-port "$PI05_PORT" \
                    --results-dir "$RESULTS_DIR" \
                    $extra_flags \
                    $SAVE_VIDEO \
                    $SHOW_EVERY \
                    2>&1 | tee "$log_file"; then

                touch "$sentinel"
                echo ">>> [$run_idx/$TOTAL_RUNS] DONE: $label  $(timestamp)"
            else
                n_failed=$(( n_failed + 1 ))
                failed_runs+=("$label")
                echo ">>> [$run_idx/$TOTAL_RUNS] FAILED: $label — no sentinel written, will retry on re-run"
            fi

        done
    done
done

echo ""
echo "========================================================================"
echo "  Benchmark finished: $(timestamp)"
echo "  Results:     $RESULTS_DIR/"
echo "  CBF HDF5:    $RESULTS_DIR/cbf_dataset/"
echo ""

if [[ $n_failed -gt 0 ]]; then
    echo "  FAILED RUNS ($n_failed) — re-run this script to retry:"
    for r in "${failed_runs[@]}"; do echo "    $r"; done
    echo ""
fi

echo "  Aggregate CSVs:"
echo "    ls $RESULTS_DIR/*_agg.csv"
echo ""
echo "  Merge CBF training data for OpenVLA-OFT fine-tuning:"
echo "    python merge_cbf_dataset.py \\"
echo "        --input $RESULTS_DIR/cbf_dataset \\"
echo "        --output cbf_finetune_data.h5"
echo ""
echo "  Quick results table:"
echo "    python -c \""
echo "import glob, csv"
echo "rows = []"
echo "for f in sorted(glob.glob('$RESULTS_DIR/*_agg.csv')):"
echo "    rows += list(csv.DictReader(open(f)))"
echo "print(f'{\\\"scene\\\":<45} {\\\"mode\\\":<6} {\\\"lvl\\\":<4} {\\\"CAR%\\\":>7} {\\\"TSR%\\\":>7} {\\\"Coll%\\\":>7}')"
echo "for r in rows:"
echo "    print(f'{r[\\\"scene\\\"]:<45} {r[\\\"mode\\\"]:<6} {r.get(\\\"safety_level\\\",\\\"-\\\"):<4} {float(r[\\\"car_pct\\\"]):>7.1f} {float(r[\\\"tsr_pct\\\"]):>7.1f} {float(r[\\\"collision_rate_pct\\\"]):>7.1f}')"
echo "    \""
echo "========================================================================"

exit $(( n_failed > 0 ? 1 : 0 ))
