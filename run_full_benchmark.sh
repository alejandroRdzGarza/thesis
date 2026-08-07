#!/usr/bin/env bash
# run_full_benchmark.sh — VLA+CBF full benchmark runner
#
# Runs all four evaluation configurations sequentially with logging.
# Requires:  conda activate libero  (simulation environment on this machine)
#            VLA server reachable at localhost:8000  (via RunPod tunnel)
#
# Usage:
#   bash run_full_benchmark.sh [EPISODES] [--save-video] [--show-every N]
#
# Examples:
#   bash run_full_benchmark.sh 10                      # 10 eps, no video
#   bash run_full_benchmark.sh 10 --save-video         # save MP4 for every episode
#   bash run_full_benchmark.sh 10 --show-every 1       # live cv2 window every episode
#   bash run_full_benchmark.sh 10 --save-video --show-every 5
#
# Timing at N=10 eps/task:
#   libero_spatial     10 tasks × N × 2 modes × ~30s/ep ≈ 100 min
#   safelibero_spatial  4 tasks × N × 2 modes × ~45s/ep ≈  60 min
#   Total ≈ 2.5–3 hours
#
# Results are written to results_libero/ as:
#   {scene}_{mode}_episodes.csv   — per-episode records
#   {scene}_{mode}_agg.csv        — aggregate stats
#   *.log                         — full console output
#   videos/{scene}/ep_XXXX.mp4   — episode recordings (if --save-video)

set -euo pipefail

# ── Argument parsing ──────────────────────────────────────────────────────────
EPISODES=10
SAVE_VIDEO=""
SHOW_EVERY=""
VLA="pi05"
PI05_HOST="127.0.0.1"
PI05_PORT=8000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --save-video)   SAVE_VIDEO="--save-video"; shift ;;
        --show-every)   SHOW_EVERY="--show-every $2"; shift 2 ;;
        --show-every=*) SHOW_EVERY="--show-every ${1#*=}"; shift ;;
        --vla)          VLA="$2"; shift 2 ;;
        --pi05-host)    PI05_HOST="$2"; shift 2 ;;
        --pi05-port)    PI05_PORT="$2"; shift 2 ;;
        [0-9]*)         EPISODES="$1"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

RESULTS_DIR="results_libero"
mkdir -p "$RESULTS_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "  VLA+CBF Full Benchmark"
echo "  episodes/task : ${EPISODES}"
echo "  save video    : ${SAVE_VIDEO:-off}"
echo "  show every    : ${SHOW_EVERY:-off}"
echo "  results dir   : ${RESULTS_DIR}/"
echo "  started       : $(timestamp)"
echo "============================================================"

# ── 1. LIBERO-Spatial — PLAIN ─────────────────────────────────────────────────
echo ""
echo ">>> [1/4] LIBERO-Spatial — PLAIN  started: $(timestamp)"
python run_libero_benchmark.py \
    --suite libero_spatial \
    --all \
    --episodes "$EPISODES" \
    --mode plain \
    --horizon 220 \
    --replan-steps 8 \
    --results-dir "$RESULTS_DIR" \
    --vla "$VLA" --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
    $SAVE_VIDEO $SHOW_EVERY \
    2>&1 | tee "$RESULTS_DIR/libero_spatial_plain.log"
echo ">>> [1/4] done: $(timestamp)"

# ── 2. LIBERO-Spatial — CBF ───────────────────────────────────────────────────
echo ""
echo ">>> [2/4] LIBERO-Spatial — CBF  started: $(timestamp)"
python run_libero_benchmark.py \
    --suite libero_spatial \
    --all \
    --episodes "$EPISODES" \
    --mode cbf \
    --horizon 220 \
    --replan-steps 8 \
    --results-dir "$RESULTS_DIR" \
    --vla "$VLA" --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
    $SAVE_VIDEO $SHOW_EVERY \
    2>&1 | tee "$RESULTS_DIR/libero_spatial_cbf.log"
echo ">>> [2/4] done: $(timestamp)"

# ── 3. SafeLIBERO-Spatial LI — PLAIN ─────────────────────────────────────────
echo ""
echo ">>> [3/4] SafeLIBERO-Spatial LI — PLAIN  started: $(timestamp)"
python run_libero_benchmark.py \
    --suite safelibero_spatial \
    --safety-level I \
    --all \
    --episodes "$EPISODES" \
    --mode plain \
    --horizon 300 \
    --replan-steps 8 \
    --results-dir "$RESULTS_DIR" \
    --vla "$VLA" --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
    $SAVE_VIDEO $SHOW_EVERY \
    2>&1 | tee "$RESULTS_DIR/safelibero_spatial_LI_plain.log"
echo ">>> [3/4] done: $(timestamp)"

# ── 4. SafeLIBERO-Spatial LI — CBF ───────────────────────────────────────────
echo ""
echo ">>> [4/4] SafeLIBERO-Spatial LI — CBF  started: $(timestamp)"
python run_libero_benchmark.py \
    --suite safelibero_spatial \
    --safety-level I \
    --all \
    --episodes "$EPISODES" \
    --mode cbf \
    --horizon 300 \
    --replan-steps 8 \
    --results-dir "$RESULTS_DIR" \
    --vla "$VLA" --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
    $SAVE_VIDEO $SHOW_EVERY \
    2>&1 | tee "$RESULTS_DIR/safelibero_spatial_LI_cbf.log"
echo ">>> [4/4] done: $(timestamp)"

echo ""
echo "============================================================"
echo "  Benchmark complete: $(timestamp)"
echo "  Results: ${RESULTS_DIR}/"
if [[ -n "$SAVE_VIDEO" ]]; then
echo "  Videos:  ${RESULTS_DIR}/videos/"
fi
echo ""
echo "  Next step — generate figures:"
echo "    python analysis/benchmark_analysis.py --results-dir ${RESULTS_DIR}"
echo "============================================================"
