#!/usr/bin/env bash
# run_full_benchmark.sh — VLA+CBF full benchmark runner
#
# Runs all four evaluation configurations sequentially with logging.
# Requires:  conda activate libero  (simulation environment on this machine)
#            VLA server reachable at localhost:8000  (via RunPod tunnel)
#
# Usage:
#   bash run_full_benchmark.sh [EPISODES_PER_TASK]
#
# Timing at N=10 eps/task (220-step horizon):
#   libero_spatial    10 tasks × N × 2 modes × ~30s/ep ≈ 100 min
#   safelibero_spatial 4 tasks × N × 2 modes × ~45s/ep ≈  60 min
#   Total ≈ 2.5–3 hours (overnight on Mac with RunPod tunnel)
#
# Results are written to results_libero/ as:
#   {scene}_{mode}_episodes.csv   — per-episode records
#   {scene}_{mode}_agg.csv        — aggregate stats
#   *_run.log                     — full console output
#
# After the run, generate figures on this machine:
#   python analysis/benchmark_analysis.py

set -euo pipefail

EPISODES=${1:-10}
RESULTS_DIR="results_libero"
mkdir -p "$RESULTS_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "============================================================"
echo "  VLA+CBF Full Benchmark"
echo "  episodes/task : ${EPISODES}"
echo "  results dir   : ${RESULTS_DIR}/"
echo "  started       : $(timestamp)"
echo "============================================================"

# ── 1. LIBERO-Spatial baseline (no obstacles) ─────────────────────────────────
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
    2>&1 | tee "$RESULTS_DIR/libero_spatial_plain.log"
echo ">>> [1/4] done: $(timestamp)"

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
    2>&1 | tee "$RESULTS_DIR/libero_spatial_cbf.log"
echo ">>> [2/4] done: $(timestamp)"

# ── 2. SafeLIBERO-Spatial Level I (obstacle near target) ─────────────────────
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
    2>&1 | tee "$RESULTS_DIR/safelibero_spatial_LI_plain.log"
echo ">>> [3/4] done: $(timestamp)"

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
    2>&1 | tee "$RESULTS_DIR/safelibero_spatial_LI_cbf.log"
echo ">>> [4/4] done: $(timestamp)"

echo ""
echo "============================================================"
echo "  Benchmark complete: $(timestamp)"
echo "  Results: ${RESULTS_DIR}/"
echo ""
echo "  Next step — generate figures:"
echo "    python analysis/benchmark_analysis.py --results-dir ${RESULTS_DIR}"
echo "============================================================"
