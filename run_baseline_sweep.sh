#!/usr/bin/env bash
# run_baseline_sweep.sh — Full SafeLIBERO baseline sweep: resumable + server-resilient.
#
# Runs plain vs cbf (AEGIS-faithful) over all suites × levels × tasks, one task per
# invocation so a dropped π0.5 connection only loses the current task. Already-finished
# tasks are skipped (resume), and each task is retried after waiting for the server to
# come back. Results are organised into per-suite/level folders.
#
# Usage:
#   bash run_baseline_sweep.sh                       # full sweep, defaults below
#   EPISODES=20 bash run_baseline_sweep.sh           # override episodes
#   SUITES="safelibero_spatial" bash run_baseline_sweep.sh   # one suite only
#   OUT=results_myrun bash run_baseline_sweep.sh
#
# Then summarise:  python experiments/summarize_results.py --dir <OUT>

set -uo pipefail   # NOT -e: one failure must not abort the whole sweep

SUITES=${SUITES:-"safelibero_spatial safelibero_object safelibero_goal"}
LEVELS=${LEVELS:-"I II"}
NTASKS=${NTASKS:-4}
EPISODES=${EPISODES:-10}
HORIZON=${HORIZON:-400}
REPLAN=${REPLAN:-5}
PI05_HOST=${PI05_HOST:-127.0.0.1}
PI05_PORT=${PI05_PORT:-8000}
OUT=${OUT:-results_baseline_21_jul}
MAX_RETRIES=${MAX_RETRIES:-4}
WAIT_SECS=${WAIT_SECS:-30}

mkdir -p "$OUT"
ts() { date '+%Y-%m-%d %H:%M:%S'; }

server_up() {
  python - "$PI05_HOST" "$PI05_PORT" <<'PY' 2>/dev/null
import socket, sys
s = socket.socket(); s.settimeout(5)
try:
    s.connect((sys.argv[1], int(sys.argv[2]))); s.close()
except Exception:
    sys.exit(1)
PY
}

wait_for_server() {
  local n=0
  until server_up; do
    n=$((n+1))
    echo "  [wait] π0.5 unreachable at ${PI05_HOST}:${PI05_PORT} (check #${n}) — retry in ${WAIT_SECS}s"
    sleep "$WAIT_SECS"
  done
}

task_done() {   # both plain+cbf agg CSVs exist for this suite/level/task
  local dir=$1 suite=$2 task=$3 level=$4
  local t; t=$(printf 't%02d' "$task")
  ls "$dir"/*"${suite}"*"${t}"*"L${level}"_plain_agg.csv >/dev/null 2>&1 \
    && ls "$dir"/*"${suite}"*"${t}"*"L${level}"_cbf_agg.csv >/dev/null 2>&1
}

echo "============================================================"
echo "  SafeLIBERO baseline sweep   started $(ts)"
echo "  suites: ${SUITES}   levels: ${LEVELS}   tasks: ${NTASKS}"
echo "  episodes/task: ${EPISODES}   horizon: ${HORIZON}   out: ${OUT}"
echo "============================================================"

n_ok=0; n_skip=0; n_fail=0
for suite in $SUITES; do
  for level in $LEVELS; do
    dir="${OUT}/${suite}_L${level}"
    mkdir -p "$dir"
    for (( task=0; task<NTASKS; task++ )); do
      if task_done "$dir" "$suite" "$task" "$level"; then
        echo "[skip] ${suite} L${level} t${task} (already done)"; n_skip=$((n_skip+1)); continue
      fi
      ok=0
      for (( attempt=1; attempt<=MAX_RETRIES; attempt++ )); do
        wait_for_server
        echo ""; echo "=== ${suite} L${level} t${task}  attempt ${attempt}/${MAX_RETRIES}  $(ts) ==="
        python run_libero_benchmark.py \
          --suite "$suite" --safety-level "$level" \
          --task "$task" --mode both --episodes "$EPISODES" \
          --vla pi05 --replan-steps "$REPLAN" --horizon "$HORIZON" \
          --pi05-host "$PI05_HOST" --pi05-port "$PI05_PORT" \
          --results-dir "$dir" 2>&1 | tee -a "${dir}/t$(printf '%02d' $task).log"
        rc=${PIPESTATUS[0]}
        if [[ $rc -eq 0 ]] && task_done "$dir" "$suite" "$task" "$level"; then
          echo "[ok] ${suite} L${level} t${task}"; ok=1; break
        fi
        echo "[retry] ${suite} L${level} t${task} failed (rc=${rc}) — waiting ${WAIT_SECS}s"
        sleep "$WAIT_SECS"
      done
      if [[ $ok -eq 1 ]]; then n_ok=$((n_ok+1)); else
        echo "[FAIL] ${suite} L${level} t${task} gave up after ${MAX_RETRIES} attempts"; n_fail=$((n_fail+1)); fi
    done
  done
done

echo ""; echo "============================================================"
echo "  SWEEP COMPLETE $(ts)   ok=${n_ok} skipped=${n_skip} failed=${n_fail}"
echo "  Summarise:  python experiments/summarize_results.py --dir ${OUT}"
echo "============================================================"
[[ $n_fail -eq 0 ]]
