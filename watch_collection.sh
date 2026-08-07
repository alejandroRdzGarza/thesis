#!/bin/bash
# Live view of a collect_bc_all run. The parent process only prints when a whole shard finishes
# (~35 rollouts, ~35 min), so it looks stalled long before it is; the per-shard logs are where the
# progress actually is.  usage: ./watch_collection.sh [round_dir]
D="${1:-results_distill/planner_A}"
while true; do
  clear
  echo "$D   $(date '+%H:%M:%S')"
  echo "clean demos so far: $(find "$D" -name '*_trace.npz' 2>/dev/null | wc -l | tr -d ' ')"
  echo "shards complete   : $(find "$D" -name '.complete' 2>/dev/null | wc -l | tr -d ' ') / 24"
  echo
  for f in "$D"/*/collect.log; do
    [ -f "$f" ] || continue
    printf "  %-32s %s\n" "$(basename "$(dirname "$f")")" \
      "$(grep -oE '\[[0-9]+/[0-9]+\][^|]*\|[^|]*\|[^|]*' "$f" | tail -1)"
  done
  sleep 20
done
