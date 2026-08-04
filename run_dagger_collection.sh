#!/usr/bin/env bash
# run_dagger_collection.sh — Full SafeLIBERO DAgger collection with π0.5 + CBF
#
# Covers the complete SafeLIBERO benchmark:
#   4 suites × 4 tasks × 2 levels × 50 episodes = 1,600 episodes
#
# PREREQUISITES:
#   1. On gadwall-l (in tmux):
#        BASE=/cs/student/project_msc/2025/rai/jesusr01
#        cd $BASE/openpi
#        XLA_PYTHON_CLIENT_PREALLOCATE=false UV_CACHE_DIR=$BASE/.uv-cache \
#          uv run scripts/serve_policy.py --env LIBERO
#
#   2. Tunnel open on Mac (keep running):
#        GPU_HOST=gadwall-l bash runpod/tunnel_ucl.sh
#
#   3. Rsync latest code to UCL first:
#        rsync -avz --exclude='__pycache__' --exclude='.git' \
#          --exclude='openpi' --exclude='data' \
#          . jesusr01@knuckles.cs.ucl.ac.uk:/cs/student/project_msc/2025/rai/jesusr01/thesis/
#
# USAGE:
#   bash run_dagger_collection.sh                          # full run, 50 ep/scenario
#   bash run_dagger_collection.sh --episodes 20            # quick test
#   bash run_dagger_collection.sh --suite safelibero_spatial  # single suite only
#   bash run_dagger_collection.sh --out data/dagger_r1     # second round
#
# Resume-safe: existing ep_NNNN.npz files are skipped automatically.
# Fault-tolerant: individual episode/task crashes are logged and skipped.

set -euo pipefail

EPISODES=50
OUT="data/dagger_r0"
PI05_HOST="127.0.0.1"
PORT=8000
LOG_DIR="logs/dagger_collection"
SUITES_ARG=""   # empty = run all 4 suites
DAGGER_ROUND=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --episodes)      EPISODES="$2";      shift 2 ;;
        --out)           OUT="$2";           shift 2 ;;
        --pi05-host)     PI05_HOST="$2";     shift 2 ;;
        --port)          PORT="$2";          shift 2 ;;
        --suite)         SUITES_ARG="$2";    shift 2 ;;
        --dagger-round)  DAGGER_ROUND="$2";  shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

# All 4 SafeLIBERO suites
if [[ -n "$SUITES_ARG" ]]; then
    SUITES=($SUITES_ARG)     # space-separated list ok: --suite "safelibero_object safelibero_goal"
else
    SUITES=(safelibero_spatial safelibero_object safelibero_goal)   # 3 baseline suites (drop 'long')
fi
LEVELS=(I II)

mkdir -p "$LOG_DIR" "$OUT"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

total_combos=$(( ${#SUITES[@]} * ${#LEVELS[@]} ))
total_eps=$(( total_combos * 4 * EPISODES ))

echo "============================================================"
echo "  DAgger Collection — π0.5 + CBF — Full SafeLIBERO"
echo "  suites         : ${SUITES[*]}"
echo "  levels         : I  II"
echo "  tasks/suite    : 4"
echo "  episodes/combo : ${EPISODES}"
echo "  total episodes : ${total_eps}  (${total_combos} suite×level combos × 4 tasks)"
echo "  output         : ${OUT}/"
echo "  π0.5 server    : ws://${PI05_HOST}:${PORT}"
echo "  dagger round   : ${DAGGER_ROUND}"
echo "  started        : $(timestamp)"
echo "  Resume-safe    : existing ep_*.npz files will be skipped"
echo "============================================================"

# Verify server is reachable before committing to a long run
echo -n "  Checking π0.5 server ... "
if ! ${PY:-python} -c "
import socket, sys
s = socket.socket()
s.settimeout(5)
try:
    s.connect(('${PI05_HOST}', ${PORT}))
    s.close()
    print('OK')
except Exception as e:
    print(f'UNREACHABLE: {e}')
    sys.exit(1)
"; then
    echo ""
    echo "  ERROR: Cannot reach π0.5 server at ${PI05_HOST}:${PORT}"
    echo "  Make sure the tunnel is running: GPU_HOST=gadwall-l bash runpod/tunnel_ucl.sh"
    exit 1
fi

combo=0
for suite in "${SUITES[@]}"; do
    for level in "${LEVELS[@]}"; do
        combo=$((combo + 1))
        tag="${suite}_L${level}"
        log="${LOG_DIR}/${tag}_$(date +%Y%m%d_%H%M%S).log"

        echo ""
        echo "============================================================"
        echo "  COMBO [${combo}/${total_combos}]: ${tag}"
        echo "  started: $(timestamp)"
        echo "============================================================"

        ${PY:-python} -m experiments.collect_obstacle_data \
            --suite        "$suite"    \
            --safety-level "$level"   \
            --all-tasks               \
            --episodes     "$EPISODES" \
            --correction   cbf        \
            --cbf-near-goal-off       \
            --vla          pi05       \
            --pi05-host    "$PI05_HOST" \
            --openvla-port "$PORT"    \
            --out          "$OUT"     \
            --dagger-round "$DAGGER_ROUND" \
            2>&1 | tee "$log" || {
            echo "  [ERROR] combo ${tag} exited with error — see ${log}"
            echo "  Continuing to next combo..."
        }

        echo "  COMBO [${combo}/${total_combos}]: ${tag}  done: $(timestamp)"
    done
done

echo ""
echo "============================================================"
echo "  ALL DONE: $(timestamp)"
echo "  Dataset: ${OUT}/"
total_npz=$(find "$OUT" -name "*.npz" 2>/dev/null | wc -l | tr -d ' ')
echo "  Total episodes saved: ${total_npz} / ${total_eps}"
echo ""
echo "  Next: convert to LeRobot format:"
echo "    cd openpi && uv run python ../experiments/convert_cbf_npz_to_lerobot.py \\"
echo "        --data-dir ../${OUT} --success-only"
echo "============================================================"
