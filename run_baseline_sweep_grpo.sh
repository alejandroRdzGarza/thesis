#!/usr/bin/env bash
# run_baseline_sweep_grpo.sh — the "before RL" reference for the flow-SDE GRPO thesis result.
#
# Evaluates the BASE pi05 policy (no RL) across the full SafeLIBERO grid, both WITH the CBF
# shield and WITHOUT it, using the SAME corrected-metric rollout path the GRPO eval uses
# (auto_goal from BDDL, success = env.check_success only, standard check_ontop). Deterministic:
# noise_level=0 (ODE) + K=1, so one honest rollout per initial state.
#
# This is independent of any running GRPO training (read-only on the base checkpoint, separate
# OUT dir) — safe to run on UCL in parallel with a RunPod GRPO loop.
#
#   source experiments/rl_env.sh          # sets PY, PYTHONPATH, and usually CKPT
#   CKPT=$CKPT ./run_baseline_sweep_grpo.sh 2>&1 | tee baseline_sweep.log
#
# Resumable: any config whose round_summary.json exists is skipped. Aggregates at the end.
set -euo pipefail

CKPT=${CKPT:?set CKPT to the base pi05_libero checkpoint dir}
CONFIG=${CONFIG:-pi05_libero_cbf}
OUT=${OUT:-results_baseline_v2}
SUITES=${SUITES:-"safelibero_spatial safelibero_object safelibero_goal"}
LEVELS=${LEVELS:-"I II"}
TASKS=${TASKS:-"0 1 2 3"}
EPISODES=${EPISODES:-"0 1 2 3"}
HORIZON=${HORIZON:-300}
REPLAN=${REPLAN:-5}
NUM_STEPS=${NUM_STEPS:-10}
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

echo "==================================================================="
echo " SafeLIBERO baseline sweep (base pi05, corrected metrics)"
echo "   suites : $SUITES"
echo "   levels : $LEVELS   tasks: $TASKS   episodes: [$EPISODES]"
echo "   modes  : cbf + nocbf   (ODE, noise=0, K=1)   -> $OUT/"
echo "==================================================================="

n_done=0; n_fail=0
for suite in $SUITES; do
  for level in $LEVELS; do
    for task in $TASKS; do
      for mode in cbf nocbf; do
        tag="${suite}_L${level}_t${task}_${mode}"
        dir="$OUT/$tag"
        if [[ -f "$dir/round_summary.json" ]]; then
          echo "[skip] $tag (already done)"; continue
        fi
        cbf_flag=""; [[ "$mode" == "nocbf" ]] && cbf_flag="--no-cbf"
        echo; echo "=== $tag ==="
        if $PY -m experiments.rl_rollout_local \
              --config "$CONFIG" --checkpoint "$CKPT" \
              --suite "$suite" --level "$level" --task "$task" \
              --episodes $EPISODES --K 1 \
              --horizon "$HORIZON" --replan "$REPLAN" \
              --num-steps "$NUM_STEPS" --noise-level 0 --sde-type cps \
              $cbf_flag --out "$dir"; then
          n_done=$((n_done + 1))
        else
          echo "[FAIL] $tag — continuing"; n_fail=$((n_fail + 1))
        fi
      done
    done
  done
done

echo; echo "sweep finished: $n_done ok, $n_fail failed. Aggregating -> $OUT/baseline_summary.csv"
OUT="$OUT" "$PY" - <<'PY'
import os, json, csv, glob
out = os.environ["OUT"]
rows = []
for sm in sorted(glob.glob(os.path.join(out, "*", "round_summary.json"))):
    tag = os.path.basename(os.path.dirname(sm))
    # tag = safelibero_<suite>_L<level>_t<task>_<mode>
    try:
        rest, mode = tag.rsplit("_", 1)
        pre, task = rest.rsplit("_t", 1)
        suite, level = pre.rsplit("_L", 1)
    except ValueError:
        suite = level = task = mode = "?"
    with open(sm) as f:
        s = json.load(f)
    rows.append({
        "suite": suite, "level": level, "task": task, "mode": mode,
        "n": s.get("n_rollouts"),
        "success_rate": s.get("success_rate"),
        "collision_rate": s.get("robot_caused_collision_rate"),
        "mean_cbf_penalty": s.get("mean_cbf_penalty"),
        "mean_reward": s.get("mean_reward"),
    })
if rows:
    csv_path = os.path.join(out, "baseline_summary.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"\nwrote {csv_path}  ({len(rows)} configs)\n")
    hdr = f"{'suite':<20}{'L':<3}{'t':<3}{'mode':<7}{'succ':>7}{'coll':>7}{'cbf_pen':>9}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        print(f"{r['suite']:<20}{r['level']:<3}{r['task']:<3}{r['mode']:<7}"
              f"{r['success_rate']!s:>7}{r['collision_rate']!s:>7}{r['mean_cbf_penalty']!s:>9}")
else:
    print("no round_summary.json found — nothing to aggregate")
PY
echo "done."
