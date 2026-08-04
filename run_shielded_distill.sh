#!/usr/bin/env bash
# run_shielded_distill.sh — the full-grid shield-as-expert distillation run, end to end.
#
# This is Exp 005b scaled from one task to the WHOLE benchmark. 005b already worked on
# safelibero_spatial task 0: two BC rounds gave 0.69 success / 0.00 collision unshielded, against
# base π0.5's 0.88 / 0.94 — safety fully internalized at 78% of base success. Its stated caveats
# were exactly (a) single task, (b) 16-rollout eval. This addresses both.
#
#   [1] round 0   collect base π0.5 + CBF demos over all suites × levels × tasks (train inits)
#   [2]           BC-distil π0.5 on the clean, shield-active demos            → r1_ckpt
#   [3] round 1   collect r1_ckpt + CBF demos (the student's OWN states = DAgger)
#   [4]           BC-distil from BASE on round0 + round1 aggregated           → r2_ckpt
#   [5] eval      {base, r1, r2} × {no-CBF, CBF} on HELD-OUT inits
#   [6]           pooled Wilson-CI table
#
# ── STOP AT TWO BC ROUNDS. THIS IS NOT A TUNING KNOB. ───────────────────────────────────────
# The shield-as-expert teacher degrades: each round imitates the shield's caution everywhere, not
# only where it was needed, so success erodes monotonically. Measured in Exp 005b:
#   round 1: 0.81   round 2: 0.69 ←keeper   round 3: 0.56   round 4: 0.50   round 6: 0.12
# The success filter SLOWS this; it does not stop it. The knee is 1-2 rounds. ROUNDS>2 will make
# your numbers worse, not better — this is documented, not speculative (RESULTS_LOG.md Exp 005/005b).
#
# Each round retrains from the BASE checkpoint on the AGGREGATED data (true DAgger, pi_N =
# BC(base, D_0..N)) rather than warm-starting the LoRA, so the base anchors every round.
#
#   source /workspace/thesis/experiments/rl_env_runpod.sh
#   tmux new -s distill
#   CKPT=$CKPT ./run_shielded_distill.sh
#
# Resumable: every stage whose output directory already exists is skipped, so a killed run picks
# up where it stopped. All output is tee'd to $OUT/run_<timestamp>.log.
set -euo pipefail

CKPT=${CKPT:?set CKPT to the BASE pi05_libero checkpoint dir}
OUT=${OUT:-results_shielded}
PY=${PY:-python}

SUITES=${SUITES:-"safelibero_spatial safelibero_object safelibero_goal"}
LEVELS=${LEVELS:-"I II"}
TASKS=${TASKS:-"0 1 2 3"}
TRAIN_INITS=${TRAIN_INITS:-$(seq 0 11)}     # demos are collected from these initial states
# DISJOINT held-out inits, never trained on. Only 5 per scene because the stats POOL across all
# 24 scenes: 5 × 24 = 120 rollouts per policy arm, which is already a tighter Wilson CI than the
# 16-rollout eval Exp 005b was criticised for. Spending the budget on more SCENES beats more inits
# per scene — scene diversity is the thing the single-task result lacked.
EVAL_INITS=${EVAL_INITS:-$(seq 35 39)}
ROUNDS=${ROUNDS:-2}                          # BC rounds. See the warning above before raising.

EPOCHS=${EPOCHS:-20}                         # 20+; the old "BC gives 0% success" was 2-epoch underfit
LR=${LR:-1e-4}
MINIBATCH=${MINIBATCH:-8}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}                   # 4 ≈ 2× faster, some action-quality loss
MIN_CBF_ACTS=${MIN_CBF_ACTS:-1}              # demand the shield actually fired in a kept demo

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

mkdir -p "$OUT"
LOG="$OUT/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1        # everything from here on is logged AND shown

NSCENES=$(( $(echo $SUITES | wc -w) * $(echo $LEVELS | wc -w) * $(echo $TASKS | wc -w) ))
NTRAIN=$(echo $TRAIN_INITS | wc -w | tr -d ' ')
NEVAL=$(echo $EVAL_INITS | wc -w | tr -d ' ')
NCOLLECT=$((ROUNDS*NSCENES*NTRAIN))
NEVALTOT=$(( (ROUNDS+1)*2*NSCENES*NEVAL ))

banner() { echo; echo "════════════════════════════════════════════════════════════════════════"; \
           echo " $*  ·  $(date '+%H:%M:%S')"; \
           echo "════════════════════════════════════════════════════════════════════════"; }
stage_t0=0
start() { stage_t0=$SECONDS; echo "── $* ──"; }
finish() { echo "   ✔ $* ($((SECONDS-stage_t0)) s)"; }

banner "SHIELDED DISTILLATION  ·  $NSCENES scenes  ·  $ROUNDS BC round(s)"
echo "  base checkpoint : $CKPT"
echo "  suites/levels   : [$SUITES] × [$LEVELS] × tasks [$TASKS]"
echo "  train inits     : $NTRAIN per scene   → $((NSCENES*NTRAIN)) rollouts per collection round"
echo "  held-out eval   : $NEVAL per scene    → $((NSCENES*NEVAL)) rollouts per policy-arm"
echo "  BC              : $EPOCHS epochs, lr $LR, minibatch $MINIBATCH"
echo "  output          : $OUT     log: $LOG"
echo
echo "  ROUGH BUDGET (≈1.5 min/rollout at --num-steps $NUM_STEPS, GPU-bound, single process):"
echo "    collection : $ROUNDS rounds × $((NSCENES*NTRAIN)) = $NCOLLECT rollouts"
echo "    eval       : $((ROUNDS+1)) ckpts × 2 shield-modes × $((NSCENES*NEVAL)) = $NEVALTOT rollouts"
echo "    TOTAL      ≈ $(( (NCOLLECT + NEVALTOT) * 3 / 120 )) hours   (+ ~1-2 h of BC training)"
echo
echo "  If that is too long, in order of least damage to the result:"
echo "    NUM_STEPS=4                  ~2× faster everywhere (some action-quality loss)"
echo "    TRAIN_INITS=\"\$(seq 0 7)\"     fewer demos per scene, keeps full scene coverage"
echo "    LEVELS=II                    half the grid; II is where the shield matters most"
echo "    ROUNDS=1                     round-0 BC only — still the headline result, no DAgger"
echo "  Do NOT cut scenes to save time: multi-scene coverage is the whole point of this run."

# ── collection + BC, per round ───────────────────────────────────────────────────────────
prev_ckpt="$CKPT"
round_dirs=""
for r in $(seq 0 $((ROUNDS-1))); do
    DEMOS="$OUT/round${r}_demos"
    RCKPT="$OUT/round${r}_ckpt"

    banner "ROUND $r  ·  collect"
    if [ -d "$DEMOS" ] && [ -f "$DEMOS/manifest.csv" ]; then
        echo "   demos exist → skip  ($(( $(wc -l < "$DEMOS/manifest.csv") - 1 )) demos)"
    else
        start "rolling out $( [ "$r" = 0 ] && echo 'BASE π0.5' || echo "round $((r-1)) student" ) + CBF shield"
        $PY -m experiments.collect_shielded_demos \
            --config pi05_libero_cbf --checkpoint "$prev_ckpt" \
            --suites $SUITES --levels $LEVELS --tasks $TASKS --episodes $TRAIN_INITS \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 \
            --min-cbf-acts "$MIN_CBF_ACTS" --out "$DEMOS"
        finish "round $r collection"
    fi
    round_dirs="$round_dirs $DEMOS"

    banner "ROUND $r  ·  BC-distil (from BASE, on aggregated data)"
    if [ -d "$RCKPT" ]; then
        echo "   checkpoint exists → skip"
    else
        start "flow_bc_train on [$round_dirs ]"
        # Always from $CKPT, never from prev_ckpt: pi_N = BC(base, D_0..N) keeps the base as the
        # anchor, which is what stops the round-over-round drift from compounding into the LoRA.
        $PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$CKPT" \
            --round $round_dirs --out "$RCKPT" --success-only \
            --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"
        finish "round $r BC"
    fi
    prev_ckpt="$RCKPT"
done

# ── held-out eval: every checkpoint × {no-CBF, CBF} ──────────────────────────────────────
eval_arm() {                       # name, ckpt, cbf_flag
    local name="$1" ckpt="$2" flag="$3"
    for s in $SUITES; do for l in $LEVELS; do for t in $TASKS; do
        local od="$OUT/eval_$name/${s}_L${l}_t${t}"
        [ -d "$od" ] && continue
        $PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$ckpt" \
            --suite "$s" --level "$l" --task "$t" --episodes $EVAL_INITS --K 1 \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 $flag --out "$od" \
            || echo "   [warn] $name $s L$l t$t failed — continuing"
    done; done; done
    echo "   ✔ $name done"
}

banner "HELD-OUT EVAL  ·  inits [$(echo $EVAL_INITS)]  ·  never trained on"
start "base π0.5"
eval_arm base_nocbf "$CKPT" "--no-cbf"
eval_arm base_cbf   "$CKPT" ""
for r in $(seq 0 $((ROUNDS-1))); do
    eval_arm "r$((r+1))_nocbf" "$OUT/round${r}_ckpt" "--no-cbf"
    eval_arm "r$((r+1))_cbf"   "$OUT/round${r}_ckpt" ""
done
finish "eval"

# ── pooled CI table ──────────────────────────────────────────────────────────────────────
banner "RESULTS  ·  pooled Wilson 95% CIs"
ARGS=""
for a in base_nocbf base_cbf $(for r in $(seq 1 $ROUNDS); do echo "r${r}_nocbf r${r}_cbf"; done); do
    [ -d "$OUT/eval_$a" ] && ARGS="$ARGS ${a}=$OUT/eval_$a"
done
$PY -m experiments.sweep_eval_stats $ARGS | tee "$OUT/stats_table.txt"

banner "DONE"
echo "  table : $OUT/stats_table.txt"
echo "  log   : $LOG"
echo
echo "  READ IT LIKE THIS:"
echo "   • headline  = rN_nocbf collision ≪ base_nocbf collision at comparable success"
echo "   • pick the KNEE across rounds (005b: r2 was best; success erodes after)"
echo "   • rN_cbf success may fall BELOW base_cbf — expected, not a bug: once the policy"
echo "     already avoids, the shield over-corrects it off the grasp. The claim is"
echo "     shield-FREE operation plus a lower CBF-activation proxy, not stacking with the shield."
