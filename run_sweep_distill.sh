#!/usr/bin/env bash
# run_sweep_distill.sh — the "real" distillation result: collect shielded classical demos across
# ALL tasks of a suite (train inits), BC-distil π0.5, and eval FOUR policies on HELD-OUT inits, then
# print a CI table. Answers, with statistics, "does distilling the CBF-shielded expert give a
# shield-free VLA that is safe AND capable, and does it need the shield less?"
#
#   train split  : all tasks × TRAIN_INITS   → shielded classical demos → BC (EPOCHS)
#   held-out eval: all tasks × EVAL_INITS (disjoint), deterministic (noise 0), for:
#       base+noCBF       base π0.5, no shield        (baseline: raw safety/capability)
#       distilled+noCBF  distilled, no shield        (HEADLINE: safety internalized?)
#       base+CBF         base π0.5, shielded         (shield activation baseline)
#       distilled+CBF    distilled, shielded         (does it still trigger the shield?)
#
#   source experiments/rl_env_runpod.sh
#   CKPT=$CKPT ./run_sweep_distill.sh 2>&1 | tee sweep_distill.log
# Resumable: any stage whose output dir exists is skipped. Scale via the env vars below.
set -euo pipefail

CKPT=${CKPT:?set CKPT to the BASE pi05_libero checkpoint dir}
SUITE=${SUITE:-safelibero_spatial}
LEVEL=${LEVEL:-II}
TRAIN_INITS=${TRAIN_INITS:-$(seq 0 19)}     # initial states used for demos + training
EVAL_INITS=${EVAL_INITS:-$(seq 20 34)}      # DISJOINT held-out inits for eval (generalization)
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
MINIBATCH=${MINIBATCH:-8}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
OUT=${OUT:-results_sweep}
PY=${PY:-python}
# Which policies to eval (space-separated): base_nocbf distilled_nocbf base_cbf distilled_cbf
POLICIES=${POLICIES:-"base_nocbf distilled_nocbf base_cbf distilled_cbf"}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

DEMOS="$OUT/demos"
DIST_CKPT="$OUT/distilled_ckpt"

# Auto-enumerate the suite's tasks.
TASKS=$($PY -c "from libero.libero import benchmark as b; d=b.get_benchmark_dict(); s=d['$SUITE'](safety_level='$LEVEL') if '$SUITE'.startswith('safelibero_') else d['$SUITE'](); print(' '.join(str(i) for i in range(s.get_num_tasks())))")
NTASKS=$(echo "$TASKS" | wc -w | tr -d ' ')

echo "==================================================================="
echo " SWEEP DISTILL  |  $SUITE / L$LEVEL  |  $NTASKS tasks: [$TASKS]"
echo "   train inits: [$(echo $TRAIN_INITS)]   held-out eval inits: [$(echo $EVAL_INITS)]"
echo "   BC epochs=$EPOCHS   policies: $POLICIES"
echo "==================================================================="

# ── [1] collect shielded classical demos across all tasks (train inits) ──────────────────
if [ -d "$DEMOS" ]; then
    echo "### [1/4] demos exist ($DEMOS) → skip collection ###"
else
    echo "### [1/4] collect shielded classical demos: $NTASKS tasks × train inits ###"
    $PY -m experiments.collect_classical_demos --suite "$SUITE" --level "$LEVEL" \
        --tasks $TASKS --episodes $TRAIN_INITS --horizon "$HORIZON" --out "$DEMOS"
fi

# ── [2] BC-distil π0.5 on the CLEAN demos ────────────────────────────────────────────────
if [ -d "$DIST_CKPT" ]; then
    echo "### [2/4] distilled checkpoint exists ($DIST_CKPT) → skip BC ###"
else
    echo "### [2/4] BC-distil ($EPOCHS epochs, --success-only) ###"
    $PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$CKPT" \
        --round "$DEMOS" --out "$DIST_CKPT" --success-only \
        --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"
fi

# ── [3] held-out eval, per policy × task ─────────────────────────────────────────────────
run_policy() {
    local name="$1" ckpt="$2" cbf_flag="$3"     # cbf_flag: "--no-cbf" or ""
    for t in $TASKS; do
        local od="$OUT/eval_$name/task$t"
        if [ -d "$od" ]; then
            echo "  [$name] task$t exists → skip"; continue
        fi
        echo "  [$name] task$t (held-out inits)"
        $PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$ckpt" \
            --suite "$SUITE" --level "$LEVEL" --task "$t" --episodes $EVAL_INITS --K 1 \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 $cbf_flag --out "$od"
    done
}

echo "### [3/4] held-out eval: policies × $NTASKS tasks ###"
for pol in $POLICIES; do
    case "$pol" in
        base_nocbf)      run_policy base_nocbf      "$CKPT"      "--no-cbf" ;;
        distilled_nocbf) run_policy distilled_nocbf "$DIST_CKPT" "--no-cbf" ;;
        base_cbf)        run_policy base_cbf        "$CKPT"      "" ;;
        distilled_cbf)   run_policy distilled_cbf   "$DIST_CKPT" "" ;;
        *) echo "  [warn] unknown policy '$pol' — skipping" ;;
    esac
done

# ── [4] pooled CI table ──────────────────────────────────────────────────────────────────
echo "### [4/4] pooled statistics (Wilson 95% CIs) ###"
ARGS=""
for pol in $POLICIES; do
    [ -d "$OUT/eval_$pol" ] && ARGS="$ARGS ${pol}=$OUT/eval_$pol"
done
$PY -m experiments.sweep_eval_stats $ARGS | tee "$OUT/stats_table.txt"
echo; echo "SWEEP done → table: $OUT/stats_table.txt"
