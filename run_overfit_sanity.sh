#!/usr/bin/env bash
# run_overfit_sanity.sh — is the BC MACHINERY sound? Hard-overfit π0.5's LoRA on ONE clean expert
# demo, then eval that SAME episode (no-CBF, ODE). Decisive diagnostic before any more DAgger work:
#
#   reproduces the demo   → BC works; the DAgger failure is coverage + the non-Markov expert
#                           (fix = make the classical expert stateless/reactive).
#   cannot reproduce it   → upstream bug (action normalization / obs pipeline / LoRA capacity);
#                           fixing the expert would be wasted effort.
#
# A single demo is ~60 examples; 60 epochs ≈ 400 steps ≈ 8 min. If overfit BC can't reproduce a
# trajectory it trained hundreds of times on, generalization was never the problem.
#
#   source experiments/rl_env_runpod.sh
#   CKPT=$CKPT ./run_overfit_sanity.sh 2>&1 | tee overfit.log
set -euo pipefail

CKPT=${CKPT:?set CKPT to the BASE pi05_libero checkpoint dir}
DEMOS=${DEMOS:-results_distill/round0_demos}   # round-0 CLEAN classical demos
LEVEL=${LEVEL:-II}
OUT=${OUT:-results_distill/overfit}
EPOCHS=${EPOCHS:-60}
LR=${LR:-2e-4}
NTH=${NTH:-0}                                   # which clean demo (0 = first)
NUM_STEPS=${NUM_STEPS:-10}
HORIZON=${HORIZON:-300}
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

IFS=$'\t' read -r TRACE SUITE TASK EP < <($PY -m experiments.pick_clean_demo "$DEMOS" "$NTH")
echo "==================================================================="
echo " OVERFIT SANITY  |  one clean demo → hard BC → reproduce same episode"
echo "   demo: $SUITE / t$TASK / ep$EP"
echo "   $TRACE"
echo "==================================================================="

ODIR="$OUT/one_demo"; rm -rf "$ODIR"; mkdir -p "$ODIR"
cp "$TRACE" "$ODIR/"
BN=$(basename "$TRACE")
printf 'trace_path,r_success,robot_caused_collision,suite,task,episode\n%s,1.5,0,%s,%s,%s\n' \
    "$(cd "$ODIR" && pwd)/$BN" "$SUITE" "$TASK" "$EP" > "$ODIR/manifest.csv"

echo; echo "### [1/2] hard-overfit BC on the single demo ($EPOCHS epochs, lr=$LR) ###"
$PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$CKPT" \
    --round "$ODIR" --out "$OUT/ckpt" --epochs "$EPOCHS" --lr "$LR"

echo; echo "### [2/2] eval the SAME episode, NO CBF, ODE (noise 0) — reproduce the demo? ###"
$PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$OUT/ckpt" \
    --suite "$SUITE" --level "$LEVEL" --task "$TASK" --episodes "$EP" --K 1 \
    --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 --no-cbf --out "$OUT/eval"

echo; echo "==================================================================="
echo " overfit eval → $OUT/eval/round_summary.json"
echo "   success_rate 1.0  → BC machinery is SOUND (fix the expert / coverage next)"
echo "   success_rate 0.0  → upstream bug (normalization / obs / capacity) — fix FIRST"
echo "==================================================================="
