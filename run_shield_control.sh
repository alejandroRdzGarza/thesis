#!/bin/bash
# run_shield_control.sh — the CONTROL ABLATION for the shielded-distillation result.
#
# The headline result trains on rollouts filtered by success AND collision-free, collected WITH the
# CBF shield. That confounds two mechanisms:
#   (a) the policy imitates the shield's avoidance corrections, and
#   (b) plain success-filtering — training on your own good episodes improves a policy by itself,
#       and successful episodes correlate with not knocking things over.
# Only (a) supports the thesis claim; (b) is a known self-improvement effect that would make the
# shield incidental. This script separates them.
#
# Design: TWO arms trained on an IDENTICAL number of demos, filtered identically, differing only in
# whether the shield was on during collection.
#
#     noshield   N demos, shield OFF during collection   <- control
#     matched    N demos, shield ON  during collection   <- comparison
#
# N is whatever the unshielded collection yielded; the shielded set is subsampled down to it. The
# 186-demo r1 set is NOT part of this comparison — it is the headline arm, trained on everything.
# Matching is between the two arms here; N only needs to be LARGE enough that both arms train to a
# level where a difference is detectable (a tiny N floors both and proves nothing).
#
# Hyperparameters are copied from run_shielded_distill.sh so the arms are comparable to r1 and to
# each other. EPOCHS in particular must stay at 20: flow_bc_train DEFAULTS to 2, and a 2-epoch run
# underfits so badly it reads as "no improvement" regardless of the data.
#
#   ./run_shield_control.sh          # after the --no-cbf collection has finished
set -euo pipefail

PY=${PY:-/workspace/openpi/.venv/bin/python}
BASE=${BASE:-/workspace/openpi_cache/openpi-assets/checkpoints/pi05_libero}
OUT=${OUT:-results_shielded}
NOSHIELD_DEMOS=$OUT/round0_demos_noshield
SHIELD_DEMOS=$OUT/round0_demos

# matched to run_shielded_distill.sh
EPOCHS=${EPOCHS:-20}
LR=${LR:-1e-4}
MINIBATCH=${MINIBATCH:-8}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
EVAL_INITS=${EVAL_INITS:-"35 36 37 38 39"}
SUITES="safelibero_spatial safelibero_object safelibero_goal"
LEVELS="I II"
TASKS="0 1 2 3"

[ -f "$NOSHIELD_DEMOS/manifest.csv" ] || { echo "no control demos at $NOSHIELD_DEMOS — run the --no-cbf collection first"; exit 1; }

N=$(tail -n +2 "$NOSHIELD_DEMOS/manifest.csv" | wc -l | tr -d ' ')
M=$(tail -n +2 "$SHIELD_DEMOS/manifest.csv"   | wc -l | tr -d ' ')
echo "control (no shield): $N demos     shielded pool: $M demos"
[ "$N" -lt 20 ] && echo "  WARNING: N=$N is small; both arms may floor and the test will be inconclusive."

# ── 1. subsample the shielded set down to N ──────────────────────────────────────────────
$PY - <<EOF
import csv, random, pathlib
N = $N
rows = list(csv.DictReader(open("$SHIELD_DEMOS/manifest.csv")))
random.seed(0)                                   # fixed seed => reproducible subsample
keep = random.sample(rows, min(N, len(rows)))
out = pathlib.Path("$OUT/round0_demos_matched"); out.mkdir(parents=True, exist_ok=True)
with open(out / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
print(f"  matched set: {len(keep)} demos")
EOF

# ── 2. train both arms, identical settings ───────────────────────────────────────────────
for ARM in noshield matched; do
    DEMOS=$OUT/round0_demos_$ARM
    CKPT=$OUT/${ARM}_ckpt
    [ -d "$CKPT" ] && { echo "  $ARM already trained — skipping"; continue; }
    echo ">>> training $ARM  ($(date '+%H:%M'))"
    $PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$BASE" \
        --round "$DEMOS" --out "$CKPT" --success-only \
        --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"
done

# ── 3. evaluate both SHIELD-OFF on the held-out inits ────────────────────────────────────
# Shield off is the whole point: the claim is that safety became intrinsic, so the test must run
# without the thing being distilled.
for ARM in noshield matched; do
    for s in $SUITES; do for l in $LEVELS; do for t in $TASKS; do
        od="$OUT/eval_${ARM}_nocbf/${s}_L${l}_t${t}"
        [ -d "$od" ] && continue                 # resumable
        $PY -m experiments.rl_rollout_local --config pi05_libero_cbf \
            --checkpoint "$OUT/${ARM}_ckpt" \
            --suite "$s" --level "$l" --task "$t" --episodes $EVAL_INITS --K 1 \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 --no-cbf \
            --out "$od" || echo "   [warn] $ARM $s L$l t$t failed — continuing"
    done; done; done
    echo "   ✔ eval $ARM done"
done

# ── 4. the answer ────────────────────────────────────────────────────────────────────────
echo
$PY -m experiments.sweep_eval_stats \
    "base (no distil)=$OUT/eval_base_nocbf" \
    "control: N=$N no shield=$OUT/eval_noshield_nocbf" \
    "matched: N=$N shielded=$OUT/eval_matched_nocbf" \
    "r1: N=$M shielded=$OUT/eval_r1_nocbf"
cat <<'EOF'

  READ IT LIKE THIS — compare CONTROL against MATCHED (same N, same filter, same training):
    control collision stays high    -> the shield is doing the work. Thesis claim holds.
    control collision falls as much -> the gain was success-filtering; the shield is incidental.
    control falls partway           -> both contribute; report the split. Most likely, and the
                                       most interesting: it quantifies what the shield adds
                                       BEYOND self-improvement, which nobody has measured.
EOF
