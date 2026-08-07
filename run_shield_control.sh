#!/bin/bash
# run_shield_control.sh — END-TO-END control ablation for the shielded-distillation result.
# Collect -> size-match -> train both arms -> evaluate both -> print the answer. One command.
#
# WHY THIS EXISTS
# The headline result (shield-free collision 82.5% -> 17.5%) trains on rollouts filtered by success
# AND collision-free, collected WITH the CBF shield. Two mechanisms are bundled in that filter:
#   (a) the policy imitates the shield's avoidance corrections   <- the thesis claim
#   (b) plain success-filtering: training on your own good episodes improves a policy by itself,
#       and successful episodes correlate with not knocking things over   <- already known, and
#       would make the shield incidental
# Nothing in the six-arm eval distinguishes them. This does.
#
# DESIGN
# Two arms, an IDENTICAL number of demos, filtered identically, differing ONLY in whether the
# shield ran during collection:
#     noshield   N demos, shield OFF   <- control
#     matched    N demos, shield ON    <- comparison
# N is whatever the unshielded collection yields; the shielded pool is subsampled down to it. The
# 186-demo r1 set is NOT in this comparison — it is the headline arm, trained on everything, and
# appears in the final table only as context. Equality is between the two arms here. N must be
# large enough that both train to a level where a difference can show; a tiny N floors both and
# proves nothing, which is why the collection budget is generous.
#
#   ./run_shield_control.sh                 # everything, resumable
#   COLLECT_INITS="$(seq 0 19)" ./run_shield_control.sh    # half the collection budget
#
# Resumable at every stage: finished collection, trained checkpoints and completed eval scenes are
# all detected and skipped, so a crash or a disconnect costs only the step in flight.
set -euo pipefail

PY=${PY:-/workspace/openpi/.venv/bin/python}
BASE=${BASE:-/workspace/openpi_cache/openpi-assets/checkpoints/pi05_libero}
OUT=${OUT:-results_shielded}
SHIELD_DEMOS=${SHIELD_DEMOS:-$OUT/round0_demos}          # the existing 186-demo shielded pool
NOSHIELD_DEMOS=$OUT/round0_demos_noshield
MATCHED_DEMOS=$OUT/round0_demos_matched

# Collection budget. 35 inits x 24 scenes = 840 rollouts ~ 21 h at ~1.5 min each. Unshielded yield
# is expected ~15% (base is 58.3% TSR / 82.5% collision), so ~126 demos. Inits 35-49 stay held out.
COLLECT_INITS=${COLLECT_INITS:-$(seq 0 34)}

# Matched to run_shielded_distill.sh so both arms are comparable to r1 and to each other.
EPOCHS=${EPOCHS:-20}          # never 2 — see flow_bc_train --epochs help
LR=${LR:-1e-4}
MINIBATCH=${MINIBATCH:-8}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
EVAL_INITS=${EVAL_INITS:-"35 36 37 38 39"}
SUITES="safelibero_spatial safelibero_object safelibero_goal"
LEVELS="I II"
TASKS="0 1 2 3"

say() { echo "[$(date '+%H:%M:%S')] $*"; }

[ -f "$SHIELD_DEMOS/manifest.csv" ] || { say "no shielded pool at $SHIELD_DEMOS"; exit 1; }

# ── 1. collect UNSHIELDED demos, same success+clean filter ───────────────────────────────
if [ -f "$NOSHIELD_DEMOS/.done" ]; then
    say "collection already complete — skipping"
else
    say ">>> [1/5] collecting unshielded demos (this is the long pole, ~21 h)"
    $PY -m experiments.collect_shielded_demos \
        --checkpoint "$BASE" --no-cbf \
        --episodes $COLLECT_INITS \
        --out "$NOSHIELD_DEMOS"
    touch "$NOSHIELD_DEMOS/.done"
fi

N=$(tail -n +2 "$NOSHIELD_DEMOS/manifest.csv" | wc -l | tr -d ' ')
M=$(tail -n +2 "$SHIELD_DEMOS/manifest.csv"   | wc -l | tr -d ' ')
say "control yielded $N demos; shielded pool has $M"
if [ "$N" -lt 20 ]; then
    say "WARNING: N=$N is small. Both arms may floor, making the comparison inconclusive."
    say "         The low yield is itself reportable: safe demos are hard to get without a shield."
fi

# ── 2. subsample the shielded pool to N ──────────────────────────────────────────────────
say ">>> [2/5] size-matching the shielded set to N=$N"
$PY - <<EOF
import csv, random, pathlib
rows = list(csv.DictReader(open("$SHIELD_DEMOS/manifest.csv")))
random.seed(0)                                    # fixed seed => reproducible subsample
keep = random.sample(rows, min($N, len(rows)))
out = pathlib.Path("$MATCHED_DEMOS"); out.mkdir(parents=True, exist_ok=True)
with open(out / "manifest.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(keep)
print(f"  matched set: {len(keep)} demos")
EOF

# ── 3. train both arms, identical settings ───────────────────────────────────────────────
for ARM in noshield matched; do
    CKPT=$OUT/${ARM}_ckpt
    if [ -d "$CKPT" ]; then say "[3/5] $ARM already trained — skipping"; continue; fi
    say ">>> [3/5] training $ARM (N=$N, epochs=$EPOCHS)"
    $PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$BASE" \
        --round "$OUT/round0_demos_$ARM" --out "$CKPT" --success-only \
        --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"
done

# ── 4. evaluate both SHIELD-OFF on held-out inits ────────────────────────────────────────
# Shield off is the point: the claim is that safety became intrinsic, so the test must run without
# the thing being distilled.
for ARM in noshield matched; do
    say ">>> [4/5] evaluating $ARM (shield OFF, inits $EVAL_INITS)"
    for s in $SUITES; do for l in $LEVELS; do for t in $TASKS; do
        od="$OUT/eval_${ARM}_nocbf/${s}_L${l}_t${t}"
        # Test for the MANIFEST, not the directory. A scene that crashed part-way leaves the
        # directory behind, so a -d test would treat partial data as complete and silently fold
        # it into the results; the manifest is only written once the scene finishes.
        [ -f "$od/manifest.csv" ] && continue
        $PY -m experiments.rl_rollout_local --config pi05_libero_cbf \
            --checkpoint "$OUT/${ARM}_ckpt" \
            --suite "$s" --level "$l" --task "$t" --episodes $EVAL_INITS --K 1 \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 --no-cbf \
            --out "$od" || say "   [warn] $ARM $s L$l t$t failed — continuing"
    done; done; done
done

# ── 5. the answer ────────────────────────────────────────────────────────────────────────
say ">>> [5/5] results"
echo
$PY -m experiments.sweep_eval_stats \
    "base (no distil)=$OUT/eval_base_nocbf" \
    "CONTROL N=$N no shield=$OUT/eval_noshield_nocbf" \
    "MATCHED N=$N shielded=$OUT/eval_matched_nocbf" \
    "r1 N=$M shielded=$OUT/eval_r1_nocbf"
cat <<'EOF'

  READ IT LIKE THIS — compare CONTROL against MATCHED. Same N, same filter, same training; the
  only difference is whether the shield ran during collection.

    control collision stays high     -> the shield is doing the work. Thesis claim holds.
    control collision falls as much  -> the gain was success-filtering; the shield is incidental,
                                        and the result must be reframed as self-improvement.
    control falls partway            -> both contribute. Most likely, and the most interesting:
                                        it quantifies what the shield adds BEYOND self-improvement,
                                        which nobody has measured.

  Note the CBF activation proxy already fell 0.533 -> 0.265 between base and r1. Success-filtering
  alone does not predict that: a policy that merely finishes tasks more often would not need LESS
  shield correction. That is independent evidence for the first row, which this test confirms or
  overturns directly.
EOF
