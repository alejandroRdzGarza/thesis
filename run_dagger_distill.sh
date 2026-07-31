#!/usr/bin/env bash
# run_dagger_distill.sh — Exp 007 DAgger rounds: fix the covariate shift left by offline BC.
#
# BC-then-DAgger recipe. run_distill_round0.sh does the OFFLINE half (BC the classical MPC-CBF
# expert on its own clean demos → results_distill/round0_ckpt). Offline BC alone fails (round 0:
# 0% success / 44% collision) because the student visits states the expert never demonstrated.
# DAgger fixes exactly that: roll the STUDENT out UNSHIELDED so it visits its own (failing) state
# distribution, LABEL every state it touches with the fixed classical expert (the correct action
# there), aggregate with all prior data, and retrain from the base each round.
#
#   for N in 1..ROUNDS:
#     roll out round{N-1}_ckpt  --label-controller --no-cbf   → round{N}_demos   (student states, expert labels)
#     BC(base, round0_demos + round1_demos + … + round{N}_demos)                 → round{N}_ckpt   (NO --success-only)
#     no-CBF eval round{N}_ckpt                                                    → round{N}_eval_nocbf
#
# Retraining from the BASE each round (not warm-starting the LoRA) keeps this true dataset-aggregation
# DAgger: pi_N = BC(base, D_0..N). --success-only is OFF for DAgger — the whole point is to keep the
# expert's recovery labels at the student's BAD states (the classical expert is a FIXED controller,
# so its labels are always correct; the Exp 005 success-erosion came from a degrading shield-expert,
# which does not apply here).
#
# Resumable: a round whose _ckpt already exists is skipped. Run round 0 first:
#   CKPT=$CKPT ./run_distill_round0.sh
#   CKPT=$CKPT ./run_dagger_distill.sh 2>&1 | tee dagger.log
set -euo pipefail

CKPT=${CKPT:?set CKPT to the BASE pi05_libero checkpoint dir (each DAgger round trains from it)}
SUITE=${SUITE:-safelibero_spatial}
LEVEL=${LEVEL:-II}
ROUNDS=${ROUNDS:-3}
# Student-rollout collection (the DAgger state distribution): one task, several initial states.
ROLLOUT_TASK=${ROLLOUT_TASK:-0}
ROLLOUT_EPISODES=${ROLLOUT_EPISODES:-0 1 2 3 4 5 6 7}
K=${K:-2}                       # rollouts per initial state (student is stochastic → varied states)
# BC hyperparams (match round 0).
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-2}
MINIBATCH=${MINIBATCH:-8}
# No-CBF eval after each round.
EVAL_TASK=${EVAL_TASK:-0}
EVAL_EPISODES=${EVAL_EPISODES:-0 1 2 3 4 5 6 7}
HORIZON=${HORIZON:-300}
NUM_STEPS=${NUM_STEPS:-10}
NOISE=${NOISE:-0.5}             # student rollout exploration (>0 → visits more of its own distribution)
OUT=${OUT:-results_distill}
PY=${PY:-python}

export MUJOCO_GL=${MUJOCO_GL:-egl}
export PYTHONPATH=${PYTHONPATH:-.}

ROUND0_DEMOS="$OUT/round0_demos"
[ -d "$ROUND0_DEMOS" ] || { echo "ERROR: $ROUND0_DEMOS missing — run run_distill_round0.sh first"; exit 1; }

echo "==================================================================="
echo " DAgger distill  |  $SUITE / L$LEVEL  |  $ROUNDS round(s)"
echo "   student rollout: t$ROLLOUT_TASK ep[$ROLLOUT_EPISODES] K=$K noise=$NOISE (UNSHIELDED)"
echo "   each round: BC(base, round0 + all dagger demos) → no-CBF eval"
echo "==================================================================="

AGG_DIRS=("$ROUND0_DEMOS")     # aggregated dataset grows each round

for N in $(seq 1 "$ROUNDS"); do
    PREV=$((N - 1))
    if [ "$PREV" -eq 0 ]; then PREV_CKPT="$OUT/round0_ckpt"; else PREV_CKPT="$OUT/round${PREV}_ckpt"; fi
    DEMOS="$OUT/round${N}_demos"
    CKPT_OUT="$OUT/round${N}_ckpt"
    EVAL_DIR="$OUT/round${N}_eval_nocbf"

    [ -d "$PREV_CKPT" ] || { echo "ERROR: previous checkpoint $PREV_CKPT missing"; exit 1; }
    AGG_DIRS+=("$DEMOS")       # this round's demos join the aggregate (built below)

    echo; echo "########################  DAgger round $N  ########################"

    if [ -d "$CKPT_OUT" ]; then
        echo "  [round $N] $CKPT_OUT exists → skipping collect+train (resume)"
    else
        echo; echo "### [$N.1] roll out student ($PREV_CKPT) UNSHIELDED, classical expert LABELS each state ###"
        if [ -d "$DEMOS" ]; then
            echo "  [round $N] $DEMOS exists → reusing collected rollouts"
        else
            $PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$PREV_CKPT" \
                --suite "$SUITE" --level "$LEVEL" --task "$ROLLOUT_TASK" --episodes $ROLLOUT_EPISODES \
                --K "$K" --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level "$NOISE" \
                --no-cbf --label-controller --out "$DEMOS"
        fi

        echo; echo "### [$N.2] BC(base, aggregated demos ${AGG_DIRS[*]}) — no --success-only ###"
        $PY -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint "$CKPT" \
            --round "${AGG_DIRS[@]}" --out "$CKPT_OUT" \
            --lr "$LR" --epochs "$EPOCHS" --minibatch "$MINIBATCH"
    fi

    echo; echo "### [$N.3] headline eval: NO CBF shield — did the student internalize the expert's safety? ###"
    if [ -d "$EVAL_DIR" ]; then
        echo "  [round $N] $EVAL_DIR exists → skipping eval"
    else
        $PY -m experiments.rl_rollout_local --config pi05_libero_cbf --checkpoint "$CKPT_OUT" \
            --suite "$SUITE" --level "$LEVEL" --task "$EVAL_TASK" --episodes $EVAL_EPISODES --K 2 \
            --horizon "$HORIZON" --num-steps "$NUM_STEPS" --noise-level 0 --no-cbf --out "$EVAL_DIR"
    fi
    echo "  round $N no-CBF eval → $EVAL_DIR/round_summary.json"
done

echo; echo "==================================================================="
echo " DAgger done. Per-round no-CBF summaries:"
for N in $(seq 1 "$ROUNDS"); do
    echo "   round $N: $OUT/round${N}_eval_nocbf/round_summary.json"
done
echo "==================================================================="
