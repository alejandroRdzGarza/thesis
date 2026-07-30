# Supervisor Meeting Summary — Making π0.5 Internalize Collision-Safety
**Date:** 2026-07-29 · **Student:** Alex Rodriguez

## Research question
Can π0.5 (a flow-matching VLA) *internalize* collision-safety in SafeLIBERO so it rarely needs
the external CBF shield — measured as **CBF-activation rate** and **collision-rate-without-shield**
dropping while **task success holds**? Novelty: turning a reactive safety filter into a training
signal so the policy learns to make the filter redundant.

## Setup (constant across all experiments)
- **Task (single, for method discovery):** `safelibero_object / level II / task 0`, 4 initial states.
- **Model:** π0.5 (`pi05_libero`), fine-tuning **only the action-expert LoRA**, VLM backbone frozen
  (per your guidance). ~22M trainable / 3.4B.
- **Shield:** ellipsoid-barrier QP (CBF), filters the translational action each control step.
- **Headline metric:** no-CBF eval — success + robot-caused-collision with the shield removed.

---

## Approach 1 — Flow-SDE GRPO (on-policy RL)  →  ❌ structural failure

**Idea.** Convert π0.5's deterministic ODE action sampler into a marginal-preserving SDE
(Flow-GRPO, Liu et al. 2025) → tractable per-step log-prob → GRPO. Roll out K times under the
shield, reward `= w·success − w·collision − w·cbf_activation + w·progress`, group-relative
advantage, update the LoRA. "Shielded policy gradient": the shield is part of the environment, so
the policy is rewarded for needing it less.

| Exp | Change | Result | Read |
|-----|--------|--------|------|
| 001 | lr 1e-5, shielded-only | flat / no-op | lr too low, policy didn't move |
| 002 | lr 5e-5, w_cbf 1.5, shielded-only | success held ~0.85, **CBF-activation flat ~0.33, no-CBF collision stuck at 1.0** | shield masks collisions during rollout → collision penalty has no gradient; only the weak activation signal remains, dominated by success |
| 003 | lr 5e-5, **mix 50% unshielded** rollouts | **COLLAPSE** — success → 0 *even with shield on* by round 2 | real collisions now enter the reward, but the strong gradient diverged the LoRA + the collision penalty punishes goal-directed motion → policy learns the trivial "do nothing = zero collisions" hack |
| 004 | lr 2e-5, **shield-anneal** 0.85→0.40 | collapse fixed (success held) but **learned zero safety** (no-CBF collision flat ~0.94) | back in the "too gentle" regime |

**Conclusion (the key result of Approach 1).** The *only* setting that moved the policy's behavior
(003) collapsed it; every setting stable enough to preserve the task (002, 004) learned no
avoidance. **The learning signal and the destabilizing signal are the same knob.** Scalar-reward
flow-GRPO cannot separate "learn the spatial detour around the obstacle" from "erode goal-reaching,"
because the obstacle sits on the path to the goal and a single scalar episode reward gives **no
per-step spatial credit**. This is a structural limit, not an untuned hyperparameter — and it
becomes the *motivation* for Approach 2.

---

## Approach 2 — Shield-as-expert DAgger (imitation)  →  ✅ first positive signal

**Insight.** The CBF shield *already computes the correct safe action at every step* — a dense,
per-step supervised signal we were throwing away. So drop the scalar reward: roll out fully
shielded, then **behavior-clone** the LoRA (via π0.5's native flow-matching loss) to reproduce the
shield-corrected action. Structurally **cannot** reward-hack to inaction (the target action *does
the task safely*), and it supplies the per-step spatial credit the scalar reward lacked. DAgger =
collect under the *current* policy's distribution, retrain, repeat.
*(Note: an earlier DAgger attempt failed due to a CBF bug that disabled the shield during the
approach phase → labels ≈ base actions. That's fixed; the shield now genuinely corrects.)*

**Exp 005 — result (no-CBF eval across DAgger rounds):**

| Round | no-CBF success | no-CBF collision |
|-------|---------------|------------------|
| base π0.5 | ~0.88 | ~0.94 |
| **0 (1 BC round)** | **0.69** | **0.125** ← knee |
| 1 | 0.25 | 0.06 |
| 2–4 | ~0.06–0.25 | ~0.00 |

- **After one BC round, unshielded collision crashed 0.94 → 0.125** — the safety internalization
  all four RL runs failed to produce. Modest success cost (0.88 → 0.69).
- Over further rounds the policy **over-imitates the shield's caution** → collision → ~0 but success
  collapses. Clean monotonic safety/success tradeoff; **the knee (1–2 BC rounds) is the keeper.**

**Exp 005b — with the automated success filter (`results_dagger_succ`):** imitate only rollouts
that succeeded *and* stayed collision-free (read from the manifest, no manual labelling).

| BC rounds | no-CBF success | no-CBF collision |
|-----------|---------------|------------------|
| base | ~0.88 | ~0.94 |
| 1 | 0.81 | 0.125 |
| **2** | **0.69** | **0.00** ← keeper (`round2_ckpt`) |
| 3 | 0.56 | 0.06 |
| 4–6 | 0.50 → 0.12 | ~0.00 |

- **Confirmed result:** `round2_ckpt` = **0.69 success / 0.00 collision** unshielded (vs base
  0.88 / 0.94) — the policy *fully* internalized avoidance (0 collisions in 16 unshielded rollouts)
  at a 0.69/0.88 success retention.
- The filter **slows but doesn't stop** the erosion (still crashes by round 6) → **early-stopping at
  1–2 rounds is the operating point.** Next lever: soft advantage-weighted BC to flatten it further.

---

## Where we are
- **Best checkpoint:** `results_dagger_succ/round2_ckpt` (2 BC rounds, success-filtered):
  **0.69 success / 0.00 collision** no-CBF, vs base **~0.88 / ~0.94**. Fully internalized avoidance
  (0 collisions unshielded) at 78% of base success. Success filter **confirmed** to preserve
  success (vs unfiltered which crashed to ~0.06).
- **Caveats:** single task; 16-rollout eval (±~0.12 noise, needs ≥20-ep confirmation); erosion means
  early-stopping (1–2 rounds) is required.

## Next steps (proposed)
1. **Confirm the success filter** (task 0, ~3 rounds): does it hold success while dropping collision?
2. **Validate on a diverse sample** (~5 configs across suites/levels) — guards against the
   single-task result being an outlier; validation must be multi-task by definition.
3. **Rigorous eval:** ≥20 episodes, corrected metrics, 2×2 = {base, DAgger} × {plain, CBF}, reporting
   TSR / collision / CBF-activation / intervention magnitude.
4. **Scale to full benchmark** for final numbers (once the sample is consistent).

## Questions for you
- **Scope of generalization** the thesis needs: per-task across the benchmark, vs a *single policy*
  trained on many tasks (stronger claim)?
- Is **"scalar RL fails structurally → dense shield-imitation works"** a sufficient contribution, or
  push for full-benchmark generalization numbers?
- With CBF *on*, we expect **CBF-activation to drop** (the thesis point) but TSR is uncertain — is the
  reduced-shield-reliance metric the right headline, or do you want TSR-under-shield as primary?

## One-line takeaway
Scalar-reward RL structurally can't teach obstacle-avoidance here (mapped 4 failure modes);
**imitating the CBF shield as a dense per-step expert does** — driving unshielded collisions from
**0.94 → 0.00** while retaining **0.69/0.88 success** (2 BC rounds, success-filtered) — i.e. the
policy learned to be safe on its own, no shield needed. Erosion past round 2 → early-stopping.
