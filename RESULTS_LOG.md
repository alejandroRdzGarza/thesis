# Results Log — π0.5 + CBF Safety (thesis)

Single, hand-maintained journal of every experiment we run, newest first. Each entry records
the **question**, the **exact config**, the **numbers**, an honest **read**, and the **next
step it motivates**. This is the source for the thesis write-up — auto-generated reports under
`figures/` are disposable; this file is not.

**Research question.** Can π0.5 (a flow-matching VLA) *internalize* collision-safety in
SafeLIBERO so it rarely needs the external CBF shield — measured as `cbf_activation_rate`
dropping across RL rounds while collision-rate-without-shield drops and task success holds?

---

## Exp 004 — Shield-anneal curriculum + lower lr (`results_grpo_v4`)
**Date:** 2026-07-28   ·   **Status:** 🟡 implemented, queued to run

**Hypothesis.** Exp 003 collapsed because the strong mixed-reward gradient at lr 5e-5 diverged
the LoRA and the collision penalty flooded a fresh policy (94% unshielded collisions) into the
inaction attractor. Introducing the collision gradient *gently* (curriculum) with *smaller
steps* (lower lr) should let the policy learn avoidance without losing task competence.

**Change (vs Exp 003).** (1) **lr 5e-5 → 2e-5** (Exp 003 diverged in 2 rounds). (2) **Shield-
anneal curriculum:** `shield_prob` starts 0.85 (round 0, mostly shielded — keeps task skill) and
ramps linearly to 0.40 (final round), so unshielded exposure grows as the policy improves.
`SHIELD_PROB_START/END` in `run_grpo_training.sh`. Everything else = Exp 003.

**What to watch:** `shielded_success_rate` must **stay up** (~0.9) this time — if it holds while
`unshielded_collision_rate` declines across rounds, we've threaded the needle. If success still
craters, the RL-from-scalar-reward approach may need a rethink (e.g. shield-as-DAgger-expert, or
per-step credit) rather than more knob-tuning.

---

## Exp 003 — Mixed shielded/unshielded rollouts (`results_grpo_v3`)
**Date:** 2026-07-28   ·   **Status:** ❌ policy collapse (reward-hacked to inaction)

**Question.** Does mixing unshielded rollouts (so real collisions enter the reward) give a direct
safety gradient that drops the unshielded collision rate?

**Config.** = Exp 002 + `shield_prob=0.5` (half each group's rollouts unshielded), lr 5e-5,
w_cbf 1.5, eval 8 states. Ran on RunPod RTX 4090. 6 rounds.

**Results.**

| Round | shielded success | overall success | unshielded collision | no-CBF collision |
|---|---|---|---|---|
| 0 | 0.94 | 0.91 | 1.00 | 0.94 |
| 1 | 0.94 | 0.81 | 0.94 | 0.88 |
| 2 | **0.00** | 0.00 | 0.63 | 0.00 |
| 3 | 0.00 | 0.00 | 0.00 | 0.00 |
| 4 | 0.00 | 0.00 | 0.00 | 0.06 |
| 5 | 0.00 | 0.00 | 0.06 | 0.00 |

**Read (honest).** ❌ **Collapse.** Collisions → 0, but **success → 0 too, even with the shield
on**. The policy didn't learn avoidance — it learned to *do nothing* (the trivial zero-collision
optimum). The mechanism worked (16/16 shielded/unshielded split; unshielded collisions did enter
the reward), but the outcome is a reward-hack.

**Root cause (two, collapse pattern disambiguates).** (1) **Divergence:** success died even in the
*shielded* condition by round 2 — the shield can't cause that, so the LoRA weights diverged.
Exp 002 was stable at the same lr because its gradient was flat; Exp 003's mixed reward is a much
stronger, higher-variance signal, so lr 5e-5 overshot. (2) **Collision penalty punishes progress:**
the goal is near the obstacle, so ~all forward motion collides early; penalizing it suppresses
goal-directed actions → retreat to inaction. Brackets the failure with Exp 002: too weak → nothing;
too strong/sudden → collapse.

**Next step → Exp 004.** Lower lr (smaller steps) + shield-anneal curriculum (introduce the
collision gradient gently, preserving task competence).

---

## Exp 003 setup notes (superseded by results above)
**Date:** 2026-07-28

**Hypothesis.** If the shield masking is the root cause (Exp 002), then running a *fraction* of
each group's rollouts **unshielded** — so real collisions enter the reward — will give GRPO a
direct safety gradient, driving down the unshielded collision rate and CBF activation while
success holds.

**Change (vs Exp 002).** New `shield_prob` knob in the rollout harness (`rl_rollout.py`
`shield_schedule` / `collect_group`): with `shield_prob=0.5`, half of each group's K rollouts run
with the shield and half unshielded. Unshielded collisions now fire `w_direct_collision`, so the
within-group reward spans **unshielded-safe-success ≈ 1.8 > shielded-success ≈ 1.31 >
unshielded-collide-success ≈ 0.8** — a Δ≈1.0 gradient toward internalized avoidance (vs the flat
~0.49 CBF-only signal before). Per-round summary now reports `unshielded_collision_rate` and
`shielded_cbf_penalty` separately. Eval strengthened to 8 initial states (was 4) to cut the ±0.12
noise. Config otherwise = Exp 002 (lr 5e-5, w_cbf 1.5, cps noise 0.7, LoRA action-head).

**What to watch:** `unshielded_collision_rate` (training) and no-CBF eval collision should trend
**down**; `shielded_cbf_penalty` should move toward 0; success should hold ~0.9.

---

## Exp 002 — Flow-SDE GRPO, tuning iteration 1 (`results_grpo_v2`)
**Date:** 2026-07-28   ·   **Status:** ❌ negative on the core claim (useful ablation)

**Question.** Does stronger optimization (lr 5e-5) + a louder safety signal (w_cbf 1.5) make the
shielded-policy-gradient setup reduce CBF reliance, after Exp 001 came back flat?

**Config.**
- Task: `safelibero_object / level II / task 0`, episodes [0,1,2,3], K=8 → 32 shielded rollouts/round.
- Method: flow-SDE GRPO (cps sampler, noise 0.7, 10 denoise steps), LoRA action-head only (backbone frozen, bf16), clip 0.2.
- **lr 5e-5**, reward `w_success 1.5 / w_direct_collision 1.0 / w_cbf_rate 1.5 / w_progress 0.3`.
- 6 rounds. Rounds 0–1 on RunPod A40; rounds 2–5 on UCL smew-l (resumed via LoRA-ckpt transfer + `base.txt` repoint). horizon 300, replan 5.
- Eval: no-CBF, deterministic-ish (noise 0), K=4 → **16 rollouts** (4 initial states).

**Results.**

| Round | no-CBF success | no-CBF collision ↓ | with-CBF success | **CBF activation** ↓ (≈ −pen/1.5) |
|---|---|---|---|---|
| 0 | 0.81 | 0.81 | — | — |
| 1 | 1.00 | 0.625 | — | — |
| 2 | 1.00 | **0.50** | 0.97 | 0.33 |
| 3 | 1.00 | 0.75 | 0.91 | 0.31 |
| 4 | 0.94 | 0.69 | 0.91 | 0.34 |
| 5 | 0.75 | **0.94** | 0.94 | 0.33 |

**Read (honest).**
- ✅ **Task performance held** — with-CBF success ~0.91–0.97, with-CBF collision 0.0 every round. No collapse (unlike lr 2e-4 in Exp 001 notes). The shielded-policy-gradient loop is *stable*.
- ⚠️ **Early signal, didn't hold** — no-CBF collision fell 0.81→0.625→0.50 (rounds 0–2), then bounced back to 0.94 by round 5, with success regressing 1.0→0.75.
- ❌ **The headline metric never moved** — CBF activation rate is dead flat at ~0.33 across all rounds. The core thesis claim (policy learns to need the shield less) did **not** happen.
- **Caveat:** the eval is underpowered — 4 initial states × K=4 = 16 rollouts, SE on a ~0.5 rate ≈ ±0.12. The round-to-round bouncing is within noise; the flat 0.33 is the only clearly-not-noise signal.

**Root cause.** **The shield masks the collision signal.** Every training rollout is shielded, so
the executed trajectory never collides → the `w_direct_collision` term is always 0 in training
data → it contributes zero gradient. The *only* safety teacher left is the CBF-activation
penalty (~−0.49), which is weak and dominated by the success term in the group-relative
advantage. So the policy optimizes for success, ignores the shield-usage nudge, and never
internalizes obstacle avoidance — exactly what the flat 0.33 and the high unshielded collision
rate show.

**Thesis framing.** Valid negative/ablation: *"shielded-only on-policy rollouts are insufficient
to internalize safety, because the shield removes the very collision events the policy must learn
from — motivating a mixed shielded/unshielded collection scheme."*

**Next step → Exp 003.** Mix unshielded rollouts into each group so real collisions enter the
reward (live `w_direct_collision` gradient); densify the shield signal (correction magnitude, not
just binary activation); strengthen the eval (≥8 initial states).

---

## Exp 001 — Flow-SDE GRPO, first run (`results_grpo`)
**Date:** ~2026-07-26   ·   **Status:** ❌ flat (motivated Exp 002)

**Question.** Does the shielded-policy-gradient flow-SDE GRPO loop reduce CBF reliance at all?

**Config.** Same task/method as Exp 002 but **lr 1e-5**, **w_cbf_rate 0.5**, eval K=1. 4 rounds.

**Results (summary).** Completely flat: `mean_cbf_penalty ≈ −0.19` every round (noise), no-CBF
collision stuck at 1.0, with-CBF success held ~0.85 (no collapse).

**Read.** lr 1e-5 was effectively a no-op (|g|≈3e-4, policy didn't move); w_cbf 0.5 was drowned
out by w_success 1.5. Motivated the Exp 002 tuning (lr 5e-5, w_cbf 1.5, eval K=4).

---

## Baseline (SafeLIBERO benchmark, base π0.5)
**Status:** ⚠️ existing `results_full_benchmark_*` / `results_baseline_*` predate the metric-fix
commit `b30ae85` (2026-07-22, goal-resolution + check_ontop audit) → their TSR is inflated for
object/goal suites and **not directly comparable** to the corrected-metric RL evals. A spot-check
(or the `run_baseline_sweep_grpo.sh` re-run) is needed before using baseline numbers in the
write-up. They carry the right columns though (plain vs cbf: TSR, collision_rate, CAR).
