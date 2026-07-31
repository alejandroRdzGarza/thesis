# Results Log — π0.5 + CBF Safety (thesis)

Single, hand-maintained journal of every experiment we run, newest first. Each entry records
the **question**, the **exact config**, the **numbers**, an honest **read**, and the **next
step it motivates**. This is the source for the thesis write-up — auto-generated reports under
`figures/` are disposable; this file is not.

**Research question.** Can π0.5 (a flow-matching VLA) *internalize* collision-safety in
SafeLIBERO so it rarely needs the external CBF shield — measured as `cbf_activation_rate`
dropping across RL rounds while collision-rate-without-shield drops and task success holds?

---

## Exp 005 — Shield-as-expert DAgger (imitation) (`results_dagger`)
**Date:** 2026-07-29   ·   **Status:** 🟡 implemented + CPU-tested, ready to run (dry-run first)

**The pivot.** Scalar-reward RL (001–004) can't separate learning-to-avoid from destroying the
task. So drop the scalar reward: the CBF shield already computes the *correct safe action at
every step*, so train the LoRA to **imitate** it (dense per-step supervision). Structurally
can't reward-hack to inaction — the target action does the task safely — and gives the spatial
credit the scalar reward lacked. The earlier "DAgger R0 fail" was a CBF bug (`_in_gc_range`
disabled the shield during approach → labels ≈ base actions → no signal); that's fixed, and the
GRPO runs confirm the shield now genuinely corrects (unshielded collide ~100%, shielded don't).

**Pipeline (per round).** (1) roll out current policy **fully shielded** (`shield_prob=1`),
recording each executed shield-corrected action (`QueryTrace.shielded_actions`, new). (2) BC the
LoRA with the model's native flow-matching loss (`Pi0.compute_loss`) to reproduce those actions
(`flow_bc.py` / `flow_bc_train.py`). (3) roll out no-CBF → measure learned safety. Repeat (DAgger:
data is collected under the *current* policy's state distribution). `run_dagger_{round,training}.sh`.

**Impl notes.** Shielded actions are env-space → normalized to model space by running them through
the policy's own input transform (LiberoInputs passes `actions` → Normalize), so targets match
π0.5 training exactly; padded 7→action_dim with zeros. CPU-tested: trace roundtrip (incl.
variable-length + back-compat), chunk assembly/tail-pad, patch applies to 15a9616. **GPU-pending:**
run `flow_bc_train --dry-run` on the pod first (prints Observation/actions shapes + one
compute_loss) to validate the transform path before a full run.

**What to watch:** no-CBF collision should now actually **fall** across rounds while success holds
— the outcome all four RL runs failed to produce.

---

## Exp 006 — Classical MPC-CBF expert (`sweep_classical`)
**Date:** 2026-07-30   ·   **Status:** ✅ strong on `object`; needs a place-on-surface mode for spatial/goal

**Why:** Exp 005 "expert" was the co-degrading student+shield (shielded_success spiralled
0.78→0.28) — no fixed anchor (supervisor's diagnosis). Replaced with a FIXED classical expert:
scripted pick-place state machine (privileged BDDL object+goal poses) + **MPC-CBF** anticipatory
obstacle avoidance (receding-horizon QP, curves around; reactive CBF alone deadlocked at the
boundary) + the reactive CBF as the hard-safety backstop. No VLA → fast, doesn't degrade.

**Full sweep (3 suites × 2 levels × 4 tasks × 4 episodes = 96 rollouts, classical only):**

| suite | success | collision | n |
|---|---|---|---|
| **safelibero_object** | **94%** | 3% | 32 |
| safelibero_goal | 16% | 9% | 32 |
| safelibero_spatial | 0% | 6% | 32 |
| overall | 36.5% | 6.2% | 96 |

**Exp 006c — trajectory-quality pass (multi-obstacle MPC).** Before distilling, verified the
*successful* trajectories are actually good (the Exp 005 mistake was trusting metrics over path
quality). Found: single-obstacle MPC lurched (reactive CBF fought it, **activation ~0.5**) and
the placement dropped the bowl. Fixes: (1) **multi-obstacle MPC** — feed all nearby scene objects
into the keep-out QP (not just the one detected obstacle) → smooth anticipatory paths through
clutter; (2) careful xy-locked descent + rim-pinch grasp for bowls; (3) place-on-surface set-down.
Result on spatial_t1: **CBF activation 0.52 → 0.17, jerk 0.021 → 0.012, 0 collision**, and it now
*lowers* the object instead of dropping it (user-confirmed on video). Residual: motion is a bit
slow (env's compliant OSC at the elevated workspace — env-limited, cosmetic for BC).

**Read.** The expert **generalizes across the object suite** (4 objects × 2 levels, ~94%, ~0
collision) — a fixed high-quality safe teacher, exactly what the pivot needed. spatial/goal fail
because they're a different task type ("place bowl ON a surface", elevated starts): the carry-high-
then-drop placement is wrong for setting a bowl down, and bowl grasp needs different heights. Not
an approach flaw — a per-suite controller mode (place-on-surface + bowl grasp).

**Next:** distill the VLA on the `object`-suite classical demos (BC, `--success-only`) → the core
sanity check: does the VLA overfit → internalize safety from a *good, fixed* expert (unlike the
degrading Exp 005 demos)? Then decide whether to add the place-on-surface mode for full-benchmark.

---

## Exp 006b — Rim-pinch grasp + MPC-CBF (classical expert, spatial/goal)
**Date:** 2026-07-30   ·   **Status:** ✅ spatial solved (69%/0 collision); goal grasp works, placement collides

**Root cause of the earlier 0% (found by live local debugging):** the akita bowl is **11 cm wide
> the 8 cm gripper**, so a top-down straddle grasp is *physically impossible* (EE stalls ~5 cm
above center). Probed the gripper: fingers close along **world Y**. Fix = **rim-pinch grasp**:
offset the grip point +5 cm along Y so one finger drops inside the bowl, one outside → pinch the
rim on close. Detect bowls by name → `grasp_mode=rim` (cartons stay top-down); track the full 3-D
grasp offset so placement drives the *object* (not the offset EE) to the goal. Also added careful
(xy-locked, speed-capped) descent to kill OSC coupling drift, and place-on-surface (`on`) mode.

**Sweep (LII, tasks 0–3, 4 episodes):**

| suite | success | collision |
|---|---|---|
| **safelibero_spatial** | **69%** | **0%** |
| safelibero_goal | 44% | 62% |

Spatial per-task: t0 50, t1 100, t2 100, t3 25. Object suite unaffected (4/4).

**Read.** Spatial (the target suite) went **0% → 69% at 0% collision** — a clean, safe, fixed
expert; with `--success-only` that's plenty of optimal-safe demos to distill. Remaining:
(1) spatial t0/t3 lag (hardcoded +Y rim offset / approach doesn't adapt to some layouts);
(2) goal collides in placement (the `on` set-down drives 15 cm below the goal → crashes the bowl/
arm into the plate). Both fixable. Next: polish those OR proceed to BC distillation on spatial.

---

## Exp 005b — DAgger + success filter (`results_dagger_succ`)
**Date:** 2026-07-30   ·   **Status:** ✅ confirmed positive (single task); early-stop required

**Change vs Exp 005:** `--success-only` (imitate only succeeded + collision-free shielded rollouts).

| BC rounds | no-CBF success | no-CBF collision |
|-----------|---------------|------------------|
| base | ~0.88 | ~0.94 |
| 1 | 0.81 | 0.125 |
| **2** | **0.69** | **0.00** ← keeper (`round2_ckpt`) |
| 3 | 0.56 | 0.06 |
| 4 | 0.50 | 0.125 |
| 5 | 0.44 | 0.00 |
| 6 | 0.125 | 0.00 |

**Read.** ✅ The filter preserves success far better than unfiltered (round 4: 0.44 vs 0.25) and
`round2_ckpt` = **0.69 / 0.00** unshielded (base 0.88 / 0.94) — full safety internalization at 78%
of base success. ⚠️ Filter *slows but doesn't stop* the over-caution erosion (crashes by round 6),
so **early-stopping at 1–2 rounds is the operating point.** Next: soft advantage-weighted BC to
flatten the erosion; then a diverse multi-task sample (single-task outlier risk) + ≥20-ep eval.

---

## Exp 004 — Shield-anneal curriculum + lower lr (`results_grpo_v4`)
**Date:** 2026-07-28   ·   **Status:** ❌ flat on safety (stable but learns nothing)

**Change (vs Exp 003).** lr 5e-5 → 2e-5; shield-anneal `shield_prob` 0.85 → 0.40 over 6 rounds.

**Results.**

| Round | shield_prob | shielded success | unshielded collision | no-CBF collision | shielded CBF-pen |
|---|---|---|---|---|---|
| 0 | 0.85 | 0.75 | 1.00 | 0.94 | −0.63 |
| 1 | 0.76 | 0.88 | 1.00 | 0.94 | −0.55 |
| 2 | 0.67 | 0.80 | 1.00 | 1.00 | −0.56 |
| 3 | 0.58 | 0.70 | 1.00 | 0.94 | −0.60 |
| 4 | 0.49 | 0.69 | 0.88 | 0.94 | −0.62 |
| 5 | 0.40 | 0.83 | 1.00 | 0.88 | −0.62 |

**Read.** ✅ Collapse fixed — `shielded_success` held ~0.7–0.88, no divergence. ❌ Zero safety
learning — `unshielded_collision_rate` pinned ~1.0, no-CBF collision flat ~0.94, CBF activation
flat ~0.4. Stable but flat.

**Decisive cross-run conclusion (001–004).** The only setting that moved the policy's behavior
(003, lr 5e-5 + mixed) collapsed it to inaction; every setting stable enough to preserve the task
(002, 004) learns no avoidance. **The learning signal and the destabilizing signal are the same
knob** — scalar-reward flow-GRPO cannot separate "learn the spatial detour around the obstacle"
from "erode goal-reaching," because the obstacle lies on the path to the goal and a single scalar
episode reward gives no per-step spatial credit. This is a **structural limit of scalar-reward RL
here**, not an untuned hyperparameter — a legitimate negative-result contribution.

**Next → Exp 005 (pivot).** Switch to a **dense per-step signal**: shield-as-expert imitation —
the CBF already computes the safe action at every step, so train the LoRA to imitate that
corrected action (DAgger-style). Structurally cannot reward-hack to inaction (the target action
*does the task safely*) and gives per-step spatial supervision the scalar reward lacked.
(NB: an earlier DAgger attempt "R0 failed" — check what broke before reusing that framing.)

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
