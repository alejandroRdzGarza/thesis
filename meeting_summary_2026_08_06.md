# Supervisor meeting — 2026-08-06

Covering 30 July – 6 August (since the 29 July meeting).

---

## Headline

The single-task distillation result from last meeting **replicated and scaled**. A full-grid run
(3 suites × 2 levels × 4 tasks) is on the pod now, roughly two-thirds through. Intermediate
numbers already show the core claim: after one round of behaviour cloning, the policy triggers
the CBF **half as often** and needs corrections **a quarter the size**, at **unchanged task
success**.

There was also a significant course correction mid-week: the classical scripted expert was
**dropped as the primary teacher** in favour of the shielded policy itself. Evidence below.

---

## 1. Exp 006 — classical MPC-CBF expert (30 Jul – 3 Aug)

Built in response to your diagnosis that Exp 005's shield-expert had no fixed anchor (its
`shielded_success` spiralled 0.78 → 0.28). Replaced it with a scripted pick-place controller on
privileged BDDL poses, plus a receding-horizon MPC keep-out and the reactive CBF as backstop.

Full grid, 96 rollouts: **object 94%**, goal 16%, spatial 0% — overall 36.5%.

Root cause of the spatial/goal 0%, found by live debugging: the akita bowl is **11 cm wide against
an 8 cm gripper**, so a top-down straddle grasp is physically impossible. Fixed with a **rim-pinch
grasp** (offset the grip point along the gripper's closing axis so one finger drops inside the
bowl). Spatial went **0% → 69% at 0% collision**.

---

## 2. Exp 007 — distilling the classical expert (31 Jul – 3 Aug)

Two bugs found and fixed, both instructive:

- **DAgger made things worse** (collision 44% → 75% across rounds). The expert was a *latched
  phase state machine*, not a Markov policy — when the student drove, the expert's hidden phase
  desynced from the observation and the same image got contradictory labels. Rewrote it as a
  **stateless reactive oracle** with phase inferred from observables. Validated identical to the
  old driver on-trajectory.
- **The DAgger label recorded the expert's *unshielded* nominal action**, which itself collides
  ~60% of the time. So DAgger was training on unsafe targets. Fixed by running the nominal through
  the same CBF the driver uses.

**The important finding:** the earlier "offline BC gives 0% success" result was **2-epoch
underfitting, not a barrier**. At 20 epochs with a success filter:

| policy (safelibero_spatial, 4 tasks × 15 held-out inits, n=60) | success | collision |
|---|---|---|
| base π0.5, no shield | 51.7% | 90.0% |
| **distilled, no shield** | **51.7%** | **15.0%** |
| base π0.5 + shield | 80.0% | 11.7% |
| distilled + shield | 38.3% | 0.0% |

Collision CIs non-overlapping. Safety internalised at identical task success, on held-out inits.
CBF activation 0.589 → 0.142.

---

## 3. Course correction — the teacher (4–5 Aug)

The premise for Exp 006 was that a scripted controller could give ~100% success / 100% collision
avoidance. **That premise did not hold**, and I now have the evidence for why:

- The classical expert tops out around **42% clean-demo yield** overall, 0% on several scenes.
- π0.5 + CBF on the AEGIS-matched baseline runs at **TSR 79.2 / CAR 87.5 ≈ 69% clean yield**.
- Scripted manipulation in an OSC end-effector action space with clutter is its own research
  problem. Three real bugs were found in one day of debugging (grasp side chosen toward the
  obstacle, grasp target aimed at the object's centre rather than its top, a latching contact
  detector that deadlocked episodes) and two scenes remain unsolved.

There is also a learning-theoretic reason to prefer the shielded policy: the classical expert emits
saturated P-control and QP deltas — a **different action distribution** from π0.5's flow-matched
chunks — so imitating it drags the policy off its pretrained prior. Shielded π0.5 actions are
π0.5's own actions minus a small safety projection: a much smaller delta to learn. This plausibly
explains why the classical distillation capped at ~52% success while base+shield reaches 79%.

**Also relevant:** the original reasons for abandoning π0.5+CBF were bugs that are now fixed — the
`_in_gc_range` gate that disabled the shield during the entire approach phase (so "corrected"
labels equalled base actions), and the 2-epoch underfit.

**Decision taken:** shielded π0.5 is the teacher; the classical expert is retained as a documented
**ablation** ("privileged scripted expert vs shielded-policy expert"), which is a genuine result
rather than wasted work.

---

## 4. Per-scene teacher work (4 Aug) — now an ablation

Before the pivot, built a per-scene teacher profile system: grasp side chosen by measured obstacle
clearance, grasp target derived from the object's true mesh geometry, planner keep-out relaxed
where it blocked the grasp corridor.

Concrete finding: the rim-grasp offset was hard-coded to +Y, and on **every** `safelibero_goal`
level-I scene the obstacle sits on the +Y side — so the teacher was told to reach 5 cm *toward* the
obstacle, the barrier refused, and all 16 episodes froze in APPROACH. Choosing the side by
clearance took those tasks from **0/12 to 6/12 at zero collisions**.

Also characterised `safelibero_spatial` t2/t3 (bowls wedged on a stove and in a cabinet) as an
**OSC_POSE action-space limitation**, not a tuning problem: a collision-free arm configuration
exists and IK finds it, but end-effector-only control cannot hold it — OSC resolves the null space
its own way and the arm drifts back into the obstacle during transport. Documented as future work.

---

## 5. Full-grid run (in progress)

24 scenes × 12 train inits per round, 2 BC rounds, then held-out eval of
{base, r1, r2} × {no-CBF, CBF} on inits 35–39 with pooled Wilson CIs.

| | round 0 (base π0.5) | round 1 (distilled) |
|---|---|---|
| rollouts | 288 | 288 |
| success | 202 (70.1%) | 200 (69.4%) |
| clean (success + collision-free) | 186 | 182 |
| **shield-free clean episodes** | **0** | **16** |
| **CBF activation rate** | **0.331** | **0.162** |
| **mean \|correction\|** | **0.335** | **0.081** |

Task capability flat (202 vs 200, well inside sampling error at n=288). Shield reliance halved,
correction magnitude quartered, and 16 episodes completed cleanly with **no intervention at all**
versus zero before.

**Caveat I want to be explicit about:** all of the above is measured *with the shield present*. The
claim is about the policy without it. The no-CBF eval arms are the actual test and have not run
yet. Low activation under shielding does not guarantee low collision unshielded, because the shield
also shapes the state distribution. Exp 005b's single-task precedent (0.94 → 0.125 unshielded
collision) is the reason to expect it transfers.

~31 h of compute remaining: round-2 BC (~14 h) then eval (~18 h).

---

## 6. A result that may be worth a paragraph

Exp 005b showed success eroding from round 1 (0.88 → 0.81) and crashing by round 6, which is why
early-stopping at 1–2 rounds became the operating point. **This run shows no erosion at round 1
at all.** The difference is data scale and diversity: 005b used ~32 demos from one task, this uses
186 across 24 scenes. If that holds, the erosion was substantially a **small-data artifact**, not
an intrinsic property of shield-as-expert distillation. One data point per condition so far — but
it's a hypothesis my own two runs speak to.

---

## Questions for you

1. **Teacher decision** — do you agree with dropping the scripted expert to a documented ablation?
   It reverses the direction from the 29 July meeting, so I want it explicit rather than assumed.
2. **Scope for the remaining time** — with ~31 h of compute left and the report due in September,
   is the priority (a) finish this run and write, (b) add a second seed for the headline arms, or
   (c) attempt the per-step correction-weighted BC variant?
3. **Publication** — is a workshop paper realistic here, and would you want to be involved in venue
   choice? Main-track submission would need real-robot or perception-based geometry, which is out
   of scope for the thesis timeline.
4. **The AEGIS comparison** — we exceed their reported CAR, but partly because ground-truth
   obstacle geometry removes their perception gap. I plan to state that caveat prominently rather
   than claim a clean win. Confirm that's the right call.
