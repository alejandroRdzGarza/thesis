# Verified paper findings and rewrite-risk register

Two documents in one, because they answer the same question from opposite ends.
**Part A** records what was found in each paper actually read in full, so it never has to be
re-derived. **Part B** is the risk register: which thesis claims are load-bearing, what finding
would break each, and which unread paper could contain it.

The purpose of Part B is to make a late-breaking rewrite impossible rather than unlikely. A rewrite
only happens if a *load-bearing* claim breaks. So every load-bearing claim is listed with its
failure condition and the paper that could trigger it, and those papers are read **before** the
prose that depends on them.

Status legend: **VERIFIED** = read in the PDF directly, quotations checked.
**CSV** = from `paper_insights.csv` / `paper_relevance_matrix.csv` only, not independently checked.

---

# Part A — Findings from papers read in full

## 1. VLSA / AEGIS — VERIFIED 2026-08-18
*The primary baseline. SafeLIBERO and the collision metric come from here.*

- **CAR is defined as "the percentage of strictly collision-free episodes"** — no numeric
  threshold. The thesis's L1 `sum|Δp| > 1 mm` test is therefore *this work's own
  operationalisation*, not theirs. Method §3.1 now says so.
- **ETS is "the average episode length (including timeouts)"** — a different quantity from the
  thesis's first-success-step-over-successes. Do not present as comparable.
- **Their shield is end-effector-only**: "because our current formulation solely constrains the
  end-effector, unconstrained kinematic links may occasionally collide." The thesis additionally
  constrains links 3–7 and the hand, so the arm-link constraint is a genuine extension, and §4.5
  shows arm links are where residual collisions concentrate.
- Their CBF uses `α(h) = 10h`, `k = 10` — identical to this work's `K_CBF = 10`.
- Table I: CAR 77.9% / 68.9% (translational / full action space) vs π0.5's 18.7% / 17.1%;
  TSR 68.1% / 67.5% vs 50.9% / 57.8%. QP costs 0.356 ms/step, ~1.86% of the cycle.
- **They independently report "Safety-Induced Distribution Shift"**: enforcement can drive the
  policy out of distribution, where it "may behave erratically and fail to recover." This is the
  same effect measured here as the shield-stacking success loss, and it is the strongest available
  argument for internalisation.
- The earlier "published reference 87.5%" figure was fabricated — it came from their own
  `collect_shielded_demos.py` docstring, not the paper. Real 3-suite figure is 71.9%.

## 2. ROAD-VLA — VERIFIED 2026-08-19
*Closest conceptual comparator. Nearly preempted the novelty claim.*

- **Teacher is a logit-perturbed copy of the student itself** (their Eq. 10): action-token logits
  nudged by a calibrated, agreement-gated advantage, then forward-KL token-level distillation along
  the student's own on-policy rollout. **No foreign controller appears anywhere in the paper.**
- Their policy-improvement bound holds *only while the teacher stays proximal to the current policy*.
- **The privileged-teacher rejection is TEXTUAL-ONLY.** Their §4.2.1 is titled "Text-guided
  $\mathcal{I}$" and all three rejected variants are language templates.
- Table 2 (VR-UnseenTable OOD, 3 seeds): full 91.5±1.2, PPO 87.2±3.6, **MCTS PI 75.8±2.0** (the
  headline text-only number, 11.4 pts *below* PPO), RelSpatial PI and Plan+RelSpatial PI both
  4.68±0.0. Table 1 average: ID 85→88, OOD 69→73, degradation 16.3→14.6.
- Their explanation supports the thesis's language-channel negative: "post-training VLA weakens the
  general reasoning of the LLM backbone" and "a modality gap prevents the discrete text from
  providing the precise grounding required for continuous control."
- **Positioning:** their proximity principle *predicts* the planner-distillation failure but never
  tests a foreign low-level controller. That experiment is this thesis's contribution — a gap they
  anticipate, not a disagreement. They *are* prior art for "distil a corrective signal into a VLA",
  so §1.3 must claim the empirical boundary, not the idea.
- CSV correction: the headline text-only number is 75.8%, not 4.68%.

## 3. SAFE-GIL (ICRA 2025) — VERIFIED 2026-08-19
***This one changed a thesis claim.***

- HJ reachability computes the disturbance `d*(x)` that maximally steers the system toward the
  failure set (their Eq. 4); it is injected into the expert's action during collection (Eq. 5,
  Algorithm 1). The **fixed** expert is dragged into safety-critical states and its recovery
  behaviour recorded. Plain BC on that data.
- **Their expert is an MPC (navigation) or PID (taxiing) — foreign and privileged, not the
  learner — and safety transfers fine.** This falsified the thesis's previous claim that "what
  transfers is correction of the policy's own behaviour, not instruction from outside it."
  Restated across Conclusion, Discussion §5.1 and Results §4.4 as: **the operative variable is the
  state distribution the supervision covers.** The planner failed on coverage, not on foreignness.
- Their ablation proves targeting is the active ingredient: Gaussian noise BC, uniform noise BC and
  DART **all fail** to improve safety. Only the reachability-computed disturbance works.
- Table I is a safety/performance **tradeoff**: collision 0.19→0.11→0.07 as d_max 0.2→0.4→0.6, but
  cost 2975→3089→3156, "as the training data shifts towards more unsafe states, the task
  performance of the learned policy decreases." **The thesis improved safety AND success together**
  — a better outcome than the closest prior method, and worth stating.
- Their Fig. 5 makes the thesis's design-time argument independently: "While BC+Filter is able to
  maintain system safety, the overall performance of the system is still limited by the underlying
  policy… the agent isn't able to return to the centerline, whereas SAFE-GIL recovers much earlier.
  This highlights the need for design-time methods."
- **Divergence to be honest about:** they find SAFE-GIL+Filter complementary "without much
  degradation in task performance." The thesis measured a success *loss* from stacking.
- Scale caveat: MLP policies, 2D unicycle / X-Plane taxiing / quadrotor. Not a VLA. 10 seeds (nav),
  5 seeds (taxiing).

## 4. IntervenGen (IROS 2024) — VERIFIED 2026-08-19
*Second independent confirmation of the state-coverage account.*

- Rolls out the **current policy** in new configurations so recorded mistakes are that policy's
  genuine failures ("the generated mistake will reflect the genuine behavior of the policy in the
  new configuration"), detects each on contact, then transforms a **human** recovery segment onto
  the state reached via SE(3) transform + open-loop replay.
- **Recovery actions are foreign (human); only the states are the policy's own.** Together with
  SAFE-GIL — different teacher, different route to the failure states — two methods now locate the
  active ingredient in state coverage rather than teacher identity. The corrected claim no longer
  rests on one paper.
- Generated demos kept only if they complete the task; they also filter segments to "prevent the
  imitation of mistakes."
- Results: up to **39× robustness from 10 human interventions**; a policy trained on data
  synthesised from 10 interventions beats one trained on 100 full interventions by 24%, at 12% of
  the collection time. 4 sim + 1 physical task, BC-RNN / robomimic, 50 trials.
- **Scope limits that favour this thesis:** their Assumptions 1–3 require a delta-pose action
  space, a **known sequence of object-centric subtasks**, and object poses observable at each
  subtask boundary. The shield needs none of these. Their distribution shift originates in
  object-pose estimation error, so their claim is about robustness to perception error rather than
  collision safety.

## 5. Guided by Guardrails — VERIFIED 2026-08-19
***The control that makes this thesis's result non-obvious.***

- **CBF Filter condition: shield active throughout RL training, then removed → safety is NOT
  retained.** The agent collides once the filter is disabled, and their critic visualisation shows
  **high positive value inside the obstacle** — the hazard was never represented. Superficially the
  same protocol as this thesis, opposite outcome.
- **Why the thesis differs:** their agent's actions are *overridden* while the objective stays a
  scalar reward, so no gradient ever points toward the corrected action. Here the corrected action
  *is* the imitation target. Their negative therefore disposes of the deflationary reading of
  Chapter 4 — that any policy trained under a shield would end up safe — and shows the imitation
  objective is doing the work.
- **CBF Decay** (executed action = β·a_cbf + (1−β)·a_rl, β decayed to 0) **does** internalise both
  goal-reaching and obstacle avoidance, and behaviour survives removal; it even learns recovery.
  A second, non-imitation route to the same destination.
- Scalar-reward findings support §4.7's RL negative: SAC alone attains near-zero return; CBF Reward
  likewise fails; they conclude "scalar rewards might not be sufficient."
- **Their future work proposes this thesis:** "we could provide a safety based privileged
  information (e.g., CBF Interventions) during simulation in the actor-critic framework, just like
  children are aware of the training wheels during learning."
- Scale caveat: unicycle model, 1.5 m × 1.5 m arena, SAC, 10 seeds, sim2real to a four-wheel robot.
  Not a VLA.

---

## 6. IL_with_CBF (CDC 2022, Cosner, Yue, Ames) — VERIFIED 2026-08-19
*Closes risk C2. The thesis's "no formal guarantee" claim survives and can now name which
conditions fail.*

Their Theorem 2 does transfer safety from a CBF-based expert to a learned end-to-end controller,
but only as **ISSf with respect to an EXPANDED safe set** `C_δ`, where δ grows with data sparsity
`r3` and learning error `M_e` — not the original set. It requires all of:

- the expert is a **TR-OP controller** (their Def. 5): a *robust* CBF-QP with tunable margins
  φ, a, b for matched disturbances and state uncertainty;
- **control-affine known dynamics** `ẋ = f(x) + g(x)u`, locally Lipschitz;
- **CBF-compliancy** (their Def. 6): data density `r1` covering the safe-set boundary ∂C (16),
  bounded learning error `M_e` on the dataset (17), and a known **Lipschitz constant** for the
  learned policy composed with the sensor map (18).

**None of these hold in this thesis.** The shield here is a *nominal* CBF-QP with an empirically
tuned lag buffer, not a robust TR-OP controller; the plant is a 7-DoF arm under OSC + IK with
contact, not a known control-affine model; and π0.5 is a 3-billion-parameter flow-matching policy
with no bounded learning error and no controlled Lipschitz constant. The barrier also acts on the
translational channel only, with rotation unconstrained. So the claim "the student is not shown to
inherit forward invariance" is correct, and §5.3 can now say *precisely why* rather than hedging.

Two further usable points. Their **Remark 1** separates the two goals cleanly: "since our goal is
to transfer safety guarantees from the expert controller to the learned controller rather than to
exactly mimic the expert behavior, we show that forward-invariance can be achieved despite
compounding errors if the expert controller enforces robust forward-invariance" — safety transfer
is not behaviour-cloning fidelity, which is the thesis's §4.5 arm-link finding in theoretical form.
And they state that learning from CBF demonstrations in simulation was already known without
guarantees, so **this thesis is not the first to learn safety from CBF-related supervision** — its
novelty is the VLA setting, the removal of the runtime layer, and the teacher-source boundary.

## 7. ConBaT (ICRA 2024) — VERIFIED 2026-08-19
*Closes half of risk C1.*

**Confirmed: it retains inference-time optimization.** Abstract: "During deployment, we employ a
lightweight online optimization to find actions that ensure future states lie within the learned
safe set." Method §III-B: they "use back-propagation to rectify π to satisfy the CBF conditions in
test." So the correction is gradient-based action rectification at test time — a learned barrier
that still optimizes online, exactly as the relevance matrix says. The thesis's no-runtime-filter
property is distinct.

Also worth citing: their critic is trained from **binary safe/unsafe labels only** ("The
supervision signals for learning the critic are just binary labels indicating states' safety,
bypassing the need for hand-crafting complex signals"), and it "operates in embedding space instead
of state space." That contrasts usefully with this thesis's geometric, privileged barrier — theirs
needs less supervision but yields no geometric guarantee; this one needs ground-truth geometry but
produces an interpretable constraint.

---

# Part B — Rewrite-risk register

Each row is a claim the thesis cannot lose without a structural rewrite, the finding that would
break it, and the unread paper most likely to contain that finding. **Read RISK: HIGH papers before
writing the prose that depends on them.**

| # | Load-bearing claim | What would break it | Where it lives | Paper at risk | Risk |
|---|---|---|---|---|---|
| C1 | Distilling a CBF shield into a VLA and **removing it at deployment** is new | A paper that already distils a safety filter and drops it at inference | §1.3, §2.3, Ch 6 ¶1 | ~~ConBaT~~ (confirmed retains online optimization), **Latent Policy Barrier** | **HALF-CLOSED** |
| C2 | The student does **not** inherit formal forward invariance — an empirical result only | Their conditions turn out to be satisfiable here | Ch 6 ¶2, §5.3 | IL_with_CBF | **CLOSED** — none of their three conditions hold here; §5.3 can now name them |
| C3 | The scalar-RL negative is about the tested reward design, not safe RL as a class | A SafeLIBERO safe-RL result that makes our RL arm look under-tuned rather than scoped | §4.7, Ch 6 ¶4 | **SafeDojo** (evaluates on SafeLIBERO) | **HIGH** |
| C4 | Supervision transfers by **state coverage**, not teacher identity | A paper where a foreign teacher on foreign states transfers safety anyway | §4.4, §5.1, Ch 6 ¶3 | LPB, SafeVLA | MEDIUM |
| C5 | Shield exposure alone does not internalise safety; the imitation objective does | Already resolved — Guided by Guardrails **confirms** it | §5.1 | — | **CLOSED** |
| C6 | The arm-link constraint extends AEGIS's end-effector-only barrier | Already resolved — quoted from their limitations | §3.2, §4.2, §4.5 | — | **CLOSED** |
| C7 | Language cannot carry the safety constraint | Already corroborated by ROAD-VLA's three text variants | §4.7 | — | **CLOSED** |
| C8 | Reproduces AEGIS's unshielded baseline (17.5%/58.3% vs 17.3%/58.9%) | Already verified against their Table I | §4.2 | — | **CLOSED** |

## Reading order dictated by the register

1. **IL_with_CBF** (7 pp) — C2. Guards the thesis's most important *negative* claim.
2. **Latent Policy Barrier** (19 pp) — C1 + C4. "Very close to your focus on the policy's own
   rollout distribution"; the question is whether it removes the runtime step or keeps it.
3. **ConBaT** (8 pp) — C1. Same question.
4. **SafeDojo** (20 pp) — C3. The only other SafeLIBERO method here; its numbers may need reporting.
5. SafeVLA (32 pp) — C4, positioning only.

## Low-risk remainder — read only if time allows

| Paper | Why it cannot force a rewrite |
|---|---|
| **VITA-VLA** | Generic action-distillation precedent. Supports plausibility; carries no safety claim. |
| **RT-VLA** | Deployment-efficiency motivation only, and the matrix already caps how it may be cited (no latency was measured here). |
| **π0 / π0.5** | Base-model description. Cannot conflict with a safety result. |
| **CBF.pdf** | Definitions for §2.2 and §3.3. Needed for citation, not for risk. |
| **Collision-avoidance-with-CBFs** | Cite only if the sphere construction is presented as following theirs; it is not. |
| **Flow-GRPO / GRPO** | Implementation lineage for the RL ablation. Appendix material. |

## Standing rule

Any claim written before its RISK: HIGH paper is verified must be phrased so that a bounded edit
can repair it — scoped to this benchmark and setup, with the comparison named rather than implied.
Every such site carries a `[CITE: … CSV, VERIFY]` marker, so the blast radius of a late finding is
always enumerable by grepping for `VERIFY`.
