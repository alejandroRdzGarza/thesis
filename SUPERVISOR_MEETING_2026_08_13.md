# Supervisor meeting — final results

Prepared 2026-08-12. Experiments are closed; the remaining work is writing.
Every number below is measured. Where something is uncertain or unresolved it says so explicitly.

---

## 1. The headline

Evaluated on **held-out initial states** (training used inits 0–34, evaluation uses 35–39), across
all 24 SafeLIBERO scenes, 5 rollouts each → **n = 120 per arm**. 95% Wilson intervals.

| policy | TSR | collision | CAR | ETS |
|---|---|---|---|---|
| base π0.5, no shield | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] | 17.5% | 138.6 |
| base + CBF shield | 71.7% [63.0, 79.0] | 13.3% [8.4, 20.6] | 86.7% | 160.6 |
| **distilled (r1), no shield** | **82.5% [74.7, 88.3]** | **19.2% [13.1, 27.1]** | **80.8%** | 157.3 |
| distilled (r2), no shield | 80.8% [72.9, 86.9] | 17.5% [11.7, 25.3] | 82.5% | 158.8 |

**The claim: a runtime safety filter's behaviour can be absorbed into the policy.** Collision falls
82.5% → 19.2% *with no shield running at inference*, and success rises 58.3% → 82.5%. Both intervals
are disjoint from base. The distilled policy recovers most of the shield's safety (CAR 80.8% vs
86.7%) without needing it at deployment.

One round is enough — r1 and r2 overlap. Round 2 did not erode either, which matters because an
earlier DAgger-style experiment in this project degraded across rounds.

**Cost, reported rather than buried:** ETS rises 138.6 → 157.3, about 14% slower.

---

## 2. Why the classical controller did not work

This is the point that most needs discussing, since the original expectation was that a classical
controller would reach ~100% success and ~100% collision avoidance.

### 2a. The controller itself: 38% clean, and bimodal

A sampling-based expert was built properly — RRT-Connect in joint space with clearance inflation,
IK, and playback as OSC deltas — and iterated through seven versions with measured diagnostics at
each step. Final yield over the full grid: **318 usable demonstrations from 840 rollouts (38%)**.

The distribution is the interesting part. It is **not uniformly mediocre — it is bimodal**:

| scene | clean demos / 35 |
|---|---|
| object LI t3 | **35 (100%)** |
| goal LI t0 | 26 |
| object LII t0 | 14 |
| spatial LII t2 | 14 |
| **object LI t1, object LI t2, spatial LII t0, spatial LII t1, goal LI t3, goal LII t1** | **0** |

Six of 24 scenes produced **zero** demonstrations. So the planner solves some geometries perfectly
and others not at all, rather than being uniformly imperfect.

Two distinct causes, which should not be conflated:

- **Scope**: `goal LI t3` is *"open the top drawer and put the bowl inside"* — two-stage
  manipulation that a pick-and-place waypoint plan cannot express at all. Zero here is a statement
  about what was implemented, not a failure of the method.
- **Capability**: the other five are reachable geometry the planner attempts and loses — elevated
  and occluded placements (stove, cabinet), which earlier per-task analysis traced to
  kinematic/OSC-posture limits rather than occlusion or collision.

### 2b. Distilling from it fails completely

| policy | TSR | collision |
|---|---|---|
| base π0.5 | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] |
| **planner-distilled** | **54.2% [45.3, 62.8]** | **80.0% [72.0, 86.2]** |
| self-distilled (r1) | 82.5% [74.7, 88.3] | 19.2% [13.1, 27.1] |

The planner-distilled student is **statistically indistinguishable from the undistilled base**. It
learned essentially nothing.

This is a controlled comparison: same starting checkpoint, same objective, learning rate and batch
size, **matched on gradient steps to within 1%** (15,988 vs 16,140 — matched on *steps* rather than
epochs, because the planner's horizon-900 episodes yield ~412 training examples each against ~35
for a shielded rollout, so equal epochs would have meant 212,000 steps vs 16,140), same evaluation
code, same 24 scenes, same held-out states. The only difference is where the demonstrations came
from.

**So: distilling safety from a foreign expert fails; distilling it from the policy's own shielded
rollouts succeeds.** That contrast is the sharpest result in the thesis, and it required building
the planner to obtain — the negative arm is what gives the positive one meaning.

### 2c. The explanation we tested and rejected

The intuitive account was observation aliasing: π0.5 sees an image plus 8-D proprio with no joint
angles, so a joint-space plan is not a function of what the policy observes. **Measured, and it is
wrong.** At matched observation radius the planner's data is roughly **2× less** aliased than the
shielded data at every radius — the teacher that fails has the *cleaner* observation-to-action
mapping. The prediction was recorded before the number was computed.

Surviving explanations: distribution shift (the student never visits the planner's states at
inference) and action-distribution mismatch (saturated P-control/QP deltas against π0.5's
flow-matched chunks). These are not separated by the current evidence.

### 2d. Honest position on "100% is achievable"

Nothing measured supports 100% for this controller on this benchmark, and the failure is
structured rather than a tuning problem: seven iterations moved the aggregate between 35% and 40%
while individual fixes traded one scene against another. Two of the six zero-scenes are outside
what pick-and-place can express at all.

That said, this is a statement about *this* planner. A task-and-motion planner with drawer
primitives would address the scope cases. It would not obviously address the capability cases, and
it would not change section 2b — the distillation failure is about the teacher being foreign, not
about it being imperfect.

---

## 3. What the shield can and cannot reach

Per-body attribution, recovered from evaluation logs. `scene_object` excluded: it fires on 120/120
episodes of every arm because the obstacle resting on its surface registers a permanent contact.

| policy | collided | gripper | arm link | held object |
|---|---|---|---|---|
| base, no shield | 99 | 71 | 17 | 36 |
| base + shield | 16 | **0** | 15 | 0 |
| r1, no shield | 23 | 7 | 13 | 9 |
| r2, no shield | 21 | 8 | 8 | 7 |

**The shield eliminates the end-effector channel entirely** — 71 → 0, and zero across all 360
shielded episodes. Of its 16 residual collisions, 15 are arm links.

**So the 13.3% floor is a scope limit, not a leak.** The barrier constrains end-effector spheres;
arm links and the carried object are not in it, while the metric scores every body. The corollary
matters for reading the headline: **r1's 17.5% shield-free is within a few points of the practical
ceiling of the supervision it received**, not a degraded copy of a perfect teacher.

**A second transfer channel.** Arm-link collisions fall 17 → 8, in a channel the barrier does not
constrain (the shielded baseline still shows 15). That must come from the selection criterion, not
the corrections. Safety transfers two ways: imitation where the filter acts, outcome selection
where it does not. *Caution: those intervals overlap (14.2% [9.0, 21.6] vs 6.7% [3.4, 12.7]), so
this is suggestive; the gripper result is not in doubt.*

---

## 4. The confound, and the control that resolves it

The obvious objection to the headline: the demonstrations were filtered on success *and*
collision-freedom, so maybe the gain is ordinary success-filtering and the shield is incidental.

A matched control was run. Both arms: **85 demonstrations, identical filter, identical training**;
the only difference is whether the shield was active during collection.

| arm | shield at collection | TSR | collision |
|---|---|---|---|
| control | **off** | 50.0% [41.2, 58.8] | **84.2% [76.6, 89.6]** |
| matched | **on** | 75.8% [67.4, 82.6] | **26.7% [19.6, 35.2]** |

**Success-filtering alone achieves nothing** — the control is indistinguishable from undistilled
base (82.5%). **The shield's corrections are what transfers.** Intervals disjoint.

Supporting evidence: the CBF activation proxy falls 0.533 → 0.265 after distillation (the shield
finds less to correct — success-filtering does not predict that), and the demonstrations are
pervasively shield-shaped (barrier active on a median 32% of control steps, mean correction norm
0.33 against actions bounded at ±1).

Also worth reporting: **safe demonstrations are ~2× harder to collect without a shield** — 18%
yield unshielded against 39% shielded.

---

## 5. Approaches tested and rejected

Four mechanisms were tried and did not work. All are characterised rather than blank negatives.

| approach | result |
|---|---|
| **Scalar-reward RL** (flow-SDE GRPO) | Failed. The learning signal and the destabilising signal are the same knob. |
| **CBF-guided sampling** — steer the denoising velocity instead of projecting the output | Implemented and verified (λ=1 reproduces the projection's endpoint exactly on a synthetic barrier). Never activates in practice: the EE's closest approach is **0.231 m** against a 0.15 m barrier on episodes that still collide. **Guidance inherits the barrier's end-effector scope.** |
| **Safety via the language channel** — a clause appended to the instruction | No safety benefit (collision 70.0% → 68.3%, near-coincident intervals) but a **23% slowdown** and ~12 points of lost success. The policy responds to the text without grounding it geometrically — the appearance of caution without its substance. |
| **Best-of-N selection** — sample K chunks, simulate each, execute the safest | Negative as a method, but the measurement is a finding. The fraction of safe candidates *equals* the fraction of queries where all K are safe, at every K and noise level tested — so candidates are **near-perfectly correlated** in safety outcome. **Safety is state-determined, not sample-determined.** Doubling K bought 8.8 points; maxing noise bought 3.7. |

The last one also explains why *episode-level* filtering works while *action-level* selection
cannot: episode-level selection chooses between trajectories that reached different states, which
is where the variation actually is.

---

## 6. Two defects found in the benchmark

Both affect any level-II result produced with unmodified SafeLIBERO, so they are reported in the
thesis rather than an appendix.

- **Non-determinism.** Parked off-scene obstacles overflow MuJoCo's contact buffer (ncon = 5000);
  on overflow contacts are dropped order-dependently. **6 of 26 re-run episodes (23%) scored
  differently.** Fixed by clearing collision flags on parked bodies — five repeats then byte-identical.
- **Phantom collisions.** Objects spawning unsupported fall under gravity with the arm stationary —
  a moka pot fell **102 mm** — and the displacement is charged to the robot. Fixed with 60 settling
  steps; residual 0.00 mm. Ordering matters: settling must come *after* the parked-contact fix.

---

## 7. Open issues — to declare, not hide

**Seed variance is unquantified, and there is an unresolved measurement discrepancy.** Every arm is
a single training run. A replication was attempted: two extra seeds trained on a second pod both
scored ~79% collision, but so did the *original* r1 checkpoint when re-evaluated on that same pod
(17.5% on pod 1 vs 79.2% on pod 2, same weights). So this is a **cross-environment discrepancy, not
a failed replication** — and it is currently unresolved. Pod 1 is authoritative for every number in
the thesis: it produced all six arms, and its baseline agrees with the AEGIS-matched published
reference (CAR 86.7% vs 87.5%).
*Proposed handling: state seed variance as unquantified, with the replication attempt and the
discrepancy described honestly.*

**Privileged geometry.** The barrier uses ground-truth obstacle geometry from the simulator, not
perception. Results are an upper bound for this class of shield with the perception gap removed.

**Single environment.** SafeLIBERO only. No claim of generality across benchmarks or robots.

**Six scenes have no planner demonstrations**, so ~30 of the 120 rollouts in section 2b measure
teacher coverage rather than distillation. Both the 24-scene and 18-scene slices should be reported.

---

## 8. Validity checks that passed

- **Vanilla LIBERO**: the same checkpoint and evaluation code score **5/5 success, 0/5 collision**
  on obstacle-free LIBERO-Spatial. The degraded SafeLIBERO figures reflect task difficulty, not a
  broken pipeline.
- **AEGIS agreement**: base + shield measured CAR **86.7%** against the published **87.5%** — 0.8
  points apart on an independently reported baseline.
- **Determinism**: after the contact fix, five repeated runs of the same episodes are byte-identical.

---

## 9. Questions for the meeting

1. **Is the seed/pod discrepancy handling acceptable** — declare seed variance unquantified and
   describe the discrepancy — or should the replication be resolved before submission (~1 day:
   fix EGL on pod 1, re-evaluate the seed checkpoints there)?
2. **Framing of the classical-controller result.** It is presented as a *boundary result* — the
   negative arm that gives the positive one meaning — rather than as a failed experiment. Does that
   match your expectation of what that work was for?
3. **Is the contribution correctly stated?** Proposed: not "we built a safer VLA" but *"we
   characterise what limits safety mechanisms for VLAs — filters teach only what they perceive,
   outcome selection reaches further than the filter can express, and neither language nor
   generative uncertainty provides a usable safety signal"*, with the self- vs cross-policy
   distillation boundary as the central result.
4. **Are the two benchmark defects worth foregrounding** as a contribution, given they affect
   anyone else's level-II numbers?
5. **Scope of remaining work.** Experiments are closed; three weeks for writing. Any objection?
