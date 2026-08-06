# Thesis status — safety internalisation for Vision-Language-Action models

**MSc Robotics and Computation, UCL · COMP0247**
**Written 2026-08-06. Self-contained: assumes no prior knowledge of the project.**

This document exists so a reader who has not followed the work can judge what has been achieved,
how solid it is, and what is still open. Results are labelled **verified**, **in progress** or
**preliminary**. Where a number has a caveat that weakens it, the caveat is stated next to the
number rather than in a footnote.

---

## 1. Research question

Vision-Language-Action (VLA) models — large pretrained policies that map camera images plus a
natural-language instruction directly to robot actions — can follow instructions but have no
concept of collision safety. The standard remedy is to bolt on an external safety filter that
projects unsafe actions to safe ones at every timestep.

That remedy is unsatisfying in two ways: the filter is needed forever, and it costs task success,
because it repeatedly pushes the policy into states its training never covered.

**The question:** can the safety be moved *inside* the policy, so that the external filter becomes
largely redundant rather than permanently necessary?

**Sub-questions:**
1. Does a Control Barrier Function (CBF) filter actually make a modern VLA safe, and at what cost?
2. Can that filter's behaviour be transferred into the policy's own weights?
3. Can the degree of remaining reliance on the filter be *measured*, not just asserted?

Sub-question 3 is where the main novelty sits: **CBF activation rate** — how often the external
filter has to intervene — is used as a primary evaluation metric, not just as a diagnostic.

---

## 2. Setup

| | |
|---|---|
| Policy | π0.5 (Physical Intelligence), PaliGemma backbone, flow-matching action head |
| Robot | Franka Panda in MuJoCo, via the LIBERO/robosuite wrapper |
| Control | OSC_POSE — 7-D actions: end-effector position/orientation deltas + gripper |
| Benchmark | **SafeLIBERO**: LIBERO with one obstacle placed in the workspace per episode |
| Scale | 3 suites (spatial / object / goal) × 2 difficulty levels × 4 tasks = 24 scenes |
| Hardware | RunPod A40 for training and policy rollouts; local CPU for planning and classical control |

**Metrics** (matched to the AEGIS baseline so numbers are comparable):
- **TSR** — task success, from the benchmark's own `check_success` predicate.
- **CAR** — collision avoidance rate = 100 − collision rate; a collision is the obstacle being
  displaced by >1 mm (AEGIS's threshold, stricter than the 2 mm used earlier in this project).
- **ETS** — mean control steps to success, over successful episodes only.
- **CBF activation rate** — fraction of steps on which the safety filter modified the action.

---

## 3. What has been built

Approximately 30 modules under `experiments/`, all under version control. The substantial pieces:

- **CBF safety layer** — ellipsoid barrier with a QP projection, obstacle geometry fitted as a
  minimum-volume enclosing ellipsoid over the object's true mesh point cloud, plus a sphere
  decomposition variant. Validated in isolation with a server-free diagnostic harness that drives
  the arm at the obstacle deliberately.
- **Collision attribution** — a contact-graph search that determines whether the robot caused an
  obstacle displacement directly, via a push chain through another object, or not at all.
- **Flow-SDE / GRPO stack** — a port of Flow-GRPO's ODE→SDE conversion giving tractable action
  log-probabilities for a flow-matching policy, wired into π0.5 and validated end-to-end on GPU.
- **Distillation pipeline** — demo collection, behaviour cloning on the flow-matching loss, DAgger
  rounds, and a held-out evaluation harness reporting Wilson confidence intervals.
- **Two expert demonstrators** — a scripted MPC-CBF pick-and-place controller, and a joint-space
  RRT planner (below).

---

## 4. Results

### 4.1 The CBF filter works, and matches published work — **verified**

Full grid, 240 episodes per mode, AEGIS-matched settings:

| | CAR | TSR |
|---|---|---|
| π0.5 alone | 15.4 | 60.0 |
| π0.5 + CBF | **87.5** | **79.2** |
| AEGIS (published, translational) | 77.3 | 76.3 |

CAR improved in all 6 suite/level combinations; TSR in 5 of 6.

**Caveat, stated prominently because it matters:** obstacle geometry here comes from simulator
ground truth, whereas AEGIS infers it from vision and depth. That removes their perception error
(worth roughly 10 CAR points). These numbers are therefore the *analytic ceiling* of the filter,
not a like-for-like win over a perception-limited system.

**A structural finding.** Breaking collisions down by which body caused them:

```
gripper       84  →  0     eliminated
held object   15  →  0     eliminated
arm links     21  → 21     unchanged
```

An end-effector barrier cannot reason about where the elbow is. This limitation is *predicted* by
the method and *confirmed* by measurement, and it motivates the whole-arm planner in §4.4.

### 4.2 Scalar-reward RL cannot separate safety from task competence — **verified negative result**

Four experiments (001–004) with online flow-matching RL. Every configuration landed in one of two
regimes: unstable enough to move the policy's behaviour, which collapsed it to inaction (success →
0); or stable enough to preserve the task, which learned no avoidance at all (unshielded collision
pinned near 1.0).

The interpretation is not "needs more tuning". The obstacle lies *on the path to the goal*, so a
single scalar episode reward provides no per-step spatial credit distinguishing "detour around the
obstacle" from "stop reaching for the goal". **The learning signal and the destabilising signal are
the same knob.**

This is a legitimate contribution: it establishes *why* the dense-supervision approach that follows
is necessary, rather than merely preferred.

### 4.3 Safety can be distilled — **verified on one suite, full grid in progress**

Switching to dense per-step supervision: the CBF already computes the correct safe action at every
timestep, so the policy is trained to imitate that corrected action directly.

Full spatial suite, 4 tasks × 15 **held-out** initial states, n=60 per policy, Wilson 95% CIs:

| policy | success | collision | CBF activation |
|---|---|---|---|
| base π0.5, no shield | 51.7% | 90.0% | — |
| **distilled, no shield** | **51.7%** | **15.0%** | — |
| base π0.5 + shield | 80.0% | 11.7% | 0.589 |
| distilled + shield | 38.3% | 0.0% | **0.142** |

**The headline: collision 90% → 15% at identical task success, with no filter at all.** Collision
confidence intervals do not overlap. Evaluated on initial states never trained on, so this is
generalisation rather than memorisation.

Two honest qualifications:
- The distilled policy still collides 15% unshielded. Safety is substantially internalised, not
  perfected.
- Distilled + shield is *worse* than base + shield on success (38% vs 80%). Once the policy already
  avoids, the shield over-corrects it off the grasp. **The value of distillation here is
  shield-free operation, not stacking with the shield** — an important framing point, since the
  naive expectation is the opposite.

**In progress:** the same recipe scaled to all 24 scenes with a stronger teacher (the shielded
policy rather than a scripted controller). Collection is complete and shows the mechanism
reproducing at scale:

| | round 0 (base π0.5) | round 1 (after one BC round) |
|---|---|---|
| task success | 202/288 | 200/288 |
| clean demos | 186 | 182 |
| **shield-free clean episodes** | **0** | **16** |
| **CBF activation rate** | **0.331** | **0.162** |
| mean correction magnitude | 0.335 | 0.081 |

Task capability flat, shield reliance halved, correction magnitude quartered, and 16 episodes
completed with no intervention at all against zero before. The held-out evaluation of these
checkpoints is running; **the shield-free collision rate — the actual claim — is not yet measured
at this scale.**

An additional observation worth investigating: the earlier single-task experiment showed success
eroding across DAgger rounds, which forced early stopping. At full scale (186 demos across 24
scenes vs ~32 demos on one task) **no erosion appears at round 1**, suggesting that erosion was
substantially a small-data artefact rather than a property of the method.

### 4.4 A whole-arm planner as expert demonstrator — **preliminary**

Motivated by §4.1's residual: an end-effector filter provably cannot prevent arm-link collisions,
so a teacher that reasons about the whole arm can demonstrate something the shield cannot.

Built: joint-space RRT-Connect with collision checking against true geometry, grasp-pose search
over the object's measured shape, transport planning that carries the held object in the swept
volume, and forward kinematics converting joint paths to end-effector traces — so the demos are in
the same action space π0.5 outputs and are reproducible by the student.

**Verified:** RRT plans executed through the operational-space controller stay collision-free on
12/12 test rollouts, despite up to 4.66 rad of drift between the planned and executed arm
configuration. This disproved an earlier conclusion in this project that joint-space planning could
not transfer to an end-effector-controlled policy.

**Preliminary:** as a full pick-and-place teacher, the best configuration reaches ~58% success /
40% collision, ~40% clean demos, against ~36–45% success for the scripted controller. Adding a
planning clearance margin took arm-link collisions from 5 to 1 — the residual the shield cannot fix.

This line is *not* required for the main result and is best presented as an ablation on teacher
design.

---

## 5. Limitations

Stated plainly; several are consequences of scope rather than oversights.

1. **Simulation only.** No physical robot. The safety claim is about a simulated Franka Panda.
2. **Privileged geometry.** Obstacle shape and pose come from the simulator, not perception. This
   is what makes the CBF numbers exceed AEGIS's and it must be stated wherever they are compared.
3. **Non-reproducible rollouts — found and fixed 2026-08-06.** Re-running 26 identical episodes
   originally produced a different score on 6 of them (~23%). Cause: SafeLIBERO scenes carry the
   obstacle variants for every safety level and park the unused ones far outside the workspace,
   where they sit interpenetrating the ground plane and generate thousands of contacts. That
   overflowed MuJoCo's contact buffer (`ncon = 5000`), and when the buffer overflows, which
   contacts get dropped is order-dependent — so identical episodes could diverge. Fixed by clearing
   `contype`/`conaffinity` on parked bodies, removing them from contact generation. Verified: zero
   overflow warnings, and three identical runs now produce identical outcomes.
   **Numbers produced before this fix carry roughly 23% per-episode noise** and should be re-run
   before any fine-grained comparison is reported; the large effects (90% → 15% collision) are far
   outside that noise and stand.
4. **Residual unshielded collision.** The distilled policy still collides ~15% of the time without
   the shield. The claim is reduced reliance, not eliminated need.
5. **Distillation and the shield conflict.** Stacking them reduces success; they are alternatives,
   not complements.
6. **One benchmark scene is out of scope.** `goal` level-I task 3 is "open the top drawer and put
   the bowl inside" — a two-stage articulated-object task no pick-and-place teacher can express.
7. **Single seed** for most training runs; no seed-variance study.

---

## 6. Contributions

1. **CBF activation rate as a primary evaluation metric** for VLA safety — quantifying how much a
   policy still depends on its external safety filter, rather than only whether it is safe with one
   attached. Measured to fall ~4× under distillation.
2. **A measured negative result** establishing that scalar-reward RL cannot separate safety
   learning from task destruction in this setting, with a mechanistic explanation.
3. **Evidence that shield behaviour is distillable**: collision 90% → 15% at unchanged task
   success, on held-out initial states, with no filter at eval time.
4. **A characterisation of what an end-effector safety filter structurally cannot do** — gripper and
   held-object collisions eliminated, arm-link collisions untouched — together with a whole-arm
   planning teacher that addresses precisely that gap.
5. **An open, reproducible SafeLIBERO evaluation harness** with metrics matched to published work.

---

## 7. Current state and remaining work

**Running now:** held-out evaluation of the full-grid distillation — six policy arms across 24
scenes. This produces the headline table.

**Immediate:**
- Fix the simulator non-determinism (§5.3) before reporting any fine-grained comparison.
- Complete the evaluation and produce the confidence-interval table.

**Then, in priority order:**
1. Write up. The report is due in September; a scaffold exists at `thesis/main.tex`.
2. Optionally finish the planner teacher as an ablation on teacher design.
3. Optionally a second seed on the headline arms.

**Explicitly out of scope:** real-robot transfer, perception-based obstacle estimation,
articulated-object manipulation.

---

## 8. Notes for a reader estimating a mark

Against the UCL PG project criteria:

- **Background and organisation** — the work is positioned against a specific recent baseline
  (AEGIS) with metrics deliberately matched to it, and against the flow-matching RL literature
  (Flow-GRPO, πRL). The progression from filter → RL → imitation is driven by measured results at
  each stage rather than chosen up front.
- **Difficulty and achievement** — the deliverables include a working CBF safety layer, a ported
  and GPU-validated online RL stack for a flow-matching policy, two expert demonstrators, a
  distillation pipeline, and an evaluation harness. The primary result (collision 90% → 15% at
  unchanged success, shield-free, on held-out states) is a genuine positive finding.
- **Analysis** — several conclusions in this project were reached, measured, and then *overturned
  by further measurement*: the erosion attributed to the method turned out to be a data-scale
  artefact; a claimed action-space limitation was disproved by direct test; the metric
  non-determinism in §5.3 was found by deliberately re-running scored episodes. The negative RL
  result is characterised rather than merely reported.
- **Known weaknesses** — simulation only, privileged geometry, single seed, and a residual 15%
  unshielded collision rate. These are stated in §5 rather than left for a reader to find.

The strongest claim this work can support is: *distilling a CBF safety filter into a VLA reduces
unshielded collisions roughly six-fold at no cost to task success, and reduces the policy's
dependence on the filter roughly four-fold, in simulation with privileged obstacle geometry.*

