# The classical controller alone — characterisation and failure analysis

Requested by supervisor 2026-08-13: evaluate the controller by itself, with full privileged
knowledge, and diagnose why it does not do better. Built from the full-grid collection
(`results_distill/planner_A`): 24 scenes x 36 attempts = 864 rollouts, ground-truth geometry.

## Per-scene yield

Clean = succeeded AND displaced no obstacle. Overall: **318/864 = 37%**.

| scene | clean/36 | scene | clean/36 |
|---|---|---|---|
| object_LI_t3 | **35 (97%)** | spatial_LI_t2 | 17 (47%) |
| object_LII_t1 | **34 (94%)** | spatial_LI_t1 | 14 (39%) |
| goal_LI_t0 | 26 (72%) | object_LII_t0 | 14 (39%) |
| goal_LI_t1 | 26 (72%) | spatial_LII_t2 | 14 (39%) |
| goal_LI_t2 | 22 (61%) | spatial_LI_t0 | 12 (33%) |
| spatial_LII_t3 | 20 (56%) | object_LII_t2 | 12 (33%) |
| object_LII_t3 | 19 (53%) | spatial_LI_t3 | 11 (31%) |
| goal_LII_t2 | 17 (47%) | goal_LII_t0 | 9 (25%) |
| | | object_LI_t0 | 8 (22%) |
| | | goal_LII_t3 | 8 (22%) |
| **zero:** goal_LII_t1, goal_LI_t3, object_LI_t1, object_LI_t2, spatial_LII_t0, spatial_LII_t1 | **0** | | |

**The distribution is bimodal, not uniformly mediocre.** Two scenes exceed 94%; six produce nothing.
A single aggregate figure (37%) misrepresents both ends.

**It is not a difficulty effect.** Three of the six zero-yield scenes are LEVEL I
(goal_LI_t3, object_LI_t1, object_LI_t2), while spatial_LII_t3 reaches 56% and object_LII_t1
reaches 94%. Failure is task-specific, not level-specific — which rules out "level II is simply
harder" as the explanation.

## Where the failures occur

Phase occupancy from the per-step logs of the six zero-yield scenes. (Counts are phase mentions
across all steps, so they measure time spent in a phase rather than terminal outcome — except for
goal_LII_t1, where only one phase ever appears.)

```
goal_LII_t1     PLAN_FAILED 3150   (no other phase EVER reached)
goal_LI_t3      DONE 1553, TRANSPORT 394, RELEASE 332, APPROACH 277
object_LI_t1    APPROACH 1509, TRANSPORT 714, DESCEND 507, LIFT 420   (no PLACE)
object_LI_t2    APPROACH 1228, TRANSPORT 1022, DESCEND 451, PLACE 22
spatial_LII_t0  APPROACH 1437, DONE 545, TRANSPORT 277
spatial_LII_t1  DONE 1214, APPROACH 495, TRANSPORT 330

for contrast, the two best scenes:
object_LI_t3    APPROACH 413, TRANSPORT 296, DESCEND 204, PLACE 65
object_LII_t1   TRANSPORT 673, APPROACH 420, DESCEND 183, PLACE 78
```

**Three distinct mechanisms, not one:**

1. **Planning infeasibility** — `goal_LII_t1`. PLAN_FAILED is the only phase ever reached: RRT plus
   the IK/clearance ladder never returns a collision-free plan. Purely geometric; execution is never
   attempted.

2. **Task inexpressibility** — `goal_LI_t3`. The plan executes to completion (DONE dominant) and the
   task is still not satisfied. This is *"open the top drawer and put the bowl inside"*: two-stage
   manipulation that a pick-and-place waypoint sequence cannot represent. Zero here is a statement
   about what was implemented, not about the method.

3. **Execution/tracking failure** — `object_LI_t1`, `object_LI_t2`, `spatial_LII_t0/t1`. Time
   concentrates in APPROACH and TRANSPORT and PLACE is barely or never reached, so a plan exists but
   the arm does not track it to placement. The successful scenes are distinguished precisely by
   reaching PLACE (65 and 78 occurrences).

## What was tried, and what it moved

Seven iterations, each with a measured diagnosis rather than a guess:

| version | change | CLEAN (n=48) |
|---|---|---|
| v2 | clearance inflation | 40% |
| v3 | clearance ladder + IK retry | 35% |
| v4 | fallbacks | 29% |
| v5 | payload speed limit (fixed object slipping from the gripper) | 38% |
| v6 | scene settling before measurement | 38% |
| v8 | four fixes from visual inspection | 35% |
| v10 | orientation gate on waypoint arrival | **40%** |

The aggregate never leaves the 35–40% band. Individual fixes are real and diagnosed — the
orientation gate took the closing error from 35.5 deg to 0.0 deg on a scene that had regressed to
zero — but each one trades scenes against each other rather than lifting the total. That pattern,
across seven attempts, is the evidence that the residual is structural rather than a tuning
deficit.

## Conclusion

With full privileged geometry the controller reaches 37% clean demonstrations, bimodally
distributed, failing through three separable mechanisms of which one (task inexpressibility) is
outside the scope of what was built and one (planning infeasibility) is purely geometric.

A task-and-motion planner with drawer primitives would address mechanism 2. It would not obviously
address mechanism 3, which is where most of the lost yield sits. Neither would change the
distillation result, which concerns the teacher being *foreign* to the policy rather than being
imperfect: the student trained on these 318 clean demonstrations is statistically indistinguishable
from the undistilled base policy.

## A physical constraint mistaken for a control failure

One earlier failure is worth recording because of how it was diagnosed. Several bowl tasks sat at
0% success with the end-effector stalling roughly 5 cm above the object centre, which reads as a
controller or IK problem. It was neither. The akita bowl is 11 cm wide and the gripper opens to
8 cm, so a top-down straddle grasp is *physically impossible* — no controller could have solved it.

The fix was a change of grasp strategy rather than of control. Probing the gripper established that
the fingers close along the world $y$ axis, so offsetting the grip point by 5 cm along that axis
drops one finger inside the bowl and one outside, pinching the rim on close. Bowls are selected for
this mode by name while cartons retain the top-down grasp, and the full three-dimensional grasp
offset is tracked so that placement drives the *object* to the goal rather than the offset
end-effector. A speed-capped, $xy$-locked descent was added alongside it to remove operational-space
coupling drift. The spatial suite went from 0% to 69% with zero collisions.

This is reported for two reasons. It is the clearest case in this project of a metric-level failure
whose cause lay outside the system being tuned, and it bears on how Section 4.4's coverage gaps
should be read: the teacher's remaining failures were investigated for this class of explanation and
the elevated-target failures survived that check, which is what makes the kinematic-limit reading
credible rather than merely convenient.
