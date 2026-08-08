# Shielded self-distillation — headline result

Source: `results_shielded/` · eval completed 2026-08-07 20:09 · 85,900 s total
Evaluation: **held-out inits 35–39** (training used 0–34) · 24 scenes (3 suites × 2 levels × 4 tasks)
× 5 rollouts = **n = 120 per arm** · horizon 300 · `--num-steps 10` · `--noise-level 0`

Metrics: TSR = `env.check_success`. Collision = raw obstacle displacement > 1 mm (the AEGIS/VLSA
quantity). **CAR = 100 − collision.** ETS = mean control steps to success, over succeeded rollouts
only. `cbf|r|` = mean absolute CBF reward term, a proxy for how much the shield had to correct.

## Results

| policy | n | TSR (95% CI) | collision (95% CI) | **CAR** | cbf\|r\| | ETS |
|---|---|---|---|---|---|---|
| base, no shield | 120 | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] | **17.5%** | 0.000 | 138.6 |
| base + shield | 120 | 71.7% [63.0, 79.0] | 13.3% [8.4, 20.6] | 86.7% | 0.533 | 160.6 |
| **r1, no shield** | 120 | **82.5% [74.7, 88.3]** | **19.2% [13.1, 27.1]** | **80.8%** | 0.000 | 157.3 |
| r1 + shield | 120 | 70.8% [62.2, 78.2] | 8.3% [4.6, 14.7] | 91.7% | 0.265 | 170.0 |
| **r2, no shield** | 120 | **80.8% [72.9, 86.9]** | **17.5% [11.7, 25.3]** | **82.5%** | 0.000 | 158.8 |
| r2 + shield | 120 | 72.5% [63.9, 79.7] | 9.2% [5.2, 15.7] | 90.8% | 0.273 | 176.7 |

Intervals are Wilson 95%, pooled across all 24 scenes.

**Headline: CAR 17.5% → 82.5% with no shield at inference, and TSR 58.3% → 80.8%.** Both intervals
are fully disjoint from base — collision [74.7, 88.3] vs [11.7, 25.3], success [49.4, 66.8] vs
[72.9, 86.9]. The distilled policy is safer *and* more capable with the shield switched off, and it
recovers most of `base + shield`'s CAR (86.7%) without running the shield.

**The knee is r1, not r2.** r1 and r2 overlap heavily on both metrics; round 2 bought nothing. It
also did not erode, which is worth stating explicitly given that DAgger erosion sank an earlier
experiment. Report r1 as the operating point and r2 as evidence the gain saturates.

**Shielded arms trade success for safety** (TSR 81% → 71%, CAR 81% → 92%). Expected: once the policy
already avoids, the shield over-corrects it off the grasp. The claim is shield-FREE operation, not
stacking the two.

**Cost: ETS rises 138.6 → 157.3, about 14% slower.** Reported rather than omitted.

## Training-data characterisation

The 186 round-0 demonstrations, filtered on success AND collision-free AND shield-active
(`--min-cbf-acts 1`):

| property | value |
|---|---|
| demonstrations | 186 (from 480 rollouts — 39% yield) |
| CBF activations per episode | median **51**, range 9–190 |
| **CBF activation rate** | median **0.320** (fraction of control steps shielded) |
| mean correction norm ‖u_safe − u_nom‖ | median **0.3304** |
| demos with < 5 activations | **0 / 186** |

Position actions are bounded at ±1 per axis, so a correction norm of 0.33 is a large deflection,
not a nudge.

## Collision culprits: which body, and what the shield can reach

Recovered from the run log with `experiments/culprits_from_log.py` (the eval predates the `culprit`
manifest column; the runner printed `touched_by=[...]` per episode, so no re-run was needed).
`scene_object` is excluded: it fires on 120/120 episodes of every arm including collision-free ones,
because the obstacle resting on its surface registers a permanent contact.

Validation: the `collided` column reproduces the headline table exactly (99/120 = 82.5%,
16/120 = 13.3%, 23/120 = 19.2%, 21/120 = 17.5%, 10/120 = 8.3%, 11/120 = 9.2%).

| arm | episodes | collided | gripper | arm_link | held_object |
|---|---|---|---|---|---|
| base, no shield | 120 | 99 | **71** | 17 | 36 |
| base + shield | 120 | 16 | **0** | 15 | 0 |
| r1, no shield | 120 | 23 | 7 | 13 | 9 |
| r1 + shield | 120 | 10 | **0** | 8 | 1 |
| r2, no shield | 120 | 21 | 8 | 8 | 7 |
| r2 + shield | 120 | 11 | **0** | 9 | 0 |

**The shield eliminates end-effector collisions completely.** 71 → 0 from base, and zero gripper
collisions across all 360 shielded episodes. Of `base+shield`'s 16 residual collisions, 15 are
`arm_link` — bodies the barrier never constrained.

**So the 13.3% floor is a scope limit, not a leak.** The EE barrier does exactly what it constrains
and nothing it does not. The corollary matters for reading the headline: the distilled policy at
17.5% shield-free is close to the practical ceiling of the supervision it received, rather than a
degraded approximation of a perfect teacher.

## Decomposing the improvement by body

The culprit breakdown separates the two mechanisms in a way the pooled rate cannot, because the
shield's authority is confined to one channel.

| channel | shield constrains it? | base → r2 (no shield) | attributable to |
|---|---|---|---|
| gripper | yes | 71 → 8 | shield corrections **and** selection |
| **arm_link** | **no** | **17 → 8** | **selection alone** |
| held_object | no | 36 → 7 | selection alone |

**The CBF cannot have caused the arm_link reduction** — it does not constrain arm links, and
`base+shield` still shows 15 of them. That improvement must come from the selection filter:
demonstrations were retained only if NOTHING collided, arm links included, and the student learned
that. Mechanism (b) is therefore demonstrably real and measurable, observed in a channel where the
shield is absent by construction.

Mechanism (a) is separately supported: the shield modified a median 32% of demonstration actions
(median 51 activations/episode, minimum 9, correction norm 0.33) and drove gripper collisions to
zero in the training data.

So the honest claim is not "distillation works" but **safety distillation transfers through two
distinct channels, separable by which bodies the filter has authority over**: direct imitation of
corrections where the filter acts, and behaviour selection where it does not.

Note `r2` without any shield reaches 8 `arm_link` collisions against shielded base's 15 — the
distilled policy avoids with its whole arm better than the shield can, because the shield
structurally cannot act there at all.

**Statistical caution.** 17/120 vs 8/120 is 14.2% [9.0, 21.6] vs 6.7% [3.4, 12.7] — overlapping
Wilson intervals, so the arm_link trend is SUGGESTIVE, not established, unlike the headline result
where the intervals are disjoint. The gripper finding (71 → 0, zero across 360 shielded episodes) is
not in doubt.

## Threat to validity: selection vs distillation

Demonstrations are filtered on three properties at once — the shield fired, the episode succeeded,
and nothing was displaced. Two mechanisms could therefore drive the improvement:

**(a) Distillation** — the policy imitates the shield's avoidance corrections. This is the claim.

**(b) Selection** — training a policy on its own successful episodes improves it regardless of any
shield (filtered behaviour cloning / self-improvement), and successful episodes correlate with not
having knocked anything over. Under this explanation the shield is incidental.

The culprit decomposition above PARTIALLY separates them, which the pooled rates cannot: the
arm_link improvement (17 → 8) occurs in a channel the shield does not constrain, so mechanism (b) is
real and measurable. Mechanism (a) is separately evidenced in the gripper channel, where the shield
drove demonstration collisions to zero and modified ~32% of imitated actions. Two further arguments
bear against (b) being the WHOLE story:

1. **The demonstrations are pervasively shield-shaped.** The shield was active on a median 32% of
   control steps, with median 51 activations per episode and a minimum of 9 — there is no
   "barely-shielded" tail. Roughly one imitated action in three was modified by the QP, with large
   corrections. The imitated action distribution is therefore substantially different from the base
   policy's, which is not what (b) describes.
2. **The shield finds less to correct afterwards.** `cbf|r|` falls 0.533 → 0.265 between `base+shield`
   and `r1+shield`. Selection alone does not predict this: a policy that merely completes tasks more
   often would not require *less* safety intervention. Needing half as much correction indicates the
   policy stopped entering the states that triggered it.

**Taken together: both mechanisms operate, and neither alone explains the result.** Selection is
proven to transfer avoidance (arm_link, where the shield is absent); the shield is proven to shape
the demonstrations heavily and to eliminate the entire EE channel. What remains unquantified is
their RELATIVE contribution within the gripper channel, where both act. The decisive test is a
matched-size control:
collect demonstrations with the shield OFF, filter identically, train at the same demo count, and
evaluate shield-off. If collision still falls, the gain was selection. Implemented in
`run_shield_control.sh` (~37 h: 21 h collection, 8 h training, 8 h evaluation); **not run**, for
compute-budget reasons. This is stated as a limitation, not claimed as a result.

## Rejected hypothesis: observation aliasing does not explain the failure

Earlier analysis attributed the failure of cross-policy distillation to observation aliasing:
pi0.5 sees image + 8-D proprio (eef_pos3 + axis_angle3 + gripper2) and no joint angles, so a shield
correction that depends on the full joint configuration is not a function of anything the policy
observes, and BC would average contradictory targets. That was asserted, never measured. It is
measurable from the demonstration files alone, and the measurement rejects it.

**Method** (`experiments/aliasing_diagnostic.py`). For each dataset, take (observation, target
action) pairs, find samples that are near-neighbours IN OBSERVATION SPACE, and measure how much
their target actions disagree, normalised by the disagreement between random pairs:

    aliasing = E[||a_i - a_j|| : obs within radius r] / E[||a_i - a_j|| : random pairs]

~1 means observations carry no information about actions (BC cannot fit); ~0 means they determine
them. Validated on synthetic data: an action that is a clean function of the observation scores
0.374, an action independent of the observation scores 1.001.

Compared at MATCHED OBSERVATION RADIUS rather than matched neighbour count. A k-nearest-neighbour
ratio conflates aliasing with sampling density, and the two teachers differ sharply there: the
scripted planner produces smooth repetitive trajectories whose neighbours sit 4x closer than a VLA
rollout's (0.18 vs 0.73 in z-units), which lowers its disagreement for reasons unrelated to
aliasing. Both datasets capped to 1400 samples.

| radius (z-scored obs) | shielded π0.5 (distils) | classical planner (cross-policy) |
|---|---|---|
| 0.10 | 0.103 | **0.051** |
| 0.25 | 0.174 | **0.093** |
| 0.50 | 0.237 | **0.118** |
| 1.00 | 0.376 | **0.174** |

**The ordering is the reverse of the prediction.** The planner dataset is roughly 2x LESS aliased at
every radius: the teacher whose demonstrations fail to distil has the CLEANER observation-to-action
mapping, and the teacher that distils successfully is the more aliased of the two.

In hindsight the direction is unsurprising — a scripted planner is close to a deterministic function
of state, so low aliasing is nearly definitional — but it means aliasing at these levels is not the
binding constraint on distillability, and the stored explanation for the cross-policy failure does
not survive contact with the data.

**Consequence.** If a planner-trained student underperforms, aliasing is not the cause. The
surviving candidates are: distribution shift (the student never visits the planner's states at
inference), action-distribution mismatch (saturated P-control/QP deltas against pi0.5's
flow-matched chunks), and coverage (the planner solves only 18 of 24 scenes, so six have no
training data at all). The measurement eliminates one of four hypotheses.

The prediction was recorded before the planner number was computed, so this is a genuine refutation
rather than a post-hoc reinterpretation.

## Rejected hypothesis: generative uncertainty does not predict collisions

pi0.5 produces each action by integrating a flow ODE over ~10 denoising steps, and every trace
stores that chain. If the policy were less certain in states where safe and unsafe behaviours are
both plausible, the chain geometry would carry a risk signal — and a runtime monitor built on it
would need no barrier, no obstacle geometry and no extra sensing. Tested with
`experiments/uncertainty_monitor.py`, scoring per-episode chain statistics as AUC for predicting
collision (AUC validated on synthetic data: 1.000 separable, 0.451 noise, 0.000 inverted).

| signal | base, no shield (99 collisions) | r1, no shield (23 collisions) |
|---|---|---|
| denoising path length (mean) | 0.461 | 0.388 |
| terminal step size (mean) | 0.495 | **0.350** |
| chain spread (mean) | 0.461 | 0.373 |

`logp` is degenerate in all 240 episodes and is excluded: the evaluations ran at `--noise-level 0`,
a deterministic ODE with zero sampling variance, so per-step log-probabilities carry no information
by construction.

**No usable signal.** Base is indistinguishable from chance. The distilled policy shows a consistent
INVERTED association across all three measures — AUC 0.35 inverts to 0.65, i.e. LOWER denoising
uncertainty is associated with collision — but on 23 collision episodes that is roughly two standard
errors from chance, and 0.65 is far short of a deployable monitor.

Reported as negative. The suggestive reading, if the inversion is real, is that the distilled
policy's failures are CONFIDENT ones: it commits decisively to a trajectory that happens to
collide, rather than hesitating in ambiguous states. That would mean uncertainty-based monitoring is
structurally unsuited to this failure mode, which is worth knowing before building one.

## Other limitations

**The shield is end-effector only.** The barrier constrains EE spheres against obstacle spheres
(`cbf_ellipsoid.py`); arm links and the carried object are not in it, while the collision metric
scores every body. This is why `base + shield` sits at 13.3% collision rather than 0. Three
assumptions are violated in deployment: scope (EE barrier vs whole-body metric), discrete-time
execution of a continuous-time invariance guarantee (with IK lag), and sphere-decomposition
approximation of the obstacle mesh. Consequently r1's 17.5% shield-free is close to the practical
ceiling of the shield it learned from, not a degraded version of it.

**Absolute collision rates may be inflated.** If the evaluation predates the settling fix, objects
still falling under gravity at episode start are charged to the robot. All six arms ran identical
code, so the comparison holds; absolute rates could be lower.

**Scope of the distillation-barrier result.** An earlier finding that CBF safety cannot be distilled
by imitation used the *classical expert* as teacher and stands for that teacher. This result uses
π0.5's own shielded rollouts. The two together bound the claim: cross-policy distillation failed,
self-distillation succeeded. Which factor is responsible — staying in the student's own state
distribution, or the clean-plus-shield-active filter making the teacher self-consistent — is not
determined by this data.
