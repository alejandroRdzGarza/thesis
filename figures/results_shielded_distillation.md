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

## Threat to validity: selection vs distillation

Demonstrations are filtered on three properties at once — the shield fired, the episode succeeded,
and nothing was displaced. Two mechanisms could therefore drive the improvement:

**(a) Distillation** — the policy imitates the shield's avoidance corrections. This is the claim.

**(b) Selection** — training a policy on its own successful episodes improves it regardless of any
shield (filtered behaviour cloning / self-improvement), and successful episodes correlate with not
having knocked anything over. Under this explanation the shield is incidental.

Nothing in the six-arm evaluation distinguishes them. Two quantitative arguments bear against (b):

1. **The demonstrations are pervasively shield-shaped.** The shield was active on a median 32% of
   control steps, with median 51 activations per episode and a minimum of 9 — there is no
   "barely-shielded" tail. Roughly one imitated action in three was modified by the QP, with large
   corrections. The imitated action distribution is therefore substantially different from the base
   policy's, which is not what (b) describes.
2. **The shield finds less to correct afterwards.** `cbf|r|` falls 0.533 → 0.265 between `base+shield`
   and `r1+shield`. Selection alone does not predict this: a policy that merely completes tasks more
   often would not require *less* safety intervention. Needing half as much correction indicates the
   policy stopped entering the states that triggered it.

**These make (b) implausible; they do not eliminate it.** The decisive test is a matched-size control:
collect demonstrations with the shield OFF, filter identically, train at the same demo count, and
evaluate shield-off. If collision still falls, the gain was selection. Implemented in
`run_shield_control.sh` (~37 h: 21 h collection, 8 h training, 8 h evaluation); **not run**, for
compute-budget reasons. This is stated as a limitation, not claimed as a result.

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
