# Chapter 4 — Experiments and Results (DRAFT)

Plain prose, no formatting. Iterate here, port to LaTeX when settled.
Numbers are final unless marked PENDING.

---

## 4.1 Experimental setup

All experiments use the SafeLIBERO benchmark: three task suites (spatial, object, goal), two
safety levels, and four tasks per suite, giving 24 distinct scenes. Each scene provides 50
randomised initial states. Initial states 0–34 are used for demonstration collection and
training; states 35–49 are held out and used only for evaluation. Every result reported here
evaluates five held-out initial states per scene, giving n = 120 rollouts per policy.

Rollouts run for a horizon of 300 control steps. The policy is queried every five steps and
emits an action chunk, from which the next five actions are executed. Action sampling uses ten
denoising steps at a noise level of zero, making the sampler a deterministic ODE; rollouts are
therefore reproducible given the same initial state.

Four metrics are reported. Task success rate (TSR) is the benchmark's own success predicate.
Collision is raw obstacle displacement exceeding one millimetre, the same quantity reported by
AEGIS, and collision-avoidance rate (CAR) is its complement. Execution time to success (ETS) is
the mean number of control steps to completion, computed over successful rollouts only — a
failed rollout has no completion time, and averaging in the horizon would make a policy appear
faster the more often it fails. Finally, the CBF activation proxy reports how much the shield
had to correct, and is used as an independent signal of whether a policy has internalised
avoidance.

All proportions are reported with 95% Wilson confidence intervals rather than normal
approximations, because rates near zero and one at n = 120 are poorly served by the latter.

**Comparing training runs.** Where two distilled policies are compared, they are matched on the
number of gradient steps rather than the number of epochs. This matters because episode length
differs by teacher: a shielded VLA rollout at horizon 300 yields roughly 35 training examples,
while a scripted-planner episode at horizon 900 yields roughly 412. Matching epochs would have
given one policy 16,140 gradient steps and the other approximately 212,000 — not a controlled
comparison but a different experiment. The two distilled policies compared in Section 4.4 were
trained for 16,140 and 15,988 steps respectively, a difference of under one percent.

---

## 4.2 Does the shield make π0.5 safe?

The first question is whether a control barrier function shield, applied as a runtime filter,
makes a pretrained VLA safe on this benchmark at all.

| policy | n | TSR (95% CI) | collision (95% CI) | CAR |
|---|---|---|---|---|
| base, no shield | 120 | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] | 17.5% |
| base + shield | 120 | 71.7% [63.0, 79.0] | 13.3% [8.4, 20.6] | 86.7% |

Unshielded, π0.5 displaces the obstacle in 82.5% of episodes. With the shield active this falls
to 13.3%, a reduction of 69 percentage points with disjoint confidence intervals.

Success also rises, from 58.3% to 71.7%. This is not incidental: a collision frequently derails
the episode, knocking the target object out of reach or displacing the goal, so preventing
collisions helps the policy complete the task. This point is worth holding onto, because in
Section 4.3 the same shield applied to an already-avoidant policy has the opposite effect on
success, and the two observations are only consistent once the mechanism is stated.

**Caveat on privileged information.** The barrier is constructed from ground-truth obstacle
geometry taken directly from the simulator, not from perception. A deployed system would have to
estimate that geometry from sensors, and would inherit the resulting error. The numbers here
should therefore be read as an upper bound on what this class of shield achieves, with the
perception gap removed by construction.

As an external check, the shielded baseline can be compared against the AEGIS-matched reference
for the same configuration, which reports CAR 87.5%. The measurement here is 86.7% — agreement
within 0.8 percentage points on an independently published baseline, which supports the view
that the pipeline is behaving as intended rather than producing an artefact.

---

## 4.3 Can the shield's safety be internalised?

A runtime filter must run forever. It occupies inference budget on every control step, requires
obstacle geometry at deployment, and is a component that can fail. The central question of this
work is whether the behaviour it induces can instead be absorbed into the policy, so that the
policy is safe with the filter switched off.

Demonstrations were collected by rolling out the shielded policy and retaining episodes that
succeeded, displaced nothing, and in which the shield actually intervened at least once. That
last criterion matters: a clean episode in which the shield never fired is ordinary base-policy
behaviour and teaches nothing about safety. The retained set was 186 demonstrations from 480
rollouts, a 39% yield. These were behaviour-cloned into the policy's action head, and the
procedure was repeated for a second round.

| policy | n | TSR (95% CI) | collision (95% CI) | CAR | cbf\|r\| | ETS |
|---|---|---|---|---|---|---|
| base, no shield | 120 | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] | 17.5% | — | 138.6 |
| base + shield | 120 | 71.7% [63.0, 79.0] | 13.3% [8.4, 20.6] | 86.7% | 0.533 | 160.6 |
| **round 1, no shield** | 120 | **82.5% [74.7, 88.3]** | **19.2% [13.1, 27.1]** | **80.8%** | — | 157.3 |
| round 1 + shield | 120 | 70.8% [62.2, 78.2] | 8.3% [4.6, 14.7] | 91.7% | 0.265 | 170.0 |
| **round 2, no shield** | 120 | **80.8% [72.9, 86.9]** | **17.5% [11.7, 25.3]** | **82.5%** | — | 158.8 |
| round 2 + shield | 120 | 72.5% [63.9, 79.7] | 9.2% [5.2, 15.7] | 90.8% | 0.273 | 176.7 |

The headline result is the third row. With no shield running at inference, the distilled policy
displaces the obstacle in 19.2% of episodes, against 82.5% for the policy it was distilled from —
and its success rate rises from 58.3% to 82.5%. Both intervals are disjoint from the base
policy's. The distilled policy is simultaneously safer and more capable than its own teacher,
without the teacher's filter.

It is worth noting how close this comes to the shielded baseline: CAR 80.8% shield-free against
86.7% with the shield active. Most of the shield's safety benefit survives its removal.

**One round is enough.** Rounds one and two overlap on both metrics, so the second round buys
nothing measurable. It also does not erode performance, which is worth stating explicitly: an
earlier DAgger-style experiment in this project degraded across rounds, and the absence of that
degradation here is informative rather than merely convenient.

**Stacking the shield costs success.** Applying the shield to the distilled policy improves
safety further (CAR 80.8% to 91.7%) but reduces success (82.5% to 70.8%). This is the mirror
image of Section 4.2 and has the same explanation. The shield operates by projecting the policy's
action onto the safe set. When the policy frequently proposes unsafe actions, that projection is
a net benefit. When the policy already avoids the obstacle, the projection instead perturbs
actions that were already both safe and competent, displacing them from the distribution the
policy was trained to produce — safe, but no longer a well-formed grasp. The claim of this work
is therefore about shield-free operation, not about stacking the two.

**The cost of internalised caution.** ETS rises from 138.6 to 157.3 steps, roughly 14% slower.
The distilled policy takes a more conservative route. This is a real cost and is reported rather
than omitted.

**Characterising the training data.** Because the demonstrations are the mechanism, their content
is worth quantifying. Across the 186 retained demonstrations the shield was active on a median of
32% of control steps, with a median of 51 activations per episode, a minimum of nine, and no
demonstration below five. The mean correction norm was 0.33 against actions bounded at ±1 per
axis. Roughly one imitated action in three was therefore modified by the barrier, and modified
substantially — the imitated action distribution differs materially from the base policy's own.

---

## 4.4 Does the source of the demonstrations matter?

Section 4.3 distilled the policy from its own shielded rollouts. An obvious alternative is to
distil from a stronger, purpose-built expert. This section reports that comparison, and it is the
sharpest result in this chapter.

A sampling-based motion planner was implemented as a privileged expert: RRT-Connect in joint
space with clearance inflation, followed by inverse kinematics and playback as operational-space
deltas. Unlike the shielded policy, this expert is self-safe by construction rather than by
correction. It produced 318 demonstrations that succeeded and displaced nothing, across 18 of the
24 scenes — six scenes it could not solve at all.

| policy | n | TSR (95% CI) | collision (95% CI) |
|---|---|---|---|
| base, no shield | 120 | 58.3% [49.4, 66.8] | 82.5% [74.7, 88.3] |
| **planner-distilled, no shield** | 120 | **54.2% [45.3, 62.8]** | **80.0% [72.0, 86.2]** |
| self-distilled (round 1), no shield | 120 | 82.5% [74.7, 88.3] | 19.2% [13.1, 27.1] |

The planner-distilled policy is statistically indistinguishable from the undistilled base policy
on both metrics. It learned essentially nothing.

This is a controlled contrast. Both students start from the same checkpoint, are trained with the
same objective, learning rate and batch size, are matched on gradient steps to within one
percent, and are evaluated by the same code on the same 24 scenes and the same held-out initial
states. The only difference is where the demonstrations came from. Distilling safety from a
foreign expert failed; distilling it from the policy's own shielded rollouts succeeded.

**Caveat.** Six of the 24 scenes contained no planner demonstrations, so roughly 30 of the 120
evaluation rollouts test the expert's coverage rather than the distillation itself. Restricting
to the 18 scenes with training data does not rescue the result — the pooled figure is already
flat against base — but both slices should be reported, since the 24-scene number conflates two
distinct failures.

**Why it fails** is addressed in Section 4.6. The intuitive explanation — that the expert's
corrections depend on state the policy cannot observe — was tested and rejected.

---

## 4.5 What can the shield not reach?

The shield reduces collisions to 13.3% but not to zero. This section identifies what the residual
consists of, using per-body collision attribution recovered from the evaluation logs. Contacts
attributed to other scene objects are excluded throughout: that category fires on every episode
of every arm, including collision-free ones, because the obstacle resting on its supporting
surface registers a permanent contact.

| policy | episodes | collided | gripper | arm link | held object |
|---|---|---|---|---|---|
| base, no shield | 120 | 99 | 71 | 17 | 36 |
| base + shield | 120 | 16 | **0** | 15 | 0 |
| round 1, no shield | 120 | 23 | 7 | 13 | 9 |
| round 1 + shield | 120 | 10 | **0** | 8 | 1 |
| round 2, no shield | 120 | 21 | 8 | 8 | 7 |
| round 2 + shield | 120 | 11 | **0** | 9 | 0 |

**The shield eliminates the end-effector channel completely.** Gripper collisions fall from 71 to
zero, and across all 360 shielded episodes there is not a single one. Of the 16 residual
collisions in the shielded baseline, 15 involve an arm link.

This reframes the 13.3% floor. The barrier constrains end-effector spheres against obstacle
spheres; arm links and the carried object are not represented in it at all, while the collision
metric scores every body. The shield is therefore not leaking — it is working perfectly within an
end-effector-only scope, and the residual consists entirely of bodies it was never given
authority over. The corollary matters for reading Section 4.3: the distilled policy's 17.5%
shield-free is within a few points of the practical ceiling of the supervision it received,
rather than a degraded approximation of a perfect teacher.

**A second transfer channel.** Arm-link collisions fall from 17 to 8 between the base and
twice-distilled policies. The barrier cannot have caused this, since it does not constrain arm
links and the shielded baseline still shows 15. That improvement must come from the selection
criterion: demonstrations were retained only if nothing at all was displaced, arm links included,
and the student learned that. Safety therefore transfers through two distinguishable channels —
imitation of the filter's corrections where the filter has authority, and outcome selection where
it does not.

*Statistical caution:* 17/120 against 8/120 is 14.2% [9.0, 21.6] versus 6.7% [3.4, 12.7]. Those
intervals overlap, so the arm-link trend is suggestive rather than established. The gripper result
is not in doubt.

**Independent corroboration.** A separate experiment measured the end-effector's closest approach
to the obstacle during level-II episodes that were scored as collisions. The minimum was 0.231 m,
against a barrier radius of 0.15 m — the end-effector never came close, yet the episode collided.
Three independent measurements now point at the same conclusion.

---

## 4.6 What explains the improvement?

PENDING — control ablation running. Draft the confound framing, the two rejected hypotheses
(aliasing; generative uncertainty), and leave the control result as a gap to fill.

---

## 4.7 Are there other channels for safety?

Every mechanism examined so far acts on the action: a shield projects it, a guided sampler steers
it, behaviour cloning reshapes the policy that produces it. A vision-language-action model exposes
a channel none of these use, and one a classical controller does not possess at all — the
instruction itself. This section asks whether safety can be requested in language.

The experiment is minimal by design. The same checkpoint, the same seed, the same held-out initial
states and the same evaluation code were used for both arms; the only difference is a single clause
appended to the task instruction. Twelve level-II scenes were evaluated at five initial states
each, giving n = 60 per arm. Level II was chosen because collisions concentrate there, making it
the sensitive test.

The exact strings were, for a representative scene:

> **plain:** `pick up the black bowl between the plate and the ramekin and place it on the plate`
>
> **safety:** `pick up the black bowl between the plate and the ramekin and place it on the plate,
> being careful not to touch or knock over any other objects`

The suffix is reproduced verbatim because the result is a negative, and a negative about language
is only as strong as the language tested. The phrasing chosen is generic: it names no object, gives
no spatial reference, and states the constraint as a manner adverbial ("being careful not to")
rather than as a hard prohibition. Each of those is a design choice that could plausibly matter,
and none was varied.

| condition | n | TSR (95% CI) | collision (95% CI) | ETS |
|---|---|---|---|---|
| plain instruction | 60 | 66.7% [54.1, 77.3] | 70.0% [57.5, 80.1] | 132.4 |
| safety clause appended | 60 | 55.0% [42.5, 66.9] | 68.3% [55.8, 78.7] | 162.4 |

**The clause produces no safety benefit.** Collisions fall by 1.7 percentage points, with
confidence intervals that are almost coincident. Whatever the policy does in response to the
instruction, it does not avoid the obstacle more often.

**But the policy is not ignoring the text.** Execution time rises from 132.4 to 162.4 steps, a 23%
slowdown, and success falls by 11.7 points. The per-body attribution shows no meaningful
redistribution either (gripper 4 to 7, arm link 8 to 5, held object 5 to 6, all within noise). The
policy responds to the safety clause by acting more slowly and completing fewer tasks, without
touching anything less often.

The natural reading is that the safety language elicits a general behavioural prior — hesitancy,
reduced speed — rather than a grounded avoidance behaviour. The model has learned that
safety-flavoured instructions correspond to cautious-looking motion, but that association is not
connected to the geometry of the scene. It produces the appearance of caution without its
substance.

This is a characterised negative rather than an empty one, and the distinction matters: it
separates "the model ignores safety language" from "the model responds to safety language but does
not ground it in collision avoidance", and only the second tells you anything about what such
models have learned. It also bears directly on any pipeline that plans to encode operator
knowledge as natural-language annotations — safety intent expressed in language should not be
assumed to transfer into safe behaviour without verification.

*Caveats.* Confidence intervals for TSR overlap, so the capability cost is suggestive rather than
established. ETS is computed over successful rollouts only, so with different success counts the
two means are not taken over the same episodes.

Most importantly, **one phrasing was tested, and the result is a claim about that phrasing.** The
conclusion supported is that this generic clause does not ground; the conclusion NOT supported is
that language cannot carry a safety constraint. Several variations are untested and each attacks a
different hypothesis about why it failed:

- **Naming the obstacle** — `without touching the wine bottle` rather than "any other objects". If
  the failure is that the policy cannot resolve which objects are meant, naming should fix it, and
  the object names are available from the scene.
- **Spatial grounding** — `the wine bottle on your left`. Tests whether the model can bind a
  constraint to a referent it must locate visually, which is the harder and more useful case.
- **Prohibition rather than manner** — `do not touch anything else` instead of "being careful not
  to". The observed effect was a general slowdown, which is what a manner adverbial would predict;
  an imperative might elicit avoidance instead of hesitancy.
- **Positive rather than negative framing** — `keep clear of the other objects`. Negation is known
  to be handled poorly by language-conditioned models, and the tested phrasing contains two
  negations ("not to touch or knock over").

That last point is the one I would test first. The measured behaviour — slower, less successful,
equally collision-prone — is precisely what one would expect if the model registered the clause's
cautious register while failing to parse its negated content.

### CBF-guided sampling: steering generation rather than correcting it

PENDING — write up as a scope-limited negative. lambda=1 reproduces the projection endpoint exactly
on a synthetic barrier; on level-II scenes guidance never activates because the end-effector's
closest approach is 0.231 m against a 0.15 m barrier while collisions still occur. Guidance
inherits the end-effector scope of the barrier it steers by. Ties to Section 4.5.

### Best-of-N selection

PENDING — not yet run.

---

## 4.8 Two defects in the benchmark

PENDING — non-determinism from contact-buffer overflow (23% of episodes), and phantom collisions
from unsettled objects (102 mm). Both fixed; both affect anyone else's level-II numbers.
