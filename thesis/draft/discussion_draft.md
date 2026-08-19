# Chapter 5 — Discussion (DRAFT, ~1,000 words)

## 5.1 Why imitation succeeds where scalar-reward RL fails (~300 words)

Both approaches attempted in this work aim at the same target — move the policy toward the
behaviour the shield induces — and they differ in the signal used to do it.

Scalar-reward reinforcement learning communicates that target through a single number per
trajectory. The policy must infer, from a reward that fell, which of several hundred actions was
responsible. In a flow-matching policy the only lever available for that inference is the sampling
noise: exploration and credit assignment are the same knob. Increasing the noise enough to
distinguish good actions from bad also degrades the actions being evaluated, so the signal that
drives learning is the signal that destabilises it. This is not a tuning difficulty but a
structural one, and it is consistent with what was observed: the RL attempts in this project did
not converge to a policy that was both safer and competent.

Imitation replaces the scalar with a dense, per-step target. The shield does not report that a
trajectory was unsafe; it reports what should have been done instead, at every control step where
it intervened, in the same action space the policy emits. Credit assignment is therefore not
inferred but given. The learning signal is also on-distribution by construction: the corrected
actions were produced from states the policy itself visited, so the supervision does not require
the policy to be evaluated in regions it never reaches.

The measured contrast in Section 4.4 sharpens this. Cross-policy distillation, where the targets
were dense and per-step but came from a *foreign* expert, failed as completely as RL did. Dense
supervision is therefore not sufficient on its own — it must also be supervision at states the
policy actually reaches.

It is tempting to state this more strongly, as the claim that what transfers is correction of the
policy's own behaviour and never instruction from outside it. That claim would be false, and the
literature already falsifies it. SAFE-GIL trains a behaviour-cloning policy on demonstrations from
an MPC or PID expert — foreign controllers, not the learner — and obtains substantially safer
policies than vanilla behaviour cloning. What makes those demonstrations work is not the teacher's
identity but where they are collected: Hamilton-Jacobi reachability is used to compute the
disturbance that most steers the system toward failure, that disturbance is injected during
collection, and the fixed expert is thereby forced to demonstrate recovery from precisely the
states a learner's errors would produce. Their ablation confirms that this targeting is the active
ingredient, since Gaussian noise, uniform noise and DART all fail to reproduce the benefit.

IntervenGen reaches the same place by a different route. It rolls out the *current* policy in new
scene configurations so that the mistakes it records are that policy's genuine failures rather than
replayed ones, detects each mistake on contact, and transforms a *human* recovery segment onto the
state reached. The recovery actions are again foreign to the policy; only the states are its own.
Two methods, with different teachers — an optimal controller in one case, a human operator in the
other — and different means of arriving at the failure states, therefore locate the active
ingredient in the same place.

The correct reading of Section 4.4 is therefore narrower and more useful. The planner did not fail
because it was foreign. It failed because its trajectories were generated from its own initial
conditions and were never conditioned on where $\pi_{0.5}$ goes wrong, so the states it labelled
and the states the student needed labelled did not overlap. The shield succeeded because its
corrections are applied, by construction, at states the policy reached on its own. What the two
results share with SAFE-GIL is a single account: the operative variable is the *state distribution
the supervision covers*, and self-distillation is one way — not the only way — of guaranteeing that
coverage. Its practical advantage over the adversarial route is that no disturbance model is needed,
because the policy visits its own failure states unprompted.

---

## 5.2 The cost of internalised caution (~250 words)

[FIGURE 5.1 — Shield stacking. TSR and collision for the distilled policy with and without the shield
still running, against the base policy under both. Shows that re-applying the shield to a policy that
no longer needs it costs success, which is the concrete measurement behind AEGIS's own
"safety-induced distribution shift" and behind this section's argument. | figures/fig3_summary.pdf —
CHECK whether it already contains the stacked arm. | EXISTS-OR-MAKE figures/fig3_summary.pdf]

Internalised safety is not free, and this work measured its price in two forms.

The first is time. The distilled policy takes 157.3 control steps to complete a task against 138.6
for the base policy, roughly 14% slower. It takes a more conservative route, and on a benchmark
where speed is not scored, that cost is invisible in the headline metrics. In a production setting
with cycle-time targets it would not be.

The second is more interesting, because it is a cost of *combining* mechanisms rather than of
distillation itself. Applying the shield to the already-distilled policy improves safety further
(CAR 80.8% to 91.7%) but reduces success from 82.5% to 70.8%. The mechanism is visible in how the
shield works: it projects the proposed action onto the safe set, minimising deviation subject to
the barrier constraint. When the policy frequently proposes unsafe actions that projection is a net
benefit — which is exactly what Section 4.2 shows, where the shield *raises* base success from
58.3% to 71.7%. When the policy already avoids the obstacle, the same projection perturbs actions
that were already both safe and competent, displacing them from the distribution the policy was
trained to produce. The result is an action that satisfies the barrier and no longer executes a
well-formed grasp.

This effect is not peculiar to the setup here. The authors of the benchmark report the same
phenomenon independently, observing that enforcing safety with their layer "can drive the system
into out-of-distribution states" in which "the policy may behave erratically and fail to recover".
That an author-acknowledged limitation of runtime shielding is precisely what distillation removes
is, if anything, the clearest argument for internalisation: the distilled policy is safe without
ever being pushed off its own distribution.

Notably, the anticipated cost did not appear. A second distillation round neither improved nor
eroded performance, in contrast to an earlier DAgger-style experiment in this project that degraded
across rounds. One round was sufficient, and iterating was not harmful.

---

[FIGURE 5.2 — OPTIONAL. Per-scene scatter of collision rate before vs after distillation, one point
per scene, diagonal marked. Would show whether the improvement is uniform or driven by a subset,
which is the question a sceptical reader asks of any pooled 82.5% to 19.2%. |
figures/fig4_episode_scatter.pdf may already be close. | EXISTS-OR-ADAPT figures/fig4_episode_scatter.pdf]

## 5.3 Limitations (~300 words)

**Simulation only.** Every result is from SafeLIBERO. No physical robot was used and no claim of
sim-to-real transfer is made. The two benchmark defects documented in Section 4.8 are themselves a
reminder that a simulator's measurements are artefacts of its implementation as much as of the
policy under test.

**Privileged geometry.** The barrier is constructed from obstacle poses and meshes read directly
from the simulator. A deployed system would estimate that geometry from perception and inherit the
resulting error, so every shielded figure reported here is an upper bound for this class of filter.
The distilled policy, by contrast, uses no geometry at inference — which is a point in favour of
internalisation, but does not remove the dependency during *training*.

**Approximate robot model.** The barrier represents the end-effector with three fitted spheres and
each arm link with three point samples carrying a single radius, and infers link velocities through
a damped Jacobian pseudo-inverse rather than optimising them directly. Section 4.5 argues the
residual collisions concentrate where that model is coarsest; that argument is an interpretation of
the attribution data, not an independent measurement of constraint fidelity.

**Single training seed.** Each distilled policy is one training run and seed variance is
unquantified. A replication was attempted but could not be reconciled across compute environments,
and is reported rather than omitted. This is the weakest methodological point in the work.

**Single policy and action space.** One vision-language-action model, one flow-matching action
head, one operational-space action representation. The chunk-level and denoising-time results
depend on that architecture and do not obviously generalise to policies that emit single actions.

**Attribution caveats.** The per-body attribution is a documented lower bound that can miss delayed
or indirect contacts, and the scene-object category is excluded as an artefact rather than
interpreted.

---

## 5.4 Future work (~150 words)

The most direct extension is a **whole-body barrier with a faithful robot model** — proper link
geometry and directly constrained link velocities rather than a pseudo-inverse estimate. Section
4.5 predicts this would lower the shield's own floor; whether the improvement then distils is a
separate and testable question.

**Perception-grounded barriers** would remove the privileged-geometry dependency and measure how
much of the reported safety survives estimation error.

**Dynamic obstacles and human proximity** would test whether the same distillation holds when the
constraint moves, which is the case that matters for the deployment settings this work is motivated
by.

Finally, the boundary result invites a **mechanism study**: cross-policy distillation failed, and
observation aliasing was measured and rejected as the explanation. Distribution shift and
action-distribution mismatch remain, and separating them would say something general about when
demonstrations transfer between controllers.
