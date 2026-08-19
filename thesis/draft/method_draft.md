# Chapter 3 — Method (DRAFT)

<!-- MARKER CONVENTIONS USED IN THIS FILE
     [CITE: bibkey — what it is being cited *for*, plus verification status]
     [FIGURE X.Y — <title>. What it shows | where the data/asset comes from | STATUS]
       STATUS is one of: EXISTS <path>  |  TO MAKE  |  OPTIONAL
     [TABLE X.Y — ...] likewise. Every marker names the claim it supports, so a
     figure that turns out to support nothing can be cut without argument. -->

Plain prose. Describes only what Chapter 4 uses.

---

## 3.1 The SafeLIBERO benchmark

[TABLE 3.1 — The benchmark grid: 3 suites x 2 obstacle levels x 4 tasks = 24 scenes, 50 initial
states each, with the train/held-out split (0-34 / 35-49) shown explicitly. Cheap to produce and it
prevents the reader from having to reconstruct the design from prose. | From the suite definitions.
| TO MAKE]

SafeLIBERO extends the LIBERO manipulation benchmark by introducing an obstacle into each scene,
turning a pure task-completion problem into one where the policy must reach a goal without
disturbing its surroundings. Three task suites are used — spatial, object and goal — each
containing four tasks, and each task is instantiated at two safety levels that differ in how close
the obstacle sits to the natural path between the object and its target. This gives 24 distinct
scenes, and each provides 50 randomised initial states.

The robot is a 7-DOF Franka Panda under operational-space control. The policy emits seven-dimensional
actions — three translation deltas, three rotation deltas, and a gripper command — which the
controller converts to joint torques. Observations are two RGB images resized to 224x224 (a static scene camera and a wrist camera), an
eight-dimensional proprioceptive vector, and a natural-language instruction. The proprioceptive
vector follows the LIBERO convention used by the policy: end-effector position, axis-angle
orientation, and the two gripper joint positions.

That the proprioceptive vector contains no joint angles becomes relevant later. The shield has
access to the full configuration through the simulator and constrains the arm links accordingly
(Section 3.2); the policy imitating it does not, and must reproduce that behaviour from end-effector
pose and images alone.

### Metrics

**Task success rate (TSR)** is the benchmark's own success predicate.

**Collision** is recorded when the obstacle's position departs from its position at the start of
the episode by more than one millimetre, measured as the sum of absolute component differences,

    sum_k | p_k(t) - p_k(0) |  >  0.001 m

and the episode is marked collided if this holds at any step. The benchmark's originating work
defines collision avoidance as the proportion of *strictly collision-free* episodes without stating
a numeric tolerance; the one-millimetre L1 threshold used here is this work's operationalisation of
that definition, chosen to be strict enough that a genuine contact cannot pass it while remaining
above simulator noise. **Collision-avoidance rate (CAR)** is its complement.

Two properties of this definition matter. It is a **cumulative displacement** test against the
episode's initial pose, not a per-step contact test, so an obstacle nudged early stays flagged.
And it counts the obstacle being *moved* by any means: it does not require that the robot touched
it, so an object pushed into the obstacle by the carried payload counts. The reference pose is
taken after the settling procedure of Section 3.6, without which gravity alone would trip the
threshold.

**Execution time to success (ETS)** is the control step at which the success predicate first
holds, averaged over successful rollouts only. A failed rollout has no completion time, and
substituting the horizon would make a policy appear faster the more often it fails. Note that the
originating work uses the same name for a different quantity — mean episode length including
timeouts — so ETS values here are internally comparable but should not be read against theirs.

**CBF activation rate** is the fraction of control steps on which the shield altered the action,
counted when the projected action differs from the nominal one by more than 1e-4. It is an
independent signal of whether a policy has internalised avoidance: a policy that needs less
correction is behaving more safely, whatever the collision count says. Chapter 4 also reports a
related reward-term proxy where the per-step rate was not recorded.

### Evaluation protocol

Initial states 0–34 are used for demonstration collection and training. States 35–49 are held out
and used only for evaluation, so no reported figure is measured on a configuration the policy was
trained on. Every evaluation uses five held-out states per scene across all 24 scenes, giving 120
rollouts per evaluated policy. Rollouts run for 300 control steps; the policy is queried every
five steps and emits an action chunk from which the next five actions are executed. Sampling uses
ten denoising steps at noise level zero, making the sampler a deterministic ODE, so a rollout is
reproducible given its initial state.

Proportions are reported with 95% Wilson intervals rather than normal approximations, which behave
poorly at rates near zero and one for n = 120.

---

## 3.2 Obstacle geometry and the barrier

The obstacle's collision mesh is read from the simulator, sampled into a point cloud, and
decomposed into a small set of spheres. Sphere decomposition is used rather than a single bounding
volume because a bounding sphere over a non-convex object is a poor approximation — an earlier
version of this work measured a 5.5 cm carton as 16 cm across, which was enough to distort the
grasp it produced.

[FIGURE 3.1 — Sphere decomposition of the robot and an obstacle. Three panels: (a) a SafeLIBERO
obstacle mesh beside its sphere set, with the bounding-sphere alternative overlaid to show the
5.5 cm carton measured as 16 cm across; (b) the three-sphere end-effector model with radii and
offsets annotated; (c) the arm-link spheres on links 3-7 and the hand, in a representative
configuration. This is the most load-bearing figure in the chapter: Section 4.5's culprit analysis
is unreadable without it, and panel (c) is what distinguishes this barrier from AEGIS's
end-effector-only formulation. | Render from the MuJoCo model plus the decomposition code that
produces the sphere sets. | TO MAKE]

The robot is represented by spheres on both sides of the barrier.

**End-effector.** Three spheres in the end-effector frame approximate the Franka hand: a palm
sphere of radius 4.8 cm at 5.6 cm behind the grasp site, and two finger-pad spheres of radius 2.0 cm
offset 3.6 cm either side at 10.5 cm depth. A single sphere would either miss the fingers or
inflate the palm.

**Arm links.** Links 3 through 7 and the hand body are each assigned a radius (5.5 cm to 9.0 cm,
set from each link's maximum transverse extent plus a 15–21 mm margin) and sampled at three
positions along the link axis, so that a long link is not represented by its origin alone.

For each (robot sphere i, obstacle sphere j) pair the barrier is

    h_ij = || p_i - c_j ||^2 - ( r_i + r_j )^2

with h > 0 the safe set. Differentiating along the rigid-body velocity gives the constraint row

    [ 2 * scale * (p_i - c_j)^T R_1 ] u  +  k h_ij  >=  0,      k = 10

where u is the commanded end-effector velocity in the body frame and R_1 its rotation. For arm
links, whose velocity is not the commanded one, the mapping v_link ~= J_link J_ee^+ (scale R_1 u)
is used, with J_ee^+ a damped pseudo-inverse (lambda = 1e-3).

Three properties of this construction matter for interpreting Chapter 4, and are stated here rather
than discovered later.

*The robot model is approximate, and unevenly so.* The end-effector is covered by three spheres
fitted to the hand geometry; each arm link is covered by three point samples carrying one sphere
radius. The end-effector is therefore represented considerably more faithfully than the links.

*Links 1 and 2 are not constrained at all*, being close to the base and rarely near the obstacle in
these scenes.

*Arm-link velocities are estimated, not commanded.* The QP optimises an end-effector velocity, and
each link's motion is inferred through a damped Jacobian pseudo-inverse. Where that inverse is
ill-conditioned — near singularities, or when the null-space motion the OSC controller actually
applies differs from the minimum-norm solution — the predicted link velocity and the realised one
diverge, and the constraint is enforced against the prediction.

*The geometry is privileged.* Obstacle poses and sizes are read from the simulator rather than
estimated from the policy's observations, so the shield operates on information a deployed system
would have to perceive. The originating work instead derives obstacle geometry through
vision-language grounding and fused depth, and attributes its residual collisions primarily to that
upstream perception pipeline rather than to the controller. The figures reported here are therefore
an upper bound for this class of filter.

*Arm links are constrained here but not in the originating work*, whose formulation "solely
constrains the end-effector" and which notes that its unconstrained kinematic links may
consequently collide. Extending the barrier to the links is a deliberate difference from that
baseline, and Section 4.2 reports its effect.

---

## 3.3 The CBF shield

[FIGURE 3.2 — The shield as a block diagram. Observation into pi0.5, nominal action out, QP
projection against the active constraint rows, executed action to the simulator, and the corrected
action tapped off as the distillation target. One arrow should be marked as the difference between
this work and a runtime filter: the tap exists, and at deployment the QP block is deleted. If only
one diagram survives into the thesis, this is the one, because it carries the whole argument. |
Draw. | TO MAKE]

At each control step the policy's proposed translation is treated as a nominal end-effector
velocity u_nom, and the shield solves

    min_u || u - u_nom ||^2      s.t.   a_m^T u + b_m >= 0  for every constraint row m

a quadratic program in three variables whose rows are the end-effector/obstacle pairs and the
arm-link pairs of Section 3.2  [CITE: cbf_theory — cite the *formulation*: safety as forward
invariance of a superlevel set, enforced as an affine inequality in the control, with the QP
returning the minimum-norm admissible departure from the nominal action. This is the standard form
being instantiated, so cite it here rather than deriving it. CITE: collision_cbf_ellipsoidal — cite
only if the sphere-pair distance construction of 3.2 is presented as following theirs; if the
spheres are this work's own simplification, say so and do not cite it as a source]. When no constraint is active the solution is u_nom and the action
passes through unchanged. The problem is solved with OSQP through CVXPY  [CITE: osqp, cvxpy — software citations, needed for
reproducibility; neither is a claim]; where CVXPY is unavailable
the implementation falls back to SLSQP and, if that fails, to the unconstrained action, emitting a
warning — a silent fallback would be indistinguishable from a shield that was never needed.

An intervention is recorded when the returned action differs from the nominal by more than 1e-4,
which is what the CBF activation rate of Section 3.1 counts.

Two discrete-time adjustments are required because the barrier's guarantee is continuous-time. The
effective radius is inflated by a lag buffer covering the distance travelled between control steps
and the inverse-kinematics tracking error; this value was tuned upward during development after
collisions were observed at nominally safe margins. Rotation is not constrained: the barrier acts
on the translational channel only.

## 3.4 Collision attribution

The headline collision metric answers *whether* the obstacle moved, not *what* moved it. Because
the barrier constrains only the end-effector, the distinction is essential to interpreting the
residual, so each episode additionally records which bodies contacted the obstacle, grouped as
gripper, arm link, held object, and other scene objects.

Two cautions apply to this attribution and are observed throughout Chapter 4. Contacts with other
scene objects fire on essentially every episode, including collision-free ones, because an obstacle
resting on its supporting surface registers a permanent contact; that category is therefore
excluded from analysis rather than interpreted. And the contact-graph attribution is a documented
lower bound — it can miss delayed or indirect pushes — so demonstration filtering uses the raw
displacement flag rather than the attributed one, which is the stricter choice.

---

## 3.5 Teachers

Two sources of demonstrations are compared, and the comparison between them is the central
experiment of this work.

### 3.5.1 The shielded policy as its own teacher

The first teacher is the policy itself, running under the shield. Rollouts are collected across the
training initial states and retained if they satisfy three criteria: the episode succeeded, the
obstacle was never displaced, and the shield intervened at least once. The third criterion matters —
a clean episode in which the barrier never engaged is ordinary base-policy behaviour and carries no
safety signal, so imitating it would degenerate into copying the policy that is already being
trained.

### 3.5.2 A privileged scripted expert

[CITE: liu2023libero — cite for LIBERO, the task definitions and the BDDL scene specification that
the planner reads object and goal poses from. CITE: rrt_connect — cite the planner algorithm itself.]

The second teacher is a sampling-based motion planner with full access to scene geometry:
RRT-Connect in joint space with clearance inflation applied to the robot's links and to any carried
object, followed by inverse kinematics and playback as operational-space deltas. Unlike the
shielded policy, this expert is safe by construction rather than by correction, and it represents
the natural expectation that a purpose-built classical controller should provide better
demonstrations than a corrected neural policy.

Its behaviour is characterised in its own right in Chapter 4, because whether that expectation holds
is itself a result.

---

## 3.6 Distillation

[FIGURE 3.3 — The two teachers and the two data paths, side by side. Left: shielded self-distillation
— pi0.5 rolls out, the shield corrects, the corrected actions at the states the policy reached become
targets. Right: the privileged planner — RRT-Connect plans from its own initial conditions and
produces trajectories at states the policy may never visit. The figure should make the state-coverage
difference visible, since that is the mechanism Chapters 4 and 5 argue for and it is far easier to
see than to read. | Draw, optionally overlaying real end-effector traces from the two demo sets to
show the distributions genuinely differ. | TO MAKE — high value, this is the figure that explains the
thesis's central negative result.]

Retained demonstrations are behaviour-cloned using the model's native flow-matching loss  [CITE:
physicalintelligence2025pi05 — cite for the *architecture and objective*: PaliGemma backbone, the
flow-matching action expert, and action chunking, all of which are inherited rather than designed
here. CITE: pi0 — cite for the flow-matching action-expert formulation that pi0.5 derives from]. Only the
parameters selected by the training configuration's trainable filter receive gradients — a LoRA
adapter on the action head — while the vision-language backbone is frozen; the saved artefact is
the adapter alone, reapplied over the base checkpoint at load time. The imitation target is the sequence of actions actually executed — that is,
post-shield — so the policy learns the corrected behaviour rather than its own original proposals.
Actions are normalised by passing them through the policy's own input transform, so the target
matches the distribution the model was pretrained on rather than a hand-rolled normalisation.

### Comparing training runs

Where two distilled policies are compared they are matched on **gradient steps**, not epochs. This
is necessary because episode length differs between teachers: a shielded rollout at horizon 300
yields roughly 35 training examples, while a planner episode at horizon 900 yields roughly 412.
Matching epochs would have given one policy 16,140 gradient steps and the other approximately
212,000, which is not a controlled comparison but a different experiment. The two students compared
in Chapter 4 were trained for 16,140 and 15,988 steps respectively.

### Two corrections to the benchmark

Two defects in the simulation had to be fixed before any measurement was trustworthy; both are
reported with their evidence in Chapter 4. Unused obstacles are parked off-scene by SafeLIBERO in a
heap dense enough to overflow MuJoCo's contact buffer, and on overflow contacts are discarded
order-dependently, making episodes non-reproducible. Separately, objects spawned in unsupported
poses fall under gravity while the arm is stationary, and that displacement is charged to the
robot. The fixes are to clear collision flags on parked bodies and to allow the scene to settle
before measurement begins — in that order, since settling before the first fix reintroduces the
non-determinism it is meant to precede.
