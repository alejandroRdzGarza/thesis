# Chapter 3 — Method (DRAFT)

Plain prose. Describes only what Chapter 4 uses.

---

## 3.1 The SafeLIBERO benchmark

SafeLIBERO extends the LIBERO manipulation benchmark by introducing an obstacle into each scene,
turning a pure task-completion problem into one where the policy must reach a goal without
disturbing its surroundings. Three task suites are used — spatial, object and goal — each
containing four tasks, and each task is instantiated at two safety levels that differ in how close
the obstacle sits to the natural path between the object and its target. This gives 24 distinct
scenes, and each provides 50 randomised initial states.

The robot is a 7-DOF Franka Panda under operational-space control. The policy emits seven-dimensional
actions — three translation deltas, three rotation deltas, and a gripper command — which the
controller converts to joint torques. Observations are two 224x224 RGB images (a static scene
camera and a wrist camera), an eight-dimensional proprioceptive vector comprising end-effector
position, axis-angle orientation and gripper state, and a natural-language instruction.

That the proprioceptive vector contains no joint angles becomes relevant later: any safety
behaviour that depends on the arm's configuration rather than its end-effector pose is conditioned
on information the policy does not receive.

### Metrics

**Task success rate (TSR)** is the benchmark's own success predicate.

**Collision** is defined as displacement of the obstacle by more than one millimetre over the
episode, which is the quantity reported by AEGIS and is adopted here for comparability.
**Collision-avoidance rate (CAR)** is its complement. This definition is deliberately strict and
worth stating plainly: it counts the obstacle being *moved*, by any means, and does not require
that the robot touched it directly. An object knocked into the obstacle by the carried payload
counts.

**Execution time to success (ETS)** is the mean number of control steps to completion, computed
over successful rollouts only. A failed rollout has no completion time, and including the horizon
would make a policy appear faster the more often it fails.

**CBF activation rate** records how frequently the shield intervened, and serves as an independent
signal of whether a policy has internalised avoidance: a policy that needs less correction is
behaving more safely, regardless of what the collision count says.

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

The obstacle's collision geometry is taken from the simulator as a mesh, from which a point cloud
is sampled and decomposed into a small set of spheres. The barrier is then defined pairwise between
end-effector spheres and obstacle spheres:

    h = || c_ee - c_obs ||^2 - ( r_ee + r_obs )^2

with h > 0 denoting the safe set. Sphere decomposition is used rather than a single bounding volume
because a bounding sphere over a non-convex object is a poor approximation — an early version of
this work measured a 5.5 cm carton as 16 cm across, which was enough to distort the grasp.

Two consequences of this construction matter for the results and are stated here rather than
discovered later. First, the barrier is defined **over the end-effector only**: arm links and any
carried object do not appear in it, while the collision metric of Section 3.1 scores every body.
Second, the geometry is **privileged** — read directly from the simulator rather than estimated from
the policy's observations — so the shield operates with information a deployed system would have to
perceive.

---

## 3.3 The CBF shield

At each control step the policy's proposed action is treated as a nominal end-effector velocity and
projected onto the safe set by solving a quadratic program that minimises deviation from the
nominal action subject to the barrier's decrease condition. When the nominal action is already safe
the constraint is inactive and the action passes through unchanged; when it is not, the QP returns
the nearest action satisfying the constraint.

Two practical adjustments were required. The barrier is enforced in discrete time while the
guarantee is continuous-time, so a lag buffer inflates the effective radius to account for the
distance travelled between control steps and for inverse-kinematics tracking error. Where the QP
solver is unavailable the implementation falls back to an unconstrained action rather than
failing; this is recorded, because a silent fallback would otherwise be indistinguishable from a
shield that never needed to act.

---

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

Retained demonstrations are behaviour-cloned into the policy's action head using the model's native
flow-matching loss, with the vision-language backbone frozen and a LoRA adapter trained on the
action head alone. The imitation target is the sequence of actions actually executed — that is,
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
