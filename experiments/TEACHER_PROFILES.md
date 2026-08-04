# Per-scene teachers for demo collection

## Why

Distillation needs demos that are both **successful** and **collision-free**. The classical
MPC-CBF expert produces those — but only where its single universal config happens to suit the
scene. The last full-grid sweep (`sweep_final/per_config.csv`, 24 scenes × 4 inits) shows the
gap plainly:

| suite | success |
|---|---|
| `safelibero_object` | ~87% |
| `safelibero_spatial` | ~38% |
| `safelibero_goal` | ~9% |

Overall 44.8% success, 7.3% collision. Roughly half the grid produces no usable demos at all.

The failures are **not** random noise — they cluster by scene geometry. Every one of the 16
`safelibero_goal` level-I episodes ended frozen in `APPROACH`, arm parked ~0.22 m from the
obstacle, having never touched the object.

## The finding that motivated this

A wide bowl doesn't fit the 8 cm gripper, so it's grasped by **pinching the rim**: the grip point
is offset ~5 cm along the gripper's closing axis so one finger drops inside the bowl and one
outside. The gripper keeps its reset orientation (the controller commands zero rotation), so that
axis is world **Y** — and the offset was hard-coded to **+Y**.

On every `safelibero_goal` LI scene the obstacle sits on the **+Y** side of the object. The teacher
was therefore instructed to reach 5 cm *towards* the obstacle. The barrier refused, correctly, and
the arm hovered until the horizon expired.

Flipping the side to −Y on those four tasks: **0/12 → 6/12 success, 0 collisions.**

Nothing about the controller was wrong. The *scene* decides which side is reachable, and one
global constant cannot encode that.

## Design

Two layers, resolved per episode in `run_libero_trial`.

### 1. Geometry-driven (`teacher_profiles.derive_profile`)

Rules keyed to measurable scene properties, never to a task index:

| rule | trigger | effect |
|---|---|---|
| grasp side | rim grasp | pinch the side whose grip point **and** hover point clear the obstacles by more; ties keep +Y so working scenes don't move |
| crowded corridor | grip-point clearance < `radius_buffer + activate_margin` | drop the planner's extra keep-out buffer, lower the hover height |
| elevated pick | object z > 1.00 (stove / cabinet) | short lift, slower descent, more contact patience |
| elevated placement | goal z > 1.00 | lower carry height, gentler set-down |
| long transport | > 0.35 m | wider carry margin for the held object |
| obstacle near goal | goal clearance < `radius_buffer + activate_margin`, surface placement | tighter release centring, gentler set-down |

### 2. Tuned per-task overrides (`teacher_profiles.json`)

For what geometry can't predict. Written by `tune_teacher.py`, layered on top of the geometric
strategy — `flip_side` in particular is expressed *relative* to the geometric choice, so an
override stays meaningful if the scene or the clearance rule changes.

## Safety is unchanged

Every one of these knobs is on the **planner** (the MPC nominal). The reactive ellipsoid CBF that
provides the hard guarantee is untouched, runs every step, and still filters the executed action.
Relaxing the MPC buffer only stops the planner refusing corridors the barrier itself permits — a
profile cannot make a demo unsafe, only make one exist.

One related planner fix went in alongside: a keep-out sphere that **swallows the target** makes the
QP infeasible, which surfaces as the arm parked on the boundary. Such spheres are now shrunk to
just inside the target.

## Running it

```bash
PY=/Users/alexrdzgarza/miniforge3/envs/libero/bin/python
export PYTHONPATH=.

# 1. Where does the teacher stand today? (24 scenes × 4 inits, ~4 h on the Mac)
$PY -m experiments.sweep_classical_expert --levels I II --out-dir sweep_profiles --horizon 300

# 2. Tune whatever it still fails. Phase-directed: only knobs that could fix the phase the
#    failures die in are tried, and the bottleneck is recomputed after each improvement.
$PY -m experiments.tune_teacher --from-sweep sweep_profiles/per_config.csv --max-success 0.75

# …or one scene at a time
$PY -m experiments.tune_teacher --suites safelibero_goal --levels II --tasks 1 --episodes 0 1 2

# 3. Re-sweep to confirm, then collect demos — profiles are picked up automatically
$PY -m experiments.collect_classical_demos --suite safelibero_goal --level I \
    --tasks 0 1 2 3 --episodes 0 1 2 3 4 5 6 7 --out results_distill/round0
```

Every rollout prints the profile it resolved:

```
[teacher] auto+tight [grasp_offset=[0. -0.05 0.]  mpc=radius_buffer=0.0]
          (rim grasp on −Y (clearance 0.186 m); tight grasp corridor → MPC buffer relaxed, CBF unchanged)
```

`tune_teacher` checkpoints each winner to `teacher_profiles.json` as it finds it, so an
interrupted run keeps its progress.

## What this does and doesn't change for the thesis

The student is still **one** VLA trained on **one** demo format. It never sees which teacher
produced a demo — only (observation → safe action). Specialising the teacher raises how many
clean, safe demos exist to distil; it does not make the distillation itself per-task.

Known limit, unchanged by this work: `safelibero_spatial` t2/t3 (bowls wedged on a stove and in a
cabinet) remain unsolved. That is an OSC_POSE action-space limitation — the arm configuration
needed to avoid the obstacle while grasping isn't expressible through end-effector-only control —
not a config that tuning can find. See the per-task-teacher notes.
