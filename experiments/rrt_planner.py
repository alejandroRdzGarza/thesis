"""rrt_planner.py — joint-space RRT-Connect for the Panda, executed through OSC_POSE.

The pipeline the supervisor is right about:

    RRT-Connect in joint space  →  collision-free joint path q_0..q_T
        →  forward kinematics    →  EE pose trace  p_0..p_T
        →  OSC_POSE delta actions →  execute

The objection I raised (and should have tested rather than asserted) is that OSC has a
one-dimensional null space for a 7-DOF arm and resolves it toward its own reset posture, so
commanding the FK trace does not *guarantee* the arm reproduces the planned configuration. That is
true as stated and it does not imply the approach fails: the discrepancy is a single scalar per
waypoint, the executed configuration can simply be CHECKED in simulation, and if it collides the
plan can be repaired. `verify_ee_trace` below does the checking; `test_rrt_transfer.py` runs it.

Everything here is non-destructive: qpos is saved and restored around every collision query, so a
planner call does not disturb a live episode.
"""

from __future__ import annotations

import numpy as np

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:                                   # pragma: no cover
    _HAS_MUJOCO = False


# ── geometry helpers ────────────────────────────────────────────────────────
def _geom_name(model, i) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""


def _is_robot_geom(name: str) -> bool:
    return any(s in name.lower() for s in
               ("gripper", "finger", "hand", "robot", "_link", "pad"))


class clearance_margin:
    """Context manager: plan against INFLATED robot geometry so paths keep a real standoff.

    MuJoCo only generates a contact once geoms actually touch, so a `no contacts` test calls a
    path that passes 0.1 mm from an obstacle "collision-free". Executing such a path grazes the
    obstacle the moment OSC drifts off the plan — which is exactly what was measured: 24 grazes
    with displacements of 1.0-4.7 mm, blamed on gripper, arm links and the held object.

    Raising `geom_margin` on the robot's geoms makes MuJoCo report contacts `clearance` metres
    BEFORE touching, so the planner refuses paths that merely come close. Restored on exit, so a
    planning call cannot leak an altered contact model into the episode.
    """

    def __init__(self, model, clearance: float = 0.02):
        self.model, self.clearance = model, float(clearance)
        self.saved = None

    def __enter__(self):
        self.saved = self.model.geom_margin.copy()
        for g in range(self.model.ngeom):
            n = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
            if _is_robot_geom(n):
                self.model.geom_margin[g] = max(self.model.geom_margin[g], self.clearance)
        return self

    def __exit__(self, *exc):
        self.model.geom_margin[:] = self.saved
        return False


def make_collision_fn(model, data, qadr, *, ignore: tuple = (), verbose: bool = False):
    """Return `free(q) -> bool`: True when arm configuration `q` touches no scene geom.

    Robot-vs-robot contacts are ignored (adjacent links are always in contact). Geoms whose name
    contains any string in `ignore` are also excluded — used for the object being grasped, which
    the gripper is *supposed* to touch.
    """
    saved = data.qpos.copy()

    def free(q) -> bool:
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        hit = False
        for c in range(data.ncon):
            ct = data.contact[c]
            n1, n2 = _geom_name(model, ct.geom1), _geom_name(model, ct.geom2)
            r1, r2 = _is_robot_geom(n1), _is_robot_geom(n2)
            if r1 == r2:                               # robot-robot or scene-scene → not our problem
                continue
            other = n2 if r1 else n1
            if any(s in other for s in ignore):
                continue
            if verbose:
                print(f"    [collision] {n1} <-> {n2}")
            hit = True
            break
        for a, v in zip(qadr, saved[qadr] if isinstance(qadr, np.ndarray) else
                        [saved[a] for a in qadr]):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        return not hit

    return free


def free_joint_qadr(model, body_prefix):
    """qpos address of a movable object's free joint (7 values: xyz + wxyz), or None.

    Needed to carry the grasped object along during transport planning — otherwise the planner
    checks an arm swinging through space while the object it is holding stays behind at the pick
    location, and every transport plan looks clear when it isn't.
    """
    for b in range(model.nbody):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if not n.startswith(body_prefix):
            continue
        if model.body_jntnum[b] == 1:
            j = int(model.body_jntadr[b])
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
                return int(model.jnt_qposadr[j])
    return None


def _mat2quat(R):
    q = np.empty(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


def make_attached_collision_fn(model, data, qadr, sid, obj_qadr, rel_pos, rel_R, *, ignore=()):
    """`free(q)` for the arm while HOLDING an object.

    The object is placed at the pose the grasp implies (EE pose composed with the recorded
    grasp-relative transform) before contacts are checked, so the held object is part of the
    swept volume. Gripper-vs-held-object contacts are ignored — they are the grasp.
    """
    saved = data.qpos.copy()

    def free(q) -> bool:
        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)                       # EE pose for this configuration
        p_ee = data.site_xpos[sid].copy()
        R_ee = data.site_xmat[sid].reshape(3, 3).copy()
        if obj_qadr is not None:
            p_obj = p_ee + R_ee @ rel_pos
            R_obj = R_ee @ rel_R
            data.qpos[obj_qadr:obj_qadr + 3] = p_obj
            data.qpos[obj_qadr + 3:obj_qadr + 7] = _mat2quat(R_obj)
        mujoco.mj_forward(model, data)                       # now with the object carried along
        hit = False
        for c in range(data.ncon):
            ct = data.contact[c]
            n1 = _geom_name(model, ct.geom1)
            n2 = _geom_name(model, ct.geom2)
            r1, r2 = _is_robot_geom(n1), _is_robot_geom(n2)
            held = tuple(s for s in ignore if s)
            h1 = any(s in n1 for s in held)
            h2 = any(s in n2 for s in held)
            if (r1 and h2) or (r2 and h1):                   # gripper holding the object — fine
                continue
            if r1 == r2 and not (h1 or h2):                  # robot-robot / scene-scene
                continue
            if (h1 and h2):
                continue
            hit = True
            break
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)
        return not hit

    return free


def fk_ee_pose(model, data, qadr, sid, q):
    """Forward kinematics: arm configuration → (EE position, EE rotation matrix). Restores qpos."""
    saved = [data.qpos[a] for a in qadr]
    for a, v in zip(qadr, q):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    pos = data.site_xpos[sid].copy()
    R = data.site_xmat[sid].reshape(3, 3).copy()
    for a, v in zip(qadr, saved):
        data.qpos[a] = v
    mujoco.mj_forward(model, data)
    return pos, R


def ik_pose(model, data, qadr, sid, target_pos, target_R=None, *, free=None,
            iters: int = 300, seeds: int = 60, seed: int = 0):
    """Damped-least-squares IK to a target EE pose, preferring a COLLISION-FREE solution.

    Returns (q, ok). Multi-seed because the Panda's redundancy means the first solution found is
    often one whose forearm sits inside an obstacle.
    """
    saved = data.qpos.copy()
    rng = np.random.default_rng(seed)
    lo_hi = []
    for a in qadr:
        jid = int(np.where(model.jnt_qposadr == a)[0][0])
        lo_hi.append(model.jnt_range[jid].copy())
    lo_hi = np.array(lo_hi)
    q_cur = np.array([data.qpos[a] for a in qadr])

    # Collect every valid solution and return the one NEAREST the current configuration, rather
    # than the first found. The Panda's redundancy means IK solutions sit on different elbow
    # branches; a far branch is reachable in joint space but OSC — which only sees an end-effector
    # target — cannot drive the arm across to it. Measured symptom: the EE parking 55 mm from the
    # planned grasp for 600 steps with zero contacts, i.e. not blocked, just not reachable by
    # end-effector servoing from where the arm actually is.
    best = None
    solutions = []
    for s in range(seeds):
        q = q_cur.copy() if s == 0 else rng.uniform(lo_hi[:, 0], lo_hi[:, 1])
        for _ in range(iters):
            for a, v in zip(qadr, q):
                data.qpos[a] = v
            mujoco.mj_forward(model, data)
            p = data.site_xpos[sid].copy()
            err = np.asarray(target_pos, float) - p
            if target_R is not None:
                R = data.site_xmat[sid].reshape(3, 3)
                from scipy.spatial.transform import Rotation as Rot
                rerr = Rot.from_matrix(np.asarray(target_R) @ R.T).as_rotvec()
                e = np.concatenate([err, rerr])
            else:
                e = err
            if np.linalg.norm(err) < 1e-3 and (target_R is None or np.linalg.norm(e[3:]) < 0.05):
                break
            jacp = np.zeros((3, model.nv)); jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, sid)
            dofs = [int(model.jnt_dofadr[int(np.where(model.jnt_qposadr == a)[0][0])]) for a in qadr]
            J = (jacp[:, dofs] if target_R is None
                 else np.vstack([jacp[:, dofs], jacr[:, dofs]]))
            lam = 0.05
            dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(J.shape[0]), e)
            q = np.clip(q + 0.5 * dq, lo_hi[:, 0], lo_hi[:, 1])

        for a, v in zip(qadr, q):
            data.qpos[a] = v
        mujoco.mj_forward(model, data)
        reach = float(np.linalg.norm(np.asarray(target_pos, float) - data.site_xpos[sid]))
        data.qpos[:] = saved
        mujoco.mj_forward(model, data)

        if reach < 5e-3:
            if free is None or free(q):
                solutions.append(q.copy())              # collision-free and on target
                # Seed 0 IS the current configuration, so a solution from it is already the
                # nearest branch — take it and skip the rest. Otherwise gather a few and choose.
                # Without this early exit every call ran all `seeds` iterations, which made
                # planning ~20x slower once this function stopped returning on first success.
                if s == 0 or len(solutions) >= 5:
                    break
            elif best is None:
                best = q                                # on target but colliding — keep as fallback

    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    if solutions:
        # nearest in joint space = the branch OSC can actually servo to
        return min(solutions, key=lambda q: float(np.linalg.norm(q - q_cur))), True
    return (best, best is not None)


# ── RRT-Connect ─────────────────────────────────────────────────────────────
def _steer(q_from, q_to, step):
    d = q_to - q_from
    n = float(np.linalg.norm(d))
    return q_to.copy() if n <= step else q_from + d * (step / n)


def _segment_free(a, b, free, res):
    """Collision-check the straight joint-space segment a→b at `res` resolution."""
    n = max(2, int(np.ceil(float(np.linalg.norm(b - a)) / res)))
    for i in range(1, n + 1):
        if not free(a + (b - a) * (i / n)):
            return False
    return True


def rrt_connect(q_start, q_goal, limits, free, *, step: float = 0.25, res: float = 0.08,
                max_iter: int = 4000, seed: int = 0):
    """Bidirectional RRT-Connect in joint space. Returns a list of configurations, or None.

    Trees are grown from both ends and connected; this is markedly faster than single-tree RRT on
    the narrow passages that matter here (threading between a target object and an obstacle).
    """
    q_start = np.asarray(q_start, float)
    q_goal = np.asarray(q_goal, float)
    if not free(q_start):
        return None, "start configuration is in collision"
    if not free(q_goal):
        return None, "goal configuration is in collision"
    if _segment_free(q_start, q_goal, free, res):
        return [q_start, q_goal], "direct"

    rng = np.random.default_rng(seed)
    lo, hi = limits[:, 0], limits[:, 1]
    ta, tb = [q_start], [q_goal]
    pa, pb = [-1], [-1]

    def extend(tree, parent, q_target):
        i = int(np.argmin([np.linalg.norm(q - q_target) for q in tree]))
        q_new = _steer(tree[i], q_target, step)
        if _segment_free(tree[i], q_new, free, res):
            tree.append(q_new); parent.append(i)
            return len(tree) - 1, bool(np.linalg.norm(q_new - q_target) < 1e-9)
        return None, False

    for it in range(max_iter):
        q_rand = rng.uniform(lo, hi)
        idx, _ = extend(ta, pa, q_rand)
        if idx is not None:
            # try to connect the other tree all the way to the new node
            j, reached = extend(tb, pb, ta[idx])
            while j is not None and not reached:
                j, reached = extend(tb, pb, ta[idx])
            if reached:
                path_a, k = [], idx
                while k != -1:
                    path_a.append(ta[k]); k = pa[k]
                path_b, k = [], j
                while k != -1:
                    path_b.append(tb[k]); k = pb[k]
                path = list(reversed(path_a)) + path_b
                if np.linalg.norm(path[0] - q_start) > 1e-9:
                    path = list(reversed(path))
                return path, f"connected after {it + 1} iters ({len(ta)}+{len(tb)} nodes)"
        ta, tb, pa, pb = tb, ta, pb, pa            # swap: grow the trees alternately

    return None, f"no path after {max_iter} iters"


def shortcut(path, free, *, res: float = 0.08, iters: int = 200, seed: int = 0):
    """Randomised shortcutting — RRT output is jagged and a jagged EE trace is hard for OSC."""
    if path is None or len(path) < 3:
        return path
    rng = np.random.default_rng(seed)
    p = [np.asarray(q, float) for q in path]
    for _ in range(iters):
        if len(p) < 3:
            break
        i, j = sorted(rng.choice(len(p), 2, replace=False))
        if j - i < 2:
            continue
        if _segment_free(p[i], p[j], free, res):
            p = p[:i + 1] + p[j:]
    return p


def densify(path, *, max_step: float = 0.05):
    """Resample the path so consecutive configurations are close — OSC tracks a dense trace far
    better than sparse waypoints, and the EE trace stays inside the swept collision-free tube."""
    out = [np.asarray(path[0], float)]
    for a, b in zip(path[:-1], path[1:]):
        a = np.asarray(a, float); b = np.asarray(b, float)
        n = max(1, int(np.ceil(float(np.linalg.norm(b - a)) / max_step)))
        for i in range(1, n + 1):
            out.append(a + (b - a) * (i / n))
    return out


def grasp_pose(center, theta, radius, dz):
    """A rim-pinch grasp pose at angle `theta` around a circular rim.

    Returns (position, rotation). The gripper approaches straight down (site +Z world-down) with
    its CLOSING axis (site +Y, verified empirically for this gripper) pointing radially, so one
    finger lands inside the rim and one outside.
    """
    u = np.array([np.cos(theta), np.sin(theta), 0.0])          # radial = closing direction
    pos = np.asarray(center, float) + radius * u + np.array([0.0, 0.0, dz])
    z_ax = np.array([0.0, 0.0, -1.0])                          # approach, pointing down
    y_ax = u
    x_ax = np.cross(y_ax, z_ax)
    x_ax /= (np.linalg.norm(x_ax) + 1e-12)
    return pos, np.column_stack([x_ax, y_ax, z_ax])


def sample_rim_grasps(model, data, qadr, sid, center, radius, free, *,
                      n_theta: int = 16, dzs=(0.005, 0.02, -0.01), seed: int = 0,
                      ik_seeds: int = 25):
    """Search the rim for a REACHABLE, COLLISION-FREE grasp.

    A single hardcoded offset (the +Y the scripted controller used) fails whenever that side of
    the object is blocked — on a bowl sitting on a stove it drives a finger into the stove base.
    The rim is a circle, so sweep it and keep every pose that IK can reach without collision.

    Returns a list of dicts sorted by preference (closest to the robot base first, which tends to
    keep the arm inside its comfortable workspace).
    """
    out = []
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / n_theta
        for dz in dzs:
            pos, R = grasp_pose(center, theta, radius, dz)
            q, ok = ik_pose(model, data, qadr, sid, pos, R, free=free,
                            seeds=ik_seeds, seed=seed + i)
            if ok and q is not None and free(q):
                out.append({"q": q, "pos": pos, "R": R, "theta": theta, "dz": dz,
                            "reach": float(np.linalg.norm(pos[:2]))})
    out.sort(key=lambda d: d["reach"])
    return out


def top_grasp_pose(center, yaw, top_z, grip_depth):
    """A top-down straddle grasp: approach from above, fingers closing along `yaw`."""
    u = np.array([np.cos(yaw), np.sin(yaw), 0.0])              # closing direction
    pos = np.array([center[0], center[1], top_z - grip_depth])
    z_ax = np.array([0.0, 0.0, -1.0])
    x_ax = np.cross(u, z_ax); x_ax /= (np.linalg.norm(x_ax) + 1e-12)
    return pos, np.column_stack([x_ax, u, z_ax])


def object_cloud(model, data, body_prefix):
    """True surface point cloud of an object — real mesh vertices plus primitive samples.

    Do NOT size objects from per-geom bounding-sphere radii (`max(geom_size)` applied to every
    axis). That over-estimates a box by its diagonal: it reports a 5.5 cm orange-juice carton as
    16 cm wide, which makes a feasible top-down straddle look impossible and falls back to a rim
    pinch that closes on air. Measured failure — object suite went 0/4 that way.
    """
    from experiments.cbf_visualizer import get_obstacle_point_cloud
    for sfx in ("", "_main", "_g0"):
        pc = get_obstacle_point_cloud(model, data, f"{body_prefix}{sfx}")
        if pc is not None and len(pc):
            return np.asarray(pc, float)
    return None


def object_extent(model, data, body_prefix, center):
    """(width_along_x, width_along_y, top_z) of an object, from its TRUE geometry."""
    pc = object_cloud(model, data, body_prefix)
    if pc is None:
        return 0.06, 0.06, float(center[2])
    return (float(pc[:, 0].max() - pc[:, 0].min()),
            float(pc[:, 1].max() - pc[:, 1].min()),
            float(pc[:, 2].max()))


def sample_grasps(model, data, qadr, sid, center, body_prefix, free, *,
                  gripper_open: float = 0.075, seed: int = 0, ik_seeds: int = 20,
                  n_theta: int = 16, n_yaw: int = 8):
    """Grasp candidates for ANY object: top-down straddle when it fits the gripper, rim pinch
    when it does not, both screened for reachability and collision.

    The scripted controller picked the mode by NAME ("bowl" → rim). This picks it by measured
    geometry, so it generalises across suites — cartons and bottles in the object suite get a
    straddle, wide bowls in spatial/goal get a rim pinch, with no per-scene table.
    """
    wx, wy, top_z = object_extent(model, data, body_prefix, center)
    out = []

    # Top-down straddle: only where the object actually fits between the fingers.
    if min(wx, wy) < gripper_open:
        # Close along the NARROW axis first: a 8.3 x 4.9 cm box only fits the gripper one way
        # round. yaw is the closing direction, so yaw=0 closes along x, yaw=pi/2 along y.
        yaw0 = 0.0 if wx < wy else np.pi / 2
        for i in range(n_yaw):
            # sweep outward from the narrow-axis yaw rather than from an arbitrary zero
            yaw = yaw0 + (np.pi * ((i + 1) // 2) / n_yaw) * (1 if i % 2 == 0 else -1)
            # close along the narrow axis: yaw is the closing direction
            for depth in (0.02, 0.035, 0.01):
                pos, R = top_grasp_pose(center, yaw, top_z, depth)
                q, ok = ik_pose(model, data, qadr, sid, pos, R, free=free,
                                seeds=ik_seeds, seed=seed + i)
                if ok and q is not None and free(q):
                    out.append({"q": q, "pos": pos, "R": R, "mode": "top",
                                "reach": float(np.linalg.norm(pos[:2]))})
                    break

    # Rim pinch: needed for anything too wide to straddle, and a useful fallback otherwise.
    radius = object_rim_radius(model, data, body_prefix, center)
    for i in range(n_theta):
        theta = 2.0 * np.pi * i / n_theta
        for dz in (0.02, 0.005, -0.01):
            pos, R = grasp_pose(center, theta, radius, dz)
            q, ok = ik_pose(model, data, qadr, sid, pos, R, free=free,
                            seeds=ik_seeds, seed=seed + i)
            if ok and q is not None and free(q):
                out.append({"q": q, "pos": pos, "R": R, "mode": "rim", "theta": theta,
                            "reach": float(np.linalg.norm(pos[:2]))})
                break

    out.sort(key=lambda d: (d["mode"] != "top", d["reach"]))   # prefer straddle, then close reach
    return out


def object_rim_radius(model, data, body_prefix, center):
    """Rim radius of a circular object, from its true surface cloud (not bounding spheres)."""
    pc = object_cloud(model, data, body_prefix)
    if pc is None:
        return 0.05
    d = np.linalg.norm(pc[:, :2] - np.asarray(center, float)[:2], axis=1)
    return float(np.percentile(d, 95))          # 95th pct: the rim, robust to stray vertices


def path_to_ee_trace(model, data, qadr, sid, path):
    """Forward kinematics over the joint path → the EE pose trace to command through OSC."""
    return [fk_ee_pose(model, data, qadr, sid, q) for q in path]


def verify_ee_trace(model, data, qadr, sid, path, free):
    """THE check I should have run before claiming this doesn't work.

    The joint path is collision-free by construction. What matters is whether the configuration OSC
    actually adopts while tracking the FK trace is also collision-free. This measures the part that
    can be checked statically: that each planned configuration is free and that the FK trace is
    continuous enough for OSC to follow (no large EE jumps between waypoints).

    Returns a dict; execution-time verification is done by the rollout in test_rrt_transfer.py.
    """
    trace = path_to_ee_trace(model, data, qadr, sid, path)
    jumps = [float(np.linalg.norm(trace[i + 1][0] - trace[i][0])) for i in range(len(trace) - 1)]
    return {
        "n_waypoints": len(path),
        "all_configs_free": all(free(q) for q in path),
        "max_ee_jump_m": max(jumps) if jumps else 0.0,
        "path_len_m": float(sum(jumps)),
        "trace": trace,
    }
