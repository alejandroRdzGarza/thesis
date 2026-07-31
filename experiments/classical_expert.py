"""
classical_expert.py — scripted pick-and-place controller for generating optimal SAFE expert
demos (thesis Exp 006 pivot: replace the degrading VLA+CBF "expert" with a fixed classical one).

Uses privileged simulation state (object + goal poses) to drive a phase state machine that emits
a 7-D OSC_POSE nominal action (world-frame EE delta + gripper). The CALLER applies the CBF for
safety, so the recorded demo is: an optimal task-completing trajectory that satisfies the same
barrier the VLA must internalize. Deterministic, ~100% success, no VLA inference → fast + a clean
BC target.

v1 = state machine + P-control nominal + reactive CBF (validate quality on video first).
v2 (planned) = swap the transit nominal for an MPC-CBF horizon QP for smooth anticipatory paths.

This module is env-agnostic and CPU-testable: the state machine takes poses in, actions out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def resolve_pick_and_place(env, obs: dict) -> dict | None:
    """From the BDDL goal predicate, return the object to pick and the goal position.

    Predicate is e.g. ['in', 'orange_juice_1', 'basket_1_contain_region'] → pick 'orange_juice_1',
    place at the goal region site. Returns {'obj_key','obj_pos','goal_pos'} or None if unresolved.
    """
    from experiments.libero_runner import resolve_goal_from_bddl
    try:
        goal_state = env.env.parsed_problem["goal_state"]
    except Exception:
        return None
    if not goal_state or len(goal_state[0]) < 2:
        return None
    predicate = str(goal_state[0][0]).lower()         # 'on' (surface) | 'in' (container) | ...
    obj_name = goal_state[0][1]                        # the manipulated object (predicate arg 1)
    obj_key = f"{obj_name}_pos"
    if obs is None or obj_key not in obs:
        return None
    goal_pos = resolve_goal_from_bddl(env, obs)
    if goal_pos is None:
        return None
    return {
        "obj_key": obj_key,
        "obj_pos": np.asarray(obs[obj_key][:3], dtype=float),
        "goal_pos": np.asarray(goal_pos, dtype=float),
        # 'in' → drop into a container (basket); anything else ('on') → set down onto a surface.
        "place_mode": "in" if predicate == "in" else "on",
        # Wide bowls don't fit the 8 cm gripper → pinch the rim instead of straddling top-down.
        "grasp_mode": "rim" if "bowl" in obj_name.lower() else "top",
    }


# Gripper commands in OSC_POSE / π0.5 convention: +1 = close, −1 = open.
_GRIP_OPEN, _GRIP_CLOSE = -1.0, 1.0


class _Obs:
    """Lightweight obstacle for the MPC (pos + safety_radius), used to pass an INFLATED radius
    while carrying an object — the MPC only protects the EE, but the held object extends beyond
    it and would otherwise graze obstacles near the goal."""
    __slots__ = ("pos", "safety_radius")
    def __init__(self, pos, safety_radius):
        self.pos = pos
        self.safety_radius = safety_radius


@dataclass
class MPCConfig:
    """Receding-horizon QP that curves the EE around the obstacle (anticipatory, unlike a
    reactive CBF filter which just stalls at the boundary)."""
    horizon: int = 25
    u_max: float = 1.0          # max per-step action (OSC units, [-1,1])
    step_scale: float = 0.05    # metres moved per unit action (OSC output_max) — MPC plant scale
    w_u: float = 0.05           # effort weight
    w_smooth: float = 0.30      # smoothness weight (curved, not jerky, paths)
    radius_buffer: float = 0.03 # keep this much MORE clearance than the reactive CBF's radius, so
    #                             the MPC anticipates and the shield rarely fires (smooth, not lurchy).
    #                             Small — routing too wide swings the arm into other (unmodelled) objects.
    activate_margin: float = 0.09   # engage the keep-out early (anticipatory) when the straight
    #                                 path comes this close to the (buffered) obstacle sphere
    sqp_iters: int = 3          # re-linearize the keep-out around the solution this many times
    kp_fallback: float = 20.0   # P-control gain used when the path is clear / the QP fails


def _keepout_normal(ref_k, c, lateral):
    """Half-space normal for linearizing ‖P−c‖ ≥ r at ref_k; lateral when ref_k ≈ c."""
    d = ref_k - c
    nd = float(np.linalg.norm(d))
    return lateral if nd < 1e-3 else d / nd


def mpc_safe_delta(p, target, obstacle, cfg: MPCConfig) -> np.ndarray:
    """First EE delta of an optimal horizon path to `target` that stays outside the obstacle
    safety sphere. Receding-horizon (called every step). Anticipatory: seeds a bent path through
    a via-point beside the obstacle, then SQP-refines the linearized keep-out — so it curves
    AROUND instead of stalling at the boundary like a reactive filter. Falls back to P-control
    when the path is clear or the QP fails (the reactive CBF downstream is the hard-safety net)."""
    p = np.asarray(p, float); target = np.asarray(target, float)
    c = np.asarray(obstacle.pos, float)
    # Keep MORE clearance than the reactive CBF's radius so the MPC does the avoidance and the
    # shield rarely has to correct → smooth anticipatory paths instead of lurchy over-corrections.
    r = float(getattr(obstacle, "safety_radius", 0.10)) + cfg.radius_buffer
    to = target - p
    dist_to = float(np.linalg.norm(to))
    if dist_to > 1e-9:
        tt = float(np.clip(np.dot(c - p, to) / dist_to**2, 0.0, 1.0))
        closest = p + tt * to
    else:
        tt, closest = 0.0, p.copy()
    # Path already clear of the obstacle → straight P-control, no QP.
    if np.linalg.norm(closest - c) > r + cfg.activate_margin:
        return np.clip(cfg.kp_fallback * (target - p), -1.0, 1.0)

    import cvxpy as cp
    N = cfg.horizon
    path_dir = to / (dist_to + 1e-9)
    lateral = np.cross(path_dir, np.array([0.0, 0.0, 1.0]))
    ln = np.linalg.norm(lateral)
    lateral = lateral / ln if ln > 1e-6 else np.array([0.0, 1.0, 0.0])
    # Seed a BENT reference: p → via (beside the obstacle) → target, so the first linearization
    # already has lateral normals (a straight ref through the centre is degenerate).
    via = closest + lateral * (r + cfg.activate_margin)

    def _seed(s):
        if tt < 1e-6:
            return p + (target - p) * s
        return p + (via - p) * (s / tt) if s <= tt else via + (target - via) * ((s - tt) / (1 - tt))

    ref = np.array([_seed((k + 1) / N) for k in range(N)])

    first_u = None
    for _ in range(cfg.sqp_iters):
        u = cp.Variable((N, 3))
        P, acc = [], p
        for k in range(N):
            acc = acc + cfg.step_scale * u[k]      # real plant scale (OSC action → metres)
            P.append(acc)
        cost, cons = 0, []
        for k in range(N):
            cost += cp.sum_squares(P[k] - target) + cfg.w_u * cp.sum_squares(u[k])
            if k > 0:
                cost += cfg.w_smooth * cp.sum_squares(u[k] - u[k - 1])
            cons.append(cp.norm(u[k], "inf") <= cfg.u_max)
            n = _keepout_normal(ref[k], c, lateral)
            cons.append(n @ (P[k] - c) >= r)
        try:
            cp.Problem(cp.Minimize(cost), cons).solve(solver=cp.OSQP, warm_start=True)
        except Exception:
            break
        if u.value is None or not np.all(np.isfinite(u.value)):
            break
        ref = np.array([p + cfg.step_scale * np.sum(u.value[: k + 1], axis=0)
                        for k in range(N)])        # re-linearize around the executed-scale path
        first_u = np.clip(np.asarray(u.value[0]).ravel(), -1.0, 1.0)

    return first_u if first_u is not None else np.clip(cfg.kp_fallback * (target - p), -1.0, 1.0)


@dataclass
class ControllerConfig:
    kp: float = 20.0            # P-gain; delta = clip(kp·error, ±1). ~1/output_max (OSC ~0.05 m/step).
    pos_tol: float = 0.02       # m, waypoint-reached tolerance
    xy_tol: float = 0.015       # m, tighter XY alignment before descending / placing
    approach_h: float = 0.12    # m above the object to hover before descending
    grasp_dz: float = 0.005     # m above the object centre to aim for (descent is contact-based)
    rim_offset: float = 0.05    # rim grasp: offset the grip point along the closing axis (world Y)
    #                             by ~bowl radius so one finger drops inside, one outside → pinch rim
    lift_h: float = 0.18        # m to lift the grasped object before transporting
    goal_clear_h: float = 0.22  # m above the goal to carry the object — high enough that the
    #                             carton's BOTTOM clears the basket rim before lowering in
    place_dz: float = 0.02      # extra m above the goal for the object CENTRE when releasing
    grasp_hold: int = 8         # control steps to hold while the gripper closes
    release_hold: int = 5       # control steps to hold open after releasing
    descend_xy_lock: float = 0.012  # only lower z once XY error is under this (keeps the grip centred)
    descend_z_cap: float = 0.30 # cap the per-step descent command so OSC coupling doesn't drift XY
    carry_margin: float = 0.04  # extra obstacle clearance while transiting with a held object (its
    #                             extent beyond the EE) — not during PLACE (goal may be near the obstacle)
    place_xy_tol: float = 0.035 # looser XY tolerance to release over the basket (vs tight transit tol)
    place_timeout: int = 50     # steps: always release by now (carton drops into the basket from above)
    setdown_reach: float = 0.06 # 'on' mode: how far below the goal to aim — gentle, capped descent
    #                             stops the object ON the surface on contact (bigger slammed the plate)
    # Descent is "until contact": transition when the EE stops making downward progress
    # (bottomed out on the object/surface), which is robust to unknown object heights.
    stall_eps: float = 0.0015   # m of downward progress per step below which we count a stall
    stall_patience: int = 12    # consecutive stalled steps → treat as contact / reached


@dataclass
class PickPlaceController:
    """Phase state machine → 7-D OSC_POSE nominal action. Caller adds CBF safety."""
    cfg: ControllerConfig = field(default_factory=ControllerConfig)
    use_mpc: bool = True
    mpc_cfg: MPCConfig = field(default_factory=MPCConfig)
    place_mode: str = "in"      # 'in' = drop into a container (basket); 'on' = set down onto a surface
    grasp_mode: str = "top"     # 'top' = straddle over the object; 'rim' = pinch a wide bowl's rim
    phase: str = "APPROACH"
    grasp_offset: float | None = None       # EE_z − object_z at grasp (scalar, for debug)
    grasp_offset_vec: object = None          # EE − object at grasp (3-D; carries the rim XY offset)
    _timer: int = 0
    _stall: int = 0
    _last_z: float | None = None
    _descending: bool = True
    _obstacle: object = None

    PHASES = ("APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "DONE")

    def reset(self):
        self.phase = "APPROACH"
        self.grasp_offset = None
        self.grasp_offset_vec = None
        self._timer = 0
        self._stall = 0
        self._last_z = None
        self._descending = True

    def _enter(self, phase: str):
        self.phase = phase
        self._timer = 0
        self._stall = 0
        self._last_z = None
        self._descending = True

    def _contact(self, ee_z: float) -> bool:
        """True once the EE is COMMANDED down but stops moving (bottomed out) for `stall_patience`
        steps. Skips steps where we intentionally paused the descent to re-centre (else the pause
        false-triggers contact and grasps high)."""
        if not self._descending:
            self._stall = 0
            self._last_z = ee_z
            return False
        if self._last_z is not None and (self._last_z - ee_z) < self.cfg.stall_eps:
            self._stall += 1
        else:
            self._stall = 0
        self._last_z = ee_z
        return self._stall >= self.cfg.stall_patience

    def _careful_descend(self, ee_pos, target, grip) -> np.ndarray:
        """Lower onto a target with XY LOCKED: full-gain XY correction, but only descend once
        centred and at a capped speed — so OSC kinematic coupling can't drift the grip off the
        object during a fast vertical move (the spatial/goal grasp-miss failure).

        Sets `self._descending` so the caller's contact detector only counts a REAL physical
        stall (commanded down but not moving), not our intentional re-centre pauses (which were
        false-triggering an 11 cm-high grasp on the high-cabinet bowls)."""
        ee = np.asarray(ee_pos, float); target = np.asarray(target, float)
        dxy = np.clip(self.cfg.kp * (target[:2] - ee[:2]), -1.0, 1.0)
        if np.linalg.norm(ee[:2] - target[:2]) < self.cfg.descend_xy_lock:
            dz = float(np.clip(self.cfg.kp * (target[2] - ee[2]), -self.cfg.descend_z_cap, 1.0))
            self._descending = True
        else:
            dz = 0.0                                  # off-centre → re-centre before lowering
            self._descending = False
        return np.array([dxy[0], dxy[1], dz, 0.0, 0.0, 0.0, grip], dtype=np.float64)

    def _goto(self, ee_pos, target, grip) -> np.ndarray:
        """EE delta toward `target` (world frame), zero rotation, given gripper cmd.

        Uses MPC-CBF (curves around the obstacle, anticipatory) when an obstacle is known and
        use_mpc is set; otherwise plain P-control. The runner's reactive CBF is still applied
        downstream as the hard-safety backstop."""
        obs = self._obstacle
        if obs is not None and self.phase in ("LIFT", "TRANSPORT"):
            # Transiting with a held object → inflate the obstacle so the object (offset from the
            # EE) clears it en route. NOT during PLACE — the goal itself may be near the obstacle.
            obs = _Obs(np.asarray(obs.pos, float),
                       float(getattr(obs, "safety_radius", 0.10)) + self.cfg.carry_margin)
        if self.use_mpc and obs is not None:
            u = mpc_safe_delta(ee_pos, target, obs, self.mpc_cfg)
        else:
            u = np.clip(self.cfg.kp * (np.asarray(target) - np.asarray(ee_pos)), -1.0, 1.0)
        return np.array([u[0], u[1], u[2], 0.0, 0.0, 0.0, grip], dtype=np.float64)

    def act(self, ee_pos, obj_pos, goal_pos, *, obstacle=None,
            table_z: float | None = None) -> tuple[np.ndarray, str]:
        """One control step. `obj_pos` is the live object position (moves once grasped);
        `obstacle` (ObstacleConfig) enables MPC-CBF avoidance. Returns (action7, phase)."""
        self._obstacle = obstacle
        c = self.cfg
        ee = np.asarray(ee_pos, dtype=float)
        obj = np.asarray(obj_pos, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        base_z = table_z if table_z is not None else obj[2]

        # Rim grasp: offset the grip POINT along the gripper's closing axis (world Y) by ~bowl
        # radius, so one finger drops inside the bowl and one outside → pinch the rim wall on close.
        gp = obj.copy()
        if self.grasp_mode == "rim":
            gp = gp + np.array([0.0, c.rim_offset, 0.0])
        # After grasp, the object hangs at (EE − grasp_offset_vec); to put it at a target we drive
        # the EE to target + grasp_offset_vec.
        gov = (np.asarray(self.grasp_offset_vec, float)
               if self.grasp_offset_vec is not None else np.zeros(3))

        if self.phase == "APPROACH":                      # hover above the grip point, gripper open
            tgt = np.array([gp[0], gp[1], gp[2] + c.approach_h])
            if np.linalg.norm(ee[:2] - gp[:2]) < c.xy_tol and abs(ee[2] - tgt[2]) < c.pos_tol:
                self._enter("DESCEND")
            return self._goto(ee, tgt, _GRIP_OPEN), self.phase

        if self.phase == "DESCEND":                       # lower onto the grip point (XY-locked) until contact
            tgt = np.array([gp[0], gp[1], gp[2] + c.grasp_dz])
            centred = np.linalg.norm(ee[:2] - gp[:2]) < c.xy_tol
            if centred and (np.linalg.norm(ee - tgt) < c.pos_tol or self._contact(ee[2])):
                self.grasp_offset_vec = (ee - obj).copy()   # 3-D offset (carries the rim XY offset)
                self.grasp_offset = float(ee[2] - obj[2])
                self._enter("GRASP")
            return self._careful_descend(ee, tgt, _GRIP_OPEN), self.phase

        if self.phase == "GRASP":                         # hold + close the gripper
            self._timer += 1
            if self._timer >= c.grasp_hold:
                self._enter("LIFT")
            return self._goto(ee, ee, _GRIP_CLOSE), self.phase

        if self.phase == "LIFT":                           # raise the grasped object
            tgt = np.array([ee[0], ee[1], base_z + c.lift_h])
            if ee[2] >= base_z + c.lift_h - c.pos_tol:
                self._enter("TRANSPORT")
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "TRANSPORT":                      # carry so the OBJECT ends above the goal
            ee_goal_xy = goal[:2] + gov[:2]                # EE xy that puts the object over the goal
            tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + c.goal_clear_h])
            if np.linalg.norm(ee[:2] - ee_goal_xy) < c.xy_tol and abs(ee[2] - tgt[2]) < c.pos_tol:
                self._enter("PLACE")
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "PLACE":
            self._timer += 1
            ee_goal_xy = goal[:2] + gov[:2]                # EE xy that places the OBJECT over the goal
            centred = np.linalg.norm(ee[:2] - ee_goal_xy) < c.place_xy_tol
            if self.place_mode == "on":
                # Set the object DOWN onto the surface with a GENTLE capped descent, release on
                # contact (object bottom rests on the surface). A big/fast set-down slammed and
                # displaced the plate → 'held_object'/'scene_object' collisions on the goal suite.
                tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + gov[2] - c.setdown_reach])
                if (centred and self._contact(ee[2])) or self._timer >= 3 * c.place_timeout:
                    self._enter("RELEASE")
                return self._careful_descend(ee, tgt, _GRIP_CLOSE), self.phase
            else:
                # Drop into a container from above (basket): place the OBJECT centre at goal+place_dz.
                tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + c.place_dz + gov[2]])
                reached = np.linalg.norm(ee - tgt) < c.pos_tol or self._contact(ee[2])
                if (centred and reached) or self._timer >= c.place_timeout:
                    self._enter("RELEASE")
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "RELEASE":                        # open + hold
            self._timer += 1
            if self._timer >= c.release_hold:
                self._enter("DONE")
            return self._goto(ee, ee, _GRIP_OPEN), self.phase

        # DONE — hold position, gripper open.
        return self._goto(ee, ee, _GRIP_OPEN), self.phase


# ── CPU self-test: drive a single-integrator EE through all phases (no env) ──────────────────
if __name__ == "__main__":
    rng = np.random.default_rng(0)
    ee = np.array([0.0, 0.0, 0.30])
    obj = np.array([0.15, 0.10, 0.02])
    goal = np.array([-0.10, 0.25, 0.05])
    ctrl = PickPlaceController(); ctrl.reset()
    obj_live = obj.copy()
    grasped = False
    seen = []
    for t in range(600):
        a, ph = ctrl.act(ee, obj_live if not grasped else ee - np.array([0, 0, 0.005]), goal,
                         table_z=obj[2])
        if not seen or seen[-1] != ph:
            seen.append(ph)
        # crude single-integrator plant (OSC ~0.05 m per unit action); grasp attaches the object
        ee = ee + 0.05 * a[:3]
        if ph == "GRASP":
            grasped = True
        if ph == "RELEASE":
            grasped = False
        if ph == "DONE":
            break
    reached_goal = np.linalg.norm(ee[:2] - goal[:2]) < 0.05
    ok = seen == list(PickPlaceController.PHASES) and reached_goal
    print("phase sequence:", " → ".join(seen))
    print(f"ended near goal: {reached_goal}  (ee={np.round(ee,3)}, goal={np.round(goal,3)})")
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
