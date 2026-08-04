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


def mpc_safe_delta(p, target, obstacles, cfg: MPCConfig) -> np.ndarray:
    """First EE delta of an optimal horizon path to `target` that stays outside EVERY nearby
    object's keep-out sphere (MULTI-obstacle). `obstacles` is a list of objects with .pos and
    .safety_radius. Receding-horizon; anticipatory: seeds a bent path past the most-blocking
    obstacle, then SQP-refines the linearized keep-out for all active obstacles at once — so the
    arm curves smoothly through clutter instead of lurching (reactive CBF) or clipping an
    unmodelled object. Falls back to P-control when the path is clear or the QP fails."""
    p = np.asarray(p, float); target = np.asarray(target, float)
    if not isinstance(obstacles, (list, tuple)):
        obstacles = [obstacles]
    to = target - p
    dist_to = float(np.linalg.norm(to))
    path_dir = to / (dist_to + 1e-9)

    # Which obstacles the straight path comes near → the ones to constrain (keeps the QP small).
    active = []          # (centre, radius, closest-approach distance)
    for ob in obstacles:
        if ob is None:
            continue
        c = np.asarray(ob.pos, float)
        r = float(getattr(ob, "safety_radius", 0.05)) + cfg.radius_buffer
        # A keep-out that SWALLOWS the target makes the QP infeasible, and an infeasible QP
        # stalls the arm on the boundary (the APPROACH freeze). Shrink such a sphere to just
        # inside the target so the path can still be planned; the reactive CBF downstream is
        # untouched, so hard safety does not depend on this.
        d_tgt = float(np.linalg.norm(target - c))
        if d_tgt < r:
            r = max(d_tgt - 0.01, 0.0)
        tt = float(np.clip(np.dot(c - p, to) / dist_to**2, 0.0, 1.0)) if dist_to > 1e-9 else 0.0
        d = float(np.linalg.norm((p + tt * to) - c))
        if r > 1e-6 and d < r + cfg.activate_margin:
            active.append((c, r, d))
    if not active:
        return np.clip(cfg.kp_fallback * (target - p), -1.0, 1.0)

    import cvxpy as cp
    N = cfg.horizon
    lateral = np.cross(path_dir, np.array([0.0, 0.0, 1.0]))
    ln = np.linalg.norm(lateral)
    lateral = lateral / ln if ln > 1e-6 else np.array([0.0, 1.0, 0.0])
    # Seed a bent reference past the MOST-blocking (closest) obstacle so the first linearization
    # already curves; SQP then refines around all active obstacles.
    cb, rb, _ = min(active, key=lambda a: a[2])
    ttb = float(np.clip(np.dot(cb - p, to) / (dist_to**2 + 1e-9), 0.0, 1.0))
    via = (p + ttb * to) + lateral * (rb + cfg.activate_margin)

    def _seed(s):
        if ttb < 1e-6:
            return p + (target - p) * s
        return p + (via - p) * (s / ttb) if s <= ttb else via + (target - via) * ((s - ttb) / (1 - ttb))

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
            for (c, r, _d) in active:                # keep clear of EVERY active obstacle
                cons.append(_keepout_normal(ref[k], c, lateral) @ (P[k] - c) >= r)
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
    descend_xy_lock: float = 0.020  # lower z once XY error is under this. MUST be >= xy_tol (0.015):
    #                                 if tighter, an extended reach (elevated bowls) that can only
    #                                 hold XY to ~0.012 oscillates at the boundary and never descends.
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
    # ── Reactive (stateless) phase inference: the "phase" is DERIVED from the current observable
    # state each call, so the same observation always maps to the same action (a Markov oracle —
    # required for DAgger relabelling, where the STUDENT drives and the expert labels its states).
    grip_closed: float = 0.04   # gripper finger separation (m) below which it has SECURED an object
    #                             (open ≈ 0.08; still-closing ≈ 0.06; secured rim ≈ 0.004). Must be
    #                             tight enough that 'holding' means done-closing, else the arm lifts
    #                             mid-close and grabs air (matches the machine's lift at grip≈0.03).
    hold_lift: float = 0.03     # fallback (no gripper signal): object lifted >this above rest height
    hold_reach: float = 0.10    # object XY within this of the EE to count as held (covers rim offset)
    hold_z: float = 0.06        # …and object within this Z of the EE — so a MISSED grab (arm rises,
    #                             object left on the table) is detected as not-holding → retry
    descend_start_h: float = 0.15  # aligned in XY and EE within this of the grip point → descend/grasp
    grasp_z_tol: float = 0.02   # EE within grasp_dz+this of the object z → close the gripper (grasp)
    place_done_xy: float = 0.05  # object within this XY of the goal …
    place_done_z: float = 0.05   # …and this Z above the goal, and not held → task done (hold, open)
    release_on_dz: float = 0.015  # 'on': release once the object is within this of the goal surface
    release_in_dz: float = 0.05   # 'in': release once the object centre is within this above the goal


@dataclass
class PickPlaceController:
    """Phase state machine → 7-D OSC_POSE nominal action. Caller adds CBF safety."""
    cfg: ControllerConfig = field(default_factory=ControllerConfig)
    use_mpc: bool = True
    mpc_cfg: MPCConfig = field(default_factory=MPCConfig)
    place_mode: str = "in"      # 'in' = drop into a container (basket); 'on' = set down onto a surface
    grasp_mode: str = "top"     # 'top' = straddle over the object; 'rim' = pinch a wide bowl's rim
    grasp_offset_xy: object = None   # 3-vector obj→grip point, set per scene by teacher_profiles
    #                             (which SIDE of a wide bowl to pinch). None → legacy +Y rim_offset.
    profile_name: str = "default"    # which teacher profile configured this controller (for logs)
    reactive: bool = True       # True = stateless Markov oracle (phase inferred from observables);
    #                             False = the legacy latched state machine (kept for A/B comparison)
    phase: str = "APPROACH"
    grasp_offset: float | None = None       # EE_z − object_z at grasp (scalar, for debug)
    grasp_offset_vec: object = None          # EE − object at grasp (3-D; carries the rim XY offset)
    _timer: int = 0
    _stall: int = 0
    _last_z: float | None = None
    _descend_min_z: float | None = None   # lowest EE z reached in the current descent (contact test)
    _descending: bool = True
    _obstacles: object = None    # list of nearby objects to avoid (multi-obstacle MPC)

    PHASES = ("APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "DONE")

    def grip_point(self, obj_pos) -> np.ndarray:
        """Where the EE must go to grasp `obj_pos`.

        A wide bowl doesn't fit the 8 cm gripper, so it's pinched by the RIM: the grip point is
        offset ~a bowl radius along the gripper's closing axis (world Y, since the controller
        commands zero rotation). WHICH side is a per-scene decision — reaching towards a nearby
        obstacle makes the barrier refuse the approach — so `grasp_offset_xy` is set by
        `teacher_profiles.resolve_profile`. Falls back to the legacy +Y offset when unset, and the
        runner uses this same method for the IK grasp orientation so both agree on the target.
        """
        gp = np.asarray(obj_pos, dtype=float).copy()
        if self.grasp_offset_xy is not None:
            return gp + np.asarray(self.grasp_offset_xy, dtype=float)
        if self.grasp_mode == "rim":
            gp = gp + np.array([0.0, self.cfg.rim_offset, 0.0])
        return gp

    def reset(self):
        self.phase = "APPROACH"
        self.grasp_offset = None
        self.grasp_offset_vec = None
        self._timer = 0
        self._stall = 0
        self._last_z = None
        self._descend_min_z = None
        self._descending = True

    def _enter(self, phase: str):
        self.phase = phase
        self._timer = 0
        self._stall = 0
        self._last_z = None
        self._descend_min_z = None
        self._descending = True

    def _contact(self, ee_z: float) -> bool:
        """True once a commanded descent BOTTOMS OUT: the EE stops reaching new lows for
        `stall_patience` steps. Min-tracking (not consecutive-step) makes it robust to the ±few-mm
        bounce when the gripper hits the object/shelf. Only counts steps where we're actually
        descending (self._descending); intentional re-centre pauses neither count nor reset, so a
        paused descent still detects contact instead of false-triggering high."""
        if not self._descending:
            return self._stall >= self.cfg.stall_patience
        if self._descend_min_z is None or ee_z < self._descend_min_z - self.cfg.stall_eps:
            self._descend_min_z = ee_z         # new low → still making downward progress
            self._stall = 0
        else:
            self._stall += 1                   # no new low → bottoming out
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
        obs_list = self._obstacles or []
        if obs_list and self.phase in ("LIFT", "TRANSPORT"):
            # Transiting with a held object → inflate every obstacle so the held object (offset
            # from the EE) clears them en route. NOT during PLACE — the goal may be near an object.
            obs_list = [_Obs(np.asarray(o.pos, float),
                             float(getattr(o, "safety_radius", 0.05)) + self.cfg.carry_margin)
                        for o in obs_list]
        if self.use_mpc and obs_list:
            u = mpc_safe_delta(ee_pos, target, obs_list, self.mpc_cfg)
        else:
            u = np.clip(self.cfg.kp * (np.asarray(target) - np.asarray(ee_pos)), -1.0, 1.0)
        return np.array([u[0], u[1], u[2], 0.0, 0.0, 0.0, grip], dtype=np.float64)

    def _holding(self, ee, obj, gripper, base_z) -> bool:
        """Observable grasp test — a Markov signal of 'currently carrying THIS object', no latched
        flag. Primary: the gripper has CLOSED (finger separation < grip_closed) AND the object is
        horizontally at the EE. This avoids the lift chicken-and-egg (the object can't be 'lifted'
        until we lift it, but we only lift once holding). Fallback if no gripper signal is given:
        object lifted above its rest height and near the EE."""
        xy_near = np.linalg.norm(ee[:2] - obj[:2]) < self.cfg.hold_reach
        if gripper is not None:
            return xy_near and gripper < self.cfg.grip_closed and abs(ee[2] - obj[2]) < self.cfg.hold_z
        return xy_near and (obj[2] - base_z) > self.cfg.hold_lift

    def act(self, ee_pos, obj_pos, goal_pos, *, obstacle=None, obstacles=None,
            table_z: float | None = None, gripper: float | None = None) -> tuple[np.ndarray, str]:
        """One control step → (action7, phase). Reactive (default) infers the phase from the current
        observable state (incl. the gripper finger separation `gripper`) so it's a Markov oracle;
        `reactive=False` runs the legacy machine (ignores `gripper`)."""
        if obstacles is not None:
            self._obstacles = list(obstacles)
        elif obstacle is not None:
            self._obstacles = [obstacle]
        else:
            self._obstacles = []
        if self.reactive:
            return self._act_reactive(ee_pos, obj_pos, goal_pos, table_z, gripper)
        return self._act_statemachine(ee_pos, obj_pos, goal_pos, table_z)

    def _act_reactive(self, ee_pos, obj_pos, goal_pos, table_z, gripper) -> tuple[np.ndarray, str]:
        """Stateless pick-place: phase DERIVED from (ee, obj, goal) each call. Same tuned targets
        and _goto/_careful_descend as the machine, but no latched phase/timer/contact state — so the
        SAME observation always yields the SAME action (valid DAgger oracle from any student state).

        On the expert's own trajectory the inferred phase tracks the machine's latched phase, so the
        driver behaves the same; off-trajectory (student-visited states) it stays consistent instead
        of desyncing — the fix for the DAgger collision blow-up."""
        c = self.cfg
        ee = np.asarray(ee_pos, dtype=float)
        obj = np.asarray(obj_pos, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        base_z = table_z if table_z is not None else obj[2]

        gp = self.grip_point(obj)
        gov = ee - obj                                    # LIVE grasp offset (valid whenever holding)
        self.grasp_offset_vec = gov.copy()                # keep for the end-of-episode debug dump
        self.grasp_offset = float(ee[2] - obj[2])
        holding = self._holding(ee, obj, gripper, base_z)

        # Terminal: object resting at the goal and no longer held → task done (hold, gripper open).
        if not holding and (np.linalg.norm(obj[:2] - goal[:2]) < c.place_done_xy
                            and (obj[2] - goal[2]) < c.place_done_z):
            self.phase = "DONE"
            return self._goto(ee, ee, _GRIP_OPEN), self.phase

        if not holding:                                   # ── PICK: align, descend, close ──
            # MISSED GRASP → reopen and retry. Without this the controller deadlocks: once the
            # descent bottoms out, `_contact` latches (we command zero motion, so the EE stops
            # reaching new lows and `_stall` only grows), so every later step re-enters GRASP and
            # the arm can never descend again. Purely observable — the gripper has finished closing
            # yet the object isn't with us — so the oracle stays Markov.
            if gripper is not None and gripper < c.grip_closed:
                self.phase = "APPROACH"
                self._stall = 0; self._descend_min_z = None   # let the next descent start fresh
                tgt = np.array([gp[0], gp[1], gp[2] + c.approach_h])
                return self._goto(ee, tgt, _GRIP_OPEN), self.phase
            aligned = np.linalg.norm(ee[:2] - gp[:2]) < c.xy_tol
            if not aligned or ee[2] > gp[2] + c.descend_start_h:
                self.phase = "APPROACH"                   # hover above the grip point (avoidance on)
                self._stall = 0; self._descend_min_z = None   # reset descent-stall tracking on (re)approach
                tgt = np.array([gp[0], gp[1], gp[2] + c.approach_h])
                return self._goto(ee, tgt, _GRIP_OPEN), self.phase
            if ee[2] <= gp[2] + c.grasp_dz + c.grasp_z_tol:
                self.phase = "GRASP"                      # at the grip height → close + hold
                return self._goto(ee, ee, _GRIP_CLOSE), self.phase
            # Careful XY-locked descent. Grasp on CONTACT too: if commanded down but the EE bottoms
            # out (elevated bowls — stove/cabinet — where the gripper hits the object/shelf before
            # the EE frame reaches obj_z), treat the stall as contact and close (the pure geometric
            # trigger above never fires there). _careful_descend sets _descending so _contact only
            # counts a real stall, not the intentional re-centre pauses.
            tgt = np.array([gp[0], gp[1], gp[2] + c.grasp_dz])
            descend_action = self._careful_descend(ee, tgt, _GRIP_OPEN)
            if self._contact(ee[2]):
                self.phase = "GRASP"
                return self._goto(ee, ee, _GRIP_CLOSE), self.phase
            self.phase = "DESCEND"
            return descend_action, self.phase

        # ── PLACE: object is held ──
        ee_goal_xy = goal[:2] + gov[:2]                   # EE xy that puts the OBJECT over the goal
        over_goal = np.linalg.norm(ee[:2] - ee_goal_xy) < c.xy_tol
        # ELEVATED pick (bowl on a stove/cabinet, base_z high): lift only enough to clear the
        # surface — a full lift_h would drive the EE near the upward reach limit, and the goal is
        # usually LOWER anyway (transport descends to it).
        _lift_h = c.lift_h if base_z < 1.0 else 0.08
        lifted = ee[2] >= base_z + _lift_h - c.pos_tol    # gate on EE height (reliably reaches target)
        if not lifted and not over_goal:
            self.phase = "LIFT"                           # raise to clear (carry-inflated MPC)
            tgt = np.array([ee[0], ee[1], base_z + _lift_h])
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase
        if not over_goal:
            self.phase = "TRANSPORT"                      # carry so the object ends over the goal
            tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + c.goal_clear_h])
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase
        # Over the goal → lower and release. Release is triggered by the OBJECT reaching placement
        # height (observable), not a timer.
        if self.place_mode == "on":
            if (obj[2] - goal[2]) <= c.release_on_dz:
                self.phase = "RELEASE"
                return self._goto(ee, ee, _GRIP_OPEN), self.phase
            self.phase = "PLACE"
            tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + gov[2] - c.setdown_reach])
            return self._careful_descend(ee, tgt, _GRIP_CLOSE), self.phase
        else:
            if (obj[2] - goal[2]) <= c.release_in_dz:
                self.phase = "RELEASE"
                return self._goto(ee, ee, _GRIP_OPEN), self.phase
            self.phase = "PLACE"
            tgt = np.array([ee_goal_xy[0], ee_goal_xy[1], goal[2] + c.place_dz + gov[2]])
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

    def _act_statemachine(self, ee_pos, obj_pos, goal_pos, table_z) -> tuple[np.ndarray, str]:
        """Legacy latched-phase state machine (kept for A/B vs the reactive oracle)."""
        c = self.cfg
        ee = np.asarray(ee_pos, dtype=float)
        obj = np.asarray(obj_pos, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        base_z = table_z if table_z is not None else obj[2]

        # Rim grasp: offset the grip POINT along the gripper's closing axis (world Y) by ~bowl
        # radius, so one finger drops inside the bowl and one outside → pinch the rim wall on close.
        gp = self.grip_point(obj)
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


# ── CPU self-test: drive a single-integrator EE (with a simulated gripper) through a pick-place ──
if __name__ == "__main__":
    ee = np.array([0.0, 0.0, 0.30])
    obj0 = np.array([0.15, 0.10, 0.02])
    goal = np.array([-0.10, 0.25, 0.05])
    ctrl = PickPlaceController(reactive=True); ctrl.reset()
    obj = obj0.copy()
    grip = 0.08            # finger separation: open ≈ 0.08, closed ≈ 0
    attached = False
    grab_off = np.zeros(3)
    seen = []
    for t in range(800):
        a, ph = ctrl.act(ee, obj, goal, table_z=obj0[2], gripper=grip)
        if not seen or seen[-1] != ph:
            seen.append(ph)
        # single-integrator EE (OSC ~0.05 m per unit action)
        ee = ee + 0.05 * a[:3]
        # gripper closes/opens toward the command over a few steps
        grip = np.clip(grip + (-0.012 if a[6] > 0 else +0.012), 0.0, 0.08)
        # attach the object once the (closing) gripper is around it; carry it; drop when opened
        if not attached and grip < ctrl.cfg.grip_closed and np.linalg.norm(ee - obj) < 0.05:
            attached = True; grab_off = ee - obj
        if attached:
            if a[6] > 0:
                obj = ee - grab_off
            else:
                attached = False          # released → object stays where dropped
        if ph == "DONE":
            break
    placed = np.linalg.norm(obj[:2] - goal[:2]) < 0.05 and abs(obj[2] - goal[2]) < 0.08
    ok = "DONE" in seen and placed
    print("phase sequence:", " → ".join(seen))
    print(f"object placed: {placed}  (obj={np.round(obj,3)}, goal={np.round(goal,3)})")
    print("SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
