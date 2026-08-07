"""planner_expert.py — a full pick-and-place teacher built on joint-space planning.

Where the scripted `PickPlaceController` reacts step by step from hand-tuned targets, this plans
the whole episode up front: search the object's geometry for a reachable collision-free grasp,
RRT-Connect to it, close, lift, RRT to the placement **with the held object carried along in the
collision check**, set down, release. Forward kinematics turns each joint path into an
end-effector pose trace, which is commanded as OSC_POSE deltas — the same action space π0.5
outputs, so the resulting demos are student-reproducible.

Measured basis for this existing at all: RRT → FK → OSC executes collision-free on 12/12
safelibero_spatial LII rollouts, with up to 4.66 rad of joint drift between the planned and
executed configuration. OSC abandons the planned arm posture; collision-freedom survives anyway.

Interface-compatible with PickPlaceController, so `run_libero_trial(controller=...)` picks up the
CBF shield, collision attribution, metrics and BC-trace recording unchanged. The one addition is
`plan()`, which the runner calls once at episode start because planning needs `model`/`data`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_GRIP_OPEN, _GRIP_CLOSE = -1.0, 1.0


@dataclass
class PlannerConfig:
    pre_grasp_h: float = 0.10     # hover height above the grasp before descending
    lift_h: float = 0.15          # how far to lift after grasping
    place_clear_h: float = 0.12   # height above the goal to arrive at before setting down
    grasp_hold: int = 12          # steps to hold while the fingers close
    release_hold: int = 8
    reach_tol: float = 0.02       # m, waypoint-reached tolerance
    max_steps_per_wp: int = 20
    strict_max_steps: int = 90   # budget to ARRIVE at a grasp/release pose before
    #                              closing the gripper; travelling waypoints keep the
    #                              cheaper max_steps_per_wp timeout
    kp: float = 20.0
    krot: float = 4.0
    descend_steps: int = 6        # interpolation points on the straight-line descents
    ik_seeds: int = 20
    try_candidates: int = 10      # grasps to attempt before giving up on the scene
    # closed-loop servo (final grasp and final set-down)
    servo_xy_lock: float = 0.020  # centre in XY to within this before descending — a fast vertical
    #                               move under OSC drags the grip point off the object. MUST stay
    #                               >= what OSC can actually hold: at 0.012 an extended reach
    #                               oscillates at the boundary and never descends at all (this is
    #                               a known failure of the scripted controller, re-hit here).
    servo_max_steps: int = 140    # hard cap so a servo can never hang the episode; on expiry it
    #                               attempts the grasp anyway rather than descending forever
    servo_z_cap: float = 0.25     # cap the descent command so OSC coupling cannot induce XY drift
    servo_grasp_tol: float = 0.012
    servo_place_tol: float = 0.020
    stall_eps: float = 0.0015     # m of downward progress below which a step counts as stalled
    stall_patience: int = 12      # consecutive stalled steps = contact reached
    place_dz: float = 0.010       # release the object this far above the goal surface
    clearance: float = 0.02       # m of standoff the PLANNER must keep from the scene.
    #                               Planning to zero clearance produced 24 grazing
    #                               contacts (1.0-4.7 mm displacements) as soon as OSC
    #                               drifted off the plan. The grasp itself is exempt —
    #                               it has to touch the object.


@dataclass
class PlannerExpert:
    """Plans once, then plays the plan back as OSC_POSE deltas."""
    cfg: PlannerConfig = field(default_factory=PlannerConfig)
    seed: int = 0
    phase: str = "PLAN"
    planned: bool = False
    plan_error: str = ""
    _wps: list = field(default_factory=list)   # [{pos, R, grip, hold, phase}]
    _i: int = 0
    _steps_on_wp: int = 0
    _held: int = 0
    reach_failed: bool = False   # never arrived at a grasp/release pose
    _stall: int = 0              # steps the descent has made no downward progress
    _min_z: object = None        # lowest EE z reached during the current descent
    _closing: int = 0            # steps spent closing the fingers

    # PickPlaceController compatibility (the runner sets these; the planner derives its own).
    place_mode: str = "on"
    grasp_mode: str = "top"
    grasp_offset_xy: object = None
    profile_name: str = "planner"
    grasp_offset: object = None
    grasp_offset_vec: object = None
    _rot_ref: object = None      # latest EE quaternion, pushed in by the runner each step
    # Standoffs to try, in order. Most scenes plan at the full 2 cm; narrow passages fall
    # back rather than returning no plan at all.
    clearance_ladder: tuple = (0.02, 0.01, 0.0)

    def reset(self):
        self.phase = "PLAN"
        self.planned = False
        self.plan_error = ""
        self._wps = []
        self._i = 0
        self._steps_on_wp = 0
        self._held = 0
        self.reach_failed = False
        self._stall = 0
        self._min_z = None
        self._closing = 0

    # ── planning ────────────────────────────────────────────────────────────
    def plan(self, model, data, qadr, jnt_rng, sid, ctx) -> bool:
        """Build the whole waypoint list. Returns False if the scene could not be planned."""
        from experiments import rrt_planner as P

        obj = np.asarray(ctx["obj_pos"], float)
        goal = np.asarray(ctx["goal_pos"], float)
        body = ctx["obj_key"].replace("_pos", "")
        c = self.cfg

        free = P.make_collision_fn(model, data, qadr, ignore=(body,))
        q_start = np.array([data.qpos[a] for a in qadr])
        if not free(q_start):
            self.plan_error = "start configuration in collision"
            return False

        # Grasp POSES are searched WITHOUT clearance — the fingers have to reach the object.
        cands = P.sample_grasps(model, data, qadr, sid, obj, body, free,
                                seed=self.seed, ik_seeds=c.ik_seeds)
        if not cands:
            self.plan_error = "no reachable collision-free grasp"
            return False

        obj_qadr = P.free_joint_qadr(model, body)
        from collections import Counter
        stage = Counter()          # which planning stage rejected each candidate

        # Order candidates by whether they make the PLACEMENT reachable. The grasp offset is
        # carried to the goal (the EE must go to goal + offset to put the object on the goal), so a
        # grasp on the far side of the object pushes the placement outward — measured on goal LII
        # t0, where every top candidate put the place pose at x=0.166 y=0.291, outside the arm's
        # workspace, and IK failed there even with collision checking switched off entirely.
        goal_xy = np.asarray(goal, float)[:2]
        for cd in cands:
            cd["place_reach"] = float(np.linalg.norm(goal_xy + (cd["pos"] - obj)[:2]))
        # Primary key is grasp quality (straddle first, then closest reach) — ordering by
        # placement reach instead raised planning to 96% but dropped clean demos 40% -> 29%,
        # because it selects grasps that are convenient for the PLACE and poor for the PICK.
        # Placement reach is a tiebreak only; unreachable placements are caught by trying
        # more candidates rather than by reordering.
        cands.sort(key=lambda d: ({"top": 0, "rim": 1, "obb": 2}.get(d["mode"], 3),
                                  round(d["reach"], 2), d["place_reach"]))

        # Travelled paths keep a standoff, but a fixed one costs scenes: 2 cm closed the narrow
        # passages on 2 of 24 and made them unplannable. Back off instead of failing — a 1 cm
        # plan is worth far more than no plan, and most scenes take the full 2 cm.
        for clr in self.clearance_ladder:
            for cand in cands[:c.try_candidates]:
                wps = []
                p_grasp, R_grasp = cand["pos"], cand["R"]
                p_pre = p_grasp + np.array([0.0, 0.0, c.pre_grasp_h])

                # 1. plan to the pre-grasp hover
                q_pre, ok = P.ik_pose(model, data, qadr, sid, p_pre, R_grasp, free=free,
                                      seeds=c.ik_seeds, seed=self.seed)
                if not ok:
                    stage["ik_pregrasp"] += 1
                    continue
                with P.clearance_margin(model, clr):
                    free_clear = P.make_collision_fn(model, data, qadr, ignore=(body,))
                    path, _why = P.rrt_connect(q_start, q_pre, jnt_rng, free_clear, seed=self.seed)
                    if path is not None:
                        path = P.shortcut(path, free_clear, seed=self.seed)
                if path is None:
                    stage["rrt_approach"] += 1
                    continue
                path = P.densify(path, max_step=0.05)
                _tr = P.path_to_ee_trace(model, data, qadr, sid, path)
                for _k, (p, R) in enumerate(_tr):
                    # The LAST approach waypoint must actually be reached: the descent that follows
                    # only moves in z, so any XY error left here is carried straight into the grasp
                    # (measured: 56 mm of XY error at the grip point, fingers closing beside the object).
                    wps.append(dict(pos=p, R=R, grip=_GRIP_OPEN, hold=0, phase="APPROACH",
                                    strict=(_k == len(_tr) - 1)))

                # 2-3. descend and close, CLOSED-LOOP on the live object pose. The planned pose is
                # only a prediction; driving to it blind closed the fingers 55 mm from the object
                # (measured, no contacts — OSC simply does not converge to every pose IK can reach).
                # Servoing re-aims at wherever the object ACTUALLY is on each step.
                wps.append(dict(pos=p_grasp, R=R_grasp, grip=_GRIP_OPEN, hold=0,
                                phase="DESCEND", servo="grasp",
                                grasp_off=(p_grasp - obj).copy()))

                # 4. lift straight up
                p_lift = p_grasp + np.array([0.0, 0.0, c.lift_h])
                for k in range(1, c.descend_steps + 1):
                    wps.append(dict(pos=p_grasp + (p_lift - p_grasp) * (k / c.descend_steps),
                                    R=R_grasp, grip=_GRIP_CLOSE, hold=0, phase="LIFT"))

                # 5. transport, WITH the held object in the collision check
                rel_pos = R_grasp.T @ (obj - p_grasp)          # object pose in the EE frame at grasp
                rel_R = R_grasp.T @ np.eye(3)
                free_held = P.make_attached_collision_fn(model, data, qadr, sid, obj_qadr,
                                                         rel_pos, rel_R, ignore=(body,))
                q_lift, ok = P.ik_pose(model, data, qadr, sid, p_lift, R_grasp, free=free,
                                       seeds=c.ik_seeds, seed=self.seed)
                if not ok:
                    stage["ik_lift"] += 1
                    continue
                # Drive the EE to wherever puts the OBJECT over the goal. The full 3-D grasp offset
                # matters, not just its z: a rim pinch holds the object ~a rim radius to one side, so
                # aiming the EE at the goal leaves the object short by exactly that much (measured:
                # 4.4 cm miss against a 4.7 cm rim radius).
                grasp_off = p_grasp - obj                      # EE − object, at the moment of grasp
                p_place = np.array([goal[0], goal[1], goal[2]]) + grasp_off + np.array([0, 0, 0.01])
                # Approach the placement from as high as is REACHABLE. A fixed hover height puts
                # the pre-place pose outside the workspace when the goal is already elevated (a
                # cabinet top), and the whole scene then fails to plan — measured as ik_preplace
                # rejecting all 18 candidate/clearance combinations on goal LII t0.
                q_pp, ok, p_preplace = None, False, None
                for _h in (c.place_clear_h, 0.08, 0.05, 0.02):
                    cand_pp = p_place + np.array([0.0, 0.0, _h])
                    q_pp, ok = P.ik_pose(model, data, qadr, sid, cand_pp, R_grasp, free=free_held,
                                         seeds=c.ik_seeds, seed=self.seed)
                    if ok:
                        p_preplace = cand_pp
                        break
                if not ok:
                    stage["ik_preplace"] += 1
                    continue
                # NB uses `clr`, not c.clearance: the transport leg was pinned at the fixed 2 cm
                # while only the approach followed the ladder, so a scene reported as failing "at
                # all clearances" had in fact only ever been tried at one.
                #
                # Transport must clear whatever sits between pick and place while CARRYING the
                # object; when it fails the usual reason is too little height, so retry from a
                # higher lift before giving this candidate up.
                tpath = None
                for _lift_extra in (0.0, 0.08, 0.16):
                    if _lift_extra > 0.0:
                        p_lift_try = p_grasp + np.array([0.0, 0.0, c.lift_h + _lift_extra])
                        q_try, ok_l = P.ik_pose(model, data, qadr, sid, p_lift_try, R_grasp,
                                                free=free, seeds=c.ik_seeds, seed=self.seed)
                        if not ok_l:
                            continue
                        q_lift, p_lift = q_try, p_lift_try
                    with P.clearance_margin(model, clr):
                        free_held_clear = P.make_attached_collision_fn(
                            model, data, qadr, sid, obj_qadr, rel_pos, rel_R, ignore=(body,))
                        tpath, _why = P.rrt_connect(q_lift, q_pp, jnt_rng, free_held_clear,
                                                    seed=self.seed)
                        if tpath is not None:
                            tpath = P.shortcut(tpath, free_held_clear, seed=self.seed)
                    if tpath is not None:
                        break
                if tpath is None:
                    stage["rrt_transport"] += 1
                    continue
                tpath = P.densify(tpath, max_step=0.05)
                for p, R in P.path_to_ee_trace(model, data, qadr, sid, tpath):
                    wps.append(dict(pos=p, R=R, grip=_GRIP_CLOSE, hold=0, phase="TRANSPORT"))

                # 6-7. set down and release, CLOSED-LOOP. Driving to a placement computed from the
                # grasp offset predicted at t=0 ignores how the object ACTUALLY ended up in the
                # gripper; re-deriving the offset from the live object and EE each step makes the
                # placement self-correct for grasp slip.
                wps.append(dict(pos=p_place, R=R_grasp, grip=_GRIP_CLOSE, hold=0,
                                phase="PLACE", servo="place"))
                wps.append(dict(pos=p_place, R=R_grasp, grip=_GRIP_OPEN,
                                hold=c.release_hold, phase="RELEASE"))
                wps.append(dict(pos=p_place + np.array([0.0, 0.0, 0.10]), R=R_grasp,
                                grip=_GRIP_OPEN, hold=0, phase="RETREAT"))

                self._wps = wps
                self.planned = True
                self.grasp_mode = cand["mode"]
                self.phase = "APPROACH"
                return True

        self.plan_error = (f"no plan at any clearance {self.clearance_ladder}; "
                           f"rejections by stage: {dict(stage)}")
        return False

    # ── closed-loop servos ──────────────────────────────────────────────────
    def _servo(self, wp, ee, obj, goal, gripper):
        """Drive the last few centimetres from OBSERVED state. Returns (action7, phase_done).

        Two cases, both re-derived every step rather than replayed from the plan:

        grasp  target = live object pose + the grasp offset chosen at plan time. XY is locked
               before descending (a fast vertical move under OSC drags the grip point off the
               object), and the fingers close on CONTACT — detected as the descent ceasing to
               reach new lows — rather than at a predicted height, which is what makes it robust
               to the object not being exactly where the plan assumed.

        place  target = goal + (live EE - live object). Using the LIVE offset means the placement
               corrects for however the object actually sits in the gripper, instead of trusting
               the offset predicted at the moment of grasp.
        """
        c = self.cfg
        act = np.zeros(7)
        if self._rot_ref is not None:
            from scipy.spatial.transform import Rotation as _R
            Rcur = _R.from_quat(np.asarray(self._rot_ref, float)).as_matrix()
            act[3:6] = np.clip(c.krot * _R.from_matrix(wp["R"] @ Rcur.T).as_rotvec(), -1.0, 1.0)

        if wp["servo"] == "grasp":
            target = obj + np.asarray(wp["grasp_off"], float)
            # already closing? hold still and finish the grip
            if self._closing > 0:
                self._closing += 1
                act[6] = _GRIP_CLOSE
                return act, self._closing > c.grasp_hold
            dxy = target[:2] - ee[:2]
            act[6] = _GRIP_OPEN
            if (float(np.linalg.norm(dxy)) > c.servo_xy_lock
                    and self._steps_on_wp < c.servo_max_steps // 2):
                act[:2] = np.clip(c.kp * dxy, -1.0, 1.0)      # centre first, do not descend yet
                return act, False
            act[:2] = np.clip(c.kp * dxy, -1.0, 1.0)
            act[2] = float(np.clip(c.kp * (target[2] - ee[2]), -c.servo_z_cap, 1.0))
            # contact test: the descent stops finding new lows
            if self._min_z is None or ee[2] < self._min_z - c.stall_eps:
                self._min_z, self._stall = float(ee[2]), 0
            else:
                self._stall += 1
            reached = float(np.linalg.norm(target - ee)) < c.servo_grasp_tol
            if reached or self._stall >= c.stall_patience or self._steps_on_wp >= c.servo_max_steps:
                self._closing = 1
                act[:3] = 0.0
                act[6] = _GRIP_CLOSE
            return act, False

        # place: put the OBJECT on the goal, using the offset it actually has right now
        live_off = ee - obj
        target = goal + live_off + np.array([0.0, 0.0, c.place_dz])
        act[6] = _GRIP_CLOSE
        dxy = target[:2] - ee[:2]
        if (float(np.linalg.norm(dxy)) > c.servo_xy_lock
                and self._steps_on_wp < c.servo_max_steps // 2):
            act[:2] = np.clip(c.kp * dxy, -1.0, 1.0)
            return act, False
        act[:2] = np.clip(c.kp * dxy, -1.0, 1.0)
        act[2] = float(np.clip(c.kp * (target[2] - ee[2]), -c.servo_z_cap, 1.0))
        if self._min_z is None or ee[2] < self._min_z - c.stall_eps:
            self._min_z, self._stall = float(ee[2]), 0
        else:
            self._stall += 1
        done = (float(np.linalg.norm(target - ee)) < c.servo_place_tol
                or self._stall >= c.stall_patience
                or self._steps_on_wp >= c.servo_max_steps)
        return act, done

    # ── execution ───────────────────────────────────────────────────────────
    def act(self, ee_pos, obj_pos, goal_pos, *, obstacle=None, obstacles=None,
            table_z=None, gripper=None):
        """Follow the plan. Signature matches PickPlaceController so the runner is unchanged."""
        from scipy.spatial.transform import Rotation as Rot

        if not self.planned or self._i >= len(self._wps):
            self.phase = "DONE" if self.planned else "PLAN_FAILED"
            return np.array([0, 0, 0, 0, 0, 0, _GRIP_OPEN], float), self.phase

        wp = self._wps[self._i]
        self.phase = wp["phase"]
        ee = np.asarray(ee_pos, float)
        self._steps_on_wp += 1

        # ── closed-loop waypoints ────────────────────────────────────────────
        # Everything else replays a pose fixed at plan time. These two re-derive their target from
        # what is ACTUALLY observed, because they are the two places where centimetre precision
        # decides the outcome and where an open-loop plan measurably fails.
        if wp.get("servo"):
            act, done = self._servo(wp, ee, np.asarray(obj_pos, float),
                                    np.asarray(goal_pos, float), gripper)
            if done:
                self._i += 1
                self._steps_on_wp = 0
                self._stall = 0
                self._min_z = None
            return act, self.phase

        err = wp["pos"] - ee

        reached = float(np.linalg.norm(err)) < self.cfg.reach_tol
        if wp.get("strict") and not reached and self._steps_on_wp < self.cfg.strict_max_steps:
            act = np.zeros(7)                       # keep servoing until we actually arrive
            act[:3] = np.clip(self.cfg.kp * err, -1.0, 1.0)
            if self._rot_ref is not None:
                from scipy.spatial.transform import Rotation as _R
                Rcur = _R.from_quat(np.asarray(self._rot_ref, float)).as_matrix()
                act[3:6] = np.clip(self.cfg.krot *
                                   _R.from_matrix(wp["R"] @ Rcur.T).as_rotvec(), -1.0, 1.0)
            act[6] = wp["grip"]
            return act, self.phase
        if wp.get("strict") and not reached:
            self.reach_failed = True
        if wp["hold"] > 0:
            # A hold waypoint closes or opens the gripper, so it MUST be in position first.
            # Advancing on a step timeout here was closing the fingers 5-6 cm short of the object
            # (measured: 55 mm and 60 mm tracking error, fingers shut to 0.0015 on empty air,
            # object never moved) — the single largest failure mode of this teacher.
            if not reached and self._steps_on_wp < self.cfg.strict_max_steps:
                act = np.zeros(7)
                act[:3] = np.clip(self.cfg.kp * err, -1.0, 1.0)
                if self._rot_ref is not None:
                    from scipy.spatial.transform import Rotation as _R
                    Rcur = _R.from_quat(np.asarray(self._rot_ref, float)).as_matrix()
                    act[3:6] = np.clip(self.cfg.krot *
                                       _R.from_matrix(wp["R"] @ Rcur.T).as_rotvec(), -1.0, 1.0)
                # hold the PREVIOUS gripper command while still travelling — closing early is
                # exactly the bug this branch exists to prevent
                act[6] = self._wps[self._i - 1]["grip"] if self._i > 0 else _GRIP_OPEN
                return act, self.phase
            if self._steps_on_wp >= self.cfg.strict_max_steps and not reached:
                self.reach_failed = True     # could not get to the grasp/release pose
            if self._held >= wp["hold"]:
                self._held = 0
                self._i += 1
                self._steps_on_wp = 0
            else:
                self._held += 1
        elif reached or self._steps_on_wp >= self.cfg.max_steps_per_wp:
            self._i += 1
            self._steps_on_wp = 0

        act = np.zeros(7)
        act[:3] = np.clip(self.cfg.kp * err, -1.0, 1.0)
        if self._rot_ref is not None:
            Rcur = Rot.from_quat(np.asarray(self._rot_ref, float)).as_matrix()
            drot = Rot.from_matrix(wp["R"] @ Rcur.T).as_rotvec()
            act[3:6] = np.clip(self.cfg.krot * drot, -1.0, 1.0)
        act[6] = wp["grip"]
        return act, self.phase

    def set_ee_quat(self, quat):
        self._rot_ref = quat
