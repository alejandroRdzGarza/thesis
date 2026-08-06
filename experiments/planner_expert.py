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
    kp: float = 20.0
    krot: float = 4.0
    descend_steps: int = 6        # interpolation points on the straight-line descents
    ik_seeds: int = 20
    try_candidates: int = 6       # grasps to attempt before giving up on the scene


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

    # PickPlaceController compatibility (the runner sets these; the planner derives its own).
    place_mode: str = "on"
    grasp_mode: str = "top"
    grasp_offset_xy: object = None
    profile_name: str = "planner"
    grasp_offset: object = None
    grasp_offset_vec: object = None
    _rot_ref: object = None      # latest EE quaternion, pushed in by the runner each step

    def reset(self):
        self.phase = "PLAN"
        self.planned = False
        self.plan_error = ""
        self._wps = []
        self._i = 0
        self._steps_on_wp = 0
        self._held = 0

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

        cands = P.sample_grasps(model, data, qadr, sid, obj, body, free,
                                seed=self.seed, ik_seeds=c.ik_seeds)
        if not cands:
            self.plan_error = "no reachable collision-free grasp"
            return False

        obj_qadr = P.free_joint_qadr(model, body)
        _, _, top_z = P.object_extent(model, data, body, obj)

        for cand in cands[:c.try_candidates]:
            wps = []
            p_grasp, R_grasp = cand["pos"], cand["R"]
            p_pre = p_grasp + np.array([0.0, 0.0, c.pre_grasp_h])

            # 1. plan to the pre-grasp hover
            q_pre, ok = P.ik_pose(model, data, qadr, sid, p_pre, R_grasp, free=free,
                                  seeds=c.ik_seeds, seed=self.seed)
            if not ok:
                continue
            path, _why = P.rrt_connect(q_start, q_pre, jnt_rng, free, seed=self.seed)
            if path is None:
                continue
            path = P.densify(P.shortcut(path, free, seed=self.seed), max_step=0.05)
            for p, R in P.path_to_ee_trace(model, data, qadr, sid, path):
                wps.append(dict(pos=p, R=R, grip=_GRIP_OPEN, hold=0, phase="APPROACH"))

            # 2. straight-line descent onto the grasp, 3. close
            for k in range(1, c.descend_steps + 1):
                wps.append(dict(pos=p_pre + (p_grasp - p_pre) * (k / c.descend_steps),
                                R=R_grasp, grip=_GRIP_OPEN, hold=0, phase="DESCEND"))
            wps.append(dict(pos=p_grasp, R=R_grasp, grip=_GRIP_CLOSE,
                            hold=c.grasp_hold, phase="GRASP"))

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
                continue
            # Drive the EE to wherever puts the OBJECT over the goal. The full 3-D grasp offset
            # matters, not just its z: a rim pinch holds the object ~a rim radius to one side, so
            # aiming the EE at the goal leaves the object short by exactly that much (measured:
            # 4.4 cm miss against a 4.7 cm rim radius).
            grasp_off = p_grasp - obj                      # EE − object, at the moment of grasp
            p_place = np.array([goal[0], goal[1], goal[2]]) + grasp_off + np.array([0, 0, 0.01])
            p_preplace = p_place + np.array([0.0, 0.0, c.place_clear_h])
            q_pp, ok = P.ik_pose(model, data, qadr, sid, p_preplace, R_grasp, free=free_held,
                                 seeds=c.ik_seeds, seed=self.seed)
            if not ok:
                continue
            tpath, _why = P.rrt_connect(q_lift, q_pp, jnt_rng, free_held, seed=self.seed)
            if tpath is None:
                continue
            tpath = P.densify(P.shortcut(tpath, free_held, seed=self.seed), max_step=0.05)
            for p, R in P.path_to_ee_trace(model, data, qadr, sid, tpath):
                wps.append(dict(pos=p, R=R, grip=_GRIP_CLOSE, hold=0, phase="TRANSPORT"))

            # 6. set down, 7. release, 8. retreat
            for k in range(1, c.descend_steps + 1):
                wps.append(dict(pos=p_preplace + (p_place - p_preplace) * (k / c.descend_steps),
                                R=R_grasp, grip=_GRIP_CLOSE, hold=0, phase="PLACE"))
            wps.append(dict(pos=p_place, R=R_grasp, grip=_GRIP_OPEN,
                            hold=c.release_hold, phase="RELEASE"))
            wps.append(dict(pos=p_place + np.array([0.0, 0.0, 0.10]), R=R_grasp,
                            grip=_GRIP_OPEN, hold=0, phase="RETREAT"))

            self._wps = wps
            self.planned = True
            self.grasp_mode = cand["mode"]
            self.phase = "APPROACH"
            return True

        self.plan_error = "no candidate produced a complete plan (grasp/transport unreachable)"
        return False

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
        err = wp["pos"] - ee
        self._steps_on_wp += 1

        reached = float(np.linalg.norm(err)) < self.cfg.reach_tol
        if wp["hold"] > 0:
            # a hold waypoint (closing/opening the gripper) advances on a step count, not distance
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
