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
    obj_name = goal_state[0][1]                       # the manipulated object (predicate arg 1)
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
    }


# Gripper commands in OSC_POSE / π0.5 convention: +1 = close, −1 = open.
_GRIP_OPEN, _GRIP_CLOSE = -1.0, 1.0


@dataclass
class ControllerConfig:
    kp: float = 20.0            # P-gain; delta = clip(kp·error, ±1). ~1/output_max (OSC ~0.05 m/step).
    pos_tol: float = 0.02       # m, waypoint-reached tolerance
    xy_tol: float = 0.015       # m, tighter XY alignment before descending / placing
    approach_h: float = 0.12    # m above the object/goal to hover before descending
    grasp_dz: float = 0.005     # m above the object centre to close the gripper
    lift_h: float = 0.18        # m to lift the grasped object before transporting
    place_dz: float = 0.04      # m above the goal to release
    grasp_hold: int = 8         # control steps to hold while the gripper closes
    release_hold: int = 5       # control steps to hold open after releasing


@dataclass
class PickPlaceController:
    """Phase state machine → 7-D OSC_POSE nominal action. Caller adds CBF safety."""
    cfg: ControllerConfig = field(default_factory=ControllerConfig)
    phase: str = "APPROACH"
    _timer: int = 0

    PHASES = ("APPROACH", "DESCEND", "GRASP", "LIFT", "TRANSPORT", "PLACE", "RELEASE", "DONE")

    def reset(self):
        self.phase = "APPROACH"
        self._timer = 0

    def _goto(self, ee_pos, target, grip) -> np.ndarray:
        """P-control EE delta toward `target` (world frame), zero rotation, given gripper cmd."""
        delta = np.clip(self.cfg.kp * (np.asarray(target) - np.asarray(ee_pos)), -1.0, 1.0)
        return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, grip], dtype=np.float64)

    def act(self, ee_pos, obj_pos, goal_pos, *, table_z: float | None = None) -> tuple[np.ndarray, str]:
        """One control step. `obj_pos` is the live object position (moves once grasped).
        Returns (action7, phase). Phase 'DONE' means the task should be complete."""
        c = self.cfg
        ee = np.asarray(ee_pos, dtype=float)
        obj = np.asarray(obj_pos, dtype=float)
        goal = np.asarray(goal_pos, dtype=float)
        base_z = table_z if table_z is not None else obj[2]

        if self.phase == "APPROACH":                      # hover above the object, gripper open
            tgt = np.array([obj[0], obj[1], obj[2] + c.approach_h])
            if np.linalg.norm(ee[:2] - obj[:2]) < c.xy_tol and abs(ee[2] - tgt[2]) < c.pos_tol:
                self.phase = "DESCEND"
            return self._goto(ee, tgt, _GRIP_OPEN), self.phase

        if self.phase == "DESCEND":                       # lower onto the object, still open
            tgt = np.array([obj[0], obj[1], obj[2] + c.grasp_dz])
            if np.linalg.norm(ee - tgt) < c.pos_tol:
                self.phase, self._timer = "GRASP", 0
            return self._goto(ee, tgt, _GRIP_OPEN), self.phase

        if self.phase == "GRASP":                         # hold + close the gripper
            self._timer += 1
            if self._timer >= c.grasp_hold:
                self.phase = "LIFT"
            return self._goto(ee, ee, _GRIP_CLOSE), self.phase

        if self.phase == "LIFT":                           # raise the grasped object
            tgt = np.array([ee[0], ee[1], base_z + c.lift_h])
            if ee[2] >= base_z + c.lift_h - c.pos_tol:
                self.phase = "TRANSPORT"
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "TRANSPORT":                      # carry to above the goal (obstacle-avoiding transit)
            tgt = np.array([goal[0], goal[1], goal[2] + c.approach_h])
            if np.linalg.norm(ee[:2] - goal[:2]) < c.xy_tol and abs(ee[2] - tgt[2]) < c.pos_tol:
                self.phase = "PLACE"
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "PLACE":                          # lower to the goal surface
            tgt = np.array([goal[0], goal[1], goal[2] + c.place_dz])
            if np.linalg.norm(ee - tgt) < c.pos_tol:
                self.phase, self._timer = "RELEASE", 0
            return self._goto(ee, tgt, _GRIP_CLOSE), self.phase

        if self.phase == "RELEASE":                        # open + hold
            self._timer += 1
            if self._timer >= c.release_hold:
                self.phase = "DONE"
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
