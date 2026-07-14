"""
Grasp Verification & Recovery (GVR).

Detects two failure modes that suppress TSR in VLA + CBF pipelines:

  PHANTOM GRASP  — Gripper closes on air near the target.
    Signal: gripper command = +1 for T_PHANTOM consecutive steps with no
            object lifted above its initial table height.
    Recovery: force-open gripper → let VLA re-align → force-close with
              downward nudge.

  HOVERING OSCILLATION  — VLA enters OOD state near the object after
    CBF deflection and outputs micro-corrections that circle the EE
    without committing to a downward grasp.
    Signal: EE XY within hover_radius of the closest graspable object
            for T_HOVER consecutive steps, no grasp confirmed.
    Recovery: inject downward velocity nudge + force gripper close.

Usage
-----
    from experiments.gvr import GVR

    gvr = GVR()

    # Call once per episode before the step loop:
    gvr.reset()

    # Inside the step loop, after CBF filtering, before env.step():
    safe_action, gvr_log = gvr.update(
        action        = safe_action,
        ee_pos        = ee_pos,
        obs           = obs,
        obj_pos_keys  = _obj_pos_keys,
        obj_initial_z = _obj_initial_z,
        grasp_flag    = _grasp_flag,
        obstacle_keys = _obstacle_key_set,
    )
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from enum import Enum


class _Phase(Enum):
    NORMAL          = "normal"
    PHANTOM_OPEN    = "phantom_open"    # forcing gripper open + slight lift
    PHANTOM_WAIT    = "phantom_wait"    # gripper open, VLA re-aligns
    PHANTOM_CLOSE   = "phantom_close"   # forcing close + downward nudge
    HOVER_RESCUE    = "hover_rescue"    # downward nudge + force close


@dataclass
class GVR:
    """Grasp Verification & Recovery state machine."""

    # ── Detection thresholds ────────────────────────────────────────────
    phantom_threshold:   int   = 25     # consecutive closing steps before phantom fires
    hover_threshold:     int   = 40     # consecutive hover steps before rescue fires
    hover_radius:        float = 0.12   # 12 cm EE-to-object XY to count as hovering
    phantom_prox_radius: float = 0.15   # phantom only fires when EE within 15 cm of object
    lift_threshold:      float = 0.015  # 1.5 cm object rise to confirm grasp

    # ── Recovery durations (steps) ───────────────────────────────────────
    open_steps:    int = 5    # phase 1: force gripper open
    wait_steps:    int = 8    # phase 2: gripper stays open, VLA repositions
    close_steps:   int = 6    # phase 3: force close + downward nudge
    rescue_steps:  int = 6    # hover rescue: downward nudge + force close

    # ── Recovery magnitudes ──────────────────────────────────────────────
    open_lift:      float = 0.005  # upward delta per step during OPEN (m)
    close_nudge:    float = 0.008  # downward delta per step during CLOSE (m)
    rescue_nudge:   float = 0.010  # downward delta per step during HOVER_RESCUE (m)

    # ── Internal state (not exposed in __init__) ─────────────────────────
    _phase:          _Phase = field(default_factory=lambda: _Phase.NORMAL, init=False, repr=False)
    _phase_step:     int    = field(default=0, init=False, repr=False)
    _phantom_count:  int    = field(default=0, init=False, repr=False)
    _hover_count:    int    = field(default=0, init=False, repr=False)

    def reset(self) -> None:
        """Reset all state — call at the start of every episode."""
        self._phase         = _Phase.NORMAL
        self._phase_step    = 0
        self._phantom_count = 0
        self._hover_count   = 0

    def update(
        self,
        action:        np.ndarray,        # safe 7-D action (after CBF)
        ee_pos:        np.ndarray,        # world EE position (3,)
        obs:           dict,              # current observation dict
        obj_pos_keys:  list[str],         # non-robot, non-obstacle _pos keys
        obj_initial_z: dict[str, float],  # initial z per obj key
        grasp_flag:    bool,              # True if an object is currently lifted
        obstacle_keys: set[str] | None = None,  # keys to exclude from hover scan
    ) -> tuple[np.ndarray, str]:
        """
        Apply GVR logic and return (modified_action, phase_label).

        Does not mutate the input array — returns a copy.
        """
        action = action.copy()

        if obstacle_keys is None:
            obstacle_keys = set()

        # ── Evaluate environment signals ──────────────────────────────────
        any_lifted      = self._check_any_lifted(obs, obj_pos_keys, obj_initial_z, obstacle_keys)
        gripper_closing = float(action[6]) > 0
        closest_xy      = self._closest_graspable_xy(ee_pos, obs, obj_pos_keys, obstacle_keys)

        # ── Update counters and trigger transitions in NORMAL phase ───────
        if self._phase == _Phase.NORMAL:
            # Phantom: gripper closing with nothing lifted AND EE close to an
            # object (proximity guard prevents counting during the far-field
            # approach where the VLA outputs close-gripper prematurely).
            ee_near_obj = (closest_xy is not None and closest_xy < self.phantom_prox_radius)
            if gripper_closing and not any_lifted and not grasp_flag and ee_near_obj:
                self._phantom_count += 1
            else:
                self._phantom_count = 0

            # Hover: EE circling near an object (at grasping height), no grasp
            if (not grasp_flag
                    and closest_xy is not None
                    and closest_xy < self.hover_radius
                    and ee_pos[2] < 1.05):      # below placement height
                self._hover_count += 1
            else:
                self._hover_count = 0

            if self._phantom_count >= self.phantom_threshold:
                self._enter_phase(_Phase.PHANTOM_OPEN)
            elif self._hover_count >= self.hover_threshold:
                self._enter_phase(_Phase.HOVER_RESCUE)

        # ── Abort recovery immediately if any lift is detected ───────────
        # any_lifted fires as soon as an object rises 1.5 cm above its
        # initial z — 1 step before grasp_flag (which also requires the
        # gripper to be closing).  Checking both prevents GVR from forcing
        # the gripper open the very step a real grasp is happening.
        if (any_lifted or grasp_flag) and self._phase != _Phase.NORMAL:
            self._enter_phase(_Phase.NORMAL)
            return action, self._phase.value

        # ── Execute recovery overrides ────────────────────────────────────
        if self._phase == _Phase.PHANTOM_OPEN:
            action[6]   = -1.0                   # force open
            action[2]  += self.open_lift         # slight upward clear
            self._phase_step += 1
            if self._phase_step >= self.open_steps:
                self._enter_phase(_Phase.PHANTOM_WAIT)

        elif self._phase == _Phase.PHANTOM_WAIT:
            action[6] = -1.0                     # keep gripper open
            self._phase_step += 1
            if self._phase_step >= self.wait_steps:
                self._enter_phase(_Phase.PHANTOM_CLOSE)

        elif self._phase == _Phase.PHANTOM_CLOSE:
            action[6]  =  1.0                    # force close
            action[2] -= self.close_nudge        # nudge toward object
            self._phase_step += 1
            if self._phase_step >= self.close_steps:
                self._enter_phase(_Phase.NORMAL)

        elif self._phase == _Phase.HOVER_RESCUE:
            action[2] -= self.rescue_nudge       # push down toward object
            action[6]  =  1.0                   # force gripper close
            self._phase_step += 1
            if self._phase_step >= self.rescue_steps:
                self._enter_phase(_Phase.NORMAL)

        return action, self._phase.value

    # ── Private helpers ───────────────────────────────────────────────────

    def _enter_phase(self, phase: _Phase) -> None:
        self._phase      = phase
        self._phase_step = 0

    def _check_any_lifted(
        self,
        obs:           dict,
        obj_pos_keys:  list[str],
        obj_initial_z: dict[str, float],
        obstacle_keys: set[str],
    ) -> bool:
        for k in obj_pos_keys:
            if k in obstacle_keys or k not in obs or k not in obj_initial_z:
                continue
            if float(obs[k][2]) - obj_initial_z[k] > self.lift_threshold:
                return True
        return False

    def _closest_graspable_xy(
        self,
        ee_pos:        np.ndarray,
        obs:           dict,
        obj_pos_keys:  list[str],
        obstacle_keys: set[str],
    ) -> float | None:
        """XY distance to the nearest non-obstacle object, or None."""
        best: float | None = None
        for k in obj_pos_keys:
            if k in obstacle_keys or k not in obs:
                continue
            obj_pos = np.array(obs[k], dtype=float)
            # Skip objects at goal height (plate/target) — those aren't graspable targets
            if obj_pos[2] > 1.02:
                continue
            xy = float(np.linalg.norm(ee_pos[:2] - obj_pos[:2]))
            if best is None or xy < best:
                best = xy
        return best

    @property
    def phase(self) -> str:
        return self._phase.value

    @property
    def phantom_count(self) -> int:
        return self._phantom_count

    @property
    def hover_count(self) -> int:
        return self._hover_count
