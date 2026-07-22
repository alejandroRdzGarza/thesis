"""
Per-step metrics collection and end-of-run summary for one experiment trial.

StepRecord holds both the lightweight metrics fields (written to CSV) and
the richer Phase-1 dataset fields (written to .npz by save_dataset()).
"""

from __future__ import annotations
import csv
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StepRecord:
    # ── Core metrics (always populated, written to CSV) ─────────────────────
    step: int
    ee_pos: np.ndarray          # (3,) end-effector xyz
    min_dist: float             # min distance from any arm link to closest obstacle
    closest_obstacle: str       # name of that obstacle
    closest_body: str           # name of the arm link that is closest
    cbf_triggered: bool
    cbf_correction_norm: float  # ||u_safe - u_nom||
    violation: bool             # min_dist < safety_radius of that obstacle

    # ── Phase-1 dataset fields ───────────────────────────────────────────────
    q:         np.ndarray       # (7,)  joint positions at this step
    u_nom:     np.ndarray       # (7,)  nominal joint velocity (IK output, pre-CBF)
    u_safe:    np.ndarray       # (7,)  post-CBF velocity (= u_nom in plain mode)
    h_values:  list             # [float] min h(q) per obstacle (positive = safe)
    vla_delta: np.ndarray       # (7,)  raw VLA action [dx,dy,dz,droll,dpitch,dyaw,grip]

    # ── Fields with defaults (must follow all non-default fields) ───────────
    collision_flag: bool = False  # SafeLIBERO ground-truth: obstacle displaced > 0.001 m
    ghost_pos: np.ndarray = None

    # ── Optional (only when collect_dataset=True) ────────────────────────────
    image: np.ndarray | None = None   # (224, 224, 3) uint8, static_cam frame

    # ── CBF fine-tuning dataset fields (only when collect_cbf_data=True) ─────
    wrist_image:    np.ndarray | None = None  # (224, 224, 3) uint8
    eef_quat:       np.ndarray | None = None  # (4,) xyzw
    gripper_qpos:   np.ndarray | None = None  # (2,)
    safe_cartesian: np.ndarray | None = None  # (7,) CBF-corrected Cartesian action

    # ── Rich research logging fields ─────────────────────────────────────────
    goal_dist:           float = float("inf")  # ||ee_pos - goal_pos|| (m); inf if no goal
    gripper_open:        bool  = True          # True when gripper action <= 0 (open)
    vla_query:           bool  = False         # True on the step a new VLA chunk arrived
    action_nom_xyz_mag:  float = 0.0           # ||u_nom[:3]||
    action_safe_xyz_mag: float = 0.0           # ||u_safe[:3]||
    obstacle_pos: np.ndarray | None = None     # (3,) closest obstacle centroid
    ee_vel:       np.ndarray | None = None     # (3,) ee_pos[t]-ee_pos[t-1]; set by tracker
    # Failure-mode analysis fields
    grasp_detected: bool  = False              # True when arm has confirmed object grasp
    obj_dist:       float = float("inf")       # ||ee_pos - closest_target_object|| (m)


# Threshold for deadlock detection: EE displacement over a window
_DEADLOCK_THRESH_M  = 0.003   # less than 3 mm movement
_DEADLOCK_WINDOW    = 20      # over 20 consecutive steps


class MetricsTracker:
    """Accumulates per-step data and computes summary statistics at the end."""

    def __init__(self, scene_name: str, mode: str, model_name: str = ""):
        self.scene_name = scene_name
        self.mode = mode        # "plain" or "cbf"
        self.model_name = model_name
        self._records: list[StepRecord] = []
        self._prev_ee_pos: np.ndarray | None = None
        self._path_length: float = 0.0
        self._goal_reach_step: int | None = None
        self._collision_step: int | None = None

        # Jerk tracking (||u_nom[:3]_t - u_nom[:3]_{t-1}||²)
        self._prev_u_nom: np.ndarray | None = None
        self._jerk_series: list[float] = []

        # Gripper transition tracking
        self._prev_gripper_open: bool | None = None
        self._gripper_close_transitions: int = 0   # open -> close
        self._gripper_open_transitions: int = 0    # close -> open
        self._first_close_step: int | None = None

        # Deadlock detection
        self._deadlock_window: list[np.ndarray] = []
        self._deadlock_steps: int = 0

        # CBF intervention run lengths and h values
        self._cbf_active_count: int = 0
        self._cbf_active_runs: list[int] = []
        self._cbf_h_when_active: list[float] = []

        # VLA query count
        self._vla_queries: int = 0

        # Grasp and object-distance tracking
        self._first_grasp_step: int | None = None
        self._obj_dist_at_grasp: float = float("inf")

    def record(self, rec: StepRecord, goal_pos: np.ndarray, goal_tolerance: float):
        # Compute EE velocity and set on record
        if self._prev_ee_pos is not None:
            rec.ee_vel = rec.ee_pos - self._prev_ee_pos
            self._path_length += float(np.linalg.norm(rec.ee_vel))
        else:
            rec.ee_vel = np.zeros(3, dtype=np.float32)
        self._prev_ee_pos = rec.ee_pos.copy()

        self._records.append(rec)

        # Goal reach (EE-distance fallback; primary path is mark_goal_reached)
        if (self._goal_reach_step is None
                and goal_tolerance > 0
                and np.linalg.norm(rec.ee_pos - goal_pos) < goal_tolerance):
            self._goal_reach_step = rec.step

        if self._collision_step is None and rec.collision_flag:
            self._collision_step = rec.step

        # Jerk: squared difference of consecutive nominal XYZ commands
        if self._prev_u_nom is not None:
            delta = rec.u_nom[:3] - self._prev_u_nom[:3]
            self._jerk_series.append(float(np.dot(delta, delta)))
        self._prev_u_nom = rec.u_nom.copy()

        # Gripper transitions
        if self._prev_gripper_open is not None:
            if self._prev_gripper_open and not rec.gripper_open:   # open -> close
                self._gripper_close_transitions += 1
                if self._first_close_step is None:
                    self._first_close_step = rec.step
            elif not self._prev_gripper_open and rec.gripper_open: # close -> open
                self._gripper_open_transitions += 1
        self._prev_gripper_open = rec.gripper_open

        # Deadlock detection: EE moves < 3 mm over a 20-step window
        self._deadlock_window.append(rec.ee_pos.copy())
        if len(self._deadlock_window) > _DEADLOCK_WINDOW:
            self._deadlock_window.pop(0)
            window_spread = float(np.max([
                np.linalg.norm(p - self._deadlock_window[0])
                for p in self._deadlock_window
            ]))
            if window_spread < _DEADLOCK_THRESH_M:
                self._deadlock_steps += 1

        # CBF intervention runs and h-value tracking
        if rec.cbf_triggered:
            self._cbf_active_count += 1
            if rec.h_values:
                self._cbf_h_when_active.append(float(min(rec.h_values)))
        else:
            if self._cbf_active_count > 0:
                self._cbf_active_runs.append(self._cbf_active_count)
                self._cbf_active_count = 0

        if rec.vla_query:
            self._vla_queries += 1

        # First confirmed grasp and object distance at that moment
        if rec.grasp_detected and self._first_grasp_step is None:
            self._first_grasp_step = rec.step
            self._obj_dist_at_grasp = rec.obj_dist

    def mark_goal_reached(self, step: int):
        """Explicitly mark the goal as reached at a given step.

        Used when success is detected via info['success'] rather than EE
        position distance (e.g. SafeLIBERO where goal_pos is None).
        """
        if self._goal_reach_step is None:
            self._goal_reach_step = step

    def summary(self) -> dict:
        if not self._records:
            return {}

        # Close out any in-progress CBF run
        if self._cbf_active_count > 0:
            self._cbf_active_runs.append(self._cbf_active_count)

        min_dists   = [r.min_dist for r in self._records]
        violations  = [r for r in self._records if r.violation]
        activations = [r for r in self._records if r.cbf_triggered]
        corrections = [r.cbf_correction_norm for r in activations]

        # Path efficiency: straight-line dist / total path length
        start_pos = self._records[0].ee_pos
        end_pos   = self._records[-1].ee_pos
        straight  = float(np.linalg.norm(end_pos - start_pos))
        path_eff  = round(straight / self._path_length, 4) if self._path_length > 1e-4 else 1.0

        # Goal distance stats
        goal_dists = [r.goal_dist for r in self._records if r.goal_dist < 1e8]

        # Object distance stats
        obj_dists = [r.obj_dist for r in self._records if r.obj_dist < 1e8]

        return {
            "scene":                      self.scene_name,
            "mode":                       self.mode,
            "model_name":                 self.model_name,
            "total_steps":                len(self._records),
            # Safety
            "min_dist_overall":           round(min(min_dists), 4),
            "mean_min_dist":              round(float(np.mean(min_dists)), 4),
            "violation_steps":            len(violations),
            "violation_rate":             round(len(violations) / len(self._records), 4),
            # CBF
            "cbf_activations":            len(activations),
            "cbf_activation_rate":        round(len(activations) / len(self._records), 4),
            "cbf_mean_correction_norm":   round(float(np.mean(corrections)), 4) if corrections else 0.0,
            "cbf_max_correction_norm":    round(float(max(corrections)), 4)      if corrections else 0.0,
            "cbf_mean_h_when_active":     round(float(np.mean(self._cbf_h_when_active)), 4)
                                          if self._cbf_h_when_active else float("inf"),
            "cbf_intervention_mean_dur":  round(float(np.mean(self._cbf_active_runs)), 2)
                                          if self._cbf_active_runs else 0.0,
            "cbf_intervention_max_dur":   int(max(self._cbf_active_runs))
                                          if self._cbf_active_runs else 0,
            # Trajectory
            "path_length_m":              round(self._path_length, 4),
            "path_efficiency":            path_eff,
            "total_jerk":                 round(float(sum(self._jerk_series)), 6)        if self._jerk_series else 0.0,
            "mean_jerk":                  round(float(np.mean(self._jerk_series)), 6)    if self._jerk_series else 0.0,
            # Gripper
            "gripper_close_transitions":  self._gripper_close_transitions,
            "gripper_open_transitions":   self._gripper_open_transitions,
            "first_close_step":           self._first_close_step if self._first_close_step is not None else -1,
            # Goal
            "goal_dist_min":              round(min(goal_dists), 4) if goal_dists else float("inf"),
            "goal_dist_final":            round(goal_dists[-1], 4)  if goal_dists else float("inf"),
            "goal_reached":               self._goal_reach_step is not None,
            "goal_reach_step":            self._goal_reach_step if self._goal_reach_step is not None else -1,
            # Deadlock and VLA
            "deadlock_steps":             self._deadlock_steps,
            "vla_queries":                self._vla_queries,
            # Collision
            "collision_detected":         self._collision_step is not None,
            "collision_step":             self._collision_step if self._collision_step is not None else -1,
            # Grasp
            "grasp_achieved":             self._first_grasp_step is not None,
            "first_grasp_step":           self._first_grasp_step if self._first_grasp_step is not None else -1,
            "obj_dist_at_grasp":          round(self._obj_dist_at_grasp, 4) if self._obj_dist_at_grasp < 1e8 else float("inf"),
            # Object distance
            "obj_dist_min":               round(min(obj_dists), 4) if obj_dists else float("inf"),
            "obj_dist_final":             round(obj_dists[-1], 4) if obj_dists else float("inf"),
        }

    def get_records(self) -> list:
        """Return the raw per-step records for diagnostic analysis."""
        return self._records

    def save_step_log(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            # Position and velocity
            "step", "ee_x", "ee_y", "ee_z", "ee_vx", "ee_vy", "ee_vz",
            # Goal
            "goal_dist",
            # Safety
            "min_dist", "h_min",
            "closest_obstacle", "closest_body",
            "cbf_triggered", "cbf_correction_norm", "violation", "collision_flag",
            # Action
            "gripper_open", "vla_query",
            "action_nom_xyz_mag", "action_safe_xyz_mag",
            # Raw VLA action components
            "vla_dx", "vla_dy", "vla_dz",
            "vla_droll", "vla_dpitch", "vla_dyaw", "vla_grip",
            # Obstacle
            "obs_x", "obs_y", "obs_z",
            # Failure-mode analysis
            "grasp_detected", "obj_dist",
        ]
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                vel = r.ee_vel if r.ee_vel is not None else np.zeros(3)
                vla = r.vla_delta if r.vla_delta is not None else np.zeros(7)
                obs_p = r.obstacle_pos
                writer.writerow({
                    "step":                 r.step,
                    "ee_x":                 round(float(r.ee_pos[0]), 5),
                    "ee_y":                 round(float(r.ee_pos[1]), 5),
                    "ee_z":                 round(float(r.ee_pos[2]), 5),
                    "ee_vx":                round(float(vel[0]), 5),
                    "ee_vy":                round(float(vel[1]), 5),
                    "ee_vz":                round(float(vel[2]), 5),
                    "goal_dist":            round(float(r.goal_dist), 5) if r.goal_dist < 1e8 else "",
                    "min_dist":             round(r.min_dist, 5),
                    "h_min":                round(float(min(r.h_values)), 5) if r.h_values else "",
                    "closest_obstacle":     r.closest_obstacle,
                    "closest_body":         r.closest_body,
                    "cbf_triggered":        int(r.cbf_triggered),
                    "cbf_correction_norm":  round(r.cbf_correction_norm, 6),
                    "violation":            int(r.violation),
                    "collision_flag":       int(r.collision_flag),
                    "gripper_open":         int(r.gripper_open),
                    "vla_query":            int(r.vla_query),
                    "action_nom_xyz_mag":   round(r.action_nom_xyz_mag, 6),
                    "action_safe_xyz_mag":  round(r.action_safe_xyz_mag, 6),
                    "vla_dx":               round(float(vla[0]), 6),
                    "vla_dy":               round(float(vla[1]), 6),
                    "vla_dz":               round(float(vla[2]), 6),
                    "vla_droll":            round(float(vla[3]), 6),
                    "vla_dpitch":           round(float(vla[4]), 6),
                    "vla_dyaw":             round(float(vla[5]), 6),
                    "vla_grip":             round(float(vla[6]), 6),
                    "obs_x":               round(float(obs_p[0]), 4) if obs_p is not None else "",
                    "obs_y":               round(float(obs_p[1]), 4) if obs_p is not None else "",
                    "obs_z":               round(float(obs_p[2]), 4) if obs_p is not None else "",
                    "grasp_detected":      int(r.grasp_detected),
                    "obj_dist":            round(float(r.obj_dist), 5) if r.obj_dist < 1e8 else "",
                })

    def save_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(s.keys()))
            writer.writeheader()
            writer.writerow(s)

    def save_analysis_npz(self, path: str | Path):
        """Save full trajectory arrays for offline analysis.

        Saved arrays (all shape (T, ...)):
            ee_pos              (T, 3)   float32
            ee_vel              (T, 3)   float32
            u_nom               (T, 7)   float32
            u_safe              (T, 7)   float32
            vla_delta           (T, 7)   float32
            h_values            (T, K)   float32  (K = max obstacles; inf-padded)
            q                   (T, 7)   float32
            cbf_triggered       (T,)     bool
            violation           (T,)     bool
            collision_flag      (T,)     bool
            min_dist            (T,)     float32
            cbf_correction_norm (T,)     float32
            goal_dist           (T,)     float32
            gripper_open        (T,)     bool
            vla_query           (T,)     bool
            action_nom_xyz_mag  (T,)     float32
            action_safe_xyz_mag (T,)     float32
            obstacle_pos        (T, 3)   float32  (nan if no obstacle)
            jerk_series         (T-1,)   float32
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        T = len(self._records)
        if T == 0:
            return

        # Stack h_values with inf-padding to handle variable obstacle counts
        hv_lists = [r.h_values if r.h_values else [float("inf")] for r in self._records]
        K = max(len(h) for h in hv_lists)
        h_arr = np.full((T, K), fill_value=float("inf"), dtype=np.float32)
        for i, h in enumerate(hv_lists):
            h_arr[i, :len(h)] = h

        obs_pos_arr = np.full((T, 3), fill_value=float("nan"), dtype=np.float32)
        for i, r in enumerate(self._records):
            if r.obstacle_pos is not None:
                obs_pos_arr[i] = r.obstacle_pos

        arrays = {
            "ee_pos":               np.stack([r.ee_pos for r in self._records]).astype(np.float32),
            "ee_vel":               np.stack([
                                        r.ee_vel if r.ee_vel is not None else np.zeros(3)
                                        for r in self._records
                                    ]).astype(np.float32),
            "u_nom":                np.stack([r.u_nom for r in self._records]).astype(np.float32),
            "u_safe":               np.stack([r.u_safe for r in self._records]).astype(np.float32),
            "vla_delta":            np.stack([r.vla_delta for r in self._records]).astype(np.float32),
            "h_values":             h_arr,
            "q":                    np.stack([r.q for r in self._records]).astype(np.float32),
            "cbf_triggered":        np.array([r.cbf_triggered for r in self._records]),
            "violation":            np.array([r.violation for r in self._records]),
            "collision_flag":       np.array([r.collision_flag for r in self._records]),
            "min_dist":             np.array([r.min_dist for r in self._records], dtype=np.float32),
            "cbf_correction_norm":  np.array([r.cbf_correction_norm for r in self._records], dtype=np.float32),
            "goal_dist":            np.array([r.goal_dist for r in self._records], dtype=np.float32),
            "gripper_open":         np.array([r.gripper_open for r in self._records]),
            "vla_query":            np.array([r.vla_query for r in self._records]),
            "action_nom_xyz_mag":   np.array([r.action_nom_xyz_mag for r in self._records], dtype=np.float32),
            "action_safe_xyz_mag":  np.array([r.action_safe_xyz_mag for r in self._records], dtype=np.float32),
            "obstacle_pos":         obs_pos_arr,
            "jerk_series":          np.array(self._jerk_series, dtype=np.float32),
            "grasp_detected":       np.array([r.grasp_detected for r in self._records]),
            "obj_dist":             np.array([r.obj_dist for r in self._records], dtype=np.float32),
        }

        np.savez_compressed(path, **arrays)

    def save_dataset(self, path: str | Path):
        """Save Phase-1 training dataset as a compressed .npz file.

        Arrays saved:
            q           (T, 7)       joint positions
            u_nom       (T, 7)       nominal joint velocity (pre-CBF)
            u_safe      (T, 7)       CBF-filtered velocity (= u_nom in plain mode)
            h_values    (T, n_obs)   min CBF barrier value per obstacle
            vla_delta   (T, 7)       raw VLA action
            ee_pos      (T, 3)       end-effector position
            cbf_triggered (T,)       bool
            violation   (T,)         bool
            min_dist    (T,)         float
            images      (T,224,224,3) uint8  — only if images were captured
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        arrays = {
            "q":             np.stack([r.q         for r in self._records]).astype(np.float32),
            "u_nom":         np.stack([r.u_nom     for r in self._records]).astype(np.float32),
            "u_safe":        np.stack([r.u_safe    for r in self._records]).astype(np.float32),
            "h_values":      np.array([r.h_values  for r in self._records], dtype=np.float32),
            "vla_delta":     np.stack([r.vla_delta for r in self._records]).astype(np.float32),
            "ee_pos":        np.stack([r.ee_pos    for r in self._records]).astype(np.float32),
            "cbf_triggered": np.array([r.cbf_triggered for r in self._records]),
            "violation":     np.array([r.violation      for r in self._records]),
            "min_dist":      np.array([r.min_dist       for r in self._records], dtype=np.float32),
        }

        if any(r.image is not None for r in self._records):
            arrays["images"] = np.stack([r.image for r in self._records]).astype(np.uint8)

        np.savez_compressed(path, **arrays)

    def save_hdf5(self, path: str | Path, instruction: str, demo_key: str = "demo_0"):
        """Save CBF fine-tuning dataset as LIBERO-compatible HDF5.

        Matches the format expected by OpenVLA-OFT's fine-tuning scripts:
          data/<demo_key>/obs/agentview_image        (T, 224, 224, 3) uint8
          data/<demo_key>/obs/robot0_eye_in_hand_image (T, 224, 224, 3) uint8
          data/<demo_key>/obs/robot0_eef_pos         (T, 3)  float32
          data/<demo_key>/obs/robot0_eef_quat        (T, 4)  float32
          data/<demo_key>/obs/robot0_gripper_qpos    (T, 2)  float32
          data/<demo_key>/actions                    (T, 7)  float32  ← CBF-corrected
          data/<demo_key>/vla_actions                (T, 7)  float32  ← raw VLA
          data/<demo_key>/cbf_triggered              (T,)    bool
          data/<demo_key>/h_values                   (T,)    float32
          data.attrs["num_demos"]  = 1
          data.attrs["env_name"]   = scene_name
        """
        import h5py
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        recs = [r for r in self._records if r.safe_cartesian is not None]
        if not recs:
            return

        with h5py.File(path, "w") as f:
            grp = f.require_group(f"data/{demo_key}")

            obs = grp.require_group("obs")
            if recs[0].image is not None:
                obs.create_dataset("agentview_image",
                    data=np.stack([r.image for r in recs]).astype(np.uint8),
                    compression="lzf")
            if recs[0].wrist_image is not None:
                obs.create_dataset("robot0_eye_in_hand_image",
                    data=np.stack([r.wrist_image for r in recs]).astype(np.uint8),
                    compression="lzf")
            obs.create_dataset("robot0_eef_pos",
                data=np.stack([r.ee_pos for r in recs]).astype(np.float32))
            if recs[0].eef_quat is not None:
                obs.create_dataset("robot0_eef_quat",
                    data=np.stack([r.eef_quat for r in recs]).astype(np.float32))
            if recs[0].gripper_qpos is not None:
                obs.create_dataset("robot0_gripper_qpos",
                    data=np.stack([r.gripper_qpos for r in recs]).astype(np.float32))

            grp.create_dataset("actions",
                data=np.stack([r.safe_cartesian for r in recs]).astype(np.float32))
            grp.create_dataset("vla_actions",
                data=np.stack([r.vla_delta for r in recs]).astype(np.float32))
            grp.create_dataset("cbf_triggered",
                data=np.array([r.cbf_triggered for r in recs]))
            grp.create_dataset("h_values",
                data=np.array([r.h_values[0] if r.h_values else float("inf")
                               for r in recs], dtype=np.float32))

            grp.attrs["language_instruction"] = instruction
            grp.attrs["num_steps"] = len(recs)

            f["data"].attrs["num_demos"] = 1
            f["data"].attrs["env_name"]  = self.scene_name
