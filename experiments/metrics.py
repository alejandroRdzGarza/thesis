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
    # ── Metrics fields (always populated, written to CSV) ───────────────────
    step: int
    ee_pos: np.ndarray          # (3,) end-effector xyz
    min_dist: float             # min distance from any arm link to closest obstacle
    closest_obstacle: str       # name of that obstacle
    closest_body: str           # name of the arm link that is closest
    cbf_triggered: bool
    cbf_correction_norm: float  # ||u_safe - u_nom||
    violation: bool             # min_dist < safety_radius of that obstacle

    # ── Phase-1 dataset fields (always populated, written to .npz) ──────────
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


class MetricsTracker:
    """Accumulates per-step data and computes summary statistics at the end."""

    def __init__(self, scene_name: str, mode: str):
        self.scene_name = scene_name
        self.mode = mode        # "plain" or "cbf"
        self._records: list[StepRecord] = []
        self._prev_ee_pos: np.ndarray | None = None
        self._path_length: float = 0.0
        self._goal_reach_step: int | None = None
        self._collision_step: int | None = None

    def record(self, rec: StepRecord, goal_pos: np.ndarray, goal_tolerance: float):
        self._records.append(rec)

        if self._prev_ee_pos is not None:
            self._path_length += float(np.linalg.norm(rec.ee_pos - self._prev_ee_pos))
        self._prev_ee_pos = rec.ee_pos.copy()

        if (self._goal_reach_step is None
                and np.linalg.norm(rec.ee_pos - goal_pos) < goal_tolerance):
            self._goal_reach_step = rec.step

        if self._collision_step is None and rec.collision_flag:
            self._collision_step = rec.step

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

        min_dists   = [r.min_dist for r in self._records]
        violations  = [r for r in self._records if r.violation]
        activations = [r for r in self._records if r.cbf_triggered]
        corrections = [r.cbf_correction_norm for r in activations]

        return {
            "scene":                    self.scene_name,
            "mode":                     self.mode,
            "total_steps":              len(self._records),
            "min_dist_overall":         round(min(min_dists), 4),
            "mean_min_dist":            round(float(np.mean(min_dists)), 4),
            "violation_steps":          len(violations),
            "violation_rate":           round(len(violations) / len(self._records), 4),
            "cbf_activations":          len(activations),
            "cbf_activation_rate":      round(len(activations) / len(self._records), 4),
            "cbf_mean_correction_norm": round(float(np.mean(corrections)), 4) if corrections else 0.0,
            "cbf_max_correction_norm":  round(float(max(corrections)), 4)  if corrections else 0.0,
            "path_length_m":            round(self._path_length, 4),
            "goal_reached":             self._goal_reach_step is not None,
            "goal_reach_step":          self._goal_reach_step if self._goal_reach_step is not None else -1,
            "collision_detected":       self._collision_step is not None,
            "collision_step":           self._collision_step if self._collision_step is not None else -1,
        }

    def get_records(self) -> list:
        """Return the raw per-step records for diagnostic analysis."""
        return self._records

    def save_step_log(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            fieldnames = [
                "step", "ee_x", "ee_y", "ee_z",
                "min_dist", "h_min",
                "closest_obstacle", "closest_body",
                "cbf_triggered", "cbf_correction_norm", "violation", "collision_flag",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in self._records:
                writer.writerow({
                    "step":                 r.step,
                    "ee_x":                 round(float(r.ee_pos[0]), 4),
                    "ee_y":                 round(float(r.ee_pos[1]), 4),
                    "ee_z":                 round(float(r.ee_pos[2]), 4),
                    "min_dist":             round(r.min_dist, 4),
                    "h_min":                round(float(min(r.h_values)), 4),
                    "closest_obstacle":     r.closest_obstacle,
                    "closest_body":         r.closest_body,
                    "cbf_triggered":        int(r.cbf_triggered),
                    "cbf_correction_norm":  round(r.cbf_correction_norm, 6),
                    "violation":            int(r.violation),
                    "collision_flag":       int(r.collision_flag),
                })

    def save_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(s.keys()))
            writer.writeheader()
            writer.writerow(s)

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
