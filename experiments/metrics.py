"""
Per-step metrics collection and end-of-run summary for one experiment trial.
"""

from __future__ import annotations
import csv
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class StepRecord:
    step: int
    ee_pos: np.ndarray          # end-effector xyz
    min_dist: float             # min distance from any arm link to closest obstacle
    closest_obstacle: str       # name of that obstacle
    closest_body: str           # name of the arm link that is closest
    cbf_triggered: bool
    cbf_correction_norm: float  # ||u_safe - u_nom||
    violation: bool             # min_dist < safety_radius of that obstacle


class MetricsTracker:
    """Accumulates per-step data and computes summary statistics at the end."""

    def __init__(self, scene_name: str, mode: str):
        self.scene_name = scene_name
        self.mode = mode        # "plain" or "cbf"
        self._records: list[StepRecord] = []
        self._prev_ee_pos: np.ndarray | None = None
        self._path_length: float = 0.0
        self._goal_reach_step: int | None = None

    def record(self, rec: StepRecord, goal_pos: np.ndarray, goal_tolerance: float):
        self._records.append(rec)

        # Accumulate path length
        if self._prev_ee_pos is not None:
            self._path_length += float(np.linalg.norm(rec.ee_pos - self._prev_ee_pos))
        self._prev_ee_pos = rec.ee_pos.copy()

        # First step where EE reaches goal
        if (self._goal_reach_step is None
                and np.linalg.norm(rec.ee_pos - goal_pos) < goal_tolerance):
            self._goal_reach_step = rec.step

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
        }

    def save_step_log(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            fieldnames = [
                "step", "ee_x", "ee_y", "ee_z",
                "min_dist", "closest_obstacle", "closest_body",
                "cbf_triggered", "cbf_correction_norm", "violation",
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
                    "closest_obstacle":     r.closest_obstacle,
                    "closest_body":         r.closest_body,
                    "cbf_triggered":        int(r.cbf_triggered),
                    "cbf_correction_norm":  round(r.cbf_correction_norm, 6),
                    "violation":            int(r.violation),
                })

    def save_summary(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        s = self.summary()
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(s.keys()))
            writer.writeheader()
            writer.writerow(s)
