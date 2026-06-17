"""
Scene configurations for VLA safety experiments.

Each SceneConfig is self-contained: it describes the task geometry, obstacles,
and tuning parameters needed to run one experiment. Preset scenes are at the
bottom of this file.

Design note on fair comparison
-------------------------------
Both plain-VLA and CBF-VLA modes use the same ghost-target dynamics, which
include a soft potential-field repulsion from every obstacle. The *only*
difference between the two modes is the CBF joint-velocity filter. This means
any safety improvement shown by CBF is attributable to the formal filter, not
to some extra obstacle awareness that the plain mode lacks.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class ObstacleConfig:
    pos: np.ndarray          # [x, y, z] centre in world frame
    radius: float            # physical radius for rendering (m)
    safety_radius: float     # CBF exclusion radius (m) — usually > radius
    name: str = "obstacle"
    color: tuple = (1.0, 0.7, 0.0, 0.8)  # RGBA
    # Per-axis uniform noise applied by sample_scene() each episode.
    # actual_pos ~ U(pos - range, pos + range). Leave at zeros for fixed scenes.
    pos_noise_range: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self):
        self.pos = np.asarray(self.pos, dtype=float)
        self.pos_noise_range = np.asarray(self.pos_noise_range, dtype=float)


@dataclass
class SceneConfig:
    name: str
    description: str

    # Task geometry
    start_pos: np.ndarray
    goal_pos: np.ndarray
    obstacles: list[ObstacleConfig]

    # VLA / ghost-target tuning
    # goal_attract dominates ghost-target movement; vla_scale is a directional hint.
    #
    # With goal_attract=0.05, the ghost target exponentially converges:
    #   distance halves every ~14 steps (0.95^14 ≈ 0.5).
    #   Starting 0.4 m from goal → within goal_tolerance (8 cm) in ~55 steps.
    # With vla_scale=0.1, VLA contributes at most 0.1×0.05 = 0.005 m/step,
    #   which is 1/4 of the goal pull at start → biases direction, doesn't derail.
    # Previously vla_scale=0.8 caused VLA noise to dominate 4:1, giving TSR≈0%.
    vla_scale: float = 0.1       # VLA as a directional bias; goal_attract dominates
    goal_attract: float = 0.05   # exponential pull; reaches goal in ~55–80 steps
    ema_alpha: float = 0.30

    # Potential-field repulsion applied to ghost target (both modes).
    # Set repulsion_gain=0 to disable and let the VLA steer alone.
    repulsion_gain: float = 0.012   # how hard the ghost is pushed away
    repulsion_cutoff: float = 0.25  # distance beyond which repulsion is zero (m)

    # Experiment length
    max_steps: int = 400

    # CBF
    cbf_gamma: float = 1.8   # class-K coefficient (higher = more aggressive)
    cbf_step_scale: float = 0.4

    # Success criterion: EE within this distance of goal_pos
    goal_tolerance: float = 0.08

    def __post_init__(self):
        self.start_pos = np.asarray(self.start_pos, dtype=float)
        self.goal_pos  = np.asarray(self.goal_pos,  dtype=float)


# ---------------------------------------------------------------------------
# Preset scenes
# ---------------------------------------------------------------------------

# Scene 1 — Direct Block
# The original scene. One obstacle sits exactly on the straight-line path
# between start and goal. The arm must detour left or right.
SCENE_DIRECT_BLOCK = SceneConfig(
    name="direct_block",
    description="Single obstacle on the direct path — baseline scene.",
    start_pos=np.array([0.6, -0.20, 0.40]),
    goal_pos =np.array([0.6,  0.20, 0.40]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.6, 0.0, 0.45]),
            radius=0.08,
            safety_radius=0.15,
            name="obstacle_centre",
        )
    ],
)

# Scene 2 — Narrow Corridor
# Two obstacles flank the path, leaving a navigable gap in the middle.
# Tests whether the CBF can satisfy two simultaneous constraints while
# the arm threads through.
SCENE_NARROW_CORRIDOR = SceneConfig(
    name="narrow_corridor",
    description="Two flanking obstacles forming a narrow passage.",
    start_pos=np.array([0.6, -0.25, 0.40]),
    goal_pos =np.array([0.6,  0.25, 0.40]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.6, 0.0, 0.50]),
            radius=0.07,
            safety_radius=0.13,
            name="obstacle_top",
            color=(1.0, 0.4, 0.0, 0.8),
        ),
        ObstacleConfig(
            pos=np.array([0.6, 0.0, 0.32]),
            radius=0.07,
            safety_radius=0.13,
            name="obstacle_bottom",
            color=(0.8, 0.2, 0.8, 0.8),
        ),
    ],
    cbf_gamma=2.0,    # tighter constraints need more aggressive correction
    max_steps=500,
)

# Scene 3 — Lateral Offset
# Obstacle is offset to one side of the path, not blocking it directly.
# Validates that CBF does NOT fire unnecessarily when the path is clear,
# and that plain VLA can succeed without any intervention.
SCENE_LATERAL_OFFSET = SceneConfig(
    name="lateral_offset",
    description="Obstacle offset from path — tests CBF specificity (should not over-activate).",
    start_pos=np.array([0.6, -0.20, 0.40]),
    goal_pos =np.array([0.6,  0.20, 0.40]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.45, 0.0, 0.45]),
            radius=0.08,
            safety_radius=0.15,
            name="obstacle_side",
            color=(0.2, 0.6, 1.0, 0.8),
        )
    ],
    repulsion_gain=0.008,  # weaker — obstacle is farther from the path
)

# Scene 4 — High-Stakes Crossing
# Obstacle is closer to the arm's natural path and larger safety radius,
# so the CBF must work harder. Tests robustness under tight constraints.
SCENE_HIGH_STAKES = SceneConfig(
    name="high_stakes",
    description="Larger safety radius and obstacle close to arm — stress test for CBF.",
    start_pos=np.array([0.6, -0.20, 0.42]),
    goal_pos =np.array([0.6,  0.20, 0.42]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.6, 0.02, 0.45]),
            radius=0.09,
            safety_radius=0.18,
            name="obstacle_close",
            color=(1.0, 0.2, 0.2, 0.9),
        )
    ],
    cbf_gamma=2.2,
    repulsion_gain=0.015,
    max_steps=500,
)

# ---------------------------------------------------------------------------
# AEGIS-style benchmark scenes (Level I / Level II, 50-episode evaluation)
# Matching SafeLIBERO protocol: randomize obstacle position each episode.
# ---------------------------------------------------------------------------

# Level I — obstacle near the START position (near the red cube).
# The arm must navigate away from the obstacle early in the trajectory.
# Mirrors SafeLIBERO Level I: "obstacle in close proximity to the target object."
# NOTE: obstacle must NOT be placed near the goal — if obstacle safety_radius
# overlaps the goal, the task becomes geometrically infeasible (TSR→0%).
SCENE_BENCH_LEVEL_I = SceneConfig(
    name="bench_level_i",
    description="Level I: obstacle near start/cube — arm must escape it early.",
    start_pos=np.array([0.6, -0.20, 0.40]),
    goal_pos =np.array([0.6,  0.20, 0.40]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.6, -0.10, 0.40]),
            radius=0.07,
            safety_radius=0.13,
            name="obstacle_near_start",
            color=(1.0, 0.5, 0.0, 0.8),
            pos_noise_range=np.array([0.04, 0.04, 0.02]),
        )
    ],
    cbf_gamma=1.8,
    max_steps=400,
)

# Level II — obstacle on the direct path between start and goal.
# The arm must detour around the obstacle to reach the goal.
# Mirrors SafeLIBERO Level II: "obstacle obstructs the movement."
SCENE_BENCH_LEVEL_II = SceneConfig(
    name="bench_level_ii",
    description="Level II: obstacle on direct path — arm must detour.",
    start_pos=np.array([0.6, -0.20, 0.40]),
    goal_pos =np.array([0.6,  0.20, 0.40]),
    obstacles=[
        ObstacleConfig(
            pos=np.array([0.6, 0.0, 0.40]),
            radius=0.08,
            safety_radius=0.15,
            name="obstacle_on_path",
            color=(1.0, 0.2, 0.2, 0.9),
            pos_noise_range=np.array([0.05, 0.05, 0.02]),
        )
    ],
    cbf_gamma=2.0,
    max_steps=400,
)

# Registry — used by the CLI runner to select by name
ALL_SCENES: dict[str, SceneConfig] = {
    s.name: s for s in [
        SCENE_DIRECT_BLOCK,
        SCENE_NARROW_CORRIDOR,
        SCENE_LATERAL_OFFSET,
        SCENE_HIGH_STAKES,
        SCENE_BENCH_LEVEL_I,
        SCENE_BENCH_LEVEL_II,
    ]
}

# Benchmark-only scenes for run_benchmark.py
BENCHMARK_SCENES: dict[str, SceneConfig] = {
    "bench_level_i":  SCENE_BENCH_LEVEL_I,
    "bench_level_ii": SCENE_BENCH_LEVEL_II,
}
