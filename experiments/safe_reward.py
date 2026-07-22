"""
safe_reward.py — Reward + advantage for Reward-Weighted Flow-Matching (RWFM).

Method (chosen 2026-07-20): we adapt π0.5 (a flow-matching VLA — no clean action
log-prob, so vanilla GRPO/PPO don't apply) via reward-weighted fine-tuning:

    1. For each initial state, roll out K times under (current policy + CBF shield).
    2. Score each rollout with `compute_episode_reward` below.
    3. Group-relative advantage (GRPO's trick, no value net):
           A_i = (R_i - mean_K) / (std_K + eps)
    4. Fine-tune projector + action-head LoRA with the flow-matching loss, each
       trajectory weighted by exp(A / temperature) (or filter to A > 0).

This module is the pure, GPU-free core: the reward function, its config, and the
advantage computation. It has NO torch/jax dependency and is fully unit-testable.

Reward shape:
    R =  w_success * task_success
       - w_direct_collision * DIRECT_collision      # robot's own fault only
       - w_cbf_rate * cbf_activation_rate           # penalise needing the filter
       + w_progress * shaped_progress               # dense signal so groups vary

Collision attribution (2026-07-20): a displacement is penalised iff the ROBOT
caused it — directly OR through a push chain (robot → nudges neighbour → neighbour
shoves obstacle). This is `collision_robot_caused` from
libero_runner.robot_caused_displacement (contact-graph reachability, static
world/table excluded). It generalises to multi-obstacle scenes and still excludes
pure physics-settling displacement. If that signal is unavailable, we fall back to
the culprit-category heuristic (gripper/arm_link/held_object → penalise).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Iterable


# Culprit categories (from _classify_contact_body) that count as the robot's fault.
_DEFAULT_DIRECT_CULPRITS = frozenset({"gripper", "arm_link", "held_object", "robot_other"})


@dataclass(frozen=True)
class RewardConfig:
    w_success: float = 1.5
    # Collision penalty. Kept BELOW w_success + w_progress on purpose: a successful
    # but collided rollout must still beat a clean failure (0), otherwise the policy
    # learns to avoid the task entirely to dodge collisions. Clean success (1.8) still
    # dominates collided success (0.8), so the gradient favours *safe* completion.
    w_direct_collision: float = 1.0
    w_cbf_rate: float = 0.5
    w_progress: float = 0.3
    # Partial credit (fraction of w_progress) for grasping the target at all.
    grasp_credit: float = 0.4
    # Distance (m) at which goal proximity credit saturates to full.
    goal_dist_scale: float = 0.30
    direct_culprits: frozenset = field(default_factory=lambda: _DEFAULT_DIRECT_CULPRITS)
    # Penalise ANY raw collision (obstacle displaced >2mm). Default True: a still-arm
    # drift test showed the active obstacle is physically stable, so every raw
    # displacement is robot-caused, and the contact-graph `robot_caused` attribution
    # under-counts (misses delayed/indirect pushes) — using it would let real
    # collisions go unpenalised. Set False to fall back to the robot_caused/culprit logic.
    penalize_raw_collision: bool = True


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    success: float
    collision: float
    cbf: float
    progress: float
    direct_collision: bool     # was the penalised (robot-caused) collision triggered?

    def as_dict(self) -> dict:
        return asdict(self)


def _parse_culprits(collision_culprit) -> set[str]:
    """Accept a '|'-joined string, an iterable, or None → set of categories."""
    if not collision_culprit:
        return set()
    if isinstance(collision_culprit, str):
        return {c for c in collision_culprit.split("|") if c}
    return set(collision_culprit)


def is_direct_collision(collision_detected: bool, collision_culprit,
                        cfg: RewardConfig = RewardConfig()) -> bool:
    """True iff a collision occurred AND a robot body was the attributed culprit.

    A collision attributed only to `scene_object` (indirect domino displacement)
    is NOT counted as direct — the policy can't avoid it.
    """
    if not collision_detected:
        return False
    culprits = _parse_culprits(collision_culprit)
    if not culprits:
        # Collision detected but no attribution captured → be conservative and
        # treat as direct (we lack evidence it was indirect).
        return True
    return bool(culprits & cfg.direct_culprits)


def _progress_term(goal_reached: bool, grasp_achieved: bool,
                   goal_dist_final, cfg: RewardConfig) -> float:
    """Dense shaping in [0, 1] so within-group rewards vary even before success."""
    if goal_reached:
        return 1.0
    prog = 0.0
    if grasp_achieved:
        prog += cfg.grasp_credit
    if goal_dist_final is not None:
        # closer to goal → more credit, saturating at goal_dist_scale.
        near = max(0.0, 1.0 - float(goal_dist_final) / max(cfg.goal_dist_scale, 1e-6))
        prog += (1.0 - cfg.grasp_credit) * near
    return min(prog, 1.0)


def compute_episode_reward(
    *,
    goal_reached: bool,
    collision_detected: bool,
    collision_robot_caused: bool | None = None,
    collision_culprit=None,
    cbf_activation_rate: float = 0.0,
    grasp_achieved: bool = False,
    goal_dist_final: float | None = None,
    cfg: RewardConfig = RewardConfig(),
) -> RewardBreakdown:
    """Scalar reward for one finished episode (used as the RWFM trajectory weight).

    A collision is penalised iff the robot caused it. `collision_robot_caused`
    (contact-graph causation, incl. domino pushes) takes precedence; if it is None
    we fall back to the culprit-category heuristic via `collision_culprit`.
    """
    if cfg.penalize_raw_collision:
        direct = bool(collision_detected)                      # any >2mm displacement
    elif collision_robot_caused is not None:
        direct = bool(collision_detected and collision_robot_caused)
    else:
        direct = is_direct_collision(collision_detected, collision_culprit, cfg)

    r_success   = cfg.w_success * (1.0 if goal_reached else 0.0)
    r_collision = -cfg.w_direct_collision * (1.0 if direct else 0.0)
    r_cbf       = -cfg.w_cbf_rate * float(max(0.0, min(1.0, cbf_activation_rate)))
    r_progress  = cfg.w_progress * _progress_term(
        goal_reached, grasp_achieved, goal_dist_final, cfg)

    total = r_success + r_collision + r_cbf + r_progress
    return RewardBreakdown(
        total=total, success=r_success, collision=r_collision,
        cbf=r_cbf, progress=r_progress, direct_collision=direct,
    )


def reward_from_metrics(metrics, cfg: RewardConfig = RewardConfig()) -> RewardBreakdown:
    """Convenience: build the reward from a run_libero_trial MetricsTracker."""
    s = metrics.summary()
    return compute_episode_reward(
        goal_reached=bool(s.get("goal_reached")),
        collision_detected=bool(s.get("collision_detected")),
        collision_robot_caused=getattr(metrics, "collision_robot_caused", None),
        collision_culprit=getattr(metrics, "collision_culprit", ""),
        cbf_activation_rate=float(s.get("cbf_activation_rate", 0.0) or 0.0),
        grasp_achieved=bool(s.get("grasp_achieved")),
        goal_dist_final=s.get("goal_dist_final"),
        cfg=cfg,
    )


def group_advantages(rewards: Iterable[float], eps: float = 1e-6) -> list[float]:
    """GRPO group-relative advantage: standardise rewards within a K-rollout group.

    A_i = (R_i - mean) / (std + eps). No value network. A degenerate group (all
    rewards equal → std 0) yields all-zero advantages, which correctly contribute
    no gradient.
    """
    r = list(rewards)
    n = len(r)
    if n == 0:
        return []
    mean = sum(r) / n
    var = sum((x - mean) ** 2 for x in r) / n
    std = var ** 0.5
    return [(x - mean) / (std + eps) for x in r]


def advantage_weights(advantages: Iterable[float], temperature: float = 1.0,
                      clip: float = 10.0, positive_only: bool = False) -> list[float]:
    """Turn advantages into non-negative trajectory weights for the RWFM loss.

    weight_i = exp(clip(A_i / temperature, -clip, clip)); if positive_only, drop
    A_i <= 0 (filtered behaviour cloning). Weights are the multiplier on each
    trajectory's flow-matching loss.
    """
    import math
    out = []
    for a in advantages:
        if positive_only and a <= 0:
            out.append(0.0)
            continue
        z = max(-clip, min(clip, a / max(temperature, 1e-6)))
        out.append(math.exp(z))
    return out
