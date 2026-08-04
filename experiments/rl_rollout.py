"""
rl_rollout.py — Reward-Weighted Flow-Matching rollout harness (step 2).

Design (disk-export, offline-first): for each initial state ("group"), roll out the
current policy K times under the CBF shield, score each rollout, compute GRPO
group-relative advantages, turn them into per-trajectory loss weights, and write a
manifest. openpi's flow-matching trainer (step 3) then reads the manifest and
weights each trajectory's loss by `weight`.

    group g (initial state s_g)
      ├─ rollout 0 → traj_0.h5, reward R_0
      ├─ …
      └─ rollout K-1 → traj_{K-1}.h5, reward R_{K-1}
    advantages A_k = (R_k - mean_g)/(std_g+ε)   → weights w_k = exp(A_k/τ)
    manifest.csv: group_id, rollout_id, traj_path, reward, advantage, weight, …

The scoring / advantage / manifest logic here is pure and unit-tested with mock
rollouts (no env, no server). `collect_group` / `run_collection` call
run_libero_trial and therefore need the π0.5 server; they're structured so the
GPU-free logic can be tested independently.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

from experiments.safe_reward import (
    RewardConfig, RewardBreakdown, group_advantages, advantage_weights,
    reward_from_metrics,
)


@dataclass
class Rollout:
    group_id: int              # index of the initial state
    rollout_id: int            # k within the group
    traj_path: str             # saved trajectory (LeRobot/hdf5) for RWFM/BC training
    reward: RewardBreakdown
    advantage: float = 0.0
    weight: float = 0.0
    trace_path: str = ""       # flow-SDE policy trace (.npz) for on-policy GRPO; "" if none
    shielded: bool = True      # was the CBF shield active during this rollout?
    # Control step at which the task was completed (-1 if never). This is ETS, the third metric
    # in the AEGIS/VLSA table alongside CAR and TSR — it was already in metrics.summary() as
    # `goal_reach_step` but never reached the manifest, so no eval could report it.
    ets: int = -1

    def manifest_row(self) -> dict:
        rb = self.reward
        return {
            "group_id": self.group_id,
            "rollout_id": self.rollout_id,
            "traj_path": self.traj_path,
            "trace_path": self.trace_path,
            "shielded": int(self.shielded),
            "reward": round(rb.total, 6),
            "advantage": round(self.advantage, 6),
            "weight": round(self.weight, 6),
            "r_success": round(rb.success, 6),
            "r_collision": round(rb.collision, 6),
            "r_cbf": round(rb.cbf, 6),
            "r_progress": round(rb.progress, 6),
            # NB despite the name this carries the PENALISED collision flag, which under the
            # default RewardConfig (penalize_raw_collision=True) is the RAW >1 mm obstacle
            # displacement — the same quantity AEGIS/VLSA report, not the contact-graph
            # attribution. Flipping that config silently changes what this column means.
            "robot_caused_collision": int(rb.direct_collision),
            "ets": self.ets,
        }


# ── GPU-free scoring / advantage / manifest ─────────────────────────────────
def score_group(rollouts: list[Rollout], temperature: float = 1.0,
                positive_only: bool = False) -> list[Rollout]:
    """Assign GRPO group-relative advantages and RWFM loss weights, in place."""
    rewards = [r.reward.total for r in rollouts]
    advs = group_advantages(rewards)
    weights = advantage_weights(advs, temperature=temperature,
                                positive_only=positive_only)
    for r, a, w in zip(rollouts, advs, weights):
        r.advantage = a
        r.weight = w
    return rollouts


def write_manifest(rollouts: list[Rollout], path: str | Path) -> Path:
    """Write the trajectory→weight manifest (CSV) the trainer consumes."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [r.manifest_row() for r in rollouts]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    return path


def collection_summary(rollouts: list[Rollout]) -> dict:
    """Aggregate stats for a collection round — the per-round training log.

    With mixed shielding (shield_prob < 1), the shielded and unshielded subsets measure
    different things and must be reported separately:
      • shielded   rollouts → CBF activation rate (via mean_cbf_penalty): does the raw
        policy still need the shield?  (the headline "internalized safety" proxy)
      • unshielded rollouts → collision rate WITHOUT the net: does the policy actually
        avoid the obstacle on its own?  (the direct safety-learning signal)
    """
    n = len(rollouts)
    if n == 0:
        return {}
    succ = sum(1 for r in rollouts if r.reward.success > 0)
    coll = sum(1 for r in rollouts if r.reward.direct_collision)
    mean_r = sum(r.reward.total for r in rollouts) / n
    # cbf term = -w_cbf * rate → recover mean rate proxy via the cbf reward sign
    mean_cbf_pen = sum(r.reward.cbf for r in rollouts) / n

    shielded = [r for r in rollouts if r.shielded]
    unshielded = [r for r in rollouts if not r.shielded]
    summary = {
        "n_rollouts": n,
        "n_shielded": len(shielded),
        "n_unshielded": len(unshielded),
        "success_rate": round(succ / n, 4),
        "robot_caused_collision_rate": round(coll / n, 4),
        "mean_reward": round(mean_r, 4),
        "mean_cbf_penalty": round(mean_cbf_pen, 4),
        "mean_weight": round(sum(r.weight for r in rollouts) / n, 4),
    }
    # CBF activation is only meaningful where the shield was on.
    if shielded:
        summary["shielded_cbf_penalty"] = round(
            sum(r.reward.cbf for r in shielded) / len(shielded), 4)
        summary["shielded_success_rate"] = round(
            sum(1 for r in shielded if r.reward.success > 0) / len(shielded), 4)
    # The unshielded collision rate is the direct measure of learned obstacle avoidance.
    if unshielded:
        summary["unshielded_collision_rate"] = round(
            sum(1 for r in unshielded if r.reward.direct_collision) / len(unshielded), 4)
        summary["unshielded_success_rate"] = round(
            sum(1 for r in unshielded if r.reward.success > 0) / len(unshielded), 4)
    return summary


# ── Rollout collection (needs the π0.5 server) ──────────────────────────────
def shield_schedule(K: int, shield_prob: float) -> list[bool]:
    """Which of a group's K rollouts run WITH the CBF shield.

    Deterministic split (not random) so every group is guaranteed a mix: the first
    round(K*shield_prob) rollouts are shielded, the rest unshielded. This guarantees
    within-group reward variance between shielded-safe and unshielded-collide/​safe
    rollouts, which is what gives GRPO a real safety gradient.
        shield_prob=1.0 → all shielded (legacy); 0.0 → all unshielded (eval); 0.5 → half.
    """
    n_shielded = int(round(K * max(0.0, min(1.0, shield_prob))))
    return [k < n_shielded for k in range(K)]


def collect_group(
    env,
    group_id: int,
    episode_idx: int,
    K: int,
    run_kwargs: dict,
    out_dir: str | Path,
    reward_cfg: RewardConfig = RewardConfig(),
    shield_prob: float = 1.0,
    save_traj: bool = True,
) -> list[Rollout]:
    """Run K rollouts of one initial state, saving each trajectory + its reward.

    `run_kwargs` are forwarded to run_libero_trial (env, instruction, goal_pos, vla='pi05',
    etc.). `shield_prob` sets the fraction of the K rollouts that run under the CBF shield;
    the remainder run UNSHIELDED so the policy's raw actions actually hit the obstacle and
    the collision penalty enters the reward (the fix for shielded-only masking). Each
    rollout's per-rollout `use_cbf` overrides whatever is in run_kwargs.

    `save_traj`: also write the per-rollout .h5 trajectory (obs/action). The GRPO trainer
    only consumes the .npz policy trace, so the .h5 is dead weight for on-policy GRPO and is
    disabled by default from rl_rollout_local (it was blowing the disk quota).
    """
    from experiments.libero_runner import run_libero_trial
    from experiments.policy_trace import save_episode_trace

    out_dir = Path(out_dir)
    schedule = shield_schedule(K, shield_prob)
    rollouts: list[Rollout] = []
    for k in range(K):
        use_shield = schedule[k]
        traj_path = (str(out_dir / f"g{group_id:04d}" / f"rollout_{k:02d}.h5")
                     if save_traj else "")
        metrics = run_libero_trial(
            env=env,
            episode_idx=episode_idx,
            collect_cbf_data=save_traj,
            cbf_dataset_path=(traj_path or None),
            **{**run_kwargs, "use_cbf": use_shield},
        )
        rb = reward_from_metrics(metrics, reward_cfg)
        # On-policy GRPO: if run_libero_trial recorded a flow-SDE policy trace
        # (metrics.policy_trace, populated on-box when record_policy_trace=True),
        # persist it next to the trajectory for the GRPO update.
        trace_path = ""
        trace = getattr(metrics, "policy_trace", None)
        if trace:
            trace_path = str(out_dir / f"g{group_id:04d}" / f"rollout_{k:02d}_trace.npz")
            save_episode_trace(trace, trace_path)
        try:
            _ets = int(metrics.summary().get("goal_reach_step", -1))
        except Exception:
            _ets = -1
        rollouts.append(Rollout(group_id=group_id, rollout_id=k, traj_path=traj_path,
                                reward=rb, trace_path=trace_path, shielded=use_shield, ets=_ets))
    return rollouts


def run_collection(
    make_env_fn,
    group_episode_indices: list[int],
    K: int,
    run_kwargs: dict,
    out_dir: str | Path,
    reward_cfg: RewardConfig = RewardConfig(),
    temperature: float = 1.0,
    positive_only: bool = False,
    shield_prob: float = 1.0,
    save_traj: bool = True,
) -> dict:
    """One RWFM collection round over several initial states → manifest + summary.

    `make_env_fn()` returns (env, initial_states); each group index selects one
    initial state. `shield_prob` is the fraction of each group's K rollouts run under
    the CBF shield (the rest unshielded — see collect_group). `save_traj` toggles the
    (GRPO-unused) .h5 trajectory dump. Writes `manifest.csv` and `round_summary.json`.
    """
    out_dir = Path(out_dir)
    env, init_states = make_env_fn()
    # SafeLIBERO needs the per-episode init states forwarded so each rollout can
    # set_init_state(episode_idx); inject them into the shared run_kwargs.
    run_kwargs = {**run_kwargs, "initial_states": init_states}
    all_rollouts: list[Rollout] = []
    try:
        for gid, ep_idx in enumerate(group_episode_indices):
            group = collect_group(env, gid, ep_idx, K, run_kwargs, out_dir, reward_cfg,
                                  shield_prob=shield_prob, save_traj=save_traj)
            score_group(group, temperature=temperature, positive_only=positive_only)
            all_rollouts.extend(group)
    finally:
        try:
            env.close()
        except Exception:
            pass

    manifest = write_manifest(all_rollouts, out_dir / "manifest.csv")
    summary = collection_summary(all_rollouts)
    with open(out_dir / "round_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  RWFM round → {manifest}")
    print(f"  summary: {summary}")
    return summary
