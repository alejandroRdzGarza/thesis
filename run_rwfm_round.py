#!/usr/bin/env python
"""
run_rwfm_round.py — Collect ONE round of Reward-Weighted Flow-Matching rollouts.

Runs locally (talks to the π0.5 server over the tunnel, like the benchmark). For
each initial state ("group") it rolls out the current policy K times under the CBF
shield, scores each rollout (safe_reward), computes GRPO group advantages + RWFM
weights, saves each trajectory as HDF5, and writes manifest.csv + round_summary.json.

Prereqs: π0.5 served on UCL + tunnel up (see reference-ucl-server memory).

Example (round 0, base π0.5):
    python run_rwfm_round.py \
        --suite safelibero_spatial --level I --task 0 \
        --groups 8 --K 6 --horizon 400 \
        --out results_rwfm/round0

Then convert + train (see experiments/RWFM_RUNBOOK.md).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from experiments.libero_runner import make_libero_env
from experiments.safe_reward import RewardConfig
from experiments.rl_rollout import run_collection

# Per-(suite,task) placement goal — mirror run_libero_benchmark's table if set there.
_DEFAULT_GOAL = None   # None → run_libero_trial uses its own default / auto goal.


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--suite", default="safelibero_spatial")
    p.add_argument("--level", default="I", choices=["I", "II"])
    p.add_argument("--task", type=int, default=0)
    p.add_argument("--groups", type=int, default=8,
                   help="number of initial states (each rolled out K times)")
    p.add_argument("--group-start", type=int, default=0,
                   help="first episode index used as a group's initial state")
    p.add_argument("--K", type=int, default=6, help="rollouts per group")
    p.add_argument("--horizon", type=int, default=400)
    p.add_argument("--replan-steps", type=int, default=5)
    p.add_argument("--safety-radius", type=float, default=0.18)
    p.add_argument("--pi05-host", default="127.0.0.1")
    p.add_argument("--pi05-port", type=int, default=8000)
    p.add_argument("--no-cbf-shield", action="store_true",
                   help="disable the CBF shield during rollouts (NOT recommended)")
    # reward weights
    p.add_argument("--w-success", type=float, default=1.5)
    p.add_argument("--w-collision", type=float, default=1.0)
    p.add_argument("--w-cbf", type=float, default=0.5)
    p.add_argument("--w-progress", type=float, default=0.3)
    # advantage → weight
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--positive-only", action="store_true",
                   help="filtered-BC: keep only above-average rollouts (weight 0/1-ish)")
    p.add_argument("--out", required=True, help="output dir for this round")
    args = p.parse_args()

    reward_cfg = RewardConfig(
        w_success=args.w_success, w_direct_collision=args.w_collision,
        w_cbf_rate=args.w_cbf, w_progress=args.w_progress,
    )

    def make_env_fn():
        env, lang, init_states = make_libero_env(
            task_suite=args.suite, task_idx=args.task,
            safety_level=args.level, horizon=args.horizon,
        )
        # Stash instruction so run_kwargs can read it (closure).
        make_env_fn.instruction = lang
        return env, init_states

    # Build once to grab the instruction (env is rebuilt inside run_collection).
    _env0, _ = make_env_fn()
    try:
        _env0.close()
    except Exception:
        pass

    run_kwargs = dict(
        obstacles=[],
        instruction=make_env_fn.instruction,
        goal_pos=_DEFAULT_GOAL,
        goal_tolerance=0.15,
        use_cbf=not args.no_cbf_shield,
        auto_detect_obstacle=args.suite.startswith("safelibero_"),
        obstacle_safety_radius=args.safety_radius,
        replan_steps=args.replan_steps,
        horizon=args.horizon,
        vla="pi05",
        pi05_host=args.pi05_host,
        pi05_port=args.pi05_port,
        aegis_faithful=True,          # every-step CBF shield, no heuristics
        save_results=False,
    )

    groups = list(range(args.group_start, args.group_start + args.groups))
    print(f"RWFM round: suite={args.suite} L{args.level} task={args.task}  "
          f"groups={groups}  K={args.K}  shield={not args.no_cbf_shield}")

    run_collection(
        make_env_fn=make_env_fn,
        group_episode_indices=groups,
        K=args.K,
        run_kwargs=run_kwargs,
        out_dir=Path(args.out),
        reward_cfg=reward_cfg,
        temperature=args.temperature,
        positive_only=args.positive_only,
    )


if __name__ == "__main__":
    main()
