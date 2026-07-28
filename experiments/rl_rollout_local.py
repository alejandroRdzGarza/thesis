"""
rl_rollout_local.py — co-located flow-SDE GRPO rollout collection (Option B, on UCL GPU).

Env + π0.5 model + CBF shield all live in ONE process (no websocket server). For each
initial state ("group") we roll out the current stochastic policy K times under the CBF
shield, recording for every VLA query the denoising chain + logp_old (the policy trace),
then score each rollout with safe_reward and assign GRPO group-relative advantages.

Pipeline (per rollout):
  sample_actions_with_logprob (flow-SDE)  →  QueryTrace(chain, logp_old, sigmas, …)
      →  CBF shield (run_ellipsoid_cbf, numpy)  →  env.step  →  reward + advantages

Outputs (under --out):
  g{gid}/rollout_{k}.h5          trajectory (obs/action) for reference / RWFM fallback
  g{gid}/rollout_{k}_trace.npz   flow-SDE policy trace for the GRPO update
  manifest.csv                   group_id, rollout_id, trace_path, reward, advantage, …
  round_summary.json             success / collision / mean-reward for the round

This reuses the tested harness end-to-end: run_libero_trial (CBF + attribution + reward
metrics) via a policy_fn override, and rl_rollout.run_collection (scoring + manifest).
The only GPU-bound piece is the model forward inside sample_actions_with_logprob.

Run on UCL (headless):
  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.rl_rollout_local \
      --config pi05_libero --checkpoint /path/to/pi05_libero \
      --suite safelibero_object --level II --task 0 \
      --episodes 0 1 2 3 --K 8 --out results_grpo/round0
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def build_policy_fn(pol_lp):
    """Wrap a PolicyWithLogprob into run_libero_trial's policy_fn contract.

    policy_fn(img_rgb, wrist_rgb, state, instruction, num_actions)
        -> (list[np.ndarray(7,)], QueryTrace)
    """
    from experiments.policy_trace import from_flow_sde_roll

    def policy_fn(img_rgb, wrist_rgb, state, instruction, num_actions):
        obs_dict = {
            "observation/image": np.asarray(img_rgb),
            "observation/wrist_image": np.asarray(wrist_rgb),
            "observation/state": np.asarray(state, dtype=np.float64),
            "prompt": instruction,
        }
        roll = pol_lp.infer_with_logprob(obs_dict)
        acts = np.asarray(roll["actions"], dtype=np.float64)      # (ah, 7)
        chunk = [acts[i].copy() for i in range(min(num_actions, len(acts)))]
        # Store the raw model inputs so the GRPO trainer can recompute logp_new
        # (re-applies the policy's input transform → Observation → compute_chain_logp).
        obs_raw = {
            "image": np.asarray(img_rgb, dtype=np.uint8),
            "wrist_image": np.asarray(wrist_rgb, dtype=np.uint8),
            "state": np.asarray(state, dtype=np.float32),
            "prompt": instruction,
        }
        return chunk, from_flow_sde_roll(roll, obs=obs_raw)

    return policy_fn


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf",
                    help="openpi TrainConfig (LoRA) used to build the model/transforms; "
                         "partial load handles a base checkpoint at round 0")
    ap.add_argument("--checkpoint", required=True,
                    help="checkpoint dir (contains params/ + assets/) — base or LoRA-updated")
    ap.add_argument("--suite", default="safelibero_object",
                    help="LIBERO/SafeLIBERO suite name")
    ap.add_argument("--level", default="II", choices=["I", "II"], help="SafeLIBERO safety level")
    ap.add_argument("--task", type=int, default=0, help="task index within the suite")
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="initial-state indices; each is one GRPO group")
    ap.add_argument("--K", type=int, default=8, help="rollouts per group (group size)")
    ap.add_argument("--out", default="results_grpo/round0", help="output directory")
    ap.add_argument("--horizon", type=int, default=300, help="max control steps per rollout")
    ap.add_argument("--replan", type=int, default=5, help="control steps between VLA queries")
    # flow-SDE sampling knobs
    ap.add_argument("--num-steps", type=int, default=10, help="flow-SDE denoising steps")
    ap.add_argument("--noise-level", type=float, default=0.7, help="SDE exploration noise (0 = ODE)")
    ap.add_argument("--sde-type", default="cps", choices=["cps", "sde"])
    ap.add_argument("--seed", type=int, default=0)
    # GRPO / reward knobs
    ap.add_argument("--temperature", type=float, default=1.0, help="RWFM weight temperature")
    ap.add_argument("--no-cbf", action="store_true",
                    help="disable the CBF shield entirely (headline eval: measure learned safety)")
    ap.add_argument("--shield-prob", type=float, default=1.0,
                    help="fraction of each group's K rollouts run WITH the CBF shield; the rest "
                         "run unshielded so real collisions enter the reward (0.5 = half/half). "
                         "1.0 = legacy shielded-only. --no-cbf forces 0.")
    args = ap.parse_args()

    # Imports deferred so --help works without JAX / LIBERO installed.
    from openpi.policies.policy_logprob import PolicyWithLogprob
    from openpi.training import config as _config

    from experiments.libero_runner import make_libero_env
    from experiments.load_policy import create_policy_partial
    from experiments.rl_rollout import run_collection
    from experiments.safe_reward import RewardConfig

    print(f"Loading policy: config={args.config}  checkpoint={args.checkpoint}")
    train_cfg = _config.get_config(args.config)
    # Partial load so a base checkpoint initializes the LoRA config at round 0 and a saved
    # LoRA checkpoint loads at round N (both behave correctly; base gives lora_b=0).
    policy = create_policy_partial(train_cfg, args.checkpoint)
    pol_lp = PolicyWithLogprob(
        policy, num_steps=args.num_steps, noise_level=args.noise_level,
        sde_type=args.sde_type, seed=args.seed,
    )
    policy_fn = build_policy_fn(pol_lp)
    # --no-cbf forces a fully unshielded run (eval); otherwise --shield-prob sets the mix.
    shield_prob = 0.0 if args.no_cbf else args.shield_prob
    print(f"  flow-SDE: num_steps={args.num_steps} noise_level={args.noise_level} "
          f"sde_type={args.sde_type}  shield_prob={shield_prob:.2f} "
          f"({'off' if shield_prob == 0 else ('full' if shield_prob >= 1 else 'mixed')})")

    # run_collection builds the env and forwards run_kwargs (incl. the instruction) to
    # every rollout, so resolve the task language once up front, then hand run_collection
    # a factory that returns (env, init_states) for the same suite/task.
    def make_env_fn():
        env, _language, init_states = make_libero_env(
            task_suite=args.suite, task_idx=args.task,
            safety_level=args.level, horizon=args.horizon,
        )
        return env, init_states

    _probe_env, _lang, _ = make_libero_env(
        task_suite=args.suite, task_idx=args.task,
        safety_level=args.level, horizon=args.horizon,
    )
    try:
        _probe_env.close()
    except Exception:
        pass

    run_kwargs = dict(
        obstacles=[],
        instruction=_lang,
        goal_pos=None,
        auto_goal=True,          # resolve BDDL goal for distance shaping in the reward
        use_geo_success=False,   # success = env.check_success ONLY (authoritative)
        use_cbf=shield_prob > 0,          # per-rollout use_cbf is overridden in collect_group
        vla="pi05",                       # translational-only default matches AEGIS
        auto_detect_obstacle=True,
        aegis_faithful=True,
        replan_steps=args.replan,
        horizon=args.horizon,
        policy_fn=policy_fn,
        record_policy_trace=True,
        scene_name=f"{args.suite}_L{args.level}_t{args.task}",
    )

    Path(args.out).mkdir(parents=True, exist_ok=True)
    summary = run_collection(
        make_env_fn=make_env_fn,
        group_episode_indices=list(args.episodes),
        K=args.K,
        run_kwargs=run_kwargs,
        out_dir=args.out,
        reward_cfg=RewardConfig(),
        temperature=args.temperature,
        shield_prob=shield_prob,
        save_traj=False,          # GRPO uses only the .npz trace; skip the big .h5 (disk quota)
    )
    print(f"\nRound complete → {args.out}")
    print(f"summary: {summary}")


if __name__ == "__main__":
    main()
