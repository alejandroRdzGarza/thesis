"""
record_expert_demos.py — render the shield-corrected "expert" rollouts to mp4 for visual
inspection. The whole point of Exp 005's pivot rests on these demos being GOOD; if the VLA+CBF
trajectories are hesitant / fight the shield / barely succeed, imitating them can't produce a
good policy (supervisor's "check the demos" point). This lets you watch them.

Reuses the exact rollout stack (create_policy_partial + PolicyWithLogprob + run_libero_trial),
so what you see is precisely what flow_bc_train imitates.

Examples (on the pod):
  # the EXPERT (VLA + CBF shield) — what BC imitates:
  $PY -m experiments.record_expert_demos --checkpoint $CKPT --episodes 0 1 2 3 --out-dir demos
  # the BASE policy WITHOUT the shield, for comparison (does the shield help or hurt?):
  $PY -m experiments.record_expert_demos --checkpoint $CKPT --episodes 0 1 2 3 --no-cbf --out-dir demos
Then scp the mp4s to your Mac to watch.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf")
    ap.add_argument("--checkpoint", required=True, help="policy checkpoint (base or a round ckpt)")
    ap.add_argument("--suite", default="safelibero_object")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out-dir", default="expert_demos")
    ap.add_argument("--no-cbf", action="store_true",
                    help="record WITHOUT the shield (base/student behavior) for comparison")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--noise-level", type=float, default=0.0,
                    help="0 = deterministic ODE (cleanest for inspection)")
    ap.add_argument("--sde-type", default="cps", choices=["cps", "sde"])
    args = ap.parse_args()

    from openpi.policies.policy_logprob import PolicyWithLogprob
    from openpi.training import config as _config

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.load_policy import create_policy_partial
    from experiments.rl_rollout_local import build_policy_fn

    print(f"Loading policy: {args.checkpoint}", flush=True)
    train_cfg = _config.get_config(args.config)
    policy = create_policy_partial(train_cfg, args.checkpoint)
    pol_lp = PolicyWithLogprob(policy, num_steps=args.num_steps, noise_level=args.noise_level,
                               sde_type=args.sde_type, seed=0)
    policy_fn = build_policy_fn(pol_lp)

    env, lang, init_states = make_libero_env(
        task_suite=args.suite, task_idx=args.task, safety_level=args.level, horizon=args.horizon)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = "nocbf" if args.no_cbf else "cbf"
    print(f"\nRecording {len(args.episodes)} demos ({tag}) → {out}/\n", flush=True)

    results = []
    for ep in args.episodes:
        vid = str(out / f"{args.suite}_L{args.level}_t{args.task}_ep{ep}_{tag}.mp4")
        m = run_libero_trial(
            env=env, episode_idx=ep, instruction=lang, initial_states=init_states,
            obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
            use_cbf=not args.no_cbf, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
            replan_steps=args.replan, horizon=args.horizon,
            policy_fn=policy_fn, record_policy_trace=False,
            scene_name=f"{args.suite}_L{args.level}_t{args.task}",
            save_video=vid,
        )
        s = m.summary()
        results.append((ep, bool(s["goal_reached"]), bool(s["collision_detected"]),
                        s.get("cbf_activations"), vid))
        print(f"  ep{ep} [{tag}]: TSR={s['goal_reached']}  collision={s['collision_detected']}  "
              f"CBF_acts={s.get('cbf_activations')}  → {vid}", flush=True)

    try:
        env.close()
    except Exception:
        pass

    n = len(results)
    succ = sum(1 for r in results if r[1])
    coll = sum(1 for r in results if r[2])
    print(f"\nSummary ({tag}): success {succ}/{n}  collided {coll}/{n}  → videos in {out}/", flush=True)
    print("  scp them to your Mac to watch, e.g.:")
    print(f"  scp -P <PORT> -i ~/.ssh/id_ed25519 'root@<IP>:{out.resolve()}/*.mp4' .", flush=True)


if __name__ == "__main__":
    main()
