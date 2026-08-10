"""rollout_bon.py — best-of-N safety selection, evaluated against a matched K=1 control.

THE COMPARISON. Selection needs DISTINCT candidates, so the sampler must be stochastic. Every
other arm in this project ran at --noise-level 0, a deterministic ODE, where all K samples would
be identical and best-of-N a no-op. A stochastic policy also performs differently from a
deterministic one, so the existing base numbers are NOT the right reference. The control is
therefore K=1 at the SAME noise level:

    --k 1   control     stochastic sampling, no selection
    --k 4   best-of-N   same sampler, safest candidate executed

Anything else would confound selection with simply turning sampling noise on.

CHOOSING THE NOISE LEVEL, which is measurable rather than a guess. Too low and the K candidates are
near-identical, so selection has nothing to choose between; too high and they are all bad. The
diagnostic is candidate DIVERSITY, printed for the first few queries: how many of the K were
collision-free, and the spread of their displacements. If every query yields K safe or K unsafe
candidates, selection cannot act and the noise level is wrong — that is a property of the setup,
not a result about safety.

THE HEADLINE DIAGNOSTIC is no_safe_rate: the fraction of queries where NO candidate avoided
contact. It separates two very different worlds and should be read before TSR or CAR:
  low  -> safety is a SAMPLING problem. The policy can behave safely but does not reliably pick
          it, so rejection sampling and then distilling the selections should work.
  high -> safety is a CAPABILITY problem. No safe behaviour exists to select, and no amount of
          selection or self-distillation creates one.

  python -m experiments.rollout_bon --checkpoint $CK --suite safelibero_spatial --level II \
      --task 0 --episodes 35 36 37 38 39 --k 4 --noise-level 0.7 --out results_bon/k4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, nargs="+", default=[35, 36, 37, 38, 39])
    ap.add_argument("--k", type=int, default=4,
                    help="candidates per query. 1 = the matched control (same sampler, no selection)")
    ap.add_argument("--noise-level", type=float, default=0.7,
                    help="sampler stochasticity. MUST be > 0 for k > 1: at 0 the sampler is a "
                         "deterministic ODE and all K candidates are identical.")
    ap.add_argument("--score-full-chunk", action="store_true",
                    help="score all H actions rather than the executed prefix (ablation)")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if args.k > 1 and args.noise_level <= 0:
        raise SystemExit("--k > 1 with --noise-level 0 is a no-op: the sampler is deterministic, "
                         "so all K candidates are identical. Use --noise-level 0.7 or similar.")

    from openpi.training import config as _config
    from openpi.policies.policy_logprob import PolicyWithLogprob
    from experiments.libero_runner import (make_libero_env, run_libero_trial,
                                           detect_safelibero_obstacle)
    from experiments.load_policy import create_policy_partial
    from experiments.rl_rollout_local import build_policy_fn
    from experiments.best_of_n import BestOfNSelector, verify_state_restore

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.suite}_L{args.level}_t{args.task}"
    print(f"\n=== {tag}  K={args.k}  noise={args.noise_level}  episodes={args.episodes} ===",
          flush=True)

    cfg = _config.get_config(args.config)
    policy = create_policy_partial(cfg, args.checkpoint)
    pol_lp = PolicyWithLogprob(policy, num_steps=args.num_steps,
                               noise_level=args.noise_level, sde_type="cps", seed=0)
    base_policy_fn = build_policy_fn(pol_lp)

    env, lang, inits = make_libero_env(task_suite=args.suite, task_idx=args.task,
                                       safety_level=args.level, horizon=args.horizon)
    env.reset(); env.set_init_state(inits[args.episodes[0]])

    # GATE. Every candidate must be scored from an identical world; a lossy rewind would make the
    # selection a measurement of simulator drift. Measured exact (4.4e-16) once the robot and
    # gripper objects are snapshotted alongside the sim state.
    if not verify_state_restore(env):
        raise SystemExit("simulator rewind is lossy — fix before trusting any best-of-N number")

    rows = []
    for ep in args.episodes:
        env.reset(); obs0 = env.set_init_state(inits[ep])
        ob = detect_safelibero_obstacle(env, obs0)
        body = getattr(ob, "body_name", None) or getattr(ob, "name", None) if ob else None
        print(f"  ep{ep}: obstacle body = {body}", flush=True)
        sel = BestOfNSelector(env, base_policy_fn, k=args.k, exec_steps=args.replan,
                              score_full_chunk=args.score_full_chunk, obstacle_body=body)

        # Candidate diversity on the first few queries: if every query yields K safe or K unsafe,
        # selection cannot act and the NOISE LEVEL is wrong — a property of the setup, not a
        # result about safety. Reported before any outcome number so it cannot be read backwards.
        inner, seen = sel.__call__, {"n": 0}
        def traced(*a, **k):
            r = inner(*a, **k)
            if seen["n"] < 3:
                ks = sel.stats["k_safe"][-1]
                sys.stderr.write(f"    [diversity] query {seen['n']+1}: "
                                 f"{ks}/{args.k} candidates safe\n"); sys.stderr.flush()
                seen["n"] += 1
            return r

        import contextlib, io
        with contextlib.redirect_stdout(io.StringIO()):
            m = run_libero_trial(
                env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                use_cbf=False, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                replan_steps=args.replan, horizon=args.horizon,
                policy_fn=traced, record_policy_trace=False, scene_name=tag)

        s, gs = m.summary(), sel.summary()
        rows.append({"episode": ep, "success": int(bool(s["goal_reached"])),
                     "collision": int(bool(s["collision_detected"])), **gs})
        print(f"  ep{ep}: TSR={rows[-1]['success']} coll={rows[-1]['collision']}  "
              f"no_safe {gs['no_safe_rate']:.0%}  mean safe {gs['mean_safe_candidates']:.2f}/{args.k}"
              f"  all-K-safe {gs['all_k_safe_rate']:.0%}", flush=True)

    n = len(rows)
    summ = {"scene": tag, "k": args.k, "noise_level": args.noise_level, "n": n,
            "tsr": sum(r["success"] for r in rows) / n,
            "collision": sum(r["collision"] for r in rows) / n,
            "car": 1 - sum(r["collision"] for r in rows) / n,
            "no_safe_rate": float(np.mean([r["no_safe_rate"] for r in rows])),
            "mean_safe_candidates": float(np.mean([r["mean_safe_candidates"] for r in rows])),
            "rows": rows}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))

    print(f"\n  TSR {summ['tsr']:.0%}   CAR {summ['car']:.0%}")
    print(f"  no_safe_rate {summ['no_safe_rate']:.1%}   "
          f"mean safe candidates {summ['mean_safe_candidates']:.2f}/{args.k}")
    if args.k > 1:
        msc = summ["mean_safe_candidates"]
        if msc >= args.k - 1e-6:
            print("  NOTE: every candidate was safe on essentially every query. Selection had "
                  "nothing to choose\n  between — lower the bar (longer chunk scoring) or accept "
                  "that this scene is not contested.")
        elif msc <= 1e-6:
            print("  NOTE: no candidate was ever safe. This is the CAPABILITY case: selection "
                  "cannot help, and\n  distilling selections would not either. Report it as such.")
    print(f"  -> {out}/summary.json\n")


if __name__ == "__main__":
    main()
