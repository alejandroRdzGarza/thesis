"""rollout_guided.py — evaluate CBF-GUIDED sampling against post-hoc projection.

THE EXPERIMENT. The shield currently projects: pi0.5 emits an action, the QP pushes it onto the
safe set. That costs capability — the distilled policy scores TSR 82.5% unshielded and 70.8% with
the shield stacked on, because projection moves a coherent action off the policy's manifold.
Guidance instead steers the denoising velocity, so the sample stays on the manifold and the safe
action is one the policy itself would plausibly have produced.

    --lam 0.0   control      unguided, THROUGH THE SAME CODE PATH
    --lam 0.5   half         partial correction, closest to the policy's own behaviour
    --lam 1.0   full         matches the projection's endpoint, reached through the flow
    --project   baseline     the existing post-hoc shield (use_cbf=True, no guidance)

CONTROL ARM. --lam 0.0 here, NOT the stock sampler. The Python-unrolled loop differs from the
jitted lax.scan by ~1.4% of mean|action| (eager vs fused XLA with bf16 params; both paths are
individually deterministic). Running every arm through this script makes that offset common-mode
so it cancels; comparing against the stock sampler would fold it into the result.

READ fire_rate BEFORE THE HEADLINE NUMBERS. If the barrier never had anything to correct, a null
result means the wiring was inert, not that guidance failed — and that is invisible from TSR and
collision rate alone.

  python -m experiments.rollout_guided --suite safelibero_spatial --level II --task 0 \
      --episodes 35 36 37 38 39 --lam 0.5 --out results_guided/spatial_LII_t0_lam0.5
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

# JAX/flax emit hundreds of DeprecationWarnings per rollout, burying the progress line and any
# real error. filterwarnings() alone does NOT hold — jax/absl reset the filters during import — so
# PYTHONWARNINGS is set before those imports happen. Both are kept: the env var covers the import
# phase, the filters cover anything that re-enables them later.
import os as _os
_os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episodes", type=int, nargs="+", default=[35, 36, 37, 38, 39])
    ap.add_argument("--lam", type=float, default=0.5,
                    help="guidance strength. 0 = control (unguided, same code path); "
                         "1.0 = the projection's endpoint, reached through the flow.")
    ap.add_argument("--project", action="store_true",
                    help="BASELINE: post-hoc CBF projection instead of guidance (the current shield)")
    ap.add_argument("--n-guide", type=int, default=3)
    ap.add_argument("--margin", type=float, default=0.0,
                    help="extra clearance beyond r_ee + r_obstacle, in metres")
    ap.add_argument("--r-ee", type=float, default=0.05, help="EE bounding-sphere radius")
    ap.add_argument("--dt", type=float, default=1.0,
                    help="action-to-displacement scale used by the barrier. The env action is an "
                         "OSC delta, so this converts it to the metres the EE would move.")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--num-steps", type=int, default=10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from openpi.training import config as _config
    from experiments.libero_runner import (make_libero_env, run_libero_trial,
                                           detect_safelibero_obstacle)
    from experiments.load_policy import create_policy_partial
    from experiments.rl_rollout_local import build_policy_fn
    from experiments.guided_policy import GuidedPolicy
    from experiments.cbf_guidance import (extract_action_scale, make_guidance_source,
                                          GuidanceStats)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.suite}_L{args.level}_t{args.task}"
    mode = "project" if args.project else f"guided(lam={args.lam})"
    print(f"\n=== {tag}  mode={mode}  episodes={args.episodes} ===", flush=True)

    cfg = _config.get_config(args.config)
    policy = create_policy_partial(cfg, args.checkpoint)
    scale = extract_action_scale(policy)

    env, lang, inits = make_libero_env(task_suite=args.suite, task_idx=args.task,
                                       safety_level=args.level, horizon=args.horizon)

    rows = []
    for ep in args.episodes:
        stats = GuidanceStats()
        guidance_source = None

        if not args.project and args.lam != 0.0:
            # The obstacle is static within an episode, so detect once and close over it. The EE
            # position is read per query from the observation, so no runner hook is needed.
            env.reset()
            obs0 = env.set_init_state(inits[ep])
            ob = detect_safelibero_obstacle(env, obs0)
            if ob is None:
                print(f"  ep{ep}: no obstacle detected — running unguided", flush=True)
            else:
                spheres = [(np.asarray(ob.pos, float), float(ob.safety_radius))]
                guidance_source = make_guidance_source(
                    spheres, r_ee=args.r_ee, action_scale=scale, dt=args.dt,
                    margin=args.margin, n_guide=args.n_guide, stats=stats)

        pol = GuidedPolicy(policy, num_steps=args.num_steps, noise_level=0.0, sde_type="cps",
                           seed=0, guidance_source=guidance_source,
                           lam=(0.0 if args.project else args.lam))
        _inner = build_policy_fn(pol)
        _t0, _n = time.time(), [0]

        def policy_fn(*a, **k):
            """Live one-line progress. The runner's own per-step prints are interleaved with JAX
            noise, so this is the only place a human can see whether the rollout is moving."""
            out = _inner(*a, **k)
            _n[0] += 1
            gs = stats.summary()
            el = time.time() - _t0
            done = _n[0] * args.replan
            # STDERR, not stdout: the runner prints per-step lines to stdout, and its newlines
            # break a \r-updated line. stderr stays clean and interleaves with nothing.
            sys.stderr.write(
                f"\r  ep{ep}  step {done:>4}/{args.horizon}  "
                f"{done/max(args.horizon,1):>4.0%}  {el:5.0f}s  "
                f"guidance {gs['fired']}/{gs['calls']} ({gs['fire_rate']:>4.0%})  "
                f"mean|d| {gs['mean_norm']:.4f}   ")
            sys.stderr.flush()
            return out

        # The runner's per-step prints go to a log rather than the terminal, so the progress line
        # is readable. Nothing is lost — runner.log holds the full detail for debugging.
        _runlog = open(out / "runner.log", "a")
        with contextlib.redirect_stdout(_runlog):
            m = run_libero_trial(
                env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                use_cbf=bool(args.project),      # projection baseline uses the existing shield
                vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                replan_steps=args.replan, horizon=args.horizon,
                policy_fn=policy_fn, record_policy_trace=False, scene_name=tag)
        _runlog.close()

        sys.stderr.write("\n"); sys.stderr.flush()
        s = m.summary()
        gs = stats.summary()
        rows.append({"episode": ep, "success": int(bool(s["goal_reached"])),
                     "collision": int(bool(s["collision_detected"])),
                     "ets": int(s.get("goal_reach_step") or -1), **{f"g_{k}": v for k, v in gs.items()}})
        print(f"  ep{ep}: TSR={rows[-1]['success']} coll={rows[-1]['collision']}  "
              f"guidance fired {gs['fired']}/{gs['calls']} ({gs['fire_rate']:.0%}) "
              f"mean|d| {gs['mean_norm']:.4f}", flush=True)

    n = len(rows)
    succ = sum(r["success"] for r in rows)
    coll = sum(r["collision"] for r in rows)
    fired = sum(r["g_fired"] for r in rows)
    calls = sum(r["g_calls"] for r in rows)
    summary = {"scene": tag, "mode": mode, "n": n,
               "tsr": succ / n if n else 0.0, "collision": coll / n if n else 0.0,
               "car": 1.0 - (coll / n if n else 0.0),
               "guidance_fire_rate": (fired / calls) if calls else 0.0,
               "guidance_calls": calls, "rows": rows}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    print(f"\n  TSR {succ}/{n} = {summary['tsr']:.0%}   CAR {summary['car']:.0%}   "
          f"guidance fired {fired}/{calls} ({summary['guidance_fire_rate']:.0%})")
    if calls and fired == 0:
        print("  WARNING: guidance never fired. A null result here says the wiring is inert, not "
              "that the method failed — check obstacle detection, --dt and --margin before "
              "concluding anything.")
    print(f"  -> {out}/summary.json\n")


if __name__ == "__main__":
    main()
