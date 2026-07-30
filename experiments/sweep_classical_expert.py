"""
sweep_classical_expert.py — evaluate the classical MPC-CBF expert across the SafeLIBERO grid.

Runs the scripted pick-place + MPC-CBF controller over {suites × levels × tasks × episodes},
collects success / collision / final-phase statistics, and writes:
  <out>/per_episode.csv      one row per rollout (success, collision, final_phase)
  <out>/per_config.csv       aggregate per (suite, level, task)
  <out>/summary.txt          human-readable tables (per config, by suite, overall)
  <out>/logs/<config>_ep<N>.log   full per-episode runner output (incl. the classical debug dump)
  <out>/sweep.log            the console summary (progress + tables)

The console stays clean (progress + tables); the verbose per-step runner output is redirected to
per-episode log files so it's there for debugging without flooding the terminal.

Run on the Mac libero env (no VLA/GPU needed — classical controller only):
  python -m experiments.sweep_classical_expert \
      --suites safelibero_spatial safelibero_object safelibero_goal \
      --levels II --tasks 0 1 2 3 --episodes 0 1 2 3
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+",
                    default=["safelibero_spatial", "safelibero_object", "safelibero_goal"])
    ap.add_argument("--levels", nargs="+", default=["II"], choices=["I", "II"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out-dir", default="sweep_classical")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--save-video", action="store_true",
                    help="also save an mp4 per episode (slow + large; off by default for a stats sweep)")
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.classical_expert import PickPlaceController

    out = Path(args.out_dir); (out / "logs").mkdir(parents=True, exist_ok=True)
    if args.save_video:
        (out / "videos").mkdir(exist_ok=True)
    sweep_log = open(out / "sweep.log", "a")

    def say(msg=""):
        print(msg, flush=True)
        sweep_log.write(msg + "\n"); sweep_log.flush()

    controller = PickPlaceController()
    per_ep, per_cfg = [], []
    n_cfg = len(args.suites) * len(args.levels) * len(args.tasks)
    say(f"=== classical MPC-CBF expert sweep  {time.strftime('%Y-%m-%d %H:%M')} ===")
    say(f"    {n_cfg} configs × {len(args.episodes)} episodes = {n_cfg * len(args.episodes)} rollouts")
    say("")
    t_start = time.time()

    cfg_i = 0
    for suite in args.suites:
        for level in args.levels:
            for task in args.tasks:
                cfg_i += 1
                tag = f"{suite}_L{level}_t{task}"
                try:
                    env, lang, init_states = make_libero_env(
                        task_suite=suite, task_idx=task, safety_level=level, horizon=args.horizon)
                except Exception as e:
                    say(f"[{cfg_i}/{n_cfg}] {tag}: ENV ERROR — skipped ({e})")
                    continue

                n = succ = coll = 0
                phases: dict[str, int] = {}
                for ep in args.episodes:
                    controller.reset()
                    vid = str(out / "videos" / f"{tag}_ep{ep}.mp4") if args.save_video else None
                    ok = cl = False
                    ph = "ERROR"
                    logf = out / "logs" / f"{tag}_ep{ep}.log"
                    try:
                        with open(logf, "w") as lf, contextlib.redirect_stdout(lf):
                            m = run_libero_trial(
                                env=env, episode_idx=ep, instruction=lang, initial_states=init_states,
                                obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                                use_cbf=True, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                                replan_steps=args.replan, horizon=args.horizon, controller=controller,
                                scene_name=tag, save_video=vid)
                        s = m.summary()
                        ok, cl, ph = bool(s["goal_reached"]), bool(s["collision_detected"]), controller.phase
                    except Exception as e:
                        with open(logf, "a") as lf:
                            lf.write(f"\nEXCEPTION: {e}\n")
                    n += 1; succ += int(ok); coll += int(cl)
                    phases[ph] = phases.get(ph, 0) + 1
                    per_ep.append(dict(suite=suite, level=level, task=task, episode=ep,
                                       success=int(ok), collision=int(cl), final_phase=ph))
                try:
                    env.close()
                except Exception:
                    pass

                sr, cr = (succ / n if n else 0.0), (coll / n if n else 0.0)
                per_cfg.append(dict(suite=suite, level=level, task=task, n=n,
                                    success_rate=round(sr, 3), collision_rate=round(cr, 3), phases=phases))
                say(f"[{cfg_i}/{n_cfg}] {tag:<32} success {succ}/{n} ({sr:>4.0%})   "
                    f"collision {coll}/{n} ({cr:>4.0%})   phases={phases}")

    # ── write CSVs ──────────────────────────────────────────────────────────
    with open(out / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "level", "task", "episode",
                                          "success", "collision", "final_phase"])
        w.writeheader(); w.writerows(per_ep)
    with open(out / "per_config.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "level", "task", "n", "success_rate", "collision_rate"])
        w.writeheader()
        for r in per_cfg:
            w.writerow({k: r[k] for k in ["suite", "level", "task", "n", "success_rate", "collision_rate"]})

    # ── summary tables ──────────────────────────────────────────────────────
    def block(title):
        say(""); say("=" * 60); say(f" {title}"); say("=" * 60)

    block("PER-CONFIG")
    say(f"{'config':<34}{'success':>9}{'collision':>11}")
    for r in per_cfg:
        say(f"{r['suite']+' L'+r['level']+' t'+str(r['task']):<34}"
            f"{r['success_rate']:>8.0%}{r['collision_rate']:>10.0%}")

    block("BY SUITE")
    for suite in args.suites:
        sc = [r for r in per_cfg if r["suite"] == suite]
        tn = sum(r["n"] for r in sc)
        if not tn:
            continue
        ss = sum(r["success_rate"] * r["n"] for r in sc) / tn
        cc = sum(r["collision_rate"] * r["n"] for r in sc) / tn
        say(f"  {suite:<26} success {ss:>4.0%}   collision {cc:>4.0%}   (n={tn})")

    block("OVERALL")
    N = sum(r["n"] for r in per_cfg)
    if N:
        S = sum(r["success_rate"] * r["n"] for r in per_cfg) / N
        C = sum(r["collision_rate"] * r["n"] for r in per_cfg) / N
        say(f"  success {S:.1%}   collision {C:.1%}   ({N} rollouts, {time.time()-t_start:.0f}s)")
    say(f"\n  CSVs + per-episode logs → {out}/")
    sweep_log.close()


if __name__ == "__main__":
    main()
