"""sweep_planner_expert.py — evaluate the PLANNING teacher across the SafeLIBERO grid.

Same harness and metrics as sweep_classical_expert, so the two teachers are directly comparable:
success and collision come from run_libero_trial with the CBF shield on, exactly as during demo
collection.

  PYTHONPATH=. python -m experiments.sweep_planner_expert \
      --suites safelibero_spatial safelibero_object safelibero_goal \
      --levels I II --tasks 0 1 2 3 --episodes 0 1 2
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import time
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+",
                    default=["safelibero_spatial", "safelibero_object", "safelibero_goal"])
    ap.add_argument("--levels", nargs="+", default=["I", "II"], choices=["I", "II"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--out-dir", default="sweep_planner")
    ap.add_argument("--horizon", type=int, default=500)
    ap.add_argument("--replan", type=int, default=1,
                    help="the planner is queried every control step (it is playing back a plan)")
    ap.add_argument("--no-cbf", action="store_true",
                    help="run WITHOUT the shield — measures whether the planner is SELF-SAFE, "
                         "which is what makes it a better distillation target than the scripted "
                         "expert (that one collides ~60%% unshielded)")
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.planner_expert import PlannerExpert

    out = Path(args.out_dir); (out / "logs").mkdir(parents=True, exist_ok=True)
    rows, per_cfg = [], []
    t0 = time.time()
    scenes = [(s, l, t) for s in args.suites for l in args.levels for t in args.tasks]
    print(f"=== planner-expert sweep · {len(scenes)} scenes × {len(args.episodes)} episodes "
          f"· shield {'OFF' if args.no_cbf else 'ON'} ===\n")

    for i, (suite, level, task) in enumerate(scenes, 1):
        tag = f"{suite}_L{level}_t{task}"
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env, lang, inits = make_libero_env(task_suite=suite, task_idx=task,
                                                   safety_level=level, horizon=args.horizon)
        except Exception as e:
            print(f"[{i}/{len(scenes)}] {tag}: ENV ERROR — {e}")
            continue

        n = succ = coll = planned = 0
        phases = Counter()
        for ep in args.episodes:
            ctrl = PlannerExpert()
            ok = cl = False
            ph = "ERROR"
            logf = out / "logs" / f"{tag}_ep{ep}.log"
            try:
                with open(logf, "w") as lf, contextlib.redirect_stdout(lf):
                    m = run_libero_trial(
                        env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                        obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                        use_cbf=not args.no_cbf, vla="pi05", auto_detect_obstacle=True,
                        aegis_faithful=True, replan_steps=args.replan, horizon=args.horizon,
                        controller=ctrl, scene_name=tag,
                        teacher_suite=suite, teacher_level=level, teacher_task=task)
                s = m.summary()
                ok, cl = bool(s["goal_reached"]), bool(s["collision_detected"])
                ph = ctrl.phase
            except Exception as e:
                with open(logf, "a") as lf:
                    lf.write(f"\nEXCEPTION: {type(e).__name__}: {e}\n")
                ph = f"ERR:{type(e).__name__}"
            n += 1; succ += int(ok); coll += int(cl); planned += int(ctrl.planned)
            phases[ph] += 1
            rows.append(dict(suite=suite, level=level, task=task, episode=ep,
                             planned=int(ctrl.planned), success=int(ok), collision=int(cl),
                             final_phase=ph, plan_error=ctrl.plan_error))
        with contextlib.suppress(Exception):
            env.close()

        per_cfg.append(dict(suite=suite, level=level, task=task, n=n, planned=planned,
                            success=succ, collision=coll))
        print(f"[{i}/{len(scenes)}] {tag:<34} planned {planned}/{n}  success {succ}/{n}  "
              f"collision {coll}/{n}   {dict(phases)}", flush=True)

    with open(out / "per_episode.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["suite", "level", "task", "episode", "planned",
                                          "success", "collision", "final_phase", "plan_error"])
        w.writeheader(); w.writerows(rows)

    N = sum(r["n"] for r in per_cfg)
    if N:
        P = sum(r["planned"] for r in per_cfg)
        S = sum(r["success"] for r in per_cfg)
        C = sum(r["collision"] for r in per_cfg)
        print("\n" + "=" * 70)
        print(f"  planned {P}/{N} ({P/N:.0%})   success {S}/{N} ({S/N:.0%})   "
              f"collision {C}/{N} ({C/N:.0%})   [{time.time()-t0:.0f}s]")
        print("=" * 70)
        for suite in args.suites:
            sc = [r for r in per_cfg if r["suite"] == suite]
            tn = sum(r["n"] for r in sc)
            if tn:
                print(f"  {suite:<26} success {sum(r['success'] for r in sc)/tn:>4.0%}   "
                      f"collision {sum(r['collision'] for r in sc)/tn:>4.0%}")
    print(f"\n  → {out}/per_episode.csv")


if __name__ == "__main__":
    main()
