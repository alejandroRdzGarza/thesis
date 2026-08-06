"""record_planner_failures.py — re-run the planner teacher's FAILING scenes and save mp4s.

Reads a sweep's per_episode.csv, picks out every episode that did not succeed, replays it with
video recording on, and writes one clip per failure named so the failure mode is obvious from the
filename:

    <suite>_L<level>_t<task>_ep<N>__<phase>__<why>.mp4

so you can sort the folder and watch one representative of each mode rather than all of them.

  PYTHONPATH=. python -m experiments.record_planner_failures \
      --sweep sweep_planner_allsuites --out videos_planner_failures
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="sweep_planner_allsuites")
    ap.add_argument("--out", default="videos_planner_failures")
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--no-cbf", action="store_true", default=True)
    ap.add_argument("--with-cbf", dest="no_cbf", action="store_false")
    ap.add_argument("--limit", type=int, default=0, help="0 = every failure")
    ap.add_argument("--only-suite", default=None)
    ap.add_argument("--include-planfail", action="store_true",
                    help="also record scenes where planning failed. Off by default: the\n                          clip is 900 steps of a stationary arm, which shows nothing.")
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.planner_expert import PlannerExpert

    rows = list(csv.DictReader(open(Path(args.sweep) / "per_episode.csv")))
    fails = [r for r in rows if not int(r["success"])]
    if not args.include_planfail:
        fails = [r for r in fails if int(r["planned"])]
    if args.only_suite:
        fails = [r for r in fails if r["suite"] == args.only_suite]
    if args.limit:
        fails = fails[:args.limit]

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(fails)} failing episodes → recording to {out}/\n")

    made = []
    for i, r in enumerate(fails, 1):
        suite, level, task, ep = r["suite"], r["level"], int(r["task"]), int(r["episode"])
        phase = r["final_phase"]
        why = ("planfail" if not int(r["planned"]) else
               "grasp_missed" if phase in ("DONE", "RELEASE", "RETREAT") else
               f"stuck_{phase.lower()}")
        name = f"{suite.replace('safelibero_','')}_L{level}_t{task}_ep{ep}__{phase}__{why}.mp4"
        vid = str((out / name).resolve())
        print(f"[{i}/{len(fails)}] {name}", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env, lang, inits = make_libero_env(task_suite=suite, task_idx=task,
                                                   safety_level=level, horizon=args.horizon)
                ctrl = PlannerExpert()
                m = run_libero_trial(
                    env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                    obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                    use_cbf=not args.no_cbf, vla="pi05", auto_detect_obstacle=True,
                    aegis_faithful=True, replan_steps=1, horizon=args.horizon,
                    controller=ctrl, scene_name=f"{suite}_L{level}_t{task}", save_video=vid,
                    teacher_suite=suite, teacher_level=level, teacher_task=task)
            s = m.summary()
            made.append((name, lang, bool(s["goal_reached"]), bool(s["collision_detected"])))
            with contextlib.suppress(Exception):
                env.close()
        except Exception as e:
            print(f"    recording failed: {type(e).__name__}: {e}")

    print(f"\n{len(made)} clips in {out}/\n")
    print("WATCH LIST — grouped by what to look for:")
    groups = {}
    for name, lang, ok, coll in made:
        groups.setdefault(name.split("__")[-1].replace(".mp4", ""), []).append((name, lang))
    hints = {
        "planfail": "no plan was found — the clip shows the arm idle; the question is whether a "
                    "human could see a viable grasp the sampler missed",
        "grasp_missed": "the plan runs to completion but the object is left behind — watch the "
                        "GRIP moment: do the fingers close on the object, or beside/above it?",
    }
    for g, items in sorted(groups.items()):
        print(f"\n  [{g}]  {hints.get(g, 'watch where the motion goes wrong')}")
        for name, lang in items:
            print(f"      {name}")
            print(f"          task: {lang}")


if __name__ == "__main__":
    main()
