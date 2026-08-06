"""record_planner_episodes.py — record a LABELLED sample so the metrics can be checked by eye.

Recording only failures cannot validate a metric: it catches successes wrongly scored as failures
(goal LI t3 was one — a two-stage drawer task the pick-and-place teacher can never satisfy) but it
cannot catch failures wrongly scored as successes. This records both, plus every episode flagged
as a collision, and puts the recorded numbers in the filename so each clip can be checked against
what the metric claimed:

    <scene>_ep<N>__TSR-<0|1>__COLL-<0|1>__objgoal-<cm>cm__<phase>.mp4

What to check, per clip:
  TSR-1   the object really does end up on/in the target
  TSR-0   it really does not (and is not a task the teacher cannot express, like opening a drawer)
  COLL-1  the arm, gripper or carried object really does touch/shift the obstacle
  COLL-0  nothing is touched

The overlay burned into each frame by the runner shows step, obstacle distance and the live
collision flag, so a disagreement between the overlay and the filename is itself informative.

  PYTHONPATH=. python -m experiments.record_planner_episodes \
      --sweep sweep_planner_servo --out videos_metric_check
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import re
from pathlib import Path


def _log_metrics(sweep: Path, r: dict) -> tuple:
    """(object displacement, object→goal distance) from the episode's own debug dump."""
    p = sweep / "logs" / f"{r['suite']}_L{r['level']}_t{r['task']}_ep{r['episode']}.log"
    try:
        t = p.read_text()
    except OSError:
        return float("nan"), float("nan")
    mv = re.search(r"moved ([\d.]+) m", t)
    gd = re.search(r"object→goal  dist  : ([\d.]+)", t)
    return (float(mv.group(1)) if mv else float("nan"),
            float(gd.group(1)) if gd else float("nan"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="sweep_planner_servo")
    ap.add_argument("--out", default="videos_metric_check")
    ap.add_argument("--horizon", type=int, default=900)
    ap.add_argument("--max-success", type=int, default=8,
                    help="how many SUCCESSES to record (to check for false positives)")
    ap.add_argument("--max-fail", type=int, default=12)
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.planner_expert import PlannerExpert

    sweep = Path(args.sweep)
    rows = list(csv.DictReader(open(sweep / "per_episode.csv")))
    planned = [r for r in rows if int(r["planned"])]
    succ = [r for r in planned if int(r["success"])][:args.max_success]
    fail = [r for r in planned if not int(r["success"])][:args.max_fail]
    # every collision-flagged episode is worth seeing, whichever bucket it fell in
    coll = [r for r in planned if int(r["collision"]) and r not in succ and r not in fail][:6]
    todo = succ + fail + coll

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    print(f"recording {len(succ)} successes + {len(fail)} failures + {len(coll)} extra "
          f"collision episodes -> {out}/\n")

    made = []
    for i, r in enumerate(todo, 1):
        suite, level, task, ep = r["suite"], r["level"], int(r["task"]), int(r["episode"])
        moved, objgoal = _log_metrics(sweep, r)
        name = (f"{suite.replace('safelibero_','')}_L{level}_t{task}_ep{ep}"
                f"__TSR-{r['success']}__COLL-{r['collision']}"
                f"__objgoal-{objgoal*100:.0f}cm__{r['final_phase']}.mp4")
        vid = str((out / name).resolve())
        print(f"[{i}/{len(todo)}] {name}", flush=True)
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                env, lang, inits = make_libero_env(task_suite=suite, task_idx=task,
                                                   safety_level=level, horizon=args.horizon)
                ctrl = PlannerExpert()
                m = run_libero_trial(
                    env=env, episode_idx=ep, instruction=lang, initial_states=inits,
                    obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                    use_cbf=False, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                    replan_steps=1, horizon=args.horizon, controller=ctrl,
                    scene_name=f"{suite}_L{level}_t{task}", save_video=vid,
                    teacher_suite=suite, teacher_level=level, teacher_task=task)
            s = m.summary()
            made.append((name, lang, r, bool(s["goal_reached"]), bool(s["collision_detected"]),
                         moved, objgoal))
            with contextlib.suppress(Exception):
                env.close()
        except Exception as e:
            print(f"    recording failed: {type(e).__name__}: {e}")

    # A re-run that disagrees with the sweep means the rollout is NOT deterministic, which would
    # undermine every number in the table — worth knowing before trusting any of it.
    print(f"\n{len(made)} clips in {out}/")
    mismatch = [(n, r, ok, cl) for n, _l, r, ok, cl, _m, _g in made
                if ok != bool(int(r["success"])) or cl != bool(int(r["collision"]))]
    if mismatch:
        print(f"\n  !! {len(mismatch)} episode(s) scored DIFFERENTLY on re-run than in the sweep.")
        print("     Rollouts are not reproducible, so single-episode numbers carry real noise:")
        for n, r, ok, cl in mismatch:
            print(f"       {n}\n         sweep: TSR={r['success']} COLL={r['collision']}   "
                  f"re-run: TSR={int(ok)} COLL={int(cl)}")
    else:
        print("  re-runs reproduced the sweep's scoring exactly (deterministic).")

    print("\nWATCH ORDER")
    print("  1. the TSR-1 clips — is the object genuinely on/in the target?")
    print("  2. the TSR-0 clips — is it genuinely not, and is the task even expressible")
    print("     as pick-and-place? (goal LI t3 is 'open the top drawer and put the bowl inside')")
    print("  3. the COLL-1 clips — does anything actually touch the obstacle? The metric fires on")
    print("     >1 mm of obstacle displacement, which includes being nudged by another object.")


if __name__ == "__main__":
    main()
