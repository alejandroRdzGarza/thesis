"""tune_teacher.py — search a per-task teacher profile for each SafeLIBERO scene.

`teacher_profiles.derive_profile` covers the failure modes that follow from geometry alone
(grasp side, elevated pick, crowded corridor). What it can't know is where a scene needs a knob
moved for reasons that aren't in the geometry — how patient the descent has to be before the
contact test fires, how high to carry, how tightly to centre a set-down. Rather than hand-tune 24
scenes, this searches them: greedy coordinate descent over a small candidate set, N episodes per
evaluation, keeping whatever beats the auto profile.

The winners are written to `experiments/teacher_profiles.json`, which `resolve_profile` layers on
top of the geometric strategy. Demo collection then picks them up with no further wiring.

    # tune only the scenes a prior sweep failed, 4 inits each
    python -m experiments.tune_teacher --from-sweep sweep_profiles/per_config.csv --max-success 0.75

    # tune one scene explicitly
    python -m experiments.tune_teacher --suites safelibero_goal --levels II --tasks 3 --episodes 4

Scoring is success-first with a collision penalty (`--collision-weight`). The CBF shield is on
during the search exactly as it is during collection, so a profile is only rewarded for outcomes
the shield actually allows.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import time
from pathlib import Path

# Candidate knob settings, in priority order. Each entry is (where, field, values); the search
# tries every value for one knob, keeps the best, then moves to the next knob (coordinate descent),
# so cost is linear in the number of knobs rather than exponential.
KNOBS: list[tuple[str, str, list]] = [
    ("grasp", "flip_side", [False, True]),      # pinch the other rim side (overrides the geometric pick)
    ("cfg", "approach_h", [0.12, 0.09, 0.16]),  # hover height before descending
    ("cfg", "rim_offset", [0.05, 0.04, 0.06]),  # how far along the rim to pinch
    ("mpc", "radius_buffer", [0.03, 0.0, 0.05]),  # planner keep-out beyond the CBF radius
    ("cfg", "descend_z_cap", [0.30, 0.20]),     # descent speed (XY drift vs stalling high)
    ("cfg", "stall_patience", [12, 16, 20]),    # steps of no progress before calling it contact
    ("cfg", "lift_h", [0.18, 0.12, 0.08]),      # lift before transporting
    ("cfg", "goal_clear_h", [0.22, 0.16, 0.12]),  # carry height above the goal
    ("cfg", "setdown_reach", [0.06, 0.04, 0.08]),  # how far below the goal to press on set-down
    ("cfg", "place_xy_tol", [0.035, 0.025]),    # centring tolerance before releasing
]


def _score(succ: float, coll: float, w_coll: float) -> float:
    return succ - w_coll * coll


def evaluate(env, lang, init_states, episodes, horizon, replan, suite, level, task,
             candidate: dict) -> tuple[float, float, list[str]]:
    """Run `episodes` inits under one candidate profile → (success_rate, collision_rate, phases)."""
    from experiments.libero_runner import run_libero_trial
    from experiments.classical_expert import PickPlaceController
    import experiments.teacher_profiles as TP

    # Inject the candidate as the per-task override for the duration of this evaluation.
    key = TP.profile_key(suite, level, task)
    saved = TP.PROFILE_OVERRIDES.get(key)
    TP.PROFILE_OVERRIDES[key] = candidate
    try:
        succ = coll = 0
        phases: list[str] = []
        for ep in episodes:
            ctrl = PickPlaceController()
            if candidate.get("cfg", {}).get("rim_offset") is not None:
                # rim_offset feeds the geometric side choice, so it must be set BEFORE resolution.
                ctrl.cfg.rim_offset = candidate["cfg"]["rim_offset"]
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    m = run_libero_trial(
                        env=env, episode_idx=ep, instruction=lang, initial_states=init_states,
                        obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False,
                        use_cbf=True, vla="pi05", auto_detect_obstacle=True, aegis_faithful=True,
                        replan_steps=replan, horizon=horizon, controller=ctrl,
                        scene_name=f"tune_{suite}_L{level}_t{task}", save_video=None,
                        teacher_suite=suite, teacher_level=level, teacher_task=task)
                s = m.summary()
                succ += int(bool(s["goal_reached"])); coll += int(bool(s["collision_detected"]))
                phases.append(ctrl.phase)
            except Exception as e:
                phases.append(f"ERR:{type(e).__name__}")
        n = max(len(episodes), 1)
        return succ / n, coll / n, phases
    finally:
        if saved is None:
            TP.PROFILE_OVERRIDES.pop(key, None)
        else:
            TP.PROFILE_OVERRIDES[key] = saved


def _with_knob(candidate: dict, where: str, field: str, value) -> dict:
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in candidate.items()}
    if where == "grasp":
        out["flip_side"] = value
    else:
        out.setdefault(where, {})[field] = value
    return out


def tune_task(suite: str, level: str, task: int, episodes: list[int], *, horizon: int,
              replan: int, w_coll: float, say) -> dict | None:
    """Greedy coordinate descent over KNOBS for one scene. Returns the winning override, or None
    if the geometric profile already wins (nothing to store)."""
    from experiments.libero_runner import make_libero_env

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            env, lang, init_states = make_libero_env(task_suite=suite, task_idx=task,
                                                     safety_level=level, horizon=horizon)
    except Exception as e:
        say(f"  {suite} L{level} t{task}: ENV ERROR — skipped ({e})")
        return None

    # Baseline: the geometric profile with nothing overridden.
    base_cand: dict = {}
    best_s, best_c, ph = evaluate(env, lang, init_states, episodes, horizon, replan,
                                  suite, level, task, base_cand)
    best_score = _score(best_s, best_c, w_coll)
    best_cand = base_cand
    say(f"  auto            success {best_s:>4.0%}  collision {best_c:>4.0%}  score {best_score:+.3f}  {ph}")

    for where, field, values in KNOBS:
        for v in values:
            cand = _with_knob(best_cand, where, field, v)
            if cand == best_cand:
                continue
            s, c, ph = evaluate(env, lang, init_states, episodes, horizon, replan,
                                suite, level, task, cand)
            sc = _score(s, c, w_coll)
            mark = ""
            if sc > best_score + 1e-9:
                best_score, best_s, best_c, best_cand = sc, s, c, cand
                mark = "  ← best"
            say(f"  {field}={v!r:<8}  success {s:>4.0%}  collision {c:>4.0%}  score {sc:+.3f}{mark}")
            if best_s >= 1.0 and best_c <= 0.0:
                break                                  # perfect on this scene — stop spending rollouts
        if best_s >= 1.0 and best_c <= 0.0:
            break

    with contextlib.suppress(Exception):
        env.close()

    if not best_cand or best_cand == {"flip_side": False}:
        say(f"  → geometric profile already best (success {best_s:.0%}, collision {best_c:.0%})")
        return None
    out = {k: v for k, v in best_cand.items() if not (k == "flip_side" and not v)}
    out["name"] = f"tuned_{suite.replace('safelibero_','')}_L{level}_t{task}"
    out["_result"] = {"success": round(best_s, 3), "collision": round(best_c, 3),
                      "n": len(episodes)}
    say(f"  → tuned: success {best_s:.0%}  collision {best_c:.0%}  {json.dumps({k: v for k, v in out.items() if not k.startswith('_')})}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", nargs="+",
                    default=["safelibero_spatial", "safelibero_object", "safelibero_goal"])
    ap.add_argument("--levels", nargs="+", default=["I", "II"], choices=["I", "II"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3],
                    help="init indices to score each candidate on (more = less noise, slower)")
    ap.add_argument("--from-sweep", type=Path,
                    help="a per_config.csv; only tune the scenes it shows as failing")
    ap.add_argument("--max-success", type=float, default=0.75,
                    help="with --from-sweep: tune scenes at or below this success rate")
    ap.add_argument("--collision-weight", type=float, default=0.5,
                    help="score = success − w·collision")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    ap.add_argument("--out", type=Path, default=None, help="default: experiments/teacher_profiles.json")
    ap.add_argument("--log", type=Path, default=Path("tune_teacher.log"))
    args = ap.parse_args()

    from experiments.teacher_profiles import save_overrides, profile_key

    targets = [(s, l, t) for s in args.suites for l in args.levels for t in args.tasks]
    if args.from_sweep:
        keep = set()
        for r in csv.DictReader(open(args.from_sweep)):
            if float(r["success_rate"]) <= args.max_success:
                keep.add((r["suite"], r["level"], int(r["task"])))
        targets = [t for t in targets if t in keep]

    logf = open(args.log, "a")

    def say(msg=""):
        print(msg, flush=True)
        logf.write(msg + "\n"); logf.flush()

    say(f"=== teacher tuning  {time.strftime('%Y-%m-%d %H:%M')} ===")
    say(f"    {len(targets)} scenes × {len(args.episodes)} inits per candidate")
    t0 = time.time()
    winners: dict[str, dict] = {}
    for i, (suite, level, task) in enumerate(targets, 1):
        say(f"\n[{i}/{len(targets)}] {suite} L{level} t{task}")
        w = tune_task(suite, level, task, args.episodes, horizon=args.horizon,
                      replan=args.replan, w_coll=args.collision_weight, say=say)
        if w:
            winners[profile_key(suite, level, task)] = w
            save_overrides({profile_key(suite, level, task): w}, args.out)   # checkpoint as we go

    say(f"\n{len(winners)} tuned profile(s) written to "
        f"{args.out or 'experiments/teacher_profiles.json'}  ({time.time()-t0:.0f}s)")
    logf.close()


if __name__ == "__main__":
    main()
