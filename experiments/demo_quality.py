"""demo_quality.py — flag demos that PASS the success+collision filter but are bad anyway.

Behaviour cloning copies everything in a demonstration, not just its outcome. A demo can succeed,
never touch the obstacle, and still teach the student something you do not want: a gripper that
closes 4 cm from the object and gets lucky, a path that wanders, a trajectory that skims 3 mm past
an obstacle without displacing it, a fumbled regrasp. The success + collision-free filter cannot
see any of that, which is why "the metrics look fine" is not the same as "the demos are good" —
a lesson this project already learned once (Exp 006c, after Exp 005 trusted metrics over paths).

Watching every clip does not scale. These properties are measurable, and every one of them is
already recorded by MetricsTracker, so no new instrumentation is needed:

    obj_dist_at_grasp        how far the gripper was from the object when it closed
    gripper_close_transitions  more than one close = fumbled and regrasped
    deadlock_steps           time spent stalled
    min_dist_overall         closest approach — a near-miss teaches unsafe proximity
    mean_jerk                jerky demos teach jerky policies
    path_efficiency          straight-line distance / path travelled
    goal_reach_step          a struggle takes longer than a clean run

Thresholds are mostly RELATIVE to the collected set (worst decile) rather than invented constants,
so the filter adapts to whatever the teacher's normal behaviour looks like. The few absolute
thresholds are ones that can be justified physically and are documented inline.

    PYTHONPATH=. python -m experiments.demo_quality --round results_distill/planner_A

Prints a per-demo table of flags and writes `quality.csv`. Use `--drop` to write a filtered
manifest keeping only demos with no flags, so BC trains on the clean subset.
"""

from __future__ import annotations

import argparse
import csv

from pathlib import Path

# Absolute red flags. Each is a property that is wrong regardless of what the rest of the set does.
ABS_RULES = {
    # Coarse backstop for a grip point planned somewhere absurd. Deliberately loose: this does NOT
    # detect the gripper closing on air. obj_dist_at_grasp measures the EE site, which sits ~5-6 cm
    # behind the fingertips, so a real grasp and a close-on-air read almost the same (measured on
    # one scene: 55.3 mm succeeding, 57.7 mm failing). That failure mode is instead caught by the
    # success filter itself — an episode that grasps nothing does not reach the goal — so it never
    # reaches the training set. Detecting it directly would need the HELD object's pose tracked,
    # which the runner does not currently record (obj_dist follows the nearest target object and
    # switches between them mid-episode).
    "grasp_far":     ("obj_dist_at_grasp", lambda v: v > 0.15),
    # One close per pick. More means it fumbled and retried, and BC would learn the fumble.
    "regrasped":     ("gripper_close_transitions", lambda v: v > 1),
    "stalled":       ("deadlock_steps", lambda v: v > 0),
    # Passing within a centimetre without displacing anything is luck, not safety, and it is
    # exactly the behaviour a safety-distillation student should NOT copy.
    "near_miss":     ("min_dist_overall", lambda v: v < 0.01),
}
# Relative flags: worst decile of this particular collection.
REL_RULES = ["mean_jerk", "goal_reach_step"]
REL_LOWER_BETTER = {"mean_jerk": True, "goal_reach_step": True}


def _f(row, key, default=float("nan")):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, help="a collection dir containing manifest.csv")
    ap.add_argument("--drop", action="store_true",
                    help="write manifest_clean.csv keeping only unflagged demos")
    ap.add_argument("--pct", type=float, default=90.0,
                    help="percentile above which a relative metric is flagged (default 90)")
    args = ap.parse_args()

    rd = Path(args.round)
    rows = list(csv.DictReader(open(rd / "manifest.csv")))
    if not rows:
        raise SystemExit(f"no demos in {rd}/manifest.csv")

    # A check whose column is absent from the manifest must be reported as SKIPPED, never as
    # passed — an old collection predating the quality columns would otherwise look flawless.
    missing = sorted({name for name, (key, _) in ABS_RULES.items() if key not in rows[0]})

    # relative thresholds from this set
    thresholds = {}
    for k in REL_RULES:
        vals = [_f(r, k) for r in rows]
        vals = [v for v in vals if v == v and v >= 0]
        if len(vals) >= 5:
            vals.sort()
            # Linear interpolation between order statistics, NOT vals[int(n*pct/100)]: that indexes
            # the largest element for small n, making the threshold equal to the worst value, so
            # `v > thr` could never fire and the relative checks silently did nothing.
            pos = (len(vals) - 1) * args.pct / 100.0
            lo = int(pos)
            hi = min(lo + 1, len(vals) - 1)
            thresholds[k] = vals[lo] + (pos - lo) * (vals[hi] - vals[lo])

    flagged = []
    for r in rows:
        flags = []
        for name, (key, bad) in ABS_RULES.items():
            v = _f(r, key)
            if v == v and bad(v):
                flags.append(f"{name}({v:.3g})")
        for k, thr in thresholds.items():
            v = _f(r, k)
            if v == v and v > thr:
                flags.append(f"{k}_outlier({v:.3g}>{thr:.3g})")
        r["_flags"] = "|".join(flags)
        if flags:
            flagged.append(r)

    print(f"\n{len(rows)} demos in {rd}")
    if missing:
        print(f"  note: these fields are absent from the manifest, so their checks were skipped: "
              f"{missing}\n  (re-collect to populate them)")
    print(f"  {len(rows) - len(flagged)} clean   {len(flagged)} FLAGGED\n")

    if flagged:
        print(f"  {'demo':<52}flags")
        for r in flagged:
            name = Path(r.get("trace_path", "?")).name.replace("_trace.npz", "")
            print(f"  {name:<52}{r['_flags']}")

    with open(rd / "quality.csv", "w", newline="") as f:
        cols = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(rows)
    print(f"\n  → {rd}/quality.csv")

    if args.drop:
        keep = [r for r in rows if not r["_flags"]]
        with open(rd / "manifest_clean.csv", "w", newline="") as f:
            cols = [c for c in rows[0].keys() if c != "_flags"]
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            for r in keep:
                w.writerow({k: r[k] for k in cols})
        print(f"  → {rd}/manifest_clean.csv  ({len(keep)} demos kept)")

    print("\n  Spot-check the FLAGGED demos on video rather than the whole set — that is the point:")
    print("  the filter narrows what needs a human eye, it does not replace one.")


if __name__ == "__main__":
    main()
