"""culprits_from_log.py — recover WHICH BODY collided, from a run log, without re-running anything.

`sweep_eval_stats` reports a culprit breakdown from a `culprit` manifest column. Evaluations run
before that column existed have no such field, and re-running a six-arm eval to recover it costs
~24 GPU-hours. But the runner already printed the information per episode:

    Done — TSR=False  collision=False  CBF=74 acts  violations=0  touched_by=['arm_link', 'scene_object']

so it can be parsed back out of the log for free.

WHY IT MATTERS HERE. The CBF shield constrains the END-EFFECTOR only (EE spheres vs obstacle
spheres), while the collision metric scores every body — gripper, arm links, the carried object,
other scene objects. Two questions follow, and both are answered by this breakdown:

  1. Does the shield's residual collision rate consist of bodies it never constrained? If so, 13.3%
     is the scope limit of an EE barrier rather than a leak, and a distilled policy at 17.5% is
     near the practical ceiling of its teacher rather than a degraded copy of it.

  2. Did the STUDENT generalise beyond the teacher? If base collides substantially via `arm_link`
     and the distilled policy does not, it learned whole-arm avoidance from end-effector-only
     supervision — behaviour the shield never demonstrated directly.

    python -m experiments.culprits_from_log --log results_shielded/run_20260804_234046.log

Attribution: episodes are buffered and assigned to the arm named by the NEXT output-path line,
because the runner prints per-episode results before naming the directory it writes them to.
"""

from __future__ import annotations

import argparse
import collections
import re
from pathlib import Path

TOUCHED = re.compile(r"touched_by=\[([^\]]*)\]")
# e.g. "... → results_shielded/eval_r2_cbf/safelibero_spatial_LI_t1/manifest.csv"
ARMPATH = re.compile(r"eval_([A-Za-z0-9_]+)/")
COLLIDED = re.compile(r"collision=(True|False)")

CATS = ["gripper", "arm_link", "held_object", "scene_object"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, type=Path)
    args = ap.parse_args()

    text = args.log.read_text(errors="ignore").splitlines()

    pending: list[tuple[set, bool]] = []            # episodes seen since the last arm marker
    per_arm: dict[str, list[tuple[set, bool]]] = collections.defaultdict(list)

    for line in text:
        m = TOUCHED.search(line)
        if m:
            bodies = {b.strip().strip("'\"") for b in m.group(1).split(",") if b.strip()}
            c = COLLIDED.search(line)
            pending.append((bodies, c.group(1) == "True" if c else bool(bodies)))
            continue
        a = ARMPATH.search(line)
        if a and pending:
            per_arm[a.group(1)].extend(pending)
            pending = []

    if not per_arm:
        raise SystemExit(
            f"no 'touched_by=[...]' lines attributable to an eval_<arm>/ path in {args.log}.\n"
            "The log may predate that print, or use a different layout — check with:\n"
            f"  grep -c touched_by {args.log}")

    print(f"\n  COLLISION CULPRITS recovered from {args.log.name}")
    print(f"  (episodes in which each body touched the obstacle; an episode can have several)\n")
    print(f"  {'arm':<22}{'episodes':>9}{'collided':>10}" + "".join(f"{c:>14}" for c in CATS))
    for arm in sorted(per_arm):
        eps = per_arm[arm]
        n = len(eps)
        ncol = sum(1 for _b, c in eps if c)
        counts = {c: sum(1 for b, _c in eps if c in b) for c in CATS}
        print(f"  {arm:<22}{n:>9}{ncol:>10}" + "".join(f"{counts[c]:>14}" for c in CATS))

    print("\n  READ IT LIKE THIS")
    print("   • The shield constrains the END-EFFECTOR only. Residual collisions concentrated in")
    print("     arm_link / held_object / scene_object are OUTSIDE its scope — evidence that its")
    print("     floor is a scope limit, not a leak.")
    print("   • If a distilled arm shows far fewer arm_link collisions than base, the student")
    print("     generalised past its teacher: whole-arm avoidance from EE-only supervision.\n")


if __name__ == "__main__":
    main()
