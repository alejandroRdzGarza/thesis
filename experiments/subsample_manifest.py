"""subsample_manifest.py — size a demo round to a TARGET NUMBER OF GRADIENT STEPS.

Two BC runs are only comparable if the policy takes a comparable number of gradient steps. Matching
`--epochs` does not achieve that when the rounds differ in episode length: measured here, a shielded
π0.5 rollout (horizon 300) yields ~35 training examples, while a planner episode (horizon 900)
yields ~267 — 7.6x denser. So "20 epochs each" meant 16,140 steps for the shielded round and
~212,000 for the planner round, an 88-hour run that is not a controlled comparison but a different
experiment.

This picks how many demos to keep so that `traces x examples_per_trace x epochs / minibatch` lands
near a target step count, and samples them EVENLY ACROSS SCENES rather than at random — a random
draw over-represents scenes the teacher happened to solve often (goal LI t3 has 35 demos, several
scenes have 8), which would skew what the student sees.

    # match r1's 16,140 steps, 4 epochs
    python -m experiments.subsample_manifest --round results_distill/planner_A \
        --target-steps 16140 --epochs 4

Writes `manifest_matched.csv`. Pass --activate to make it the manifest the trainer reads
(the original is preserved as manifest_full.csv).

examples-per-trace is MEASURED from a few traces by default rather than assumed, because it is the
quantity the whole calculation hinges on and it differs per collection.
"""

from __future__ import annotations

import argparse
import collections
import csv
import random
from pathlib import Path


def measure_examples_per_trace(rows: list[dict], n_sample: int = 8) -> float:
    """Average training examples per trace.

    Reads ONLY the `n_queries` scalar from each .npz. np.load on a compressed archive is lazy, so
    nothing else is decompressed — critical because a planner trace holds ~337 queries of paired
    224x224 images, and load_episode_trace() on even a few of them was enough to get this script
    OOM-killed (silently, since SIGKILL leaves no traceback).
    """
    import numpy as np
    random.seed(0)
    sample = random.sample(rows, min(n_sample, len(rows)))
    counts = []
    print(f"  measuring examples/trace from {len(sample)} traces (header only, no image decode) ...",
          flush=True)
    for i, r in enumerate(sample, 1):
        path = Path(r["trace_path"])
        if not path.exists():
            print(f"    [{i}/{len(sample)}] MISSING {path.name}", flush=True)
            continue
        try:
            with np.load(path, allow_pickle=True) as z:
                n = int(z["n_queries"]) if "n_queries" in z.files else 0
        except Exception as e:
            print(f"    [{i}/{len(sample)}] unreadable ({type(e).__name__}) {path.name}", flush=True)
            continue
        counts.append(n)
        print(f"    [{i}/{len(sample)}] {path.name}: {n} queries", flush=True)
    if not counts:
        raise SystemExit("could not read any trace — are the paths rebased for this machine? "
                         "(see experiments/rebase_manifest.py)")
    return sum(counts) / len(counts)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--target-steps", type=int, default=16140,
                    help="gradient steps to match. Default 16140 = the r1 shielded round, so a "
                         "student trained on this data is comparable to it.")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--examples-per-trace", type=float, default=None,
                    help="skip measurement and use this value")
    ap.add_argument("--manifest", default="manifest.csv")
    ap.add_argument("--activate", action="store_true",
                    help="make the subsample the manifest the trainer reads; the original is kept "
                         "as manifest_full.csv")
    args = ap.parse_args()

    rd: Path = args.round
    print(f"\n  reading {rd/args.manifest} ...", flush=True)
    rows = list(csv.DictReader(open(rd / args.manifest)))
    if not rows:
        raise SystemExit(f"no rows in {rd/args.manifest}")
    print(f"  {len(rows)} demos in manifest", flush=True)

    ept = args.examples_per_trace or measure_examples_per_trace(rows)
    print(f"  -> {ept:.0f} examples per trace", flush=True)
    print("  selecting demos evenly across scenes ...", flush=True)
    # steps = traces * ept * epochs / minibatch  ->  solve for traces
    n_target = max(1, round(args.target_steps * args.minibatch / (ept * args.epochs)))
    n_target = min(n_target, len(rows))

    by_scene = collections.defaultdict(list)
    for r in rows:
        by_scene[Path(r["trace_path"]).parent.name].append(r)
    n_scenes = len(by_scene)
    per_scene = max(1, n_target // n_scenes)

    random.seed(0)
    keep: list[dict] = []
    for scene in sorted(by_scene):
        rs = by_scene[scene]
        keep += random.sample(rs, min(per_scene, len(rs)))
    # Top up from the scenes that still have demos left, so integer division does not undershoot.
    if len(keep) < n_target:
        pool = [r for r in rows if r not in keep]
        keep += random.sample(pool, min(n_target - len(keep), len(pool)))

    est_steps = int(len(keep) * ept * args.epochs / args.minibatch)
    out = rd / "manifest_matched.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader(); w.writerows(keep)

    print(f"\n  measured   : {ept:.0f} examples per trace")
    print(f"  available  : {len(rows)} demos across {n_scenes} scenes")
    print(f"  keeping    : {len(keep)} demos (~{per_scene}/scene)")
    print(f"  epochs     : {args.epochs}   minibatch {args.minibatch}")
    print(f"  ESTIMATED  : {est_steps} gradient steps   (target {args.target_steps})")
    print(f"  -> {out}")

    if args.activate:
        full = rd / "manifest_full.csv"
        if not full.exists():
            (rd / args.manifest).rename(full)
        out.replace(rd / args.manifest)
        print(f"  ACTIVATED: {args.manifest} is now the subsample; original kept as {full.name}")
        print(f"  revert with:  mv {full} {rd/args.manifest}")

    print("\n  Check the trainer's own 'N BC steps' line before letting it run — examples/trace is "
          "an average\n  and the true count will differ. If it is far off, adjust --epochs and "
          "re-run this.\n")


if __name__ == "__main__":
    main()
