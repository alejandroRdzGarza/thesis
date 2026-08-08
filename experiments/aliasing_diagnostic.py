"""aliasing_diagnostic.py — predict whether a demo set can be behaviour-cloned, BEFORE training.

Behaviour cloning fits a function obs -> action. If a dataset contains samples whose observations
are nearly identical but whose target actions differ, no policy of any capacity can fit both: the
target is not a function of the observation. The optimiser averages the contradictory targets, and
the result is a policy that is confidently wrong in exactly the states where precision matters.

This project asserted that failure mode but never measured it. The claim was:

    The shield's correction depends on UNOBSERVED state. Arm-link avoidance is a function of the
    full joint configuration, but pi0.5 observes image + 8-D proprio (eef_pos3 + axis_angle3 +
    gripper2) — no joint angles. So the safe action is not a function of the observation; BC over
    states that look identical but have different arm configs averages contradictory corrections.

That is testable from the demo files alone.

METHOD
For each dataset, take (observation, target-action) pairs from the traces, find each sample's
nearest neighbours IN OBSERVATION SPACE, and measure how much their target actions disagree.
Normalise by the disagreement between RANDOM pairs, which is what you would see if the observation
carried no information about the action at all:

    aliasing ratio = E[ ||a_i - a_j|| : obs neighbours ] / E[ ||a_i - a_j|| : random pairs ]

    ~1.0  observations do not disambiguate actions -> BC CANNOT fit this dataset
    ~0.0  observations determine actions           -> BC can fit it

The ratio is scale-free, so datasets with different action magnitudes are directly comparable — the
point of the exercise, since the whole question is why one teacher distils and another does not.

    python -m experiments.aliasing_diagnostic --round results_shielded/round0_demos --label shielded
    python -m experiments.aliasing_diagnostic --round results_distill/planner_A  --label planner

INTERPRETATION CAVEAT, which matters and should be reported alongside any number this prints:
the primary measurement uses PROPRIO ONLY, because that is where the unobserved-joint-configuration
argument lives. But the policy also sees images, which may disambiguate states that proprio alone
confuses. --with-image adds a crude visual descriptor (downsampled greyscale) so the comparison can
be made both ways. A crude descriptor is a LOWER BOUND on what the vision encoder resolves, so
image-conditioned aliasing measured here is an over-estimate.
"""

from __future__ import annotations

import argparse
import glob
import random
from pathlib import Path

import numpy as np


def collect_pairs(round_dir: Path, max_traces: int, with_image: bool, img_dim: int = 8):
    """(obs_features, target_actions) over a round's traces."""
    from experiments.policy_trace import load_episode_trace

    paths = sorted(glob.glob(str(round_dir / "**" / "*_trace.npz"), recursive=True))
    if not paths:
        raise SystemExit(f"no traces under {round_dir}")
    random.seed(0)
    if len(paths) > max_traces:
        paths = random.sample(paths, max_traces)

    feats, acts = [], []
    for p in paths:
        for q in load_episode_trace(p):
            if q.shielded_actions is None or q.obs is None:
                continue
            state = np.asarray(q.obs.get("state"), dtype=np.float32).ravel()
            if state.size == 0:
                continue
            f = state
            if with_image:
                im = q.obs.get("image")
                if im is not None:
                    im = np.asarray(im, dtype=np.float32)
                    if im.ndim == 3:
                        im = im.mean(axis=2)                      # greyscale
                    s = max(1, im.shape[0] // img_dim)
                    small = im[::s, ::s][:img_dim, :img_dim].ravel() / 255.0
                    f = np.concatenate([state, small])
            # The BC target is the executed chunk; compare the FIRST executed action, which is the
            # one the policy must commit to from this observation.
            feats.append(f)
            acts.append(np.asarray(q.shielded_actions, dtype=np.float32)[0])
    if not feats:
        raise SystemExit(f"no usable (obs, action) pairs in {round_dir}")
    return np.stack(feats), np.stack(acts)


def aliasing_ratio(F: np.ndarray, A: np.ndarray, k: int, n_sample: int, seed: int = 0):
    """Neighbour action disagreement / random-pair disagreement."""
    rng = np.random.default_rng(seed)
    # Standardise each observation dimension so no single unit dominates the distance.
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)

    n = len(Fz)
    idx = rng.choice(n, size=min(n_sample, n), replace=False)

    try:
        from scipy.spatial import cKDTree
        tree = cKDTree(Fz)
        _, nbr = tree.query(Fz[idx], k=k + 1)          # k+1: the first hit is the point itself
        nbr = nbr[:, 1:]
    except ImportError:
        nbr = np.stack([rng.choice(n, size=k, replace=False) for _ in idx])
        print("  (scipy unavailable — neighbours are RANDOM, so the ratio will read ~1.0 "
              "regardless; install scipy for a real measurement)")

    d_nbr = np.mean([np.linalg.norm(A[i] - A[j]) for i, row in zip(idx, nbr) for j in row])

    # Random pairs: the disagreement expected when the observation carries no information.
    ri, rj = rng.choice(n, size=len(idx) * k), rng.choice(n, size=len(idx) * k)
    d_rnd = np.mean(np.linalg.norm(A[ri] - A[rj], axis=1))

    # How close neighbours actually are, so "neighbour" can be sanity-checked rather than assumed.
    nd = np.mean([np.linalg.norm(Fz[i] - Fz[j]) for i, row in zip(idx, nbr) for j in row])
    return d_nbr, d_rnd, (d_nbr / d_rnd if d_rnd > 0 else float("nan")), nd


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--round", required=True, type=Path)
    ap.add_argument("--label", default=None)
    ap.add_argument("--max-traces", type=int, default=40,
                    help="traces to sample (each holds hundreds of queries)")
    ap.add_argument("--k", type=int, default=8, help="neighbours per sample")
    ap.add_argument("--n-sample", type=int, default=4000, help="query points to evaluate")
    ap.add_argument("--max-samples", type=int, default=1400,
                    help="cap the REFERENCE set size. Essential for comparing datasets: neighbour "
                         "distance falls as the set grows, so a denser dataset scores a lower ratio "
                         "for reasons unrelated to aliasing. A planner round yields ~337 queries per "
                         "trace against ~35 for a shielded round, so uncapped the two are not "
                         "comparable at all. Set the SAME value for every dataset you compare.")
    ap.add_argument("--with-image", action="store_true",
                    help="append a crude downsampled-greyscale descriptor to the observation")
    args = ap.parse_args()

    label = args.label or args.round.name
    F, A = collect_pairs(args.round, args.max_traces, args.with_image)
    if args.max_samples and len(F) > args.max_samples:
        rng = np.random.default_rng(0)
        sel = rng.choice(len(F), size=args.max_samples, replace=False)
        F, A = F[sel], A[sel]
    d_nbr, d_rnd, ratio, nd = aliasing_ratio(F, A, args.k, args.n_sample)

    print(f"\n  dataset            : {label}")
    print(f"  samples            : {len(F)} (obs dim {F.shape[1]}, action dim {A.shape[1]})"
          f"{'  [capped for comparability]' if args.max_samples else ''}")
    print(f"  observation source : {'proprio + downsampled image' if args.with_image else 'proprio only'}")
    print(f"  mean neighbour distance (z-scored obs) : {nd:.4f}")
    print(f"  action disagreement, neighbours        : {d_nbr:.4f}")
    print(f"  action disagreement, random pairs      : {d_rnd:.4f}")
    print(f"\n  ALIASING RATIO     : {ratio:.3f}")
    if ratio > 0.8:
        print("    -> observations barely disambiguate actions. BC should struggle badly here;")
        print("       more data or epochs will not help, because the target is not a function")
        print("       of what the policy can see.")
    elif ratio > 0.5:
        print("    -> partial aliasing. BC will fit the reliable part and average the rest.")
    else:
        print("    -> observations largely determine actions. BC should fit this dataset.")
    print("\n  Compare ratios ACROSS datasets — the absolute value depends on k and on how the")
    print("  observation is built, so it is a relative measure, not a threshold.\n")


if __name__ == "__main__":
    main()
