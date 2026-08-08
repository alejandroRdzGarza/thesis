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
    """(obs_features, target_actions) over a round's traces.

    Reads the npz keys DIRECTLY rather than via load_episode_trace. The archive stores `obs_state`
    (Q,8) and `obs_image` (Q,224,224,3) as separate keys and np.load is lazy per key, so a
    proprio-only run never decompresses a single image. Building full QueryTrace objects instead
    inflates a 55 MB planner trace to hundreds of MB of pixels — enough to get this script
    OOM-killed, silently, since SIGKILL leaves no traceback.
    """
    import gc

    paths = sorted(glob.glob(str(round_dir / "**" / "*_trace.npz"), recursive=True))
    if not paths:
        raise SystemExit(f"no traces under {round_dir}")
    random.seed(0)
    if len(paths) > max_traces:
        paths = random.sample(paths, max_traces)

    feats, acts = [], []
    print(f"  reading {len(paths)} traces"
          f"{' (+images)' if with_image else ' (proprio only — images never decompressed)'} ...",
          flush=True)
    for i, p in enumerate(paths, 1):
        try:
            with np.load(p, allow_pickle=True) as z:
                if not (bool(z["has_obs"]) and bool(z["has_shielded"])):
                    print(f"    [{i}/{len(paths)}] skipped (no obs/targets): {Path(p).name}", flush=True)
                    continue
                state = np.asarray(z["obs_state"], dtype=np.float32)          # (Q, 8)
                sa = np.asarray(z["shielded_actions"], dtype=np.float32)      # (Q, L, 7)
                lens = np.asarray(z["shielded_lens"], dtype=np.int32)
                if with_image:
                    im = np.asarray(z["obs_image"], dtype=np.float32)         # the expensive key
                    g = im.mean(axis=3)                                       # (Q,H,W) greyscale
                    st = max(1, g.shape[1] // img_dim)
                    small = g[:, ::st, ::st][:, :img_dim, :img_dim]
                    small = small.reshape(len(g), -1) / 255.0
                    del im, g
                else:
                    small = None
        except Exception as e:
            print(f"    [{i}/{len(paths)}] unreadable ({type(e).__name__}): {Path(p).name}", flush=True)
            continue

        n = 0
        for qi in range(len(state)):
            if lens[qi] <= 0:
                continue
            f = state[qi] if small is None else np.concatenate([state[qi], small[qi]])
            feats.append(f)
            # The action the policy must commit to from this observation: the first executed one.
            acts.append(sa[qi, 0])
            n += 1
        del state, sa, lens, small
        gc.collect()
        if i % 5 == 0 or i == len(paths):
            print(f"    [{i}/{len(paths)}] {len(feats)} samples so far", flush=True)

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


def aliasing_curve(F: np.ndarray, A: np.ndarray, radii, n_sample: int, seed: int = 0):
    """Action disagreement at FIXED observation distances, normalised by random-pair disagreement.

    The k-NN ratio conflates two things: how well observations determine actions, and how densely
    the dataset samples observation space. A scripted controller produces smooth repetitive
    trajectories whose states cluster tightly, so its k nearest neighbours are far closer than a
    VLA rollout's (measured: 0.18 vs 0.73 in z-units) and its disagreement is mechanically lower —
    for reasons unrelated to aliasing. Evaluating at the SAME observation radius in both datasets
    removes that, so the numbers are comparable.
    """
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    Fz = (F - F.mean(0)) / (F.std(0) + 1e-8)
    tree = cKDTree(Fz)
    idx = rng.choice(len(Fz), size=min(n_sample, len(Fz)), replace=False)
    ri, rj = rng.choice(len(Fz), size=20000), rng.choice(len(Fz), size=20000)
    d_rnd = float(np.mean(np.linalg.norm(A[ri] - A[rj], axis=1)))

    out = []
    for r in radii:
        ds, npairs = [], 0
        for i in idx:
            for j in tree.query_ball_point(Fz[i], r):
                if j != i:
                    ds.append(np.linalg.norm(A[i] - A[j])); npairs += 1
            if npairs > 40000:
                break
        out.append((r, (float(np.mean(ds)) / d_rnd if ds else float("nan")), npairs))
    return out, d_rnd


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
    print("\n  AT MATCHED OBSERVATION RADIUS (density-independent — use THIS to compare datasets)")
    print(f"  {'radius (z)':>12}{'aliasing':>11}{'pairs':>10}")
    try:
        curve, _ = aliasing_curve(F, A, [0.1, 0.25, 0.5, 1.0], args.n_sample)
        for r, v, npairs in curve:
            print(f"  {r:>12.2f}{v:>11.3f}{npairs:>10}")
    except ImportError:
        print("    (scipy unavailable)")

    print("\n  Compare ratios ACROSS datasets — the absolute value depends on k and on how the")
    print("  observation is built, so it is a relative measure, not a threshold.\n")


if __name__ == "__main__":
    main()
