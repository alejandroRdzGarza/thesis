"""
curate_dataset.py — inspect and filter a collected obstacle-conditioned dataset.

Usage:
    # Print stats and apply filters (dry run)
    python -m experiments.curate_dataset --dataset data/obs_cond_dataset/safelibero_spatial_t00_LI

    # Actually copy kept episodes to a new directory
    python -m experiments.curate_dataset \
        --dataset data/obs_cond_dataset/safelibero_spatial_t00_LI \
        --out data/obs_cond_curated/safelibero_spatial_t00_LI \
        --min-cbf-steps 5 --max-discard-rate 0.30
"""

from __future__ import annotations
import argparse
import shutil
from pathlib import Path

import numpy as np


def analyse_episode(path: Path) -> dict:
    d = np.load(path)
    labels    = d["label"]
    n_total   = len(labels)
    n_nom     = int((labels == 0).sum())
    n_cbf     = int((labels == 1).sum())
    n_disc    = int((labels == -1).sum())   # already filtered out; kept for info

    # obs_dist is stored in obs_feat[:, 3] (the 4th element)
    obs_feat  = d["obs_feat"]              # (N, 4)
    obs_dists = obs_feat[:, 3]            # normalised dist ∈ [0,1]

    # correction magnitude (if saved)
    corr_mags = d["corr_mag"] if "corr_mag" in d else np.zeros(n_total)

    return {
        "file":        path,
        "n_total":     n_total,
        "n_nom":       n_nom,
        "n_cbf":       n_cbf,
        "n_disc":      n_disc,
        "cbf_frac":    n_cbf / max(n_total, 1),
        "disc_rate":   n_disc / max(n_total + n_disc, 1),
        "min_obs_dist": float(obs_dists.min()) if len(obs_dists) else 1.0,
        "mean_corr":   float(corr_mags[corr_mags > 0].mean()) if (corr_mags > 0).any() else 0.0,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset",          required=True,
                   help="Path to episode directory (.npz files)")
    p.add_argument("--out",              default=None,
                   help="Output directory for curated episodes (omit = dry run)")
    p.add_argument("--min-cbf-steps",   type=int,   default=5,
                   help="Min CBF-corrected steps per episode (default 5)")
    p.add_argument("--max-discard-rate",type=float, default=0.30,
                   help="Max fraction of discarded steps per episode (default 0.30)")
    p.add_argument("--min-obs-dist",    type=float, default=None,
                   help="Optional: only keep episodes where min obs_dist < X (0-1 normalised)")
    args = p.parse_args()

    src = Path(args.dataset)
    files = sorted(src.glob("ep_*.npz"))
    if not files:
        print(f"No .npz files found in {src}")
        return

    stats = [analyse_episode(f) for f in files]

    # ── Print per-episode table ────────────────────────────────────────────────
    print(f"\n{'File':<20}  {'total':>6}  {'nom':>5}  {'cbf':>5}  "
          f"{'disc':>5}  {'cbf%':>6}  {'disc%':>6}  {'min_d':>6}  {'corr':>6}  keep?")
    print("-" * 90)

    keep, drop = [], []
    for s in stats:
        ok = (s["n_cbf"] >= args.min_cbf_steps and
              s["disc_rate"] <= args.max_discard_rate)
        if args.min_obs_dist is not None:
            ok = ok and (s["min_obs_dist"] < args.min_obs_dist)

        tag = "✓" if ok else "✗"
        print(f"  {s['file'].name:<18}  {s['n_total']:>6}  {s['n_nom']:>5}  "
              f"{s['n_cbf']:>5}  {s['n_disc']:>5}  "
              f"{s['cbf_frac']*100:>5.1f}%  {s['disc_rate']*100:>5.1f}%  "
              f"{s['min_obs_dist']:>6.3f}  {s['mean_corr']:>6.4f}  {tag}")
        (keep if ok else drop).append(s)

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n  Episodes: {len(keep)} kept / {len(drop)} dropped / {len(stats)} total")
    if keep:
        kept_steps = sum(s["n_total"] for s in keep)
        kept_cbf   = sum(s["n_cbf"]   for s in keep)
        kept_nom   = sum(s["n_nom"]   for s in keep)
        print(f"  Kept steps : {kept_steps}  (nom={kept_nom}  cbf={kept_cbf}  "
              f"cbf_frac={kept_cbf/max(kept_steps,1)*100:.1f}%)")
        print(f"  Recommend weighting cbf steps ×{max(1, round(kept_nom/max(kept_cbf,1)))}"
              f" in training to balance labels.")

    # ── Copy if --out specified ────────────────────────────────────────────────
    if args.out and keep:
        dst = Path(args.out)
        dst.mkdir(parents=True, exist_ok=True)
        for s in keep:
            shutil.copy2(s["file"], dst / s["file"].name)
        print(f"\n  Curated dataset written to {dst}/  ({len(keep)} episodes)")
    elif args.out:
        print("\n  Nothing to copy — all episodes dropped.")
    else:
        print("\n  Dry run — pass --out <dir> to copy kept episodes.")


if __name__ == "__main__":
    main()
