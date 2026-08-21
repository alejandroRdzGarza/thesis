"""make_fig_state_coverage.py — Figure 4.4: the state-coverage mechanism, measured.

THE CLAIM UNDER TEST. Section 4.4 argues the planner's demonstrations failed to transfer because
they do not cover the states pi0.5 actually reaches, not because the planner is a foreign
controller. That is a measurable statement: take the base policy's own visited states and ask how
close each demonstration set comes to them.

WHY SAMPLE SIZE MUST BE MATCHED -- this is the whole methodology. Planner episodes run to horizon
900 and shielded rollouts to horizon 300, so per scene the planner contributes roughly thirty times
more state points. Nearest-neighbour coverage rises monotonically with sample density, so the raw
comparison is rigged in the planner's favour and shows almost no difference (37.7% vs 32.8%). Both
sets are therefore subsampled to min(n_shielded, n_planner) points, averaged over 30 resamples.
Matched, the gap opens to 37.7% vs 22.8% and the shielded set covers better in 15 of 18 scenes.

WHAT THIS DOES NOT SHOW. Coverage is computed on end-effector position only -- three of the eight
observed dimensions, and not joint configuration. The shield's corrections are known to depend on
joint configuration the student never observes, so this figure supports the coverage account
without exhausting it.

SCENES. The 18 scenes holding both demonstration types. Six of the 24 have no planner
demonstrations at all, which is itself reported in Section 4.4 and is not a defect of this script.

    PYTHONPATH=. python -m experiments.make_fig_state_coverage
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

OUT = Path("figures")
SHIELDED = "pod_backup/results_shielded/round1_demos/{sc}_ep*trace.npz"
PLANNER = "results_distill/planner_A/{sc}/{sc}_ep*trace.npz"
BASE = "pod_backup/results_shielded/eval_base_nocbf/{sc}/*/rollout_*_trace.npz"
RADIUS = 0.02          # metres; "covered" means a demo state within 2 cm
RESAMPLES = 30
HERO = "safelibero_object_LII_t1"   # largest gap; used for the right-hand panel


def ee(path: str) -> np.ndarray | None:
    d = np.load(path, allow_pickle=True)
    return d["obs_state"][:, :3] if "obs_state" in d.files else None


def load(pattern: str, sc: str) -> np.ndarray | None:
    xs = [x for x in (ee(f) for f in glob.glob(pattern.format(sc=sc))) if x is not None]
    return np.vstack(xs) if xs else None


def main() -> int:
    rng = np.random.default_rng(0)
    scenes = sorted({re.search(r"(safelibero_\w+?_L(?:I|II)_t\d)_ep", f).group(1)
                     for f in glob.glob(SHIELDED.format(sc="*"))})

    rows, hero = [], None
    for sc in scenes:
        S, P, B = load(SHIELDED, sc), load(PLANNER, sc), load(BASE, sc)
        if S is None or P is None or B is None:
            continue
        m = min(len(S), len(P))
        cov = []
        for _ in range(RESAMPLES):
            Ss = S[rng.choice(len(S), m, replace=False)]
            Ps = P[rng.choice(len(P), m, replace=False)]
            cov.append([(cKDTree(Ss).query(B)[0] < RADIUS).mean(),
                        (cKDTree(Ps).query(B)[0] < RADIUS).mean()])
        cs, cp = np.array(cov).mean(0)
        rows.append((sc, cs, cp))
        if sc == HERO:
            hero = (S, P, B)

    assert rows, "no scenes had all three sources; check the paths at the top of this file"
    cs = np.array([r[1] for r in rows]); cp = np.array([r[2] for r in rows])
    wins = int((cs > cp).sum())

    fig = plt.figure(figsize=(10.6, 4.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15], wspace=0.28)

    # ---- (a) per-scene coverage, shielded vs planner -----------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    lim = max(cs.max(), cp.max()) * 100 * 1.12
    ax.plot([0, lim], [0, lim], ls="--", c="grey", lw=1.1, zorder=1)
    ax.scatter(cp * 100, cs * 100, s=58, c="#2E6F9E", edgecolor="black",
               linewidth=0.7, zorder=3)
    ax.fill_between([0, lim], [0, lim], [lim, lim], color="#2E6F9E", alpha=0.07, zorder=0)
    ax.text(lim * 0.06, lim * 0.93, "shielded covers better", fontsize=9.5,
            color="#2E6F9E", style="italic", va="top")
    ax.set_xlabel("planner demonstrations: base states covered (%)", fontsize=10)
    ax.set_ylabel("shielded demonstrations: base states covered (%)", fontsize=10)
    ax.set_title(f"(a) Coverage of the base policy's own states\n"
                 f"one point per scene; shielded wins {wins}/{len(rows)}",
                 fontsize=11, pad=8)
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.grid(alpha=0.25, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- (b) the distributions themselves, on the widest-gap scene ---------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    S, P, B = hero
    ax2.scatter(P[:, 0], P[:, 2], s=5, c="#C0504D", alpha=0.30, label="planner demos", zorder=2)
    ax2.scatter(S[:, 0], S[:, 2], s=5, c="#2E6F9E", alpha=0.55, label="shielded demos", zorder=3)
    ax2.scatter(B[:, 0], B[:, 2], s=13, facecolor="none", edgecolor="black",
                linewidth=0.55, label="base policy states", zorder=4)
    ax2.set_xlabel("end-effector $x$ (m)", fontsize=10)
    ax2.set_ylabel("end-effector $z$ (m)", fontsize=10)
    ax2.set_title(f"(b) {HERO.replace('safelibero_', '').replace('_', ' ')}\n"
                  f"widest gap: {cs[[r[0] for r in rows].index(HERO)]*100:.0f}% vs "
                  f"{cp[[r[0] for r in rows].index(HERO)]*100:.0f}% covered",
                  fontsize=11, pad=8)
    ax2.legend(fontsize=9, loc="best", framealpha=0.92)
    ax2.grid(alpha=0.25, lw=0.6); ax2.set_axisbelow(True)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)

    fig.text(0.5, -0.06,
             f"Coverage = fraction of base-policy states within {RADIUS*100:.0f} cm of a "
             f"demonstration state. Both sets subsampled to equal size per scene "
             f"({RESAMPLES} resamples).",
             ha="center", fontsize=9, style="italic")

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_state_coverage.{ext}", bbox_inches="tight", dpi=200)

    print(f"\n  matched sample size, {RESAMPLES} resamples, radius {RADIUS} m, {len(rows)} scenes\n")
    print(f"  {'scene':30s} {'shielded':>9s} {'planner':>9s}")
    for sc, a, b in rows:
        print(f"  {sc:30s} {a:9.1%} {b:9.1%}{'   *' if a > b else ''}")
    print(f"\n  pooled: shielded {cs.mean():.1%}  planner {cp.mean():.1%}  "
          f"(shielded better in {wins}/{len(rows)} scenes)")
    print(f"  -> {OUT}/fig_state_coverage.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
