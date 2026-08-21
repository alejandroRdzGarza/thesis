"""make_fig_shield_efficacy.py — Figure 4.1: does the shield make pi0.5 safe?

Two panels, base vs base+shield, on the two metrics that answer the section's question: task
success (arrow up = better) and collision rate (arrow down = better). Wilson 95% intervals as bars.

WHY COUNTS, NOT PERCENTAGES. The source table reports rounded percentages (58.3%, 82.5%, ...). A
Wilson interval computed from a rounded rate is not the interval that was reported, and the
discrepancy shows up in the third significant figure -- enough to make the figure disagree with the
table beside it. So the integer success/collision counts are stored here and every rate and
interval is recomputed from them. The script asserts that the recomputed rates round back to the
published ones, so a transcription error fails loudly instead of producing a plausible wrong figure.

SCOPE. Pooled over all 24 scenes (3 suites x 2 levels x 4 tasks), n = 120 held-out rollouts per
arm. A per-suite breakdown is NOT produced: the per-scene records for this evaluation were not
retained locally, and splitting the pooled figure by suite would require inventing them.

    python -m experiments.make_fig_shield_efficacy
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")
N = 120

# Counts, from figures/results_shielded_distillation.md (eval 2026-08-07, held-out inits 35-39).
# k_success / k_collision out of N. Percentages in that table are these counts over 120.
ARMS = {
    "Base $\\pi_{0.5}$\n(no shield)": {"success": 70, "collision": 99},
    "Base $\\pi_{0.5}$\n+ CBF shield": {"success": 86, "collision": 16},
}

# What the thesis text reports, to one decimal. The script refuses to run if the counts disagree.
EXPECTED = {
    "Base $\\pi_{0.5}$\n(no shield)": {"success": 58.3, "collision": 82.5},
    "Base $\\pi_{0.5}$\n+ CBF shield": {"success": 71.7, "collision": 13.3},
}


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    """Wilson score interval. Returns (point, lo, hi) as percentages."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return 100.0 * p, 100.0 * (centre - half), 100.0 * (centre + half)


def main() -> int:
    # ---- guard: counts must reproduce the published percentages -------------------------------
    for arm, counts in ARMS.items():
        for metric, k in counts.items():
            got = round(100.0 * k / N, 1)
            want = EXPECTED[arm][metric]
            assert abs(got - want) < 0.05, (
                f"{arm} {metric}: {k}/{N} = {got}% but the thesis reports {want}%. "
                f"Fix the count, do not fix the assertion."
            )

    labels = list(ARMS.keys())
    panels = [
        ("Task success rate", "success", "#2E6F9E", r"$\uparrow$"),
        ("Collision rate", "collision", "#C0504D", r"$\downarrow$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    x = np.arange(len(labels))

    for ax, (title, metric, colour, arrow) in zip(axes, panels):
        pts, los, his = [], [], []
        for arm in labels:
            p, lo, hi = wilson(ARMS[arm][metric], N)
            pts.append(p); los.append(p - lo); his.append(hi - p)

        bars = ax.bar(x, pts, width=0.55, color=colour, alpha=0.85,
                      edgecolor="black", linewidth=0.7, zorder=2)
        ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=5, capthick=1.2, zorder=3)

        for xi, (bar, p) in enumerate(zip(bars, pts)):
            ax.text(xi, p + his[xi] + 3.0, f"{p:.1f}%", ha="center", va="bottom",
                    fontsize=10, fontweight="bold")

        ax.set_title(f"{title} {arrow}", fontsize=11.5, pad=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0, 100); ax.set_ylabel("percent of rollouts", fontsize=10)
        ax.grid(axis="y", alpha=0.28, zorder=0, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle("Effect of the CBF runtime shield on base $\\pi_{0.5}$   "
                 "($n=120$ held-out rollouts per arm, Wilson 95% intervals)",
                 fontsize=11.5, y=1.02)
    fig.tight_layout()

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_shield_efficacy.{ext}", bbox_inches="tight", dpi=200)

    # ---- echo the numbers so the caption and table can be checked against them ---------------
    print(f"\n  n = {N} per arm, pooled over 24 scenes\n")
    print(f"  {'arm':32s} {'metric':10s} {'point':>7s}  {'95% Wilson':>16s}")
    for arm in labels:
        for metric in ("success", "collision"):
            p, lo, hi = wilson(ARMS[arm][metric], N)
            print(f"  {arm.replace(chr(10), ' '):32s} {metric:10s} {p:6.1f}%  [{lo:5.1f}, {hi:5.1f}]")
    print(f"\n  -> {OUT}/fig_shield_efficacy.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
