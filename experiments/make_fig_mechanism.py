"""make_fig_mechanism.py — Figures for Section 4.6: what explains the improvement.

Produces two figures.

fig_matched_control.pdf — the confound killer. Three conditions on both metrics: the undistilled
base, a student trained on demonstrations collected with the shield SWITCHED OFF but filtered
identically, and a student trained on shielded demonstrations subsampled to the same count. The
only difference between the last two is whether the shield was active during collection, so the
gap between them is attributable to the shield and to nothing else in the pipeline. This is the
first objection any examiner raises about Section 4.3, so the answer should be visible.

fig_aliasing.pdf — a rejected explanation, plotted because the point is that the ordering is
INVERTED. The prior account said the planner fails because its corrections are not a function of
what the policy observes, which predicts higher action disagreement among observation-neighbours in
the planner set. The measurement puts the planner BELOW the shielded set at every radius: the
teacher that fails to distil has the cleaner observation-to-action mapping. A table states that; a
plot makes the reversal impossible to miss.

Comparison is at matched observation RADIUS rather than matched neighbour count, because a
nearest-neighbour ratio conflates aliasing with sampling density and the planner's trajectories are
far denser.

    PYTHONPATH=. python -m experiments.make_fig_mechanism
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

# (label, k_success, k_collision, shield_during_collection)
CONTROL = [
    ("Undistilled\nbase",                    70,  99, None),
    ("Control\nshield OFF\nduring collection", 60, 101, False),
    ("Matched\nshield ON\nduring collection",  91,  32, True),
]
EXPECTED = [(58.3, 82.5), (50.0, 84.2), (75.8, 26.7)]

# Action disagreement among observation-neighbours, normalised by random-pair disagreement.
RADII = [0.10, 0.25, 0.50, 1.00]
SHIELDED = [0.103, 0.174, 0.237, 0.376]
PLANNER = [0.051, 0.093, 0.118, 0.174]


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * p, 100.0 * (c - h), 100.0 * (c + h)


def fig_matched_control():
    for (lab, ks, kc, _), (ws, wc) in zip(CONTROL, EXPECTED):
        for k, w, nm in ((ks, ws, "success"), (kc, wc, "collision")):
            got = round(100.0 * k / N, 1)
            assert abs(got - w) < 0.06, f"{lab} {nm}: {k}/{N}={got}% vs reported {w}%"

    labels = [c[0] for c in CONTROL]
    x = np.arange(len(CONTROL))
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4))

    for ax, (title, idx, base_c) in zip(
            axes, [("Task success rate $\\uparrow$", 1, "#2E6F9E"),
                   ("Collision rate $\\downarrow$", 2, "#C0504D")]):
        pts, los, his = [], [], []
        for c in CONTROL:
            p, lo, hi = wilson(c[idx], N)
            pts.append(p); los.append(p - lo); his.append(hi - p)
        # grey = no distillation; pale = shield off; saturated = shield on
        cols = ["#9E9E9E", "#D8B9B8" if idx == 2 else "#AFC4D6", base_c]
        ax.bar(x, pts, width=0.6, color=cols, edgecolor="black", linewidth=0.8, zorder=2)
        ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=5, capthick=1.2, zorder=3)
        for i, p in enumerate(pts):
            ax.text(i, p + his[i] + 3, f"{p:.1f}%", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold")
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 100); ax.set_ylabel("percent of rollouts", fontsize=10)
        ax.grid(axis="y", alpha=0.28, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    fig.text(0.5, -0.10,
             "Both distilled conditions use 85 demonstrations passing an identical filter "
             "(succeeded and displaced nothing).\nThe conditions differ only in whether the "
             "shield was active while those demonstrations were generated.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_matched_control.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_aliasing():
    fig, ax = plt.subplots(figsize=(6.4, 4.3))
    ax.plot(RADII, SHIELDED, "o-", color="#2E6F9E", lw=2.0, ms=7,
            label="shielded demos (distil successfully)", zorder=3)
    ax.plot(RADII, PLANNER, "s--", color="#C0504D", lw=2.0, ms=7,
            label="planner demos (do not distil)", zorder=3)
    ax.fill_between(RADII, PLANNER, SHIELDED, color="grey", alpha=0.13, zorder=1)

    ax.annotate("the prior account predicted\nthe planner ABOVE this line",
                xy=(0.62, 0.29), xytext=(0.30, 0.335), fontsize=9, style="italic",
                color="#555555",
                arrowprops=dict(arrowstyle="->", color="#777777", lw=1.0))

    ax.set_xlabel("observation neighbourhood radius (z-scored)", fontsize=10.5)
    ax.set_ylabel("action disagreement among neighbours\n(normalised by random pairs)", fontsize=10)
    ax.set_title("Observation aliasing does not explain the failure", fontsize=11.5, pad=10)
    ax.legend(fontsize=9.5, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.28, lw=0.6); ax.set_axisbelow(True)
    ax.set_xlim(0.05, 1.05); ax.set_ylim(0, max(SHIELDED) * 1.25)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_aliasing.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig_matched_control()
    fig_aliasing()

    print(f"\n  {'condition':34s}{'TSR':>8s} {'95% CI':>14s}  {'coll':>7s} {'95% CI':>14s}")
    for c in CONTROL:
        ps, ls, hs = wilson(c[1], N); pc, lc, hc = wilson(c[2], N)
        print(f"  {c[0].replace(chr(10),' '):34s}{ps:7.1f}% [{ls:5.1f},{hs:5.1f}] "
              f"{pc:7.1f}% [{lc:5.1f},{hc:5.1f}]")
    print("\n  aliasing ratio (lower = better determined by the observation):")
    for r, s, p in zip(RADII, SHIELDED, PLANNER):
        print(f"    radius {r:.2f}   shielded {s:.3f}   planner {p:.3f}   "
              f"planner is {s/p:.2f}x cleaner")
    print(f"\n  -> {OUT}/fig_matched_control.pdf")
    print(f"  -> {OUT}/fig_aliasing.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
