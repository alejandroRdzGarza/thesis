"""make_fig_internalisation.py — Figure 4.2, the headline figure.

Three conditions on two metrics: the base policy, the base policy with the shield running, and the
self-distilled policy with NO shield at inference. The figure has to carry the whole claim on its
own, because it is the one a skimming reader looks at: safety up, success up, filter switched off.

DESIGN. Shield-free arms are drawn solid, the shielded arm hatched, and the legend says which is
which -- the single most misreadable thing about this result is someone assuming the distilled bar
still has a shield behind it. A footline states n and the interval type so the figure survives
being lifted out of the document.

WHY COUNTS, NOT PERCENTAGES. Same reason as make_fig_shield_efficacy.py: a Wilson interval computed
from a rounded rate differs from the reported one in the third significant figure, which would make
this figure disagree with the table beside it. Integer counts are stored and everything is
recomputed; an assertion fails loudly if a count does not round back to the published percentage.

    PYTHONPATH=. python -m experiments.make_fig_internalisation
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

OUT = Path("figures")
N = 120

# From figures/results_shielded_distillation.md (eval 2026-08-07, held-out inits 35-39).
# (label, k_success, k_collision, shield_at_inference)
ARMS = [
    ("Base $\\pi_{0.5}$",              70, 99, True),
    ("Base $\\pi_{0.5}$\n+ shield",    86, 16, False),
    ("Self-distilled\n(round 1)",      99, 23, True),
]
# shield_at_inference is encoded as "shield_free" below; kept positional for readability.
SHIELD_FREE = [True, False, True]

EXPECTED = [(58.3, 82.5), (71.7, 13.3), (82.5, 19.2)]

C_FREE, C_SHIELD = "#2E6F9E", "#8FAFC7"
C_FREE_R, C_SHIELD_R = "#C0504D", "#DDA6A4"


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float, float]:
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * p, 100.0 * (c - h), 100.0 * (c + h)


def main() -> int:
    for (label, ks, kc, _), (want_s, want_c) in zip(ARMS, EXPECTED):
        for k, want, name in ((ks, want_s, "success"), (kc, want_c, "collision")):
            got = round(100.0 * k / N, 1)
            assert abs(got - want) < 0.05, (
                f"{label} {name}: {k}/{N} = {got}% but the thesis reports {want}%. "
                f"Fix the count, not the assertion.")

    labels = [a[0] for a in ARMS]
    x = np.arange(len(ARMS))

    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.3))
    panels = [
        ("Task success rate $\\uparrow$", 1, C_FREE, C_SHIELD),
        ("Collision rate $\\downarrow$",  2, C_FREE_R, C_SHIELD_R),
    ]

    for ax, (title, idx, col_free, col_shield) in zip(axes, panels):
        pts, los, his = [], [], []
        for arm in ARMS:
            p, lo, hi = wilson(arm[idx], N)
            pts.append(p); los.append(p - lo); his.append(hi - p)

        for i, (p, free) in enumerate(zip(pts, SHIELD_FREE)):
            ax.bar(i, p, width=0.6,
                   color=col_free if free else col_shield,
                   hatch="" if free else "///",
                   edgecolor="black", linewidth=0.8, zorder=2)

        ax.errorbar(x, pts, yerr=[los, his], fmt="none", ecolor="black",
                    elinewidth=1.2, capsize=5, capthick=1.2, zorder=3)
        for i, p in enumerate(pts):
            ax.text(i, p + his[i] + 3.0, f"{p:.1f}%", ha="center", va="bottom",
                    fontsize=10.5, fontweight="bold")

        ax.set_title(title, fontsize=12, pad=10)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
        ax.set_ylim(0, 100); ax.set_ylabel("percent of rollouts", fontsize=10)
        ax.grid(axis="y", alpha=0.28, zorder=0, linewidth=0.6); ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    # The legend is load-bearing: it is what stops a reader assuming the distilled bar is shielded.
    fig.legend(handles=[
        Patch(facecolor="white", edgecolor="black", label="no shield at inference"),
        Patch(facecolor="white", edgecolor="black", hatch="///", label="shield running"),
    ], loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.06))

    fig.text(0.5, -0.13, "$n=120$ held-out rollouts per condition, pooled over 24 scenes; "
                         "error bars are Wilson 95% intervals",
             ha="center", fontsize=9, style="italic")

    fig.tight_layout()
    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_internalisation.{ext}", bbox_inches="tight", dpi=200)

    print(f"\n  {'condition':22s} {'TSR':>7s} {'95% CI':>15s}   {'coll':>7s} {'95% CI':>15s}")
    for arm in ARMS:
        ps, ls, hs = wilson(arm[1], N)
        pc, lc, hc = wilson(arm[2], N)
        print(f"  {arm[0].replace(chr(10),' '):22s} {ps:6.1f}% [{ls:5.1f},{hs:5.1f}]   "
              f"{pc:6.1f}% [{lc:5.1f},{hc:5.1f}]")
    print(f"\n  -> {OUT}/fig_internalisation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
