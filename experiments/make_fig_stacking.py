"""make_fig_stacking.py — Figure 5.1: the cost of combining mechanisms.

WHY A SLOPE CHART AND NOT BARS. The claim of Section 5.2 is not that the shield helps or that it
hurts; it is that it does BOTH, depending on what it is applied to. Attaching the shield raises the
base policy's success by 13.4 points and lowers the distilled policy's by 11.7. Four bars leave the
reader to compute that reversal; two crossing lines show it. The crossing point is the argument.

The right panel repeats the construction for collision to make clear that the reversal is confined
to the success channel: the shield reduces collisions in both cases, and monotonically. Safety and
capability come apart, and only one of them reverses.

Numbers from figures/results_shielded_distillation.md (eval 2026-08-07, n = 120 per condition).

    PYTHONPATH=. python -m experiments.make_fig_stacking
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

# (label, colour, k_success_off, k_coll_off, k_success_on, k_coll_on)
ARMS = [
    ("Base $\\pi_{0.5}$",        "#8C8C8C", 70, 99, 86, 16),
    ("Self-distilled (round 1)", "#2E6F9E", 99, 23, 85, 10),
]
EXPECTED = [((58.3, 82.5), (71.7, 13.3)), ((82.5, 19.2), (70.8, 8.3))]


def wilson(k, n, z=1.959963985):
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * p, 100.0 * (c - h), 100.0 * (c + h)


def main() -> int:
    for arm, ((so, co), (sn, cn)) in zip(ARMS, EXPECTED):
        for k, want in ((arm[2], so), (arm[3], co), (arm[4], sn), (arm[5], cn)):
            got = round(100.0 * k / N, 1)
            assert abs(got - want) < 0.06, f"{arm[0]}: {k}/{N}={got}% vs reported {want}%"

    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5))
    x = [0, 1]

    for ax, (title, i_off, i_on, better) in zip(axes, [
            ("Task success rate $\\uparrow$", 2, 4, "up"),
            ("Collision rate $\\downarrow$", 3, 5, "down")]):
        placed = []
        for label, colour, *ks in ARMS:
            p_off, lo_off, hi_off = wilson(ks[i_off - 2], N)
            p_on, lo_on, hi_on = wilson(ks[i_on - 2], N)
            ax.plot(x, [p_off, p_on], "o-", color=colour, lw=2.4, ms=9,
                    label=label, zorder=3)
            ax.errorbar(x, [p_off, p_on],
                        yerr=[[p_off - lo_off, p_on - lo_on], [hi_off - p_off, hi_on - p_on]],
                        fmt="none", ecolor=colour, elinewidth=1.3, capsize=4, zorder=2)
            delta = p_on - p_off
            y_lab = p_on
            while any(abs(y_lab - q) < 5.5 for q in placed):   # avoid coincident endpoints
                y_lab -= 5.5
            placed.append(y_lab)
            ax.annotate(f"{delta:+.1f} pp", xy=(1.05, y_lab), fontsize=10,
                        color=colour, va="center", fontweight="bold")
            ax.text(-0.06, p_off, f"{p_off:.1f}%", ha="right", va="center", fontsize=9.5)

        ax.set_xticks(x); ax.set_xticklabels(["shield off", "shield on"], fontsize=10.5)
        ax.set_xlim(-0.42, 1.42); ax.set_ylim(0, 100)
        ax.set_title(title, fontsize=12, pad=10)
        ax.set_ylabel("percent of rollouts", fontsize=10)
        ax.grid(axis="y", alpha=0.28, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].legend(fontsize=9.5, loc="lower left", framealpha=0.95)
    axes[0].text(0.5, 0.90, "the lines cross", ha="center", fontsize=10,
                 style="italic", color="#444444", transform=axes[0].transAxes)

    fig.suptitle("Attaching the shield helps the base policy and harms the distilled one",
                 fontsize=12.5, y=1.02)
    fig.text(0.5, -0.06,
             "The reversal is confined to the success channel. Collisions fall in both cases: "
             "safety and capability come apart,\nand only one of them reverses. "
             "$n = 120$ per condition, Wilson 95% intervals.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_stacking.{ext}", bbox_inches="tight", dpi=200)

    print(f"\n  {'arm':28s}{'TSR off':>9s}{'TSR on':>9s}{'delta':>9s}   "
          f"{'coll off':>9s}{'coll on':>9s}{'delta':>9s}")
    for label, colour, ks_off, kc_off, ks_on, kc_on in ARMS:
        so = wilson(ks_off, N)[0]; sn = wilson(ks_on, N)[0]
        co = wilson(kc_off, N)[0]; cn = wilson(kc_on, N)[0]
        print(f"  {label.replace('$','').replace(chr(92)+'pi_{0.5}','pi0.5'):28s}"
              f"{so:8.1f}%{sn:8.1f}%{sn-so:+8.1f} {co:9.1f}%{cn:8.1f}%{cn-co:+8.1f}")
    print(f"\n  -> {OUT}/fig_stacking.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
