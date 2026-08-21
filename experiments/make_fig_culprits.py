"""make_fig_culprits.py — Figure 4.5: what the shield can and cannot reach.

THE POINT. The shield removes the end-effector channel completely -- zero gripper collisions across
all 360 shielded episodes -- while arm-link contacts survive it almost untouched. That asymmetry is
what licenses the arm-link comparison against the published end-effector-only formulation
(Section 4.2) and what the fidelity argument in 4.5 rests on.

WHY GROUPED AND NOT STACKED. The culprit categories OVERLAP: a single episode can be attributed to
more than one body, so the base row is 71 + 17 + 36 = 124 attributions across only 99 collided
episodes. Stacking would imply the parts sum to the whole and inflate the apparent total by a
quarter. Bars are therefore grouped, each read against the episode count, and the caption says so.

EXCLUDED. Contacts attributed to other scene objects. That category fires on every episode of every
condition including collision-free ones, because an obstacle resting on its supporting surface
registers a permanent contact. Counting it would make every condition look identical.

    PYTHONPATH=. python -m experiments.make_fig_culprits
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

# From the per-body attribution recovered by experiments/culprits_from_log.py.
# (label, shield_running, collided_episodes, gripper, arm_link, held_object)
ROWS = [
    ("Base",            False, 99, 71, 17, 36),
    ("Base\n+ shield",  True,  16,  0, 15,  0),
    ("Round 1",         False, 23,  7, 13,  9),
    ("Round 1\n+ shield", True, 10,  0,  8,  1),
    ("Round 2",         False, 21,  8,  8,  7),
    ("Round 2\n+ shield", True, 11,  0,  9,  0),
]
CULPRITS = [("gripper", "#C0504D"), ("arm link", "#E8A33D"), ("held object", "#7C6BA8")]


def wilson(k: int, n: int, z: float = 1.959963985) -> tuple[float, float]:
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100.0 * (c - h), 100.0 * (c + h)


def main() -> int:
    labels = [r[0] for r in ROWS]
    x = np.arange(len(ROWS))
    w = 0.26

    fig, ax = plt.subplots(figsize=(10.2, 4.6))

    for j, (name, colour) in enumerate(CULPRITS):
        vals = [r[3 + j] for r in ROWS]
        pos = x + (j - 1) * w
        bars = ax.bar(pos, vals, width=w, color=colour, edgecolor="black",
                      linewidth=0.7, zorder=2, label=name)
        for xi, (b, v, r) in enumerate(zip(bars, vals, ROWS)):
            if v == 0 and r[1]:                      # the punchline: zero, under a shield
                ax.text(pos[xi], 1.2, "0", ha="center", va="bottom", fontsize=10,
                        fontweight="bold", color=colour)
            elif v > 0:
                ax.text(pos[xi], v + 1.2, str(v), ha="center", va="bottom", fontsize=9)

    # episode-level collision count as a reference line per condition
    for xi, r in enumerate(ROWS):
        ax.plot([xi - 1.62 * w, xi + 1.62 * w], [r[2], r[2]], color="black",
                lw=1.4, ls=":", zorder=5)
        ax.text(xi + 1.75 * w, r[2], str(r[2]), ha="left", va="center",
                fontsize=8.5, style="italic", zorder=5)
    ax.plot([], [], color="black", lw=1.4, ls=":", label="episodes with any collision")

    # shade the shielded conditions so the zero-gripper pattern is visible at a glance
    for xi, r in enumerate(ROWS):
        if r[1]:
            ax.axvspan(xi - 0.45, xi + 0.45, color="grey", alpha=0.10, zorder=0)

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel(f"episodes attributed to this body (of {N})", fontsize=10)
    ax.set_title("Residual collisions by culprit body\n"
                 "shaded columns have the shield running at inference", fontsize=11.5, pad=10)
    ax.set_ylim(0, max(max(r[2] for r in ROWS), max(r[3] for r in ROWS)) * 1.20)
    ax.grid(axis="y", alpha=0.28, lw=0.6, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9.5, ncol=4, loc="upper right", framealpha=0.95)

    fig.text(0.5, -0.04,
             "Categories overlap: one episode may be attributed to more than one body, so bars "
             "are grouped rather than stacked and need not sum to the episode count.",
             ha="center", fontsize=9, style="italic")

    OUT.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_culprits.{ext}", bbox_inches="tight", dpi=200)

    print(f"\n  {'condition':20s}{'coll':>6s}{'grip':>6s}{'arm':>6s}{'held':>6s}   attribution sum")
    for r in ROWS:
        print(f"  {r[0].replace(chr(10),' '):20s}{r[2]:6d}{r[3]:6d}{r[4]:6d}{r[5]:6d}"
              f"   {r[3]+r[4]+r[5]:>3d} vs {r[2]} episodes")
    lo1, hi1 = wilson(17, N); lo2, hi2 = wilson(8, N)
    print(f"\n  arm-link base 17/120 = 14.2% [{lo1:.1f}, {hi1:.1f}]")
    print(f"  arm-link r2   8/120 =  6.7% [{lo2:.1f}, {hi2:.1f}]   -> intervals overlap")
    print(f"  gripper across all 360 shielded episodes: "
          f"{sum(r[3] for r in ROWS if r[1])}")
    print(f"\n  -> {OUT}/fig_culprits.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
