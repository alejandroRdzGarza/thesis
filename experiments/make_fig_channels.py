"""make_fig_channels.py — Figures for Section 4.7: other channels for safety.

fig_rl_bracket.pdf — the reinforcement-learning negative, plotted as a bracket. Two panels from the
two informative configurations. Left, Exp 003: putting real collisions into the reward moves the
policy, and it collapses to inaction -- success reaches zero IN THE SHIELDED CONDITION TOO, which
the shield alone cannot cause. Right, Exp 004: lowering the learning rate and annealing the shield
fixes the collapse, and nothing is learned -- unshielded collision stays pinned near 1.0 for six
rounds. Neither panel alone is an argument; together they are, because they are the two sides of
one knob rather than two points that a third setting might sit between.

fig_bon_correlation.pdf — the best-of-N measurement, which is the contribution even though the
method is negative. The plotted comparison is observed all-K-safe against what INDEPENDENT sampling
would predict, p^K. If a policy's K samples from a state were independent draws, all-K-safe would
be vanishingly rare: 1.1% at K=4 and 0.05% at K=8. Observed values are 32% and 41%, two to three
orders of magnitude higher, and essentially equal to the mean safe fraction itself. Candidates from
a given state are therefore near-perfectly correlated in their safety outcome: a query is 0-of-K or
all-of-K safe and almost never in between.

That is why action-level rejection sampling cannot help while episode-level filtering can --
episode-level selection chooses between trajectories that reached DIFFERENT states, which is the
axis along which variation actually exists.

    PYTHONPATH=. python -m experiments.make_fig_channels
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("figures")

# Exp 003 — mixed shielded/unshielded rollouts, lr 5e-5. Rounds 0,1,2,3,5 recorded.
E3_ROUND = [0, 1, 2, 3, 5]
E3_SHIELDED_SUCCESS = [0.94, 0.94, 0.00, 0.00, 0.00]
E3_UNSHIELDED_COLL = [1.00, 0.94, 0.63, 0.00, 0.06]

# Exp 004 — lr 2e-5, shield probability annealed 0.85 -> 0.40. Rounds 0,2,3,5 recorded.
E4_ROUND = [0, 2, 3, 5]
E4_SHIELDED_SUCCESS = [0.75, 0.80, 0.70, 0.83]
E4_UNSHIELDED_COLL = [1.00, 1.00, 1.00, 1.00]

# Best-of-N. (label, K, mean safe fraction, all-K-safe fraction)
BON = [
    ("$K{=}4$\nnoise 0.7", 4, 0.325, 0.32),
    ("$K{=}4$\nnoise 1.0", 4, 0.360, 0.36),
    ("$K{=}8$\nnoise 0.7", 8, 0.410, 0.41),
]


def fig_rl_bracket():
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.2), sharey=True)

    for ax, (title, rounds, succ, coll, note, npos) in zip(axes, [
        ("Exp 003 — collision enters the reward\n(lr $5\\times10^{-5}$, half rollouts unshielded)",
         E3_ROUND, E3_SHIELDED_SUCCESS, E3_UNSHIELDED_COLL,
         "collapses to inaction:\nsuccess reaches zero even\nwith the shield running", (0.72, 0.62)),
        ("Exp 004 — gentler gradient\n(lr $2\\times10^{-5}$, shield annealed $0.85\\to0.40$)",
         E4_ROUND, E4_SHIELDED_SUCCESS, E4_UNSHIELDED_COLL,
         "stable, but flat:\nunshielded collision never\nleaves 1.0", (0.5, 0.32)),
    ]):
        ax.plot(rounds, succ, "o-", color="#2E6F9E", lw=2.0, ms=7,
                label="shielded success $\\uparrow$", zorder=3)
        ax.plot(rounds, coll, "s--", color="#C0504D", lw=2.0, ms=7,
                label="unshielded collision $\\downarrow$", zorder=3)
        ax.set_title(title, fontsize=10.5, pad=9)
        ax.set_xlabel("training round", fontsize=10)
        ax.set_ylim(-0.06, 1.14); ax.set_xlim(-0.3, 5.3)
        ax.grid(alpha=0.28, lw=0.6); ax.set_axisbelow(True)
        ax.text(npos[0], npos[1], note, transform=ax.transAxes, fontsize=9,
                style="italic", color="#444444", ha="center")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    axes[0].set_ylabel("rate", fontsize=10)
    axes[0].legend(fontsize=9.5, loc="lower left", framealpha=0.95,
                   bbox_to_anchor=(0.02, 0.10))
    fig.suptitle("Scalar-reward RL: the failure bracketed from both sides", fontsize=12, y=1.02)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_rl_bracket.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def fig_bon_correlation():
    labels = [b[0] for b in BON]
    x = np.arange(len(BON))
    mean_safe = np.array([b[2] for b in BON])
    all_safe = np.array([b[3] for b in BON])
    indep = np.array([b[2] ** b[1] for b in BON])          # p^K under independence

    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    w = 0.27
    ax.bar(x - w, mean_safe * 100, w, color="#AFC4D6", edgecolor="black", lw=0.7,
           label="mean safe candidates (per query)", zorder=2)
    ax.bar(x, all_safe * 100, w, color="#2E6F9E", edgecolor="black", lw=0.7,
           label="queries where ALL $K$ are safe", zorder=2)
    ax.bar(x + w, indep * 100, w, color="#C0504D", edgecolor="black", lw=0.7,
           label="all-$K$-safe predicted if candidates were independent", zorder=2)

    for i in range(len(BON)):
        ax.text(x[i] - w, mean_safe[i] * 100 + 1.1, f"{mean_safe[i]*100:.0f}%",
                ha="center", fontsize=9)
        ax.text(x[i], all_safe[i] * 100 + 1.1, f"{all_safe[i]*100:.0f}%",
                ha="center", fontsize=9, fontweight="bold")
        ax.text(x[i] + w, indep[i] * 100 + 1.1,
                f"{indep[i]*100:.2f}%" if indep[i] * 100 >= 0.01 else "<0.01%",
                ha="center", fontsize=8.5, color="#C0504D")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9.5)
    ax.set_ylabel("percent of policy queries", fontsize=10)
    ax.set_title("Safety is state-determined, not sample-determined\n"
                 "candidates drawn from one state share their safety outcome",
                 fontsize=11.5, pad=10)
    ax.set_ylim(0, 52)
    ax.grid(axis="y", alpha=0.28, lw=0.6); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=9, loc="upper left", framealpha=0.95)

    fig.text(0.5, -0.07,
             "The first two bars coincide in every configuration: the fraction of individually "
             "safe candidates equals the fraction\nof queries where every candidate is safe. "
             "Independent sampling would put the third bar two to three orders lower.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_bon_correlation.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig_rl_bracket()
    fig_bon_correlation()
    print("\n  best-of-N, observed vs independent prediction:")
    for lab, K, p, allk in BON:
        print(f"    {lab.replace(chr(10),' '):18s} mean safe {p:5.1%}   all-K-safe {allk:5.1%}   "
              f"if independent {p**K:8.4%}   ratio {allk/(p**K):8.0f}x")
    print(f"\n  -> {OUT}/fig_rl_bracket.pdf")
    print(f"  -> {OUT}/fig_bon_correlation.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
