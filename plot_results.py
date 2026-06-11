"""
Plot safety experiment results: Plain VLA vs VLA + CBF

Usage:
    python plot_results.py
    python plot_results.py --save   # save to figures/ instead of showing
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

PLAIN_CSV = "results_plain.csv"
CBF_CSV   = "results_cbf.csv"
SAFETY_RADIUS = 0.15

COLORS = {
    "plain":     "#e05c5c",
    "cbf":       "#4a90d9",
    "safety":    "#f5a623",
    "violation": "#d0021b",
    "cbf_act":   "#7ed321",
}


def load(path):
    df = pd.read_csv(path)
    return df


def summary_stats(df, label):
    n = len(df)
    violations = df["violation"].sum()
    min_d      = df["min_dist"].min()
    cbf_act    = df["cbf_triggered"].sum() if "cbf_triggered" in df.columns else 0
    print(f"  {label:20s}  violations={violations}/{n}  "
          f"min_dist={min_d:.3f}m  cbf_activations={cbf_act}")
    return {"label": label, "violations": violations, "n": n,
            "min_dist": min_d, "cbf_activations": cbf_act}


def fig_timeseries(plain, cbf, save_dir=None):
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.suptitle("VLA Safety Experiment — Minimum Distance to Obstacle Over Time",
                 fontsize=13, fontweight="bold")

    for ax, df, color, label in [
        (axes[0], plain, COLORS["plain"], "Plain VLA (no safety filter)"),
        (axes[1], cbf,   COLORS["cbf"],  "VLA + CBF (multi-link safety filter)"),
    ]:
        steps = df["step"]

        # Safety radius band
        ax.axhspan(0, SAFETY_RADIUS, alpha=0.12, color=COLORS["violation"],
                   label=f"Unsafe zone (< {SAFETY_RADIUS}m)")
        ax.axhline(SAFETY_RADIUS, color=COLORS["violation"], linewidth=1.2,
                   linestyle="--", alpha=0.8, label=f"Safety radius = {SAFETY_RADIUS}m")

        # CBF activations as background ticks
        if "cbf_triggered" in df.columns:
            cbf_steps = df[df["cbf_triggered"] == 1]["step"]
            for s in cbf_steps:
                ax.axvline(s, color=COLORS["cbf_act"], alpha=0.18, linewidth=0.7)

        # Violation markers
        viols = df[df["violation"] == 1]
        if not viols.empty:
            ax.scatter(viols["step"], viols["min_dist"],
                       color=COLORS["violation"], s=18, zorder=5,
                       label=f"Violations ({len(viols)})")

        # Distance curve
        ax.plot(steps, df["min_dist"], color=color, linewidth=1.6, label=label)

        ax.set_ylabel("Min distance to obstacle (m)", fontsize=10)
        ax.set_ylim(bottom=0)
        ax.set_xlim(0, len(df) - 1)
        ax.legend(fontsize=8, loc="upper right")
        ax.set_title(label, fontsize=11)
        ax.grid(axis="y", alpha=0.3)

    axes[1].set_xlabel("Simulation step", fontsize=10)
    fig.tight_layout()
    _save_or_show(fig, save_dir, "timeseries.png")


def fig_violation_bar(stats_list, save_dir=None):
    fig, axes = plt.subplots(1, 3, figsize=(11, 4))
    fig.suptitle("Summary: Plain VLA vs VLA + CBF", fontsize=13, fontweight="bold")

    labels  = [s["label"] for s in stats_list]
    colors  = [COLORS["plain"], COLORS["cbf"]]

    # Violation %
    pcts = [s["violations"] / s["n"] * 100 for s in stats_list]
    bars = axes[0].bar(labels, pcts, color=colors, width=0.5, edgecolor="white")
    axes[0].set_title("Safety violations (%)", fontsize=10)
    axes[0].set_ylabel("% of steps in violation")
    axes[0].set_ylim(0, 100)
    for bar, pct in zip(bars, pcts):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"{pct:.1f}%", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # Min distance
    min_ds = [s["min_dist"] for s in stats_list]
    bars2 = axes[1].bar(labels, min_ds, color=colors, width=0.5, edgecolor="white")
    axes[1].axhline(SAFETY_RADIUS, color=COLORS["violation"], linestyle="--",
                    linewidth=1.4, label=f"Safety radius {SAFETY_RADIUS}m")
    axes[1].set_title("Closest approach (m)", fontsize=10)
    axes[1].set_ylabel("Min distance (m)")
    axes[1].legend(fontsize=8)
    for bar, d in zip(bars2, min_ds):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                     f"{d:.3f}m", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # CBF activations
    acts = [s["cbf_activations"] for s in stats_list]
    bars3 = axes[2].bar(labels, acts, color=colors, width=0.5, edgecolor="white")
    axes[2].set_title("CBF activations", fontsize=10)
    axes[2].set_ylabel("Steps where CBF fired")
    for bar, a in zip(bars3, acts):
        axes[2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     str(a), ha="center", va="bottom", fontsize=9, fontweight="bold")

    for ax in axes:
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    _save_or_show(fig, save_dir, "summary_bar.png")


def fig_ee_trajectory(plain, cbf, save_dir=None):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_title("End-Effector Y Trajectory (lateral motion through obstacle zone)",
                 fontsize=11, fontweight="bold")

    obs_y, obs_z = 0.0, 0.45
    circle = plt.Circle((obs_y, obs_z), SAFETY_RADIUS,
                         color=COLORS["violation"], alpha=0.15, label="Safety radius")
    inner  = plt.Circle((obs_y, obs_z), 0.08,
                         color=COLORS["safety"],   alpha=0.5,  label="Obstacle (r=0.08m)")
    ax.add_patch(circle)
    ax.add_patch(inner)

    for df, color, label in [
        (plain, COLORS["plain"], "Plain VLA"),
        (cbf,   COLORS["cbf"],  "VLA + CBF"),
    ]:
        ax.plot(df["ee_y"], df["ee_z"], color=color, linewidth=1.8,
                label=label, alpha=0.85)
        ax.scatter(df["ee_y"].iloc[0],  df["ee_z"].iloc[0],
                   marker="o", s=60, color=color, zorder=5)
        ax.scatter(df["ee_y"].iloc[-1], df["ee_z"].iloc[-1],
                   marker="*", s=120, color=color, zorder=5)

    ax.set_xlabel("Y position (m) — lateral axis", fontsize=10)
    ax.set_ylabel("Z position (m) — height axis", fontsize=10)
    ax.legend(fontsize=9)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    _save_or_show(fig, save_dir, "ee_trajectory.png")


def _save_or_show(fig, save_dir, filename):
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action="store_true",
                        help="Save figures to figures/ directory instead of showing")
    args = parser.parse_args()
    save_dir = "figures" if args.save else None

    print("Loading CSVs...")
    plain = load(PLAIN_CSV)
    cbf   = load(CBF_CSV)

    print("\nSummary:")
    stats = [
        summary_stats(plain, "Plain VLA"),
        summary_stats(cbf,   "VLA + CBF"),
    ]

    print("\nGenerating figures...")
    fig_timeseries(plain, cbf, save_dir)
    fig_violation_bar(stats, save_dir)
    fig_ee_trajectory(plain, cbf, save_dir)

    if not save_dir:
        print("\nDone. Close plot windows to exit.")
