"""
Analyze and visualize results from run_experiment.py.

Reads results/<scene>_<mode>_steps.csv and results/<scene>_<mode>_summary.csv,
produces four figures and a printed comparison table.

Usage:
    python analyze_results.py                  # reads from results/
    python analyze_results.py --results-dir my_results --out figures/
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
    "figure.dpi": 150,
})

PLAIN_COLOR = "#4878CF"
CBF_COLOR   = "#D65F5F"
SCENE_ORDER = ["direct_block", "narrow_corridor", "lateral_offset", "high_stakes"]
SCENE_LABELS = {
    "direct_block":    "Direct\nBlock",
    "narrow_corridor": "Narrow\nCorridor",
    "lateral_offset":  "Lateral\nOffset",
    "high_stakes":     "High\nStakes",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_results(results_dir: Path):
    summaries, steps = {}, {}
    for mode in ("plain", "cbf"):
        for scene in SCENE_ORDER:
            key = (scene, mode)
            sum_path  = results_dir / f"{scene}_{mode}_summary.csv"
            step_path = results_dir / f"{scene}_{mode}_steps.csv"
            if sum_path.exists():
                summaries[key] = pd.read_csv(sum_path).iloc[0].to_dict()
            if step_path.exists():
                steps[key] = pd.read_csv(step_path)
    return summaries, steps


def _get(summaries, scene, mode, col, default=np.nan):
    return summaries.get((scene, mode), {}).get(col, default)


# ---------------------------------------------------------------------------
# Figure 1: Violation rate + min distance  (main comparison)
# ---------------------------------------------------------------------------
def fig_comparison(summaries: dict, out: Path):
    scenes = [s for s in SCENE_ORDER if (s, "plain") in summaries]
    x = np.arange(len(scenes))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle("Plain VLA vs CBF-VLA: Safety Comparison", fontweight="bold")

    # --- Violation rate ---
    ax = axes[0]
    plain_v = [_get(summaries, s, "plain", "violation_rate") * 100 for s in scenes]
    cbf_v   = [_get(summaries, s, "cbf",   "violation_rate") * 100 for s in scenes]
    ax.bar(x - w/2, plain_v, w, color=PLAIN_COLOR, label="Plain VLA",  alpha=0.85)
    ax.bar(x + w/2, cbf_v,   w, color=CBF_COLOR,   label="VLA + CBF",  alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABELS[s] for s in scenes])
    ax.set_ylabel("Violation rate (%)")
    ax.set_title("Safety violation rate (lower is better)")
    ax.legend()
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    # Improvement annotations
    for i, (p, c) in enumerate(zip(plain_v, cbf_v)):
        if not (np.isnan(p) or np.isnan(c)):
            diff = p - c
            color = "green" if diff > 0 else "red"
            ax.annotate(f"{diff:+.0f}%", xy=(i, max(p, c) + 1),
                        ha="center", fontsize=9, color=color, fontweight="bold")

    # --- Minimum distance ---
    ax = axes[1]
    plain_d = [_get(summaries, s, "plain", "min_dist_overall") * 100 for s in scenes]
    cbf_d   = [_get(summaries, s, "cbf",   "min_dist_overall") * 100 for s in scenes]
    safety_radii = {"direct_block": 15, "narrow_corridor": 13,
                    "lateral_offset": 15, "high_stakes": 18}

    ax.bar(x - w/2, plain_d, w, color=PLAIN_COLOR, label="Plain VLA",  alpha=0.85)
    ax.bar(x + w/2, cbf_d,   w, color=CBF_COLOR,   label="VLA + CBF",  alpha=0.85)

    # Safety radius reference lines per scene
    for i, scene in enumerate(scenes):
        r = safety_radii.get(scene)
        if r:
            ax.plot([i - w, i + w], [r, r], color="orange",
                    linewidth=1.5, linestyle="--", alpha=0.9)
    ax.plot([], [], color="orange", linestyle="--", linewidth=1.5,
            label="Safety radius")

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_LABELS[s] for s in scenes])
    ax.set_ylabel("Minimum distance (cm)")
    ax.set_title("Closest approach to obstacle (higher is better)")
    ax.legend()
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = out / "fig1_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 2: Min-distance time series per scene
# ---------------------------------------------------------------------------
def fig_timeseries(steps: dict, summaries: dict, out: Path):
    scenes = [s for s in SCENE_ORDER if (s, "plain") in steps]
    ncols  = 2
    nrows  = (len(scenes) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(11, 4 * nrows), sharey=False)
    axes = np.array(axes).flatten()
    fig.suptitle("Minimum Arm–Obstacle Distance Over Time", fontweight="bold")

    safety_radii = {"direct_block": 0.15, "narrow_corridor": 0.13,
                    "lateral_offset": 0.15, "high_stakes": 0.18}

    for i, scene in enumerate(scenes):
        ax = axes[i]
        for mode, color, label in [("plain", PLAIN_COLOR, "Plain VLA"),
                                    ("cbf",   CBF_COLOR,   "VLA + CBF")]:
            df = steps.get((scene, mode))
            if df is not None:
                ax.plot(df["step"], df["min_dist"], color=color,
                        label=label, linewidth=1.0, alpha=0.85)

        r = safety_radii.get(scene)
        if r:
            ax.axhline(r, color="orange", linestyle="--",
                       linewidth=1.5, label=f"Safety radius ({r:.2f} m)")

        ax.set_title(SCENE_LABELS[scene].replace("\n", " "))
        ax.set_xlabel("Step")
        ax.set_ylabel("Min dist (m)")
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_path = out / "fig2_timeseries.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 3: CBF activation analysis
# ---------------------------------------------------------------------------
def fig_cbf_activity(steps: dict, summaries: dict, out: Path):
    scenes = [s for s in SCENE_ORDER if (s, "cbf") in steps]
    fig, axes = plt.subplots(len(scenes), 1, figsize=(11, 3 * len(scenes)), sharex=False)
    if len(scenes) == 1:
        axes = [axes]
    fig.suptitle("CBF Correction Norm Over Time (CBF mode only)", fontweight="bold")

    for ax, scene in zip(axes, scenes):
        df = steps.get((scene, "cbf"))
        if df is None:
            continue
        active = df[df["cbf_triggered"] == 1]
        inactive = df[df["cbf_triggered"] == 0]

        ax.scatter(inactive["step"], inactive["cbf_correction_norm"],
                   s=2, color="lightgray", label="CBF inactive", alpha=0.6)
        ax.scatter(active["step"], active["cbf_correction_norm"],
                   s=4, color=CBF_COLOR, label="CBF active", alpha=0.8)

        rate = _get(summaries, scene, "cbf", "cbf_activation_rate", 0) * 100
        ax.set_title(f"{SCENE_LABELS[scene].replace(chr(10), ' ')}  "
                     f"(activation rate: {rate:.1f}%)")
        ax.set_ylabel("||u_safe − u_nom||")
        ax.set_xlabel("Step")
        ax.legend(markerscale=3, fontsize=9)
        ax.yaxis.grid(True, linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    fig.tight_layout()
    out_path = out / "fig3_cbf_activity.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Figure 4: EE trajectory (top-down XY view)
# ---------------------------------------------------------------------------
def fig_trajectories(steps: dict, out: Path):
    scenes = [s for s in SCENE_ORDER if (s, "plain") in steps]
    ncols  = 2
    nrows  = (len(scenes) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 4.5 * nrows))
    axes = np.array(axes).flatten()
    fig.suptitle("End-Effector Trajectory (top-down view)", fontweight="bold")

    obs_positions = {
        "direct_block":    [(0.6,  0.0,  0.08)],
        "narrow_corridor": [(0.6,  0.0,  0.07), (0.6, 0.0, 0.07)],
        "lateral_offset":  [(0.45, 0.0,  0.08)],
        "high_stakes":     [(0.6,  0.02, 0.09)],
    }
    start_goals = {
        "direct_block":    ((-0.2, 0.6), (0.2, 0.6)),
        "narrow_corridor": ((-0.25, 0.6), (0.25, 0.6)),
        "lateral_offset":  ((-0.2, 0.6), (0.2, 0.6)),
        "high_stakes":     ((-0.2, 0.6), (0.2, 0.6)),
    }

    for i, scene in enumerate(scenes):
        ax = axes[i]
        for mode, color, label in [("plain", PLAIN_COLOR, "Plain VLA"),
                                    ("cbf",   CBF_COLOR,   "VLA + CBF")]:
            df = steps.get((scene, mode))
            if df is not None:
                ax.plot(df["ee_y"], df["ee_x"], color=color,
                        label=label, linewidth=1.2, alpha=0.8)
                ax.plot(df["ee_y"].iloc[0], df["ee_x"].iloc[0],
                        "o", color=color, markersize=5)

        # Obstacle circles
        for obs in obs_positions.get(scene, []):
            ox, oy, r = obs
            circle = plt.Circle((oy, ox), r, color="orange",
                                 alpha=0.35, zorder=5)
            ax.add_patch(circle)
            ax.plot(oy, ox, "+", color="darkorange", markersize=8, zorder=6)

        # Start / goal markers
        sg = start_goals.get(scene)
        if sg:
            (sy, sx), (gy, gx) = sg
            ax.plot(sy, sx, "rs", markersize=7, zorder=7, label="Start")
            ax.plot(gy, gx, "g^", markersize=7, zorder=7, label="Goal")

        ax.set_title(SCENE_LABELS[scene].replace("\n", " "))
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.4)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    out_path = out / "fig4_trajectories.png"
    fig.savefig(out_path, bbox_inches="tight")
    print(f"  Saved {out_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Console summary table
# ---------------------------------------------------------------------------
def print_table(summaries: dict):
    cols = [
        ("violation_rate",      "Viol. rate", "{:.1%}"),
        ("min_dist_overall",    "Min dist",   "{:.3f} m"),
        ("mean_min_dist",       "Mean dist",  "{:.3f} m"),
        ("cbf_activation_rate", "CBF rate",   "{:.1%}"),
        ("path_length_m",       "Path len",   "{:.3f} m"),
        ("goal_reached",        "Goal?",      "{}"),
    ]
    header = f"{'Scene':<22} {'Mode':<6}" + "".join(f"  {h:>12}" for _, h, _ in cols)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for scene in SCENE_ORDER:
        for mode in ("plain", "cbf"):
            key = (scene, mode)
            if key not in summaries:
                continue
            s = summaries[key]
            row = f"{scene:<22} {mode:<6}"
            for col, _, fmt in cols:
                val = s.get(col, "—")
                try:
                    row += f"  {fmt.format(val):>12}"
                except Exception:
                    row += f"  {'—':>12}"
            print(row)
        print("-" * len(header))

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--out", default="figures")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading results from {results_dir}/")
    summaries, steps = load_results(results_dir)

    if not summaries:
        print("No summary files found. Run experiments first with run_experiment.py.")
        return

    print_table(summaries)

    print("Generating figures...")
    fig_comparison(summaries, out_dir)
    fig_timeseries(steps, summaries, out_dir)
    fig_cbf_activity(steps, summaries, out_dir)
    fig_trajectories(steps, out_dir)

    print(f"\nAll figures saved to {out_dir}/")


if __name__ == "__main__":
    main()
