"""
DAgger round-over-round analysis: behavioral shift, failure mode breakdown,
CBF dependence reduction, EE trajectory comparison.

Usage
-----
# Compare base model vs DAgger rounds
python experiments/dagger_analysis.py \
    --rounds base_pi05 dagger_r0 dagger_r1 \
    --dirs   results_eval/base_pi05 results_eval/dagger_r0 results_eval/dagger_r1

# Include trajectory analysis from saved .npz files
python experiments/dagger_analysis.py \
    --rounds base_pi05 dagger_r0 \
    --dirs   results_eval/base_pi05 results_eval/dagger_r0 \
    --npz-dirs results_eval/base_pi05 results_eval/dagger_r0 \
    --plot

# Single suite focus
python experiments/dagger_analysis.py \
    --rounds base dagger_r0 \
    --dirs   results_eval/base_pi05 results_eval/dagger_r0 \
    --suite  safelibero_spatial

Output
------
1. Cross-round key metric table (CAR, TSR, collision, CBF activation, deadlock, path eff)
2. Failure mode breakdown per round (GRASP_NEVER / COLLISION_APPROACH / DEADLOCK / ...)
3. Behavioral shift summary (are we reducing CBF dependence?)
4. (with --npz-dirs) Per-round EE trajectory stats from saved .npz files
5. (with --plot) Matplotlib figures saved to --plot-dir
"""

from __future__ import annotations
import argparse
import csv
import glob
import os
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np

# ── Failure mode taxonomy ─────────────────────────────────────────────────────
# Priority order matters — first matching rule wins.
FAILURE_MODES = [
    "SUCCESS",
    "COLLISION_APPROACH",   # collision before grasp
    "COLLISION_POST_GRASP", # collision after grasp
    "DEADLOCK",             # stuck for many steps
    "GRASP_THEN_FAIL",      # grasped but didn't complete task
    "GRASP_NEVER",          # never approached / grasped object
]

_DEADLOCK_THRESHOLD = [30]  # mutable so main() can update it without 'global'


def classify_episode(ep: dict) -> str:
    tsr       = int(float(ep.get("tsr", 0)))
    collision = int(float(ep.get("collision", 0)))
    grasp     = int(float(ep.get("grasp_achieved", 0)))
    deadlock  = int(float(ep.get("deadlock_steps", 0)))

    if tsr:
        return "SUCCESS"
    if collision and not grasp:
        return "COLLISION_APPROACH"
    if collision and grasp:
        return "COLLISION_POST_GRASP"
    if deadlock > _DEADLOCK_THRESHOLD[0]:
        return "DEADLOCK"
    if grasp:
        return "GRASP_THEN_FAIL"
    return "GRASP_NEVER"


# ── Data loading ──────────────────────────────────────────────────────────────

def load_episodes(directory: Path, suite_filter: Optional[str] = None) -> list[dict]:
    """Load all *_episodes.csv files from a results directory."""
    rows = []
    for csv_path in sorted(directory.glob("*_episodes.csv")):
        if suite_filter and suite_filter not in csv_path.name:
            continue
        with open(csv_path) as f:
            for row in csv.DictReader(f):
                row["_source_file"] = csv_path.name
                rows.append(row)
    return rows


def load_agg(directory: Path, suite_filter: Optional[str] = None) -> list[dict]:
    """Load all *_agg.csv files from a results directory."""
    rows = []
    for csv_path in sorted(directory.glob("*_agg.csv")):
        if suite_filter and suite_filter not in csv_path.name:
            continue
        with open(csv_path) as f:
            rows.extend(csv.DictReader(f))
    return rows


def load_npz_stats(directory: Path, suite_filter: Optional[str] = None) -> dict:
    """
    Load saved .npz trajectory files and compute per-round aggregate stats.

    Returns dict with keys:
      cbf_rate_mean      — fraction of steps CBF was active (mean across episodes)
      h_min_mean         — mean minimum h value across all CBF-active steps
      correction_mean    — mean correction norm across all CBF-active steps
      path_length_mean   — mean path length (m)
      ee_final_dist_mean — mean ||ee_pos[-1] - ee_pos[0]|| (displacement)
      deadlock_frac      — fraction of episodes with ≥1 deadlock steps
      n_episodes         — total episodes loaded
    """
    pattern = "**/*.npz"
    npz_files = sorted(directory.glob(pattern))
    if not npz_files:
        npz_files = sorted(directory.glob("*.npz"))

    if suite_filter:
        npz_files = [p for p in npz_files if suite_filter in str(p)]

    cbf_rates, h_mins, corr_norms, path_lengths, displacements = [], [], [], [], []
    n_loaded = 0

    for npz_path in npz_files:
        try:
            d = np.load(npz_path)
        except Exception:
            continue
        n_loaded += 1

        triggered = d["cbf_triggered"]  # (T,) bool
        cbf_rates.append(triggered.mean())

        if triggered.any():
            h_arr = d["h_values"]        # (T, K)
            h_min_per_step = h_arr.min(axis=1)
            active_h = h_min_per_step[triggered]
            finite = active_h[np.isfinite(active_h)]
            if len(finite):
                h_mins.append(float(finite.mean()))

            corr = d["cbf_correction_norm"]
            corr_norms.append(float(corr[triggered].mean()))

        ee = d["ee_pos"]                 # (T, 3)
        vels = np.linalg.norm(np.diff(ee, axis=0), axis=1)
        path_lengths.append(float(vels.sum()))
        displacements.append(float(np.linalg.norm(ee[-1] - ee[0])))

    if n_loaded == 0:
        return {"n_episodes": 0}

    return {
        "n_episodes":         n_loaded,
        "cbf_rate_mean":      float(np.mean(cbf_rates)),
        "h_min_mean":         float(np.mean(h_mins))    if h_mins    else float("inf"),
        "correction_mean":    float(np.mean(corr_norms)) if corr_norms else 0.0,
        "path_length_mean":   float(np.mean(path_lengths)),
        "displacement_mean":  float(np.mean(displacements)),
    }


# ── Printing helpers ──────────────────────────────────────────────────────────

def _fmt(val, fmt=".2f", fallback="N/A"):
    try:
        return format(float(val), fmt)
    except (TypeError, ValueError):
        return fallback


def _mean_col(rows: list[dict], key: str) -> Optional[float]:
    vals = []
    for r in rows:
        v = r.get(key)
        if v is not None and v != "" and v != "inf":
            try:
                vals.append(float(v))
            except ValueError:
                pass
    return float(np.mean(vals)) if vals else None


def _section(title: str):
    print(f"\n{'='*78}")
    print(f"  {title}")
    print(f"{'='*78}")


# ── Analysis functions ────────────────────────────────────────────────────────

def print_cross_round_agg(rounds: list[str], all_agg: dict[str, list[dict]],
                          suite_filter: Optional[str] = None):
    """Print aggregate metric table — rows=scenes, columns=rounds."""

    metrics = [
        ("car_pct",                   "CAR%  ↑", ">7.1f"),
        ("tsr_pct",                   "TSR%  ↑", ">7.1f"),
        ("collision_rate_pct",        "Coll% ↓", ">7.1f"),
        ("mean_cbf_activation_rate",  "CBF%  ↓", ">7.3f"),
        ("mean_cbf_mean_h_active",    "h̄_cbf ↑", ">7.3f"),
        ("mean_deadlock_steps",       "Deadlk↓", ">7.1f"),
        ("mean_path_efficiency",      "PathEff↑",">7.3f"),
        ("grasp_rate_pct",            "Grasp% ↑",">7.1f"),
        ("ets",                       "ETS   ↓", ">7.1f"),
    ]

    # Build lookup: round → {scene_mode_key: row}
    lookup: dict[str, dict[str, dict]] = {}
    for rnd in rounds:
        lookup[rnd] = {}
        for row in all_agg.get(rnd, []):
            key = f"{row.get('scene','?')}_{row.get('mode','?')}_{row.get('safety_level','?')}"
            lookup[rnd][key] = row

    all_keys = sorted(set(k for d in lookup.values() for k in d))
    if not all_keys:
        print("  (no aggregate data found)")
        return

    col_w = max(9, max(len(r) for r in rounds) + 1)
    key_w = min(55, max(len(k) for k in all_keys))

    for metric, label, fmt_spec in metrics:
        _section(f"{label}")
        hdr = f"  {'Scene / Mode / Level':<{key_w}}"
        for rnd in rounds:
            hdr += f"  {rnd:>{col_w}}"
        print(hdr)
        print(f"  {'-'*key_w}" + f"  {'-'*col_w}" * len(rounds))
        for key in all_keys:
            line = f"  {key:<{key_w}}"
            for rnd in rounds:
                val = lookup[rnd].get(key, {}).get(metric, "N/A")
                try:
                    line += f"  {float(val):{fmt_spec}}"
                except (TypeError, ValueError):
                    line += f"  {'N/A':>{col_w}}"
            print(line)

    # Averages
    _section("AVERAGES ACROSS ALL SCENES (CBF mode only)")
    hdr = f"  {'Metric':<30}"
    for rnd in rounds:
        hdr += f"  {rnd:>{col_w}}"
    print(hdr)
    print(f"  {'-'*30}" + f"  {'-'*col_w}" * len(rounds))
    for metric, label, fmt_spec in metrics:
        line = f"  {label:<30}"
        for rnd in rounds:
            vals = [
                float(row[metric])
                for row in all_agg.get(rnd, [])
                if row.get("mode") == "cbf"
                and row.get(metric) not in (None, "", "inf")
                and row.get(metric) is not None
            ]
            try:
                avg = float(np.mean([float(v) for v in vals])) if vals else None
                line += f"  {avg:{fmt_spec}}" if avg is not None else f"  {'N/A':>{col_w}}"
            except Exception:
                line += f"  {'N/A':>{col_w}}"
        print(line)


def print_failure_breakdown(rounds: list[str], all_episodes: dict[str, list[dict]]):
    """Per-round failure mode counts and percentages."""
    _section("FAILURE MODE BREAKDOWN")

    col_w = max(10, max(len(r) for r in rounds) + 1)
    mode_w = max(len(m) for m in FAILURE_MODES)

    hdr = f"  {'Failure Mode':<{mode_w}}"
    for rnd in rounds:
        hdr += f"  {rnd:>{col_w}} {'%':>5}"
    print(hdr)
    print(f"  {'-'*mode_w}" + f"  {'-'*(col_w+6)}" * len(rounds))

    round_counts: dict[str, dict[str, int]] = {}
    round_totals: dict[str, int] = {}
    for rnd in rounds:
        eps = all_episodes.get(rnd, [])
        counts: dict[str, int] = defaultdict(int)
        for ep in eps:
            counts[classify_episode(ep)] += 1
        round_counts[rnd] = counts
        round_totals[rnd] = len(eps)

    for mode in FAILURE_MODES:
        line = f"  {mode:<{mode_w}}"
        for rnd in rounds:
            n = round_counts[rnd].get(mode, 0)
            total = round_totals.get(rnd, 1) or 1
            pct = 100.0 * n / total
            line += f"  {n:>{col_w}d} {pct:>5.1f}%"
        print(line)

    print(f"  {'-'*mode_w}" + f"  {'-'*(col_w+6)}" * len(rounds))
    line = f"  {'TOTAL':<{mode_w}}"
    for rnd in rounds:
        n = round_totals.get(rnd, 0)
        line += f"  {n:>{col_w}d} {'100%':>5}"
    print(line)


def print_behavioral_shift(rounds: list[str], all_episodes: dict[str, list[dict]]):
    """
    Show how the model's behavior changes across DAgger rounds.

    Key signal: CBF activation rate dropping = model learned to avoid obstacles
    on its own without needing correction.
    """
    _section("BEHAVIORAL SHIFT (CBF mode episodes)")

    metrics = [
        ("cbf_activation_rate",  "CBF activation rate ↓ (less CBF needed)"),
        ("cbf_mean_h_active",    "Mean h when CBF active ↑ (less urgent corrections)"),
        ("cbf_interv_mean_dur",  "Mean CBF intervention duration ↓"),
        ("deadlock_steps",       "Deadlock steps ↓ (less CBF-policy conflict)"),
        ("path_efficiency",      "Path efficiency ↑ (smoother trajectories)"),
        ("path_length_m",        "Path length m ↓ (more direct)"),
        ("obj_dist_min",         "Min obj distance ↓ (better approach)"),
    ]

    col_w = max(10, max(len(r) for r in rounds) + 1)
    label_w = max(len(m[1]) for m in metrics)

    hdr = f"  {'Metric':<{label_w}}"
    for rnd in rounds:
        hdr += f"  {rnd:>{col_w}}"
    print(hdr)
    print(f"  {'-'*label_w}" + f"  {'-'*col_w}" * len(rounds))

    for key, label in metrics:
        line = f"  {label:<{label_w}}"
        for rnd in rounds:
            cbf_eps = [
                ep for ep in all_episodes.get(rnd, [])
                if "cbf" in ep.get("_source_file", "")
            ]
            val = _mean_col(cbf_eps, key)
            line += f"  {_fmt(val, '.4f'):>{col_w}}"
        print(line)


def print_npz_stats(rounds: list[str], npz_dirs: list[Path],
                    suite_filter: Optional[str] = None):
    """Trajectory-level stats from saved .npz files."""
    _section("TRAJECTORY ANALYSIS (from .npz files)")

    col_w = max(12, max(len(r) for r in rounds) + 1)
    label_w = 35

    keys = [
        ("cbf_rate_mean",    "CBF step fraction ↓"),
        ("h_min_mean",       "Mean h at CBF steps ↑"),
        ("correction_mean",  "Mean correction norm ↓"),
        ("path_length_mean", "Mean path length m ↓"),
        ("displacement_mean","Mean displacement m"),
        ("n_episodes",       "N episodes loaded"),
    ]

    hdr = f"  {'Metric':<{label_w}}"
    for rnd in rounds:
        hdr += f"  {rnd:>{col_w}}"
    print(hdr)
    print(f"  {'-'*label_w}" + f"  {'-'*col_w}" * len(rounds))

    for key, label in keys:
        line = f"  {label:<{label_w}}"
        for rnd, npz_dir in zip(rounds, npz_dirs):
            stats = load_npz_stats(npz_dir, suite_filter)
            val = stats.get(key)
            if key == "n_episodes":
                line += f"  {int(val) if val is not None else 0:>{col_w}d}"
            else:
                line += f"  {_fmt(val, '.4f'):>{col_w}}"
        print(line)


def make_plots(rounds: list[str], all_agg: dict[str, list[dict]],
               all_episodes: dict[str, list[dict]], plot_dir: Path):
    """Generate matplotlib figures for the DAgger progression."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not available — skipping plots)")
        return

    plot_dir.mkdir(parents=True, exist_ok=True)

    # 1. Key metrics across rounds (CBF mode averages)
    agg_metrics = [
        ("car_pct",               "CAR % ↑"),
        ("tsr_pct",               "TSR % ↑"),
        ("collision_rate_pct",    "Collision Rate % ↓"),
        ("mean_cbf_activation_rate", "CBF Activation Rate ↓"),
        ("mean_deadlock_steps",   "Mean Deadlock Steps ↓"),
        ("mean_path_efficiency",  "Path Efficiency ↑"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("DAgger Round Progression — SafeLIBERO CBF Mode", fontsize=13)

    for ax, (metric, ylabel) in zip(axes.flat, agg_metrics):
        vals = []
        for rnd in rounds:
            cbf_rows = [r for r in all_agg.get(rnd, []) if r.get("mode") == "cbf"]
            v = _mean_col(cbf_rows, metric)
            vals.append(v)

        x = range(len(rounds))
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if valid:
            xi, yi = zip(*valid)
            ax.plot(xi, yi, "o-", linewidth=2, markersize=7)
            ax.set_xticks(list(range(len(rounds))))
            ax.set_xticklabels(rounds, rotation=20, ha="right", fontsize=8)
        ax.set_title(ylabel, fontsize=9)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = plot_dir / "dagger_agg_metrics.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")

    # 2. Failure mode stacked bars
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {
        "SUCCESS":              "#2ecc71",
        "GRASP_THEN_FAIL":      "#f39c12",
        "GRASP_NEVER":          "#e74c3c",
        "COLLISION_APPROACH":   "#c0392b",
        "COLLISION_POST_GRASP": "#e91e63",
        "DEADLOCK":             "#9b59b6",
    }

    bottoms = np.zeros(len(rounds))
    for mode in FAILURE_MODES:
        heights = []
        for rnd in rounds:
            eps = all_episodes.get(rnd, [])
            total = len(eps) or 1
            n = sum(1 for ep in eps if classify_episode(ep) == mode)
            heights.append(100.0 * n / total)
        ax.bar(rounds, heights, bottom=bottoms, label=mode,
               color=colors.get(mode, "#888"))
        bottoms += np.array(heights)

    ax.set_ylabel("Episode %")
    ax.set_title("Failure Mode Breakdown Across DAgger Rounds")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, 105)
    plt.tight_layout()
    out = plot_dir / "dagger_failure_modes.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"  Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="DAgger round-over-round analysis for SafeLIBERO benchmark"
    )
    ap.add_argument("--rounds", nargs="+", required=True,
                    help="Round labels (e.g. base_pi05 dagger_r0 dagger_r1)")
    ap.add_argument("--dirs", nargs="+", required=True,
                    help="Results directories, one per round (same order as --rounds)")
    ap.add_argument("--npz-dirs", nargs="+", default=None,
                    help="Directories containing .npz files for trajectory analysis")
    ap.add_argument("--suite", default=None,
                    help="Filter to a specific suite (e.g. safelibero_spatial)")
    ap.add_argument("--plot", action="store_true",
                    help="Generate matplotlib figures")
    ap.add_argument("--plot-dir", default="figures/dagger_analysis",
                    help="Where to save figures (default: figures/dagger_analysis)")
    ap.add_argument("--deadlock-threshold", type=int, default=_DEADLOCK_THRESHOLD[0],
                    help=f"Deadlock_steps > N → DEADLOCK failure (default {_DEADLOCK_THRESHOLD[0]})")
    args = ap.parse_args()

    _DEADLOCK_THRESHOLD[0] = args.deadlock_threshold

    if len(args.rounds) != len(args.dirs):
        ap.error("--rounds and --dirs must have the same number of entries")

    rounds = args.rounds
    dirs   = [Path(d) for d in args.dirs]

    # Load data for all rounds
    all_agg: dict[str, list[dict]] = {}
    all_episodes: dict[str, list[dict]] = {}
    for rnd, d in zip(rounds, dirs):
        all_agg[rnd]      = load_agg(d, suite_filter=args.suite)
        all_episodes[rnd] = load_episodes(d, suite_filter=args.suite)
        n_agg = len(all_agg[rnd])
        n_ep  = len(all_episodes[rnd])
        print(f"  Loaded {rnd:25s}  {n_agg} agg rows  {n_ep} episode rows  from {d}")

    # Tables
    print_cross_round_agg(rounds, all_agg, suite_filter=args.suite)
    print_failure_breakdown(rounds, all_episodes)
    print_behavioral_shift(rounds, all_episodes)

    # Optional: trajectory analysis from npz
    if args.npz_dirs:
        if len(args.npz_dirs) != len(rounds):
            print("WARNING: --npz-dirs count differs from --rounds count; skipping npz analysis")
        else:
            print_npz_stats(rounds, [Path(d) for d in args.npz_dirs], suite_filter=args.suite)

    # Optional: plots
    if args.plot:
        _section("GENERATING PLOTS")
        make_plots(rounds, all_agg, all_episodes, Path(args.plot_dir))

    print(f"\n{'='*78}\n  Done.\n{'='*78}\n")


if __name__ == "__main__":
    main()
