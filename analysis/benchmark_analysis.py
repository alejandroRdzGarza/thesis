#!/usr/bin/env python3
"""
VLA+CBF benchmark analysis — loads result CSVs and generates publication figures.

Usage:
    python analysis/benchmark_analysis.py
    python analysis/benchmark_analysis.py --results-dir results_libero --out-dir figures

Output (in --out-dir):
    fig1_tsr_comparison.{png,pdf}   — per-task TSR, plain vs CBF
    fig2_safety_analysis.{png,pdf}  — collision rate, CAR, CBF activations
    fig3_summary.{png,pdf}          — one-page aggregate overview
    results_summary.csv             — machine-readable aggregate table
    results_report.md               — markdown results report
"""

from __future__ import annotations
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":        "DejaVu Sans",
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         120,
    "savefig.dpi":        300,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.alpha":         0.25,
    "grid.linestyle":     "--",
})

# Colorblind-safe (IBM palette): blue for baseline, orange for CBF
C_PLAIN  = "#648FFF"
C_CBF    = "#FE6100"
C_DANGER = "#DC267F"   # for collision indicators


# ── Data loading ───────────────────────────────────────────────────────────────

def _parse_scene(scene: str) -> dict:
    m = re.match(r"(safelibero|libero)_(\w+)_t(\d+)(?:_L(\w+))?", scene)
    if not m:
        return {"is_safe": False, "sub_suite": "unknown", "task_id": -1, "slevel": "n/a"}
    is_safe = m.group(1) == "safelibero"
    return {
        "is_safe":   is_safe,
        "sub_suite": m.group(2),
        "task_id":   int(m.group(3)),
        "slevel":    m.group(4) or ("I" if is_safe else "n/a"),
    }


def load_agg(results_dir: Path) -> pd.DataFrame:
    """Load all *_agg.csv files into a single DataFrame."""
    rows = []
    for f in sorted(results_dir.glob("*_agg.csv")):
        try:
            rows.append(pd.read_csv(f))
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    if not rows:
        return pd.DataFrame()
    df = pd.concat(rows, ignore_index=True)
    meta = pd.DataFrame([_parse_scene(s) for s in df["scene"]])
    return pd.concat([df, meta], axis=1)


def load_episodes(results_dir: Path) -> pd.DataFrame:
    """Load all *_episodes.csv files. Adds 'scene' and 'mode' columns."""
    rows = []
    for f in sorted(results_dir.glob("*_episodes.csv")):
        m = re.match(r"(.+)_(plain|cbf)_episodes", f.stem)
        if not m:
            continue
        try:
            d = pd.read_csv(f)
            d["scene"] = m.group(1)
            d["mode"]  = m.group(2)
            meta = pd.DataFrame([_parse_scene(m.group(1))] * len(d))
            d = pd.concat([d, meta], axis=1)
            rows.append(d)
        except Exception as e:
            print(f"  [warn] {f.name}: {e}")
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def wilson_ci(k: int, n: int, z: float = 1.96) -> float:
    """Wilson score CI half-width (percentage points). Safe for n=0."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z**2 / n
    half = z / denom * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(half * 100)


# ── Plotting helpers ───────────────────────────────────────────────────────────

def _save(fig: plt.Figure, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        path = out_dir / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight")
        print(f"  → {path}")
    plt.close(fig)


def _grouped_bar(
    ax: plt.Axes,
    tasks: list[int],
    plain_vals: list[float],
    cbf_vals:   list[float],
    plain_errs: list[float],
    cbf_errs:   list[float],
    plain_ns:   list[int],
    cbf_ns:     list[int],
    ylabel: str = "",
    title:  str = "",
    ylim:   tuple = (-14, 112),
) -> None:
    x = np.arange(len(tasks))
    w = 0.38
    ax.bar(x - w/2, plain_vals, w, color=C_PLAIN, label="Plain (VLA)",
           alpha=0.88, zorder=3)
    ax.bar(x + w/2, cbf_vals,   w, color=C_CBF,   label="VLA + CBF",
           alpha=0.88, zorder=3)
    ax.errorbar(x - w/2, plain_vals, yerr=plain_errs,
                fmt="none", color="#222", capsize=3, linewidth=1.2, zorder=4)
    ax.errorbar(x + w/2, cbf_vals, yerr=cbf_errs,
                fmt="none", color="#222", capsize=3, linewidth=1.2, zorder=4)
    for xi, (np_, nc) in enumerate(zip(plain_ns, cbf_ns)):
        ax.text(xi - w/2, ylim[0] + 1, f"n={np_}", ha="center",
                va="bottom", fontsize=7, color="#666")
        ax.text(xi + w/2, ylim[0] + 1, f"n={nc}", ha="center",
                va="bottom", fontsize=7, color="#666")
    ax.set_xticks(x)
    ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_ylim(*ylim)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontweight="bold", pad=8)


def _extract_metric(
    df: pd.DataFrame, task_id: int, mode: str, col: str
) -> tuple[float, float, int]:
    """Return (value_pct, ci_half, n) for a given task+mode+metric column."""
    row = df[(df["task_id"] == task_id) & (df["mode"] == mode)]
    if len(row) == 0:
        return 0.0, 0.0, 0
    val = float(row[col].iloc[0])
    n   = int(row["n_episodes"].iloc[0])
    k   = int(round(val * n / 100))
    return val, wilson_ci(k, n), n


# ── Figure 1: per-task TSR ─────────────────────────────────────────────────────

def fig1_tsr_comparison(agg: pd.DataFrame, out_dir: Path) -> None:
    lib  = agg[~agg["is_safe"] & (agg["sub_suite"] == "spatial")]
    safe = agg[ agg["is_safe"] & (agg["sub_suite"] == "spatial")]

    has_lib  = len(lib)  > 0
    has_safe = len(safe) > 0
    if not has_lib and not has_safe:
        print("  [skip] fig1: no data")
        return

    ncols = int(has_lib) + int(has_safe)
    fig, axes = plt.subplots(1, ncols, figsize=(6.5 * ncols, 4.8))
    if ncols == 1:
        axes = [axes]
    ax_idx = 0

    def _fill(ax: plt.Axes, sub: pd.DataFrame, title: str) -> None:
        tasks = sorted(sub["task_id"].unique())
        pv, cv, pe, ce, pn, cn = [], [], [], [], [], []
        for t in tasks:
            v, e, n = _extract_metric(sub, t, "plain", "tsr_pct")
            pv.append(v); pe.append(e); pn.append(n)
            v, e, n = _extract_metric(sub, t, "cbf",   "tsr_pct")
            cv.append(v); ce.append(e); cn.append(n)
        _grouped_bar(ax, tasks, pv, cv, pe, ce, pn, cn,
                     ylabel="Task Success Rate (%)", title=title)
        ax.legend(loc="upper right")

    if has_lib:
        _fill(axes[ax_idx], lib,  "LIBERO-Spatial  ·  Task Success Rate")
        ax_idx += 1
    if has_safe:
        _fill(axes[ax_idx], safe, "SafeLIBERO-Spatial LI  ·  Task Success Rate")

    fig.suptitle("Per-Task TSR: OpenVLA (Plain) vs OpenVLA + CBF",
                 fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig1_tsr_comparison")


# ── Figure 2: safety analysis ──────────────────────────────────────────────────

def fig2_safety_analysis(agg: pd.DataFrame, ep: pd.DataFrame, out_dir: Path) -> None:
    safe = agg[agg["is_safe"] & (agg["sub_suite"] == "spatial")]
    if len(safe) == 0:
        print("  [skip] fig2: no safelibero results")
        return

    tasks = sorted(safe["task_id"].unique())
    x = np.arange(len(tasks))
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    # ── Panel A: Collision Rate ──────────────────────────────────────────────
    pv, cv, pe, ce, pn, cn = [], [], [], [], [], []
    for t in tasks:
        v, e, n = _extract_metric(safe, t, "plain", "collision_rate_pct")
        pv.append(v); pe.append(e); pn.append(n)
        v, e, n = _extract_metric(safe, t, "cbf",   "collision_rate_pct")
        cv.append(v); ce.append(e); cn.append(n)

    ax = axes[0]
    ax.bar(x - w/2, pv, w, color=C_PLAIN,  label="Plain",     alpha=0.88, zorder=3)
    ax.bar(x + w/2, cv, w, color=C_DANGER, label="VLA + CBF", alpha=0.88, zorder=3)
    ax.errorbar(x - w/2, pv, yerr=pe, fmt="none", color="#222", capsize=3, lw=1.2, zorder=4)
    ax.errorbar(x + w/2, cv, yerr=ce, fmt="none", color="#222", capsize=3, lw=1.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_ylim(0, 112); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("Collision Rate  (lower is better)")
    ax.set_title("Obstacle Collision Rate", fontweight="bold", pad=8)
    ax.legend()

    # ── Panel B: CAR ──────────────────────────────────────────────────────────
    pv, cv, pe, ce, pn, cn = [], [], [], [], [], []
    for t in tasks:
        v, e, n = _extract_metric(safe, t, "plain", "car_pct")
        pv.append(v); pe.append(e); pn.append(n)
        v, e, n = _extract_metric(safe, t, "cbf",   "car_pct")
        cv.append(v); ce.append(e); cn.append(n)

    ax = axes[1]
    ax.bar(x - w/2, pv, w, color=C_PLAIN, label="Plain",     alpha=0.88, zorder=3)
    ax.bar(x + w/2, cv, w, color=C_CBF,   label="VLA + CBF", alpha=0.88, zorder=3)
    ax.errorbar(x - w/2, pv, yerr=pe, fmt="none", color="#222", capsize=3, lw=1.2, zorder=4)
    ax.errorbar(x + w/2, cv, yerr=ce, fmt="none", color="#222", capsize=3, lw=1.2, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in tasks])
    ax.set_ylim(0, 112); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
    ax.set_ylabel("Collision Avoidance Rate  (higher is better)")
    ax.set_title("CAR — Obstacle Avoidance", fontweight="bold", pad=8)
    ax.legend()

    # ── Panel C: CBF activations per task (box plot from episode data) ────────
    ax = axes[2]

    if len(ep) > 0 and "cbf_acts" in ep.columns:
        ep_safe_cbf = ep[ep["is_safe"] & (ep["mode"] == "cbf") &
                         (ep["sub_suite"] == "spatial")]
        data_by_task = []
        task_labels  = []
        for t in tasks:
            task_ep = ep_safe_cbf[ep_safe_cbf["task_id"] == t]["cbf_acts"].dropna()
            if len(task_ep) > 0:
                data_by_task.append(task_ep.values)
                task_labels.append(f"T{t}")

        if data_by_task:
            bp = ax.boxplot(data_by_task, labels=task_labels, patch_artist=True,
                            medianprops=dict(color="white", linewidth=2),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2))
            for patch in bp["boxes"]:
                patch.set_facecolor(C_CBF)
                patch.set_alpha(0.85)
            ax.set_ylabel("CBF activations per episode")
            ax.set_title("CBF Filter Activations\n(VLA + CBF mode)", fontweight="bold", pad=8)
        else:
            ax.text(0.5, 0.5, "Episode data\nnot available",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=11, color="gray")
            ax.set_title("CBF Filter Activations", fontweight="bold", pad=8)
    else:
        # Fall back: mean CBF acts from episode CSV if it has the column
        cbf_acts_mean = []
        for t in tasks:
            row_c = safe[(safe["task_id"] == t) & (safe["mode"] == "cbf")]
            cbf_acts_mean.append(0)   # agg CSV doesn't include this; skip gracefully

        ax.bar(x, cbf_acts_mean, color=C_CBF, alpha=0.88, zorder=3)
        ax.set_xticks(x); ax.set_xticklabels([f"T{t}" for t in tasks])
        ax.set_ylabel("Mean CBF activations / episode")
        ax.set_title("CBF Filter Activations\n(VLA + CBF mode)", fontweight="bold", pad=8)

    fig.suptitle("SafeLIBERO-Spatial Level I  ·  Safety Analysis",
                 fontweight="bold", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, out_dir, "fig2_safety_analysis")


# ── Figure 3: executive summary ────────────────────────────────────────────────

def fig3_summary(agg: pd.DataFrame, out_dir: Path) -> None:
    if len(agg) == 0:
        print("  [skip] fig3: no data")
        return

    def _agg_metric(mask, mode, col):
        sub = agg[mask & (agg["mode"] == mode)]
        if len(sub) == 0:
            return 0.0, 0.0
        total_n   = int(sub["n_episodes"].sum())
        total_k   = int((sub[col] / 100 * sub["n_episodes"]).round().sum())
        val = total_k / total_n * 100 if total_n > 0 else 0.0
        return val, wilson_ci(total_k, total_n)

    lib_mask  = ~agg["is_safe"] & (agg["sub_suite"] == "spatial")
    safe_mask =  agg["is_safe"] & (agg["sub_suite"] == "spatial")

    configs = [
        ("LIBERO\nPlain",    lib_mask,  "plain", C_PLAIN, False),
        ("LIBERO\nCBF",      lib_mask,  "cbf",   C_CBF,   False),
        ("SafeLIBERO\nPlain", safe_mask, "plain", C_PLAIN, True),
        ("SafeLIBERO\nCBF",   safe_mask, "cbf",   C_CBF,   True),
    ]

    labels  = [c[0] for c in configs]
    colors  = [c[3] for c in configs]
    x = np.arange(len(configs))

    tsr_vals, tsr_errs = zip(*[_agg_metric(c[1], c[2], "tsr_pct") for c in configs])
    cr_vals,  cr_errs  = zip(*[_agg_metric(c[1], c[2], "collision_rate_pct") for c in configs])
    car_vals, car_errs = zip(*[_agg_metric(c[1], c[2], "car_pct") for c in configs])

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    def _summary_panel(ax, vals, errs, title, ylabel, note=""):
        bars = ax.bar(x, vals, color=colors, alpha=0.88, zorder=3,
                      edgecolor=["#3263c8", "#c04800", "#3263c8", "#c04800"],
                      linewidth=1.0)
        # Hatch for safelibero entries
        for i, c in enumerate(configs):
            if c[4]:   # is_safe
                bars[i].set_hatch("///")
                bars[i].set_edgecolor("#3263c8" if c[2] == "plain" else "#c04800")
        ax.errorbar(x, vals, yerr=errs, fmt="none", color="#111",
                    capsize=4, linewidth=1.4, zorder=4)
        for i, (v, e) in enumerate(zip(vals, errs)):
            ax.text(i, v + e + 2.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylim(0, 118); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold", pad=8)
        if note:
            ax.text(0.5, -0.18, note, ha="center", transform=ax.transAxes,
                    fontsize=8, color="#666", style="italic")

    _summary_panel(axes[0], tsr_vals, tsr_errs,
                   "Task Success Rate  (TSR)", "TSR  (higher = better)",
                   "primary task completion metric")
    _summary_panel(axes[1], car_vals, car_errs,
                   "Collision Avoidance Rate  (CAR)", "CAR  (higher = better)",
                   "only meaningful for SafeLIBERO scenes")
    _summary_panel(axes[2], cr_vals, cr_errs,
                   "Obstacle Collision Rate", "Collision Rate  (lower = better)",
                   "only meaningful for SafeLIBERO scenes")

    # Legend
    p1 = mpatches.Patch(facecolor=C_PLAIN,  label="Plain — VLA only (baseline)")
    p2 = mpatches.Patch(facecolor=C_CBF,    label="VLA + CBF (ours)")
    p3 = mpatches.Patch(facecolor="#aaa", hatch="///", label="SafeLIBERO (obstacle present)")
    axes[2].legend(handles=[p1, p2, p3], loc="upper right", fontsize=8)

    fig.suptitle("VLA + CBF Safety Benchmark  ·  Aggregate Results",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig3_summary")


# ── Bonus: per-episode scatter ─────────────────────────────────────────────────

def fig4_episode_scatter(ep: pd.DataFrame, out_dir: Path) -> None:
    """Per-episode TSR+ETS scatter, coloured by success."""
    if len(ep) == 0 or "ets" not in ep.columns:
        return

    safe_ep = ep[ep["is_safe"] & (ep["sub_suite"] == "spatial")]
    lib_ep  = ep[~ep["is_safe"] & (ep["sub_suite"] == "spatial")]

    targets = [(lib_ep,  "LIBERO-Spatial"),
               (safe_ep, "SafeLIBERO-Spatial LI")]
    targets = [(d, t) for d, t in targets if len(d) > 0]
    if not targets:
        return

    fig, axes = plt.subplots(1, len(targets), figsize=(6 * len(targets), 4.5))
    if len(targets) == 1:
        axes = [axes]

    for ax, (data, title) in zip(axes, targets):
        for mode, color, marker in [("plain", C_PLAIN, "o"), ("cbf", C_CBF, "^")]:
            sub = data[data["mode"] == mode]
            if len(sub) == 0:
                continue
            success = sub["tsr"] == 1
            ax.scatter(sub[success]["episode"] if "episode" in sub else range(success.sum()),
                       sub[success]["ets"],
                       color=color, marker=marker, alpha=0.75, label=f"{mode} — success", s=50)
            ax.scatter(sub[~success]["episode"] if "episode" in sub else range((~success).sum()),
                       sub[~success]["ets"],
                       color=color, marker=marker, alpha=0.30, label=f"{mode} — fail", s=30)
        ax.set_xlabel("Episode index")
        ax.set_ylabel("Episode length (steps)")
        ax.set_title(title, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Episode Outcomes — ETS vs Episode Index",
                 fontweight="bold", fontsize=12, y=1.02)
    fig.tight_layout()
    _save(fig, out_dir, "fig4_episode_scatter")


# ── Summary table ──────────────────────────────────────────────────────────────

def save_summary(agg: pd.DataFrame, ep: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── CSV ──
    cols = ["scene", "mode", "safety_level", "n_episodes",
            "tsr_pct", "car_pct", "collision_rate_pct", "ets"]
    out_df = agg[[c for c in cols if c in agg.columns]].copy()
    if "is_safe" in agg.columns and "task_id" in agg.columns:
        out_df = out_df.join(agg[["is_safe", "task_id"]])
        out_df = out_df.sort_values(["is_safe", "task_id", "mode"])
        out_df = out_df.drop(columns=["is_safe", "task_id"])
    csv_path = out_dir / "results_summary.csv"
    out_df.to_csv(csv_path, index=False)
    print(f"  → {csv_path}")

    # ── Markdown report ──
    lines = [
        "# VLA + CBF Safety Benchmark — Results Report\n",
        f"Generated from {len(agg)} scene/mode combinations.\n",
    ]

    def _agg_row(sub, mode):
        m = sub[sub["mode"] == mode]
        if len(m) == 0:
            return None
        total_n = int(m["n_episodes"].sum())
        tsr  = (m["tsr_pct"] / 100 * m["n_episodes"]).sum() / total_n * 100
        car  = (m["car_pct"] / 100 * m["n_episodes"]).sum() / total_n * 100
        cr   = (m["collision_rate_pct"] / 100 * m["n_episodes"]).sum() / total_n * 100
        ets  = (m["ets"] * m["n_episodes"]).sum() / total_n
        return total_n, tsr, car, cr, ets

    for is_safe, header in [(False, "LIBERO-Spatial  (no obstacles)"),
                             (True,  "SafeLIBERO-Spatial Level I  (obstacle present)")]:
        mask = agg["is_safe"] == is_safe if "is_safe" in agg.columns else pd.Series([True] * len(agg))
        sub  = agg[mask].sort_values(["task_id", "mode"]) if "task_id" in agg.columns else agg[mask]
        if len(sub) == 0:
            continue
        lines += [
            f"\n## {header}\n",
            "| Task | Mode | N  | TSR ↑ | CAR ↑ | Collision ↓ | ETS ↓ |",
            "|------|------|----|-------|-------|-------------|-------|",
        ]
        for _, row in sub.iterrows():
            lines.append(
                f"| T{int(row.get('task_id', 0)):02d}  | {row['mode']:<5} | "
                f"{int(row['n_episodes']):2d} | "
                f"{row['tsr_pct']:5.1f}% | "
                f"{row['car_pct']:5.1f}% | "
                f"{row['collision_rate_pct']:10.1f}% | "
                f"{row['ets']:5.1f} |"
            )
        for mode in ("plain", "cbf"):
            r = _agg_row(sub, mode)
            if r:
                n, tsr, car, cr, ets = r
                lines.append(
                    f"| **AVG** | **{mode}** | **{n}** | "
                    f"**{tsr:.1f}%** | **{car:.1f}%** | **{cr:.1f}%** | **{ets:.0f}** |"
                )

    # CBF improvement summary
    safe = agg[agg["is_safe"]] if "is_safe" in agg.columns else pd.DataFrame()
    if len(safe) > 0:
        r_plain = _agg_row(safe, "plain")
        r_cbf   = _agg_row(safe, "cbf")
        if r_plain and r_cbf:
            lines += [
                "\n## Key Findings (SafeLIBERO)\n",
                f"- **TSR change** (CBF vs Plain): {r_cbf[1] - r_plain[1]:+.1f} pp",
                f"- **Collision reduction**: {r_plain[3]:.1f}% → {r_cbf[3]:.1f}% "
                f"({r_plain[3] - r_cbf[3]:+.1f} pp)",
                f"- **CAR improvement**: {r_plain[2]:.1f}% → {r_cbf[2]:.1f}% "
                f"({r_cbf[2] - r_plain[2]:+.1f} pp)",
            ]

    md_path = out_dir / "results_report.md"
    md_path.write_text("\n".join(lines) + "\n")
    print(f"  → {md_path}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description="VLA+CBF benchmark analysis")
    ap.add_argument("--results-dir", default="results_libero",
                    help="Directory containing *_agg.csv and *_episodes.csv files")
    ap.add_argument("--out-dir", default="figures",
                    help="Output directory for figures and summary files")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir     = Path(args.out_dir)

    if not results_dir.exists():
        print(f"Error: {results_dir} not found. Run the benchmark first.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading results from {results_dir}/")
    agg = load_agg(results_dir)
    ep  = load_episodes(results_dir)

    if len(agg) == 0:
        print("No *_agg.csv files found. Run the benchmark first.")
        sys.exit(1)

    n_scenes = agg["scene"].nunique() if "scene" in agg.columns else 0
    n_ep_rows = len(ep)
    print(f"  {len(agg)} aggregate rows  ({n_scenes} unique scenes)")
    print(f"  {n_ep_rows} per-episode rows")
    print(f"Generating figures → {out_dir}/")

    fig1_tsr_comparison(agg, out_dir)
    fig2_safety_analysis(agg, ep, out_dir)
    fig3_summary(agg, out_dir)
    fig4_episode_scatter(ep, out_dir)
    save_summary(agg, ep, out_dir)

    print("\nDone. Open figures/ to review results.")


if __name__ == "__main__":
    main()
