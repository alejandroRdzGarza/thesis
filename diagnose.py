"""
Trajectory quality diagnostic for the VLA-CBF pipeline.

Runs a small number of episodes (both plain and CBF), then generates five
diagnostic figures and a health table that flags known failure modes.

Figures saved to <out_dir>/<scene_name>/:
    01_trajectory_xy.png   — top-down EE path per episode, plain vs CBF
    02_h_min_time.png      — CBF barrier h_min(t); h<0 = violation
    03_cbf_correction.png  — CBF correction norm per step (CBF mode)
    04_vla_delta.png       — ||VLA xyz delta|| per step (server contributing?)
    05_goal_dist.png       — ||EE - goal|| convergence per episode
    06_ghost_tracking.png  — ||ghost - EE|| tracking error (IK quality)

Usage:
    mjpython diagnose.py --scene bench_level_ii
    mjpython diagnose.py --scene bench_level_i  --episodes 5 --out diagnostics/
    mjpython diagnose.py --scene bench_level_ii --show          # open viewer
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from experiments.scene_config import ALL_SCENES, SceneConfig
from experiments.runner import run_trial, sample_scene

# ── Plot style ───────────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif", "font.size": 10,
    "axes.titlesize": 11, "axes.labelsize": 10,
    "figure.dpi": 130,
})
C_PLAIN = "#4878CF"
C_CBF   = "#D65F5F"
C_OBS   = "#E8A020"


# ── Data container for one episode ───────────────────────────────────────────
class EpisodeData:
    def __init__(self, mode: str, ep_idx: int, cfg: SceneConfig):
        self.mode    = mode
        self.ep_idx  = ep_idx
        self.cfg     = cfg
        self.records = []
        self.summary = {}

    # Convenience array extractors
    def ee_xy(self):         return np.array([[r.ee_pos[0], r.ee_pos[1]] for r in self.records])
    def h_min(self):         return np.array([min(r.h_values) for r in self.records])
    def cbf_norm(self):      return np.array([r.cbf_correction_norm for r in self.records])
    def vla_mag(self):       return np.array([float(np.linalg.norm(r.vla_delta[:3])) for r in self.records])
    def goal_dist(self):     return np.array([float(np.linalg.norm(r.ee_pos - self.cfg.goal_pos)) for r in self.records])
    def ghost_err(self):
        if self.records[0].ghost_pos is None:
            return None
        return np.array([float(np.linalg.norm(r.ee_pos - r.ghost_pos)) for r in self.records])


# ── Runner ───────────────────────────────────────────────────────────────────
def collect_episodes(scene_name: str, n_episodes: int,
                     headless: bool) -> tuple[list[EpisodeData], list[EpisodeData]]:
    cfg = ALL_SCENES[scene_name]
    plain_eps, cbf_eps = [], []

    for ep in range(n_episodes):
        for mode, use_cbf, store in [("plain", False, plain_eps),
                                     ("cbf",   True,  cbf_eps)]:
            print(f"  [{mode:5s}] ep {ep+1}/{n_episodes} ... ", end="", flush=True)
            ep_cfg  = sample_scene(cfg)
            metrics = run_trial(ep_cfg, use_cbf=use_cbf,
                                headless=headless, save_results=False)
            ed = EpisodeData(mode, ep, ep_cfg)
            ed.records = metrics.get_records()
            ed.summary = metrics.summary()
            store.append(ed)
            s = ed.summary
            print(f"TSR={int(s['goal_reached'])}  "
                  f"CAR={1-int(s['violation_steps']>0)}  "
                  f"CBF_acts={s['cbf_activations']:3d}  "
                  f"min_dist={s['min_dist_overall']:.3f}m")

    return plain_eps, cbf_eps


# ── Figure 1 — XY Trajectory ─────────────────────────────────────────────────
def fig_trajectory_xy(plain_eps: list[EpisodeData], cbf_eps: list[EpisodeData],
                      cfg: SceneConfig, out: Path):
    n  = len(plain_eps)
    nc = min(n, 3)
    nr = (n + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(5*nc, 4.5*nr))
    axes = np.array(axes).flatten() if n > 1 else [axes]
    fig.suptitle(f"EE Trajectory — {cfg.name} (top-down XY)", fontweight="bold")

    for i, (pe, ce) in enumerate(zip(plain_eps, cbf_eps)):
        ax = axes[i]
        pxy = pe.ee_xy()
        cxy = ce.ee_xy()
        ax.plot(pxy[:,1], pxy[:,0], color=C_PLAIN, lw=1.2, alpha=0.8,
                label=f"Plain (TSR={int(pe.summary['goal_reached'])})")
        ax.plot(cxy[:,1], cxy[:,0], color=C_CBF,   lw=1.2, alpha=0.8,
                label=f"CBF   (TSR={int(ce.summary['goal_reached'])})")

        # Start / goal
        sx, sy = cfg.start_pos[0], cfg.start_pos[1]
        gx, gy = cfg.goal_pos[0],  cfg.goal_pos[1]
        ax.plot(sy, sx, "rs", ms=8, zorder=5, label="Start")
        ax.plot(gy, gx, "g^", ms=8, zorder=5, label="Goal")

        # Obstacles (use episode-specific perturbed positions)
        for obs in ce.cfg.obstacles:
            c = plt.Circle((obs.pos[1], obs.pos[0]), obs.radius,
                           color=C_OBS, alpha=0.4, zorder=4)
            ax.add_patch(c)
            c2 = plt.Circle((obs.pos[1], obs.pos[0]), obs.safety_radius,
                            color=C_OBS, alpha=0.10, zorder=3,
                            linestyle="--", fill=False, edgecolor=C_OBS, lw=1)
            ax.add_patch(c2)

        ax.set_title(f"Episode {i+1}")
        ax.set_xlabel("Y (m)")
        ax.set_ylabel("X (m)")
        ax.legend(fontsize=8)
        ax.set_aspect("equal")
        ax.grid(True, ls="--", alpha=0.3)

    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    fig.tight_layout()
    p = out / "01_trajectory_xy.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Figure 2 — h_min(t) ──────────────────────────────────────────────────────
def fig_h_min(plain_eps: list[EpisodeData], cbf_eps: list[EpisodeData],
              cfg: SceneConfig, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"CBF Barrier h_min(t)  [h<0 = violation] — {cfg.name}",
                 fontweight="bold")

    for i, (pe, ce) in enumerate(zip(plain_eps, cbf_eps)):
        alpha = 0.6 if len(plain_eps) > 1 else 0.9
        ax.plot(pe.h_min(), color=C_PLAIN, lw=1.0, alpha=alpha,
                label="Plain" if i == 0 else "_")
        ax.plot(ce.h_min(), color=C_CBF,   lw=1.0, alpha=alpha,
                label="CBF" if i == 0 else "_")

    ax.axhline(0, color="red", lw=1.5, ls="--", label="h=0 (safety boundary)")
    ax.set_xlabel("Step")
    ax.set_ylabel("h_min (m²)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    p = out / "02_h_min_time.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Figure 3 — CBF correction norm ───────────────────────────────────────────
def fig_cbf_correction(cbf_eps: list[EpisodeData], cfg: SceneConfig, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"CBF Correction Norm ||u_safe - u_nom|| — {cfg.name}",
                 fontweight="bold")

    for i, ce in enumerate(cbf_eps):
        norm  = ce.cbf_norm()
        steps = np.arange(len(norm))
        fired = norm > 1e-4
        ax.scatter(steps[~fired], norm[~fired], s=2, color="lightgray", alpha=0.5,
                   label="Inactive" if i == 0 else "_")
        ax.scatter(steps[fired],  norm[fired],  s=4, color=C_CBF, alpha=0.8,
                   label="Active" if i == 0 else "_")

    act_rate = np.mean([ce.summary["cbf_activation_rate"] for ce in cbf_eps])
    ax.set_title(f"Mean CBF activation rate: {act_rate:.1%}  "
                 f"(target 15–60%)", fontsize=10)
    ax.set_xlabel("Step")
    ax.set_ylabel("||u_safe − u_nom||")
    ax.legend(markerscale=3)
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    p = out / "03_cbf_correction.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Figure 4 — VLA delta magnitude ───────────────────────────────────────────
def fig_vla_delta(plain_eps: list[EpisodeData], cbf_eps: list[EpisodeData],
                  cfg: SceneConfig, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"VLA xyz delta magnitude — {cfg.name}  "
                 f"(0 = VLA server silent or not contributing)", fontweight="bold")

    for i, (pe, ce) in enumerate(zip(plain_eps, cbf_eps)):
        alpha = 0.55 if len(plain_eps) > 1 else 0.85
        ax.plot(pe.vla_mag(), color=C_PLAIN, lw=0.8, alpha=alpha,
                label="Plain" if i == 0 else "_")
        ax.plot(ce.vla_mag(), color=C_CBF,   lw=0.8, alpha=alpha,
                label="CBF"   if i == 0 else "_")

    ax.axhline(0.001, color="orange", lw=1.2, ls="--",
               label="0.001 m/step (min expected VLA contribution)")
    ax.set_xlabel("Step")
    ax.set_ylabel("||VLA delta xyz|| (m)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    p = out / "04_vla_delta.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Figure 5 — Distance to goal ───────────────────────────────────────────────
def fig_goal_dist(plain_eps: list[EpisodeData], cbf_eps: list[EpisodeData],
                  cfg: SceneConfig, out: Path):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"||EE – goal|| convergence — {cfg.name}", fontweight="bold")

    for i, (pe, ce) in enumerate(zip(plain_eps, cbf_eps)):
        alpha = 0.55 if len(plain_eps) > 1 else 0.85
        ax.plot(pe.goal_dist(), color=C_PLAIN, lw=1.0, alpha=alpha,
                label="Plain" if i == 0 else "_")
        ax.plot(ce.goal_dist(), color=C_CBF,   lw=1.0, alpha=alpha,
                label="CBF"   if i == 0 else "_")

    ax.axhline(cfg.goal_tolerance, color="green", lw=1.5, ls="--",
               label=f"goal_tolerance ({cfg.goal_tolerance} m)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Distance to goal (m)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    p = out / "05_goal_dist.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Figure 6 — Ghost-target tracking error ────────────────────────────────────
def fig_ghost_tracking(cbf_eps: list[EpisodeData], cfg: SceneConfig, out: Path):
    if cbf_eps[0].ghost_err() is None:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(f"IK tracking error ||ghost – EE|| — {cfg.name}  "
                 f"(high = IK failing to follow ghost target)", fontweight="bold")

    for i, ce in enumerate(cbf_eps):
        err = ce.ghost_err()
        ax.plot(err, color=C_CBF, lw=0.9, alpha=0.6,
                label="CBF ep" if i == 0 else "_")

    ax.axhline(0.05, color="orange", lw=1.2, ls="--",
               label="50 mm (acceptable tracking error)")
    ax.set_xlabel("Step")
    ax.set_ylabel("||ghost – EE|| (m)")
    ax.legend()
    ax.grid(True, ls="--", alpha=0.3)

    fig.tight_layout()
    p = out / "06_ghost_tracking.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved {p.name}")


# ── Health table ──────────────────────────────────────────────────────────────
def _flag(ok: bool, warn: bool = False) -> str:
    if ok:   return "PASS ✓"
    if warn: return "WARN ?"
    return          "FAIL ✗"


def print_health_table(plain_eps: list[EpisodeData], cbf_eps: list[EpisodeData],
                       cfg: SceneConfig):
    tsr_plain = np.mean([e.summary["goal_reached"]        for e in plain_eps])
    tsr_cbf   = np.mean([e.summary["goal_reached"]        for e in cbf_eps])
    car_plain = np.mean([e.summary["violation_steps"] == 0 for e in plain_eps])
    car_cbf   = np.mean([e.summary["violation_steps"] == 0 for e in cbf_eps])
    act_rate  = np.mean([e.summary["cbf_activation_rate"] for e in cbf_eps])
    vla_mean  = np.mean([np.mean(e.vla_mag())             for e in cbf_eps])
    h_plain_min = min(min(e.h_min()) for e in plain_eps)
    h_cbf_min   = min(min(e.h_min()) for e in cbf_eps)
    ghost_err_mean = (np.mean([np.mean(e.ghost_err()) for e in cbf_eps])
                      if cbf_eps[0].ghost_err() is not None else None)

    rows = [
        ("TSR — CBF mode",        f"{tsr_cbf:.0%}",
         _flag(tsr_cbf >= 0.30, tsr_cbf >= 0.10),
         "≥30% expected; 0% = arm stuck or geometry broken"),

        ("TSR — plain mode",      f"{tsr_plain:.0%}",
         "INFO",
         "lower than CBF is expected"),

        ("CAR — CBF mode",        f"{car_cbf:.0%}",
         _flag(car_cbf >= 0.70, car_cbf >= 0.40),
         "≥70% expected; low = CBF QP not solving"),

        ("CAR — plain mode",      f"{car_plain:.0%}",
         _flag(car_plain < 1.00, car_plain < 0.50),  # want some violations in plain
         "< 100% confirms scene is genuinely challenging"),

        ("CBF activation rate",   f"{act_rate:.1%}",
         _flag(0.05 <= act_rate <= 0.80, act_rate < 0.05),
         "5–80% normal; 0% = CBF never fires; 100% = degenerate"),

        ("VLA delta mean |xyz|",  f"{vla_mean:.4f} m",
         _flag(vla_mean > 0.001, vla_mean > 0.0001),
         ">0.001 m = VLA actively contributing; 0 = server silent"),

        ("h_min ever <0 (plain)", f"{h_plain_min:.4f}",
         _flag(h_plain_min < 0),
         "negative in plain = scene is genuinely unsafe without CBF"),

        ("h_min ever <0 (CBF)",   f"{h_cbf_min:.4f}",
         _flag(h_cbf_min >= 0, h_cbf_min >= -0.005),
         "should stay ≥0; if negative CBF QP is failing"),
    ]

    if ghost_err_mean is not None:
        rows.append(
            ("IK tracking error mean", f"{ghost_err_mean:.3f} m",
             _flag(ghost_err_mean < 0.05, ghost_err_mean < 0.10),
             "<50 mm good; high = IK can't follow ghost target"),
        )

    w = [36, 12, 10, 50]
    sep = "─" * (sum(w) + len(w)*2)
    hdr = (f"  {'Indicator':<{w[0]}}  {'Value':>{w[1]}}  "
           f"{'Status':<{w[2]}}  {'Note'}")
    print(f"\n{'='*len(sep)}")
    print(f"  HEALTH REPORT — {cfg.name}  ({len(plain_eps)} episodes each mode)")
    print(f"{'='*len(sep)}")
    print(hdr)
    print(f"  {sep}")
    for label, val, status, note in rows:
        print(f"  {label:<{w[0]}}  {val:>{w[1]}}  {status:<{w[2]}}  {note}")
    print(f"  {sep}\n")

    # Explicit failure mode hints
    hints = []
    if tsr_cbf < 0.10:
        hints.append("STUCK: TSR≈0% in CBF mode — check goal_attract, start/goal positions, or VLA server")
    if act_rate < 0.03:
        hints.append("CBF SILENT: activation rate <3% — obstacle may be too far; check obstacle position and safety_radius")
    if act_rate > 0.90:
        hints.append("CBF ALWAYS ON: >90% activation — CBF radius too large or obstacle on arm's only path; arm may be stuck")
    if car_cbf < 0.40:
        hints.append("CBF NOT WORKING: CAR<40% in CBF mode — QP may be failing (check SLSQP convergence)")
    if vla_mean < 0.0001:
        hints.append("VLA SILENT: delta≈0 — VLA server not responding or network issue; check mjpython server on port 8000")
    if h_plain_min > 0.05:
        hints.append("SCENE TOO EASY: h_min never goes negative in plain mode — obstacle may not be on the arm's path")
    if ghost_err_mean is not None and ghost_err_mean > 0.10:
        hints.append("IK LAGGING: ghost-EE error >10 cm — IK step size too small or N_WARMUP insufficient")

    if hints:
        print("  Detected issues:")
        for h in hints:
            print(f"    ! {h}")
        print()
    else:
        print("  No obvious failure modes detected — pipeline looks healthy.\n")


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Trajectory quality diagnostic for VLA-CBF pipeline")
    parser.add_argument("--scene",    required=True,
                        help=f"Scene name. Options: {list(ALL_SCENES)}")
    parser.add_argument("--episodes", type=int, default=3,
                        help="Episodes per mode (default 3 — runs 6 total)")
    parser.add_argument("--out",      default="diagnostics",
                        help="Output directory for figures (default: diagnostics/)")
    parser.add_argument("--show",     action="store_true",
                        help="Open MuJoCo viewer for all episodes (slow but visual)")
    args = parser.parse_args()

    if args.scene not in ALL_SCENES:
        print(f"Unknown scene '{args.scene}'. Available: {list(ALL_SCENES)}")
        return

    cfg     = ALL_SCENES[args.scene]
    out_dir = Path(args.out) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Diagnosing: {args.scene}  ({args.episodes} ep × 2 modes = "
          f"{args.episodes*2} total)")
    print(f"  Output:     {out_dir}/")
    print(f"{'='*64}\n")

    plain_eps, cbf_eps = collect_episodes(
        args.scene, args.episodes, headless=not args.show
    )

    print(f"\n  Generating figures...")
    fig_trajectory_xy (plain_eps, cbf_eps, cfg, out_dir)
    fig_h_min         (plain_eps, cbf_eps, cfg, out_dir)
    fig_cbf_correction(cbf_eps,            cfg, out_dir)
    fig_vla_delta     (plain_eps, cbf_eps, cfg, out_dir)
    fig_goal_dist     (plain_eps, cbf_eps, cfg, out_dir)
    fig_ghost_tracking(cbf_eps,            cfg, out_dir)

    print_health_table(plain_eps, cbf_eps, cfg)
    print(f"  All figures → {out_dir.resolve()}/\n")


if __name__ == "__main__":
    main()
