"""
Phase-1 dataset collection for CBF-VLA safety benchmark.

Runs episodes in CBF mode only and saves per-episode .npz files containing:
    images      (T, 224, 224, 3) uint8   — static_cam frames (full fidelity)
    q           (T, 7)           float32 — joint positions
    u_nom       (T, 7)           float32 — nominal IK joint velocity (pre-CBF)
    u_safe      (T, 7)           float32 — CBF-filtered velocity
    h_values    (T, n_obs)       float32 — min CBF barrier value per obstacle
    vla_delta   (T, 7)           float32 — raw VLA action [dx,dy,dz,dr,dp,dy,g]
    ee_pos      (T, 3)           float32 — end-effector xyz
    cbf_triggered (T,)           bool    — CBF was active this step
    violation   (T,)             bool    — safety radius breached
    min_dist    (T,)             float32 — closest arm-to-obstacle distance

Each episode is saved as:
    <out_dir>/<scene_name>/ep_{N:04d}.npz

Usage:
    mjpython collect_dataset.py --scene bench_level_i --episodes 50
    mjpython collect_dataset.py --scene bench_level_ii --episodes 50 --out dataset/
    mjpython collect_dataset.py --all --episodes 50
"""

from __future__ import annotations

import argparse
import numpy as np
from pathlib import Path

from experiments.scene_config import BENCHMARK_SCENES, ALL_SCENES
from experiments.runner import run_trial, sample_scene


def collect_scene(scene_name: str, n_episodes: int, out_dir: Path,
                  show_every: int = 0):
    cfg      = ALL_SCENES[scene_name]
    save_dir = out_dir / scene_name
    save_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*64}")
    print(f"  Collecting: {scene_name}  ({n_episodes} episodes, CBF mode)")
    print(f"  Output:     {save_dir}/")
    print(f"{'='*64}")

    total_interventions = 0
    total_successes     = 0

    for ep in range(n_episodes):
        show    = show_every > 0 and ep % show_every == 0
        ep_tag  = "[viewer] " if show else "         "
        ep_path = save_dir / f"ep_{ep:04d}.npz"

        print(f"  ep {ep+1:3d}/{n_episodes} {ep_tag}", end="", flush=True)

        ep_cfg  = sample_scene(cfg)
        metrics = run_trial(
            ep_cfg,
            use_cbf=True,
            headless=not show,
            save_results=False,
            collect_dataset=True,
            dataset_path=str(ep_path),
        )

        s = metrics.summary()
        n_cbf = s.get("cbf_activations", 0)
        total_interventions += n_cbf
        total_successes     += int(s.get("goal_reached", False))
        print(f"CBF_acts={n_cbf:3d}  TSR={int(s['goal_reached'])}  "
              f"min_dist={s['min_dist_overall']:.3f}m  → {ep_path.name}")

    print(f"\n  ── {scene_name} collection summary ──")
    print(f"     Episodes   : {n_episodes}")
    print(f"     Successes  : {total_successes} / {n_episodes}  "
          f"({100*total_successes/n_episodes:.1f}%)")
    print(f"     CBF pairs  : {total_interventions} total intervention steps")
    print(f"     Avg/episode: {total_interventions/n_episodes:.1f}")

    # Quick sanity check on one saved file
    sample = np.load(save_dir / "ep_0000.npz")
    print(f"\n  .npz keys & shapes (ep_0000):")
    for k, v in sample.items():
        print(f"     {k:<16} {str(v.shape):<22} {v.dtype}")

    return total_interventions


def main():
    parser = argparse.ArgumentParser(
        description="Phase-1 CBF dataset collection (saves per-episode .npz files)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scene", help="Scene name (see --list)")
    grp.add_argument("--all",   action="store_true",
                     help="Collect from all benchmark scenes")
    grp.add_argument("--list",  action="store_true",
                     help="List available scenes and exit")
    parser.add_argument("--episodes",   type=int, default=50)
    parser.add_argument("--out",        default="dataset",
                        help="Root output directory (default: dataset/)")
    parser.add_argument("--show-every", type=int, default=0,
                        help="Open viewer every Nth episode (0=headless only)")
    args = parser.parse_args()

    if args.list:
        print("Available scenes:")
        for name, cfg in ALL_SCENES.items():
            tag = " [benchmark]" if name in BENCHMARK_SCENES else ""
            print(f"  {name:<24} {cfg.description}{tag}")
        return

    if args.scene and args.scene not in ALL_SCENES:
        print(f"Unknown scene '{args.scene}'. Use --list.")
        return

    out_dir = Path(args.out)
    scenes  = list(BENCHMARK_SCENES.keys()) if args.all else [args.scene]

    grand_total = 0
    for scene_name in scenes:
        grand_total += collect_scene(
            scene_name, args.episodes, out_dir,
            show_every=args.show_every,
        )

    print(f"\n{'='*64}")
    print(f"  Collection complete.")
    print(f"  Total CBF intervention steps: {grand_total}")
    print(f"  Dataset root: {out_dir.resolve()}/")
    if grand_total < 500:
        print(f"  WARNING: only {grand_total} intervention steps — "
              f"run more episodes for Phase 2 (target ≥500 per scene).")
    print(f"{'='*64}\n")


if __name__ == "__main__":
    main()
