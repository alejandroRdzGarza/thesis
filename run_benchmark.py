"""
AEGIS-style multi-episode benchmark for OpenVLA + CBF safety evaluation.

Matches the SafeLIBERO evaluation protocol from the AEGIS paper:
  - N_EPISODES (default 50) per (scene, mode) pair
  - Obstacle positions randomized each episode within pos_noise_range
  - Reports CAR / TSR / ETS — the three metrics from AEGIS Table 1

Metrics:
  CAR  Collision Avoidance Rate  — % of episodes with zero safety violations
  TSR  Task Success Rate         — % of episodes where goal is reached
  ETS  Execution Time Steps      — mean steps per episode (goal_reach_step if
                                   success, else max_steps)

Usage:
    mjpython run_benchmark.py --scene bench_level_i
    mjpython run_benchmark.py --scene bench_level_ii --mode cbf --episodes 10
    mjpython run_benchmark.py --all --episodes 50
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

from experiments.scene_config import ALL_SCENES, BENCHMARK_SCENES
from experiments.runner import run_trial, sample_scene

DEFAULT_EPISODES = 50


# ---------------------------------------------------------------------------
# Per-episode extraction
# ---------------------------------------------------------------------------
def _episode_record(ep_idx: int, summary: dict, max_steps: int) -> dict:
    """Convert a single-trial summary into AEGIS-compatible per-episode metrics."""
    goal_step = summary["goal_reach_step"]
    return {
        "episode":  ep_idx,
        "car":      1 if summary["violation_steps"] == 0 else 0,
        "tsr":      1 if summary["goal_reached"] else 0,
        "ets":      goal_step if summary["goal_reached"] else max_steps,
        "min_dist": summary["min_dist_overall"],
        "path_len": summary["path_length_m"],
    }


# ---------------------------------------------------------------------------
# Scene × mode runner
# ---------------------------------------------------------------------------
def run_benchmark_scene(scene_name: str, use_cbf: bool,
                        n_episodes: int, results_dir: Path,
                        show_every: int = 0) -> dict:
    cfg   = ALL_SCENES[scene_name]
    mode  = "cbf" if use_cbf else "plain"
    label = f"{scene_name}_{mode}"

    print(f"\n{'='*64}")
    vis_note = f", viewer every {show_every} ep" if show_every > 0 else ", headless"
    print(f"  Benchmark: {label}   ({n_episodes} episodes{vis_note})")
    print(f"  {cfg.description}")
    print(f"{'='*64}")

    records = []
    for ep in range(n_episodes):
        show    = show_every > 0 and ep % show_every == 0
        ep_tag  = "[viewer] " if show else "         "
        print(f"  ep {ep+1:3d}/{n_episodes} {ep_tag}", end="", flush=True)
        ep_cfg  = sample_scene(cfg)          # randomise obstacle positions
        metrics = run_trial(ep_cfg, use_cbf=use_cbf,
                            results_dir=str(results_dir),
                            headless=not show,
                            save_results=False)
        rec = _episode_record(ep, metrics.summary(), cfg.max_steps)
        records.append(rec)
        print(f"CAR={rec['car']}  TSR={rec['tsr']}  "
              f"ETS={rec['ets']:3d}  min_dist={rec['min_dist']:.3f} m")

    car = np.mean([r["car"] for r in records]) * 100
    tsr = np.mean([r["tsr"] for r in records]) * 100
    ets = np.mean([r["ets"] for r in records])

    print(f"\n  ── {label} aggregate ──")
    print(f"     CAR: {car:.1f}%   TSR: {tsr:.1f}%   ETS: {ets:.1f} steps")

    # Per-episode CSV
    ep_path = results_dir / f"bench_{label}_episodes.csv"
    with open(ep_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    # Aggregate CSV
    agg_path = results_dir / f"bench_{label}_agg.csv"
    with open(agg_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["scene", "mode", "n_episodes",
                           "car_pct", "tsr_pct", "ets"])
        writer.writeheader()
        writer.writerow({
            "scene": scene_name, "mode": mode,
            "n_episodes": n_episodes,
            "car_pct": round(car, 2),
            "tsr_pct": round(tsr, 2),
            "ets":     round(ets, 1),
        })

    print(f"  Saved {ep_path.name}  {agg_path.name}")
    return {"scene": scene_name, "mode": mode, "car": car, "tsr": tsr, "ets": ets}


# ---------------------------------------------------------------------------
# Final AEGIS-style table
# ---------------------------------------------------------------------------
def print_aegis_table(results: list[dict]):
    scene_order = list(dict.fromkeys(r["scene"] for r in results))
    width = 66
    print("\n" + "=" * width)
    print(f"  {'Method / Scene':<30}  {'CAR ↑':>9}  {'TSR ↑':>9}  {'ETS ↓':>9}")
    print("=" * width)
    for scene in scene_order:
        for mode in ("plain", "cbf"):
            match = [r for r in results if r["scene"] == scene and r["mode"] == mode]
            if not match:
                continue
            r = match[0]
            label = f"{scene} [{mode}]"
            print(f"  {label:<30}  {r['car']:>8.1f}%  {r['tsr']:>8.1f}%  {r['ets']:>9.1f}")
        print("-" * width)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="AEGIS-style multi-episode benchmark (CAR / TSR / ETS)")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--scene",   help="Scene name (see --list)")
    grp.add_argument("--all",     action="store_true",
                     help="Run all benchmark scenes (bench_level_i + bench_level_ii)")
    grp.add_argument("--list",    action="store_true",
                     help="List available scenes and exit")
    parser.add_argument("--mode",     choices=["plain", "cbf", "both"], default="both")
    parser.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--show-every", type=int, default=0,
                        help="Open MuJoCo viewer for every Nth episode (0=never). "
                             "E.g. --show-every 5 shows ep 1, 6, 11, ...")
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    if args.list:
        print("Available scenes:")
        for name, cfg in ALL_SCENES.items():
            tag = " [benchmark]" if name in BENCHMARK_SCENES else ""
            print(f"  {name:<24} {cfg.description}{tag}")
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    scenes = list(BENCHMARK_SCENES.keys()) if args.all else [args.scene]
    modes  = (["plain", "cbf"] if args.mode == "both"
              else [args.mode])

    if args.scene and args.scene not in ALL_SCENES:
        print(f"Unknown scene '{args.scene}'. Use --list to see options.")
        return

    all_results = []
    for scene_name in scenes:
        for mode in modes:
            r = run_benchmark_scene(
                scene_name, use_cbf=(mode == "cbf"),
                n_episodes=args.episodes,
                results_dir=results_dir,
                show_every=args.show_every,
            )
            all_results.append(r)

    print_aegis_table(all_results)


if __name__ == "__main__":
    main()
