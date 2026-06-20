"""
AEGIS-style benchmark using LIBERO scenes.

Each LIBERO task provides:
  - Pre-validated scene geometry (robot, table, objects)
  - Built-in success detection  (env.info["success"])
  - Language instruction        (passed directly to OpenVLA)

You supply:
  - ObstacleConfig list  (CBF safety zones around distractors / fixed posts)
  - goal_pos             (where the object should end up)

Usage:
    # List available LIBERO tasks first
    python run_libero_benchmark.py --list --suite libero_spatial

    # Run one task, both modes, 10 episodes
    python run_libero_benchmark.py --suite libero_spatial --task 0 --episodes 10

    # Run all 10 tasks in a suite
    python run_libero_benchmark.py --suite libero_spatial --all --episodes 10

    # Run without CBF to get baseline
    python run_libero_benchmark.py --suite libero_spatial --task 0 --mode plain

    # Watch 1 in every 5 episodes
    python run_libero_benchmark.py --suite libero_spatial --task 0 --show-every 5
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
from pathlib import Path

from experiments.libero_runner import (
    make_libero_env, run_libero_trial, list_tasks, obs_from_libero,
)
from experiments.scene_config import ObstacleConfig

DEFAULT_EPISODES = 10


# ── Per-task obstacle and goal configuration ───────────────────────────────────
# Edit this dict to set CBF obstacles and goal positions for each task index.
# Key: (suite_name, task_idx)   Value: {"obstacles": [...], "goal_pos": np.array}
#
# For tasks where the goal is an object's resting place (e.g. a plate on the table),
# set goal_pos to that plate's approximate world-frame position.
# If you don't know the goal pos yet, run with --list-obs to print obs keys first.
TASK_CONFIGS: dict[tuple[str, int], dict] = {
    # libero_spatial task 0: pick black bowl (between plate & ramekin) → place on plate
    # Positions measured from --list-obs at reset (z≈0.97 = table surface + object height)
    ("libero_spatial", 0): {
        "goal_pos": np.array([0.062, 0.195, 1.00]),   # slightly above plate surface
        "target_obj_key": "akita_black_bowl_1_pos",   # key for virtual-grasp tracking
        "obstacles": [
            ObstacleConfig(
                pos=np.array([-0.2149, 0.2044, 0.97]),
                radius=0.04,
                safety_radius=0.09,
                name="ramekin",
            ),
            ObstacleConfig(
                pos=np.array([-0.1875, 0.3288, 0.97]),
                radius=0.04,
                safety_radius=0.09,
                name="black_bowl_2",
            ),
            ObstacleConfig(
                pos=np.array([0.0614, 0.0346, 0.97]),
                radius=0.04,
                safety_radius=0.07,
                name="cookies",
            ),
        ],
    },
    # Add more tasks here as you run them:
    # ("libero_spatial", 1): { "goal_pos": ..., "obstacles": [...] },
}

_DEFAULT_GOAL    = np.array([0.15, 0.15, 0.90])   # fallback if task not in TASK_CONFIGS
_DEFAULT_OBS_CFG: list[ObstacleConfig] = []         # no CBF constraints by default


# ── Benchmark runner ───────────────────────────────────────────────────────────
def run_task(suite: str, task_idx: int, use_cbf: bool,
             n_episodes: int, results_dir: Path,
             show_every: int = 0, collect_dataset: bool = False,
             save_video: bool = False) -> dict:

    key  = (suite, task_idx)
    tcfg = TASK_CONFIGS.get(key, {})
    goal_pos       = tcfg.get("goal_pos",       _DEFAULT_GOAL.copy())
    obstacles      = tcfg.get("obstacles",      _DEFAULT_OBS_CFG)
    target_obj_key = tcfg.get("target_obj_key", None)

    mode       = "cbf" if use_cbf else "plain"
    scene_name = f"{suite}_t{task_idx:02d}"
    label      = f"{scene_name}_{mode}"

    print(f"\n{'='*66}")
    print(f"  LIBERO benchmark: {label}  ({n_episodes} eps)")
    print(f"{'='*66}")

    records = []
    # Create env once and reuse (reset() is called inside run_libero_trial)
    env, lang = make_libero_env(
        task_suite=suite,
        task_idx=task_idx,
        has_renderer=False,            # viewer opened per-episode below if needed
        horizon=400,
    )

    try:
        for ep in range(n_episodes):
            show   = show_every > 0 and ep % show_every == 0
            ep_tag = "[view]" if show else "      "

            print(f"  ep {ep+1:3d}/{n_episodes} {ep_tag} ...", end="", flush=True)

            ds_path = (str(results_dir / "dataset" / scene_name / f"ep_{ep:04d}.npz")
                       if collect_dataset else None)
            vid_path = (str(results_dir / "videos" / scene_name / f"ep_{ep:04d}.mp4")
                        if save_video else None)

            metrics = run_libero_trial(
                env=env,
                obstacles=obstacles,
                instruction=lang,
                goal_pos=goal_pos,
                use_cbf=use_cbf,
                scene_name=scene_name,
                collect_dataset=collect_dataset,
                dataset_path=ds_path,
                target_obj_key=target_obj_key,
                show_viewer=show,
                save_video=vid_path,
            )

            s   = metrics.summary()
            car = 1 if s["violation_steps"] == 0 else 0
            tsr = 1 if s["goal_reached"]    else 0
            ets = s["goal_reach_step"] if s["goal_reached"] else s["total_steps"]
            records.append({"episode": ep, "car": car, "tsr": tsr, "ets": ets,
                            "min_dist": s["min_dist_overall"],
                            "cbf_acts": s["cbf_activations"]})
            print(f" CAR={car}  TSR={tsr}  ETS={ets:3d}  "
                  f"min_dist={s['min_dist_overall']:.3f}m  CBF_acts={s['cbf_activations']}")
            if vid_path:
                print(f"    Video → {vid_path}")
    finally:
        env.close()

    car_pct = np.mean([r["car"] for r in records]) * 100
    tsr_pct = np.mean([r["tsr"] for r in records]) * 100
    ets_mean = np.mean([r["ets"] for r in records])

    print(f"\n  ── {label} ──  CAR={car_pct:.1f}%  TSR={tsr_pct:.1f}%  ETS={ets_mean:.1f}")

    # Save CSVs
    ep_path  = results_dir / f"{label}_episodes.csv"
    agg_path = results_dir / f"{label}_agg.csv"
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene","mode","n_episodes",
                                           "car_pct","tsr_pct","ets"])
        w.writeheader()
        w.writerow({"scene": scene_name, "mode": mode,
                    "n_episodes": n_episodes,
                    "car_pct": round(car_pct,2),
                    "tsr_pct": round(tsr_pct,2),
                    "ets":     round(ets_mean,1)})

    return {"scene": scene_name, "mode": mode,
            "car": car_pct, "tsr": tsr_pct, "ets": ets_mean}


def print_table(results: list[dict]):
    scenes = list(dict.fromkeys(r["scene"] for r in results))
    w = 70
    print("\n" + "=" * w)
    print(f"  {'Method / Scene':<34}  {'CAR ↑':>9}  {'TSR ↑':>9}  {'ETS ↓':>9}")
    print("=" * w)
    for scene in scenes:
        for mode in ("plain", "cbf"):
            match = [r for r in results if r["scene"]==scene and r["mode"]==mode]
            if not match: continue
            r = match[0]
            print(f"  {scene+' ['+mode+']':<34}  "
                  f"{r['car']:>8.1f}%  {r['tsr']:>8.1f}%  {r['ets']:>9.1f}")
        print("-" * w)
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="AEGIS benchmark on LIBERO scenes")
    p.add_argument("--suite",      default="libero_spatial",
                   choices=["libero_spatial","libero_object","libero_goal",
                            "libero_long","libero_100"],
                   help="LIBERO task suite")
    p.add_argument("--task",       type=int, default=0,
                   help="Task index within the suite (0-based)")
    p.add_argument("--all",        action="store_true",
                   help="Run all tasks in the suite")
    p.add_argument("--list",       action="store_true",
                   help="List tasks in suite and exit")
    p.add_argument("--list-obs",   action="store_true",
                   help="Reset env and print observation keys (useful for setting goal_pos)")
    p.add_argument("--mode",       choices=["plain","cbf","both"], default="both")
    p.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    p.add_argument("--show-every", type=int, default=0,
                   help="Show live cv2 viewer every N episodes (0=off)")
    p.add_argument("--save-video", action="store_true",
                   help="Save each episode as MP4 to results-dir/videos/")
    p.add_argument("--results-dir",default="results_libero")
    p.add_argument("--collect-dataset", action="store_true",
                   help="Save per-episode .npz dataset files")
    args = p.parse_args()

    if args.list:
        list_tasks(args.suite)
        return

    if args.list_obs:
        env, lang = make_libero_env(args.suite, args.task)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        print(f"\n  Observation keys for {args.suite} task {args.task}:")
        for k, v in sorted(obs.items()):
            shape = getattr(v, "shape", type(v).__name__)
            print(f"    {k:<40} {shape}")
        env.close()
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    n_tasks = 10 if args.suite in ("libero_spatial","libero_object",
                                    "libero_goal","libero_long") else 100
    tasks   = list(range(n_tasks)) if args.all else [args.task]
    modes   = (["plain","cbf"] if args.mode == "both" else [args.mode])

    all_results = []
    for task_idx in tasks:
        for mode in modes:
            r = run_task(
                suite=args.suite, task_idx=task_idx,
                use_cbf=(mode == "cbf"),
                n_episodes=args.episodes,
                results_dir=results_dir,
                show_every=args.show_every,
                collect_dataset=args.collect_dataset,
                save_video=args.save_video,
            )
            all_results.append(r)

    print_table(all_results)


if __name__ == "__main__":
    main()
