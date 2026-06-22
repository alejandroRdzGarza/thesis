"""
VLA+CBF safety benchmark on SafeLIBERO scenes.

SafeLIBERO extends LIBERO with obstacle-augmented scenes at two difficulty levels:
  Level I  — obstacle near the target object (arm must approach carefully)
  Level II — obstacle blocking the movement path (arm must detour)

Each scenario has 50 randomised episodes with varying obstacle and object positions.
Collision detection uses the AEGIS paper metric: obstacle displaced > 0.001 m.

Benchmark modes:
  cbf   — OpenVLA + Cartesian CBF safety filter (proposed method)
  plain — OpenVLA alone, no CBF (baseline)
  both  — run both modes, print comparison table

Usage:
    # List available SafeLIBERO tasks
    python run_libero_benchmark.py --list --suite safelibero_spatial --safety-level I

    # Quick 5-episode test on spatial task 0, level I, CBF only
    python run_libero_benchmark.py --suite safelibero_spatial --safety-level I --task 0 --episodes 5 --mode cbf

    # Full 50-episode evaluation, both modes
    python run_libero_benchmark.py --suite safelibero_spatial --safety-level I --task 0 --episodes 50

    # All 4 tasks in a suite
    python run_libero_benchmark.py --suite safelibero_spatial --safety-level I --all --episodes 50

    # Standard LIBERO (no obstacles, no SafeLIBERO)
    python run_libero_benchmark.py --suite libero_spatial --task 0 --episodes 10 --mode cbf
"""

from __future__ import annotations

import argparse
import csv
import numpy as np
from pathlib import Path

from experiments.libero_runner import (
    make_libero_env, run_libero_trial, list_tasks,
    detect_safelibero_obstacle, obs_from_libero,
)
from experiments.scene_config import ObstacleConfig

DEFAULT_EPISODES = 10

# ── Optional per-task goal-position overrides ──────────────────────────────────
# For SafeLIBERO, success is detected via info["success"] (LIBERO built-in),
# so goal_pos is not required.  You can supply one here to also track EE
# distance progress in the step log.  Leave as None to skip distance tracking.
#
# For standard LIBERO (no SafeLIBERO), goal_pos IS needed for success detection.
TASK_GOAL_POS: dict[tuple[str, int], np.ndarray | None] = {
    # SafeLIBERO Spatial
    ("safelibero_spatial", 0): np.array([0.062, 0.195, 1.00]),  # plate
    ("safelibero_spatial", 1): np.array([0.062, 0.195, 1.00]),
    ("safelibero_spatial", 2): np.array([0.062, 0.195, 1.00]),
    ("safelibero_spatial", 3): np.array([0.062, 0.195, 1.00]),
    # Standard LIBERO Spatial (fallback — still needs goal_pos for success detection)
    ("libero_spatial", 0): np.array([0.062, 0.195, 1.00]),
}
_DEFAULT_GOAL = np.array([0.15, 0.15, 0.90])


# ── Benchmark runner ───────────────────────────────────────────────────────────
def run_task(suite: str, task_idx: int, use_cbf: bool,
             n_episodes: int, results_dir: Path,
             safety_level: str = "I",
             cbf_gamma: float = 1.8,
             obstacle_safety_radius: float = 0.10,
             show_every: int = 0,
             collect_dataset: bool = False,
             save_video: bool = False,
             replan_steps: int = 5) -> dict:

    is_safe = suite.startswith("safelibero_")
    mode    = "cbf" if use_cbf else "plain"
    scene_name = f"{suite}_t{task_idx:02d}" + (f"_L{safety_level}" if is_safe else "")
    label  = f"{scene_name}_{mode}"

    print(f"\n{'='*70}")
    print(f"  {label}  ({n_episodes} eps)")
    if is_safe:
        print(f"  SafeLIBERO level={safety_level}  cbf_gamma={cbf_gamma}  "
              f"safety_radius={obstacle_safety_radius:.2f}m  replan_steps={replan_steps}")
    print(f"{'='*70}")

    env, lang, initial_states = make_libero_env(
        task_suite=suite,
        task_idx=task_idx,
        safety_level=safety_level,
        has_renderer=False,
        horizon=800,
    )

    goal_pos = TASK_GOAL_POS.get((suite, task_idx), _DEFAULT_GOAL.copy())

    # For SafeLIBERO: obstacles are auto-detected after env.set_init_state().
    # For standard LIBERO: obstacles must be provided here (or left empty).
    manual_obstacles: list[ObstacleConfig] = []

    records = []
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
                obstacles=manual_obstacles,
                instruction=lang,
                goal_pos=goal_pos,
                use_cbf=use_cbf,
                cbf_gamma=cbf_gamma,
                scene_name=scene_name,
                collect_dataset=collect_dataset,
                dataset_path=ds_path,
                show_viewer=show,
                save_video=vid_path,
                # SafeLIBERO params
                episode_idx=ep,
                initial_states=initial_states,
                auto_detect_obstacle=is_safe,
                obstacle_safety_radius=obstacle_safety_radius,
                replan_steps=replan_steps,
            )

            s = metrics.summary()
            # CAR: 1 if no physical collision (displacement metric for SafeLIBERO,
            #      or no CBF violation for standard LIBERO)
            if is_safe:
                car = 0 if s["collision_detected"] else 1
            else:
                car = 0 if s["violation_steps"] > 0 else 1
            tsr = 1 if s["goal_reached"] else 0
            ets = s["goal_reach_step"] if s["goal_reached"] else s["total_steps"]

            records.append({
                "episode":    ep,
                "car":        car,
                "tsr":        tsr,
                "ets":        ets,
                "collision":  int(s["collision_detected"]),
                "cbf_acts":   s["cbf_activations"],
                "min_dist":   s["min_dist_overall"],
                "violations": s["violation_steps"],
            })
            print(f" CAR={car}  TSR={tsr}  ETS={ets:3d}  "
                  f"collision={s['collision_detected']}  "
                  f"min_dist={s['min_dist_overall']:.3f}m  "
                  f"CBF_acts={s['cbf_activations']}")
            if vid_path:
                print(f"    Video → {vid_path}")

    finally:
        env.close()

    car_pct    = np.mean([r["car"]       for r in records]) * 100
    tsr_pct    = np.mean([r["tsr"]       for r in records]) * 100
    coll_rate  = np.mean([r["collision"] for r in records]) * 100
    ets_mean   = np.mean([r["ets"]       for r in records])

    print(f"\n  ── {label} ──  CAR={car_pct:.1f}%  TSR={tsr_pct:.1f}%  "
          f"CollisionRate={coll_rate:.1f}%  ETS={ets_mean:.1f}")

    ep_path  = results_dir / f"{label}_episodes.csv"
    agg_path = results_dir / f"{label}_agg.csv"
    with open(ep_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader(); w.writerows(records)
    with open(agg_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "mode", "safety_level",
                                          "n_episodes", "car_pct", "tsr_pct",
                                          "collision_rate_pct", "ets"])
        w.writeheader()
        w.writerow({
            "scene":                scene_name,
            "mode":                 mode,
            "safety_level":         safety_level if is_safe else "n/a",
            "n_episodes":           n_episodes,
            "car_pct":              round(car_pct, 2),
            "tsr_pct":              round(tsr_pct, 2),
            "collision_rate_pct":   round(coll_rate, 2),
            "ets":                  round(ets_mean, 1),
        })

    return {
        "scene":      scene_name, "mode": mode,
        "car":        car_pct,    "tsr":  tsr_pct,
        "collision":  coll_rate,  "ets":  ets_mean,
    }


def print_table(results: list[dict]):
    scenes = list(dict.fromkeys(r["scene"] for r in results))
    w = 80
    print("\n" + "=" * w)
    print(f"  {'Method / Scene':<36}  {'CAR ↑':>8}  {'TSR ↑':>8}  "
          f"{'Coll ↓':>8}  {'ETS ↓':>8}")
    print("=" * w)
    for scene in scenes:
        for mode in ("plain", "cbf"):
            match = [r for r in results if r["scene"] == scene and r["mode"] == mode]
            if not match:
                continue
            r = match[0]
            print(f"  {scene + ' [' + mode + ']':<36}  "
                  f"{r['car']:>7.1f}%  {r['tsr']:>7.1f}%  "
                  f"{r['collision']:>7.1f}%  {r['ets']:>8.1f}")
        print("-" * w)
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="VLA+CBF benchmark on SafeLIBERO/LIBERO scenes")
    p.add_argument("--suite", default="safelibero_spatial",
                   choices=[
                       "safelibero_spatial", "safelibero_object",
                       "safelibero_goal",    "safelibero_long",
                       "libero_spatial",     "libero_object",
                       "libero_goal",        "libero_90", "libero_10",
                   ],
                   help="Task suite to evaluate")
    p.add_argument("--safety-level", choices=["I", "II"], default="I",
                   help="SafeLIBERO safety level (I = obstacle near target, II = obstacle on path)")
    p.add_argument("--task",       type=int, default=0,  help="Task index (0-based)")
    p.add_argument("--all",        action="store_true",  help="Run all tasks in the suite")
    p.add_argument("--list",       action="store_true",  help="List tasks and exit")
    p.add_argument("--list-obs",   action="store_true",  help="Print observation keys after reset")
    p.add_argument("--mode",       choices=["plain", "cbf", "both"], default="both")
    p.add_argument("--episodes",   type=int, default=DEFAULT_EPISODES)
    p.add_argument("--cbf-gamma",  type=float, default=1.8,
                   help="CBF class-K coefficient (higher = more conservative)")
    p.add_argument("--safety-radius", type=float, default=0.10,
                   help="Safety exclusion radius around auto-detected obstacle (m)")
    p.add_argument("--show-every", type=int, default=0,
                   help="Show live viewer every N episodes (0=off)")
    p.add_argument("--save-video", action="store_true")
    p.add_argument("--results-dir", default="results_libero")
    p.add_argument("--collect-dataset", action="store_true")
    p.add_argument("--replan-steps", type=int, default=5,
                   help="VLA query every N control steps (AEGIS approach, default=5)")
    args = p.parse_args()

    if args.list:
        list_tasks(args.suite, safety_level=args.safety_level)
        return

    if args.list_obs:
        env, lang, _ = make_libero_env(args.suite, args.task,
                                        safety_level=args.safety_level)
        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        print(f"\n  Observation keys for {args.suite} level={args.safety_level} task {args.task}:")
        for k, v in sorted(obs.items()):
            shape = getattr(v, "shape", type(v).__name__)
            print(f"    {k:<44} {shape}")
        env.close()
        return

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    is_safe = args.suite.startswith("safelibero_")
    n_tasks = 4 if is_safe else 10
    tasks   = list(range(n_tasks)) if args.all else [args.task]
    modes   = (["plain", "cbf"] if args.mode == "both" else [args.mode])

    all_results = []
    for task_idx in tasks:
        for mode in modes:
            r = run_task(
                suite=args.suite,
                task_idx=task_idx,
                use_cbf=(mode == "cbf"),
                n_episodes=args.episodes,
                results_dir=results_dir,
                safety_level=args.safety_level,
                cbf_gamma=args.cbf_gamma,
                obstacle_safety_radius=args.safety_radius,
                show_every=args.show_every,
                collect_dataset=args.collect_dataset,
                save_video=args.save_video,
                replan_steps=args.replan_steps,
            )
            all_results.append(r)

    print_table(all_results)


if __name__ == "__main__":
    main()
