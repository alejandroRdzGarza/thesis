"""
SafeLIBERO paired comparison: plain vs CBF on identical init states.

Each episode index runs plain first, then CBF on the SAME initial_states[ep],
so results are directly comparable scene-for-scene.

Usage:
    # Single task, 5 episodes
    python run_safelibero_paired.py --suite safelibero_spatial --safety-level I --task 0 --episodes 5

    # All 4 tasks in a suite
    python run_safelibero_paired.py --suite safelibero_spatial --safety-level I --all --episodes 10

    # Both levels
    python run_safelibero_paired.py --suite safelibero_spatial --all --episodes 10 --both-levels
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from experiments.libero_runner import make_libero_env, run_libero_trial
from experiments.scene_config import ObstacleConfig

TASK_GOAL_POS: dict[tuple[str, int], np.ndarray] = {
    ("safelibero_spatial", 0): np.array([0.062, 0.195, 1.00]),
    ("safelibero_spatial", 1): np.array([0.062, 0.195, 1.00]),
    ("safelibero_spatial", 2): np.array([0.062, 0.195, 1.00]),
    ("safelibero_spatial", 3): np.array([0.062, 0.195, 1.00]),
}
_DEFAULT_GOAL = np.array([0.15, 0.15, 0.90])


def _ep_metrics(metrics, is_safe: bool) -> dict:
    s = metrics.summary()
    car = 0 if s["collision_detected"] else 1 if is_safe else (0 if s["violation_steps"] > 0 else 1)
    tsr = 1 if s["goal_reached"] else 0
    ets = s["goal_reach_step"] if s["goal_reached"] else s["total_steps"]
    return {
        "car": car,
        "tsr": tsr,
        "ets": ets,
        "collision": int(s["collision_detected"]),
        "cbf_acts": s["cbf_activations"],
        "min_dist": s["min_dist_overall"],
        "violations": s["violation_steps"],
    }


def run_paired_task(
    suite: str,
    task_idx: int,
    safety_level: str,
    n_episodes: int,
    results_dir: Path,
    horizon: int,
    replan_steps: int,
    obstacle_safety_radius: float,
    save_video: bool,
    show_every: int = 0,
) -> list[dict]:
    is_safe = suite.startswith("safelibero_")
    scene = f"{suite}_t{task_idx:02d}" + (f"_L{safety_level}" if is_safe else "")

    print(f"\n{'='*72}")
    print(f"  PAIRED  {scene}  ({n_episodes} episodes × 2 modes)")
    print(f"{'='*72}")

    env, lang, initial_states = make_libero_env(
        task_suite=suite,
        task_idx=task_idx,
        safety_level=safety_level,
        has_renderer=False,
        horizon=horizon,
    )
    goal_pos = TASK_GOAL_POS.get((suite, task_idx), _DEFAULT_GOAL.copy())

    print(f"\n  {'EP':>3}  {'— PLAIN ——————————————————————':32}  {'— CBF ————————————————————————':32}")
    print(f"  {'':>3}  {'CAR  TSR  Coll  ETS   minDist':32}  {'CAR  TSR  Coll  ETS   minDist  CBFacts':38}")
    print(f"  {'-'*3}  {'-'*32}  {'-'*38}")

    all_rows: list[dict] = []

    try:
        for ep in range(n_episodes):
            row: dict = {"episode": ep, "scene": scene}

            for mode in ("plain", "cbf"):
                use_cbf = (mode == "cbf")
                show = show_every > 0 and ep % show_every == 0
                vid_path = (
                    str(results_dir / "videos" / scene / f"ep{ep:03d}_{mode}.mp4")
                    if save_video else None
                )

                m = run_libero_trial(
                    env=env,
                    obstacles=[],           # auto-detected inside trial for SafeLIBERO
                    instruction=lang,
                    goal_pos=goal_pos,
                    use_cbf=use_cbf,
                    scene_name=f"{scene}_{mode}",
                    show_viewer=show,
                    save_video=vid_path,
                    episode_idx=ep,
                    initial_states=initial_states,
                    auto_detect_obstacle=is_safe,
                    obstacle_safety_radius=obstacle_safety_radius,
                    replan_steps=replan_steps,
                    horizon=horizon,
                )
                r = _ep_metrics(m, is_safe)
                for k, v in r.items():
                    row[f"{mode}_{k}"] = v

            # side-by-side print
            p, c = {k[6:]: v for k, v in row.items() if k.startswith("plain_")}, \
                   {k[4:]: v for k, v in row.items() if k.startswith("cbf_")}
            coll_str_p = "COLL" if p["collision"] else "    "
            coll_str_c = "COLL" if c["collision"] else "    "
            print(
                f"  {ep:>3}  "
                f"{'✓' if p['car'] else '✗'}    {'✓' if p['tsr'] else '✗'}   "
                f"{coll_str_p}  {p['ets']:>4}  {p['min_dist']:.3f}m          "
                f"{'✓' if c['car'] else '✗'}    {'✓' if c['tsr'] else '✗'}   "
                f"{coll_str_c}  {c['ets']:>4}  {c['min_dist']:.3f}m  "
                f"CBF={c['cbf_acts']}"
            )

            all_rows.append(row)

    finally:
        env.close()

    # ── Aggregate summary ──────────────────────────────────────────────────────
    def agg(mode: str, key: str) -> float:
        return float(np.mean([r[f"{mode}_{key}"] for r in all_rows]))

    print(f"\n  {'':>3}  {'PLAIN':32}  {'CBF':38}")
    print(
        f"  {'AGG':>3}  "
        f"CAR={agg('plain','car')*100:.0f}%  TSR={agg('plain','tsr')*100:.0f}%  "
        f"Coll={agg('plain','collision')*100:.0f}%  ETS={agg('plain','ets'):.0f}          "
        f"CAR={agg('cbf','car')*100:.0f}%  TSR={agg('cbf','tsr')*100:.0f}%  "
        f"Coll={agg('cbf','collision')*100:.0f}%  ETS={agg('cbf','ets'):.0f}  "
        f"CBF_acts={agg('cbf','cbf_acts'):.1f}"
    )

    # ── CSV output ─────────────────────────────────────────────────────────────
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{scene}_paired.csv"
    with open(csv_path, "w", newline="") as f:
        cols = list(all_rows[0].keys())
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"\n  Results → {csv_path}")

    return all_rows


def print_final_table(all_rows: list[dict]) -> None:
    scenes = sorted({r["scene"] for r in all_rows})
    if len(scenes) <= 1:
        return

    print(f"\n\n{'='*80}")
    print(f"  {'Scene':<38}  {'PLAIN':>22}   {'CBF':>22}")
    print(f"  {'':38}  {'CAR%  TSR%  Coll%  ETS':22}   {'CAR%  TSR%  Coll%  ETS':22}")
    print(f"  {'-'*38}  {'-'*22}   {'-'*22}")

    def agg(rows, mode, key):
        return float(np.mean([r[f"{mode}_{key}"] for r in rows]))

    for scene in scenes:
        rows = [r for r in all_rows if r["scene"] == scene]
        print(
            f"  {scene:<38}  "
            f"{agg(rows,'plain','car')*100:4.0f}%"
            f"  {agg(rows,'plain','tsr')*100:4.0f}%"
            f"  {agg(rows,'plain','collision')*100:5.0f}%"
            f"  {agg(rows,'plain','ets'):5.0f}    "
            f"{agg(rows,'cbf','car')*100:4.0f}%"
            f"  {agg(rows,'cbf','tsr')*100:4.0f}%"
            f"  {agg(rows,'cbf','collision')*100:5.0f}%"
            f"  {agg(rows,'cbf','ets'):5.0f}"
        )
    print(f"  {'='*80}")


def main() -> None:
    p = argparse.ArgumentParser(description="SafeLIBERO paired plain vs CBF comparison")
    p.add_argument("--suite", default="safelibero_spatial",
                   choices=["safelibero_spatial", "safelibero_object",
                            "safelibero_goal", "safelibero_long"])
    p.add_argument("--safety-level", choices=["I", "II"], default="I")
    p.add_argument("--task",         type=int,   default=0)
    p.add_argument("--all",          action="store_true", help="Run all 4 tasks")
    p.add_argument("--both-levels",  action="store_true", help="Run both safety levels I and II")
    p.add_argument("--episodes",     type=int,   default=5)
    p.add_argument("--horizon",      type=int,   default=500)
    p.add_argument("--replan-steps", type=int,   default=8)
    p.add_argument("--safety-radius", type=float, default=0.10)
    p.add_argument("--results-dir",  default="results_paired")
    p.add_argument("--save-video",   action="store_true")
    p.add_argument("--show-every",   type=int, default=0,
                   help="Show live viewer every N episodes (0=off)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    tasks   = list(range(4)) if args.all else [args.task]
    levels  = ["I", "II"]   if args.both_levels else [args.safety_level]

    all_rows: list[dict] = []

    for level in levels:
        for task_idx in tasks:
            rows = run_paired_task(
                suite=args.suite,
                task_idx=task_idx,
                safety_level=level,
                n_episodes=args.episodes,
                results_dir=results_dir,
                horizon=args.horizon,
                replan_steps=args.replan_steps,
                obstacle_safety_radius=args.safety_radius,
                save_video=args.save_video,
                show_every=args.show_every,
            )
            all_rows.extend(rows)

    print_final_table(all_rows)


if __name__ == "__main__":
    main()
