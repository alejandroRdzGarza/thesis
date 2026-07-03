"""
SafeLIBERO CBF-only round-robin sample.

Creates all task envs up front, then runs episodes round-robin across tasks:
  ep0/t0 → ep0/t1 → ep0/t2 → ep0/t3 → ep1/t0 → ep1/t1 → ...

This gives cross-task coverage from the first pass — early termination still
yields balanced results across every task.  CBF only (no plain baseline).

Usage
-----
  python run_safelibero_cbf_sample.py --suite safelibero_spatial --safety-level I --episodes 3 --horizon 400 --replan-steps 8 --save-video --show-every 1 --results-dir results_cbf_sample
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from experiments.libero_runner import make_libero_env, run_libero_trial
from run_safelibero_paired import TASK_GOAL_POS, _DEFAULT_GOAL, _ep_metrics


def main() -> None:
    p = argparse.ArgumentParser(description="SafeLIBERO CBF-only round-robin sample")
    p.add_argument("--suite",          default="safelibero_spatial",
                   choices=["safelibero_spatial", "safelibero_object",
                            "safelibero_goal", "safelibero_long"])
    p.add_argument("--safety-level",   choices=["I", "II"], default="I")
    p.add_argument("--both-levels",    action="store_true",
                   help="Run both safety levels I and II")
    p.add_argument("--episodes",       type=int,   default=1,
                   help="Episodes per task (default 1 for quick representative sample)")
    p.add_argument("--horizon",        type=int,   default=400)
    p.add_argument("--replan-steps",   type=int,   default=8)
    p.add_argument("--safety-radius",  type=float, default=0.10)
    p.add_argument("--results-dir",    default="results_cbf_sample")
    p.add_argument("--save-video",     action="store_true")
    p.add_argument("--show-every",     type=int,   default=0,
                   help="Show live viewer every N episodes (0=off)")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    is_safe = args.suite.startswith("safelibero_")
    levels  = ["I", "II"] if args.both_levels else [args.safety_level]

    all_rows: list[dict] = []

    for level in levels:
        print(f"\n{'='*70}")
        print(f"  Safety Level {level} — creating all 4 task envs …")
        print(f"{'='*70}")

        # ── Create all 4 envs up front ────────────────────────────────────────
        task_envs: list[tuple] = []
        for task_idx in range(4):
            env, lang, init_states = make_libero_env(
                task_suite=args.suite,
                task_idx=task_idx,
                safety_level=level,
                has_renderer=False,
                horizon=args.horizon,
            )
            goal_pos = TASK_GOAL_POS.get((args.suite, task_idx), _DEFAULT_GOAL.copy())
            task_envs.append((task_idx, env, lang, init_states, goal_pos))
            print(f"  t{task_idx}: {lang[:65]}")

        print(f"\n  {'ep':>3}  {'task':<6}  CAR  TSR  Coll  ETS   minDist  CBFacts")
        print(f"  {'-'*3}  {'-'*6}  {'-'*3}  {'-'*3}  {'-'*4}  {'-'*4}  {'-'*7}  {'-'*7}")

        try:
            # ── Round-robin: ep0/t0, ep0/t1, ep0/t2, ep0/t3, ep1/t0, … ─────
            for ep in range(args.episodes):
                for task_idx, env, lang, init_states, goal_pos in task_envs:
                    scene = f"{args.suite}_t{task_idx:02d}_L{level}"
                    show  = args.show_every > 0 and ep % args.show_every == 0
                    vid_path = (
                        str(results_dir / "videos" / scene / f"ep{ep:03d}_cbf.mp4")
                        if args.save_video else None
                    )

                    m = run_libero_trial(
                        env=env,
                        obstacles=[],
                        instruction=lang,
                        goal_pos=goal_pos,
                        use_cbf=True,
                        scene_name=f"{scene}_cbf",
                        show_viewer=show,
                        save_video=vid_path,
                        episode_idx=ep,
                        initial_states=init_states,
                        auto_detect_obstacle=is_safe,
                        obstacle_safety_radius=args.safety_radius,
                        replan_steps=args.replan_steps,
                        horizon=args.horizon,
                    )
                    r = _ep_metrics(m, is_safe)
                    coll_s = "COLL" if r["collision"] else "    "
                    print(
                        f"  {ep:>3}  t{task_idx:<5}  "
                        f"{'✓' if r['car'] else '✗'}    "
                        f"{'✓' if r['tsr'] else '✗'}    "
                        f"{coll_s}  {r['ets']:>4}  {r['min_dist']:.3f}m   "
                        f"{r['cbf_acts']}"
                    )
                    all_rows.append({
                        "level": level, "task": task_idx, "episode": ep,
                        "scene": scene, **r,
                    })

        finally:
            for _, env, *_ in task_envs:
                env.close()

        # ── Per-task aggregate for this level ─────────────────────────────────
        print(f"\n  Level {level} aggregate:")
        for task_idx in range(4):
            rows = [r for r in all_rows if r["task"] == task_idx and r["level"] == level]
            if not rows:
                continue
            print(
                f"    t{task_idx}  CAR={np.mean([r['car'] for r in rows])*100:.0f}%"
                f"  TSR={np.mean([r['tsr'] for r in rows])*100:.0f}%"
                f"  Coll={np.mean([r['collision'] for r in rows])*100:.0f}%"
                f"  ETS={np.mean([r['ets'] for r in rows]):.0f}"
                f"  CBFacts={np.mean([r['cbf_acts'] for r in rows]):.1f}"
            )

    # ── CSV ───────────────────────────────────────────────────────────────────
    if all_rows:
        csv_path = results_dir / f"{args.suite}_cbf_sample.csv"
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n  Results → {csv_path}")


if __name__ == "__main__":
    main()
