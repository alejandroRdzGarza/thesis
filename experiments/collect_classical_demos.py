"""
collect_classical_demos.py — collect optimal-safe classical-expert demos for distillation
(Exp 007, round-0 OFFLINE BC). Runs the classical MPC-CBF pick-place controller across states,
records (obs images + 8-D state, executed expert action) traces + a manifest, consumable directly
by flow_bc_train --success-only.

  classical expert rollout (record_policy_trace) → <out>/*_trace.npz + manifest.csv
      → flow_bc_train --round <out> --success-only → BC-distilled LoRA

This is the OFFLINE half of the BC-then-DAgger recipe: demos collected under the EXPERT's own
distribution. The DAgger rounds (roll out the student VLA, label its states with this same
classical expert) reuse the same trace format + trainer.

Run on the Mac libero env (no VLA/GPU needed):
  python -m experiments.collect_classical_demos --suite safelibero_spatial --level II \
      --tasks 0 1 2 3 --episodes 0 1 2 3 4 5 6 7 --out results_distill/round0
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--tasks", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--episodes", type=int, nargs="+", default=[0, 1, 2, 3])
    ap.add_argument("--out", default="results_distill/round0")
    ap.add_argument("--horizon", type=int, default=300)
    ap.add_argument("--replan", type=int, default=5)
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.classical_expert import PickPlaceController
    from experiments.policy_trace import save_episode_trace

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    controller = PickPlaceController()
    rows = []
    n = succ = coll = clean = 0
    for task in args.tasks:
        env, lang, init_states = make_libero_env(
            task_suite=args.suite, task_idx=task, safety_level=args.level, horizon=args.horizon)
        for ep in args.episodes:
            controller.reset()
            m = run_libero_trial(
                env=env, episode_idx=ep, instruction=lang, initial_states=init_states,
                obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False, use_cbf=True,
                vla="pi05", auto_detect_obstacle=True, aegis_faithful=True, replan_steps=args.replan,
                horizon=args.horizon, controller=controller, record_policy_trace=True,
                scene_name=f"{args.suite}_L{args.level}_t{task}",
                # Per-scene teacher: the controller is specialised to this task's geometry
                # (grasp side, elevation, corridor) plus any tuned override.
                teacher_suite=args.suite, teacher_level=args.level, teacher_task=task)
            s = m.summary()
            ok = bool(s["goal_reached"]); cl = bool(s["collision_detected"])
            tp = str((out / f"{args.suite}_L{args.level}_t{task}_ep{ep}_trace.npz").resolve())
            save_episode_trace(m.policy_trace, tp)
            rows.append({"trace_path": tp, "r_success": 1.5 if ok else 0.0,
                         "robot_caused_collision": int(cl),
                         "suite": args.suite, "task": task, "episode": ep})
            n += 1; succ += int(ok); coll += int(cl); clean += int(ok and not cl)
            print(f"  {args.suite} L{args.level} t{task} ep{ep}: succ={ok} coll={cl} "
                  f"queries={len(m.policy_trace)} → {Path(tp).name}", flush=True)
        try:
            env.close()
        except Exception:
            pass

    with open(out / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["trace_path", "r_success", "robot_caused_collision",
                                          "suite", "task", "episode"])
        w.writeheader(); w.writerows(rows)
    print(f"\n{n} demos: {succ} success, {coll} collided, {clean} CLEAN (success+safe) "
          f"→ {out}/manifest.csv")
    print(f"  distil with:  flow_bc_train --round {out} --success-only  "
          f"(uses the {clean} clean demos)")


if __name__ == "__main__":
    main()
