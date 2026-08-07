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
    ap.add_argument("--teacher", default="classical", choices=["classical", "planner"],
                    help="which expert generates the demos: the scripted MPC-CBF controller, or "
                         "the joint-space RRT planner")
    ap.add_argument("--randomize-seed", type=int, default=None,
                    help="base seed for TRAINING-DATA obstacle randomisation; each episode gets "
                         "seed+episode. Leave unset for canonical layouts. Never use for eval.")
    ap.add_argument("--no-cbf", dest="use_cbf", action="store_false", default=None,
                    help="collect WITHOUT the CBF shield. Default depends on the teacher: the "
                         "scripted controller NEEDS the shield (its nominal actions collide ~60%% "
                         "unshielded), whereas the planner is self-safe by construction and the "
                         "shield actively fights it — it fired on every step and stalled the "
                         "episode in LIFT.")
    ap.add_argument("--clean-only", action="store_true",
                    help="only write a trace when the episode SUCCEEDED and stayed collision-free. "
                         "BC discards the rest anyway (flow_bc_train --success-only), and each "
                         "trace carries ~60 image pairs — on a full-grid collection this is the "
                         "difference between ~4 GB and ~10 GB on disk.")
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, run_libero_trial
    from experiments.policy_trace import save_episode_trace
    from experiments.progress import Progress
    if args.teacher == "planner":
        from experiments.planner_expert import PlannerExpert as _Teacher
    else:
        from experiments.classical_expert import PickPlaceController as _Teacher
    if args.use_cbf is None:
        args.use_cbf = (args.teacher != "planner")
    print(f"  teacher={args.teacher}  shield={'ON' if args.use_cbf else 'OFF'}"
          f"  randomise={'seed ' + str(args.randomize_seed) if args.randomize_seed is not None else 'no'}",
          flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rows = []
    n = succ = coll = clean = 0
    _total = len(args.tasks) * len(args.episodes)
    prog = Progress(_total, f"collect [{args.teacher}] {args.suite} L{args.level}")
    prog.__enter__()
    for task in args.tasks:
        env, lang, init_states = make_libero_env(
            task_suite=args.suite, task_idx=task, safety_level=args.level, horizon=args.horizon)
        for ep in args.episodes:
            controller = _Teacher()
            controller.reset()
            m = run_libero_trial(
                env=env, episode_idx=ep, instruction=lang, initial_states=init_states,
                obstacles=[], goal_pos=None, auto_goal=True, use_geo_success=False, use_cbf=args.use_cbf,
                vla="pi05", auto_detect_obstacle=True, aegis_faithful=True, replan_steps=args.replan,
                horizon=args.horizon, controller=controller, record_policy_trace=True,
                scene_name=f"{args.suite}_L{args.level}_t{task}",
                # Per-scene teacher: the controller is specialised to this task's geometry
                # (grasp side, elevation, corridor) plus any tuned override.
                teacher_suite=args.suite, teacher_level=args.level, teacher_task=task,
                randomize_obstacle_seed=(None if args.randomize_seed is None
                                         else args.randomize_seed + ep))
            s = m.summary()
            ok = bool(s["goal_reached"]); cl = bool(s["collision_detected"])
            is_clean = ok and not cl
            tp = str((out / f"{args.suite}_L{args.level}_t{task}_ep{ep}_trace.npz").resolve())
            if is_clean or not args.clean_only:
                save_episode_trace(m.policy_trace, tp)
                rows.append({"trace_path": tp, "r_success": 1.5 if ok else 0.0,
                             "robot_caused_collision": int(cl),
                             "suite": args.suite, "task": task, "episode": ep})
            n += 1; succ += int(ok); coll += int(cl); clean += int(is_clean)
            prog.note(clean=int(is_clean))
            prog.step(f"t{task} ep{ep} {'KEPT' if is_clean else 'drop'}")
        try:
            env.close()
        except Exception:
            pass

    prog.__exit__()
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
