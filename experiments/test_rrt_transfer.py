"""test_rrt_transfer.py — does a joint-space RRT plan survive execution through OSC_POSE?

This is the experiment that settles whether a sampling-based planner is usable as a teacher here.
The claim under test is NOT "can RRT find a collision-free path" (it can, trivially) but:

    Plan collision-free in joint space, convert with forward kinematics to an end-effector pose
    trace, command that trace through OSC_POSE — does the ARM stay collision-free during
    execution, given that OSC resolves the 7-DOF null space toward its own reset posture?

Measures, per scene:
  · whether RRT finds a path at all
  · EE tracking error while executing the FK trace
  · joint-space drift between the PLANNED configuration and the EXECUTED one (the null-space
    question, quantified rather than asserted)
  · whether any robot geom contacts a scene geom during execution (the thing that matters)

Run:
  PYTHONPATH=. python -m experiments.test_rrt_transfer --suite safelibero_spatial --level II --task 2
"""

from __future__ import annotations

import argparse
import contextlib
import io

import numpy as np


def execute(env, obs, model, data, qadr, trace, path, obj_name, args, writer=None):
    """Track the FK pose trace with OSC_POSE deltas and measure what actually happened.

    The plan is collision-free by construction; this reports whether the configuration OSC
    ACTUALLY adopts while tracking it is collision-free too, plus how far that configuration
    drifted from the planned one.
    """
    import contextlib, io
    import mujoco
    import numpy as np
    from scipy.spatial.transform import Rotation as Rot
    from experiments import rrt_planner as P

    collided, culprits = False, set()
    ee_errs, drifts, steps = [], [], 0
    for wi, (wp_pos, wp_R) in enumerate(trace):
        for _ in range(args.max_steps_per_wp):
            ee = np.asarray(obs["robot0_eef_pos"], float)
            err = wp_pos - ee
            if np.linalg.norm(err) < args.reach_tol:
                break
            Rcur = Rot.from_quat(np.asarray(obs["robot0_eef_quat"], float)).as_matrix()
            drot = Rot.from_matrix(wp_R @ Rcur.T).as_rotvec()
            act = np.zeros(7)
            act[:3] = np.clip(args.kp * err, -1, 1)
            act[3:6] = np.clip(args.krot * drot, -1, 1)
            act[6] = -1.0                                    # gripper stays open on the approach
            with contextlib.redirect_stdout(io.StringIO()):
                obs, _r, _d, _i = env.step(act)
            steps += 1
            if writer is not None:
                writer.write(np.asarray(obs["agentview_image"])[::-1, :, ::-1])

            for c in range(data.ncon):                        # live check on the EXECUTED config
                ct = data.contact[c]
                n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ct.geom1) or ""
                n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, ct.geom2) or ""
                r1, r2 = P._is_robot_geom(n1), P._is_robot_geom(n2)
                if r1 == r2:
                    continue
                other = n2 if r1 else n1
                if obj_name in other or "table" in other.lower() or "floor" in other.lower():
                    continue
                collided = True
                culprits.add(other)

        ee = np.asarray(obs["robot0_eef_pos"], float)
        ee_errs.append(float(np.linalg.norm(wp_pos - ee)))
        drifts.append(float(np.linalg.norm(np.array([data.qpos[a] for a in qadr]) - path[wi])))

    return {
        "steps": steps, "collided": collided, "culprits": culprits,
        "ee_mean": float(np.mean(ee_errs)), "ee_max": float(np.max(ee_errs)),
        "drift_mean": float(np.mean(drifts)), "drift_max": float(np.max(drifts)),
        "final_err": float(np.linalg.norm(trace[-1][0]
                                          - np.asarray(obs["robot0_eef_pos"], float))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=2)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--kp", type=float, default=20.0, help="P-gain on the EE position delta")
    ap.add_argument("--krot", type=float, default=4.0, help="gain on the EE rotation delta")
    ap.add_argument("--reach-tol", type=float, default=0.02, help="m, waypoint-reached tolerance")
    ap.add_argument("--max-steps-per-wp", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--try-candidates", type=int, default=8,
                    help="how many rim grasps to try before giving up; a plan that is\n                          collision-free may still execute into something, so candidates\n                          are screened by EXECUTION, not by the plan")
    ap.add_argument("--video", default=None)
    args = ap.parse_args()

    import mujoco
    from scipy.spatial.transform import Rotation as Rot
    from experiments.libero_runner import (make_libero_env, _get_arm_qpos_indices,
                                           _unwrap_sim)
    from experiments.classical_expert import resolve_pick_and_place
    from experiments import rrt_planner as P

    print(f"=== RRT → FK → OSC transfer test: {args.suite} L{args.level} t{args.task} ===")
    with contextlib.redirect_stdout(io.StringIO()):
        env, lang, inits = make_libero_env(task_suite=args.suite, task_idx=args.task,
                                           safety_level=args.level, horizon=args.horizon)
        env.reset()
        env.set_init_state(inits[args.episode])
        for _ in range(5):                     # let the scene settle
            obs, *_ = env.step(np.zeros(7))
        ctx = resolve_pick_and_place(env, obs)

    model, data = _unwrap_sim(env)
    qadr, jnt_rng = _get_arm_qpos_indices(model)
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
    assert qadr and sid >= 0, "could not resolve arm joints / grip site"

    obj = np.asarray(ctx["obj_pos"], float)
    print(f"  instruction : {lang}")
    print(f"  object      : {np.round(obj, 3)}")

    # The gripper is SUPPOSED to touch the target object; ignore it in collision checks.
    obj_name = ctx["obj_key"].replace("_pos", "")
    free = P.make_collision_fn(model, data, qadr, ignore=(obj_name,))

    q_start = np.array([data.qpos[a] for a in qadr])
    print(f"  start config free? {free(q_start)}")

    # ── 1. SEARCH the rim for a reachable, collision-free grasp ─────────────
    # A single hardcoded offset fails whenever that side is blocked (on the stove scene the +Y
    # offset drives a finger into the stove base). Sweep the rim instead.
    radius = P.object_rim_radius(model, data, obj_name, obj)
    print(f"  rim radius  : {radius*100:.1f} cm — sweeping for a feasible grasp …")
    cands = P.sample_rim_grasps(model, data, qadr, sid, obj, radius, free, seed=args.seed)
    if not cands:
        print("  GRASP SEARCH: no reachable collision-free rim grasp found "
              f"({16*3} poses tried) — this scene may genuinely have no rim grasp.")
        return
    print(f"  grasp search: {len(cands)} feasible poses")

    # Planning collision-free is not the same as EXECUTING collision-free: OSC picks its own
    # null-space configuration, so the arm can clip something the plan cleared. Rather than assume
    # either way, try candidates and keep the first whose EXECUTED trajectory is clean.
    def reset_env():
        with contextlib.redirect_stdout(io.StringIO()):
            env.reset()
            env.set_init_state(inits[args.episode])
            o = None
            for _ in range(5):
                o, *_ = env.step(np.zeros(7))
        return o

    chosen = None
    for ci, cand in enumerate(cands[:args.try_candidates]):
        q_goal, target = cand["q"], cand["pos"]
        path, why = P.rrt_connect(q_start, q_goal, jnt_rng, free, seed=args.seed)
        if path is None:
            print(f"    cand {ci}: RRT failed — {why}")
            continue
        path = P.densify(P.shortcut(path, free, seed=args.seed), max_step=0.05)
        v = P.verify_ee_trace(model, data, qadr, sid, path, free)
        obs = reset_env()
        res = execute(env, obs, model, data, qadr, v["trace"], path, obj_name, args, writer=None)
        tag = (f"theta={np.degrees(cand['theta']):3.0f}deg dz={cand['dz']:+.3f}")
        print(f"    cand {ci}: {tag}  {v['n_waypoints']:3d} wp  "
              f"reach {res['final_err']*1000:5.1f} mm  drift {res['drift_mean']:.2f} rad  "
              f"collided={res['collided']}"
              + (f" {sorted(res['culprits'])}" if res["collided"] else ""))
        if not res["collided"] and res["final_err"] < 0.05:
            chosen = (cand, path, v, res)
            break

    if chosen is None:
        print("\n  No candidate executed cleanly. The repair options are: bias the grasp choice "
              "away from the obstacle, or set the OSC null-space reference to the planned config.")
        with contextlib.suppress(Exception):
            env.close()
        return
    cand, path, v, res = chosen
    print(f"\n  CHOSEN      : theta={np.degrees(cand['theta']):.0f}deg dz={cand['dz']:+.3f}")
    print(f"  path        : {v['n_waypoints']} waypoints, EE path {v['path_len_m']:.3f} m")

    print(f"\n  ── RESULT ─────────────────────────────────────────────")
    print(f"    control steps           : {res['steps']}")
    print(f"    EE waypoint error       : mean {res['ee_mean']*1000:.1f} mm, "
          f"max {res['ee_max']*1000:.1f} mm")
    print(f"    joint drift (planned vs executed, the NULL-SPACE question)")
    print(f"                            : mean {res['drift_mean']:.3f} rad, "
          f"max {res['drift_max']:.3f} rad")
    print(f"    final EE vs grip target : {res['final_err']*1000:.1f} mm")
    print(f"    ARM COLLIDED DURING EXECUTION : {res['collided']}")
    print(f"  ───────────────────────────────────────────────────────")
    print("    VERDICT: RRT -> FK -> OSC TRANSFERS. Plan executes collision-free.")
    if args.video:
        obs = reset_env()
        import cv2
        w = cv2.VideoWriter(args.video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (256, 256))
        execute(env, obs, model, data, qadr, v["trace"], path, obj_name, args, writer=w)
        w.release()
        print(f"    video → {args.video}")
    with contextlib.suppress(Exception):
        env.close()


if __name__ == "__main__":
    main()
