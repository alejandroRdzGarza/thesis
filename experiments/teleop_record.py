"""teleop_record.py — hand-demonstrate a task with the keyboard and record a demo trace usable by
the distillation pipeline (flow_bc_train). For the hard tasks (bowl wedged against an obstacle)
where the scripted expert struggles: you drive the optimal trajectory once, we record it.

Records, exactly like collect_classical_demos, per-query obs (224² agentview + wrist + 8-D proprio +
prompt) and the executed OSC_POSE actions in between → <out>/*_trace.npz + manifest.csv. Mix these
with the scripted demos (flow_bc_train --round demos_scripted demos_teleop) or point a sweep at them.

Run ON THE MAC (opens a window — needs a display; not over headless SSH):
  MUJOCO_GL= PYTHONPATH=. python -m experiments.teleop_record \
      --suite safelibero_spatial --level II --task 3 --episodes 0 1 2 3 --out demos_teleop

Controls are printed at start (robosuite keyboard). Typical: arrows/wasd-style move, keys rotate,
SPACE toggles the gripper. Press the RESET key to END the current episode (saved if it succeeded,
or always with --save-fails). Success is auto-detected and also ends the episode.
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="demos_teleop")
    ap.add_argument("--replan", type=int, default=5, help="record one query obs every N steps (match training)")
    ap.add_argument("--horizon", type=int, default=1500)
    ap.add_argument("--pos-sensitivity", type=float, default=1.5)
    ap.add_argument("--rot-sensitivity", type=float, default=1.5)
    ap.add_argument("--save-fails", action="store_true", help="save episodes even if success wasn't detected")
    args = ap.parse_args()

    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action

    from experiments.libero_runner import make_libero_env, _preprocess, _build_proprio
    from experiments.policy_trace import QueryTrace, save_episode_trace

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    env, lang, init = make_libero_env(task_suite=args.suite, task_idx=args.task,
                                      safety_level=args.level, has_renderer=True, horizon=args.horizon)
    robot = env.env.robots[0]
    kb = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    try:
        kb._display_controls()
    except Exception:
        pass
    print(f'\nTELEOP  {args.suite} L{args.level} t{args.task}:  "{lang}"')
    print("  Drive the arm; SPACE toggles the gripper; press the RESET key to END/SAVE an episode.\n")

    def _success():
        try:
            return bool(env.check_success())
        except Exception:
            try:
                return bool(env._check_success())
            except Exception:
                return False

    def _obs_query(obs):
        img = _preprocess(obs["agentview_image"]) if "agentview_image" in obs else np.zeros((224, 224, 3), np.uint8)
        wri = _preprocess(obs["robot0_eye_in_hand_image"]) if "robot0_eye_in_hand_image" in obs else np.zeros((224, 224, 3), np.uint8)
        return {"image": np.asarray(img, np.uint8), "wrist_image": np.asarray(wri, np.uint8),
                "state": np.asarray(_build_proprio(obs), np.float32), "prompt": lang}

    import time as _time
    import mujoco
    import mujoco.viewer as _mjv

    rows = []
    for ep in args.episodes:
        obs = env.reset()
        if init is not None:
            obs = env.set_init_state(init[ep])
        for _ in range(20):                       # settle
            obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
        # OffScreenRenderEnv has no on-screen render() — drive the display with MuJoCo's own passive
        # viewer bound to the sim's model/data (fetched AFTER reset, which may rebuild the sim).
        model = env.sim.model._model
        data = env.sim.data._data
        kb.start_control()
        print(f"--- episode {ep}: GO (press q to finish/save) ---", flush=True)

        queries, bufs = [], []
        t = 0; ok = False
        with _mjv.launch_passive(model, data) as viewer:
            while t < args.horizon and viewer.is_running():
                _t0 = _time.time()
                action, _grasp = input2action(device=kb, robot=robot,
                                               active_arm="right", env_configuration="single-arm-opposed")
                if action is None:                # reset key (q) → end episode
                    break
                if t % args.replan == 0:          # start a new query
                    queries.append(QueryTrace(chain=np.zeros((2, 1, 1), np.float32),
                                              logp_old=np.zeros(1, np.float32),
                                              sigmas=np.array([1.0, 0.0], np.float32),
                                              noise_level=0.0, sde_type="teleop", obs=_obs_query(obs)))
                    bufs.append([])
                obs, _r, done, _info = env.step(action)
                bufs[-1].append(np.asarray(action, np.float32)[:7].copy())
                viewer.sync()
                t += 1
                if _success():
                    ok = True
                    print(f"  *** SUCCESS at step {t} ***", flush=True)
                    break
                if done:
                    break
                _time.sleep(max(0.0, 0.05 - (_time.time() - _t0)))   # ~20 Hz, controllable by hand

        for q, b in zip(queries, bufs):           # attach executed actions
            if b:
                q.shielded_actions = np.asarray(b, np.float32)
        usable = [q for q in queries if q.shielded_actions is not None]
        if usable and (ok or args.save_fails):
            tp = str((out / f"{args.suite}_L{args.level}_t{args.task}_ep{ep}_teleop_trace.npz").resolve())
            save_episode_trace(usable, tp)
            rows.append({"trace_path": tp, "r_success": 1.5 if ok else 0.0,
                         "robot_caused_collision": 0, "suite": args.suite, "task": args.task, "episode": ep})
            print(f"  saved {len(usable)} queries → {Path(tp).name}  (success={ok})", flush=True)
        else:
            print(f"  episode {ep}: not saved (success={ok}, queries={len(usable)})", flush=True)

    if rows:
        mpath = out / "manifest.csv"
        exists = mpath.exists()
        with open(mpath, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["trace_path", "r_success", "robot_caused_collision",
                                              "suite", "task", "episode"])
            if not exists:
                w.writeheader()
            w.writerows(rows)
        print(f"\n{len(rows)} teleop demos → {mpath}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
