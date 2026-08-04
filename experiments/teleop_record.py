"""teleop_record.py — STAGE 1 of hand-demonstration: drive a task on-screen with the keyboard and
record (sim_state, action) per step. macOS-robust combo:
  - env with NO robosuite rendering (has_renderer=False, has_offscreen_renderer=False) → no GL
    context conflict (that caused the 'NSWindow on non-main thread' crash),
  - MuJoCo's OWN passive viewer for the window (main-thread-safe under mjpython),
  - pynput Keyboard for input.
Images are NOT rendered here; STAGE 2 (teleop_to_trace.py) regenerates them offscreen from the states.

Run under **mjpython** (the MuJoCo viewer needs it on macOS), and grant your terminal Accessibility
(System Settings ▸ Privacy & Security ▸ Accessibility) so pynput can read keys, then restart it:
  MUJOCO_GL= PYTHONPATH=. /Users/alexrdzgarza/miniforge3/envs/libero/bin/mjpython \
      -m experiments.teleop_record --suite safelibero_spatial --level II --task 3 --episodes 0 1 2

Controls: w/a/s/d move XY · r/f up-down · z/x·t/g·c/v rotate · SPACE gripper · q = end+save episode.
"""
from __future__ import annotations

import argparse
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
    ap.add_argument("--horizon", type=int, default=3000)
    ap.add_argument("--pos-sensitivity", type=float, default=1.5)
    ap.add_argument("--rot-sensitivity", type=float, default=1.5)
    ap.add_argument("--save-fails", action="store_true")
    args = ap.parse_args()

    import mujoco
    import mujoco.viewer as mjv
    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action

    from experiments.libero_runner import make_libero_env

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # Pure-physics env: no robosuite renderer at all → only MuJoCo's passive viewer touches the GPU.
    env, lang, init = make_libero_env(task_suite=args.suite, task_idx=args.task, safety_level=args.level,
                                      has_renderer=False, has_offscreen_renderer=False,
                                      use_camera_obs=False, horizon=args.horizon)
    robot = env.env.robots[0]
    device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    env.reset()                                       # build the sim once (viewer binds to it)
    model = env.sim.model._model
    data = env.sim.data._data
    print(f'\nTELEOP  {args.suite} L{args.level} t{args.task}:  "{lang}"')
    print("  w/a/s/d move · r/f up-down · z/x t/g c/v rotate · SPACE gripper · q = end+save\n")

    def _success():
        try:
            return bool(env.check_success())
        except Exception:
            return False

    saved = 0
    with mjv.launch_passive(model, data) as viewer:
        for ep in args.episodes:
            if init is not None:
                env.set_init_state(init[ep])          # no sim rebuild → viewer stays valid
            for _ in range(20):
                env.step([0, 0, 0, 0, 0, 0, -1])
            viewer.sync()
            device.start_control()
            print(f"--- episode {ep}: GO (press q to finish/save) ---", flush=True)

            states, actions = [], []
            t = 0; ok = False
            while t < args.horizon and viewer.is_running():
                t0 = time.time()
                action, _grasp = input2action(device=device, robot=robot,
                                               active_arm="right", env_configuration="single-arm-opposed")
                if action is None:                    # q pressed → end episode
                    break
                states.append(env.get_sim_state().copy())
                env.step(action)
                actions.append(np.asarray(action, np.float32)[:7].copy())
                viewer.sync()
                t += 1
                if _success():
                    ok = True
                    print(f"  *** SUCCESS at step {t} ***", flush=True)
                    break
                time.sleep(max(0.0, 0.05 - (time.time() - t0)))

            if states and (ok or args.save_fails):
                fp = out / f"{args.suite}_L{args.level}_t{args.task}_ep{ep}_teleop.npz"
                np.savez_compressed(fp, states=np.asarray(states), actions=np.asarray(actions),
                                    success=ok, suite=args.suite, level=args.level, task=args.task,
                                    episode=ep, prompt=lang)
                saved += 1
                print(f"  saved {len(states)} steps → {fp.name}  (success={ok})", flush=True)
            else:
                print(f"  episode {ep}: not saved (success={ok}, steps={len(states)})", flush=True)
            if not viewer.is_running():
                break

    print(f"\n{saved} teleop recordings → {out}/   next: regenerate traces (offscreen):")
    print(f"  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.teleop_to_trace "
          f"--in {out} --suite {args.suite} --level {args.level} --task {args.task}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
