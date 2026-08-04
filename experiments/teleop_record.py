"""teleop_record.py — STAGE 1 of hand-demonstration: drive a task on-screen with the keyboard and
record (sim_state, action) per step. No offscreen renderer during teleop (that conflicts with the
on-screen window on macOS), and keys come from the render WINDOW (viewer callbacks), so no pynput /
Accessibility permission is needed. Follows the SafeLIBERO fork's proven collect_demonstration flow.

Then STAGE 2 (teleop_to_trace.py) replays the saved states offscreen to regenerate the 224² obs +
build a demo trace the distillation pipeline can consume.

Run ON THE MAC with plain python (NOT mjpython) — the on-screen window must have focus for keys:
  MUJOCO_GL= PYTHONPATH=. python -m experiments.teleop_record \
      --suite safelibero_spatial --level II --task 3 --episodes 0 1 2 --out demos_teleop

Controls (printed at start): w/a/s/d move XY · r/f up-down · z/x·t/g·c/v rotate · SPACE gripper ·
q = end+save the episode. Success is auto-detected.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--episodes", type=int, nargs="+", default=[0])
    ap.add_argument("--out", default="demos_teleop")
    ap.add_argument("--horizon", type=int, default=2000)
    ap.add_argument("--pos-sensitivity", type=float, default=1.5)
    ap.add_argument("--rot-sensitivity", type=float, default=1.5)
    ap.add_argument("--save-fails", action="store_true")
    args = ap.parse_args()

    from robosuite.devices import Keyboard
    from robosuite.utils.input_utils import input2action

    from experiments.libero_runner import make_libero_env

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    # On-screen only (no offscreen renderer → no macOS GL conflict); images regenerated in stage 2.
    env, lang, init = make_libero_env(task_suite=args.suite, task_idx=args.task, safety_level=args.level,
                                      has_renderer=True, has_offscreen_renderer=False,
                                      use_camera_obs=False, horizon=args.horizon)
    renv = env.env                                    # underlying robosuite env (render + viewer + robots)
    robot = renv.robots[0]
    device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
    renv.render()                                     # create the viewer
    renv.viewer.add_keypress_callback("any", device.on_press)
    renv.viewer.add_keyup_callback("any", device.on_release)
    renv.viewer.add_keyrepeat_callback("any", device.on_press)
    print(f'\nTELEOP  {args.suite} L{args.level} t{args.task}:  "{lang}"')
    print("  window keys: w/a/s/d move · r/f up-down · z/x t/g c/v rotate · SPACE gripper · q = end+save\n")

    def _success():
        try:
            return bool(env.check_success())
        except Exception:
            return False

    saved = 0
    for ep in args.episodes:
        env.reset()
        if init is not None:
            env.set_init_state(init[ep])
        for _ in range(20):                           # settle
            env.step([0, 0, 0, 0, 0, 0, -1])
        device.start_control()
        print(f"--- episode {ep}: GO (press q to finish/save) ---", flush=True)

        states, actions = [], []
        t = 0; ok = False
        while t < args.horizon:
            renv.render()
            action, _grasp = input2action(device=device, robot=robot,
                                          active_arm="right", env_configuration="single-arm-opposed")
            if action is None:                        # q pressed → end episode
                break
            states.append(env.get_sim_state().copy())  # sim state BEFORE the action (for stage-2 obs)
            env.step(action)
            actions.append(np.asarray(action, np.float32)[:7].copy())
            t += 1
            if _success():
                ok = True
                print(f"  *** SUCCESS at step {t} ***", flush=True)
                break

        if states and (ok or args.save_fails):
            fp = out / f"{args.suite}_L{args.level}_t{args.task}_ep{ep}_teleop.npz"
            np.savez_compressed(fp, states=np.asarray(states), actions=np.asarray(actions),
                                success=ok, suite=args.suite, level=args.level, task=args.task,
                                episode=ep, prompt=lang)
            saved += 1
            print(f"  saved {len(states)} steps → {fp.name}  (success={ok})", flush=True)
        else:
            print(f"  episode {ep}: not saved (success={ok}, steps={len(states)})", flush=True)

    print(f"\n{saved} teleop recordings → {out}/")
    print(f"Now regenerate demo traces (offscreen, can run on the pod):")
    print(f"  python -m experiments.teleop_to_trace --in {out} --suite {args.suite} "
          f"--level {args.level} --task {args.task}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
