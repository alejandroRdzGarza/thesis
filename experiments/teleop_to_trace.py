"""teleop_to_trace.py — STAGE 2: turn teleop (state,action) recordings into demo traces the
distillation pipeline consumes. Offscreen (no window) — runs headless on the Mac or the pod.

For each <in>/*_teleop.npz (from teleop_record.py) it regenerates the 224² agentview+wrist obs and
8-D proprio from the saved sim states (env.regenerate_obs_from_state) and writes a *_trace.npz +
appends manifest.csv — identical format to collect_classical_demos.

  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.teleop_to_trace \
      --in demos_teleop --suite safelibero_spatial --level II --task 3 --out demos_teleop
Then mix into the distill:  flow_bc_train --round demos_scripted demos_teleop
"""
from __future__ import annotations

import argparse
import csv
import glob
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", default="demos_teleop", help="dir with *_teleop.npz")
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=3)
    ap.add_argument("--out", default=None, help="output dir for traces (default: same as --in)")
    ap.add_argument("--replan", type=int, default=5, help="record one query obs every N steps (match training)")
    args = ap.parse_args()

    from experiments.libero_runner import make_libero_env, _preprocess, _build_proprio
    from experiments.policy_trace import QueryTrace, save_episode_trace

    inp = Path(args.inp)
    out = Path(args.out) if args.out else inp
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(str(inp / f"{args.suite}_L{args.level}_t{args.task}_ep*_teleop.npz")))
    if not files:
        raise SystemExit(f"no *_teleop.npz for {args.suite} L{args.level} t{args.task} under {inp}")

    env, lang, _ = make_libero_env(task_suite=args.suite, task_idx=args.task, safety_level=args.level,
                                   has_renderer=False, has_offscreen_renderer=True, use_camera_obs=True)
    env.reset()

    def _obs_query(obs):
        img = _preprocess(obs["agentview_image"]) if "agentview_image" in obs else np.zeros((224, 224, 3), np.uint8)
        wri = _preprocess(obs["robot0_eye_in_hand_image"]) if "robot0_eye_in_hand_image" in obs else np.zeros((224, 224, 3), np.uint8)
        return {"image": np.asarray(img, np.uint8), "wrist_image": np.asarray(wri, np.uint8),
                "state": np.asarray(_build_proprio(obs), np.float32), "prompt": lang}

    rows = []
    for f in files:
        d = np.load(f, allow_pickle=True)
        states, actions = d["states"], d["actions"]
        ok = bool(d["success"]); ep = int(d["episode"])
        queries, bufs = [], []
        for t in range(len(actions)):
            if t % args.replan == 0:
                obs = env.regenerate_obs_from_state(states[t])
                queries.append(QueryTrace(chain=np.zeros((2, 1, 1), np.float32),
                                          logp_old=np.zeros(1, np.float32),
                                          sigmas=np.array([1.0, 0.0], np.float32),
                                          noise_level=0.0, sde_type="teleop", obs=_obs_query(obs)))
                bufs.append([])
            bufs[-1].append(np.asarray(actions[t], np.float32))
        for q, b in zip(queries, bufs):
            if b:
                q.shielded_actions = np.asarray(b, np.float32)
        usable = [q for q in queries if q.shielded_actions is not None]
        tp = str((out / f"{args.suite}_L{args.level}_t{args.task}_ep{ep}_teleop_trace.npz").resolve())
        save_episode_trace(usable, tp)
        rows.append({"trace_path": tp, "r_success": 1.5 if ok else 0.0, "robot_caused_collision": 0,
                     "suite": args.suite, "task": args.task, "episode": ep})
        print(f"  ep{ep}: {len(usable)} queries → {Path(tp).name}  (success={ok})", flush=True)

    mpath = out / "manifest.csv"
    exists = mpath.exists()
    with open(mpath, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["trace_path", "r_success", "robot_caused_collision",
                                           "suite", "task", "episode"])
        if not exists:
            w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} teleop traces → {mpath}")
    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    main()
