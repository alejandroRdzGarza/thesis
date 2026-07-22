#!/usr/bin/env python
"""
convert_rwfm_to_lerobot.py — RWFM rollouts (+manifest) → weighted LeRobot dataset.

Reads a round's manifest.csv (traj_path, weight, advantage) and the per-rollout HDF5
trajectories saved by run_libero_trial(collect_cbf_data=True), and writes a LeRobot
dataset that STOCK openpi train.py consumes — no openpi code change needed.

Weighting modes (how the RWFM advantage enters training):
  --weight-mode filter    : DEFAULT. Keep only rollouts with advantage > --adv-threshold
                            (group-relative filtered BC = RAFT / hard-RWR). A legitimate,
                            citable method; zero openpi changes; the most stable v1.
  --weight-mode duplicate : write each trajectory round(weight * --dup-scale) times
                            (coarse integer oversampling; bloats epochs — weakest option).
  --weight-mode none      : write every rollout once (ablation / debugging).

The faithful RWR/AWR is the soft per-sample LOSS weighting (multiply each sample's
flow-matching loss by exp(A/τ)) — the intended MAIN method, implemented via the openpi
edits in experiments/RWFM_TRAINING.md and tested on-box. `filter` here is the v1 that
runs today with no openpi changes; the ablation filter→soft-weight is a clean result.

Run in the openpi/lerobot env (needs the `lerobot` package + h5py), e.g. on UCL:
    uv run python ../experiments/convert_rwfm_to_lerobot.py \
        --manifest results_rwfm/round0/manifest.csv \
        --repo-name safelibero_rwfm_round0 --weight-mode duplicate
"""
from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

CONTROL_HZ = 20
IMAGE_SHAPE = (224, 224, 3)
PROPRIO_DIM = 8
ACTION_DIM = 7
DEMO_KEY = "demo_0"


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """[x,y,z,w] → axis-angle (3,). Matches libero_runner._quat2axisangle."""
    q = np.asarray(quat, dtype=np.float64)
    w = min(1.0, max(-1.0, q[3]))
    den = np.sqrt(1.0 - w * w)
    if den < 1e-8:
        return np.zeros(3)
    angle = 2.0 * np.arccos(w)
    return (q[:3] / den) * angle


def _load_traj(h5_path: Path):
    """Return (images, wrists, states8, actions, instruction) or None."""
    import h5py
    if not h5_path.exists():
        return None
    with h5py.File(h5_path, "r") as f:
        grp = f[f"data/{DEMO_KEY}"]
        obs = grp["obs"]
        if "agentview_image" not in obs or "actions" not in grp:
            return None
        imgs   = obs["agentview_image"][:]
        wrists = (obs["robot0_eye_in_hand_image"][:]
                  if "robot0_eye_in_hand_image" in obs else imgs)
        eef_pos = obs["robot0_eef_pos"][:]
        eef_quat = (obs["robot0_eef_quat"][:] if "robot0_eef_quat" in obs
                    else np.tile([0, 0, 0, 1], (len(eef_pos), 1)))
        grip = (obs["robot0_gripper_qpos"][:] if "robot0_gripper_qpos" in obs
                else np.zeros((len(eef_pos), 2)))
        actions = grp["actions"][:]
        lang = grp.attrs.get("language_instruction", "pick and place the object")
        if isinstance(lang, bytes):
            lang = lang.decode()
    states = np.stack([
        np.concatenate([eef_pos[i], _quat2axisangle(eef_quat[i]), grip[i][:2]])
        for i in range(len(eef_pos))
    ]).astype(np.float32)
    return imgs, wrists, states, actions.astype(np.float32), str(lang)


def _copies_for(weight: float, advantage: float, mode: str,
                adv_threshold: float, dup_scale: float, dup_cap: int) -> int:
    if mode == "none":
        return 1
    if mode == "filter":
        return 1 if advantage > adv_threshold else 0
    if mode == "duplicate":
        return max(0, min(dup_cap, int(round(weight * dup_scale))))
    raise ValueError(mode)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--manifest", nargs="+", required=True,
                   help="one or more round manifest.csv files")
    p.add_argument("--repo-name", required=True)
    p.add_argument("--weight-mode", choices=["filter", "duplicate", "none"],
                   default="filter")
    p.add_argument("--adv-threshold", type=float, default=0.0,
                   help="filter mode: keep rollouts with advantage above this")
    p.add_argument("--dup-scale", type=float, default=2.0,
                   help="duplicate mode: copies = round(weight * dup_scale)")
    p.add_argument("--dup-cap", type=int, default=8,
                   help="duplicate mode: max copies per trajectory")
    p.add_argument("--push-to-hub", action="store_true")
    args = p.parse_args()

    from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

    out = HF_LEROBOT_HOME / args.repo_name
    if out.exists():
        print(f"Removing existing dataset at {out}")
        shutil.rmtree(out)

    ds = LeRobotDataset.create(
        repo_id=args.repo_name, robot_type="panda", fps=CONTROL_HZ,
        features={
            "image":       {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "wrist_image": {"dtype": "image", "shape": IMAGE_SHAPE, "names": ["height", "width", "channel"]},
            "state":       {"dtype": "float32", "shape": (PROPRIO_DIM,), "names": ["state"]},
            "actions":     {"dtype": "float32", "shape": (ACTION_DIM,), "names": ["actions"]},
        },
        image_writer_threads=10, image_writer_processes=5,
    )

    rows = []
    for m in args.manifest:
        with open(m) as f:
            for r in csv.DictReader(f):
                rows.append(r)

    n_written = n_frames = n_skipped = 0
    for r in rows:
        traj = _load_traj(Path(r["traj_path"]))
        if traj is None:
            n_skipped += 1
            continue
        copies = _copies_for(
            float(r["weight"]), float(r["advantage"]), args.weight_mode,
            args.adv_threshold, args.dup_scale, args.dup_cap)
        if copies <= 0:
            n_skipped += 1
            continue
        imgs, wrists, states, actions, lang = traj
        T = len(imgs)
        for _ in range(copies):
            for i in range(T):
                ds.add_frame({
                    "image": imgs[i], "wrist_image": wrists[i],
                    "state": states[i], "actions": actions[i], "task": lang,
                })
            ds.save_episode()
            n_written += 1
            n_frames += T

    print(f"\nRWFM dataset '{args.repo_name}': {n_written} episodes "
          f"({n_frames} frames) from {len(rows)} rollouts "
          f"({n_skipped} skipped, mode={args.weight_mode})")
    print(f"  → {out}")
    if args.push_to_hub:
        ds.push_to_hub()


if __name__ == "__main__":
    main()
