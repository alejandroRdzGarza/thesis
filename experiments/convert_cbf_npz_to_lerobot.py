"""
Convert CBF-corrected DAgger episodes (collect_obstacle_data.py output) to LeRobot format
for fine-tuning π0.5 with the openpi training pipeline.

The converter reads all ep_*.npz files under --data-dir (recursively) and writes a
LeRobot dataset that the openpi train.py script can consume directly.

Action target: `cbf_act` by default (CBF-corrected actions — what we want the model to
imitate). Pass --action-key nom_act to use nominal VLA actions instead (ablation).

Usage
-----
  # Convert all episodes (successful and failed):
  python -m experiments.convert_cbf_npz_to_lerobot --data-dir data/dagger_round0

  # Only successful episodes (recommended for DAgger):
  python -m experiments.convert_cbf_npz_to_lerobot --data-dir data/dagger_round0 --success-only

  # Combine multiple collection rounds:
  python -m experiments.convert_cbf_npz_to_lerobot \
      --data-dir data/dagger_round0 data/dagger_round1 \
      --repo-name pi05_libero_cbf_dagger_r01

Then run in the openpi environment:
  cd openpi
  uv run scripts/compute_norm_stats.py --config-name pi05_libero_cbf
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_cbf \
      --exp-name dagger_round0 --overwrite
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

REPO_NAME = "pi05_libero_cbf_dagger"

IMAGE_SHAPE    = (224, 224, 3)   # matches _preprocess() output in collect_obstacle_data.py
PROPRIO_DIM    = 8               # eef_pos(3) + axis_angle(3) + gripper_qpos(2)
ACTION_DIM     = 7               # xyz(3) + axis_angle(3) + gripper(1)
CONTROL_HZ     = 20              # LIBERO OSC_POSE control frequency


def main(
    data_dirs:    list[str],
    repo_name:    str  = REPO_NAME,
    success_only: bool = False,
    action_key:   str  = "cbf_act",
    push_to_hub:  bool = False,
) -> None:
    try:
        from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
    except ImportError:
        raise RuntimeError(
            "lerobot not installed in this environment.\n"
            "Run this script inside the openpi uv environment:\n"
            "  cd openpi && uv run python ../experiments/convert_cbf_npz_to_lerobot.py ..."
        )

    # Gather all episode files across all input directories
    npz_files: list[Path] = []
    for d in data_dirs:
        found = sorted(Path(d).rglob("ep_*.npz"))
        print(f"  {d}: {len(found)} episode files")
        npz_files.extend(found)

    if not npz_files:
        print("No ep_*.npz files found — check --data-dir paths.")
        return

    print(f"\nTotal episodes found: {len(npz_files)}")

    # Wipe any previous version of this dataset
    output_path = HF_LEROBOT_HOME / repo_name
    if output_path.exists():
        print(f"Removing existing dataset at {output_path}")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="panda",
        fps=CONTROL_HZ,
        features={
            "image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": IMAGE_SHAPE,
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (PROPRIO_DIM,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (ACTION_DIM,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    n_ep_written = 0
    n_ep_skipped = 0
    n_steps_total = 0

    for npz_path in npz_files:
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            print(f"  [skip] {npz_path.name}: load error — {e}")
            n_ep_skipped += 1
            continue

        # Success filter — skip failed episodes (bad training signal)
        if success_only:
            success_val = data.get("success", None)
            if success_val is None or int(success_val.flat[0]) == 0:
                n_ep_skipped += 1
                continue

        # Grasp filter — skip episodes where robot never grasped the object.
        # These episodes contain only approach + obstacle avoidance, never task
        # completion, so fine-tuning on them teaches avoidance without reward.
        grasp_val = data.get("grasp_step", None)
        if grasp_val is not None and int(grasp_val.flat[0]) == -1:
            n_ep_skipped += 1
            continue

        imgs     = data["img"]          # (T, H, W, 3) uint8
        wrists   = data["wrist"]        # (T, H, W, 3) uint8
        proprios = data["proprio"]      # (T, 8) float32
        actions  = data[action_key]     # (T, 7) float32  ← cbf_act or nom_act
        lang_arr = data.get("lang", np.array(["pick and place the object"]))
        lang     = str(lang_arr.flat[0])
        T        = len(imgs)

        if T == 0:
            n_ep_skipped += 1
            continue

        for i in range(T):
            dataset.add_frame({
                "image":       imgs[i],
                "wrist_image": wrists[i],
                "state":       proprios[i].astype(np.float32),
                "actions":     actions[i].astype(np.float32),
                "task":        lang,
            })

        dataset.save_episode()
        n_ep_written += 1
        n_steps_total += T

        if n_ep_written % 20 == 0:
            print(f"  {n_ep_written} episodes written ({n_steps_total} steps) ...")

    print(f"\n{'='*55}")
    print(f"Conversion complete")
    print(f"  Episodes written : {n_ep_written}")
    print(f"  Episodes skipped : {n_ep_skipped}")
    print(f"  Total steps      : {n_steps_total}")
    print(f"  Action key       : {action_key}")
    print(f"  Dataset saved to : {output_path}")
    print(f"{'='*55}")
    print(f"\nNext steps (run inside openpi/):")
    print(f"  uv run scripts/compute_norm_stats.py --config-name pi05_libero_cbf")
    print(f"  XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero_cbf \\")
    print(f"      --exp-name dagger_round0 --overwrite")

    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "cbf-dagger", "safety"],
            private=True,
            push_videos=True,
            license="apache-2.0",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert CBF DAgger npz episodes to LeRobot dataset for π0.5 fine-tuning"
    )
    parser.add_argument("--data-dir", nargs="+", required=True,
                        help="One or more directories containing ep_*.npz files")
    parser.add_argument("--repo-name", default=REPO_NAME,
                        help="LeRobot dataset repo ID (used as output folder name)")
    parser.add_argument("--success-only", action="store_true",
                        help="Only include episodes where the task succeeded")
    parser.add_argument("--action-key", choices=["cbf_act", "nom_act"], default="cbf_act",
                        help="cbf_act (default): train on CBF-corrected actions; "
                             "nom_act: train on raw VLA actions (ablation)")
    parser.add_argument("--push-to-hub", action="store_true",
                        help="Push the resulting dataset to the Hugging Face Hub")
    args = parser.parse_args()

    main(
        data_dirs=args.data_dir,
        repo_name=args.repo_name,
        success_only=args.success_only,
        action_key=args.action_key,
        push_to_hub=args.push_to_hub,
    )
