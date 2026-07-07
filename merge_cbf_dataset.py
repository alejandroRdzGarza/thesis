"""
Merge per-episode CBF HDF5 files into a single LIBERO-compatible dataset
for OpenVLA-OFT fine-tuning.

Usage:
    python merge_cbf_dataset.py                         # defaults
    python merge_cbf_dataset.py --input results_cbf_dataset/cbf_dataset --output cbf_finetune_data.h5
    python merge_cbf_dataset.py --cbf-only              # drop steps where CBF never fired
    python merge_cbf_dataset.py --min-h 0.5             # keep only steps where h < 0.5
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import h5py
import numpy as np


def merge(
    input_dir: str,
    output_path: str,
    cbf_only: bool = False,
    min_h: float | None = None,
    min_steps: int = 10,
) -> None:
    files = sorted(glob.glob(f"{input_dir}/**/*.h5", recursive=True))
    if not files:
        print(f"No .h5 files found under {input_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(files)} episode files under {input_dir}")

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    demo_idx = 0
    skipped = 0
    total_steps = 0

    with h5py.File(out_path, "w") as out:
        data_grp = out.require_group("data")

        for fpath in files:
            with h5py.File(fpath, "r") as src:
                if "data/demo_0" not in src:
                    skipped += 1
                    continue

                src_demo = src["data/demo_0"]
                n_steps = src_demo.attrs.get("num_steps", 0)

                if n_steps < min_steps:
                    skipped += 1
                    continue

                # optional filtering: keep only steps where CBF was active / h was low
                if cbf_only or min_h is not None:
                    cbf_mask = src_demo["cbf_triggered"][:].astype(bool)
                    h_vals   = src_demo["h_values"][:]

                    mask = np.ones(n_steps, dtype=bool)
                    if cbf_only:
                        mask &= cbf_mask
                    if min_h is not None:
                        mask &= (h_vals < min_h)

                    if mask.sum() < min_steps:
                        skipped += 1
                        continue

                    dest_key = f"demo_{demo_idx}"
                    dest = data_grp.require_group(dest_key)

                    obs_dest = dest.require_group("obs")
                    obs_src  = src_demo["obs"]
                    for key in obs_src:
                        obs_dest.create_dataset(
                            key,
                            data=obs_src[key][mask],
                            compression="lzf" if "image" in key else None,
                        )

                    for key in ("actions", "vla_actions", "cbf_triggered", "h_values"):
                        if key in src_demo:
                            dest.create_dataset(key, data=src_demo[key][mask])

                    dest.attrs["language_instruction"] = src_demo.attrs.get(
                        "language_instruction", ""
                    )
                    dest.attrs["num_steps"] = int(mask.sum())

                else:
                    # copy entire demo as-is
                    src.copy("data/demo_0", data_grp, name=f"demo_{demo_idx}")

                total_steps += int(src_demo.attrs.get("num_steps", n_steps))
                demo_idx += 1

        data_grp.attrs["num_demos"] = demo_idx
        data_grp.attrs["env_name"]  = "safelibero"

    print(f"Merged {demo_idx} demos ({total_steps} steps) → {out_path}")
    if skipped:
        print(f"Skipped {skipped} files (too short or no matching steps)")


def main():
    p = argparse.ArgumentParser(description="Merge CBF episode HDF5 files for OpenVLA-OFT fine-tuning")
    p.add_argument("--input",    default="results_cbf_dataset/cbf_dataset",
                   help="Root directory containing per-episode .h5 files (searched recursively)")
    p.add_argument("--output",   default="cbf_finetune_data.h5",
                   help="Output merged HDF5 path")
    p.add_argument("--cbf-only", action="store_true",
                   help="Keep only timesteps where CBF actually fired (cbf_triggered=True)")
    p.add_argument("--min-h",    type=float, default=None,
                   help="Keep only timesteps where h_value < MIN_H (CBF was constraining the arm)")
    p.add_argument("--min-steps", type=int, default=10,
                   help="Skip episodes with fewer than MIN_STEPS steps (default: 10)")
    args = p.parse_args()

    merge(
        input_dir=args.input,
        output_path=args.output,
        cbf_only=args.cbf_only,
        min_h=args.min_h,
        min_steps=args.min_steps,
    )


if __name__ == "__main__":
    main()
