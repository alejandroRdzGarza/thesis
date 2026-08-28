"""make_fig_filmstrip.py — Figure 4.3: the same scene under the base and distilled policies.

WHY THIS IS BUILDABLE WITHOUT RE-RUNNING ANYTHING. Every evaluation rollout stored its full
observation stream (`obs_image`, 224x224 RGB per policy query) alongside a manifest recording
`robot_caused_collision` and `r_success` per episode. The base and round-1 arms were evaluated on
the same scenes and the same held-out initial states, indexed identically by `group_id`. Matching
on (scene, group_id) therefore yields paired episodes that differ only in the policy -- 120 of
them, of which 68 have the base colliding while the distilled policy stays clean and succeeds.

The `videos/` directory is NOT the source. Those three clips carry no record of which policy, scene
or initial state produced them, which is why the figure was blocked before this was found.

FRAME SELECTION. Episodes differ in length -- a failing rollout runs to the horizon while a
succeeding one stops early -- so frames are sampled at equal FRACTIONS of each episode rather than
at equal step indices. Both rows then show the whole arc, and the caption must say so; sampling at
matched step indices would truncate the shorter episode and misrepresent it.

    # inventory the candidates, best first
    PYTHONPATH=. python -m experiments.make_fig_filmstrip --list

    # build the figure for one pair
    PYTHONPATH=. python -m experiments.make_fig_filmstrip \
        --scene safelibero_goal_LI_t0 --group 0
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "pod_backup/results_shielded"
BASE_ARM, DIST_ARM = "eval_base_nocbf", "eval_r1_nocbf"
OUT = Path("figures")


def load_manifest(arm: str) -> dict:
    """(scene, group_id) -> dict with collision, success and trace path."""
    out = {}
    for m in glob.glob(f"{ROOT}/{arm}/*/manifest.csv"):
        scene = os.path.basename(os.path.dirname(m))
        for row in csv.DictReader(open(m)):
            out[(scene, row["group_id"])] = {
                "collision": int(row["robot_caused_collision"]),
                "success": float(row["r_success"]) > 0,
                "trace": row["trace_path"],
                "ets": row.get("ets", ""),
            }
    return out


def candidates():
    b, d = load_manifest(BASE_ARM), load_manifest(DIST_ARM)
    rows = []
    for k in sorted(set(b) & set(d)):
        rows.append((k, b[k], d[k]))
    # best first: base collides, distilled clean and successful, and base ALSO failed the task
    # (both failure modes visible in one figure)
    rows.sort(key=lambda r: (
        -(r[1]["collision"] == 1 and r[2]["collision"] == 0 and r[2]["success"]),
        -(not r[1]["success"]),
    ))
    return rows


def frames(trace_path: str, n: int) -> tuple[np.ndarray, np.ndarray]:
    """n frames sampled at equal fractions of the episode, with their step indices."""
    d = np.load(trace_path, allow_pickle=True)
    imgs = d["obs_image"]
    idx = np.linspace(0, len(imgs) - 1, n).round().astype(int)
    return imgs[idx], idx


TICK, CROSS = "\u2713", "\u2717"


def row_label(name: str, collision_free: bool, success: bool) -> str:
    """Outcome goes in the row label rather than as free text. Text anchored to a single-image
    axis but extending past it makes tight_layout reserve space for the overflow and shrink every
    panel, which is what a first attempt at side-by-side badges did."""
    return (f"{name}\n"
            f"{TICK if collision_free else CROSS} collision-free   "
            f"{TICK if success else CROSS} success")


def build(scene: str, group: str, n: int, stem: str = "fig_filmstrip",
          exts=("pdf", "png"), quiet: bool = False):
    b, d = load_manifest(BASE_ARM)[(scene, group)], load_manifest(DIST_ARM)[(scene, group)]
    fb, ib = frames(b["trace"], n)
    fd, idd = frames(d["trace"], n)

    fig, axes = plt.subplots(2, n, figsize=(2.0 * n, 4.6))
    for row, (imgs, idx, meta, name) in enumerate(
            [(fb, ib, b, "Base $\\pi_{0.5}$"), (fd, idd, d, "Self-distilled")]):
        for col in range(n):
            ax = axes[row, col]
            ax.imshow(imgs[col]); ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_edgecolor("#C0504D" if meta["collision"] else "#2E7D32")
                s.set_linewidth(2.0)
            if row == 0:
                ax.set_title(f"step {idx[col]*5}", fontsize=9, pad=4)
        axes[row, 0].set_ylabel(name, fontsize=11, labelpad=8)
        ok = not meta["collision"]
        fig.text(0.5, 0.505 - 0.485 * row,
                 f"{TICK if ok else CROSS} collision-free    "
                 f"{TICK if meta['success'] else CROSS} task success",
                 ha="center", va="top", fontsize=10.5, fontweight="bold",
                 color="#2E7D32" if ok else "#C0504D")

    fig.suptitle(scene.replace("safelibero_", "").replace("_", " ")
                 + f"   |   held-out initial state {group}", fontsize=11.5, y=0.98)
    fig.tight_layout(rect=[0, 0.04, 1, 0.95])
    fig.subplots_adjust(hspace=0.30)
    OUT.mkdir(exist_ok=True)
    for ext in exts:
        fig.savefig(OUT / f"{stem}.{ext}", bbox_inches="tight", dpi=160)
    plt.close(fig)
    if quiet:
        return
    print(f"\n  {scene}  g{group}")
    print(f"    base       collision={b['collision']}  success={b['success']}  ets={b['ets']}")
    print(f"    distilled  collision={d['collision']}  success={d['success']}  ets={d['ets']}")
    print(f"\n  -> {OUT}/fig_filmstrip.pdf")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="inventory paired episodes, best first")
    ap.add_argument("--scene"); ap.add_argument("--group")
    ap.add_argument("--frames", type=int, default=5)
    ap.add_argument("--all", action="store_true",
                    help="render every ideal pair to figures/filmstrip_candidates/ for browsing")
    a = ap.parse_args()

    rows = candidates()

    if a.all:
        ideal = [r for r in rows if r[1]["collision"] and not r[2]["collision"] and r[2]["success"]]
        d = OUT / "filmstrip_candidates"; d.mkdir(parents=True, exist_ok=True)
        for i, ((sc, g), _, _) in enumerate(ideal, 1):
            build(sc, g, a.frames, stem=f"filmstrip_candidates/{sc}_g{g}",
                  exts=("png",), quiet=True)
            print(f"  [{i:>2}/{len(ideal)}] {sc}_g{g}.png", flush=True)
        print(f"\n  {len(ideal)} candidates -> {d}/")
        print("  browse, then rebuild the one you want:")
        print("    python -m experiments.make_fig_filmstrip --scene <scene> --group <n>")
        return 0

    if a.list or not (a.scene and a.group):
        ideal = [r for r in rows if r[1]["collision"] and not r[2]["collision"] and r[2]["success"]]
        print(f"\n  {len(rows)} paired (scene, init); {len(ideal)} with base colliding and "
              f"distilled clean + successful\n")
        print(f"  {'scene':30s}{'init':>5s}   {'base':>16s}   {'distilled':>16s}")
        for (sc, g), bb, dd in ideal[:20]:
            print(f"  {sc:30s}{g:>5s}   "
                  f"{'collides':>8s} {'fails' if not bb['success'] else 'succeeds':>7s}   "
                  f"{'clean':>8s} {'succeeds':>7s}")
        print("\n  pick one and pass --scene ... --group ...")
        return 0

    build(a.scene, a.group, a.frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
