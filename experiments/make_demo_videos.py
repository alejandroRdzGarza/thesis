"""make_demo_videos.py — side-by-side MP4s of paired episodes, for a project page.

NOT for the thesis PDF. Embedded video in PDF plays only in Adobe Acrobat and is blank in Preview,
in browsers and in print, so the document itself uses the still filmstrip of
`make_fig_filmstrip.py`. These files are for a supplementary link, and for showing the result to
someone in ten seconds rather than by reading a table.

Each video puts the base policy on the left and the self-distilled policy on the right, on the SAME
scene and the SAME held-out initial state, with the outcome burned into a header strip so a clip
that gets separated from its caption still says what it shows.

Episodes differ in length -- a failing rollout runs to the horizon while a succeeding one stops
early -- so the shorter is held on its final frame rather than looping or being cut, and the header
says so. Cutting the longer one instead would hide the base policy still flailing after the
distilled policy has finished.

    PYTHONPATH=. python -m experiments.make_demo_videos --all
    PYTHONPATH=. python -m experiments.make_demo_videos --scene safelibero_object_LI_t0 --group 4
"""

from __future__ import annotations

import argparse
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from experiments.make_fig_filmstrip import DIST_ARM, BASE_ARM, candidates, load_manifest

OUT = Path("figures/demo_videos")
FPS = 8
SCALE = 2          # 224 -> 448 per panel
HEADER = 54
GREEN, RED, INK = (46, 125, 50), (192, 80, 77), (30, 30, 30)


def load_frames(path: str) -> np.ndarray:
    return np.load(path, allow_pickle=True)["obs_image"]


def compose(base: np.ndarray, dist: np.ndarray, meta_b: dict, meta_d: dict,
            scene: str, group: str) -> list[Image.Image]:
    n = max(len(base), len(dist))
    w = base.shape[2] * SCALE
    out = []
    for i in range(n):
        # hold the shorter episode on its last frame; truncating the longer one would hide the
        # base policy still failing after the distilled policy has finished
        fb = base[min(i, len(base) - 1)]
        fd = dist[min(i, len(dist) - 1)]
        canvas = Image.new("RGB", (w * 2 + 6, w + HEADER), "white")
        for k, (f, x) in enumerate(((fb, 0), (fd, w + 6))):
            canvas.paste(Image.fromarray(f).resize((w, w), Image.NEAREST), (x, HEADER))
        d = ImageDraw.Draw(canvas)
        for (meta, name, x) in ((meta_b, "Base pi0.5", 0), (meta_d, "Self-distilled", w + 6)):
            ok = not meta["collision"]
            col = GREEN if ok else RED
            d.rectangle([x, HEADER - 4, x + w, HEADER], fill=col)
            d.text((x + 8, 6), name, fill=INK)
            d.text((x + 8, 24),
                   f"{'collision-free' if ok else 'COLLISION'}   "
                   f"{'success' if meta['success'] else 'no success'}", fill=col)
        d.text((8, w + HEADER - 16), f"{scene}  init {group}   step {i*5}", fill=INK)
        out.append(canvas)
    return out


def render(scene: str, group: str, quiet: bool = False) -> Path:
    b = load_manifest(BASE_ARM)[(scene, group)]
    dd = load_manifest(DIST_ARM)[(scene, group)]
    frames = compose(load_frames(b["trace"]), load_frames(dd["trace"]), b, dd, scene, group)
    OUT.mkdir(parents=True, exist_ok=True)
    dest = OUT / f"{scene}_g{group}.mp4"
    with tempfile.TemporaryDirectory() as td:
        for i, im in enumerate(frames):
            im.save(f"{td}/{i:04d}.png")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FPS),
             "-i", f"{td}/%04d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", str(dest)],
            check=True)
    if not quiet:
        print(f"  {dest}  ({len(frames)} frames, {len(frames)/FPS:.1f}s)")
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--scene"); ap.add_argument("--group")
    a = ap.parse_args()

    if a.all:
        ideal = [r for r in candidates()
                 if r[1]["collision"] and not r[2]["collision"] and r[2]["success"]]
        for i, ((sc, g), _, _) in enumerate(ideal, 1):
            render(sc, g, quiet=True)
            print(f"  [{i:>2}/{len(ideal)}] {sc}_g{g}.mp4", flush=True)
        print(f"\n  {len(ideal)} videos -> {OUT}/")
        return 0

    if not (a.scene and a.group):
        ap.error("pass --all, or both --scene and --group")
    render(a.scene, a.group)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
