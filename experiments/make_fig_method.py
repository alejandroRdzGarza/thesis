"""make_fig_method.py — Figures for Chapter 3.

fig_spheres.pdf — the geometric model, drawn TO SCALE from the constants the shield actually uses,
not sketched. Two panels.
  (a) Why the obstacle is decomposed rather than bounded. A single bounding sphere over a
      non-convex object is a poor approximation: an earlier version of this work measured a 5.5 cm
      carton as 16 cm across, enough to distort the grasp it produced. Drawn at true relative size.
  (b) The three-sphere end-effector model in the y-z plane, with radii and offsets annotated. This
      is the panel Section 4.5 depends on: it shows how much more faithfully the end-effector is
      represented than an arm link, which is the fidelity argument for where residual collisions
      concentrate.

fig_shield_block.pdf — the shield as a data-flow diagram, and the one figure that carries the whole
thesis in one image. The policy proposes, the QP projects, the robot executes -- and the corrected
action is TAPPED OFF as the distillation target. At deployment the QP block is deleted and the tap
is what remains. Drawing the deletion explicitly is the point; a reader who sees only the runtime
path will not understand what is being claimed.

    PYTHONPATH=. python -m experiments.make_fig_method
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path("figures")

# End-effector sphere model, metres, in the end-effector frame (y, z).
PALM = dict(y=0.000, z=-0.056, r=0.048)
FINGERS = [dict(y=+0.036, z=-0.105, r=0.020), dict(y=-0.036, z=-0.105, r=0.020)]

# The bounding-sphere failure: a 5.5 cm carton measured as 16 cm across.
CARTON_W, CARTON_H = 0.055, 0.075
BOUNDING_D = 0.16

C_SPH, C_BAD, C_OBJ = "#2E6F9E", "#C0504D", "#8C8C8C"


def fig_spheres():
    fig, (axa, axb) = plt.subplots(1, 2, figsize=(10.4, 4.6))

    # ---- (a) bounding sphere vs decomposition -------------------------------------------
    axa.add_patch(Rectangle((-CARTON_W / 2, -CARTON_H / 2), CARTON_W, CARTON_H,
                            facecolor=C_OBJ, edgecolor="black", lw=1.0, alpha=0.85, zorder=3))
    axa.add_patch(Circle((0, 0), BOUNDING_D / 2, facecolor=C_BAD, alpha=0.16,
                         edgecolor=C_BAD, lw=1.6, ls="--", zorder=1))
    # a plausible 3-sphere decomposition of the same box
    for zc in (-CARTON_H / 2 + 0.018, 0.0, CARTON_H / 2 - 0.018):
        axa.add_patch(Circle((0, zc), 0.0285, facecolor=C_SPH, alpha=0.30,
                             edgecolor=C_SPH, lw=1.3, zorder=2))

    axa.annotate("bounding sphere\n16 cm across", xy=(BOUNDING_D / 2 * 0.72, 0.052),
                 xytext=(0.085, 0.085), fontsize=9.5, color=C_BAD,
                 arrowprops=dict(arrowstyle="->", color=C_BAD, lw=1.1))
    axa.annotate("object is 5.5 cm wide", xy=(CARTON_W / 2, -0.026), xytext=(0.062, -0.062),
                 fontsize=9.5, arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    axa.text(0, -0.105, "decomposition (blue) tracks the shape;\n"
                        "the bounding sphere (red) inflates it threefold",
             ha="center", fontsize=9, style="italic", color="#444444")

    axa.set_xlim(-0.13, 0.13); axa.set_ylim(-0.13, 0.13); axa.set_aspect("equal")
    axa.set_title("(a) Why the obstacle is decomposed", fontsize=11.5, pad=8)
    axa.axis("off")

    # ---- (b) the three-sphere end-effector, to scale --------------------------------------
    axb.plot([-0.012, 0.012, 0.012, -0.012, -0.012],
             [0.004, 0.004, -0.050, -0.050, 0.004], color="black", lw=1.2, zorder=4)
    axb.add_patch(Circle((PALM["y"], PALM["z"]), PALM["r"], facecolor=C_SPH, alpha=0.28,
                         edgecolor=C_SPH, lw=1.6, zorder=2))
    for f in FINGERS:
        axb.add_patch(Circle((f["y"], f["z"]), f["r"], facecolor=C_SPH, alpha=0.45,
                             edgecolor=C_SPH, lw=1.6, zorder=3))
        axb.plot([f["y"], f["y"]], [-0.050, f["z"] - 0.012], color="black", lw=2.2, zorder=1)

    axb.plot(0, 0, marker="x", ms=9, mew=2.0, color="black", zorder=5)
    axb.annotate("grasp site", xy=(0, 0), xytext=(0.045, 0.020), fontsize=9.5,
                 arrowprops=dict(arrowstyle="->", color="black", lw=1.0))
    axb.annotate(f"palm  $r={PALM['r']*100:.1f}$ cm\nat {abs(PALM['z'])*100:.1f} cm depth",
                 xy=(-PALM["r"] * 0.72, PALM["z"]), xytext=(-0.135, -0.030),
                 fontsize=9.5, color=C_SPH,
                 arrowprops=dict(arrowstyle="->", color=C_SPH, lw=1.1))
    axb.annotate(f"finger pads  $r={FINGERS[0]['r']*100:.1f}$ cm\n"
                 f"$\\pm{FINGERS[0]['y']*100:.1f}$ cm, {abs(FINGERS[0]['z'])*100:.1f} cm depth",
                 xy=(FINGERS[0]["y"] + 0.018, FINGERS[0]["z"]), xytext=(0.052, -0.128),
                 fontsize=9.5, color=C_SPH,
                 arrowprops=dict(arrowstyle="->", color=C_SPH, lw=1.1))

    axb.set_xlim(-0.15, 0.15); axb.set_ylim(-0.165, 0.055); axb.set_aspect("equal")
    axb.set_title("(b) The three-sphere end-effector model", fontsize=11.5, pad=8)
    axb.axis("off")

    fig.text(0.5, -0.02,
             "Both panels are drawn to scale from the constants the shield uses. "
             "Arm links (not shown) carry one radius each, sampled at three points along the "
             "link axis:\na coarser representation than the end-effector, which is where "
             "Section 4.5 locates the residual collisions.",
             ha="center", fontsize=9, style="italic")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_spheres.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def _box(ax, x, y, w, h, text, fc, ec="black", fs=9.5, ls="solid", alpha=1.0):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                facecolor=fc, edgecolor=ec, lw=1.4, ls=ls,
                                alpha=alpha, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, zorder=4, linespacing=1.35)


def _arrow(ax, p, q, color="black", ls="solid", lw=1.5, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=14,
                                 color=color, lw=lw, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}", zorder=2))


def fig_shield_block():
    fig, ax = plt.subplots(figsize=(10.6, 4.4))

    _box(ax, 0.01, 0.52, 0.15, 0.20, "observation\n$o_t$", "#EFEFEF")
    _box(ax, 0.21, 0.52, 0.16, 0.20, "$\\pi_{0.5}$", "#DCE7F0", fs=12)
    _box(ax, 0.43, 0.52, 0.17, 0.20, "CBF-QP\nprojection", "#F3D9D8")
    _box(ax, 0.67, 0.52, 0.15, 0.20, "robot", "#EFEFEF")
    _box(ax, 0.43, 0.10, 0.17, 0.20, "demonstration\nbuffer", "#DCE7F0")
    _box(ax, 0.21, 0.10, 0.16, 0.20, "behaviour\ncloning", "#DCE7F0")

    _arrow(ax, (0.16, 0.62), (0.21, 0.62))
    _arrow(ax, (0.37, 0.62), (0.43, 0.62))
    _arrow(ax, (0.60, 0.62), (0.67, 0.62))
    ax.text(0.40, 0.675, "$u_{\\mathrm{nom}}$", ha="center", fontsize=10)
    ax.text(0.635, 0.675, "$u^{\\star}$", ha="center", fontsize=10)

    # the tap
    _arrow(ax, (0.515, 0.52), (0.515, 0.30), color="#2E6F9E", lw=2.0)
    ax.text(0.535, 0.41, "the corrected action\nis the training target",
            fontsize=9.5, color="#2E6F9E", va="center")
    _arrow(ax, (0.43, 0.20), (0.37, 0.20), color="#2E6F9E", lw=2.0)
    _arrow(ax, (0.29, 0.30), (0.29, 0.52), color="#2E6F9E", lw=2.0, rad=0.0)
    ax.text(0.255, 0.41, "weights", fontsize=9.5, color="#2E6F9E",
            rotation=90, va="center", ha="right")

    # deployment: the QP is gone
    ax.add_patch(Rectangle((0.425, 0.505), 0.175, 0.23, fill=False,
                           edgecolor="#C0504D", lw=2.0, ls=(0, (4, 3)), zorder=5))
    ax.annotate("removed at deployment", xy=(0.60, 0.735), xytext=(0.70, 0.86),
                fontsize=10.5, color="#C0504D", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#C0504D", lw=1.6))

    ax.text(0.01, 0.93, "Training", fontsize=12, fontweight="bold")
    ax.text(0.01, 0.02,
            "The shield is present only while demonstrations are generated. What survives into "
            "deployment is the policy, not the apparatus.",
            fontsize=9.5, style="italic", color="#444444")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_shield_block.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def main() -> int:
    OUT.mkdir(exist_ok=True)
    fig_spheres()
    fig_shield_block()
    print(f"\n  end-effector model, to scale:")
    print(f"    palm      r = {PALM['r']*100:.1f} cm at z = {PALM['z']*100:.1f} cm")
    print(f"    fingers   r = {FINGERS[0]['r']*100:.1f} cm at y = "
          f"+/-{FINGERS[0]['y']*100:.1f} cm, z = {FINGERS[0]['z']*100:.1f} cm")
    print(f"    bounding-sphere failure: {CARTON_W*100:.1f} cm object -> "
          f"{BOUNDING_D*100:.0f} cm sphere ({BOUNDING_D/CARTON_W:.1f}x)")
    print(f"\n  -> {OUT}/fig_spheres.pdf")
    print(f"  -> {OUT}/fig_shield_block.pdf")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
