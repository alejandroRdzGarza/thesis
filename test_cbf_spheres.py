"""
Visual test for sphere-decomposition CBF — no VLA / GPU required.

Loads a SafeLIBERO environment, detects the obstacle, decomposes it into a
sphere cloud, then drives the arm with three scripted phases so you can see
the CBF in action:

  Phase 1 (0 – T/3)  : move straight toward the obstacle
                        → CBF deflects the arm around the obstacle
  Phase 2 (T/3 – 2T/3): lateral sweep past the obstacle
                        → sphere cloud lights up orange/red on close spheres
  Phase 3 (2T/3 – T)  : retreat to approximate start

Things to check visually:
  • 48 small spheres tracing the obstacle surface (blue → orange → red)
  • 3 cyan spheres per arm link (instead of 1 at body origin)
  • Green EE sphere turning yellow when CBF fires
  • Arm path curves around the obstacle during phase 1

Usage:
  conda activate libero
  python test_cbf_spheres.py --suite safelibero_spatial --task 0 --level I
  python test_cbf_spheres.py --suite safelibero_spatial --task 1 --level II --save-video out.mp4
"""

from __future__ import annotations

import argparse
import time

import cv2
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from experiments.cbf_ellipsoid import (
    K_CBF, EE_Q_DIAG_DEFAULT, EE_SPHERE_RADIUS,
    ee_ellipsoid_center, run_sphere_decomp_cbf,
)
from experiments.cbf_visualizer import (
    decompose_obstacle_to_spheres, install_scene_hook, push_cbf_geoms,
)
from experiments.libero_runner import (
    _ARM_LINK_CBF_RADII, _CV2_WINDOW, _DUMMY_ACTION,
    _WARMUP_STEPS, _compute_arm_link_constraints,
    _get_arm_body_ids, _get_arm_dof_indices,
    _link_sample_positions, _render_frame, _unwrap_sim,
    detect_safelibero_obstacle, make_libero_env,
)

_DISPLAY_SCALE = 2
_STATUS_BAR_H  = 64


def _preprocess(img: np.ndarray) -> np.ndarray:
    return img[::-1, ::-1].copy()


def _scripted_action(
    t: int, horizon: int,
    ee_pos: np.ndarray,
    ob_pos: np.ndarray,
    ee_start: np.ndarray,
    speed: float = 0.012,
) -> np.ndarray:
    """Three-phase scripted arm motion — no VLA needed."""
    phase = t / horizon

    if phase < 0.40:
        # Phase 1: drive straight toward the obstacle centre
        d = ob_pos - ee_pos
        n = np.linalg.norm(d)
        delta = (d / n * speed) if n > 0.01 else np.zeros(3)
    elif phase < 0.70:
        # Phase 2: lateral sweep — circle around obstacle in XY plane
        ang = (phase - 0.40) / 0.30 * 2 * np.pi
        delta = np.array([speed * np.cos(ang), speed * np.sin(ang), 0.0])
    else:
        # Phase 3: retreat toward start position
        d = ee_start - ee_pos
        n = np.linalg.norm(d)
        delta = (d / n * speed * 0.6) if n > 0.02 else np.zeros(3)

    return np.array([delta[0], delta[1], delta[2], 0.0, 0.0, 0.0, -1.0])


def main() -> None:
    ap = argparse.ArgumentParser(description="CBF sphere decomposition visual test")
    ap.add_argument("--suite",      default="safelibero_spatial",
                    choices=["safelibero_spatial", "safelibero_object",
                             "safelibero_goal",    "safelibero_long"])
    ap.add_argument("--task",       type=int,   default=0)
    ap.add_argument("--level",      default="I", choices=["I", "II"])
    ap.add_argument("--episode",    type=int,   default=0)
    ap.add_argument("--horizon",    type=int,   default=350)
    ap.add_argument("--sphere-n",   type=int,   default=48,
                    help="Obstacle surface spheres (default 48)")
    ap.add_argument("--speed",      type=float, default=0.012,
                    help="Scripted action magnitude per step (m)")
    ap.add_argument("--save-video", default=None,
                    help="Path to save MP4 (e.g. out.mp4)")
    ap.add_argument("--no-cbf",     action="store_true",
                    help="Disable CBF so you can compare deflected vs raw path")
    args = ap.parse_args()

    # ── Environment ───────────────────────────────────────────────────────
    print(f"\nLoading {args.suite} task {args.task} level {args.level} …")
    env, lang, init_states = make_libero_env(
        task_suite=args.suite, task_idx=args.task,
        safety_level=args.level, has_renderer=False, horizon=args.horizon,
    )
    print(f"  Task: {lang}")

    env.reset()
    if init_states is not None:
        obs = env.set_init_state(init_states[args.episode])
    else:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

    model, data = _unwrap_sim(env)
    arm_body_ids = _get_arm_body_ids(model)
    arm_dof_idx  = _get_arm_dof_indices(model)
    ee_body_id   = arm_body_ids[-1] if arm_body_ids else 0
    print(f"  Arm bodies: {len(arm_body_ids)}  DOFs: {arm_dof_idx}")

    # ── Warm-up ────────────────────────────────────────────────────────────
    for _ in range(_WARMUP_STEPS):
        step_out = env.step(_DUMMY_ACTION.tolist())
        obs = step_out[0] if isinstance(step_out, tuple) else step_out

    # ── Obstacle detection ─────────────────────────────────────────────────
    obstacle = detect_safelibero_obstacle(env, obs, safety_radius=0.10)
    obstacles = [obstacle] if obstacle else []

    if obstacle:
        print(f"\n  Obstacle: '{obstacle.name}'  pos={np.round(obstacle.pos, 3)}")
    else:
        print("\n  [warn] No obstacle detected — running without CBF")

    # ── Sphere decomposition ───────────────────────────────────────────────
    sphere_decomp: list | None = None
    if obstacle and not args.no_cbf:
        for suffix in ("_main", ""):
            sphere_decomp = decompose_obstacle_to_spheres(
                model, data, f"{obstacle.name}{suffix}",
                n_spheres=args.sphere_n, r_sphere=0.010, safety_margin=0.010,
            )
            if sphere_decomp is not None:
                break
        if sphere_decomp is not None:
            print(f"  Sphere decomp: {len(sphere_decomp)} spheres")
        else:
            print("  [warn] Sphere decomp failed — CBF inactive")

    # ── Render hook ────────────────────────────────────────────────────────
    viz_ok = install_scene_hook(env.sim)
    print(f"  Render hook: {'installed' if viz_ok else 'FAILED'}")

    # ── Video writer ───────────────────────────────────────────────────────
    vwriter = None
    if args.save_video:
        import os
        os.makedirs(os.path.dirname(args.save_video) or ".", exist_ok=True)
        fw = 224 * _DISPLAY_SCALE
        fh = 224 * _DISPLAY_SCALE + _STATUS_BAR_H
        vwriter = cv2.VideoWriter(
            args.save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (fw, fh),
        )
        print(f"  Saving video → {args.save_video}")

    # ── Starting EE position (for phase-3 retreat) ─────────────────────────
    ee_start = np.array(obs["robot0_eef_pos"], dtype=float)
    ob_pos   = obstacle.pos.copy() if obstacle else ee_start + np.array([0.2, 0, 0])

    print(f"\n  EE start : {np.round(ee_start, 3)}")
    print(f"  Obstacle : {np.round(ob_pos, 3)}")
    print(f"\n  Running {args.horizon} steps …  (press Q in viewer to quit)\n")
    print(f"  {'step':>4}  {'EE pos':^28}  {'h_min':>7}  {'CBF':>5}  phase")
    print(f"  {'----':>4}  {'------':^28}  {'-----':>7}  {'---':>5}  -----")

    # ── Main loop ──────────────────────────────────────────────────────────
    for t in range(args.horizon):

        ee_pos   = np.array(obs["robot0_eef_pos"], dtype=float)
        eef_quat = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
        R1       = SciRot.from_quat(eef_quat).as_matrix()

        # Scripted action (no VLA)
        raw_action = _scripted_action(
            t, args.horizon, ee_pos, ob_pos, ee_start, speed=args.speed,
        )
        safe_action  = raw_action.copy()
        cbf_triggered = False
        h_val         = float("inf")
        min_d         = float(np.linalg.norm(ee_pos - ob_pos)) if obstacle else float("inf")

        # CBF filter
        if sphere_decomp is not None and not args.no_cbf:
            arm_rows = _compute_arm_link_constraints(
                model, data, arm_body_ids, arm_dof_idx,
                ee_body_id, R1, obstacles,
            )
            u_safe, h_val, cbf_triggered = run_sphere_decomp_cbf(
                ee_pos=ee_pos,
                R1=R1,
                obstacle_spheres=sphere_decomp,
                u_nom=raw_action[:3],
                k_cbf=K_CBF,
                extra_constraints=arm_rows,
            )
            safe_action[:3] = u_safe

        # Arm sample positions (recomputed each step as arm moves)
        arm_samples = _link_sample_positions(data, arm_body_ids, n=3, model=model, radial=True)

        # Push geoms into renderer
        ee_c = ee_ellipsoid_center(ee_pos, R1)
        if viz_ok and obstacles:
            push_cbf_geoms(
                env.sim,
                ee_center=ee_c,
                ee_q=EE_Q_DIAG_DEFAULT,
                R1=R1,
                obstacles=obstacles,
                h_values=[h_val],
                arm_body_ids=arm_body_ids,
                arm_radii=_ARM_LINK_CBF_RADII,
                model=model,
                data=data,
                cbf_triggered=cbf_triggered,
                obstacle_spheres=sphere_decomp,
                arm_sample_positions=arm_samples,
            )

        # Render frame
        img_raw = obs.get("agentview_image")
        if img_raw is not None:
            img   = _preprocess(img_raw)
            phase_str = ["→ obstacle", "↔ lateral", "← retreat"][
                0 if t / args.horizon < 0.4 else (1 if t / args.horizon < 0.7 else 2)
            ]
            frame = _render_frame(
                img, t, args.horizon,
                "no-cbf" if args.no_cbf else "sphere-cbf",
                min_d, cbf_triggered, False, 0, 0, h_val,
            )
            if vwriter is not None:
                vwriter.write(frame)
            cv2.imshow(_CV2_WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Quit by user.")
                break

        # Log every 25 steps
        if t % 25 == 0:
            phase_name = ("→ obstacle" if t / args.horizon < 0.4
                          else ("↔ lateral" if t / args.horizon < 0.7 else "← retreat"))
            h_str = f"{h_val:>7.3f}" if np.isfinite(h_val) else "    inf"
            pos_s = f"[{ee_pos[0]:+.3f} {ee_pos[1]:+.3f} {ee_pos[2]:+.3f}]"
            print(f"  {t:>4}  {pos_s}  {h_str}  {'ON ' if cbf_triggered else 'off':>5}  {phase_name}")

        # Step
        step_out = env.step(safe_action.tolist())
        obs = step_out[0] if isinstance(step_out, tuple) else step_out

    # ── Cleanup ────────────────────────────────────────────────────────────
    if vwriter is not None:
        vwriter.release()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    env.close()
    print("\n  Done.")


if __name__ == "__main__":
    main()
