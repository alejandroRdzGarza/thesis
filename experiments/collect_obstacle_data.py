"""
collect_obstacle_data.py — DAgger-style data collection for obstacle-conditioned projector.

Runs SafeLIBERO episodes under the base VLA + CBF filter and records per-step tuples:
    (agentview_img, wrist_img, proprio, obs_feat, nom_action, cbf_action, h_value, label)

DAgger labeling (applied per chunk, based on minimum h in the upcoming chunk):
    h > H_SAFE     → use nominal VLA action (safe, no correction needed)
    0 < h ≤ H_SAFE → use CBF-corrected action (near boundary; model should learn avoidance)
    h < 0          → discard step (safety violation; labels are unreliable)

The dataset is saved as a directory of .npz files, one per episode, loadable by
train_obstacle_projector.py.

Usage:
    python -m experiments.collect_obstacle_data \
        --suite safelibero_spatial --task 0 --episodes 50 \
        --out data/obs_cond_dataset \
        --openvla-port 8000

The VLA server must be running (plain mode, NOT obs_cond) so that nominal actions
are true base-model outputs.  The CBF filter runs locally.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from experiments.libero_runner import (
    _build_proprio,
    _compute_obs_features,
    _OBS_MAX_RANGE,
    _post_process_vla,
    _preprocess,
    _query_openvla_chunk,
    OPENVLA_URL,
    ObstacleConfig,
    detect_safelibero_obstacle,
    make_libero_env,
)
from experiments.cbf_ellipsoid import (
    build_cbf_qp,
    ee_ellipsoid_center,
    CbfQpResult,
)

# ── DAgger labeling thresholds ────────────────────────────────────────────────
H_SAFE   = 0.15   # h above this → use nom action; below → use CBF action
H_DISCARD = 0.0   # h below this → discard (in collision; labels unreliable)


def _collect_episode(
    env,
    lang: str,
    initial_states,
    ep_idx: int,
    *,
    openvla_port: int,
    replan_steps: int,
    horizon: int,
    cbf_gamma: float,
    obstacle_safety_radius: float,
    out_dir: Path,
) -> dict:
    """Run one episode, collect DAgger-labelled steps, save to out_dir/ep_{ep_idx:04d}.npz."""

    # Reset environment
    env.reset()
    if initial_states is not None:
        env.set_init_state(initial_states[ep_idx % len(initial_states)])
    obs = env.get_obs()

    obstacles: list[ObstacleConfig] = detect_safelibero_obstacle(env, obs)
    if not obstacles:
        print(f"  ep {ep_idx}: no obstacle detected, skipping")
        return {"steps": 0, "discarded": 0, "nom_labels": 0, "cbf_labels": 0}

    # Storage buffers
    buf_img       = []  # (H, W, 3) uint8
    buf_wrist     = []  # (H, W, 3) uint8
    buf_proprio   = []  # (8,) float64
    buf_obs_feat  = []  # (4,) float64: obs_dir(3) + obs_dist(1)
    buf_nom_act   = []  # (7,) float32 — raw VLA output
    buf_cbf_act   = []  # (7,) float32 — CBF-corrected (= nom_act if h>H_SAFE)
    buf_h         = []  # float
    buf_label     = []  # 0=nom, 1=cbf, -1=discard

    action_queue: list[np.ndarray] = []
    nom_queue:    list[np.ndarray] = []  # parallel buffer tracking nom actions for each queued step

    url = f"http://127.0.0.1:{openvla_port}/act"

    for t in range(horizon):
        img_raw   = obs.get("agentview_image")
        wrist_raw = obs.get("robot0_eye_in_hand_image")
        if img_raw is None:
            obs, _, done, _ = env.step(np.zeros(7))
            if done:
                break
            continue

        img       = _preprocess(img_raw)
        wrist_img = _preprocess(wrist_raw) if wrist_raw is not None else np.zeros((224, 224, 3), dtype=np.uint8)
        state     = _build_proprio(obs)
        ee_pos    = np.array(obs["robot0_eef_pos"], dtype=float)

        # Nearest obstacle features
        dists    = [np.linalg.norm(ee_pos - ob.pos) for ob in obstacles]
        near_ob  = obstacles[int(np.argmin(dists))]
        obs_feat = _compute_obs_features(ee_pos, near_ob.pos)

        if not action_queue:
            try:
                raw_chunk = _query_openvla_chunk(img, wrist_img, state, lang,
                                                 num_actions=replan_steps, url=url)
                action_queue = [_post_process_vla(a) for a in raw_chunk]
                nom_queue    = [_post_process_vla(a) for a in raw_chunk]
            except Exception as e:
                print(f"  [{t:03d}] VLA error: {e}")
                break

        nom_action = nom_queue.pop(0)
        action     = action_queue.pop(0)

        # Compute h value (minimum over obstacles; single-sphere approximation)
        h_min = float("inf")
        for ob in obstacles:
            delta = ee_pos - ob.pos
            dist  = float(np.linalg.norm(delta))
            h     = dist - obstacle_safety_radius
            if h < h_min:
                h_min = h

        # CBF filter on xyz component only
        nom_xyz  = nom_action[:3].copy()
        safe_xyz = nom_xyz.copy()
        if h_min < H_SAFE + 0.05:  # only run QP near boundary
            try:
                # Simple repulsive correction: project away from obstacle if h < H_SAFE
                delta_nearest = ee_pos - near_ob.pos
                dist_nearest  = float(np.linalg.norm(delta_nearest)) + 1e-8
                n_hat         = delta_nearest / dist_nearest
                h             = dist_nearest - obstacle_safety_radius
                hdot_nom      = float(np.dot(n_hat, nom_xyz))
                if hdot_nom + cbf_gamma * h < 0:
                    correction  = -(hdot_nom + cbf_gamma * h) * n_hat
                    safe_xyz    = nom_xyz + correction
            except Exception:
                pass

        cbf_action = nom_action.copy()
        cbf_action[:3] = safe_xyz

        # DAgger label
        if h_min < H_DISCARD:
            label = -1   # discard
        elif h_min <= H_SAFE:
            label = 1    # use CBF action
        else:
            label = 0    # use nom action

        buf_img.append(img)
        buf_wrist.append(wrist_img)
        buf_proprio.append(state.astype(np.float32))
        buf_obs_feat.append(obs_feat.astype(np.float32))
        buf_nom_act.append(nom_action[:7].astype(np.float32))
        buf_cbf_act.append(cbf_action[:7].astype(np.float32))
        buf_h.append(float(h_min))
        buf_label.append(int(label))

        # Step env with safe action
        exec_action = cbf_action if label == 1 else nom_action
        obs, _, done, _ = env.step(exec_action)
        if done:
            break

    # Filter and save
    labels  = np.array(buf_label)
    keep    = labels >= 0
    n_total = int(keep.sum())
    n_nom   = int((labels[keep] == 0).sum())
    n_cbf   = int((labels[keep] == 1).sum())
    n_disc  = int((labels == -1).sum())

    if n_total > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"ep_{ep_idx:04d}.npz",
            img       = np.array(buf_img,     dtype=np.uint8)[keep],
            wrist     = np.array(buf_wrist,   dtype=np.uint8)[keep],
            proprio   = np.array(buf_proprio, dtype=np.float32)[keep],
            obs_feat  = np.array(buf_obs_feat,dtype=np.float32)[keep],
            nom_act   = np.array(buf_nom_act, dtype=np.float32)[keep],
            cbf_act   = np.array(buf_cbf_act, dtype=np.float32)[keep],
            h_value   = np.array(buf_h,       dtype=np.float32)[keep],
            label     = labels[keep],
        )

    print(f"  ep {ep_idx:03d}: {n_total} steps kept  "
          f"(nom={n_nom}  cbf={n_cbf}  discarded={n_disc})")
    return {"steps": n_total, "discarded": n_disc, "nom_labels": n_nom, "cbf_labels": n_cbf}


def main():
    parser = argparse.ArgumentParser(description="Collect obstacle-conditioned DAgger data")
    parser.add_argument("--suite",    default="safelibero_spatial")
    parser.add_argument("--task",     type=int, default=0)
    parser.add_argument("--safety-level", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--horizon",  type=int, default=600)
    parser.add_argument("--replan-steps", type=int, default=8)
    parser.add_argument("--cbf-gamma", type=float, default=1.8)
    parser.add_argument("--safety-radius", type=float, default=0.10)
    parser.add_argument("--openvla-port", type=int, default=8000)
    parser.add_argument("--out", default="data/obs_cond_dataset")
    args = parser.parse_args()

    out_dir = Path(args.out) / f"{args.suite}_t{args.task:02d}_LI"
    print(f"Collecting {args.episodes} episodes → {out_dir}")

    env, lang, initial_states = make_libero_env(
        task_suite=args.suite,
        task_idx=args.task,
        safety_level=args.safety_level,
        has_renderer=False,
        horizon=args.horizon,
    )

    totals = {"steps": 0, "discarded": 0, "nom_labels": 0, "cbf_labels": 0}
    for ep in range(args.episodes):
        stats = _collect_episode(
            env, lang, initial_states, ep,
            openvla_port=args.openvla_port,
            replan_steps=args.replan_steps,
            horizon=args.horizon,
            cbf_gamma=args.cbf_gamma,
            obstacle_safety_radius=args.safety_radius,
            out_dir=out_dir,
        )
        for k in totals:
            totals[k] += stats[k]

    print(f"\nDone. Total: {totals['steps']} steps  "
          f"(nom={totals['nom_labels']}  cbf={totals['cbf_labels']}  "
          f"discarded={totals['discarded']})")
    print(f"Dataset saved to {out_dir}/")


if __name__ == "__main__":
    main()
