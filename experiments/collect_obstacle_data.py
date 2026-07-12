"""
collect_obstacle_data.py — DAgger-style data collection for obstacle-conditioned projector.

Runs SafeLIBERO episodes under the base VLA and records per-step tuples:
    (agentview_img, wrist_img, proprio, obs_feat, nom_action, safe_action, dist, label)

Two correction modes (--correction):
  cbf  : reactive CBF filter — only activates when h < H_SAFE (can produce jerky corrections)
  apf  : proactive APF repulsion — smooth decay over d_influence radius (better training data)

DAgger labeling:
  CBF mode:  h > H_SAFE → nom (label=0); 0 < h ≤ H_SAFE → CBF (label=1); h < 0 → discard
  APF mode:  d > d_influence → nom (label=0); D_DISCARD ≤ d ≤ d_influence → APF (label=1); d < D_DISCARD → discard

The dataset is saved as a directory of .npz files, one per episode, loadable by
train_obstacle_projector.py.

Usage:
    # Test run — 5 episodes, check quality
    python -m experiments.collect_obstacle_data \\
        --suite safelibero_spatial --task 0 --episodes 5 \\
        --correction apf --out data/obs_cond_dataset

    # Full collection — all 10 tasks × 60 episodes
    for t in $(seq 0 9); do
        python -m experiments.collect_obstacle_data \\
            --suite safelibero_spatial --task $t --episodes 60 \\
            --correction apf --out data/obs_cond_dataset
    done

The VLA server must be running in plain mode (OPENVLA_OBS_COND=0) so that nominal
actions are true base-model outputs.
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

# ── CBF thresholds (used when --correction cbf) ───────────────────────────────
H_SAFE    = 0.15   # h above this → nom; below → CBF action
H_DISCARD = 0.0    # h below this → discard

# ── APF constants (used when --correction apf) ────────────────────────────────
D_DISCARD  = 0.05  # discard if within 5 cm of obstacle (unreliable zone)


# ─────────────────────────────────────────────────────────────────────────────
# Correction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cbf_correction(
    ee_pos: np.ndarray,
    near_ob: "ObstacleConfig",
    nom_xyz: np.ndarray,
    cbf_gamma: float,
    obstacle_safety_radius: float,
) -> tuple[np.ndarray, float]:
    """Return (safe_xyz, h_min) using CBF filter."""
    delta  = ee_pos - near_ob.pos
    d      = float(np.linalg.norm(delta)) + 1e-8
    n_hat  = delta / d
    h      = d - obstacle_safety_radius
    hdot   = float(np.dot(n_hat, nom_xyz))
    safe_xyz = nom_xyz.copy()
    if hdot + cbf_gamma * h < 0:
        correction = -(hdot + cbf_gamma * h) * n_hat
        safe_xyz   = nom_xyz + correction
    return safe_xyz, h


def _apf_correction(
    ee_pos: np.ndarray,
    near_ob: "ObstacleConfig",
    nom_xyz: np.ndarray,
    k_rep: float,
    d_influence: float,
) -> tuple[np.ndarray, float, float]:
    """Return (safe_xyz, dist_surface, corr_mag) using APF smooth repulsion.

    k_rep is DIMENSIONLESS. d_influence measured from obstacle SURFACE so geometry
    is correctly accounted for: alpha uses d_surface = d_center - ob.safety_radius.
    """
    delta     = ee_pos - near_ob.pos
    d         = float(np.linalg.norm(delta)) + 1e-8
    d_surface = max(0.0, d - near_ob.safety_radius)
    n_hat     = delta / d
    safe_xyz  = nom_xyz.copy()
    corr_mag  = 0.0
    if d_surface < d_influence:
        alpha    = (1.0 - d_surface / d_influence) ** 2
        nom_mag  = float(np.linalg.norm(nom_xyz)) + 1e-8
        corr_mag = k_rep * alpha * nom_mag
        safe_xyz = nom_xyz + corr_mag * n_hat
    return safe_xyz, d_surface, corr_mag


# ─────────────────────────────────────────────────────────────────────────────
# Episode collector
# ─────────────────────────────────────────────────────────────────────────────

def _collect_episode(
    env,
    lang: str,
    initial_states,
    ep_idx: int,
    *,
    openvla_port: int,
    replan_steps: int,
    horizon: int,
    correction: str,
    cbf_gamma: float,
    obstacle_safety_radius: float,
    k_rep: float,
    d_influence: float,
    out_dir: Path,
) -> dict:
    """Run one episode, save DAgger-labelled steps to out_dir/ep_{ep_idx:04d}.npz."""

    env.reset()
    if initial_states is not None:
        env.set_init_state(initial_states[ep_idx % len(initial_states)])
    obs = env.get_obs()

    obstacles: list[ObstacleConfig] = detect_safelibero_obstacle(env, obs)
    if not obstacles:
        print(f"  ep {ep_idx}: no obstacle detected, skipping")
        return {"steps": 0, "discarded": 0, "nom": 0, "safe": 0, "corr_mags": []}

    buf_img      = []
    buf_wrist    = []
    buf_proprio  = []
    buf_obs_feat = []
    buf_nom_act  = []
    buf_safe_act = []   # APF- or CBF-corrected (= nom_act when label=0)
    buf_dist     = []   # distance to nearest obstacle
    buf_corr_mag = []   # correction magnitude (0 if nom)
    buf_label    = []   # 0=nom, 1=safe, -1=discard

    action_queue: list[np.ndarray] = []
    nom_queue:    list[np.ndarray] = []

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

        dists   = [np.linalg.norm(ee_pos - ob.pos) for ob in obstacles]
        near_ob = obstacles[int(np.argmin(dists))]
        obs_feat = _compute_obs_features(ee_pos, near_ob.pos)

        if not action_queue:
            try:
                raw_chunk  = _query_openvla_chunk(img, wrist_img, state, lang,
                                                  num_actions=replan_steps, url=url)
                action_queue = [_post_process_vla(a) for a in raw_chunk]
                nom_queue    = [_post_process_vla(a) for a in raw_chunk]
            except Exception as e:
                print(f"  [{t:03d}] VLA error: {e}")
                break

        nom_action = nom_queue.pop(0)
        _          = action_queue.pop(0)   # unused — we recompute correction each step
        nom_xyz    = nom_action[:3].copy()

        if correction == "apf":
            safe_xyz, dist, corr_mag = _apf_correction(
                ee_pos, near_ob, nom_xyz, k_rep=k_rep, d_influence=d_influence)
            if dist < D_DISCARD:
                label = -1
            elif dist < d_influence:
                label = 1
            else:
                label = 0

        else:  # cbf
            # h over all obstacles
            h_min = float("inf")
            for ob in obstacles:
                h = float(np.linalg.norm(ee_pos - ob.pos)) - obstacle_safety_radius
                if h < h_min:
                    h_min = h
            dist = float(np.linalg.norm(ee_pos - near_ob.pos))
            if h_min < H_SAFE + 0.05:
                try:
                    safe_xyz, _ = _cbf_correction(
                        ee_pos, near_ob, nom_xyz, cbf_gamma, obstacle_safety_radius)
                except Exception:
                    safe_xyz = nom_xyz.copy()
            else:
                safe_xyz = nom_xyz.copy()
            corr_mag = float(np.linalg.norm(safe_xyz - nom_xyz))
            if h_min < H_DISCARD:
                label = -1
            elif h_min <= H_SAFE:
                label = 1
            else:
                label = 0

        safe_action = nom_action.copy()
        safe_action[:3] = safe_xyz

        buf_img.append(img)
        buf_wrist.append(wrist_img)
        buf_proprio.append(state.astype(np.float32))
        buf_obs_feat.append(obs_feat.astype(np.float32))
        buf_nom_act.append(nom_action[:7].astype(np.float32))
        buf_safe_act.append(safe_action[:7].astype(np.float32))
        buf_dist.append(float(dist))
        buf_corr_mag.append(float(corr_mag))
        buf_label.append(int(label))

        exec_action = safe_action if label == 1 else nom_action
        obs, _, done, _ = env.step(exec_action)
        if done:
            break

    labels  = np.array(buf_label)
    keep    = labels >= 0
    n_total = int(keep.sum())
    n_nom   = int((labels[keep] == 0).sum())
    n_safe  = int((labels[keep] == 1).sum())
    n_disc  = int((labels == -1).sum())
    kept_mags = np.array(buf_corr_mag)[keep]

    if n_total > 0:
        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out_dir / f"ep_{ep_idx:04d}.npz",
            img      = np.array(buf_img,      dtype=np.uint8)[keep],
            wrist    = np.array(buf_wrist,    dtype=np.uint8)[keep],
            proprio  = np.array(buf_proprio,  dtype=np.float32)[keep],
            obs_feat = np.array(buf_obs_feat, dtype=np.float32)[keep],
            nom_act  = np.array(buf_nom_act,  dtype=np.float32)[keep],
            cbf_act  = np.array(buf_safe_act, dtype=np.float32)[keep],  # key kept for trainer compat
            dist     = np.array(buf_dist,     dtype=np.float32)[keep],
            corr_mag = np.array(buf_corr_mag, dtype=np.float32)[keep],
            label    = labels[keep],
        )

    mean_mag = float(kept_mags[kept_mags > 0].mean()) if (kept_mags > 0).any() else 0.0
    print(f"  ep {ep_idx:03d}: {n_total} steps kept  "
          f"(nom={n_nom}  safe={n_safe}  disc={n_disc})  "
          f"mean_corr={mean_mag:.4f}m")
    return {"steps": n_total, "discarded": n_disc, "nom": n_nom, "safe": n_safe,
            "corr_mags": kept_mags[kept_mags > 0].tolist()}


# ─────────────────────────────────────────────────────────────────────────────
# Quality report
# ─────────────────────────────────────────────────────────────────────────────

def _quality_report(totals: dict, all_mags: list[float], correction: str) -> None:
    total = totals["steps"]
    if total == 0:
        print("\nNo steps collected — check obstacle detection and server connection.")
        return

    nom_frac  = totals["nom"]  / total * 100
    safe_frac = totals["safe"] / total * 100
    disc_frac = totals["discarded"] / (total + totals["discarded"]) * 100

    print("\n" + "=" * 55)
    print("DATA QUALITY REPORT")
    print("=" * 55)
    print(f"  Correction mode  : {correction.upper()}")
    print(f"  Total steps kept : {total}")
    print(f"  Label split      : nom={nom_frac:.1f}%  safe={safe_frac:.1f}%")
    print(f"  Discarded steps  : {totals['discarded']}  ({disc_frac:.1f}% of all steps)")

    if all_mags:
        mags = np.array(all_mags)
        print(f"\n  Correction magnitude (steps where correction applied):")
        print(f"    min={mags.min():.4f}  p25={np.percentile(mags,25):.4f}  "
              f"median={np.median(mags):.4f}  p75={np.percentile(mags,75):.4f}  "
              f"max={mags.max():.4f}")

        # ASCII histogram
        counts, edges = np.histogram(mags, bins=8)
        print(f"\n  Histogram of correction magnitudes (m):")
        bar_max = max(counts)
        for i, c in enumerate(counts):
            bar = "█" * int(c / bar_max * 30)
            print(f"    {edges[i]:.3f}-{edges[i+1]:.3f}  {bar}  ({c})")

    ideal_safe_frac = 20.0  # rough target: ~20% of steps near obstacle
    if safe_frac < 5.0:
        print(f"\n  ⚠  Very few safe-label steps ({safe_frac:.1f}%). "
              "Try decreasing --d-influence (APF) or --safety-radius (CBF).")
    elif safe_frac > 60.0:
        print(f"\n  ⚠  Many safe-label steps ({safe_frac:.1f}%). "
              "Model may overfit to avoidance. Consider decreasing --d-influence.")
    else:
        print(f"\n  ✓ Label split looks reasonable.")

    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect obstacle-conditioned DAgger data")
    parser.add_argument("--suite",       default="safelibero_spatial")
    parser.add_argument("--task",        type=int, default=0)
    parser.add_argument("--safety-level", type=int, default=1)
    parser.add_argument("--episodes",   type=int, default=60)
    parser.add_argument("--horizon",    type=int, default=600)
    parser.add_argument("--replan-steps", type=int, default=8)
    # Correction mode
    parser.add_argument("--correction", choices=["cbf", "apf"], default="apf",
                        help="cbf: reactive CBF filter; apf: smooth APF repulsion (default)")
    # APF params
    parser.add_argument("--k-rep",       type=float, default=2.0,
                        help="APF gain (dimensionless; default 2.0 = 73%% of nom action at closest approach)")
                        help="APF repulsion gain (m/step at obstacle surface, default 0.025)")
    parser.add_argument("--d-influence", type=float, default=0.20,
                        help="APF influence radius from obstacle SURFACE in metres (default 0.20)")
    # CBF params
    parser.add_argument("--cbf-gamma",  type=float, default=1.8)
    parser.add_argument("--safety-radius", type=float, default=0.10)
    # IO
    parser.add_argument("--openvla-port", type=int, default=8000)
    parser.add_argument("--out", default="data/obs_cond_dataset")
    args = parser.parse_args()

    out_dir = Path(args.out) / f"{args.suite}_t{args.task:02d}_LI"
    print(f"Collecting {args.episodes} episodes → {out_dir}")
    print(f"  correction={args.correction}  "
          + (f"k_rep={args.k_rep}  d_influence={args.d_influence}"
             if args.correction == "apf"
             else f"cbf_gamma={args.cbf_gamma}  safety_radius={args.safety_radius}"))

    env, lang, initial_states = make_libero_env(
        task_suite=args.suite,
        task_idx=args.task,
        safety_level=args.safety_level,
        has_renderer=False,
        horizon=args.horizon,
    )

    totals    = {"steps": 0, "discarded": 0, "nom": 0, "safe": 0}
    all_mags: list[float] = []

    for ep in range(args.episodes):
        stats = _collect_episode(
            env, lang, initial_states, ep,
            openvla_port=args.openvla_port,
            replan_steps=args.replan_steps,
            horizon=args.horizon,
            correction=args.correction,
            cbf_gamma=args.cbf_gamma,
            obstacle_safety_radius=args.safety_radius,
            k_rep=args.k_rep,
            d_influence=args.d_influence,
            out_dir=out_dir,
        )
        for k in ("steps", "discarded", "nom", "safe"):
            totals[k] += stats[k]
        all_mags.extend(stats["corr_mags"])

    _quality_report(totals, all_mags, args.correction)
    print(f"Dataset saved to {out_dir}/")


if __name__ == "__main__":
    main()
