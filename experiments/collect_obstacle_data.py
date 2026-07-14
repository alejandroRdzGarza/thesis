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
import sys
import time
from pathlib import Path

import numpy as np

import cv2


class _Tee:
    """Write to both stdout and a log file simultaneously."""
    def __init__(self, stream, path: Path):
        self._stream = stream
        self._file   = open(path, "w", buffering=1)
    def write(self, s):
        self._stream.write(s)
        self._file.write(s)
    def flush(self):
        self._stream.flush()
        self._file.flush()
    def fileno(self):
        return self._stream.fileno()
    def close(self):
        self._file.close()

from experiments.libero_runner import (
    _build_proprio,
    _compute_obs_features,
    _OBS_MAX_RANGE,
    _post_process_vla,
    _preprocess,
    _query_openvla_chunk,
    _render_frame,
    _unwrap_sim,
    OPENVLA_URL,
    ObstacleConfig,
    detect_safelibero_obstacle,
    make_libero_env,
)
from experiments.cbf_ellipsoid import ee_ellipsoid_center, EE_Q_DIAG_DEFAULT

try:
    from experiments.cbf_visualizer import install_scene_hook, push_cbf_geoms
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

_CV2_WINDOW = "collect_obstacle_data"

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


def _teleport_obstacle(env, obs: dict, obstacle_name: str,
                        x_range: tuple = (0.30, 0.65),
                        y_range: tuple = (-0.35, 0.35),
                        min_robot_dist: float = 0.25) -> dict:
    """Move obstacle to a random (x, y) within workspace; keep z and orientation.

    Samples uniformly within x_range × y_range with rejection on robot-base
    proximity.  Returns fresh obs after env.sim.forward().
    """
    joint_name = f"{obstacle_name}_joint0"
    try:
        jid = env.sim.model.joint_name2id(joint_name)
    except Exception:
        return obs                              # joint not found, skip
    qpos_addr = env.sim.model.jnt_qposadr[jid]
    for _ in range(50):                         # rejection sampling
        nx = np.random.uniform(*x_range)
        ny = np.random.uniform(*y_range)
        if np.sqrt(nx ** 2 + ny ** 2) >= min_robot_dist:
            break
    env.sim.data.qpos[qpos_addr]     = nx
    env.sim.data.qpos[qpos_addr + 1] = ny
    # z (qpos_addr+2) and quaternion (qpos_addr+3:7) stay unchanged
    env.sim.forward()
    # Different LIBERO env wrappers expose observations differently
    for _getter in ("_get_observations", "_get_obs", "get_observation"):
        fn = getattr(env, _getter, None)
        if fn is not None:
            return fn()
    # Final fallback: zero-action step (advances physics one tick, updates obs)
    new_obs, _, _, _ = env.step(np.zeros(7))
    return new_obs


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
    randomize_obstacle: bool = False,
    cbf_near_goal_off: bool = False,
    show: bool = False,
) -> dict:
    """Run one episode, save DAgger-labelled steps to out_dir/ep_{ep_idx:04d}.npz."""

    out_dir.mkdir(parents=True, exist_ok=True)
    _log_path   = out_dir / f"ep_{ep_idx:04d}.log"
    _tee        = _Tee(sys.stdout, _log_path)
    _real_stdout = sys.stdout
    sys.stdout   = _tee

    try:
        return _collect_episode_body(
            env=env, lang=lang, initial_states=initial_states, ep_idx=ep_idx,
            openvla_port=openvla_port, replan_steps=replan_steps, horizon=horizon,
            correction=correction, cbf_gamma=cbf_gamma,
            obstacle_safety_radius=obstacle_safety_radius,
            k_rep=k_rep, d_influence=d_influence, out_dir=out_dir,
            randomize_obstacle=randomize_obstacle, cbf_near_goal_off=cbf_near_goal_off,
            show=show,
        )
    finally:
        sys.stdout = _real_stdout
        _tee.close()
        _real_stdout.write(f"  Log → {_log_path}\n")


def _collect_episode_body(
    env, lang: str, initial_states, ep_idx: int, *,
    openvla_port, replan_steps, horizon, correction, cbf_gamma,
    obstacle_safety_radius, k_rep, d_influence, out_dir, randomize_obstacle,
    cbf_near_goal_off, show,
) -> dict:
    env.reset()
    if initial_states is not None:
        obs = env.set_init_state(initial_states[ep_idx % len(initial_states)])
    else:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result

    # detect_safelibero_obstacle returns ObstacleConfig | None — wrap in list
    _detected = detect_safelibero_obstacle(env, obs)
    obstacles  = [_detected] if _detected is not None else []
    if not obstacles:
        print(f"  ep {ep_idx}: no obstacle detected, skipping")
        return {"steps": 0, "discarded": 0, "nom": 0, "safe": 0, "corr_mags": [],
                "success": False, "collision": False, "max_lift": 0.0, "grasp_step": None}

    if randomize_obstacle:
        obs = _teleport_obstacle(env, obs, obstacles[0].name)
        _det2 = detect_safelibero_obstacle(env, obs)
        if _det2 is not None:
            obstacles = [_det2]

    # ── Visualization setup ───────────────────────────────────────────────────
    _viz_hook_ok = False
    model, data = None, None
    if show and _HAS_VIZ:
        try:
            model, data = _unwrap_sim(env)
            _viz_hook_ok = install_scene_hook(env.sim)
            cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
        except Exception as _e:
            print(f"  [warn] viz init failed: {_e}")

    # ── Collision tracking (displacement-based, matches SafeLIBERO metric) ────
    # Baseline is captured AFTER the first physics step so teleport-settling
    # micro-displacements (physics resolving the new contact state) don't
    # flag as collisions before the robot has moved.
    _initial_obstacle_pos = None   # set lazily after step 0
    _collision_flag = False
    _obs_key_coll = f"{obstacles[0].name}_pos" if obstacles else None

    # ── Object tracking ───────────────────────────────────────────────────────
    _obj_pos_keys = sorted([
        k for k in obs.keys()
        if k.endswith("_pos") and not k.startswith("robot") and "to_robot" not in k
    ])
    _obstacle_key_set = {f"{ob.name}_pos" for ob in obstacles}
    _obj_initial_z    = {k: float(obs[k][2]) for k in _obj_pos_keys if k in obs}
    _target_keys      = [k for k in _obj_pos_keys if k not in _obstacle_key_set]

    # ── Episode-level debug state ─────────────────────────────────────────────
    _success      = False
    _max_lift     = 0.0          # max z-rise of any target object
    _grasp_step   = None         # first step where lift > 2 cm
    _grasp_obj    = None

    print(f"\n  [ep {ep_idx:03d}] obstacle={obstacles[0].name if obstacles else 'none'}"
          f"  correction={correction.upper()}  lang=\"{lang}\"")
    print(f"  Objects: " + "  ".join(
        f"{k.replace('_pos','')}@z={_obj_initial_z[k]:.3f}"
        for k in _obj_pos_keys if k in _obj_initial_z
    ))

    vla_cnt = 0

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

    # CBF near-goal deactivation: once gripper has been closing for 2+ steps, stop CBF.
    # This lets the VLA complete the grasp unimpeded once it's committed to the target.
    _cbf_active          = True
    _gripper_close_steps = 0

    url = f"http://127.0.0.1:{openvla_port}/act"

    def _quat_to_euler(q):
        """quaternion [x,y,z,w] → roll,pitch,yaw in degrees."""
        x, y, z, w = q
        roll  = np.degrees(np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)))
        pitch = np.degrees(np.arcsin(np.clip(2*(w*y-z*x), -1, 1)))
        yaw   = np.degrees(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z)))
        return roll, pitch, yaw

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
        ee_quat   = np.array(obs.get("robot0_eef_quat", [0,0,0,1]), dtype=float)
        ee_euler  = _quat_to_euler(ee_quat)
        gq        = np.array(obs.get("robot0_gripper_qpos", [0.04, 0.04]), dtype=float)

        # ── Per-step full state log ───────────────────────────────────────────
        _rel_strs = []
        for _k in _obj_pos_keys:
            if _k not in obs:
                continue
            _op  = np.array(obs[_k], dtype=float)
            _rel = _op - ee_pos
            _d   = float(np.linalg.norm(_rel))
            _tag = "OBS" if _k in _obstacle_key_set else "tgt"
            _rel_strs.append(
                f"    {_tag} {_k.replace('_pos',''):30s}"
                f" pos=[{_op[0]:+.3f},{_op[1]:+.3f},{_op[2]:+.3f}]"
                f" rel=[{_rel[0]:+.3f},{_rel[1]:+.3f},{_rel[2]:+.3f}]"
                f" d={_d:.3f}m"
            )
        print(
            f"[t={t:03d}]"
            f" EE=[{ee_pos[0]:+.3f},{ee_pos[1]:+.3f},{ee_pos[2]:+.3f}]"
            f" rpy=[{ee_euler[0]:+.1f},{ee_euler[1]:+.1f},{ee_euler[2]:+.1f}]deg"
            f" grip=[{gq[0]:.4f},{gq[1]:.4f}]"
        )
        for _s in _rel_strs:
            print(_s)

        dists   = [np.linalg.norm(ee_pos - ob.pos) for ob in obstacles]
        near_ob = obstacles[int(np.argmin(dists))]
        obs_feat = _compute_obs_features(ee_pos, near_ob.pos)

        if not action_queue:
            try:
                _t0 = time.perf_counter()
                raw_chunk  = _query_openvla_chunk(img, wrist_img, state, lang,
                                                  num_actions=replan_steps, url=url)
                vla_ms = (time.perf_counter() - _t0) * 1000
                action_queue = [_post_process_vla(a) for a in raw_chunk]
                nom_queue    = [_post_process_vla(a) for a in raw_chunk]
                vla_cnt += 1
                grip_str = "CLOSE" if action_queue[0][6] > 0 else "open"
                _gq = obs.get("robot0_gripper_qpos", [0.04, 0.04])
                obj_str = ""
                if _obj_pos_keys:
                    _dists = {k: float(np.linalg.norm(np.array(obs[k]) - ee_pos))
                              for k in _obj_pos_keys if k in obs}
                    _nk = min(_dists, key=_dists.get)
                    obj_str = f"  nearest={_nk.replace('_pos','')}({_dists[_nk]:.3f}m)"
                _obj_z_str = "  ".join(
                    f"{k.replace('_pos','')}z={obs[k][2]:.3f}"
                    for k in _target_keys if k in obs
                )
                print(f"  [{t:03d}] VLA #{vla_cnt}  grip={grip_str}(q={_gq[0]:.3f},{_gq[1]:.3f})"
                      f"  EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
                      f"{obj_str}  ({vla_ms:.0f}ms)"
                      + (f"\n         objs: {_obj_z_str}" if _obj_z_str else ""))
            except Exception as e:
                print(f"  [{t:03d}] VLA error: {e}")
                break

        nom_action = nom_queue.pop(0)
        _          = action_queue.pop(0)   # unused — we recompute correction each step
        nom_xyz    = nom_action[:3].copy()

        # Track gripper state — deactivate CBF once the gripper starts closing,
        # so the VLA can grasp the target without interference.
        if cbf_near_goal_off:
            g_qpos = obs.get("robot0_gripper_qpos", np.array([0.04, 0.04]))
            if float(np.sum(np.abs(g_qpos))) < 0.015:   # both fingers nearly closed
                _gripper_close_steps += 1
                if _gripper_close_steps >= 2:
                    _cbf_active = False
            else:
                _gripper_close_steps = 0

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

            if not _cbf_active:
                # CBF deactivated near goal — execute nominal, label as nom
                safe_xyz = nom_xyz.copy()
                corr_mag = 0.0
                label    = 0
            else:
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
        _cbf_on = (label == 1)
        print(
            f"  act nom=[{nom_action[0]:+.4f},{nom_action[1]:+.4f},{nom_action[2]:+.4f}"
            f" r={nom_action[3]:+.3f},{nom_action[4]:+.3f},{nom_action[5]:+.3f}"
            f" g={nom_action[6]:+.0f}]"
            + (f"  CBF→[{safe_action[0]:+.4f},{safe_action[1]:+.4f},{safe_action[2]:+.4f}]"
               if _cbf_on else "")
        )
        step_out = env.step(exec_action)
        obs = step_out[0] if isinstance(step_out[0], dict) else step_out[0]
        done = step_out[2] if len(step_out) == 4 else (step_out[2] or step_out[3])

        # ── Lift / grasp detection ────────────────────────────────────────────
        for _k in _target_keys:
            if _k not in obs:
                continue
            _lift = float(obs[_k][2]) - _obj_initial_z.get(_k, 0.0)
            if _lift > _max_lift:
                _max_lift = _lift
            if _lift > 0.020 and _grasp_step is None:
                _grasp_step = t
                _grasp_obj  = _k
                print(f"  [{t:03d}] *** LIFT: {_k.replace('_pos','')} z={obs[_k][2]:.3f}m"
                      f" (rise={_lift:.3f}m)  EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
                      f"  grip=q[{obs.get('robot0_gripper_qpos',[0,0])[0]:.3f}]")

        # ── Task success check ────────────────────────────────────────────────
        if not _success:
            try:
                _success = bool(env.check_success())
                if _success:
                    print(f"  [{t:03d}] *** TASK SUCCESS ***")
            except Exception:
                pass

        # ── Collision detection (displacement-based) ──────────────────────────
        if _obs_key_coll and _obs_key_coll in obs:
            _curr_obs_pos = np.array(obs[_obs_key_coll], dtype=float)
            if _initial_obstacle_pos is None:
                _initial_obstacle_pos = _curr_obs_pos.copy()  # step-0 baseline
            elif not _collision_flag:
                _disp = float(np.sum(np.abs(_curr_obs_pos - _initial_obstacle_pos)))
                if _disp > 0.002:  # 2mm: matches SafeLIBERO intent, filters physics-settle noise
                    _collision_flag = True
                    print(f"  [{t:03d}] COLLISION: obstacle displaced {_disp:.4f}m")

        # ── Console status every CBF step ─────────────────────────────────────
        if label == 1:
            print(f"  [{t:03d}] CBF  h={h_min:.3f}  corr={corr_mag:.4f}m")

        # ── cv2 display ───────────────────────────────────────────────────────
        if show and img is not None:
            if _viz_hook_ok and obstacles and model is not None:
                try:
                    from scipy.spatial.transform import Rotation as _SR
                    _eef_q = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), float)
                    _R1    = _SR.from_quat(_eef_q).as_matrix()
                    _ee_c  = ee_ellipsoid_center(ee_pos, _R1)
                    push_cbf_geoms(
                        env.sim,
                        ee_center=_ee_c, ee_q=EE_Q_DIAG_DEFAULT, R1=_R1,
                        obstacles=obstacles,
                        h_values=[h_min if correction == "cbf" else dist],
                        arm_body_ids=[], arm_radii={},
                        model=model, data=data,
                        cbf_triggered=(label == 1),
                        obstacle_spheres=None, arm_sample_positions=None,
                    )
                except Exception:
                    pass
            _h_disp = h_min if correction == "cbf" else float("inf")
            frame = _render_frame(img, t, horizon, correction, dist,
                                  label == 1, _collision_flag, ep_idx, vla_cnt,
                                  h_val=_h_disp)
            cv2.imshow(_CV2_WINDOW, frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

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

    # ── Episode outcome summary ───────────────────────────────────────────────
    _outcome = "SUCCESS" if _success else ("COLLISION" if _collision_flag else "FAIL")
    print(f"  ep {ep_idx:03d} [{_outcome}]  steps={t+1}  vla={vla_cnt}"
          f"  max_lift={_max_lift:.3f}m"
          + (f"  grasped={_grasp_obj.replace('_pos','') if _grasp_obj else 'NONE'}@step={_grasp_step}"
             if _grasp_step else "  NEVER_GRASPED")
          + f"  collision={_collision_flag}"
          + f"\n         data: {n_total} steps kept (nom={n_nom} safe={n_safe} disc={n_disc})"
          + f"  mean_corr={mean_mag:.4f}m")

    return {"steps": n_total, "discarded": n_disc, "nom": n_nom, "safe": n_safe,
            "corr_mags": kept_mags[kept_mags > 0].tolist(),
            "success": _success, "collision": _collision_flag,
            "max_lift": _max_lift, "grasp_step": _grasp_step}


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
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--episodes",   type=int, default=60)
    parser.add_argument("--horizon",    type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=8)
    # Correction mode
    parser.add_argument("--correction", choices=["cbf", "apf"], default="apf",
                        help="cbf: reactive CBF filter; apf: smooth APF repulsion (default)")
    # APF params
    parser.add_argument("--k-rep",       type=float, default=2.0,
                        help="APF gain (dimensionless; default 2.0)")
    parser.add_argument("--d-influence", type=float, default=0.20,
                        help="APF influence radius from obstacle SURFACE in metres (default 0.20)")
    # CBF params
    parser.add_argument("--cbf-gamma",  type=float, default=1.8)
    parser.add_argument("--safety-radius", type=float, default=0.10)
    # Data augmentation
    parser.add_argument("--randomize-obstacle", action="store_true",
                        help="Teleport obstacle to a random workspace position each episode")
    parser.add_argument("--cbf-near-goal-off", action="store_true",
                        help="Deactivate CBF once gripper starts closing (near grasp target)")
    # IO / display
    parser.add_argument("--show", action="store_true",
                        help="Open MuJoCo viewer window (slower; useful for debugging)")
    parser.add_argument("--openvla-port", type=int, default=8000)
    parser.add_argument("--out", default="data/obs_cond_dataset")
    args = parser.parse_args()

    scene_mode = "rand" if args.randomize_obstacle else "orig"
    out_dir = Path(args.out) / f"{args.suite}_t{args.task:02d}_L{args.safety_level}_{scene_mode}"
    print(f"Collecting {args.episodes} episodes → {out_dir}")
    print(f"  correction={args.correction}  scene={scene_mode}  level={args.safety_level}  "
          + (f"k_rep={args.k_rep}  d_influence={args.d_influence}"
             if args.correction == "apf"
             else f"cbf_gamma={args.cbf_gamma}  safety_radius={args.safety_radius}"))

    env, lang, initial_states = make_libero_env(
        task_suite=args.suite,
        task_idx=args.task,
        safety_level=args.safety_level,
        has_renderer=False,   # cv2 handles display; MuJoCo windowed renderer hangs
        horizon=args.horizon,
    )

    totals    = {"steps": 0, "discarded": 0, "nom": 0, "safe": 0}
    all_mags: list[float] = []
    n_success = 0
    n_grasped = 0
    n_collision = 0

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
            randomize_obstacle=args.randomize_obstacle,
            cbf_near_goal_off=args.cbf_near_goal_off,
            show=args.show,
        )
        for k in ("steps", "discarded", "nom", "safe"):
            totals[k] += stats[k]
        all_mags.extend(stats["corr_mags"])
        if stats.get("success"):    n_success   += 1
        if stats.get("grasp_step") is not None: n_grasped += 1
        if stats.get("collision"):  n_collision += 1

    n = args.episodes
    print(f"\n{'='*55}")
    print(f"EPISODE OUTCOMES ({n} episodes)")
    print(f"  TSR        : {n_success}/{n} ({100*n_success/n:.0f}%)")
    print(f"  Grasped    : {n_grasped}/{n} ({100*n_grasped/n:.0f}%)")
    print(f"  Collisions : {n_collision}/{n} ({100*n_collision/n:.0f}%)")
    print(f"{'='*55}")
    _quality_report(totals, all_mags, args.correction)
    print(f"Dataset saved to {out_dir}/")


if __name__ == "__main__":
    main()
