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
    _query_pi05_chunk,
    _init_pi05_client,
    _render_frame,
    _unwrap_sim,
    _compute_arm_link_constraints,
    _get_arm_body_ids,
    _get_arm_dof_indices,
    OPENVLA_URL,
    ObstacleConfig,
    detect_safelibero_obstacle,
    make_libero_env,
)
from experiments.cbf_ellipsoid import (
    ee_ellipsoid_center, EE_Q_DIAG_DEFAULT,
    run_sphere_decomp_cbf, get_ee_spheres, K_CBF,
)
from scipy.spatial.transform import Rotation as _SciRot

# MuJoCo primitive geom type IDs (mjtGeom enum values — stable across versions)
_MJG_SPHERE    = 2
_MJG_CAPSULE   = 3
_MJG_ELLIPSOID = 4
_MJG_CYLINDER  = 5
_MJG_BOX       = 6
_MJG_MESH      = 7

try:
    from experiments.cbf_visualizer import install_scene_hook, push_cbf_geoms
    from experiments.cbf_visualizer import _KNOWN_OBSTACLE_Q as _KNOWN_OB_Q
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False
    _KNOWN_OB_Q: dict = {}

_CV2_WINDOW = "collect_obstacle_data"

# ── CBF thresholds (used when --correction cbf) ───────────────────────────────
H_SAFE    = 0.15   # h above this → nom; below → CBF action
H_DISCARD = 0.0    # h below this → discard

# ── APF constants (used when --correction apf) ────────────────────────────────
D_DISCARD  = 0.05  # discard if within 5 cm of obstacle (unreliable zone)


# ─────────────────────────────────────────────────────────────────────────────
# Correction helpers
# ─────────────────────────────────────────────────────────────────────────────

def _ee_r_eff(n_hat_world: np.ndarray, R1: np.ndarray) -> float:
    """Effective EE ellipsoid radius (meters) in world-frame direction n_hat_world.

    EE_Q_DIAG_DEFAULT already holds semi-axes directly: [0.040, 0.115, 0.075] m
    (x=gripper-width, y=finger-spread, z=hand-depth). Do NOT take sqrt.
    """
    semi_axes = EE_Q_DIAG_DEFAULT                    # [0.040, 0.115, 0.075] m — direct semi-axes
    n_local   = R1.T @ n_hat_world                   # direction in EE frame
    n_scaled  = n_local / semi_axes
    return 1.0 / (float(np.linalg.norm(n_scaled)) + 1e-8)


def _cbf_correction(
    ee_pos: np.ndarray,
    R1: np.ndarray,
    near_ob: "ObstacleConfig",
    nom_xyz: np.ndarray,
    cbf_gamma: float,
    obstacle_safety_radius: float,
) -> tuple[np.ndarray, float]:
    """CBF filter with ellipsoidal EE model.

    h = dist(ee_center, ob_center) - r_eff(direction) - obstacle_radius
    where r_eff is the EE ellipsoid's radius in the direction toward the obstacle.
    """
    delta    = ee_pos - near_ob.pos
    d        = float(np.linalg.norm(delta)) + 1e-8
    n_hat    = delta / d                             # world frame: obstacle → EE
    r_eff    = _ee_r_eff(n_hat, R1)                 # EE ellipsoid radius in that dir
    h        = d - r_eff - obstacle_safety_radius
    hdot     = float(np.dot(n_hat, nom_xyz))
    safe_xyz = nom_xyz.copy()
    if hdot + cbf_gamma * h < 0:
        correction = -(hdot + cbf_gamma * h) * n_hat
        safe_xyz   = nom_xyz + correction
    return safe_xyz, h


# ── Geom-geometry helpers ────────────────────────────────────────────────────

def _geom_surface_dist_and_normal(
    ee_pos: np.ndarray,
    geom_id: int,
    model,
    data,
) -> tuple[float, np.ndarray]:
    """Signed distance (m) from ee_pos to a MuJoCo geom surface, plus outward
    unit normal (geom → EE) in world frame.  Handles sphere, box, capsule,
    cylinder; falls back to the stored bounding sphere for mesh / ellipsoid.
    Negative distance means ee_pos is inside the geom.
    """
    gtype = int(model.geom_type[geom_id])
    gsize = model.geom_size[geom_id]
    gpos  = data.geom_xpos[geom_id]
    gmat  = data.geom_xmat[geom_id].reshape(3, 3)
    p     = gmat.T @ (ee_pos - gpos)   # EE in geom-local frame

    if gtype == _MJG_SPHERE:
        r   = float(gsize[0])
        d   = float(np.linalg.norm(p))
        n_l = p / (d + 1e-8)
        return d - r, gmat @ n_l

    elif gtype == _MJG_BOX:
        half = gsize[:3].copy()
        if np.all(np.abs(p) < half):           # inside — find nearest face
            margins = half - np.abs(p)
            face    = int(np.argmin(margins))
            n_l     = np.zeros(3); n_l[face] = float(np.sign(p[face]))
            return -float(margins[face]), gmat @ n_l
        else:
            closest = np.clip(p, -half, half)
            diff    = p - closest
            d       = float(np.linalg.norm(diff))
            return d, gmat @ (diff / (d + 1e-8))

    elif gtype == _MJG_CAPSULE:
        r, hh = float(gsize[0]), float(gsize[1])   # radius, half-height (local z)
        p_z   = float(np.clip(p[2], -hh, hh))
        diff  = p - np.array([0.0, 0.0, p_z])
        d     = float(np.linalg.norm(diff))
        return d - r, gmat @ (diff / (d + 1e-8))

    elif gtype == _MJG_CYLINDER:
        r, hh = float(gsize[0]), float(gsize[1])
        xy    = p[:2]
        d_xy  = float(np.linalg.norm(xy))
        d_lat = d_xy - r          # neg inside lateral surface
        d_cap = abs(float(p[2])) - hh    # neg inside cap range

        if d_lat > 0 and d_cap <= 0:     # outside lateral only
            n_l = np.array([xy[0], xy[1], 0.0]) / (d_xy + 1e-8)
            return d_lat, gmat @ n_l
        elif d_lat <= 0 and d_cap > 0:   # outside cap only
            n_l = np.array([0.0, 0.0, float(np.sign(p[2]))])
            return d_cap, gmat @ n_l
        elif d_lat > 0 and d_cap > 0:    # corner region
            d   = float(np.sqrt(d_lat**2 + d_cap**2))
            nx  = xy / (d_xy + 1e-8)
            n_l = np.array([nx[0]*d_lat, nx[1]*d_lat, float(np.sign(p[2]))*d_cap])
            n_l /= (float(np.linalg.norm(n_l)) + 1e-8)
            return d, gmat @ n_l
        else:                             # inside — use closest surface
            if d_lat > d_cap:
                n_l = np.array([xy[0], xy[1], 0.0]) / (d_xy + 1e-8)
                return d_lat, gmat @ n_l
            else:
                n_l = np.array([0.0, 0.0, float(np.sign(p[2]))])
                return d_cap, gmat @ n_l

    else:   # MESH, ELLIPSOID, or unknown — use MuJoCo's stored bounding sphere
        r   = float(model.geom_rbound[geom_id])
        d   = float(np.linalg.norm(p))
        return d - r, gmat @ (p / (d + 1e-8))


_MJG_PRIMITIVES = {_MJG_SPHERE, _MJG_CAPSULE, _MJG_ELLIPSOID, _MJG_CYLINDER, _MJG_BOX}


def _get_obstacle_geom_ids(model, obstacle_name: str) -> list[int]:
    """Return PRIMITIVE collision geom IDs for an obstacle body.
    Mesh geoms are excluded: their bounding sphere (geom_rbound) is often much
    larger than the actual object and produces wrong CBF distances.
    """
    body_id = None
    for suffix in ("", "_main", "_base", "_body"):
        try:
            body_id = int(model.body(f"{obstacle_name}{suffix}").id)
            break
        except Exception:
            continue
    if body_id is None:
        return []
    ids = []
    for g in range(model.ngeom):
        if int(model.geom_bodyid[g]) != body_id:
            continue
        if int(model.geom_contype[g]) == 0 and int(model.geom_conaffinity[g]) == 0:
            continue
        if int(model.geom_type[g]) not in _MJG_PRIMITIVES:
            continue   # skip MESH and other non-analytic types
        ids.append(g)
    return ids


def _cbf_correction_geom(
    ee_pos: np.ndarray,
    R1: np.ndarray,
    all_geom_ids: list[int],
    model,
    data,
    nom_xyz: np.ndarray,
    cbf_gamma: float,
    safety_margin: float,
) -> tuple[np.ndarray, float]:
    """CBF correction using actual obstacle geom shapes + ellipsoidal EE.

    Loops over every collision geom across all obstacles, finds the worst-case
    (minimum h) geom, and applies the 1-D CBF projection in that direction.
    """
    h_min      = float("inf")
    best_n_hat = np.array([0.0, 0.0, 1.0])

    for gid in all_geom_ids:
        d_surf, n_world = _geom_surface_dist_and_normal(ee_pos, gid, model, data)
        # EE ellipsoid radius in the direction toward this geom surface
        r_eff = _ee_r_eff(-n_world, R1)   # n_world points *away* from geom
        h = d_surf - r_eff - safety_margin
        if h < h_min:
            h_min      = h
            best_n_hat = n_world

    hdot     = float(np.dot(best_n_hat, nom_xyz))
    safe_xyz = nom_xyz.copy()
    if hdot + cbf_gamma * h_min < 0:
        correction = -(hdot + cbf_gamma * h_min) * best_n_hat
        safe_xyz   = nom_xyz + correction
    return safe_xyz, h_min


def _known_ellipsoid_barrier(
    ee_pos: np.ndarray,
    R1: np.ndarray,
    obstacles: list,
    ob_mats: dict,
    nom_xyz: np.ndarray,
    cbf_gamma: float,
    safety_margin: float,
) -> tuple[np.ndarray, float]:
    """CBF for mesh-only obstacles using known per-class ellipsoid dimensions.

    ob_mats: {obstacle_name: 3×3 world-frame rotation matrix} captured from
    MuJoCo at episode start (orientation fixed since obstacles are static).
    Falls back to EE-ellipsoid + sphere if obstacle not in _KNOWN_OB_Q.
    """
    h_min      = float("inf")
    best_n_hat = np.array([0.0, 0.0, 1.0])

    for ob in obstacles:
        p = ee_pos - ob.pos            # world-frame EE – obstacle center
        d = float(np.linalg.norm(p)) + 1e-8
        n_hat = p / d

        if ob.name in _KNOWN_OB_Q:
            # Transform p into obstacle body frame, scale by semi-axes
            ob_mat  = ob_mats.get(ob.name, np.eye(3))
            semi    = _KNOWN_OB_Q[ob.name]           # [ax, ay, az] in body frame
            p_local = ob_mat.T @ p
            scaled  = p_local / semi
            d_ell   = float(np.linalg.norm(scaled))  # 1.0 on ellipsoid surface
            # Outward normal in world frame
            grad_local = p_local / (semi ** 2)
            n_world    = ob_mat @ grad_local
            n_world   /= (float(np.linalg.norm(n_world)) + 1e-8)
            # Distance from obstacle ellipsoid surface in metres (approx)
            d_surf = (d_ell - 1.0) * float(np.min(semi))
            r_eff  = _ee_r_eff(-n_world, R1)
            h      = d_surf - r_eff - safety_margin
            n_use  = n_world
        else:
            # No known shape: EE ellipsoid + sphere obstacle
            r_eff = _ee_r_eff(n_hat, R1)
            h     = d - r_eff - getattr(ob, "safety_radius", 0.10) - safety_margin
            n_use = n_hat

        if h < h_min:
            h_min      = h
            best_n_hat = n_use

    hdot     = float(np.dot(best_n_hat, nom_xyz))
    safe_xyz = nom_xyz.copy()
    if hdot + cbf_gamma * h_min < 0:
        correction = -(hdot + cbf_gamma * h_min) * best_n_hat
        safe_xyz   = nom_xyz + correction
    return safe_xyz, h_min


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
    vla: str = "openvla",
    pi05_host: str = "127.0.0.1",
    dagger_round: int = 0,
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
            show=show, vla=vla, pi05_host=pi05_host,
            dagger_round=dagger_round,
        )
    finally:
        sys.stdout = _real_stdout
        _tee.close()
        _real_stdout.write(f"  Log → {_log_path}\n")


def _collect_episode_body(
    env, lang: str, initial_states, ep_idx: int, *,
    openvla_port, replan_steps, horizon, correction, cbf_gamma,
    obstacle_safety_radius, k_rep, d_influence, out_dir, randomize_obstacle,
    cbf_near_goal_off, show, vla: str = "openvla", pi05_host: str = "127.0.0.1",
    dagger_round: int = 0,
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

    # ── MuJoCo model/data (sphere-decomp CBF + arm-link constraints) ─────────
    model, data = None, None
    arm_body_ids, arm_dof_idx, ee_body_id = [], [], 0
    _has_model = False
    try:
        model, data = _unwrap_sim(env)
        arm_body_ids = _get_arm_body_ids(model)
        arm_dof_idx  = _get_arm_dof_indices(model)
        ee_body_id   = arm_body_ids[-1] if arm_body_ids else 0
        _has_model   = True
    except Exception as _e:
        print(f"  [warn] MuJoCo model extraction failed: {_e}")

    # ── Visualization setup ───────────────────────────────────────────────────
    _viz_hook_ok = False
    if show and _HAS_VIZ and model is not None:
        try:
            _viz_hook_ok = install_scene_hook(env.sim)
            cv2.namedWindow(_CV2_WINDOW, cv2.WINDOW_AUTOSIZE)
        except Exception as _e:
            print(f"  [warn] viz init failed: {_e}")

    # ── Sphere decomposition (once per episode; obstacles are static) ─────────
    _sphere_decomp = None
    if correction == "cbf" and _has_model and obstacles:
        try:
            from experiments.cbf_visualizer import decompose_obstacle_to_spheres
            ob0_name = obstacles[0].name
            for suffix in ("_main", ""):
                _sphere_decomp = decompose_obstacle_to_spheres(
                    model, data, f"{ob0_name}{suffix}",
                    n_spheres=48,
                    r_sphere=0.015,
                    safety_margin=0.010,
                )
                if _sphere_decomp is not None:
                    break
            if _sphere_decomp is not None:
                print(f"  Sphere decomp: {len(_sphere_decomp)} spheres for '{ob0_name}'")
            else:
                print(f"  [warn] Sphere decomp failed — falling back to geom/ellipsoid CBF")
        except Exception as _sd_e:
            print(f"  [warn] Sphere decomp import error: {_sd_e}")

    # ── Obstacle geom IDs (for geom-geometry CBF; obstacles are static) ───────
    # Maps obstacle name → list of collision geom IDs.  Captures actual shape
    # and current orientation (e.g. book face-down vs. upright) via geom_xmat.
    _obstacle_geom_ids: dict[str, list[int]] = {}
    _all_obstacle_geom_ids: list[int] = []
    _obstacle_ob_mats: dict[str, np.ndarray] = {}   # obstacle body rotation (static)
    _mesh_only_obstacles: list = []                 # obstacles with no primitive geoms

    if correction == "cbf" and _has_model and obstacles:
        import mujoco as _mj
        for _ob in obstacles:
            # Cache obstacle body rotation for known-ellipsoid path
            _bid = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, _ob.name)
            for _sfx in ("", "_main", "_base", "_body"):
                _bid2 = _mj.mj_name2id(model, _mj.mjtObj.mjOBJ_BODY, f"{_ob.name}{_sfx}")
                if _bid2 >= 0:
                    _bid = _bid2; break
            if _bid >= 0:
                _obstacle_ob_mats[_ob.name] = data.xmat[_bid].reshape(3, 3).copy()

            _gids = _get_obstacle_geom_ids(model, _ob.name)
            if _gids:
                _obstacle_geom_ids[_ob.name] = _gids
                _all_obstacle_geom_ids.extend(_gids)
                _type_names = {_MJG_SPHERE:"sphere", _MJG_BOX:"box",
                               _MJG_CAPSULE:"capsule", _MJG_CYLINDER:"cylinder",
                               _MJG_MESH:"mesh", _MJG_ELLIPSOID:"ellipsoid"}
                _desc = ", ".join(_type_names.get(int(model.geom_type[g]), "?") for g in _gids)
                print(f"  Geom CBF: '{_ob.name}' → {len(_gids)} primitive geom(s): [{_desc}]")
            elif _ob.name in _KNOWN_OB_Q:
                _mesh_only_obstacles.append(_ob)
                print(f"  Ellipsoid CBF: '{_ob.name}' → known shape {_KNOWN_OB_Q[_ob.name]}")
            else:
                _mesh_only_obstacles.append(_ob)
                print(f"  [warn] '{_ob.name}': no primitives, no known shape → sphere fallback")

    # ── Collision tracking (displacement-based, matches SafeLIBERO metric) ────
    # Baseline is captured AFTER the first physics step so teleport-settling
    # micro-displacements (physics resolving the new contact state) don't
    # flag as collisions before the robot has moved.
    _initial_obstacle_pos = None   # set lazily after step 0
    _collision_flag = False
    _obs_key_coll = f"{obstacles[0].name}_pos" if obstacles else None

    # ── Object tracking ───────────────────────────────────────────────────────
    # Filter to objects actually in the workspace (exclude off-table stored objects
    # which LIBERO places at x/y = ±10 or ±20 to keep them out of the scene).
    def _in_workspace(p):
        return abs(p[0]) < 2.0 and abs(p[1]) < 2.0 and p[2] > 0.3
    _obj_pos_keys = sorted([
        k for k in obs.keys()
        if k.endswith("_pos") and not k.startswith("robot") and "to_robot" not in k
        and _in_workspace(np.array(obs[k], dtype=float))
    ])
    _obstacle_key_set = {f"{ob.name}_pos" for ob in obstacles}
    _obj_initial_z: dict[str, float] = {}   # populated at step 5 after physics settle
    _target_keys      = [k for k in _obj_pos_keys if k not in _obstacle_key_set]

    # ── Episode-level debug state ─────────────────────────────────────────────
    _success      = False
    _max_lift     = 0.0          # max z-rise of any target object
    _grasp_step   = None         # first step where lift > 2 cm
    _grasp_obj    = None

    _ee_semi = EE_Q_DIAG_DEFAULT   # semi-axes: x=gripper-width, y=finger-spread, z=hand-depth
    print(f"\n  [ep {ep_idx:03d}] obstacle={obstacles[0].name if obstacles else 'none'}"
          f"  correction={correction.upper()}  lang=\"{lang}\""
          f"\n  EE semi-axes: x={_ee_semi[0]:.3f} y={_ee_semi[1]:.3f} z={_ee_semi[2]:.3f} m"
          f"  safety_r={obstacle_safety_radius:.3f}m  gamma={cbf_gamma:.1f}"
          f"  H_SAFE={H_SAFE:.3f}")

    vla_cnt = 0

    buf_img      = []
    buf_wrist    = []
    buf_proprio  = []
    buf_obs_feat = []
    buf_nom_act  = []
    buf_safe_act = []   # APF- or CBF-corrected (= nom_act when label=0)
    buf_dist     = []   # Euclidean distance to nearest obstacle centre
    buf_h_min    = []   # CBF barrier value h (sphere-to-sphere); inf for non-CBF steps
    buf_corr_mag = []   # correction magnitude (0 if nom)
    buf_label    = []   # 0=nom, 1=safe, -1=discard

    action_queue: list[np.ndarray] = []
    nom_queue:    list[np.ndarray] = []

    # CBF near-goal deactivation: once gripper has been closing for 2+ steps, stop CBF.
    # This lets the VLA complete the grasp unimpeded once it's committed to the target.
    _cbf_active          = True
    _gripper_close_steps = 0

    url = f"http://127.0.0.1:{openvla_port}/act"

    if vla == "pi05":
        _init_pi05_client(pi05_host, openvla_port)

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
        _R1       = _SciRot.from_quat(ee_quat).as_matrix()
        gq        = np.array(obs.get("robot0_gripper_qpos", [0.04, 0.04]), dtype=float)

        # ── Deferred initial-z capture (after physics settle at step 5) ────────
        if t == 5 and not _obj_initial_z:
            _obj_initial_z = {k: float(obs[k][2]) for k in _obj_pos_keys if k in obs}
            print(f"  [settled] Objects: " + "  ".join(
                f"{k.replace('_pos','')}@z={_obj_initial_z[k]:.3f}"
                for k in _obj_pos_keys if k in _obj_initial_z
            ))

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
                if vla == "pi05":
                    raw_chunk = _query_pi05_chunk(img, wrist_img, state, lang,
                                                  num_actions=replan_steps)
                    action_queue = list(raw_chunk)
                    nom_queue    = list(raw_chunk)
                else:
                    raw_chunk = _query_openvla_chunk(img, wrist_img, state, lang,
                                                     num_actions=replan_steps, url=url)
                    action_queue = [_post_process_vla(a) for a in raw_chunk]
                    nom_queue    = [_post_process_vla(a) for a in raw_chunk]
                vla_ms = (time.perf_counter() - _t0) * 1000
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

        _step_h = float("inf")  # CBF barrier value for this step; inf for non-CBF

        if correction == "none":
            safe_xyz = nom_xyz.copy()
            dist     = float(np.linalg.norm(ee_pos - near_ob.pos))
            corr_mag = 0.0
            label    = 0

        elif correction == "apf":
            safe_xyz, dist, corr_mag = _apf_correction(
                ee_pos, near_ob, nom_xyz, k_rep=k_rep, d_influence=d_influence)
            if dist < D_DISCARD:
                label = -1
            elif dist < d_influence:
                label = 1
            else:
                label = 0

        else:  # cbf
            dist = float(np.linalg.norm(ee_pos - near_ob.pos))

            # ── Compute h_min and worst-case outward normal ───────────────────
            # Priority: sphere decomp (tightest, shape-accurate) >
            #           primitive geoms (exact analytic) >
            #           known ellipsoid (mesh obstacles) >
            #           sphere fallback
            h_min      = float("inf")
            best_n_hat = np.array([0.0, 0.0, 1.0])  # unit normal: obstacle→EE

            if _sphere_decomp is not None:
                # Each EE sphere vs. every obstacle surface sphere — linear h in metres
                _ee_sph = get_ee_spheres(ee_pos, _R1)
                for _ee_c, _ee_r in _ee_sph:
                    for _ob_c, _ob_r in _sphere_decomp:
                        _diff = _ee_c - _ob_c
                        _d    = float(np.linalg.norm(_diff)) + 1e-8
                        _h    = _d - (_ee_r + _ob_r)
                        if _h < h_min:
                            h_min      = _h
                            best_n_hat = _diff / _d

            elif _all_obstacle_geom_ids and _has_model:
                for _gid in _all_obstacle_geom_ids:
                    _d_surf, _n_w = _geom_surface_dist_and_normal(ee_pos, _gid, model, data)
                    _h = _d_surf - _ee_r_eff(-_n_w, _R1) - obstacle_safety_radius
                    if _h < h_min:
                        h_min      = _h
                        best_n_hat = _n_w
                for _mob in _mesh_only_obstacles:
                    if _mob.name in _KNOWN_OB_Q:
                        _ob_mat = _obstacle_ob_mats.get(_mob.name, np.eye(3))
                        _semi   = _KNOWN_OB_Q[_mob.name]
                        _p_l    = _ob_mat.T @ (ee_pos - _mob.pos)
                        _d_ell  = float(np.linalg.norm(_p_l / _semi))
                        _d_s    = (_d_ell - 1.0) * float(np.min(_semi))
                        _g_l    = _p_l / (_semi ** 2)
                        _n_w2   = _ob_mat @ _g_l; _n_w2 /= float(np.linalg.norm(_n_w2)) + 1e-8
                        _h      = _d_s - _ee_r_eff(-_n_w2, _R1) - obstacle_safety_radius
                    else:
                        _d_ob   = float(np.linalg.norm(ee_pos - _mob.pos)) + 1e-8
                        _n_w2   = (ee_pos - _mob.pos) / _d_ob
                        _h      = _d_ob - _ee_r_eff(_n_w2, _R1) - getattr(_mob, "safety_radius", 0.10) - obstacle_safety_radius
                    if _h < h_min:
                        h_min = _h; best_n_hat = _n_w2

            else:
                # Last resort: EE ellipsoid vs. obstacle point / known ellipsoid
                _ob_list = _mesh_only_obstacles if _mesh_only_obstacles else obstacles
                for _ob in _ob_list:
                    if _ob.name in _KNOWN_OB_Q:
                        _ob_mat = _obstacle_ob_mats.get(_ob.name, np.eye(3))
                        _semi   = _KNOWN_OB_Q[_ob.name]
                        _p_l    = _ob_mat.T @ (ee_pos - _ob.pos)
                        _d_ell  = float(np.linalg.norm(_p_l / _semi))
                        _d_s    = (_d_ell - 1.0) * float(np.min(_semi))
                        _g_l    = _p_l / (_semi ** 2)
                        _n_w    = _ob_mat @ _g_l; _n_w /= float(np.linalg.norm(_n_w)) + 1e-8
                        _h      = _d_s - _ee_r_eff(-_n_w, _R1) - obstacle_safety_radius
                    else:
                        _d_ob = float(np.linalg.norm(ee_pos - _ob.pos)) + 1e-8
                        _n_w  = (ee_pos - _ob.pos) / _d_ob
                        _h    = _d_ob - _ee_r_eff(_n_w, _R1) - obstacle_safety_radius
                    if _h < h_min:
                        h_min = _h; best_n_hat = _n_w

            # ── Apply 1-D CBF projection ──────────────────────────────────────
            if not _cbf_active:
                safe_xyz = nom_xyz.copy()
                corr_mag = 0.0
                label    = 0
            else:
                safe_xyz = nom_xyz.copy()
                try:
                    hdot = float(np.dot(best_n_hat, nom_xyz))
                    if hdot + cbf_gamma * h_min < 0:
                        _corr_vec = -(hdot + cbf_gamma * h_min) * best_n_hat
                        safe_xyz  = nom_xyz + _corr_vec
                except Exception as _e:
                    print(f"  [warn] CBF exception: {_e}")

                corr_mag = float(np.linalg.norm(safe_xyz - nom_xyz))
                _step_h  = h_min  # capture barrier value for this step
                if h_min < H_DISCARD:
                    label = -1
                elif h_min <= H_SAFE:
                    label = 1
                else:
                    label = 0

        # Clamp corrected xyz to the same range as π0.5 nominal outputs so
        # cbf_act stays in-distribution for fine-tuning (large CBF corrections
        # near obstacles can push xyz outside [-1, 1]).
        safe_xyz = np.clip(safe_xyz, -1.0, 1.0)

        safe_action = nom_action.copy()
        safe_action[:3] = safe_xyz

        buf_img.append(img)
        buf_wrist.append(wrist_img)
        buf_proprio.append(state.astype(np.float32))
        buf_obs_feat.append(obs_feat.astype(np.float32))
        buf_nom_act.append(nom_action[:7].astype(np.float32))
        buf_safe_act.append(safe_action[:7].astype(np.float32))
        buf_dist.append(float(dist))
        buf_h_min.append(float(_step_h))
        buf_corr_mag.append(float(corr_mag))
        buf_label.append(int(label))

        # Always execute the safe action; label only controls training-data recording.
        # When label==0 (far), safe_xyz==nom_xyz so no behavioral difference.
        # When label==-1 (h<0, already violated), we MUST apply the correction to
        # push the EE back out — executing nom_action here caused the collisions.
        exec_action = safe_action
        _cbf_on = (label == 1)

        # ── Near-grasp commit ─────────────────────────────────────────────────
        # When EE is within 12 cm of the nearest graspable target: lock gripper
        # closed, force descent to target_z + 1 cm, block upward escape until
        # fingers physically close.  Mirrors the same block in libero_runner.py.
        _GC_Z_STOP  = 0.010
        _GC_DESCENT = -0.35
        _gc_near_key, _gc_near_dist = None, float("inf")
        for _gck in _target_keys:
            if _gck not in obs:
                continue
            _gcp = np.array(obs[_gck], dtype=float)
            if _gcp[2] > 1.02:   # skip objects already at goal/placement height
                continue
            _d = float(np.linalg.norm(ee_pos - _gcp))
            if _d < _gc_near_dist:
                _gc_near_dist, _gc_near_key = _d, _gck
        _in_gc_range = (correction != "none"
                        and vla != "pi05"
                        and _gc_near_key is not None
                        and _gc_near_dist < 0.12
                        and _grasp_step is None)

        # ── Pre-open: keep gripper open during far approach ──────────────────
        # OpenVLA closes the gripper from step 1 due to distribution shift.
        # Not needed for π0.5 which handles gripper timing natively.
        _PRE_OPEN_DIST = 0.15   # m — force open beyond this range
        _pre_open_active = (
            vla != "pi05"
            and correction != "none"
            and _gc_near_key is not None
            and _gc_near_dist > _PRE_OPEN_DIST
            and _grasp_step is None
        )

        if _in_gc_range:
            _gc_tgt_pos = np.array(obs[_gc_near_key], dtype=float)
            _gc_tgt_z   = float(_gc_tgt_pos[2])
            exec_action = exec_action.copy()
            exec_action[6] = 1.0   # lock gripper CLOSED
            _gc_xy_err  = _gc_tgt_pos[:2] - ee_pos[:2]
            _gc_xy_pull = np.clip(0.4 * _gc_xy_err, -0.012, 0.012)
            exec_action[0] += _gc_xy_pull[0]
            exec_action[1] += _gc_xy_pull[1]
            if ee_pos[2] > _gc_tgt_z + _GC_Z_STOP:
                exec_action[2] = min(exec_action[2], _GC_DESCENT)
                if t % 8 == 0:
                    print(f"  [{t:03d}] GC: descend+pull → "
                          f"{_gc_near_key.replace('_pos','')} "
                          f"EEz={ee_pos[2]:.3f} tgtz={_gc_tgt_z:.3f} "
                          f"xy_err=[{_gc_xy_err[0]:+.3f},{_gc_xy_err[1]:+.3f}]")
            # Block upward escape until fingers physically close
            _gc_gq = np.abs(np.array(obs.get("robot0_gripper_qpos",
                                               [0.04, 0.04]), dtype=float))
            if float(np.max(_gc_gq)) > 0.015:   # fingers still open
                exec_action[2] = min(exec_action[2], 0.0)
        elif _pre_open_active:
            exec_action = exec_action.copy()
            exec_action[6] = -1.0  # force gripper OPEN

        _h_tag = ""
        if correction == "cbf":
            if h_min < 0:
                _h_tag = f"  h={h_min:.3f}!!! VIOLATED corr={corr_mag:.4f}"
            elif h_min < H_SAFE:
                _h_tag = f"  h={h_min:.3f}!  CBF_ON corr={corr_mag:.4f}"
            else:
                _h_tag = f"  h={h_min:.3f}"
        print(
            f"  act nom=[{nom_action[0]:+.4f},{nom_action[1]:+.4f},{nom_action[2]:+.4f}"
            f" r={nom_action[3]:+.3f},{nom_action[4]:+.3f},{nom_action[5]:+.3f}"
            f" g={nom_action[6]:+.0f}]"
            + _h_tag
            + (f"  exec=[{exec_action[0]:+.4f},{exec_action[1]:+.4f},{exec_action[2]:+.4f}]"
               if _cbf_on or (correction == "cbf" and h_min < 0) else "")
            + (f"  GC→z={exec_action[2]:+.4f}" if _in_gc_range else "")
            + (f"  OPN" if _pre_open_active else "")
        )
        step_out = env.step(exec_action)
        obs = step_out[0] if isinstance(step_out[0], dict) else step_out[0]
        done = step_out[2] if len(step_out) == 4 else (step_out[2] or step_out[3])

        # ── Lift / grasp detection ────────────────────────────────────────────
        for _k in _target_keys:
            if _k not in obs or _k not in _obj_initial_z:
                continue
            _lift = float(obs[_k][2]) - _obj_initial_z[_k]
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

        # ── cv2 display ───────────────────────────────────────────────────────
        if show and img is not None:
            if _viz_hook_ok and obstacles and model is not None:
                try:
                    _ee_c  = ee_ellipsoid_center(ee_pos, _R1)
                    push_cbf_geoms(
                        env.sim,
                        ee_center=_ee_c, ee_q=EE_Q_DIAG_DEFAULT, R1=_R1,
                        obstacles=obstacles,
                        h_values=[h_min if correction == "cbf" else dist],
                        arm_body_ids=[], arm_radii={},
                        model=model, data=data,
                        cbf_triggered=(correction == "cbf" and h_min <= H_SAFE),
                        obstacle_spheres=_sphere_decomp, arm_sample_positions=None,
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
        _h_vals = np.array(buf_h_min, dtype=np.float32)[keep]
        _finite_h = _h_vals[np.isfinite(_h_vals)]
        _cbf_steps = int((labels[keep] == 1).sum())

        np.savez_compressed(
            out_dir / f"ep_{ep_idx:04d}.npz",
            # ── Per-step arrays ──────────────────────────────────────────────
            img        = np.array(buf_img,      dtype=np.uint8)[keep],
            wrist      = np.array(buf_wrist,    dtype=np.uint8)[keep],
            proprio    = np.array(buf_proprio,  dtype=np.float32)[keep],
            obs_feat   = np.array(buf_obs_feat, dtype=np.float32)[keep],
            nom_act    = np.array(buf_nom_act,  dtype=np.float32)[keep],
            cbf_act    = np.array(buf_safe_act, dtype=np.float32)[keep],
            dist       = np.array(buf_dist,     dtype=np.float32)[keep],
            h_min      = _h_vals,
            corr_mag   = np.array(buf_corr_mag, dtype=np.float32)[keep],
            label      = labels[keep],
            # ── Per-episode scalars ──────────────────────────────────────────
            lang         = np.array([lang]),
            success      = np.array([int(_success)]),
            collision    = np.array([int(_collision_flag)]),
            grasp_step   = np.array([int(_grasp_step) if _grasp_step is not None else -1]),
            # DAgger loop metadata — key metrics for the comparison table
            dagger_round  = np.array([int(dagger_round)]),
            n_cbf_steps   = np.array([_cbf_steps]),           # CBF activations/episode
            cbf_rate      = np.array([_cbf_steps / max(n_total, 1)], dtype=np.float32),
            min_h         = np.array([float(_finite_h.min()) if len(_finite_h) else float("inf")], dtype=np.float32),
            mean_h        = np.array([float(_finite_h.mean()) if len(_finite_h) else float("inf")], dtype=np.float32),
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

def _run_task(args, task_idx: int) -> tuple[dict, list[float], int, int, int]:
    """Collect episodes for a single task. Returns (totals, mags, n_success, n_grasped, n_collision)."""
    scene_mode = "rand" if args.randomize_obstacle else "orig"
    out_dir = Path(args.out) / f"{args.suite}_t{task_idx:02d}_L{args.safety_level}_{scene_mode}"
    print(f"\n  Output → {out_dir}")

    env, lang, initial_states = make_libero_env(
        task_suite=args.suite,
        task_idx=task_idx,
        safety_level=args.safety_level,
        has_renderer=False,
        horizon=args.horizon,
    )

    totals      = {"steps": 0, "discarded": 0, "nom": 0, "safe": 0}
    all_mags:     list[float] = []
    n_success   = 0
    n_grasped   = 0
    n_collision = 0

    for ep in range(args.episodes):
        # Resume: skip episodes whose .npz already exists from a previous run.
        _ep_npz = out_dir / f"ep_{ep:04d}.npz"
        if _ep_npz.exists():
            print(f"  ep {ep:03d}: already exists, skipping")
            continue

        try:
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
                vla=args.vla,
                pi05_host=args.pi05_host,
                dagger_round=args.dagger_round,
            )
        except KeyboardInterrupt:
            raise  # let Ctrl-C propagate to kill the whole run
        except Exception as _ep_err:
            print(f"\n  [ERROR] ep {ep:03d} crashed: {_ep_err} — skipping and continuing\n")
            import traceback; traceback.print_exc()
            continue

        for k in ("steps", "discarded", "nom", "safe"):
            totals[k] += stats[k]
        all_mags.extend(stats["corr_mags"])
        if stats.get("success"):                    n_success   += 1
        if stats.get("grasp_step") is not None:     n_grasped   += 1
        if stats.get("collision"):                  n_collision += 1

    n = args.episodes
    print(f"\n{'='*55}")
    print(f"TASK {task_idx} OUTCOMES ({n} episodes)")
    print(f"  TSR        : {n_success}/{n} ({100*n_success/n:.0f}%)")
    print(f"  Grasped    : {n_grasped}/{n} ({100*n_grasped/n:.0f}%)")
    print(f"  Collisions : {n_collision}/{n} ({100*n_collision/n:.0f}%)")
    print(f"{'='*55}")
    _quality_report(totals, all_mags, args.correction)
    print(f"  Data saved → {out_dir}/")

    try:
        env.close()
    except Exception:
        pass

    return totals, all_mags, n_success, n_grasped, n_collision


def main():
    parser = argparse.ArgumentParser(description="Collect obstacle-conditioned DAgger data")
    parser.add_argument("--suite",        default="safelibero_spatial")
    parser.add_argument("--task",         type=int, default=0,
                        help="Task index (ignored when --all-tasks is set)")
    parser.add_argument("--all-tasks",    action="store_true",
                        help="Run ALL tasks in the suite sequentially (for overnight collection)")
    parser.add_argument("--safety-level", default="I")
    parser.add_argument("--episodes",    type=int, default=60)
    parser.add_argument("--horizon",     type=int, default=400)
    parser.add_argument("--replan-steps", type=int, default=8)
    # Correction mode
    parser.add_argument("--correction", choices=["cbf", "apf", "none"], default="apf",
                        help="cbf: reactive CBF filter; apf: smooth APF repulsion (default); none: pure VLA")
    # APF params
    parser.add_argument("--k-rep",       type=float, default=2.0)
    parser.add_argument("--d-influence", type=float, default=0.20,
                        help="APF influence radius from obstacle SURFACE in metres (default 0.20)")
    # CBF params
    parser.add_argument("--cbf-gamma",     type=float, default=1.8)
    parser.add_argument("--safety-radius", type=float, default=0.10)
    # Data augmentation
    parser.add_argument("--randomize-obstacle", action="store_true")
    parser.add_argument("--cbf-near-goal-off",  action="store_true")
    # IO / display
    parser.add_argument("--show",        action="store_true")
    parser.add_argument("--vla",         choices=["openvla", "pi05"], default="openvla")
    parser.add_argument("--openvla-port", type=int, default=8000)
    parser.add_argument("--pi05-host",   default="127.0.0.1")
    parser.add_argument("--out",         default="data/obs_cond_dataset")
    # Iterative DAgger loop metadata
    parser.add_argument("--dagger-round", type=int, default=0,
                        help="DAgger round index (0=base VLA, 1=first fine-tune, ...). "
                             "Saved per episode so rounds can be separated or combined.")
    args = parser.parse_args()

    # Determine which tasks to run
    if args.all_tasks:
        from libero.libero import benchmark as _lb
        _bd   = _lb.get_benchmark_dict()
        _is_s = args.suite.startswith("safelibero_")
        _so   = _bd[args.suite](safety_level=args.safety_level) if _is_s else _bd[args.suite]()
        task_indices = list(range(_so.get_num_tasks()))
    else:
        task_indices = [args.task]

    scene_mode = "rand" if args.randomize_obstacle else "orig"
    print(f"{'='*60}")
    print(f"DAgger data collection")
    print(f"  suite={args.suite}  tasks={task_indices}  episodes/task={args.episodes}")
    print(f"  correction={args.correction}  vla={args.vla}  level={args.safety_level}")
    print(f"  out={args.out}/")
    print(f"{'='*60}")

    grand = {"steps": 0, "discarded": 0, "nom": 0, "safe": 0}
    grand_mags:      list[float] = []
    grand_success    = 0
    grand_grasped    = 0
    grand_collision  = 0

    import time as _time
    _t_start = _time.time()

    for i, task_idx in enumerate(task_indices):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}  ({i+1}/{len(task_indices)})")
        print(f"{'='*60}")
        try:
            totals, mags, ns, ng, nc = _run_task(args, task_idx)
        except KeyboardInterrupt:
            raise
        except Exception as _task_err:
            print(f"\n  [ERROR] Task {task_idx} crashed: {_task_err} — skipping to next task\n")
            import traceback; traceback.print_exc()
            continue

        for k in ("steps", "discarded", "nom", "safe"):
            grand[k] += totals[k]
        grand_mags.extend(mags)
        grand_success   += ns
        grand_grasped   += ng
        grand_collision += nc

        elapsed = _time.time() - _t_start
        remaining = (elapsed / (i + 1)) * (len(task_indices) - i - 1)
        print(f"\n  Elapsed: {elapsed/3600:.1f}h   Est. remaining: {remaining/3600:.1f}h")

    if len(task_indices) > 1:
        grand_ep = len(task_indices) * args.episodes
        print(f"\n{'='*60}")
        print(f"ALL TASKS COMPLETE — {len(task_indices)} tasks × {args.episodes} episodes")
        print(f"  TSR        : {grand_success}/{grand_ep} ({100*grand_success/grand_ep:.0f}%)")
        print(f"  Grasped    : {grand_grasped}/{grand_ep} ({100*grand_grasped/grand_ep:.0f}%)")
        print(f"  Collisions : {grand_collision}/{grand_ep} ({100*grand_collision/grand_ep:.0f}%)")
        _quality_report(grand, grand_mags, args.correction)
        print(f"{'='*60}")
        print(f"\nDataset saved to {args.out}/")


if __name__ == "__main__":
    main()
