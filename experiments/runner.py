"""
Experiment runner: runs one (scene, mode) trial and returns a MetricsTracker.

Architecture
------------
  High-level planner  : OpenVLA-7B, background thread, ~5–10 Hz
  Ghost-target layer  : goal attractor + VLA action + soft potential-field
                        repulsion (both modes — ensures fair comparison)
  Safety filter       : formal kinematic CBF-QP on joint velocities (CBF mode only)
  Physics / viewer    : MuJoCo, ~100 Hz
"""

from __future__ import annotations

import contextlib
import copy
import os
import tempfile
import time
import threading
import mujoco
import mujoco.viewer
import numpy as np
import cv2
import requests
import base64
from PIL import Image
import io
from scipy.optimize import minimize

from experiments.scene_config import SceneConfig, ObstacleConfig
from experiments.xml_builder import build_scene_xml, PANDA_DIR
from experiments.metrics import MetricsTracker, StepRecord

# ---------------------------------------------------------------------------
# Server / VLA constants (shared across all scenes)
# ---------------------------------------------------------------------------
OPENVLA_URL    = "http://127.0.0.1:8000/act"
# Bridge V2 prompts are short, imperative, describe the object and destination.
TEXT_GOAL      = "move the red block to the left, avoiding the obstacle."
ARM_BODY_NAMES = ["link3", "link4", "link5", "link6", "link7", "hand"]

# Franka "ready" configuration — puts EE at roughly [0.31, 0, 0.59].
# Runner drives the arm from here to cfg.start_pos during warm-up before
# the VLA and metrics are engaged.
_QPOS_READY = np.array([0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854])
N_WARMUP    = 150   # IK-only steps to reach start_pos before experiment begins


# ---------------------------------------------------------------------------
# Async VLA worker
# ---------------------------------------------------------------------------
_vla_lock    = threading.Lock()
_vla_image   = None
_delta_ema   = np.zeros(3)
_vla_running = False


# ---------------------------------------------------------------------------
# Headless viewer shim + episode randomization
# ---------------------------------------------------------------------------
class _NullViewer:
    """Drop-in replacement for the passive viewer in headless mode."""
    def sync(self): pass


@contextlib.contextmanager
def _viewer_ctx(model, data, headless: bool):
    if headless:
        yield _NullViewer()
    else:
        with mujoco.viewer.launch_passive(model, data) as v:
            yield v


def sample_scene(cfg: SceneConfig) -> SceneConfig:
    """Return a deep copy of cfg with each obstacle position randomly perturbed.

    Noise is drawn uniformly from [-pos_noise_range, +pos_noise_range] per axis.
    Scenes without pos_noise_range set (all zeros) are returned unchanged.
    Used by run_benchmark.py to implement the SafeLIBERO randomisation protocol.
    """
    new_cfg = copy.deepcopy(cfg)
    for obs in new_cfg.obstacles:
        if np.any(obs.pos_noise_range > 0):
            obs.pos += np.random.uniform(-obs.pos_noise_range, obs.pos_noise_range)
    return new_cfg


def _vla_worker(ema_alpha: float):
    global _delta_ema, _vla_running
    print("  [VLA Thread] started.")
    while _vla_running:
        with _vla_lock:
            img = _vla_image
        if img is not None:
            try:
                action = _query_openvla(img, TEXT_GOAL)
                # Bridge actions after unnorm are meter-scale deltas (~0.01–0.05 m).
                # Clip rather than tanh to preserve the action distribution.
                delta = np.clip(action[:3], -0.05, 0.05)
                with _vla_lock:
                    _delta_ema[:] = ema_alpha * _delta_ema + (1 - ema_alpha) * delta
            except Exception as e:
                print(f"  [VLA Thread] query failed: {e}")
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------
def _preprocess_rgb(img_bgr: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (224, 224)).astype(np.uint8)


def _image_to_base64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _query_openvla(img_rgb: np.ndarray, instruction: str) -> np.ndarray:
    payload = {"image_base64": _image_to_base64(img_rgb), "instruction": instruction}
    r = requests.post(OPENVLA_URL, json=payload, timeout=60)
    r.raise_for_status()
    d = r.json()
    if not d.get("action"):
        raise RuntimeError(f"VLA server error: {d}")
    return np.array(d["action"], dtype=np.float32)


# ---------------------------------------------------------------------------
# Arm monitoring
# ---------------------------------------------------------------------------
def _resolve_arm_bodies(model):
    resolved = []
    for name in ARM_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            resolved.append((name, bid))
    if not resolved:
        raise RuntimeError("No arm bodies found.")
    print(f"  Monitoring {len(resolved)} arm bodies: {[n for n, _ in resolved]}")
    return resolved


def _arm_obs_distances(data, arm_bodies: list, obstacles: list[ObstacleConfig]):
    """
    Returns (min_dist, closest_obstacle_name, closest_body_name) across all
    (link, obstacle) pairs.
    """
    min_d    = float("inf")
    min_obs  = None
    min_body = None
    for obs in obstacles:
        for name, bid in arm_bodies:
            pos  = data.xpos[bid]
            dist = float(np.linalg.norm(pos - obs.pos))
            if dist < min_d:
                min_d    = dist
                min_obs  = obs.name
                min_body = name
    return min_d, min_obs, min_body


def _any_violation(data, arm_bodies: list, obstacles: list[ObstacleConfig]) -> bool:
    for obs in obstacles:
        for _, bid in arm_bodies:
            if np.linalg.norm(data.xpos[bid] - obs.pos) < obs.safety_radius:
                return True
    return False


# ---------------------------------------------------------------------------
# Ghost-target potential-field repulsion
# Pushes the ghost target (Cartesian) away from obstacles.
# Applied in BOTH modes to ensure a fair comparison: the only difference
# between plain and CBF is the joint-velocity filter.
# ---------------------------------------------------------------------------
def _potential_repulsion(ghost_pos: np.ndarray, obstacles: list[ObstacleConfig],
                         gain: float, cutoff: float) -> np.ndarray:
    force = np.zeros(3)
    for obs in obstacles:
        diff = ghost_pos - obs.pos
        dist = np.linalg.norm(diff)
        if 0 < dist < cutoff:
            # Inverse-square repulsion, clamped at cutoff
            magnitude = gain * (1.0 / dist - 1.0 / cutoff) / (dist ** 2)
            force += magnitude * (diff / dist)
    return force


# ---------------------------------------------------------------------------
# Formal kinematic CBF-QP filter
# ---------------------------------------------------------------------------
def _run_cbf(model, data, arm_bodies: list, obstacles: list[ObstacleConfig],
             q_dot_nom: np.ndarray, gamma: float) -> tuple[np.ndarray, float]:
    """
    Solves:  min  0.5 * ||u - u_nom||^2
             s.t. for each (link, obstacle): CBF constraint satisfied

    Returns (u_safe, correction_norm).
    """
    num_joints = 7
    u_nom = np.asarray(q_dot_nom)

    constraints = []
    for obs in obstacles:
        for name, bid in arm_bodies:
            p_link = data.xpos[bid].copy()
            dist   = np.linalg.norm(p_link - obs.pos)
            h      = dist ** 2 - obs.safety_radius ** 2

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)
            J_link = jacp[:, :num_joints]

            def _con(u, p_l=p_link, J_l=J_link, h_val=h):
                diff = p_l - obs.pos
                A_i  = -2 * diff.dot(J_l)
                # Scale b_i by step_scale so the constraint is consistent
                # with the actual integration gain applied later.
                b_i  = gamma * h_val * 0.4
                return b_i - A_i.dot(u)

            constraints.append({"type": "ineq", "fun": _con})

    res = minimize(
        lambda u: 0.5 * np.sum((u - u_nom) ** 2),
        x0=u_nom,
        method="SLSQP",
        constraints=constraints,
    )

    if res.success:
        u_safe = res.x
    else:
        u_safe = np.zeros(num_joints)

    correction_norm = float(np.linalg.norm(u_safe - u_nom))
    cbf_triggered   = correction_norm > 1e-4
    return u_safe, correction_norm, cbf_triggered


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def run_trial(cfg: SceneConfig, use_cbf: bool,
              results_dir: str = "results",
              headless: bool = False,
              save_results: bool = True) -> MetricsTracker:
    """
    Runs one trial for the given scene and mode.
    Saves step-level CSV and summary CSV to results_dir/.
    Returns the populated MetricsTracker.
    """
    global _vla_image, _delta_ema, _vla_running

    mode      = "cbf" if use_cbf else "plain"
    run_label = f"{cfg.name}_{mode}"

    print(f"\n{'='*56}")
    print(f"  Scene : {cfg.name}")
    print(f"  Mode  : {mode.upper()}  —  {cfg.description}")
    print(f"  Steps : {cfg.max_steps}   CBF gamma : {cfg.cbf_gamma}")
    print(f"{'='*56}\n")

    # --- Build model from a temp XML file co-located with panda.xml ----------
    # MuJoCo resolves meshdir="assets" relative to the XML file's own directory.
    # Writing the temp file into PANDA_DIR mirrors how safety_scene.xml works,
    # so all asset paths resolve identically.
    xml_str = build_scene_xml(cfg)
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".xml", dir=str(PANDA_DIR))
    try:
        with os.fdopen(tmp_fd, "w") as f:
            f.write(xml_str)
        model = mujoco.MjModel.from_xml_path(tmp_path)
    finally:
        os.unlink(tmp_path)
    data    = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 400, 400)

    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "hand")
    ghost_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "target_ghost")
    mocap_id   = model.body_mocapid[ghost_body]

    arm_bodies = _resolve_arm_bodies(model)

    # Set arm to ready configuration and run a warm-up IK phase to reach
    # cfg.start_pos before engaging the VLA or recording metrics.
    data.qpos[:7] = _QPOS_READY
    mujoco.mj_forward(model, data)
    ghost_pos = cfg.start_pos.copy()
    data.mocap_pos[mocap_id] = ghost_pos
    mujoco.mj_forward(model, data)

    metrics = MetricsTracker(cfg.name, mode)

    # --- Async VLA worker ----------------------------------------------------
    _delta_ema[:] = 0.0
    _vla_image    = None
    _vla_running  = True
    vla_thread = threading.Thread(
        target=_vla_worker, args=(cfg.ema_alpha,), daemon=True
    )
    vla_thread.start()

    # --- Main loop -----------------------------------------------------------
    with _viewer_ctx(model, data, headless) as viewer:

        # Warm-up: pure IK, no VLA, no metrics — moves arm from ready pose
        # to cfg.start_pos so the experiment begins in the right configuration.
        print(f"  Warm-up ({N_WARMUP} steps): driving arm to start position...")
        for _ in range(N_WARMUP):
            mujoco.mj_step(model, data)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            err = cfg.start_pos - data.site_xpos[ee_site_id]
            J   = jacp[:, :7]
            dq  = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3)) @ err
            data.qpos[:7] += 0.5 * np.clip(dq, -0.2, 0.2)
            mujoco.mj_forward(model, data)
            viewer.sync()
        ee_at_start = data.site_xpos[ee_site_id].copy()
        print(f"  Warm-up done. EE at {np.round(ee_at_start, 3)} "
              f"(target {cfg.start_pos})")

        for t in range(cfg.max_steps):
            mujoco.mj_step(model, data)

            # Push latest frame to VLA worker
            renderer.update_scene(data, camera=cam_id)
            img = _preprocess_rgb(renderer.render())
            with _vla_lock:
                _vla_image = img

            # Read latest VLA action
            with _vla_lock:
                d_ema = _delta_ema.copy()

            # --- Ghost target update ----------------------------------------
            # Soft potential-field repulsion applied in BOTH modes for fairness
            repulsion = _potential_repulsion(
                ghost_pos, cfg.obstacles,
                cfg.repulsion_gain, cfg.repulsion_cutoff,
            )
            ghost_pos = (ghost_pos
                         + cfg.goal_attract * (cfg.goal_pos - ghost_pos)
                         + cfg.vla_scale * d_ema
                         + repulsion)
            data.mocap_pos[mocap_id] = ghost_pos
            mujoco.mj_forward(model, data)

            # --- Nominal IK joint velocity ----------------------------------
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            err        = ghost_pos - data.site_xpos[ee_site_id]
            J          = jacp[:, :7]
            dq_nominal = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3)) @ err
            dq_nominal = np.clip(dq_nominal, -0.2, 0.2)

            # --- CBF filter (CBF mode only) ----------------------------------
            cbf_triggered   = False
            correction_norm = 0.0
            if use_cbf:
                dq_to_apply, correction_norm, cbf_triggered = _run_cbf(
                    model, data, arm_bodies,
                    cfg.obstacles, dq_nominal, cfg.cbf_gamma,
                )
            else:
                dq_to_apply = dq_nominal

            # --- Integrate --------------------------------------------------
            data.qpos[:7] += cfg.cbf_step_scale * dq_to_apply
            mujoco.mj_forward(model, data)

            # --- Metrics ----------------------------------------------------
            min_d, closest_obs, closest_body = _arm_obs_distances(
                data, arm_bodies, cfg.obstacles
            )
            violation = _any_violation(data, arm_bodies, cfg.obstacles)

            metrics.record(
                StepRecord(
                    step=t,
                    ee_pos=data.site_xpos[ee_site_id].copy(),
                    min_dist=min_d,
                    closest_obstacle=closest_obs or "",
                    closest_body=closest_body or "",
                    cbf_triggered=cbf_triggered,
                    cbf_correction_norm=correction_norm,
                    violation=violation,
                ),
                goal_pos=cfg.goal_pos,
                goal_tolerance=cfg.goal_tolerance,
            )

            if t % 20 == 0:
                flags = []
                if violation:       flags.append("*** VIOLATION ***")
                if cbf_triggered:   flags.append("[CBF]")
                print(f"  [{t:03d}] min_dist={min_d:.3f}m  {'  '.join(flags)}")

            viewer.sync()
            if not headless:
                time.sleep(0.01)

    # --- Stop VLA thread -----------------------------------------------------
    _vla_running = False
    vla_thread.join(timeout=2.0)

    # --- Save results --------------------------------------------------------
    if save_results:
        metrics.save_step_log(f"{results_dir}/{run_label}_steps.csv")
        metrics.save_summary( f"{results_dir}/{run_label}_summary.csv")

    s = metrics.summary()
    print(f"\n{'='*56}")
    print(f"  Results — {run_label}")
    print(f"  Min dist      : {s['min_dist_overall']:.3f} m")
    print(f"  Violations    : {s['violation_steps']}/{s['total_steps']} steps  ({s['violation_rate']:.1%})")
    print(f"  CBF triggered : {s['cbf_activations']} steps  ({s['cbf_activation_rate']:.1%})")
    print(f"  Path length   : {s['path_length_m']:.3f} m")
    print(f"  Goal reached  : {s['goal_reached']}  (step {s['goal_reach_step']})")
    print(f"{'='*56}\n")

    return metrics
