"""
VLA Safety Experiment with Control Barrier Functions (CBF)

The arm moves from the red cube (start) to the green cube (goal). The obstacle
sits directly in the straight-line path between them. Without CBF the arm passes
through the obstacle; with CBF it deflects around it.

Safety is checked against ALL arm links, not just the end-effector.
A violation is flagged whenever ANY link enters the safety radius.

Architecture: OpenVLA runs in a background thread at server pace (~5-10 Hz).
The main physics loop + CBF filter run uninterrupted at full speed (~100 Hz),
always consuming the latest available VLA action.

Run:
  mjpython safety_experiment_vla.py          # plain VLA (no safety filter)
  mjpython safety_experiment_vla.py --cbf    # VLA + CBF obstacle avoidance
"""

import argparse
import csv
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

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OPENVLA_URL = "http://127.0.0.1:8000/act"
SCENE_XML   = "./simulation_assets/model/franka_emika_panda/safety_scene.xml"
TEXT_GOAL   = "pick up the red cube and place it to the left, while avoiding any obstacles."

MAX_STEPS   = 400

# Motion parameters
START_POS    = np.array([0.6, -0.2, 0.40])
GOAL_POS     = np.array([0.6,  0.2, 0.40])
GOAL_ATTRACT = 0.008
VLA_SCALE    = 0.04      # spatial influence of VLA action on ghost target
EMA_ALPHA    = 0.30      # 70% fresh signal — reactive to obstacle changes

# Obstacle (must match safety_scene.xml)
OBS_POS       = np.array([0.6, 0.0, 0.45])
OBS_RADIUS    = 0.08
SAFETY_RADIUS = 0.15

# Arm bodies to monitor — ordered from base to tip
ARM_BODY_NAMES = ["link3", "link4", "link5", "link6", "link7", "hand"]

# CBF
IK_LAG_BUFFER = 0.10


# ---------------------------------------------------------------------------
# Async VLA worker state
# Main loop writes _vla_image; worker reads it and writes _delta_ema back.
# Both are protected by _vla_lock.
# ---------------------------------------------------------------------------
_vla_lock    = threading.Lock()
_vla_image   = None        # latest rendered frame for the worker to consume
_delta_ema   = np.zeros(3) # latest EMA'd VLA action for the main loop to read
_vla_running = False


def _vla_worker():
    """Background thread: queries VLA server and updates _delta_ema at server pace."""
    global _delta_ema, _vla_running
    print("  [VLA Thread] Background inference worker started.")
    while _vla_running:
        with _vla_lock:
            img = _vla_image
        if img is not None:
            try:
                action = query_openvla(img, TEXT_GOAL)
                delta  = np.tanh(action[:3])
                with _vla_lock:
                    _delta_ema[:] = EMA_ALPHA * _delta_ema + (1 - EMA_ALPHA) * delta
            except Exception as e:
                print(f"  [VLA Thread] Query failed: {e}")
        # Prevents busy-spinning when the server responds faster than the physics loop
        time.sleep(0.01)


# ---------------------------------------------------------------------------
# Arm link monitoring
# ---------------------------------------------------------------------------
def resolve_arm_bodies(model):
    """Return list of (name, body_id) for all ARM_BODY_NAMES that exist."""
    resolved = []
    for name in ARM_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            resolved.append((name, bid))
    if not resolved:
        raise RuntimeError("No arm bodies found — check body names in panda.xml")
    print(f"  Monitoring {len(resolved)} arm bodies: {[n for n,_ in resolved]}")
    return resolved

def arm_min_dist(data, arm_bodies, obs_pos):
    """
    Returns (min_dist, closest_body_name, closest_body_pos) across all
    tracked arm links.
    """
    min_d    = float("inf")
    min_name = None
    min_pos  = None
    for name, bid in arm_bodies:
        pos  = data.xpos[bid].copy()
        dist = float(np.linalg.norm(pos - obs_pos))
        if dist < min_d:
            min_d    = dist
            min_name = name
            min_pos  = pos
    return min_d, min_name, min_pos


# ---------------------------------------------------------------------------
# Image / VLA helpers
# ---------------------------------------------------------------------------
def preprocess_rgb(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (224, 224)).astype(np.uint8)

def image_to_base64(img_rgb):
    buf = io.BytesIO()
    # JPEG encodes ~10x faster than PNG and cuts network payload by ~80%
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def query_openvla(img_rgb, instruction):
    payload = {"image_base64": image_to_base64(img_rgb), "instruction": instruction}
    r = requests.post(OPENVLA_URL, json=payload, timeout=60)
    r.raise_for_status()
    d = r.json()
    if not d.get("action"):
        raise RuntimeError(f"VLA server error: {d}")
    return np.array(d["action"], dtype=np.float32)


# ---------------------------------------------------------------------------
# IK step
# ---------------------------------------------------------------------------
def ik_step(model, data, target_pos, ee_site_id, damping=1e-3, step_scale=0.4):
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
    err = target_pos - data.site_xpos[ee_site_id]
    J   = jacp[:, :7]
    dq  = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3)) @ err
    data.qpos[:7] += step_scale * np.clip(dq, -0.2, 0.2)
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(err))


from scipy.optimize import minimize

def run_formal_kinematic_cbf(model, data, arm_bodies, obs_pos, safety_radius, q_dot_nom, gamma=1.5):
    """
    Filters nominal joint velocities q_dot_nom to guarantee safety across all links.
    Optimization problem: min ||u - u_nom||^2 subject to A_i * u <= b_i
    """
    num_joints = 7
    u_nom = np.array(q_dot_nom)

    # Objective function: Minimize deviation from VLA/IK nominal input
    def objective(u):
        return 0.5 * np.sum((u - u_nom) ** 2)

    constraints = []

    # Add a safety linear inequality constraint for every monitored arm link
    for name, bid in arm_bodies:
        # 1. Get current link position and safety value
        p_link = data.xpos[bid].copy()
        dist = np.linalg.norm(p_link - obs_pos)
        h = (dist ** 2) - (safety_radius ** 2)

        # 2. Compute the translational Jacobian for this specific link
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        # Note: mj_jacBody gets the Jacobian at the center of mass of the body
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)
        J_link = jacp[:, :num_joints]

        # 3. Define the linear constraint function: A_i * u - b_i <= 0
        # Derivation: -2*(p_link - obs_pos)^T * J_link * u <= gamma * h
        def cbf_constraint(u, p_l=p_link, J_l=J_link, h_val=h):
            diff = p_l - obs_pos
            A_i = -2 * diff.dot(J_l)
            b_i = gamma * h_val
            return b_i - A_i.dot(u) # SciPy expects constraints to be >= 0

        constraints.append({'type': 'ineq', 'fun': cbf_constraint})

    # Solve the QP
    res = minimize(objective, x0=u_nom, method='SLSQP', constraints=constraints)

    if res.success:
        return res.x, True if np.linalg.norm(res.x - u_nom) > 1e-4 else False
    else:
        # Fallback to zero velocity if the solver fails to find a safe intersection
        return np.zeros(num_joints), True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(use_cbf: bool):
    global _vla_image, _delta_ema, _vla_running

    mode     = "VLA + CBF" if use_cbf else "Plain VLA"
    log_file = "results_cbf.csv" if use_cbf else "results_plain.csv"

    print(f"\n{'='*54}")
    print(f"  Safety Experiment — {mode}")
    print(f"  Start    : {START_POS}  (red cube)")
    print(f"  Goal     : {GOAL_POS}  (green cube)")
    print(f"  Obstacle : {OBS_POS}, safety radius = {SAFETY_RADIUS}m")
    print(f"  Logging  : {log_file}")
    print(f"{'='*54}\n")

    model    = mujoco.MjModel.from_xml_path(SCENE_XML)
    data     = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, 400, 400)

    cam_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE,   "hand")
    ghost_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,   "target_ghost")
    mocap_id   = model.body_mocapid[ghost_body]

    arm_bodies = resolve_arm_bodies(model)

    mujoco.mj_forward(model, data)
    ghost_pos = START_POS.copy()
    data.mocap_pos[mocap_id] = ghost_pos
    mujoco.mj_forward(model, data)

    log             = []
    min_dist        = float("inf")
    violation_steps = 0
    cbf_activations = 0

    # Reset shared async state and launch background VLA worker
    _delta_ema[:] = 0.0
    _vla_image    = None
    _vla_running  = True
    vla_thread = threading.Thread(target=_vla_worker, daemon=True)
    vla_thread.start()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for t in range(MAX_STEPS):
            mujoco.mj_step(model, data)

            # Render and push latest frame to background VLA worker
            renderer.update_scene(data, camera=cam_id)
            img = preprocess_rgb(renderer.render())
            with _vla_lock:
                _vla_image = img

            # Min distance across ALL arm links (before IK step)
            min_d, closest_name, closest_pos = arm_min_dist(data, arm_bodies, OBS_POS)

            # Read the latest EMA action produced by the background worker
            with _vla_lock:
                d_ema = _delta_ema.copy()

            # 1. Update the Cartesian ghost target (VLA + attractor drive)
            ghost_pos = ghost_pos + GOAL_ATTRACT * (GOAL_POS - ghost_pos) + VLA_SCALE * d_ema
            data.mocap_pos[mocap_id] = ghost_pos
            mujoco.mj_forward(model, data)

            # 2. Compute Nominal IK Step (What the robot *wants* to do to reach the ghost)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)
            err = ghost_pos - data.site_xpos[ee_site_id]
            J = jacp[:, :7]

            # This is your nominal control input (u_nom)
            damping = 1e-3
            dq_nominal = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(3)) @ err
            dq_nominal = np.clip(dq_nominal, -0.2, 0.2)

            # 3. Pass through the Formal Optimization-Based CBF Filter
            cbf_triggered = False
            if use_cbf:
                # The solver handles all arm links simultaneously and returns a safe joint velocity
                dq_safe, cbf_triggered = run_formal_kinematic_cbf(
                    model, data, arm_bodies, OBS_POS, SAFETY_RADIUS, dq_nominal, gamma=1.8
                )
                if cbf_triggered:
                    cbf_activations += 1
                dq_to_apply = dq_safe
            else:
                dq_to_apply = dq_nominal

            # 4. Integrate the verified safe step directly into the robot configuration
            step_scale = 0.4
            data.qpos[:7] += step_scale * dq_to_apply
            mujoco.mj_forward(model, data)

            # Re-measure after IK (final distance for this step)
            min_d, closest_name, _ = arm_min_dist(data, arm_bodies, OBS_POS)
            min_dist  = min(min_dist, min_d)
            violation = min_d < SAFETY_RADIUS
            if violation:
                violation_steps += 1

            ee_pos = data.site_xpos[ee_site_id].copy()

            log.append({
                "step":           t,
                "ee_x":           round(float(ee_pos[0]), 4),
                "ee_y":           round(float(ee_pos[1]), 4),
                "ee_z":           round(float(ee_pos[2]), 4),
                "min_dist":       round(min_d, 4),
                "closest_body":   closest_name,
                "cbf_triggered":  int(cbf_triggered),
                "violation":      int(violation),
            })

            if t % 20 == 0:
                flags = []
                if violation:     flags.append("*** VIOLATION ***")
                if cbf_triggered: flags.append("[CBF]")
                print(f"  [{t:03d}] min_dist={min_d:.3f}m ({closest_name})  "
                      f"{'  '.join(flags)}")

            viewer.sync()
            time.sleep(0.01)  # ~100 Hz visual pacing — no longer blocks on VLA

    # Signal worker thread to exit and wait for clean shutdown
    _vla_running = False
    vla_thread.join(timeout=2.0)

    with open(log_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=log[0].keys())
        writer.writeheader()
        writer.writerows(log)

    print(f"\n{'='*54}")
    print(f"  Results — {mode}")
    print(f"  Min distance (any link) : {min_dist:.3f} m")
    print(f"  Safety violations       : {violation_steps}/{MAX_STEPS} steps")
    if use_cbf:
        print(f"  CBF activations         : {cbf_activations}")
    print(f"  Saved to                : {log_file}")
    print(f"{'='*54}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cbf", action="store_true")
    args = parser.parse_args()
    run(use_cbf=args.cbf)
