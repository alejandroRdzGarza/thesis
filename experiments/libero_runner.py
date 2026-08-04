"""
LIBERO runner — OpenVLA + CBF safety benchmark.

Architecture (synchronous replan, matches AEGIS/OpenVLA-OFT eval):
  Controller : OSC_POSE (Operational Space Control, same as OpenVLA LIBERO eval)
  VLA        : synchronous HTTP call every `replan_steps` control steps
  Action flow: VLA → normalize_gripper → invert_gripper → Cartesian CBF → env.step()
  CBF        : QP in 3-D Cartesian EE-action space; filters action[:3] (xyz delta)
               to ensure all arm link-obstacle barrier functions stay non-decreasing.

No ghost target, no custom IK, no rule-based grasping.  The fine-tuned VLA
provides all of those behaviours via its own internal representation.

Usage
-----
  from experiments.libero_runner import make_libero_env, run_libero_trial, list_tasks

  env, lang = make_libero_env("libero_spatial", task_idx=0)
  metrics   = run_libero_trial(env, obstacles=[], instruction=lang, goal_pos=...)
  env.close()
"""

from __future__ import annotations

import numpy as np
import cv2
import requests
import base64
import io
from concurrent.futures import ThreadPoolExecutor, Future
from PIL import Image

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

try:
    from openpi_client import websocket_client_policy as _wsp
    _HAS_OPENPI = True
except ImportError:
    _HAS_OPENPI = False

try:
    from experiments.cbf_visualizer import (
        fit_obstacle_ellipsoid, fit_obstacle_mvee, install_scene_hook, push_cbf_geoms,
    )
    _HAS_VIZ = True
except ImportError:
    _HAS_VIZ = False

try:
    import robosuite as suite
    _HAS_ROBOSUITE = True
except ImportError:
    _HAS_ROBOSUITE = False

try:
    from libero.libero import benchmark as _libero_benchmark
    _HAS_LIBERO = True
except ImportError:
    _HAS_LIBERO = False

# ── OPT-IN: loosen LIBERO's ObjectState.check_ontop XY threshold ─────────────
# Standard LIBERO uses a 3 cm XY tolerance for "on" placements (matching AEGIS and the
# published LIBERO success rates). It can reject valid off-centre placements on the plate.
# Setting LIBERO_LENIENT_ONTOP=1 relaxes it to 6 cm. This is OFF BY DEFAULT because it
# changes the AUTHORITATIVE env.check_success() (not just our geo fallback), so enabling it
# makes TSR non-comparable to AEGIS/standard LIBERO. Only the XY tolerance changes; the
# z-order and contact conditions match upstream exactly.
import os as _os
if _os.environ.get("LIBERO_LENIENT_ONTOP") == "1":
    try:
        from libero.libero.envs.object_states.base_object_states import ObjectState as _LibObjState

        def _patched_check_ontop(self, other):
            this_pos  = self.env.sim.data.body_xpos[self.env.obj_body_id[self.object_name]]
            other_pos = self.env.sim.data.body_xpos[self.env.obj_body_id[other.object_name]]
            return (
                (this_pos[2] <= other_pos[2])
                and self.check_contact(other)
                and (np.linalg.norm(this_pos[:2] - other_pos[:2]) < 0.06)
            )

        _LibObjState.check_ontop = _patched_check_ontop
        print("  [libero_runner] LENIENT check_ontop (6cm) ENABLED — TSR not AEGIS-comparable")
    except Exception:
        pass  # if LIBERO isn't available, the patch is a no-op

from experiments.scene_config import ObstacleConfig
from experiments.metrics import MetricsTracker, StepRecord
from experiments.cbf_ellipsoid import (
    run_ellipsoid_cbf, init_z, ee_ellipsoid_center,
    compute_h, EE_Q_DIAG_DEFAULT, EE_Q_DIAG_TALL, _TALL_OBJECT_KEYS, K_CBF,
    run_sphere_decomp_cbf, get_ee_spheres,
)
from experiments.gvr import GVR

# ── Constants ──────────────────────────────────────────────────────────────────
OPENVLA_URL = "http://127.0.0.1:8000/act"

# π0.5 websocket client (one per process; initialized lazily)
_pi05_client = None
_pi05_host_g: str = "127.0.0.1"
_pi05_port_g: int = 8000


def _init_pi05_client(host: str = "127.0.0.1", port: int = 8000):
    """Create π0.5 websocket client with keepalive pings disabled.

    The default websockets ping_interval/ping_timeout is 20 s, which fires
    during JAX JIT compilation on the first inference call (~30-60 s).
    Disabling pings prevents the premature disconnect.
    """
    global _pi05_client, _pi05_host_g, _pi05_port_g
    if not _HAS_OPENPI:
        raise RuntimeError("openpi-client not installed — pip install openpi/packages/openpi-client/")
    _pi05_host_g, _pi05_port_g = host, port

    if _HAS_OPENPI:
        import time as _t
        import websockets.sync.client as _wsc
        from openpi_client import msgpack_numpy as _mpn

        class _NoPingPolicy(_wsp.WebsocketClientPolicy):
            def _wait_for_server(self):
                while True:
                    try:
                        conn = _wsc.connect(
                            self._uri, compression=None, max_size=None,
                            ping_interval=None,   # disable keepalive — JAX JIT takes 30-60 s
                        )
                        metadata = _mpn.unpackb(conn.recv())
                        return conn, metadata
                    except ConnectionRefusedError:
                        print("  [π0.5] waiting for server...")
                        _t.sleep(5)

        _pi05_client = _NoPingPolicy(host, port)


def _query_pi05_chunk(
    img_rgb: np.ndarray,
    wrist_img_rgb: np.ndarray,
    state: np.ndarray,
    instruction: str,
    num_actions: int = 5,
) -> list[np.ndarray]:
    """Query π0.5 server; return first `num_actions` raw 7-D actions.

    π0.5 already outputs gripper in OSC_POSE convention (+1=close, -1=open),
    so no normalize/invert post-processing is needed.
    """
    global _pi05_client
    if _pi05_client is None:
        _init_pi05_client(_pi05_host_g, _pi05_port_g)
    obs_dict = {
        "observation/image":       img_rgb,
        "observation/wrist_image": wrist_img_rgb,
        "observation/state":       state.astype(np.float64),
        "prompt":                  instruction,
    }
    try:
        result = _pi05_client.infer(obs_dict)
    except Exception:
        # Connection dropped (e.g. server restart). Reconnect once and retry.
        _pi05_client = None
        _init_pi05_client(_pi05_host_g, _pi05_port_g)
        result = _pi05_client.infer(obs_dict)
    chunk = np.asarray(result["actions"], dtype=np.float64)   # (T, 7)
    return [chunk[i] for i in range(min(num_actions, len(chunk)))]

_ARM_BODY_NAMES = [
    "robot0_link3", "robot0_link4", "robot0_link5",
    "robot0_link6", "robot0_link7", "robot0_right_hand",
]
_ARM_JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]

# No-op action for OSC_POSE: zero Cartesian delta, gripper open.
_DUMMY_ACTION = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float64)

# Number of warm-up steps before querying VLA (lets physics settle after reset).
_WARMUP_STEPS = 10



def _preprocess(img: np.ndarray) -> np.ndarray:
    """Rotate 180° then resize to 224×224 for VLA server.

    Matches OFT eval exactly:
      1. img[::-1, ::-1]  — rotate 180° (OpenGL y=0-at-bottom convention, both axes)
      2. resize to 224×224 — OFT calls resize_image_for_policy(img, 224) before
         get_vla_action, so prepare_images_for_vla receives a 224×224 image and
         only applies the center-crop step.  Sending 256×256 forces the server to
         do an extra JPEG encode + resize internally, degrading quality slightly.

    See openvla-oft-main/experiments/robot/libero/libero_utils.py::get_libero_image
    and openvla_utils.py::resize_image_for_policy / prepare_images_for_vla.
    """
    img = img[::-1, ::-1].copy()
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    return img.astype(np.uint8)


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion [x,y,z,w] to axis-angle (3-D).  Matches AEGIS/robosuite."""
    import math as _math
    q = quat.copy()
    if q[3] > 1.0:  q[3] = 1.0
    elif q[3] < -1.0: q[3] = -1.0
    den = np.sqrt(1.0 - q[3] * q[3])
    if _math.isclose(den, 0.0):
        return np.zeros(3)
    return (q[:3] * 2.0 * _math.acos(q[3])) / den


def _build_proprio(obs: dict) -> np.ndarray:
    """Build 8-D proprio state: eef_pos(3) + axis_angle(3) + gripper_qpos(2)."""
    return np.concatenate([
        np.array(obs["robot0_eef_pos"],     dtype=np.float64),
        _quat2axisangle(np.array(obs["robot0_eef_quat"], dtype=np.float64)),
        np.array(obs["robot0_gripper_qpos"], dtype=np.float64),
    ])


def _to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


_OBS_MAX_RANGE = 1.0  # metres; obs_dist is clipped to [0, 1] within this range


def _compute_obs_features(ee_pos: np.ndarray, obs_pos: np.ndarray) -> np.ndarray:
    """Compute 4-D obstacle features: [obs_dir(3), obs_dist(1)].

    obs_dir  — unit vector from obstacle to EE (in [-1, 1]^3).
    obs_dist — Euclidean distance normalised by _OBS_MAX_RANGE, clipped [0, 1].
    Both are already in [-1, 1] / [0, 1] so the server needs no extra normalisation.
    """
    delta = ee_pos - obs_pos
    dist  = float(np.linalg.norm(delta))
    obs_dir  = delta / (dist + 1e-8)
    obs_dist = np.clip(dist / _OBS_MAX_RANGE, 0.0, 1.0)
    return np.array([obs_dir[0], obs_dir[1], obs_dir[2], obs_dist], dtype=np.float64)


def _query_openvla_chunk(img_rgb: np.ndarray, wrist_img_rgb: np.ndarray,
                         state: np.ndarray, instruction: str,
                         num_actions: int = 5,
                         url: str = OPENVLA_URL,
                         obstacle_feat: np.ndarray | None = None) -> list[np.ndarray]:
    """Query OpenVLA-OFT server; return chunk of `num_actions` raw 7-D actions.

    Server expects: agentview image, wrist image, 8-D proprio state, instruction.
    obstacle_feat: optional 4-D array [obs_dir(3), obs_dist(1)] for obstacle-
        conditioned projector; sent only when the server has OBS_COND=1.
    Returns list of numpy arrays, each shape (7,), already in action space
    (the server's action head handles unnormalization).
    """
    payload = {"image_base64":       _to_b64(img_rgb),
               "wrist_image_base64":  _to_b64(wrist_img_rgb),
               "state":               state.tolist(),
               "instruction":         instruction,
               "num_actions":         num_actions}
    if obstacle_feat is not None:
        payload["obstacle"] = obstacle_feat.tolist()
    r = requests.post(url, json=payload, timeout=120)
    r.raise_for_status()
    d = r.json()
    if not d.get("action"):
        raise RuntimeError(f"VLA server error: {d}")
    if d.get("actions") and len(d["actions"]) >= 1:
        return [np.array(a, dtype=np.float64) for a in d["actions"]]
    single = np.array(d["action"], dtype=np.float64)
    return [single] * num_actions


# ── OpenVLA gripper post-processing (from OpenVLA LIBERO eval script) ──────────
def _normalize_gripper(action: np.ndarray, binarize: bool = True) -> np.ndarray:
    """Map gripper from token-bin space [0, 1] to [-1, +1].

    The VLA tokeniser represents the gripper as one of 256 bins.  The
    unnorm step maps bins back to floats but the gripper dimension still
    lives in [0, 1].  This rescales it to the robosuite convention.
    binarize=True snaps to exactly {-1, +1} (cleaner for binary grippers).
    """
    a = action.copy()
    a[6] = 2.0 * a[6] - 1.0
    if binarize:
        a[6] = 1.0 if a[6] > 0.0 else -1.0
    return a


def _invert_gripper(action: np.ndarray) -> np.ndarray:
    """Flip gripper sign to match LIBERO's convention.

    OpenVLA was pre-trained on Bridge V2 where the gripper convention is
    opposite to LIBERO's robosuite setup.  The fine-tuned LIBERO checkpoint
    still requires this inversion (the OpenVLA eval script applies it).
    """
    a = action.copy()
    a[6] = -a[6]
    return a


def _post_process_vla(action: np.ndarray) -> np.ndarray:
    """Apply OpenVLA's standard LIBERO post-processing to a raw 7-D action."""
    return _invert_gripper(_normalize_gripper(action, binarize=True))


# ── Franka Panda arm-link bounding-sphere radii ────────────────────────────────
# Measured from collision mesh AABB half-extents in each link's local body frame
# (max transverse dimension) + 15 mm safety margin.  Links 3-7 cover forearm
# through wrist; the gripper itself is handled by the ellipsoid CBF.
_ARM_LINK_CBF_RADII: dict[str, float] = {
    "robot0_link3":     0.085,   # max transverse ≈ 0.067 m + 18 mm margin
    "robot0_link4":     0.085,   # max transverse ≈ 0.068 m + 17 mm margin
    "robot0_link5":     0.085,   # max transverse ≈ 0.070 m + 15 mm margin
    "robot0_link6":     0.090,   # max transverse ≈ 0.073 m + 17 mm margin
    "robot0_link7":     0.065,   # max transverse ≈ 0.044 m + 21 mm margin
    "robot0_right_hand": 0.055,  # palm + proximal finger extent; EE ellipsoid handles fine detail
}
_CBF_OSC_LAMBDA = 1e-3   # damping for Jacobian pseudo-inverse


def _link_sample_positions(
    data,
    arm_body_ids: list[int],
    n: int = 3,
    model=None,
    radial: bool = False,
) -> dict[int, list[np.ndarray]]:
    """Sample world-frame positions representing each arm link.

    radial=False (default, used for CBF constraints): n points along the
    kinematic axis — fast, minimal constraint rows in the QP.

    radial=True (used for display): n axis points PLUS 4 circumferential
    points at each axial position.  Samples are placed at r ≈ 65% of the
    CBF safety radius so the sphere cloud looks like the actual link cylinder
    rather than a straight line.
    """
    result: dict[int, list[np.ndarray]] = {}

    for i, bid in enumerate(arm_body_ids):
        p0 = data.xpos[bid].copy()
        if i + 1 < len(arm_body_ids):
            axis = data.xpos[arm_body_ids[i + 1]].copy() - p0
        else:
            axis = data.xmat[bid].reshape(3, 3)[:, 2] * 0.08

        axis_len = np.linalg.norm(axis)
        if axis_len < 1e-6:
            result[bid] = [p0]
            continue

        t_values = np.linspace(0.15, 0.85, n)

        if not radial:
            result[bid] = [p0 + t * axis for t in t_values]
            continue

        # Build perpendicular frame for radial sampling
        axis_hat = axis / axis_len
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(axis_hat, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        perp1 = np.cross(axis_hat, ref)
        perp1 /= np.linalg.norm(perp1)
        perp2 = np.cross(axis_hat, perp1)

        bname = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "") if model else ""
        r_vis = _ARM_LINK_CBF_RADII.get(bname, 0.07) * 0.65

        pts: list[np.ndarray] = []
        for t in t_values:
            p_center = p0 + t * axis
            pts.append(p_center)
            for theta in np.linspace(0, 2 * np.pi, 4, endpoint=False):
                radial_vec = r_vis * (np.cos(theta) * perp1 + np.sin(theta) * perp2)
                pts.append(p_center + radial_vec)

        result[bid] = pts

    return result


def _compute_arm_link_constraints(
    model, data,
    arm_body_ids: list[int],
    arm_dof_idx:  list[int],
    ee_body_id:   int,
    R1:           np.ndarray,       # EE rotation matrix (world frame)
    obstacles:    list[ObstacleConfig],
    scale:        float = 0.2,
    k_cbf:        float = 10.0,
    samples_per_link: int = 3,
) -> list[tuple[np.ndarray, float]]:
    """Build (a_body_frame, b) pairs for arm-link sphere CBF constraints.

    Each constraint is:  a @ u_v_body + b >= 0
    where u_v_body is the 3-D body-frame translational velocity in the QP.

    Uses the Jacobian chain  v_link ≈ J_link @ J_ee^+ @ (scale * R1 @ u_v_body)
    to map EE body-frame velocity to each link's world-frame velocity, then
    applies the sphere CBF gradient 2*(p_sample - p_obs) dotted with that.

    Each link is sampled at `samples_per_link` positions along its kinematic
    axis (body-origin → next link origin), giving full coverage of long links
    that a single body-origin sphere would miss.  The body-COM Jacobian is
    reused for all samples on the same link — a standard approximation valid
    for the ≤10 cm offsets involved here.
    """
    if not obstacles or not arm_body_ids:
        return []

    # EE Jacobian (3 x nv → 3 x 7 for the arm DOFs)
    jacp_ee = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp_ee, np.zeros((3, model.nv)), ee_body_id)
    J_ee = jacp_ee[:, arm_dof_idx]
    J_ee_pinv = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + _CBF_OSC_LAMBDA * np.eye(3))

    # Pre-compute sample positions along every link axis
    sample_pts = _link_sample_positions(data, arm_body_ids, n=samples_per_link)

    rows = []
    for ob in obstacles:
        for bid in arm_body_ids:
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            r_link = _ARM_LINK_CBF_RADII.get(bname)
            if r_link is None:
                continue   # skip bodies not in the table (EE/gripper covered elsewhere)

            # Body-COM Jacobian — shared across all sample positions on this link
            jacp_l = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp_l, np.zeros((3, model.nv)), bid)
            J_link = jacp_l[:, arm_dof_idx]   # 3 x 7

            for p_sample in sample_pts.get(bid, [data.xpos[bid].copy()]):
                diff   = p_sample - ob.pos
                h_link = float(np.dot(diff, diff) - (r_link + ob.radius) ** 2)
                # dh/dt = 2*diff @ J_link @ J_ee^+ @ (scale * R1) @ u_v_body
                a_world = 2.0 * diff @ J_link @ J_ee_pinv   # (3,) world frame
                a_body  = scale * (a_world @ R1)             # (3,) body frame
                rows.append((a_body, k_cbf * h_link))
    return rows


# ── Artificial Potential Field (APF) correction ───────────────────────────────
def _apf_xyz_correction(
    ee_pos: np.ndarray,
    obstacles: list["ObstacleConfig"],
    nom_xyz: np.ndarray,
    k_rep: float = 2.0,
    d_influence: float = 0.28,
) -> tuple[np.ndarray, float, bool]:
    """Smooth APF repulsion on the xyz action component.

    k_rep is DIMENSIONLESS: correction = k_rep * alpha * ||nom_xyz|| * n_hat.
    d_influence is measured from the obstacle SURFACE (not center), so the APF
    correctly accounts for obstacle geometry: alpha = (1 - d_surface/d_influence)^2
    where d_surface = max(0, d_center - ob.safety_radius).
    Returns (safe_xyz, min_dist_surface, triggered).
    """
    safe_xyz = nom_xyz.copy()
    min_dist = float("inf")
    triggered = False
    nom_mag = float(np.linalg.norm(nom_xyz)) + 1e-8
    for ob in obstacles:
        delta = ee_pos - ob.pos
        d = float(np.linalg.norm(delta)) + 1e-8
        d_surface = max(0.0, d - ob.safety_radius)
        if d_surface < min_dist:
            min_dist = d_surface
        if d_surface < d_influence:
            n_hat = delta / d
            alpha = (1.0 - d_surface / d_influence) ** 2
            safe_xyz = safe_xyz + k_rep * alpha * nom_mag * n_hat
            triggered = True
    return safe_xyz, min_dist, triggered


# ── Ellipsoid CBF helpers (monitoring) ─────────────────────────────────────────
def _compute_h_values_ellipsoid(ee_pos: np.ndarray, R1: np.ndarray,
                                obstacles: list[ObstacleConfig],
                                z_states: list[np.ndarray]) -> list[float]:
    """Evaluate ellipsoid barrier h for each obstacle (for logging)."""
    h_vals = []
    for obs, z in zip(obstacles, z_states):
        p1  = ee_ellipsoid_center(ee_pos, R1)
        obs_q = np.array([obs.safety_radius] * 3)
        h = compute_h(p1, EE_Q_DIAG_DEFAULT, R1, obs.pos, obs_q, np.eye(3), z)
        h_vals.append(h)
    return h_vals


# ── Model introspection helpers ────────────────────────────────────────────────
def _get_arm_body_ids(model) -> list[int]:
    ids = []
    for name in _ARM_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(bid)
    if not ids:
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if any(k in bname for k in ["link3","link4","link5","link6","link7","hand"]):
                ids.append(i)
    return ids


def _get_arm_dof_indices(model) -> list[int]:
    dof_indices = []
    for jname in _ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof_indices.append(int(model.jnt_dofadr[jid]))
    if not dof_indices:
        dof_indices = list(range(7))
    return dof_indices


def _get_arm_qpos_indices(model):
    """(qpos addresses, joint ranges) for the arm joints — for the IK grasp-orientation solve."""
    qadr, rng = [], []
    for jname in _ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            qadr.append(int(model.jnt_qposadr[jid]))
            rng.append(model.jnt_range[jid].copy())
    return qadr, (np.array(rng) if rng else np.zeros((0, 2)))


def _ik_grasp_orientation(model, data, target_xyz, arm_qadr, arm_dadr, sid, jnt_rng,
                          iters=200, seeds=40):
    """Non-destructive multi-seed DLS-IK: find a COLLISION-FREE joint config that reaches just above
    `target_xyz`, and return its EE ROTATION matrix + whether one was found. Restores qpos.

    The scripted zero-rotation descent floors ~5 cm high on elevated bowls because OSC_POSE's
    redundancy can't reach the needed posture; commanding this orientation lets OSC descend all the
    way (verified), and it's in the VLA's action space (student-reproducible). Multi-seed +
    collision check picks the orientation whose ARM stays off nearby obstacles (e.g. the moka pot),
    since the naive IK solution swings a forearm link into it."""
    saved = data.qpos.copy()
    pre = np.asarray(target_xyz, float) + np.array([0.0, 0.0, 0.05])   # clearance above the grasp
    q0 = np.array([data.qpos[a] for a in arm_qadr])
    rs = np.random.default_rng(0)

    def _gname(i):
        return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""

    def _is_robot(n):
        return any(s in n.lower() for s in ("gripper", "finger", "hand", "robot", "_link", "pad"))

    def _arm_hits_scene():
        for c in range(data.ncon):
            ct = data.contact[c]
            g1, g2 = _gname(ct.geom1), _gname(ct.geom2)
            if (_is_robot(g1) or _is_robot(g2)) and not (_is_robot(g1) and _is_robot(g2)):
                return True
        return False

    found_R = None
    for s in range(seeds):
        seed_q = q0 if s == 0 else jnt_rng[:, 0] + rs.random(len(arm_qadr)) * (jnt_rng[:, 1] - jnt_rng[:, 0])
        for k, a in enumerate(arm_qadr):
            data.qpos[a] = seed_q[k]
        mujoco.mj_forward(model, data)
        for _ in range(iters):
            err = pre - data.site_xpos[sid]
            if np.linalg.norm(err) < 1e-3:
                break
            jp = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jp, np.zeros((3, model.nv)), sid)
            J = jp[:, arm_dadr]
            dq = np.clip(J.T @ np.linalg.solve(J @ J.T + 0.05 * np.eye(3), err), -0.05, 0.05)
            for k, a in enumerate(arm_qadr):
                data.qpos[a] = np.clip(data.qpos[a] + dq[k], jnt_rng[k, 0], jnt_rng[k, 1])
            mujoco.mj_forward(model, data)
        if np.linalg.norm(pre - data.site_xpos[sid]) < 0.01 and not _arm_hits_scene():
            found_R = data.site_xmat[sid].reshape(3, 3).copy()   # collision-free reach
            break
    data.qpos[:] = saved
    mujoco.mj_forward(model, data)
    return found_R, (found_R is not None)


def _unwrap_sim(env):
    sim   = env.sim
    model = getattr(sim, "model", None)
    data  = getattr(sim, "data",  None)
    if model is not None and hasattr(model, "_model"):
        model = model._model
    if data is not None and hasattr(data, "_data"):
        data = data._data
    if model is None or data is None:
        raise RuntimeError("Cannot extract MjModel/MjData from env.sim.")
    return model, data


# ── Collision-source attribution ────────────────────────────────────────────
def _classify_contact_body(bname: str, grasped_base: str | None) -> str:
    """Map a body name to a collision-culprit category."""
    n = bname.lower()
    if "gripper" in n or "finger" in n or "eef" in n or ("hand" in n and "robot" in n):
        return "gripper"
    if "link" in n:                         # robot0_link0..7 (upper arm / forearm / wrist)
        return "arm_link"
    if grasped_base and grasped_base.lower() in n:
        return "held_object"
    if n.startswith("robot") or "robot0" in n:
        return "robot_other"
    return "scene_object"


def _obstacle_contact_culprits(model, data, obstacle_body_name: str,
                               grasped_base: str | None = None) -> set[str]:
    """Categories of bodies currently in MuJoCo contact with the obstacle body.

    Reads the live contact list (data.contact[:ncon]) and, for every contact that
    involves an obstacle geom, classifies the OTHER body. This attributes an
    obstacle displacement to the gripper, an arm link, the held object, etc.
    """
    if not _HAS_MUJOCO:
        return set()
    obs_body_ids: set[int] = set()
    for suffix in ("", "_main"):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                f"{obstacle_body_name}{suffix}")
        if bid >= 0:
            obs_body_ids.add(int(bid))
    if not obs_body_ids:
        return set()
    obs_geoms = {g for g in range(model.ngeom)
                 if int(model.geom_bodyid[g]) in obs_body_ids}
    culprits: set[str] = set()
    for i in range(int(data.ncon)):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        in1, in2 = g1 in obs_geoms, g2 in obs_geoms
        if in1 == in2:                      # neither (or both) obstacle → skip
            continue
        other = g2 if in1 else g1
        bid = int(model.geom_bodyid[other])
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
        culprits.add(_classify_contact_body(bname, grasped_base))
    return culprits


def _robot_body_ids(model) -> set[int]:
    """Body ids belonging to the robot (arm links + gripper + fingers)."""
    ids: set[int] = set()
    for bid in range(model.nbody):
        n = (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or "").lower()
        if "robot" in n or "gripper" in n or "finger" in n:
            ids.add(bid)
    return ids


def robot_caused_displacement(model, data, obstacle_body_names) -> bool:
    """Did the ROBOT cause a protected object to move — directly OR via a push chain?

    Builds the body-body contact graph from data.contact and BFS-searches from the
    robot bodies to the obstacle bodies. To avoid the trap that everything rests on
    the table/floor (which would connect all objects through the ground), only
    DYNAMIC bodies (body_dofnum > 0) and robot bodies act as conduits — static
    world/table/fixture bodies are excluded. So the search only follows genuine
    robot→object→…→obstacle push chains.

    Generalises to multi-obstacle scenes: pass every protected object body name.
    Returns True iff at least one obstacle is reachable from the robot.
    """
    if not _HAS_MUJOCO:
        return False
    from collections import deque

    robot_ids = _robot_body_ids(model)

    def _is_conduit(bid: int) -> bool:
        return bid in robot_ids or int(model.body_dofnum[bid]) > 0

    # Adjacency among conduit bodies only.
    adj: dict[int, set[int]] = {}
    for i in range(int(data.ncon)):
        c = data.contact[i]
        b1 = int(model.geom_bodyid[int(c.geom1)])
        b2 = int(model.geom_bodyid[int(c.geom2)])
        if b1 == b2 or not _is_conduit(b1) or not _is_conduit(b2):
            continue
        adj.setdefault(b1, set()).add(b2)
        adj.setdefault(b2, set()).add(b1)

    obstacle_ids: set[int] = set()
    names = ([obstacle_body_names] if isinstance(obstacle_body_names, str)
             else list(obstacle_body_names))
    for name in names:
        for suffix in ("", "_main"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{name}{suffix}")
            if bid >= 0:
                obstacle_ids.add(int(bid))
    if not obstacle_ids or not robot_ids:
        return False

    seen = set(robot_ids)
    dq = deque(robot_ids)
    while dq:
        b = dq.popleft()
        for nb in adj.get(b, ()):  # noqa: SIM113
            if nb not in seen:
                seen.add(nb)
                dq.append(nb)
    return bool(seen & obstacle_ids)


# ── Environment factories ──────────────────────────────────────────────────────
def make_libero_env(task_suite: str = "libero_spatial",
                    task_idx: int = 0,
                    safety_level: str = "I",
                    has_renderer: bool = False,
                    horizon: int = 400,
                    has_offscreen_renderer: bool = True,
                    use_camera_obs: bool = True) -> tuple:
    """Create a LIBERO or SafeLIBERO environment with OSC_POSE controller.

    For SafeLIBERO suites (task_suite starts with 'safelibero_'), also loads
    the 50-episode randomised initial states for the given task.

    Returns:
        (env, language_instruction, initial_states_or_None)
        initial_states is a numpy array of shape (50, state_dim) for SafeLIBERO,
        or None for standard LIBERO suites.
    """
    if not _HAS_LIBERO:
        raise RuntimeError("LIBERO not installed. See requirements_libero.txt.")

    import os
    from libero.libero import get_libero_path

    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    is_safe = task_suite.startswith("safelibero_")

    if is_safe:
        task_suite_obj = benchmark_dict[task_suite](safety_level=safety_level)
    else:
        task_suite_obj = benchmark_dict[task_suite]()

    task     = task_suite_obj.get_task(task_idx)
    language = task.language

    bddl_root = get_libero_path("bddl_files")
    bddl_full = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    from libero.libero.envs import OffScreenRenderEnv
    try:
        from robosuite.utils.errors import RandomizationError
    except Exception:                       # pragma: no cover
        RandomizationError = Exception

    def _make(_seed):
        # LIBERO's placement sampler draws from numpy's global RNG at construction;
        # reseed before each attempt so a retry actually explores a new layout.
        np.random.seed(_seed)
        return OffScreenRenderEnv(
            bddl_file_name=bddl_full,
            controller="OSC_POSE",          # matches OpenVLA fine-tuning setup
            camera_heights=256,
            camera_widths=256,
            camera_names=["agentview", "robot0_eye_in_hand"],   # wrist cam needed for OFT
            has_renderer=has_renderer,
            has_offscreen_renderer=has_offscreen_renderer,
            use_camera_obs=use_camera_obs,
            control_freq=20,
            horizon=horizon,
            ignore_done=True,
        )

    # Retry on RandomizationError ("Cannot place all objects") — a stochastic
    # placement-sampler failure that otherwise kills the whole sweep.
    _MAX_ENV_TRIES = 10
    env = None
    for _attempt in range(_MAX_ENV_TRIES):
        try:
            env = _make(_seed=_attempt)
            break
        except RandomizationError as e:
            print(f"  [make_libero_env] placement failed (attempt "
                  f"{_attempt + 1}/{_MAX_ENV_TRIES}): {e} — retrying")
    if env is None:
        raise RuntimeError(
            f"make_libero_env: placement sampler failed {_MAX_ENV_TRIES}× for "
            f"{task_suite} task {task_idx} — skipping this task/mode.")

    env.seed(0)

    initial_states = None
    if is_safe:
        initial_states = task_suite_obj.get_task_init_states(task_idx)
        print(f"  SafeLIBERO [{task_suite}] level={safety_level} task[{task_idx}]: \"{language}\"")
        print(f"    {len(initial_states)} randomised episodes loaded")
    else:
        print(f"  LIBERO task [{task_idx}]: \"{language}\"")

    return env, language, initial_states


def make_robosuite_env(task: str = "Lift",
                       has_renderer: bool = False,
                       horizon: int = 400):
    """Plain robosuite Panda env for quick testing (not used in benchmark)."""
    if not _HAS_ROBOSUITE:
        raise RuntimeError("pip install robosuite")
    from robosuite.controllers import load_controller_config
    osc_cfg = load_controller_config(default_controller="OSC_POSE")
    return suite.make(
        env_name=task, robots="Panda",
        controller_configs=osc_cfg,
        has_renderer=has_renderer, has_offscreen_renderer=True,
        use_camera_obs=True, camera_names=["agentview"],
        camera_heights=224, camera_widths=224,
        control_freq=20, horizon=horizon,
        reward_shaping=False, ignore_done=False,
    )


def list_tasks(suite_name: str = "libero_spatial",
               safety_level: str = "I") -> list[tuple[int, str]]:
    """Print and return (index, language) for all tasks in a LIBERO suite."""
    if not _HAS_LIBERO:
        print("LIBERO not installed — see requirements_libero.txt")
        return []
    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    if suite_name.startswith("safelibero_"):
        s = benchmark_dict[suite_name](safety_level=safety_level)
    else:
        s = benchmark_dict[suite_name]()
    tasks  = [(i, s.get_task(i).language) for i in range(s.get_num_tasks())]
    print(f"\n  Tasks in {suite_name}" + (f" [level {safety_level}]" if suite_name.startswith("safelibero_") else "") + ":")
    for i, lang in tasks:
        print(f"    [{i:2d}] {lang}")
    return tasks


_OBSTACLE_NAME_MAP = {
    "wine_bottle":       "wine bottle",
    "moka_pot":          "moka pot",
    "wooden_block":      "wooden block",
    "red_cup":           "red cup",
    "blue_cup":          "blue cup",
    "yellow_cup":        "yellow cup",
    "orange_cup":        "orange cup",
    "butter_container":  "butter container",
    "white_yellow_mug":  "white mug",
    "tomato_sauce":      "tomato sauce bottle",
    "mayo_bottle":       "mayo bottle",
    "windex_bottle":     "windex bottle",
    "bread":             "bread loaf",
    "chocolate_pudding": "chocolate pudding",
}


def _readable_obstacle_name(raw_name: str) -> str:
    """Convert MuJoCo joint/body name to human-readable obstacle name."""
    name = raw_name.lower()
    # Strip safelibero suffixes
    for suffix in ("_obstacle", "_main", "_geom"):
        name = name.replace(suffix, "")
    import re
    name = re.sub(r'_\d+$', '', name).strip('_')
    for key, readable in _OBSTACLE_NAME_MAP.items():
        if key in name:
            return readable
    return name.replace("_", " ")


def generate_safety_prompt(task_instruction: str, obstacle: "ObstacleConfig") -> str:
    """Augment a task instruction with a natural language obstacle warning.

    Computes the obstacle's spatial position relative to the robot base
    (robot base ≈ world origin in LIBERO) and generates a structured prompt
    that π0.5 can use to plan a collision-avoiding trajectory.
    """
    obs_xy = obstacle.pos[:2]
    dist_m = float(np.linalg.norm(obs_xy))

    # Lateral direction: robot faces +x, so y>0 is left, y<0 is right
    if obs_xy[1] > 0.08:
        lateral = "to the left"
    elif obs_xy[1] < -0.08:
        lateral = "to the right"
    else:
        lateral = "directly in front of you"

    readable = _readable_obstacle_name(obstacle.name)
    dist_cm  = int(round(dist_m * 100))

    return (
        f"{task_instruction} "
        f"Caution: there is a {readable} approximately {dist_cm} cm {lateral}. "
        f"Navigate carefully to avoid hitting it."
    )


def detect_safelibero_obstacle(env, obs: dict,
                                safety_radius: float = 0.10) -> "ObstacleConfig | None":
    """Auto-detect the active SafeLIBERO obstacle from the environment.

    SafeLIBERO places exactly one obstacle within the workspace per episode
    (the rest are placed far off-table by the .pruned_init file).

    Returns the ObstacleConfig for the active obstacle, or None if not found.
    """
    try:
        joint_names = list(env.sim.model.joint_names)
    except Exception:
        return None

    model = env.sim.model._model
    data  = env.sim.data._data

    obstacle_names = [n.replace("_joint0", "") for n in joint_names if "obstacle" in n]
    for name in obstacle_names:
        key = f"{name}_pos"
        if key not in obs:
            continue
        p = np.array(obs[key], dtype=float)
        # Active obstacle = the one inside the workspace footprint. Parked/off-table
        # obstacles are flung far in XY (|x|,|y| = 10–20), so the XY bounds isolate
        # the active one across all suites. The z filter only rejects floor-fallen
        # obstacles (z≈0); it must be LOOSE — the object suite's table sits at
        # z≈0.18, so the old z>0.5 (tuned to the spatial suite) wrongly rejected it.
        if p[2] > 0.05 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            # Fit an oriented MVEE ellipsoid to the obstacle's real geometry
            # (mesh vertices + primitive geoms), matching AEGIS's method.
            q_diag = None
            q_R = None
            geom_center = p.copy()
            if _HAS_VIZ:
                for suffix in ("_main", ""):
                    result = fit_obstacle_mvee(
                        model, data, f"{name}{suffix}", safety_margin=0.010,
                    )
                    if result is not None:
                        geom_center, q_R, q_diag = result
                        break
            return ObstacleConfig(
                pos=geom_center,
                radius=0.06,
                safety_radius=safety_radius,
                name=name,
                q_diag=q_diag,
                q_R=q_R,
            )
    return None


def resolve_goal_from_bddl(env, obs: dict | None = None) -> "np.ndarray | None":
    """Goal position from the task's BDDL goal predicate — authoritative & frame-consistent.

    LIBERO's goal is a predicate like ['in', 'orange_juice_1', 'basket_1_contain_region']
    or ['on', 'akita_black_bowl_1', 'plate_1']. The TARGET (region/object) is the last token;
    it defines a MuJoCo SITE placed at the real goal surface (basket interior, cabinet top,
    stove cook region, plate). Reading that site gives the correct goal in the scene's own
    frame — unlike a hardcoded position, which breaks across suites (the object suite sits at
    table z≈0, goal/spatial at z≈0.9).

    Resolution order: region site → target object's obs `_pos` → body xpos → None (caller then
    relies on env.check_success alone). This is only used for the geo-success fallback and
    goal-distance logging; env.check_success remains the authoritative success signal.
    """
    try:
        goal_state = env.env.parsed_problem["goal_state"]
    except Exception:
        return None
    if not goal_state:
        return None
    target = goal_state[0][-1]                          # region/object name (predicate arg)

    model = data = None
    if _HAS_MUJOCO:
        try:
            model, data = _unwrap_sim(env)
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, target)
            if sid >= 0:
                return data.site_xpos[sid].copy()       # authoritative goal surface
        except Exception:
            pass

    base = target
    for suf in ("_contain_region", "_cook_region", "_top_side", "_top_region", "_region", "_side"):
        base = base.replace(suf, "")
    if obs is not None and f"{base}_pos" in obs:
        return np.array(obs[f"{base}_pos"], dtype=float)
    if model is not None:
        for name in (base, f"{base}_main"):
            bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return data.xpos[bid].copy()
    return None


def obs_from_libero(env_obs: dict, object_keys: list[str],
                    safety_radius: float = 0.10) -> list[ObstacleConfig]:
    """Build ObstacleConfig list from LIBERO observation keys."""
    obstacles = []
    for key in object_keys:
        if key in env_obs:
            pos = np.array(env_obs[key][:3], dtype=float)
            obstacles.append(ObstacleConfig(
                pos=pos, radius=0.04, safety_radius=safety_radius,
                name=key.replace("_pos", ""),
            ))
        else:
            print(f"  [obs_from_libero] key '{key}' not in obs — skipping")
    return obstacles


# ── Visualization helper ───────────────────────────────────────────────────────
_DISPLAY_SCALE = 2          # upscale factor for live viewer / saved video
_STATUS_BAR_H  = 64         # height in pixels of the overlay bar (after upscale)
_CAM_RES       = 224        # preprocessed image size sent to VLA (env renders at 256, resized to 224)

# Fixed window name shared across all trials so cv2.imshow updates the same
# window rather than creating new ones.  On macOS, multiple windows with
# different names can leave "frozen" ghost windows between episodes.
_CV2_WINDOW = "LIBERO CBF Benchmark"


def _cbf_overlay(
    img_rgb: np.ndarray,
    model,
    data,
    ee_center: np.ndarray,
    ee_q_diag: np.ndarray,
    R1: np.ndarray,
    obstacles: list,
    h_values: list,
    cbf_triggered: bool,
    cam_name: str = "agentview",
) -> np.ndarray:
    """Draw translucent CBF safety regions onto img_rgb (224x224, post-flip RGB).

    Draws:
      • Obstacle safety sphere  — filled red/orange circle
      • EE ellipsoid boundary   — filled green/yellow convex hull
      • h value text next to each obstacle
    """
    import mujoco as _mujoco

    H, W = img_rgb.shape[:2]
    try:
        cam_id = _mujoco.mj_name2id(model, _mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
        if cam_id < 0:
            return img_rgb
        cam_pos = data.cam_xpos[cam_id].copy()
        cam_mat = data.cam_xmat[cam_id].reshape(3, 3).copy()
        fovy_rad = np.deg2rad(model.cam_fovy[cam_id])
        f = (H / 2) / np.tan(fovy_rad / 2)
    except Exception:
        return img_rgb

    def _proj(p_world):
        """World → pixel on the post-flip 224×224 image, or None if behind cam."""
        d = cam_mat.T @ (p_world - cam_pos)
        if d[2] >= -1e-4:          # behind or right at camera plane
            return None
        u = f * d[0] / (-d[2]) + W / 2
        v = -f * d[1] / (-d[2]) + H / 2
        # _preprocess does [::-1, ::-1] → 180° flip
        u = W - 1 - u
        v = H - 1 - v
        return int(np.clip(u, 0, W - 1)), int(np.clip(v, 0, H - 1))

    def _proj_r(radius_m, p_world):
        d = cam_mat.T @ (p_world - cam_pos)
        depth = max(-d[2], 0.01)
        return max(2, int(f * radius_m / depth))

    overlay = img_rgb.copy()

    # ── obstacle safety spheres ────────────────────────────────────────────
    for i, ob in enumerate(obstacles):
        h = h_values[i] if i < len(h_values) else float("inf")
        ctr = _proj(ob.pos)
        if ctr is None:
            continue
        r_px = _proj_r(ob.safety_radius, ob.pos)
        # BGR: red when triggered, orange when close (h<0.5), dim blue when safe
        if cbf_triggered or h < 0:
            col = (40,  40,  220)   # red
        elif h < 0.5:
            col = (0,  120, 255)    # orange
        else:
            col = (60, 100, 180)    # muted blue
        cv2.circle(overlay, ctr, r_px, col, -1)
        cv2.circle(overlay, ctr, r_px, (255, 255, 255), 1)
        label_y = max(ctr[1] - r_px - 4, 8)
        cv2.putText(overlay, f"h={h:.2f}",
                    (ctr[0] - 16, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

    # ── EE ellipsoid (sample surface points → project → convex hull) ───────
    ee_pts: list[tuple[int, int]] = []
    for theta in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        for phi in np.linspace(0.2, np.pi - 0.2, 6):
            s_vec = np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi),
            ])
            p_world = ee_center + R1 @ (ee_q_diag * s_vec)
            pt = _proj(p_world)
            if pt is not None:
                ee_pts.append(pt)

    if len(ee_pts) >= 3:
        hull = cv2.convexHull(np.array(ee_pts, dtype=np.int32))
        ee_col = (50, 200, 50) if not cbf_triggered else (0, 200, 220)  # green / yellow
        cv2.fillConvexPoly(overlay, hull, ee_col)
        cv2.polylines(overlay, [hull], True, (255, 255, 255), 1)

    # ── EE centre dot ──────────────────────────────────────────────────────
    ctr_ee = _proj(ee_center)
    if ctr_ee is not None:
        cv2.circle(overlay, ctr_ee, 3, (255, 255, 255), -1)

    # 0.40 original + 0.60 overlay → translucent effect
    return cv2.addWeighted(img_rgb, 0.40, overlay, 0.60, 0)


def _render_frame(img_rgb: np.ndarray, t: int, horizon: int, mode: str,
                  min_dist: float, cbf_triggered: bool, collision_flag: bool,
                  episode_idx: int, vla_cnt: int,
                  h_val: float = float("inf")) -> np.ndarray:
    """Return a BGR display frame: upscaled camera image + status bar.

    img_rgb has already been flipped by _preprocess (and optionally CBF overlay
    drawn on top).
    """
    s = _DISPLAY_SCALE
    h, w = img_rgb.shape[:2]
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (w * s, h * s), interpolation=cv2.INTER_NEAREST)

    bar = np.zeros((_STATUS_BAR_H, w * s, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE  = (220, 220, 220)
    GRAY   = (130, 130, 130)
    YELLOW = (0,   200, 200)
    RED    = (60,  60,  220)

    cbf_col  = YELLOW if cbf_triggered else GRAY
    coll_col = RED    if collision_flag else GRAY
    h_finite = h_val if np.isfinite(h_val) else 99.0
    h_col    = RED if h_finite < 0 else (YELLOW if h_finite < 0.5 else GRAY)

    cv2.putText(bar, f"step {t:03d}/{horizon}",                       (8,   22), font, 0.52, GRAY,    1)
    cv2.putText(bar, f"[{mode.upper()}]",                             (8,   50), font, 0.58, WHITE,   1)
    cv2.putText(bar, f"obs {min_dist:.3f}m",                          (160, 22), font, 0.52, WHITE,   1)
    cv2.putText(bar, f"CBF {'ON' if cbf_triggered else 'off'}",       (160, 50), font, 0.52, cbf_col, 1)
    cv2.putText(bar, f"h={h_finite:.2f}",                             (300, 22), font, 0.52, h_col,   1)
    cv2.putText(bar, f"collision {'YES' if collision_flag else 'no'}",(300, 50), font, 0.52, coll_col, 1)
    cv2.putText(bar, f"ep {episode_idx}  VLA #{vla_cnt}",            (390, 22), font, 0.52, GRAY,    1)

    return np.vstack([frame, bar])


# ── Main trial loop ────────────────────────────────────────────────────────────
def run_libero_trial(
    env,
    obstacles: list[ObstacleConfig],
    instruction: str,
    goal_pos: np.ndarray | None = None,
    use_cbf: bool = True,
    cbf_gamma: float = 1.8,
    goal_tolerance: float = 0.08,
    scene_name: str = "libero",
    collect_dataset: bool = False,
    collect_cbf_data: bool = False,
    save_results: bool = False,
    results_dir: str = "results",
    dataset_path: str | None = None,
    cbf_dataset_path: str | None = None,
    show_viewer: bool = False,
    save_video: str | None = None,
    # SafeLIBERO episode parameters
    episode_idx: int = 0,
    initial_states=None,
    auto_detect_obstacle: bool = False,
    obstacle_safety_radius: float = 0.10,
    # When True and goal_pos is None, resolve the goal from the BDDL goal predicate's
    # region site (frame-consistent). Feeds the geo-success fallback + goal-distance
    # logging/shaping; env.check_success stays authoritative. See resolve_goal_from_bddl.
    auto_goal: bool = False,
    # Whether the geo-distance fallback may COUNT as success. Default True (benchmark).
    # Set False for RL so success is env.check_success ONLY (authoritative, AEGIS-
    # comparable) while still getting goal-distance reward shaping from auto_goal.
    use_geo_success: bool = True,
    # VLA backend: "openvla" or "pi05"
    vla: str = "openvla",
    openvla_port: int = 8000,
    pi05_host: str = "127.0.0.1",
    pi05_port: int = 8000,
    # None = auto-select per VLA: openvla→True, pi05→False
    use_gvr: bool | None = None,
    use_gripper_hysteresis: bool | None = None,
    # None = auto: True for pi05 (match AEGIS paper), False for openvla
    translational_only: bool | None = None,
    # Synchronous replan parameters (AEGIS approach)
    replan_steps: int = 5,
    horizon: int = 400,
    # Obstacle geometry: sphere decomposition vs single ellipsoid
    use_sphere_decomp: bool = True,
    sphere_decomp_n: int = 48,
    # Obstacle-conditioned projector: send obs features to server when True
    use_obs_cond: bool = False,
    # APF mode: smooth potential-field repulsion instead of CBF QP
    use_apf: bool = False,
    apf_k_rep: float = 2.0,
    apf_d_influence: float = 0.20,
    # Safety-conditioned prompting: inject obstacle warning into language instruction
    use_safety_prompt: bool = False,
    # AEGIS-faithful mode (default): apply the CBF on EVERY step whenever an
    # obstacle is present, with the single-ellipsoid barrier and NO local
    # heuristics (no _near_goal gate, no grasp-commit/pull, no placement force-
    # release, no post-trigger slowdown, no arm-link constraints, no sphere
    # decomposition). This replicates AEGIS/VLSA main_aegis_translational.py,
    # whose only gate is a per-episode "obstacle present?" flag. Set False to
    # restore the legacy heuristic behaviour.
    aegis_faithful: bool = True,
    # Co-located RL rollout (Option B): in-process policy override. When provided,
    # actions come from policy_fn(img, wrist_img, state, instruction, num_actions) ->
    # (list[7-D action], QueryTrace | None) instead of the VLA server. This bypasses
    # the websocket entirely (env + model share the process). When record_policy_trace
    # is True, each returned QueryTrace is appended to metrics.policy_trace for the
    # on-policy GRPO update. The CBF shield, collision attribution and reward metrics
    # are untouched — only the action SOURCE changes.
    policy_fn=None,
    record_policy_trace: bool = False,
    controller=None,
    label_controller=None,
) -> MetricsTracker:
    """Run one LIBERO episode using OpenVLA + optional Cartesian CBF.

    Action pipeline (matches AEGIS synchronous replan approach):
      1. Every `replan_steps` control steps, synchronously query VLA server
         for a chunk of `replan_steps` actions.
      2. Execute each action in the chunk sequentially (one per control step).
      3. _normalize_gripper()  maps gripper [0,1] -> [-1,+1], then binarises
      4. _invert_gripper()     flips sign to match LIBERO robosuite convention
      5. _run_cartesian_cbf()  (when use_cbf=True) filters action[:3] to keep
                               all arm links outside obstacle safety zones
      6. env.step(action)      OSC_POSE controller handles Cartesian -> joint

    SafeLIBERO mode: pass initial_states (from make_libero_env) and
    auto_detect_obstacle=True to use per-episode randomised scenes and
    displacement-based collision detection matching the AEGIS paper.
    """
    if not _HAS_MUJOCO:
        raise RuntimeError("mujoco not available in this environment")

    # Only spin up the π0.5 websocket client when the VLA server is actually the action source
    # (not for an in-process policy_fn, and not for the scripted classical controller).
    if vla == "pi05" and policy_fn is None and controller is None:
        _init_pi05_client(pi05_host, pi05_port)

    # Resolve auto defaults: OpenVLA keeps original True behaviour;
    # π0.5 defaults to False (closes gripper early, making both features mis-fire).
    _is_pi05 = (vla == "pi05")
    _use_gvr               = use_gvr               if use_gvr               is not None else (not _is_pi05)
    _use_gripper_hysteresis = use_gripper_hysteresis if use_gripper_hysteresis is not None else (not _is_pi05)
    _use_translational_only = translational_only if translational_only is not None else _is_pi05

    if use_apf:
        mode = "apf"
    elif use_cbf:
        mode = "cbf"
    else:
        mode = "plain"

    # ── Video writer ─────────────────────────────────────────────────────────
    _vwriter = None
    _frame_w = _CAM_RES * _DISPLAY_SCALE
    _frame_h = _CAM_RES * _DISPLAY_SCALE + _STATUS_BAR_H
    if save_video:
        import os
        os.makedirs(os.path.dirname(save_video) or ".", exist_ok=True)
        _vwriter = cv2.VideoWriter(
            save_video, cv2.VideoWriter_fourcc(*"mp4v"), 20, (_frame_w, _frame_h))

    # ── Reset env and load SafeLIBERO episode state ──────────────────────────
    env.reset()
    if initial_states is not None:
        obs = env.set_init_state(initial_states[episode_idx])
    else:
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # ── Extract MuJoCo model/data for CBF Jacobian computation ───────────────
    model, data  = _unwrap_sim(env)
    arm_body_ids = _get_arm_body_ids(model)
    arm_dof_idx  = _get_arm_dof_indices(model)
    ee_body_id   = arm_body_ids[-1] if arm_body_ids else 0
    print(f"  Arm bodies: {len(arm_body_ids)}  Arm DOFs: {arm_dof_idx}")

    metrics = MetricsTracker(scene_name, mode, model_name=vla)
    # On-policy GRPO trace buffer (populated only when policy_fn records traces).
    metrics.policy_trace = []
    # Parallel to policy_trace: the executed shield-corrected actions per query, collected for
    # shield-as-expert DAgger BC (Exp 005). One sub-list per query, one action per control step.
    _shielded_bufs: list[list] = []

    goal_str = np.round(goal_pos, 3) if goal_pos is not None else "auto"
    print(f"\n  [{scene_name}] ep={episode_idx}  mode={mode.upper()}  "
          f"obstacles={len(obstacles)}  goal={goal_str}  replan={replan_steps}")

    # ── Warm-up: let physics settle before querying VLA ───────────────────────
    for _ in range(_WARMUP_STEPS):
        step_out = env.step(_DUMMY_ACTION.tolist())
        if isinstance(step_out, tuple):
            obs = step_out[0]
        else:
            obs = step_out

    # ── Install 3D CBF render hook (after first step initialises render ctx) ──
    _viz_hook_ok = False
    if _HAS_VIZ and (show_viewer or save_video):
        _viz_hook_ok = install_scene_hook(env.sim)

    # ── Object tracking setup ─────────────────────────────────────────────────
    # Collect absolute object position keys only — exclude robot keys and the
    # relative-vector keys (e.g. "bowl_to_robot0_eef_pos") which are NOT object
    # positions and would corrupt grasp detection.
    _obj_pos_keys = sorted([
        k for k in obs.keys()
        if k.endswith("_pos")
        and not k.startswith("robot")
        and "to_robot" not in k
    ])
    _obj_initial_z = {k: float(obs[k][2]) for k in _obj_pos_keys if k in obs}
    print(f"\n  Objects in scene ({len(_obj_pos_keys)}):")
    for k in _obj_pos_keys:
        if k in obs:
            p = np.array(obs[k])
            print(f"    {k:<44} z={p[2]:.3f}m  pos=[{p[0]:.3f},{p[1]:.3f},{p[2]:.3f}]")
    _grasp_flag          = False
    _grasped_object:      str | None = None
    _last_grasped_object: str | None = None   # persists after gripper opens
    _last_release_step:   int        = -999   # step when grasp last went True→False
    _goal_hold_steps = 0  # steps continuously holding object at goal (for force-release)

    # ── GVR: Grasp Verification & Recovery ───────────────────────────────────
    _gvr = GVR()
    _gvr.reset()

    _gripper_close_chunks = 0
    _gripper_locked       = False
    _gripper_locked_steps = 0

    # ── SafeLIBERO obstacle auto-detection ────────────────────────────────────
    # After warm-up, scan env joint names for obstacle objects that have
    # settled within the workspace bounds, then build the CBF obstacle config.
    if auto_detect_obstacle:
        detected = detect_safelibero_obstacle(env, obs, safety_radius=obstacle_safety_radius)
        if detected is not None:
            obstacles = [detected]
            print(f"  Obstacle detected: '{detected.name}' at {np.round(detected.pos, 3)}"
                  f"  r_safe={obstacle_safety_radius:.2f}m")
        else:
            obstacles = []
            print("  [warn] No obstacle found in workspace — running without CBF obstacles")

    # ── Resolve the task goal from the BDDL predicate (frame-consistent) ──────
    # Only when not supplied explicitly; used for the geo-success fallback + logging.
    if goal_pos is None and auto_goal:
        goal_pos = resolve_goal_from_bddl(env, obs)
        if goal_pos is not None:
            print(f"  Goal (from BDDL): {np.round(goal_pos, 3)}")
        else:
            print("  [warn] Could not resolve goal from BDDL — geo-success fallback disabled")

    # ── Classical scripted expert: resolve pick-and-place context once ─────────
    # When a `controller` is supplied it replaces the VLA as the action source, driven by
    # privileged sim poses (object + goal) to produce optimal safe demos. The CBF still filters.
    # A `controller` DRIVES the arm (classical expert as action source). A `label_controller`
    # LABELS the VLA's states with the expert action for DAgger (the VLA drives via policy_fn,
    # the classical expert says what it would do → recorded as the BC target). Both need the same
    # pick-and-place context + multi-obstacle avoid-list.
    def _grip_width(_obs):
        """Panda finger separation (m): open ≈ 0.08, closed-on-object ≈ 0.004–0.05. The reactive
        classical expert's observable grasp signal."""
        _q = np.asarray(_obs.get("robot0_gripper_qpos", (0.04, -0.04)), dtype=float)
        return float(_q[0] - _q[1]) if _q.shape[0] >= 2 else None

    _pp_ctx = None
    if controller is not None or label_controller is not None:
        from experiments.classical_expert import resolve_pick_and_place, _Obs as _MpcObs
        _pp_ctx = resolve_pick_and_place(env, obs)
        if _pp_ctx is None:
            raise RuntimeError("classical controller: could not resolve pick-and-place target "
                               "from the BDDL goal predicate")
        _pp_ctx["table_z"] = float(_pp_ctx["obj_pos"][2])
        _avoid = list(obstacles)                       # ObstacleConfig(s): have .pos + .safety_radius
        _tgt_key = _pp_ctx["obj_key"]
        _gp = np.asarray(_pp_ctx["goal_pos"], dtype=float)
        for _k in _obj_pos_keys:
            if _k == _tgt_key or _k not in obs:
                continue
            _pp = np.asarray(obs[_k][:3], dtype=float)
            if np.linalg.norm(_pp - _gp) < 0.12:       # the goal-surface object (plate) — don't avoid
                continue
            if abs(_pp[0]) > 1.0 or abs(_pp[1]) > 1.0:  # off-table pruned obstacle — ignore
                continue
            _avoid.append(_MpcObs(_pp, 0.045))         # clutter object, modest keep-out radius
        _pp_ctx["avoid"] = _avoid
        # IK grasp ORIENTATION: the wrist tilt that lets OSC_POSE descend to the grip point. A
        # zero-rotation descent floors ~5cm high on elevated bowls (cabinet/stove); commanding this
        # orientation reaches all the way, in the OSC_POSE action space the VLA also uses.
        _pp_ctx["grasp_R"] = None
        _grip_pt = np.asarray(_pp_ctx["obj_pos"], float).copy()
        if _pp_ctx.get("grasp_mode") == "rim":
            _rimoff = getattr(getattr(controller or label_controller, "cfg", None), "rim_offset", 0.05)
            _grip_pt = _grip_pt + np.array([0.0, _rimoff, 0.0])
        _grip_pt[2] += 0.005
        # ONLY for ELEVATED grasps (obj on a stove/cabinet, z > ~1.0): a normal top-down descent
        # reaches table-height bowls fine, and the wrist tilt REGRESSES them (breaks the table grasp).
        _elevated = float(_grip_pt[2]) > 1.0
        _arm_qadr, _jnt_rng = _get_arm_qpos_indices(model)
        _grip_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_grip_site")
        if _elevated and _grip_sid >= 0 and _arm_qadr:
            _Rg, _ik_ok = _ik_grasp_orientation(model, data, _grip_pt, _arm_qadr, arm_dof_idx,
                                                _grip_sid, _jnt_rng)
            _pp_ctx["grasp_R"] = _Rg if _ik_ok else None
            print(f"  [classical] elevated grasp (z={_grip_pt[2]:.2f}) → IK orientation: "
                  f"{'solved' if _ik_ok else 'unreachable'}")
        for _c in (controller, label_controller):      # configure whichever are in play
            if _c is None:
                continue
            _c.reset()
            if hasattr(_c, "place_mode"):
                _c.place_mode = _pp_ctx.get("place_mode", "in")
            if hasattr(_c, "grasp_mode"):
                _c.grasp_mode = _pp_ctx.get("grasp_mode", "top")
        _role = "drive" if controller is not None else "LABEL"
        print(f"  [classical:{_role}] pick '{_pp_ctx['obj_key'].replace('_pos','')}' "
              f"@ {np.round(_pp_ctx['obj_pos'],3)} → goal {np.round(_pp_ctx['goal_pos'],3)}  "
              f"[grasp={_pp_ctx.get('grasp_mode','top')} place={_pp_ctx.get('place_mode','in')}] "
              f"avoid={len(_avoid)}")

    # ── Safety-conditioned prompt ─────────────────────────────────────────────
    # Replace the bare task instruction with one that names the obstacle and its
    # spatial position relative to the robot, so π0.5 can plan around it.
    _instruction_effective = instruction
    if use_safety_prompt and obstacles:
        _instruction_effective = generate_safety_prompt(instruction, obstacles[0])
        print(f"  Safety prompt: '{_instruction_effective}'")

    # ── Sphere decomposition for active obstacle ──────────────────────────────
    # Compute once per episode (obstacles are static).  Replaces the single
    # coarse ellipsoid with N small spheres tracing the actual object surface,
    # giving the CBF accurate geometry for concave / irregular obstacles.
    # Falls back to ellipsoid CBF automatically if decomposition fails.
    _sphere_decomp: list | None = None
    if aegis_faithful and use_cbf and obstacles:
        print("  [AEGIS-faithful] CBF applied every step · single ellipsoid · "
              "no local heuristics")
    if use_cbf and use_sphere_decomp and not aegis_faithful and obstacles and _HAS_VIZ and _HAS_MUJOCO:
        from experiments.cbf_visualizer import decompose_obstacle_to_spheres
        ob0_name = obstacles[0].name
        for suffix in ("_main", ""):
            _sphere_decomp = decompose_obstacle_to_spheres(
                model, data, f"{ob0_name}{suffix}",
                n_spheres=sphere_decomp_n,
                r_sphere=0.015,   # surface marker; clearance also from EE sphere radii
                safety_margin=0.010,
            )
            if _sphere_decomp is not None:
                break
        if _sphere_decomp is not None:
            print(f"  Sphere decomp: {len(_sphere_decomp)} spheres for '{ob0_name}'")
        else:
            print(f"  [warn] Sphere decomp failed for '{ob0_name}' — falling back to ellipsoid CBF")

    # Record initial obstacle position for displacement-based collision check.
    _obstacle_name: str | None = None
    _initial_obstacle_pos: np.ndarray | None = None
    _obstacle_key_set: set[str] = set()
    if obstacles:
        _obstacle_name = obstacles[0].name
        _obstacle_key_set = {f"{ob.name}_pos" for ob in obstacles}
    _collision_flag = False
    # Collision-source attribution: what bodies touch the obstacle over the episode,
    # and specifically what was touching it at the moment displacement first fired.
    _obs_touch_seen: set[str] = set()
    _collision_culprits: set[str] = set()
    _collision_robot_caused = False   # robot caused it (directly or via push chain)
    from collections import deque as _deque
    _robot_chain_window = _deque(maxlen=8)   # recent robot→obstacle chain connectivity

    # ── Ellipsoid CBF state (per-obstacle auxiliary z direction) ─────────────
    # R1: EE rotation matrix (updated each step from eef_quat)
    # z_states: unit-sphere auxiliary directions, one per obstacle
    from scipy.spatial.transform import Rotation as _SciRot
    eef_quat_init = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
    _R1 = _SciRot.from_quat(eef_quat_init).as_matrix()   # scipy [x,y,z,w]

    _z_states: list[np.ndarray] = []
    ee_pos_init = np.array(obs.get("robot0_eef_pos", [0, 0, 0]), dtype=float)
    p1_init = ee_ellipsoid_center(ee_pos_init, _R1)
    for ob in obstacles:
        _z_states.append(init_z(p1_init, ob.pos))

    # ── DAgger self-safe expert: shield the LABEL ────────────────────────────
    # The classical expert's nominal action is NOT collision-safe on its own (it only avoids
    # obstacle POINTS at the EE; the arm/gripper still hit tall obstacles — measured ~60% collision
    # unshielded). The SELF-SAFE expert we distil is classical + CBF. During a DAgger rollout the
    # VLA drives UNSHIELDED (use_cbf=False) so it visits its own states, but each label must be the
    # SAFE action there — so we run the SAME ellipsoid CBF the driver uses on the expert's nominal,
    # with an independent z-state (the driving shield is off, so no state conflict).
    _label_z_states: list[np.ndarray] = []
    if label_controller is not None:
        for ob in obstacles:
            _label_z_states.append(init_z(p1_init, ob.pos))

    def _shield_label_action(a7, ee_p, R1):
        """Ellipsoid-CBF-correct a nominal expert action → the arm-aware self-safe label.
        Mirrors the driver's aegis-faithful ellipsoid shield (which gives 0 collision); keeps its
        own z-state. No-op if there is no obstacle geometry."""
        if not obstacles or not _label_z_states:
            return a7
        ob0 = obstacles[0]
        obs_q = ob0.q_diag if ob0.q_diag is not None else np.array([ob0.safety_radius] * 3)
        if ob0.q_R is not None:
            obs_R = ob0.q_R
        elif _HAS_MUJOCO:
            _bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob0.name)
            obs_R = data.xmat[_bid].reshape(3, 3).copy() if _bid >= 0 else np.eye(3)
        else:
            obs_R = np.eye(3)
        _ee_q1 = (EE_Q_DIAG_TALL if any(k in instruction.lower() for k in _TALL_OBJECT_KEYS)
                  else EE_Q_DIAG_DEFAULT)
        u_safe, z_new, _h, _trig = run_ellipsoid_cbf(
            ee_pos=ee_p, R1=R1, Q1_diag=_ee_q1,
            obs_pos=ob0.pos, obs_q=obs_q, z=_label_z_states[0],
            u_nom=np.asarray(a7[:3], dtype=float), k_cbf=K_CBF, scale=1.0,
            extra_lin_constraints=[], obs_R=obs_R,
        )
        _label_z_states[0] = z_new
        out = np.asarray(a7, dtype=float).copy()
        out[:3] = u_safe
        return out

    import time as _time

    # OffScreenRenderEnv is a wrapper (self.env holds the real env), so
    # env.horizon doesn't exist at the wrapper level — fall back to the
    # explicit parameter passed from run_task() rather than the default 800.
    horizon      = getattr(getattr(env, "env", None), "horizon", horizon)
    action_queue: list[np.ndarray] = []
    vla_cnt      = 0
    _current_action = _DUMMY_ACTION.copy()
    vla_raw      = np.zeros(7)
    _prev_min_d  = float("inf")   # CBF constraint distance from previous step

    _vla_executor = ThreadPoolExecutor(max_workers=1)

    try:
        for t in range(horizon):

            # ── 1. Camera observations ────────────────────────────────────
            if isinstance(obs, dict):
                img_raw   = obs.get("agentview_image")
                wrist_raw = obs.get("robot0_eye_in_hand_image")
            else:
                img_raw = wrist_raw = None

            img       = _preprocess(img_raw)   if img_raw   is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            wrist_img = _preprocess(wrist_raw) if wrist_raw is not None else np.zeros((224, 224, 3), dtype=np.uint8)
            state     = _build_proprio(obs) if isinstance(obs, dict) else np.zeros(8)

            # ── 2. Synchronous replan with fresh observations ─────────────
            if not action_queue:
                try:
                    _t0 = _time.perf_counter()
                    if controller is not None:
                        # Classical scripted expert: one action per step (queue length 1 → the
                        # controller is re-queried every control step for closed-loop behaviour).
                        _ee_now = np.array(obs["robot0_eef_pos"], dtype=float)
                        _obj_now = np.array(obs.get(_pp_ctx["obj_key"], _pp_ctx["obj_pos"]),
                                            dtype=float)
                        _nominal, _cphase = controller.act(
                            _ee_now, _obj_now, _pp_ctx["goal_pos"],
                            obstacles=_pp_ctx.get("avoid"), table_z=_pp_ctx["table_z"],
                            gripper=_grip_width(obs))
                        # Servo the wrist toward the IK grasp orientation during the pick so OSC can
                        # reach elevated bowls (zero-rotation descent floors ~5cm high).
                        if _pp_ctx.get("grasp_R") is not None and _cphase in ("DESCEND", "GRASP"):
                            _Rcur = _SciRot.from_quat(
                                np.asarray(obs.get("robot0_eef_quat", [0, 0, 0, 1]), float)).as_matrix()
                            _drot = _SciRot.from_matrix(_pp_ctx["grasp_R"] @ _Rcur.T).as_rotvec()
                            _nominal = np.asarray(_nominal, float).copy()
                            _nominal[3:6] = np.clip(4.0 * _drot, -1.0, 1.0)
                        action_queue = [_nominal.copy()]
                        vla_cnt += 1
                        if vla_cnt % 10 == 1:
                            print(f"  [{t:03d}] [classical] phase={_cphase}  "
                                  f"EE=[{_ee_now[0]:.3f},{_ee_now[1]:.3f},{_ee_now[2]:.3f}]")
                        # Record a BC demo every replan_steps: obs now + (via _shielded_bufs) the
                        # executed expert actions until the next capture. Dummy chain (BC ignores it).
                        if record_policy_trace and (t % replan_steps == 0):
                            from experiments.policy_trace import QueryTrace as _QT
                            _obs_raw = {"image": np.asarray(img, np.uint8),
                                        "wrist_image": np.asarray(wrist_img, np.uint8),
                                        "state": np.asarray(state, np.float32),
                                        "prompt": _instruction_effective}
                            metrics.policy_trace.append(_QT(
                                chain=np.zeros((2, 1, 1), np.float32),
                                logp_old=np.zeros(1, np.float32),
                                sigmas=np.array([1.0, 0.0], np.float32),
                                noise_level=0.0, sde_type="classical", obs=_obs_raw))
                            _shielded_bufs.append([])
                    elif policy_fn is not None:
                        # Co-located in-process policy (Option B): returns env-ready
                        # 7-D actions + the flow-SDE QueryTrace for this chunk.
                        raw_chunk, _qtrace = policy_fn(
                            img, wrist_img, state,
                            _instruction_effective, replan_steps)
                        action_queue = [np.asarray(a, dtype=np.float64).copy()
                                        for a in raw_chunk]
                        if _use_translational_only:
                            for a in action_queue:
                                a[3:6] = 0.0  # zero rotational deltas — match AEGIS setup
                        if record_policy_trace and _qtrace is not None:
                            metrics.policy_trace.append(_qtrace)
                            _shielded_bufs.append([])   # start this query's executed-action buffer
                    elif vla == "pi05":
                        raw_chunk = _query_pi05_chunk(img, wrist_img, state,
                                                      _instruction_effective, num_actions=replan_steps)
                        action_queue = [a.copy() for a in raw_chunk]
                        if _use_translational_only:
                            for a in action_queue:
                                a[3:6] = 0.0  # zero rotational deltas — match AEGIS paper setup
                    else:
                        _ovla_url = f"http://127.0.0.1:{openvla_port}/act"
                        # Obstacle features for obstacle-conditioned projector
                        _obs_feat = None
                        if use_obs_cond and obstacles:
                            _ee_now = np.array(obs["robot0_eef_pos"], dtype=float)
                            _dists  = [np.linalg.norm(_ee_now - ob.pos) for ob in obstacles]
                            _near   = obstacles[int(np.argmin(_dists))]
                            _obs_feat = _compute_obs_features(_ee_now, _near.pos)
                        raw_chunk = _query_openvla_chunk(img, wrist_img, state,
                                                         _instruction_effective, num_actions=replan_steps,
                                                         url=_ovla_url,
                                                         obstacle_feat=_obs_feat)
                        action_queue = [_post_process_vla(a) for a in raw_chunk]
                    if controller is None:      # VLA-only logging (raw_chunk/vla_raw not set for controller)
                        vla_ms = (_time.perf_counter() - _t0) * 1000
                        vla_raw      = raw_chunk[0].copy()
                        vla_cnt     += 1
                        ee_now = np.array(obs["robot0_eef_pos"])
                        grip_str = "CLOSE" if action_queue[0][6] > 0 else "open"
                        # CBF margin from previous step (actual constraint distance)
                        cbf_margin_str = (f"  cbf_margin={_prev_min_d:.3f}m"
                                          if _prev_min_d < float("inf") else "")
                        # Nearest scene object (for target tracking)
                        obj_dist_str = ""
                        if _obj_pos_keys:
                            dists = {k: np.linalg.norm(np.array(obs[k]) - ee_now)
                                     for k in _obj_pos_keys if k in obs}
                            near_k = min(dists, key=dists.get)
                            obj_dist_str = f"  nearest={near_k.replace('_pos','')}({dists[near_k]:.3f}m)"
                        print(f"  [{t:03d}] VLA #{vla_cnt}  grip={grip_str}"
                              f"  EE=[{ee_now[0]:.3f},{ee_now[1]:.3f},{ee_now[2]:.3f}]"
                              f"{cbf_margin_str}{obj_dist_str}  ({vla_ms:.0f}ms)")
                except Exception as e:
                    print(f"  [{t:03d}] VLA query error (holding last action): {e}")
                    action_queue = [_current_action.copy()]

            _current_action  = action_queue.pop(0)
            _is_new_vla_chunk = (len(action_queue) == replan_steps - 1)

            # ── 3. Ellipsoid CBF filter on xyz component ──────────────────
            ee_pos = np.array(obs["robot0_eef_pos"], dtype=float)

            # Update EE rotation matrix from current quaternion
            eef_quat_now = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
            _R1 = _SciRot.from_quat(eef_quat_now).as_matrix()

            safe_action     = _current_action.copy()
            cbf_triggered   = False
            correction_norm = 0.0

            # Deactivate CBF when the EE is close to the placement goal.
            # Near the target the robot needs precise control for settling
            # the object; CBF micro-corrections interfere with this.
            _near_goal = (goal_pos is not None and
                          np.linalg.norm(ee_pos - goal_pos) < goal_tolerance)

            # Pre-check grasp commit range.  When the EE is within 12 cm of a
            # graspable target, CBF is also disabled — at this range the
            # obstacle is already cleared and CBF pushes the EE AWAY from the
            # bowl (opposite direction to the commit XY pull), which would
            # counteract the grasp assist entirely.
            _gc_near_key_pre, _gc_near_dist_pre = None, float("inf")
            if not _near_goal and not _grasp_flag:
                for _gck in _obj_pos_keys:
                    if _gck in _obstacle_key_set or _gck not in obs:
                        continue
                    _gcp = np.array(obs[_gck], dtype=float)
                    if _gcp[2] > 1.02:
                        continue
                    _d = float(np.linalg.norm(ee_pos - _gcp))
                    if _d < _gc_near_dist_pre:
                        _gc_near_dist_pre, _gc_near_key_pre = _d, _gck
            _in_gc_range = (_gc_near_key_pre is not None
                            and _gc_near_dist_pre < 0.12)

            h_val = float("inf")   # reset each step; overwritten by CBF/APF block
            # AEGIS applies the CBF on every step whenever an obstacle exists.
            # The _near_goal gate is a legacy heuristic that disabled the filter
            # exactly where Level-I obstacles sit → strip it in faithful mode.
            _cbf_active = use_cbf and obstacles and (aegis_faithful or not _near_goal)
            if _cbf_active:
                # Arm-link constraints are a local addition (not in AEGIS) — off
                # in faithful mode. Otherwise gated on distance so they only fire
                # when a forearm/wrist link can actually reach the obstacle.
                _arm_link_rows: list = []
                _ee_obs_d = float(np.linalg.norm(ee_pos - obstacles[0].pos))
                if not aegis_faithful and arm_body_ids and _ee_obs_d < 0.30:
                    _arm_link_rows = _compute_arm_link_constraints(
                        model, data, arm_body_ids, arm_dof_idx, ee_body_id,
                        R1=_R1, obstacles=obstacles, scale=1.0, k_cbf=K_CBF,
                    )

                if _sphere_decomp is not None:
                    # ── Sphere-decomposition CBF ──────────────────────────────
                    u_safe_world, h_val, cbf_triggered = run_sphere_decomp_cbf(
                        ee_pos=ee_pos,
                        R1=_R1,
                        obstacle_spheres=_sphere_decomp,
                        u_nom=_current_action[:3],
                        k_cbf=K_CBF,
                        scale=1.0,
                        extra_constraints=_arm_link_rows,
                        ee_spheres=get_ee_spheres(ee_pos, _R1),
                    )
                else:
                    # ── Ellipsoid CBF (fallback) ──────────────────────────────
                    ob0   = obstacles[0]
                    z0    = _z_states[0]
                    obs_q = ob0.q_diag if ob0.q_diag is not None else np.array([ob0.safety_radius] * 3)
                    # Prefer the MVEE principal-axis orientation (q_R). Fall back
                    # to the obstacle body frame, then identity.
                    if ob0.q_R is not None:
                        obs_R = ob0.q_R
                    elif _HAS_MUJOCO:
                        _ob_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob0.name)
                        obs_R = data.xmat[_ob_bid].reshape(3, 3).copy() if _ob_bid >= 0 else np.eye(3)
                    else:
                        obs_R = np.eye(3)
                    # AEGIS uses a taller EE ellipsoid (z=0.20) for tall cartons.
                    _ee_q1 = (EE_Q_DIAG_TALL
                              if any(k in instruction.lower() for k in _TALL_OBJECT_KEYS)
                              else EE_Q_DIAG_DEFAULT)
                    u_safe_world, z_new, h_val, cbf_triggered = run_ellipsoid_cbf(
                        ee_pos=ee_pos, R1=_R1, Q1_diag=_ee_q1,
                        obs_pos=ob0.pos, obs_q=obs_q, z=z0,
                        u_nom=_current_action[:3], k_cbf=K_CBF,
                        scale=1.0,
                        extra_lin_constraints=_arm_link_rows, obs_R=obs_R,
                    )
                    _z_states[0] = z_new

                correction_norm = float(np.linalg.norm(u_safe_world - _current_action[:3]))
                # 0.6× post-trigger slowdown is a local robustness knob, not in
                # AEGIS — skip it in faithful mode.
                if cbf_triggered and not aegis_faithful:
                    u_safe_world = 0.6 * u_safe_world
                safe_action[:3] = u_safe_world

            elif use_apf and obstacles and (aegis_faithful or not _near_goal):
                apf_xyz, _apf_dist, cbf_triggered = _apf_xyz_correction(
                    ee_pos, obstacles, _current_action[:3],
                    k_rep=apf_k_rep, d_influence=apf_d_influence,
                )
                correction_norm = float(np.linalg.norm(apf_xyz - _current_action[:3]))
                safe_action[:3] = apf_xyz
                h_val = _apf_dist

            _prev_min_d = h_val   # track real CBF barrier value for VLA print line
            _gvr_phase = "normal"

            # ── 3c/3d. Local grasp/placement heuristics (NOT in AEGIS) ────
            # AEGIS passes the VLA gripper command straight through and never
            # overrides translation for grasping. Everything below — the
            # placement force-release and the near-grasp descend/pull/lock — is a
            # local TSR-recovery hack. Skip it entirely in faithful mode.
            if not aegis_faithful:
              # ── 3c. Placement heuristic ─────────────────────────────────
              # If the arm has been holding an object AT the goal for several
              # consecutive steps, the VLA is likely stuck (OOD near target).
              # Force the gripper open so the bowl can settle on the plate.
              # LIBERO OSC_POSE convention: gripper -1 = open, +1 = close.
              if _near_goal and _grasp_flag:
                _goal_hold_steps += 1
                if _goal_hold_steps >= 8:
                    safe_action[6] = -1.0  # force release
              else:
                _goal_hold_steps = 0

              # ── 3d. Near-grasp commit ───────────────────────────────────
              # Reuses _in_gc_range / _gc_near_key_pre computed above.
              # CBF is already disabled in this range, so XY pull is uncontested.
              _GC_Z_STOP  = 0.010  # stop forcing down when EE z < target_z + 1 cm
              _GC_DESCENT = -0.35  # z override while descending into grasp

              if _in_gc_range:
                _gc_tgt_pos = np.array(obs[_gc_near_key_pre], dtype=float)
                _gc_tgt_z   = float(_gc_tgt_pos[2])
                safe_action[6] = 1.0   # lock gripper CLOSED
                # XY pull toward target (CBF disabled here so this is uncontested)
                _gc_xy_err  = _gc_tgt_pos[:2] - ee_pos[:2]
                _gc_xy_pull = np.clip(0.4 * _gc_xy_err, -0.012, 0.012)
                safe_action[0] += _gc_xy_pull[0]
                safe_action[1] += _gc_xy_pull[1]
                if ee_pos[2] > _gc_tgt_z + _GC_Z_STOP:
                    safe_action[2] = min(safe_action[2], _GC_DESCENT)
                    if t % 8 == 0:
                        print(f"  [{t:03d}] GC: descend+pull → "
                              f"{_gc_near_key_pre.replace('_pos','')} "
                              f"EEz={ee_pos[2]:.3f} tgtz={_gc_tgt_z:.3f} "
                              f"xy_err=[{_gc_xy_err[0]:+.3f},{_gc_xy_err[1]:+.3f}]")
                # Block upward motion until gripper physically closes.
                # VLA often commands lift immediately after a failed pinch.
                _gc_gq = np.abs(np.array(obs.get("robot0_gripper_qpos",
                                                   [0.04, 0.04]), dtype=float))
                if float(np.max(_gc_gq)) > 0.015:   # fingers still open
                    safe_action[2] = min(safe_action[2], 0.0)
              elif (_gc_near_key_pre is not None and _gc_near_dist_pre > 0.15):
                # Pre-open: VLA closes gripper too early (distribution shift).
                # Keep it open during far approach so fingers can wrap the object
                # when the near-grasp commit descends at d < 0.12 m.
                safe_action[6] = -1.0

            # ── 4. Safety monitoring ──────────────────────────────────────
            ee_c = ee_ellipsoid_center(ee_pos, _R1)
            if obstacles:
                dists_to_obs = [float(np.linalg.norm(ee_c - ob.pos))
                                for ob in obstacles]
                min_d = min(dists_to_obs)
                _closest_obs_idx = int(np.argmin(dists_to_obs))
                _closest_obs_pos = obstacles[_closest_obs_idx].pos.copy()
                violation = any(d < ob.safety_radius
                                for d, ob in zip(dists_to_obs, obstacles))
                if _sphere_decomp is not None:
                    # h_val already holds h_min from sphere CBF; wrap for display
                    h_values = [h_val]
                else:
                    h_values = _compute_h_values_ellipsoid(ee_pos, _R1, obstacles, _z_states)
            else:
                min_d, violation = float("inf"), False
                h_values = [float("inf")]
                _closest_obs_pos = None

            # ── 5. Display ────────────────────────────────────────────────
            if show_viewer or _vwriter is not None:
                # Push 3D geoms into MuJoCo's render scene (they appear in the
                # camera image automatically — no 2D pixel projection needed).
                if _viz_hook_ok and obstacles:
                    push_cbf_geoms(
                        env.sim,
                        ee_center=ee_c,
                        ee_q=EE_Q_DIAG_DEFAULT,
                        R1=_R1,
                        obstacles=obstacles,
                        h_values=h_values,
                        arm_body_ids=[],        # arm-link spheres removed from QP and viz
                        arm_radii={},
                        model=model,
                        data=data,
                        cbf_triggered=cbf_triggered,
                        obstacle_spheres=_sphere_decomp,
                        arm_sample_positions=None,
                    )
                # The hook already injected CBF geoms into the render context.
                # env.step() (below) will render the camera through the patched
                # context, so obs["agentview_image"] on the NEXT iteration will
                # include the CBF shapes — one-step lag that is imperceptible.
                # Do NOT call _update_observables here; it advances physics and
                # causes the sim to run at 1-step cadence instead of 8.
                display_img = img
                h_display = h_values[0] if h_values else float("inf")
                frame_disp = _render_frame(
                    display_img, t, horizon, mode, min_d, cbf_triggered,
                    _collision_flag, episode_idx, vla_cnt, h_val=h_display,
                )
                if _vwriter is not None:
                    _vwriter.write(frame_disp)
                if show_viewer:
                    cv2.imshow(_CV2_WINDOW, frame_disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # Record this step's imitation TARGET into the current query's buffer:
            #  • DAgger (label_controller set): the classical EXPERT's action at the VLA's current
            #    state (the VLA drives via policy_fn; the expert says what it would do) — this is
            #    the on-policy relabel that fixes covariate shift.
            #  • otherwise: the executed shield-/classical-corrected action (offline BC / Exp 005).
            if record_policy_trace and _shielded_bufs:
                if label_controller is not None and _pp_ctx is not None:
                    _obj_lbl = np.array(obs.get(_pp_ctx["obj_key"], _pp_ctx["obj_pos"]), dtype=float)
                    _lbl_action, _ = label_controller.act(
                        np.array(obs["robot0_eef_pos"], dtype=float), _obj_lbl,
                        _pp_ctx["goal_pos"], obstacles=_pp_ctx.get("avoid"),
                        table_z=_pp_ctx["table_z"], gripper=_grip_width(obs))
                    # Self-safe expert = classical nominal + CBF: shield the label so the DAgger
                    # target is collision-safe (the nominal alone collides ~60% unshielded).
                    _lbl_action = _shield_label_action(_lbl_action, ee_pos, _R1)
                    _shielded_bufs[-1].append(np.asarray(_lbl_action, dtype=np.float32).copy())
                else:
                    _shielded_bufs[-1].append(np.asarray(safe_action, dtype=np.float32).copy())

            # ── 6. Step environment ───────────────────────────────────────
            step_out = env.step(safe_action.tolist())
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            # ── 6b. Grasp detection: gripper closing + object lifted off table ─
            gripper_closing = safe_action[6] > 0
            if gripper_closing:
                for k in _obj_pos_keys:
                    if k in obs and k in _obj_initial_z:
                        lift = float(obs[k][2]) - _obj_initial_z[k]
                        if lift > 0.02 and not _grasp_flag:
                            _grasp_flag          = True
                            _grasped_object      = k
                            _last_grasped_object = k
                            ee_g = np.array(obs["robot0_eef_pos"])
                            print(f"  [{t:03d}] *** GRASPED: {k.replace('_pos','')} "
                                  f"lifted {lift:.3f}m  "
                                  f"EE=[{ee_g[0]:.3f},{ee_g[1]:.3f},{ee_g[2]:.3f}]")
            else:
                if _grasp_flag:
                    print(f"  [{t:03d}] *** RELEASED: {_grasped_object}")
                    _last_release_step = t
                _grasp_flag     = False
                _grasped_object = None
                # _last_grasped_object intentionally kept — used for geo-success check

            # ── 7. Displacement-based collision check (SafeLIBERO metric) ─
            # Attribution: scan live MuJoCo contacts to see which bodies (gripper,
            # arm link, held object, …) are touching the obstacle right now.
            _now_touch: set[str] = set()
            if _obstacle_name is not None and _HAS_MUJOCO:
                _grasp_base = (_grasped_object or _last_grasped_object or "")
                _grasp_base = _grasp_base.replace("_pos", "") if _grasp_base else None
                _now_touch = _obstacle_contact_culprits(
                    model, data, _obstacle_name, grasped_base=_grasp_base)
                _obs_touch_seen |= _now_touch
                # Windowed robot-causation: was the robot in a contact chain to the
                # obstacle in the last few steps? (Displacement crosses threshold a
                # step or two AFTER the push, so a single-step check misses it.)
                _robot_chain_window.append(
                    robot_caused_displacement(model, data, _obstacle_name))

            if _obstacle_name is not None:
                _obs_key = f"{_obstacle_name}_pos"
                if _obs_key in obs:
                    curr_obs_pos = np.array(obs[_obs_key], dtype=float)
                    if _initial_obstacle_pos is None:
                        _initial_obstacle_pos = curr_obs_pos.copy()  # step-0 baseline
            if (not _collision_flag
                    and _obstacle_name is not None
                    and _initial_obstacle_pos is not None):
                _obs_key = f"{_obstacle_name}_pos"
                if _obs_key in obs:
                    curr_obs_pos = np.array(obs[_obs_key], dtype=float)
                    # Collision threshold matched to AEGIS/VLSA (>0.001 m displacement).
                    if np.sum(np.abs(curr_obs_pos - _initial_obstacle_pos)) > 0.001:
                        _collision_flag = True
                        # Culprit = what's touching now, else the last body seen
                        # touching the obstacle this episode.
                        _collision_culprits = _now_touch or _obs_touch_seen or {"unknown"}
                        # Did the robot cause it (direct hit OR push chain)? This is
                        # what the RL reward penalises — it counts scene_object domino
                        # pushes as the robot's fault while excluding physics settling.
                        # Use the recent window (push may precede threshold-crossing).
                        _collision_robot_caused = (
                            any(_robot_chain_window)
                            or robot_caused_displacement(model, data, _obstacle_name))
                        print(f"  [{t:03d}] COLLISION: obstacle displaced "
                              f"{np.sum(np.abs(curr_obs_pos - _initial_obstacle_pos)):.4f}m"
                              f"  culprit={sorted(_collision_culprits)}"
                              f"  robot_caused={_collision_robot_caused}")

            # ── 8. Success check ──────────────────────────────────────────
            # BDDLBaseDomain.step() sets done = self._check_success() directly
            # but never writes info["success"].  Call check_success() directly
            # on the wrapper (ControlEnv.check_success → env._check_success)
            # as the authoritative signal; done and info["success"] as backups.
            try:
                task_success = env.check_success()
            except AttributeError:
                task_success = False

            # Geometric fallback: LIBERO's ObjectState.check_ontop uses a
            # 3 cm XY threshold (base_object_states.py:92) that rejects valid
            # off-center placements on the plate/target surface.  If the
            # official check fails, we test whether the last-grasped object
            # now rests within 10 cm of goal_pos in XY at approximately table
            # height — a physically meaningful "placed on target" criterion.
            geo_success = False
            if not task_success and goal_pos is not None and use_geo_success:
                _geo_candidate = _last_grasped_object
                if _geo_candidate is None:
                    # No confirmed grasp — scan for any non-obstacle object
                    # displaced upward that is now near the goal.
                    for k in _obj_pos_keys:
                        if "obstacle" in k or k not in obs:
                            continue
                        if float(obs[k][2]) - _obj_initial_z.get(k, 0.0) > 0.01:
                            _geo_candidate = k
                            break
                if _geo_candidate and _geo_candidate in obs:
                    obj_pos_geo = np.array(obs[_geo_candidate], dtype=float)
                    xy_dist_geo = np.linalg.norm(obj_pos_geo[:2] - goal_pos[:2])
                    # Frame-relative z gate: the object should rest near the goal SURFACE
                    # (goal_pos comes from the BDDL region site). A fixed 0.88 assumed the
                    # world frame and silently failed in the object suite (surface z≈0.07).
                    z_ok = goal_pos[2] - 0.06 < obj_pos_geo[2] < goal_pos[2] + 0.12
                    # Require the object was actually released (not just the gripper
                    # action momentarily dipping to ≤ 0 between VLA chunks) AND that
                    # at least 8 steps have passed since release so the physics can
                    # settle and confirm the object landed on the target surface.
                    settled = (
                        not _grasp_flag
                        and _last_release_step > 0
                        and (t - _last_release_step) >= 8
                    )
                    if xy_dist_geo < 0.10 and z_ok and settled:
                        geo_success = True
                        print(f"  [{t:03d}] GEO-SUCCESS: {_geo_candidate} "
                              f"xy_dist={xy_dist_geo:.3f}m z={obj_pos_geo[2]:.3f}m"
                              f"  (released at step {_last_release_step})")

            success_flag = task_success or geo_success or done or bool(info.get("success", False))
            if t % 8 == 0:  # print every chunk so we can see the value live
                print(f"  [{t:03d}] check_success={task_success}  geo={geo_success}  done={done}"
                      f"  info_keys={list(info.keys())}")
            if success_flag:
                print(f"  [{t:03d}] *** TASK SUCCESS ***"
                      + (" (geo fallback)" if geo_success and not task_success else ""))
                metrics.mark_goal_reached(t)

            # ── 9. Record ─────────────────────────────────────────────────
            q_current = np.array(obs.get("robot0_joint_pos", np.zeros(7)), dtype=float)
            # Pass zero goal_pos so MetricsTracker's internal EE-distance check
            # never fires — success detection goes through mark_goal_reached()
            # (called above when info["success"] is True).
            _goal_dist = (float(np.linalg.norm(ee_pos - goal_pos))
                          if goal_pos is not None else float("inf"))

            # Distance to closest target object (non-obstacle, not yet at goal)
            _obj_dist = float("inf")
            for _k in _obj_pos_keys:
                if _k in _obstacle_key_set or _k not in obs:
                    continue
                _op = np.array(obs[_k], dtype=float)
                if goal_pos is not None and np.linalg.norm(_op[:2] - goal_pos[:2]) < 0.06:
                    continue
                _d = float(np.linalg.norm(ee_pos - _op))
                if _d < _obj_dist:
                    _obj_dist = _d
            metrics.record(
                StepRecord(
                    step=t,
                    ee_pos=ee_pos,
                    min_dist=min_d,
                    closest_obstacle="",
                    closest_body="",
                    cbf_triggered=cbf_triggered,
                    cbf_correction_norm=correction_norm,
                    violation=violation,
                    collision_flag=_collision_flag,
                    q=q_current,
                    u_nom=_current_action.copy(),
                    u_safe=safe_action.copy(),
                    h_values=h_values,
                    vla_delta=vla_raw.copy(),
                    ghost_pos=None,
                    image=img.copy() if (collect_dataset or collect_cbf_data) else None,
                    wrist_image=wrist_img.copy() if collect_cbf_data else None,
                    eef_quat=np.array(obs.get("robot0_eef_quat", [0,0,0,1]), dtype=np.float32) if collect_cbf_data else None,
                    gripper_qpos=np.array(obs.get("robot0_gripper_qpos", [0,0]), dtype=np.float32) if collect_cbf_data else None,
                    safe_cartesian=safe_action.copy() if collect_cbf_data else None,
                    # Rich research logging
                    goal_dist=_goal_dist,
                    gripper_open=bool(safe_action[6] <= 0),
                    vla_query=_is_new_vla_chunk,
                    action_nom_xyz_mag=float(np.linalg.norm(_current_action[:3])),
                    action_safe_xyz_mag=float(np.linalg.norm(safe_action[:3])),
                    obstacle_pos=_closest_obs_pos,
                    # Failure-mode analysis
                    grasp_detected=_grasp_flag,
                    obj_dist=_obj_dist,
                ),
                goal_pos=np.zeros(3),
                goal_tolerance=0.0,
            )

            if t % 20 == 0:
                flags = []
                if _grasp_flag:     flags.append(f"HOLDING:{_grasped_object.replace('_pos','') if _grasped_object else '?'}")
                if _collision_flag: flags.append("COLLISION")
                if violation:       flags.append("VIOLATION")
                if cbf_triggered:   flags.append("[CBF]")
                if success_flag:    flags.append("SUCCESS")
                d_goal = (f"{np.linalg.norm(ee_pos - goal_pos):.3f}m"
                          if goal_pos is not None else "n/a")
                grip_s = "CLOSE" if safe_action[6] > 0 else "open"
                # Object positions snapshot
                obj_snap = "  ".join(
                    f"{k.replace('_pos','')}_z={obs[k][2]:.3f}"
                    for k in _obj_pos_keys if k in obs
                )
                print(f"  [{t:03d}] EE=[{ee_pos[0]:.3f},{ee_pos[1]:.3f},{ee_pos[2]:.3f}]"
                      f"  grip={grip_s}  d_goal={d_goal}  min_obs={min_d:.3f}m"
                      f"  {'  '.join(flags)}"
                      + (f"\n         objs: {obj_snap}" if obj_snap else ""))

            # Don't cut the episode while the arm is holding the object near
            # the goal — let the placement heuristic (section 3c) run its
            # 8-step countdown and force-release first.
            _holding_near_goal = _near_goal and _grasp_flag
            if (done or success_flag) and not _holding_near_goal:
                break

    finally:
        _vla_executor.shutdown(wait=False)
        if _vwriter is not None:
            _vwriter.release()
        if show_viewer:
            # waitKey(1) before and after destroyAllWindows is required on
            # macOS to actually flush the close event from the GUI event queue.
            # Without it the window stays "frozen" on screen between episodes.
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_results:
        import os
        os.makedirs(results_dir, exist_ok=True)
        label = f"{scene_name}_{mode}"
        metrics.save_step_log(     f"{results_dir}/{label}_steps.csv")
        metrics.save_summary(      f"{results_dir}/{label}_summary.csv")
        metrics.save_analysis_npz( f"{results_dir}/{label}_trajectory.npz")

    if collect_dataset and dataset_path:
        metrics.save_dataset(dataset_path)

    # ── Classical controller: end-of-episode debug dump (why did placement succeed/fail?) ──
    if controller is not None and _pp_ctx is not None:
        _fee = np.array(obs["robot0_eef_pos"], dtype=float)
        _fobj = np.array(obs.get(_pp_ctx["obj_key"], [np.nan, np.nan, np.nan]), dtype=float)
        _fg = np.asarray(_pp_ctx["goal_pos"], dtype=float)
        _obj0 = np.asarray(_pp_ctx["obj_pos"], dtype=float)
        print("  ── classical debug ──────────────────────────────────────")
        print(f"    final phase        : {controller.phase}")
        print(f"    grasp_offset (EE−obj_z at grasp): {getattr(controller, 'grasp_offset', None)}")
        print(f"    object start  pos  : {np.round(_obj0, 3)}")
        print(f"    object FINAL  pos  : {np.round(_fobj, 3)}   (moved {np.linalg.norm(_fobj - _obj0):.3f} m)")
        print(f"    goal (basket) pos  : {np.round(_fg, 3)}")
        print(f"    object→goal  dist  : {np.linalg.norm(_fobj - _fg):.3f} m   "
              f"(xy {np.linalg.norm(_fobj[:2] - _fg[:2]):.3f}, z {_fobj[2] - _fg[2]:+.3f})")
        print(f"    EE final      pos  : {np.round(_fee, 3)}")
        if obstacles:
            print(f"    obstacle      pos  : {np.round(np.asarray(obstacles[0].pos), 3)}  "
                  f"r_safe={getattr(obstacles[0], 'safety_radius', '?')}  "
                  f"EE→obs {np.linalg.norm(_fee - np.asarray(obstacles[0].pos)):.3f} m")
        print("  ─────────────────────────────────────────────────────────")

    # Attach each query's executed shield-corrected actions to its trace (Exp 005 BC target).
    if record_policy_trace and _shielded_bufs:
        for _q, _buf in zip(metrics.policy_trace, _shielded_bufs):
            if _buf:
                _q.shielded_actions = np.asarray(_buf, dtype=np.float32)

    if collect_cbf_data and cbf_dataset_path:
        metrics.save_hdf5(cbf_dataset_path, instruction=instruction)

    s = metrics.summary()
    # Attach attribution to the metrics object so run_task can persist it to CSV
    # ('|'-joined so it survives as a single CSV cell).
    metrics.collision_culprit = "|".join(sorted(_collision_culprits)) if _collision_flag else ""
    metrics.collision_robot_caused = bool(_collision_flag and _collision_robot_caused)
    metrics.obs_touched_by    = "|".join(sorted(_obs_touch_seen))
    _culprit_str = (f"  culprit={sorted(_collision_culprits)}"
                    if _collision_flag else "")
    _touch_str = (f"  touched_by={sorted(_obs_touch_seen)}"
                  if _obs_touch_seen else "")
    print(f"  Done — TSR={s['goal_reached']}  "
          f"collision={s['collision_detected']}  "
          f"CBF={s['cbf_activations']} acts  "
          f"violations={s['violation_steps']}{_culprit_str}{_touch_str}")
    return metrics
