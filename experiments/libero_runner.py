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
from PIL import Image
from scipy.optimize import minimize

try:
    import mujoco
    _HAS_MUJOCO = True
except ImportError:
    _HAS_MUJOCO = False

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

from experiments.scene_config import ObstacleConfig
from experiments.metrics import MetricsTracker, StepRecord

# ── Constants ──────────────────────────────────────────────────────────────────
OPENVLA_URL = "http://127.0.0.1:8000/act"

_ARM_BODY_NAMES = [
    "robot0_link3", "robot0_link4", "robot0_link5",
    "robot0_link6", "robot0_link7", "robot0_right_hand",
]
_ARM_JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]

# No-op action for OSC_POSE: zero Cartesian delta, gripper open.
_DUMMY_ACTION = np.array([0., 0., 0., 0., 0., 0., -1.], dtype=np.float64)

# Number of warm-up steps before querying VLA (lets physics settle after reset).
_WARMUP_STEPS = 10

# Damping for Jacobian pseudoinverse used in the Cartesian CBF.
_CBF_OSC_LAMBDA = 1e-3


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to 224x224 uint8 RGB and flip vertically.

    LIBERO fine-tuning data was loaded with a vertical flip applied (display
    convention, y=0 at top).  OpenVLA therefore expects flipped images at
    inference.  The raw agentview_image from robosuite is in OpenGL convention
    (y=0 at bottom), so we flip here for both VLA input and display.
    """
    img = img[::-1].copy()
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224))
    return img.astype(np.uint8)


def _to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _query_openvla_chunk(img_rgb: np.ndarray, instruction: str,
                         num_actions: int = 5) -> list[np.ndarray]:
    """Query the VLA server and return a chunk of `num_actions` raw 7-D actions.

    The server's /act endpoint returns {"action": ..., "actions": [...]}.
    Falls back to repeating "action" num_actions times if server is old.
    """
    r = requests.post(OPENVLA_URL,
                      json={"image_base64":  _to_b64(img_rgb),
                            "instruction":   instruction,
                            "num_actions":   num_actions},
                      timeout=120)
    r.raise_for_status()
    d = r.json()
    if not d.get("action"):
        raise RuntimeError(f"VLA server error: {d}")
    if d.get("actions") and len(d["actions"]) >= 1:
        return [np.array(a, dtype=np.float64) for a in d["actions"]]
    # Fallback: old server that only returns a single action
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


# ── Cartesian CBF ───────────────────────────────────────────────────────────────
def _compute_h_values_cartesian(data, arm_body_ids: list[int],
                                obstacles: list[ObstacleConfig]) -> list[float]:
    """Min h(q) per obstacle over all arm links (for monitoring/logging)."""
    h_per_obs = []
    for obs in obstacles:
        h_min = min(
            float(np.linalg.norm(data.xpos[bid] - obs.pos) ** 2 - obs.safety_radius ** 2)
            for bid in arm_body_ids
        )
        h_per_obs.append(h_min)
    return h_per_obs


def _run_cartesian_cbf(
    model, data,
    arm_body_ids: list[int],
    arm_dof_idx:  list[int],
    ee_body_id:   int,
    obstacles:    list[ObstacleConfig],
    u_nom_xyz:    np.ndarray,   # shape (3,): nominal EE position delta
    gamma:        float,
) -> tuple[np.ndarray, float, bool]:
    """CBF-QP in 3-D Cartesian EE-action space.

    For each (arm-link, obstacle) pair computes the Lie derivative of the
    barrier h_{ij}(q) = ||p_i - p_obs_j||^2 - r_j^2 along the EE action
    direction using the chain rule:

        dh/dt ~= 2*(p_i - p_obs) @ J_link_i @ J_ee^+ @ u_xyz

    CBF constraint: dh/dt + gamma*h >= 0
      => [2*(p_i-p_obs) @ J_link_i @ J_ee^+] @ u_xyz  >=  -gamma * h

    Solves a small 3-D QP:
        min  ||u - u_nom||^2
        s.t. A_k @ u >= b_k   for every (link, obstacle) pair k

    Returns (u_safe_xyz, correction_norm, triggered).
    """
    if not obstacles or not arm_body_ids:
        return u_nom_xyz.copy(), 0.0, False

    # EE positional Jacobian (3 x 7)
    jacp_ee = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp_ee, np.zeros((3, model.nv)), ee_body_id)
    J_ee = jacp_ee[:, arm_dof_idx]                                          # 3x7
    J_ee_pinv = J_ee.T @ np.linalg.inv(J_ee @ J_ee.T + _CBF_OSC_LAMBDA * np.eye(3))  # 7x3

    constraints = []
    for obs in obstacles:
        for bid in arm_body_ids:
            p_link = data.xpos[bid].copy()
            diff   = p_link - obs.pos
            h      = float(np.dot(diff, diff) - obs.safety_radius ** 2)

            jacp_l = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp_l, np.zeros((3, model.nv)), bid)
            J_link = jacp_l[:, arm_dof_idx]   # 3x7

            # Effective Jacobian: link velocity per unit EE action (3x3)
            M = J_link @ J_ee_pinv
            # Constraint row: A @ u_xyz >= -gamma*h
            A = 2.0 * diff @ M    # (3,)
            b = gamma * h

            constraints.append({
                "type": "ineq",
                "fun":  lambda u, A=A, b=b: float(A @ u) + b,
            })

    res = minimize(
        lambda u: 0.5 * float(np.dot(u - u_nom_xyz, u - u_nom_xyz)),
        x0=u_nom_xyz.copy(),
        method="SLSQP",
        constraints=constraints,
    )
    u_safe         = res.x if res.success else u_nom_xyz.copy()
    corr_norm      = float(np.linalg.norm(u_safe - u_nom_xyz))
    triggered      = corr_norm > 1e-4
    return u_safe, corr_norm, triggered


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


# ── Environment factories ──────────────────────────────────────────────────────
def make_libero_env(task_suite: str = "libero_spatial",
                    task_idx: int = 0,
                    safety_level: str = "I",
                    has_renderer: bool = False,
                    horizon: int = 800) -> tuple:
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
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_full,
        controller="OSC_POSE",          # matches OpenVLA fine-tuning setup
        camera_heights=224,
        camera_widths=224,
        camera_names=["agentview"],
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,
    )

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


def detect_safelibero_obstacle(env, obs: dict,
                                safety_radius: float = 0.10) -> ObstacleConfig | None:
    """Auto-detect the active SafeLIBERO obstacle from the environment.

    SafeLIBERO BDDL files place multiple obstacle objects in the scene but only
    one of them lands within the robot workspace per episode (the others are
    placed far off-table by the .pruned_init file).  This function identifies
    the active obstacle by scanning joint names for 'obstacle' and checking
    which object position falls within the workspace bounds.

    Returns an ObstacleConfig for the active obstacle, or None if not found.
    """
    try:
        joint_names = list(env.sim.model.joint_names)
    except Exception:
        return None

    obstacle_names = [n.replace("_joint0", "") for n in joint_names if "obstacle" in n]
    for name in obstacle_names:
        key = f"{name}_pos"
        if key not in obs:
            continue
        p = np.array(obs[key], dtype=float)
        if p[2] > 0.5 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
            return ObstacleConfig(
                pos=p.copy(),
                radius=0.06,
                safety_radius=safety_radius,
                name=name,
            )
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


def _render_frame(img_rgb: np.ndarray, t: int, horizon: int, mode: str,
                  min_dist: float, cbf_triggered: bool, collision_flag: bool,
                  episode_idx: int, vla_cnt: int) -> np.ndarray:
    """Return a BGR display frame: upscaled camera image + status bar.

    img_rgb has already been flipped by _preprocess (display convention).
    """
    s = _DISPLAY_SCALE
    h, w = img_rgb.shape[:2]
    frame = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    frame = cv2.resize(frame, (w * s, h * s), interpolation=cv2.INTER_NEAREST)

    bar = np.zeros((_STATUS_BAR_H, w * s, 3), dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX
    WHITE  = (220, 220, 220)
    GRAY   = (130, 130, 130)
    GREEN  = (60,  200, 60)
    YELLOW = (0,   200, 200)
    RED    = (60,  60,  220)

    cbf_col  = YELLOW if cbf_triggered else GRAY
    coll_col = RED    if collision_flag else GRAY

    cv2.putText(bar, f"step {t:03d}/{horizon}",          (8,  22), font, 0.52, GRAY,  1)
    cv2.putText(bar, f"[{mode.upper()}]",                (8,  50), font, 0.58, WHITE, 1)
    cv2.putText(bar, f"obs {min_dist:.3f}m",             (160, 22), font, 0.52, WHITE, 1)
    cv2.putText(bar, f"CBF {'ON' if cbf_triggered else 'off'}", (160, 50), font, 0.52, cbf_col, 1)
    cv2.putText(bar, f"collision {'YES' if collision_flag else 'no'}", (310, 22), font, 0.52, coll_col, 1)
    cv2.putText(bar, f"ep {episode_idx}  VLA #{vla_cnt}", (310, 50), font, 0.52, GRAY, 1)

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
    save_results: bool = False,
    results_dir: str = "results",
    dataset_path: str | None = None,
    show_viewer: bool = False,
    save_video: str | None = None,
    # SafeLIBERO episode parameters
    episode_idx: int = 0,
    initial_states=None,
    auto_detect_obstacle: bool = False,
    obstacle_safety_radius: float = 0.10,
    # Synchronous replan parameters (AEGIS approach)
    replan_steps: int = 5,
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

    mode = "cbf" if use_cbf else "plain"

    # ── Video writer ─────────────────────────────────────────────────────────
    _vwriter = None
    _frame_w = 224 * _DISPLAY_SCALE
    _frame_h = 224 * _DISPLAY_SCALE + _STATUS_BAR_H
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

    metrics = MetricsTracker(scene_name, mode)

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

    # Record initial obstacle position for displacement-based collision check.
    _obstacle_name: str | None = None
    _initial_obstacle_pos: np.ndarray | None = None
    if obstacles:
        # Use the first obstacle's name to track position from obs dict.
        _obstacle_name = obstacles[0].name
        _obs_key = f"{_obstacle_name}_pos"
        if _obs_key in obs:
            _initial_obstacle_pos = np.array(obs[_obs_key], dtype=float).copy()
    _collision_flag = False

    horizon      = getattr(env, "horizon", 800)
    action_queue: list[np.ndarray] = []   # pending post-processed actions from last chunk
    vla_cnt      = 0                       # total VLA calls made (for display)
    _current_action = _DUMMY_ACTION.copy()
    vla_raw      = np.zeros(7)

    try:
        for t in range(horizon):

            # ── 1. Get camera observation ─────────────────────────────────
            img_raw = obs.get("agentview_image") if isinstance(obs, dict) else None
            if img_raw is not None:
                img = _preprocess(img_raw)
            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            # ── 2. Synchronous replan: query VLA when queue is empty ──────
            if not action_queue:
                try:
                    raw_chunk = _query_openvla_chunk(img, instruction,
                                                     num_actions=replan_steps)
                    vla_raw   = raw_chunk[0].copy()
                    action_queue = [_post_process_vla(a) for a in raw_chunk]
                    vla_cnt  += 1
                    print(f"  [{t:03d}] VLA #{vla_cnt}  gripper={action_queue[0][6]:+.1f}")
                except Exception as e:
                    print(f"  [{t:03d}] VLA query error (holding last action): {e}")
                    action_queue = [_current_action.copy()]

            _current_action = action_queue.pop(0)

            # ── 3. CBF filter on xyz component ────────────────────────────
            ee_pos          = np.array(obs["robot0_eef_pos"], dtype=float)
            safe_action     = _current_action.copy()
            cbf_triggered   = False
            correction_norm = 0.0

            if use_cbf and obstacles and arm_body_ids:
                u_safe_xyz, correction_norm, cbf_triggered = _run_cartesian_cbf(
                    model, data, arm_body_ids, arm_dof_idx, ee_body_id,
                    obstacles, _current_action[:3], cbf_gamma,
                )
                safe_action[:3] = u_safe_xyz

            # ── 4. Safety monitoring ──────────────────────────────────────
            if obstacles and arm_body_ids:
                dists_to_obs = [
                    float(np.linalg.norm(data.xpos[bid] - ob.pos))
                    for ob in obstacles for bid in arm_body_ids
                ]
                min_d     = min(dists_to_obs)
                violation = any(
                    np.linalg.norm(data.xpos[bid] - ob.pos) < ob.safety_radius
                    for ob in obstacles for bid in arm_body_ids
                )
                h_values = _compute_h_values_cartesian(data, arm_body_ids, obstacles)
            else:
                min_d, violation = float("inf"), False
                h_values = [float("inf")]

            # ── 5. Display ────────────────────────────────────────────────
            if show_viewer or _vwriter is not None:
                frame_disp = _render_frame(
                    img, t, horizon, mode, min_d, cbf_triggered,
                    _collision_flag, episode_idx, vla_cnt,
                )
                if _vwriter is not None:
                    _vwriter.write(frame_disp)
                if show_viewer:
                    cv2.imshow(f"LIBERO — {scene_name} [{mode}]", frame_disp)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # ── 6. Step environment ───────────────────────────────────────
            step_out = env.step(safe_action.tolist())
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            # ── 7. Displacement-based collision check (SafeLIBERO metric) ─
            if (not _collision_flag
                    and _obstacle_name is not None
                    and _initial_obstacle_pos is not None):
                _obs_key = f"{_obstacle_name}_pos"
                if _obs_key in obs:
                    curr_obs_pos = np.array(obs[_obs_key], dtype=float)
                    if np.sum(np.abs(curr_obs_pos - _initial_obstacle_pos)) > 0.001:
                        _collision_flag = True
                        print(f"  [{t:03d}] COLLISION: obstacle displaced "
                              f"{np.sum(np.abs(curr_obs_pos - _initial_obstacle_pos)):.4f}m")

            # ── 8. Success check ──────────────────────────────────────────
            success_flag = bool(info.get("success", False))
            if success_flag:
                metrics.mark_goal_reached(t)
            elif goal_pos is not None:
                success_flag = float(np.linalg.norm(ee_pos - goal_pos)) < goal_tolerance

            # ── 9. Record ─────────────────────────────────────────────────
            q_current = np.array(obs.get("robot0_joint_pos", np.zeros(7)), dtype=float)
            _gp = goal_pos if goal_pos is not None else np.zeros(3)
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
                    image=img.copy() if collect_dataset else None,
                ),
                goal_pos=_gp,
                goal_tolerance=goal_tolerance,
            )

            if t % 20 == 0:
                flags = []
                if _collision_flag: flags.append("COLLISION")
                if violation:       flags.append("VIOLATION")
                if cbf_triggered:   flags.append("[CBF]")
                if success_flag:    flags.append("SUCCESS")
                d_goal = (f"{np.linalg.norm(ee_pos - goal_pos):.3f}m"
                          if goal_pos is not None else "n/a")
                print(f"  [{t:03d}] d_goal={d_goal}  "
                      f"min_obs={min_d:.3f}m  {'  '.join(flags)}")

            if done or success_flag:
                break

    finally:
        if _vwriter is not None:
            _vwriter.release()
        if show_viewer:
            cv2.destroyAllWindows()

    # ── Save ──────────────────────────────────────────────────────────────────
    if save_results:
        import os
        os.makedirs(results_dir, exist_ok=True)
        label = f"{scene_name}_{mode}"
        metrics.save_step_log(f"{results_dir}/{label}_steps.csv")
        metrics.save_summary( f"{results_dir}/{label}_summary.csv")

    if collect_dataset and dataset_path:
        metrics.save_dataset(dataset_path)

    s = metrics.summary()
    print(f"  Done — TSR={s['goal_reached']}  "
          f"collision={s['collision_detected']}  "
          f"CBF={s['cbf_activations']} acts  "
          f"violations={s['violation_steps']}")
    return metrics
