"""
LIBERO/robosuite runner — replaces xml_builder + runner for LIBERO scenes.

What stays identical from the original pipeline
------------------------------------------------
  _run_cbf()            CBF-QP joint-velocity filter (imported from runner.py)
  _compute_h_values()   barrier value computation
  _potential_repulsion  ghost-target repulsion
  MetricsTracker        per-step logging, save_dataset()
  VLA HTTP thread       async OpenVLA queries

What changes
------------
  Physics loop  : env.step(action) instead of mj_step + qpos override
  Model/data    : env.sim.model / env.sim.data  (robosuite wraps native mujoco)
  Camera image  : obs["agentview_image"]  (no separate Renderer needed)
  EE position   : obs["robot0_eef_pos"]   (no site lookup needed)
  Success check : info["success"]          (LIBERO / robosuite built-in)
  Obstacles     : ObstacleConfig list, positions in world frame,
                  used ONLY for CBF — not injected into physics

Controller choice: JOINT_POSITION with control_delta=True mirrors our current
  data.qpos[:7] += step_scale * dq_safe  approach exactly.

Usage
-----
  from experiments.libero_runner import make_libero_env, run_libero_trial, list_tasks

  env, lang = make_libero_env("libero_spatial", task_idx=0)
  metrics   = run_libero_trial(env, obstacles=[], instruction=lang, ...)
  env.close()
"""

from __future__ import annotations

import time
import threading
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
    from robosuite.controllers import load_controller_config
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

# Franka arm body names in robosuite's Panda model.
# robosuite prefixes all robot bodies with "robot0_".
# Fallback search also checks "link3" … "link7" substring.
_ARM_BODY_NAMES = [
    "robot0_link3", "robot0_link4", "robot0_link5",
    "robot0_link6", "robot0_link7", "robot0_right_hand",
]

# Joint names used to extract arm DOF indices from the full system Jacobian.
# Critical: jacp has shape (3, model.nv). We slice only the 7 arm DOFs,
# NOT [:, :7], because robosuite's DOF ordering is not guaranteed to start
# with arm joints (gripper and object DOFs may be interleaved).
_ARM_JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 8)]

# JOINT_POSITION controller — max delta per step (rad).
# action = clip(step_scale * dq / _MAX_DQ, -1, 1) → controller maps ±1 → ±_MAX_DQ
_MAX_DQ    = 0.20
_STEP_SCALE = 0.40    # mirrors SceneConfig.cbf_step_scale

_GRASP_DIST   = 0.18  # m — EE to object to trigger grasp
_RELEASE_DIST = 0.10  # m — EE to goal to trigger release


# ── VLA async worker ───────────────────────────────────────────────────────────
_vla_lock       = threading.Lock()
_vla_image      = None
_delta_ema      = np.zeros(3)
_vla_action_raw = np.zeros(7)
_vla_running    = False
_vla_instruction = "pick up the object and place it at the goal location."


def _preprocess(img: np.ndarray) -> np.ndarray:
    """Resize to 224×224 uint8 RGB. robosuite images are already RGB but
    returned upside-down (OpenGL convention); flip vertically."""
    img = img[::-1].copy()            # flip vertically
    if img.shape[:2] != (224, 224):
        img = cv2.resize(img, (224, 224))
    return img.astype(np.uint8)


def _to_b64(img_rgb: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(img_rgb).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def _query_openvla(img_rgb: np.ndarray, instruction: str) -> np.ndarray:
    r = requests.post(OPENVLA_URL,
                      json={"image_base64": _to_b64(img_rgb),
                            "instruction":  instruction},
                      timeout=60)
    r.raise_for_status()
    d = r.json()
    if not d.get("action"):
        raise RuntimeError(f"VLA server error: {d}")
    return np.array(d["action"], dtype=np.float32)


def _vla_worker(ema_alpha: float):
    global _delta_ema, _vla_action_raw, _vla_running
    while _vla_running:
        with _vla_lock:
            img  = _vla_image
            inst = _vla_instruction
        if img is not None:
            try:
                action = _query_openvla(img, inst)
                delta  = np.clip(action[:3], -0.05, 0.05)
                with _vla_lock:
                    _delta_ema[:]      = ema_alpha * _delta_ema + (1 - ema_alpha) * delta
                    _vla_action_raw[:] = action[:7]
            except Exception:
                pass
        time.sleep(0.01)


# ── CBF helpers ────────────────────────────────────────────────────────────────
def _compute_h_values(model, data, arm_body_ids: list[int],
                      obstacles: list[ObstacleConfig]) -> list[float]:
    if not obstacles:
        return []
    h_per_obs = []
    for obs in obstacles:
        min_h = min(
            float(np.linalg.norm(data.xpos[bid] - obs.pos) ** 2
                  - obs.safety_radius ** 2)
            for bid in arm_body_ids
        )
        h_per_obs.append(min_h)
    return h_per_obs


def _run_cbf(model, data, arm_body_ids: list[int],
             arm_dof_indices: list[int],
             obstacles: list[ObstacleConfig],
             q_dot_nom: np.ndarray, gamma: float):
    """CBF-QP filter.  Returns (u_safe, u_nom, h_per_obs, correction_norm, triggered)."""
    num_joints = len(arm_dof_indices)
    u_nom = np.asarray(q_dot_nom, dtype=float)

    if not obstacles:
        return u_nom.copy(), u_nom, [], 0.0, False

    constraints = []
    h_dict = {obs.name: float("inf") for obs in obstacles}

    for obs in obstacles:
        for bid in arm_body_ids:
            p_link = data.xpos[bid].copy()
            dist   = np.linalg.norm(p_link - obs.pos)
            h      = dist ** 2 - obs.safety_radius ** 2
            h_dict[obs.name] = min(h_dict[obs.name], h)

            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            mujoco.mj_jacBody(model, data, jacp, jacr, bid)
            J_link = jacp[:, arm_dof_indices]      # only arm DOFs

            def _con(u, p_l=p_link, J_l=J_link, h_val=h):
                diff = p_l - obs.pos
                A_i  = -2 * diff.dot(J_l)
                b_i  = gamma * h_val * 0.4
                return b_i - A_i.dot(u)

            constraints.append({"type": "ineq", "fun": _con})

    res = minimize(
        lambda u: 0.5 * np.sum((u - u_nom) ** 2),
        x0=u_nom, method="SLSQP", constraints=constraints,
    )
    u_safe          = res.x if res.success else np.zeros(num_joints)
    correction_norm = float(np.linalg.norm(u_safe - u_nom))
    cbf_triggered   = correction_norm > 1e-4
    h_per_obs       = [h_dict[obs.name] for obs in obstacles]
    return u_safe, u_nom, h_per_obs, correction_norm, cbf_triggered


def _potential_repulsion(ghost_pos, obstacles, gain, cutoff):
    force = np.zeros(3)
    for obs in obstacles:
        diff = ghost_pos - obs.pos
        dist = np.linalg.norm(diff)
        if 0 < dist < cutoff:
            mag = gain * (1.0 / dist - 1.0 / cutoff) / dist ** 2
            force += mag * (diff / dist)
    return force


# ── Model introspection helpers ────────────────────────────────────────────────
def _get_arm_body_ids(model) -> list[int]:
    """Find Panda arm body IDs in robosuite's MjModel."""
    ids = []
    for name in _ARM_BODY_NAMES:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid >= 0:
            ids.append(bid)
    if not ids:
        # Fallback: substring match
        for i in range(model.nbody):
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or ""
            if any(k in bname for k in ["link3","link4","link5","link6","link7","hand"]):
                ids.append(i)
    return ids


def _get_arm_dof_indices(model) -> list[int]:
    """Return column indices in the full Jacobian (model.nv wide) for the 7 arm joints."""
    dof_indices = []
    for jname in _ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid >= 0:
            dof_indices.append(int(model.jnt_dofadr[jid]))
    if not dof_indices:
        # Last resort: assume first 7 DOFs are arm joints
        dof_indices = list(range(7))
    return dof_indices


def _unwrap_sim(env):
    """Return (MjModel, MjData) from a robosuite environment.

    robosuite 1.4.x exposes env.sim.model and env.sim.data as native mujoco objects.
    Some builds wrap them in a proxy; this unwraps both cases.
    """
    sim   = env.sim
    model = getattr(sim, "model", None)
    data  = getattr(sim, "data",  None)
    # Unwrap proxy wrappers (some robosuite versions)
    if model is not None and hasattr(model, "_model"):
        model = model._model
    if data is not None and hasattr(data, "_data"):
        data = data._data
    if model is None or data is None:
        raise RuntimeError(
            "Cannot extract MjModel/MjData from env.sim. "
            "Try env.sim._model / env.sim._data for your robosuite version.")
    return model, data


# ── Environment factories ──────────────────────────────────────────────────────
def _base_controller_cfg() -> dict:
    """JOINT_POSITION controller with incremental (delta) mode."""
    cfg = load_controller_config(default_controller="JOINT_POSITION")
    cfg["control_delta"] = True
    cfg["input_max"]     = 1.0
    cfg["input_min"]     = -1.0
    cfg["output_max"]    = _MAX_DQ    # action=1 → delta_q=0.20 rad
    cfg["output_min"]    = -_MAX_DQ
    return cfg


def make_robosuite_env(task: str = "Lift",
                       has_renderer: bool = False,
                       horizon: int = 400) -> "suite.environments.robot.RobotEnv":
    """Create a plain robosuite Panda environment (no LIBERO required).

    Good for testing the integration before loading LIBERO tasks.
    Tasks: "Lift", "PickPlace", "Stack", "NutAssembly", "TwoArmLift", ...
    """
    if not _HAS_ROBOSUITE:
        raise RuntimeError("pip install robosuite")
    return suite.make(
        env_name=task,
        robots="Panda",
        controller_configs=_base_controller_cfg(),
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["agentview"],
        camera_heights=224,
        camera_widths=224,
        control_freq=20,
        horizon=horizon,
        reward_shaping=False,
        ignore_done=False,
    )


def make_libero_env(task_suite: str = "libero_spatial",
                    task_idx: int = 0,
                    has_renderer: bool = False,
                    horizon: int = 400) -> tuple:
    """Create a LIBERO environment.

    Returns:
        (env, language_instruction)  — pass the instruction to run_libero_trial().

    Task suites:
        libero_spatial   10 tasks, same objects different placements  ← start here
        libero_object    10 tasks, different object types
        libero_goal      10 tasks, different placement goals
        libero_long      10 long-horizon multi-step tasks
        libero_100       100 diverse tasks (takes longer to load)
    """
    if not _HAS_LIBERO:
        raise RuntimeError(
            "LIBERO not installed.\n"
            "  conda create -n libero python=3.10\n"
            "  conda activate libero\n"
            "  pip install -r requirements_libero.txt\n"
            "  git clone https://github.com/Lifelong-Robot-Learning/LIBERO\n"
            "  cd LIBERO && pip install -e .")

    import os
    from libero.libero import get_libero_path

    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite]()
    task           = task_suite_obj.get_task(task_idx)
    language       = task.language

    # task.bddl_file is just the filename; env_wrapper needs the full path
    bddl_root = get_libero_path("bddl_files")
    bddl_full = os.path.join(bddl_root, task.problem_folder, task.bddl_file)

    # LIBERO provides OffScreenRenderEnv / RobosuiteEnv wrappers
    from libero.libero.envs import OffScreenRenderEnv
    env = OffScreenRenderEnv(
        bddl_file_name=bddl_full,
        controller="JOINT_POSITION",   # ControlEnv builds controller_configs internally
        camera_heights=224,
        camera_widths=224,
        camera_names=["agentview"],
        has_renderer=has_renderer,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        control_freq=20,
        horizon=horizon,
        ignore_done=True,   # LIBERO overrides done with _check_success(); this prevents
                            # robosuite from setting self.done=True on horizon exceeded,
                            # which would crash the next step() call.
    )
    # Patch output scale on the live controller so action=±1 → ±_MAX_DQ rad
    for robot in env.env.robots:
        ctrl = robot.controller
        ctrl.output_max[:] = _MAX_DQ
        ctrl.output_min[:] = -_MAX_DQ
    print(f"  LIBERO task [{task_idx}]: \"{language}\"")
    return env, language


def list_tasks(suite_name: str = "libero_spatial") -> list[tuple[int, str]]:
    """Print and return (index, language) for all tasks in a LIBERO suite."""
    if not _HAS_LIBERO:
        print("LIBERO not installed — see requirements_libero.txt")
        return []
    benchmark_dict = _libero_benchmark.get_benchmark_dict()
    suite          = benchmark_dict[suite_name]()
    tasks = [(i, suite.get_task(i).language) for i in range(suite.get_num_tasks())]
    print(f"\n  Tasks in {suite_name}:")
    for i, lang in tasks:
        print(f"    [{i:2d}] {lang}")
    return tasks


# ── Obstacle helpers ───────────────────────────────────────────────────────────
def obs_from_libero(env_obs: dict, object_keys: list[str],
                    safety_radius: float = 0.10) -> list[ObstacleConfig]:
    """Build ObstacleConfig list from LIBERO observation dict.

    Pass the keys of objects you want treated as obstacles.
    Example: obs_from_libero(obs, ["akita_black_bowl_1_pos", "wooden_tray_1_pos"])
    """
    obstacles = []
    for key in object_keys:
        if key in env_obs:
            pos = np.array(env_obs[key][:3], dtype=float)
            obstacles.append(ObstacleConfig(
                pos=pos,
                radius=0.04,
                safety_radius=safety_radius,
                name=key.replace("_pos", ""),
            ))
        else:
            print(f"  [obs_from_libero] key '{key}' not in obs — skipping")
    return obstacles


# ── Main trial loop ────────────────────────────────────────────────────────────
def run_libero_trial(
    env,
    obstacles: list[ObstacleConfig],
    instruction: str,
    goal_pos: np.ndarray,
    use_cbf: bool = True,
    cbf_gamma: float = 1.8,
    goal_attract: float = 0.01,
    vla_scale: float = 1.0,
    repulsion_gain: float = 0.012,
    repulsion_cutoff: float = 0.25,
    ema_alpha: float = 0.30,
    goal_tolerance: float = 0.08,
    scene_name: str = "libero",
    collect_dataset: bool = False,
    save_results: bool = False,
    results_dir: str = "results",
    dataset_path: str | None = None,
    target_obj_key: str | None = None,
    show_viewer: bool = False,
    save_video: str | None = None,
) -> MetricsTracker:
    """Run one episode in a robosuite/LIBERO environment with optional CBF filter.

    Args:
        env:         Environment created by make_libero_env() or make_robosuite_env().
                     Call env.reset() is done inside this function.
        obstacles:   CBF obstacle zones (world-frame positions + radii).
                     These are PURELY for the CBF filter — not physics objects.
                     Pass [] to run without safety constraints.
        instruction: Language instruction string passed to OpenVLA.
        goal_pos:    Goal position in world frame [x, y, z].
                     Used for the ghost-target attractor AND TSR fallback check.
        use_cbf:     Apply CBF-QP filter to joint velocities.
        ...          (tuning params match SceneConfig; see field docstrings there)
        collect_dataset: Save camera images in StepRecord (for .npz dataset export).
        save_results: Write CSV step log and summary to results_dir/.
        dataset_path: If set and collect_dataset=True, saves .npz here.

    Returns:
        Populated MetricsTracker (call .summary() for CAR/TSR/ETS values).
    """
    global _vla_image, _delta_ema, _vla_action_raw, _vla_running, _vla_instruction

    if not _HAS_MUJOCO:
        raise RuntimeError("mujoco not available in this environment")

    mode             = "cbf" if use_cbf else "plain"
    _vla_instruction = instruction

    # ── Video writer setup ────────────────────────────────────────────────────
    _vwriter = None
    if save_video:
        import os
        os.makedirs(os.path.dirname(save_video) or ".", exist_ok=True)
        _vwriter = cv2.VideoWriter(
            save_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            20,                          # fps ≈ control_freq
            (224, 224),
        )

    # ── Reset & initial observation ──────────────────────────────────────────
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    # ── Extract native MuJoCo model/data ─────────────────────────────────────
    model, data = _unwrap_sim(env)
    arm_body_ids  = _get_arm_body_ids(model)
    arm_dof_idx   = _get_arm_dof_indices(model)
    print(f"  Arm bodies: {len(arm_body_ids)}  Arm DOFs: {arm_dof_idx}")

    # ── Initialise ghost target at current EE position ───────────────────────
    ee_pos    = np.array(obs["robot0_eef_pos"], dtype=float)
    ghost_pos = ee_pos.copy()

    # ── Virtual grasp state ───────────────────────────────────────────────────
    # Try to read target object position from obs for virtual-grasp tracking.
    # Prefer the explicitly named key; fall back to first _pos key found.
    def _get_object_pos(ob: dict) -> np.ndarray | None:
        if target_obj_key and target_obj_key in ob:
            return np.array(ob[target_obj_key][:3], dtype=float)
        for key in ob:
            if key.endswith("_pos") and key not in ("robot0_eef_pos",):
                return np.array(ob[key][:3], dtype=float)
        return None

    object_pos    = _get_object_pos(obs)
    block_grasped = False

    metrics = MetricsTracker(scene_name, mode)

    # ── Start VLA thread ──────────────────────────────────────────────────────
    _delta_ema[:]      = 0.0
    _vla_action_raw[:] = 0.0
    _vla_image         = None
    _vla_running       = True
    vla_thread = threading.Thread(
        target=_vla_worker, args=(ema_alpha,), daemon=True
    )
    vla_thread.start()

    print(f"\n  [{scene_name}] mode={mode.upper()}  "
          f"obstacles={len(obstacles)}  goal={np.round(goal_pos, 3)}")

    horizon = getattr(env, "horizon", 400)

    try:
        for t in range(horizon):

            # ── 1. Camera → VLA worker ────────────────────────────────────
            img_raw = obs.get("agentview_image")
            if img_raw is not None:
                img = _preprocess(img_raw)
                with _vla_lock:
                    _vla_image = img
            else:
                img = np.zeros((224, 224, 3), dtype=np.uint8)

            # Live viewer and/or video recording
            if show_viewer or _vwriter is not None:
                # Annotate frame with step / goal dist / CBF status
                frame_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                cv2.putText(frame_bgr, f"t={t}", (4, 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
                if _vwriter is not None:
                    _vwriter.write(frame_bgr)
                if show_viewer:
                    cv2.imshow(f"LIBERO — {scene_name} [{mode}]", frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # ── 2. Read latest VLA output ─────────────────────────────────
            with _vla_lock:
                d_ema     = _delta_ema.copy()
                vla_delta = _vla_action_raw.copy()

            # ── 3. Ghost target update ────────────────────────────────────
            ee_pos = np.array(obs["robot0_eef_pos"], dtype=float)
            repulsion = _potential_repulsion(ee_pos, obstacles,
                                            repulsion_gain, repulsion_cutoff)
            # Ghost is anchored to the current EE each step (velocity setpoint
            # semantics): ghost = EE + VLA_direction + soft_goal_pull + repulsion.
            # Anchoring prevents the 20Hz accumulation that caused runaway drift
            # when VLA only fires at ~1Hz.
            if np.linalg.norm(d_ema) > 1e-6:
                ghost_pos = ee_pos + vla_scale * d_ema + goal_attract * (goal_pos - ee_pos)
            else:
                # VLA not yet running (cold-start): pull ghost toward goal
                ghost_pos = ee_pos + goal_attract * (goal_pos - ee_pos)
            ghost_pos = ghost_pos + repulsion

            # ── 4. Nominal IK toward ghost ────────────────────────────────
            if arm_body_ids and arm_dof_idx:
                ee_bid = arm_body_ids[-1]    # hand / right_hand = last body
                jacp   = np.zeros((3, model.nv))
                jacr   = np.zeros((3, model.nv))
                mujoco.mj_jacBody(model, data, jacp, jacr, ee_bid)
                J          = jacp[:, arm_dof_idx]      # (3, 7) arm-only Jacobian
                err        = ghost_pos - ee_pos
                dq_nominal = J.T @ np.linalg.inv(J @ J.T + 1e-3 * np.eye(3)) @ err
                dq_nominal = np.clip(dq_nominal, -0.2, 0.2)
            else:
                dq_nominal = np.zeros(7)

            q_current = np.array(obs.get("robot0_joint_pos", np.zeros(7)), dtype=float)

            # ── 5. CBF filter ─────────────────────────────────────────────
            cbf_triggered   = False
            correction_norm = 0.0
            if use_cbf and obstacles:
                dq_safe, u_nom, h_values, correction_norm, cbf_triggered = _run_cbf(
                    model, data, arm_body_ids, arm_dof_idx,
                    obstacles, dq_nominal, cbf_gamma,
                )
            else:
                dq_safe  = dq_nominal.copy()
                u_nom    = dq_nominal.copy()
                h_values = _compute_h_values(model, data, arm_body_ids, obstacles)

            # ── 6. Gripper (rule-based virtual grasp) ─────────────────────
            if object_pos is not None:
                # Update object pos from obs each step (LIBERO moves real objects)
                obj_pos_now = _get_object_pos(obs) if _get_object_pos(obs) is not None else object_pos
                dist_obj  = float(np.linalg.norm(ee_pos - obj_pos_now))
                dist_goal = float(np.linalg.norm(ee_pos - goal_pos))
                if not block_grasped and dist_obj < _GRASP_DIST:
                    block_grasped = True
                    print(f"  [Grasp] GRASPED  step={t}  dist={dist_obj:.3f} m")
                elif block_grasped and dist_goal < _RELEASE_DIST:
                    block_grasped = False
                    print(f"  [Grasp] PLACED   step={t}")
            gripper = np.array([1.0 if block_grasped else -1.0])

            # ── 7. Step environment ───────────────────────────────────────
            # Normalise dq to [-1, 1]; controller maps ±1 → ±_MAX_DQ rad
            action_arm = np.clip(_STEP_SCALE * dq_safe / _MAX_DQ, -1.0, 1.0)
            action     = np.concatenate([action_arm, gripper])
            step_out   = env.step(action)
            if len(step_out) == 4:
                obs, reward, done, info = step_out
            else:
                obs, reward, terminated, truncated, info = step_out
                done = terminated or truncated

            # ── 8. Safety monitoring ──────────────────────────────────────
            if obstacles and arm_body_ids:
                min_d = min(
                    float(np.linalg.norm(data.xpos[bid] - ob.pos))
                    for ob in obstacles for bid in arm_body_ids
                )
                violation = any(
                    np.linalg.norm(data.xpos[bid] - ob.pos) < ob.safety_radius
                    for ob in obstacles for bid in arm_body_ids
                )
            else:
                min_d, violation = float("inf"), False

            # Success: prefer env's own checker, fall back to distance
            success_flag = bool(info.get("success", False))
            if not success_flag:
                success_flag = float(np.linalg.norm(ee_pos - goal_pos)) < goal_tolerance

            # ── 9. Record ─────────────────────────────────────────────────
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
                    q=q_current,
                    u_nom=u_nom,
                    u_safe=dq_safe.copy(),
                    h_values=h_values if h_values else [float("inf")],
                    vla_delta=vla_delta,
                    ghost_pos=ghost_pos.copy(),
                    image=img.copy() if collect_dataset else None,
                ),
                goal_pos=goal_pos,
                goal_tolerance=goal_tolerance,
            )

            if t % 20 == 0:
                flags = []
                if violation:     flags.append("VIOLATION")
                if cbf_triggered: flags.append("[CBF]")
                print(f"  [{t:03d}] d_goal={np.linalg.norm(ee_pos-goal_pos):.3f}m  "
                      f"min_obs={min_d:.3f}m  {'  '.join(flags)}")

            if done:
                break

    finally:
        _vla_running = False
        vla_thread.join(timeout=2.0)
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
          f"CBF={s['cbf_activations']} acts  "
          f"violations={s['violation_steps']}")
    return metrics
