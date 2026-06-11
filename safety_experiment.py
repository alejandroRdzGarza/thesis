import mujoco
import mujoco.viewer
import numpy as np
import time

# Safety experiment configuration
MAX_STEPS = 500
SAFETY_DISTANCE = 0.1  # Minimum distance from obstacles
CBF_GAMMA = 1.0       # CBF parameter

def cbf_constraint(model, data, target_pos, current_pos, obstacle_pos):
    """
    Control Barrier Function for obstacle avoidance
    Returns modified target position that respects safety constraints
    """
    # Calculate distance to obstacle
    dist_to_obstacle = np.linalg.norm(current_pos - obstacle_pos)

    if dist_to_obstacle < SAFETY_DISTANCE:
        # Compute safe direction (away from obstacle)
        safe_dir = (current_pos - obstacle_pos) / (dist_to_obstacle + 1e-6)
        # Modify target to stay safe
        safe_target = current_pos + safe_dir * (SAFETY_DISTANCE - dist_to_obstacle)
        # Blend with original target
        alpha = min(1.0, (SAFETY_DISTANCE - dist_to_obstacle) / SAFETY_DISTANCE)
        target_pos = (1 - alpha) * target_pos + alpha * safe_target

    return target_pos

def ik_step_with_safety(model, data, target_pos, ee_site_id, obstacle_pos=None):
    """
    Inverse kinematics with safety constraints
    """
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    ee_pos = data.site_xpos[ee_site_id]

    # Apply CBF if obstacle present
    if obstacle_pos is not None:
        target_pos = cbf_constraint(model, data, target_pos, ee_pos, obstacle_pos)

    err = target_pos - ee_pos
    J = jacp[:, :7]   # Panda arm
    JT = J.T

    damping = 1e-3
    dq = JT @ np.linalg.inv(J @ JT + damping * np.eye(3)) @ err
    dq = np.clip(dq, -0.2, 0.2)

    data.qpos[:7] += 0.4 * dq
    mujoco.mj_forward(model, data)

    return np.linalg.norm(err)

def main():
    print("\n=== Safety Experiment ===")

    model = mujoco.MjModel.from_xml_path(
        "./simulation_assets/model/franka_emika_panda/scene.xml"
    )
    data = mujoco.MjData(model)

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")

    # Define obstacle position (you can modify this)
    obstacle_pos = np.array([0.3, 0.0, 0.5])

    mujoco.mj_forward(model, data)

    # Simple trajectory: move in a circle while avoiding obstacle
    center = np.array([0.4, 0.0, 0.6])
    radius = 0.2

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for t in range(MAX_STEPS):
            # Generate target position (circular trajectory)
            angle = t * 0.02
            target_pos = center + np.array([radius * np.cos(angle), radius * np.sin(angle), 0])

            # Apply safety constraints
            err = ik_step_with_safety(model, data, target_pos, ee_site_id, obstacle_pos)

            if t % 20 == 0:
                ee_pos = data.site_xpos[ee_site_id]
                dist_to_obs = np.linalg.norm(ee_pos - obstacle_pos)
                print(f"[{t:03d}] target={np.round(target_pos,3)} ee={np.round(ee_pos,3)} dist_to_obs={dist_to_obs:.3f}")

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.02)

        print("\n[DONE]")

        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()</content>
<parameter name="filePath">/Users/alexrdzgarza/Developer/ucl_masters/Thesis/VLA_Benchmark-karl/safety_experiment.py