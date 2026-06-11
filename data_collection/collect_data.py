import mujoco
import mujoco.viewer
import numpy as np
import time
import cv2
import random
import os
import json
from scipy.spatial.transform import Rotation as R
import h5py

# --- Differential IK Constants ---
DAMPING = 1e-4
K_POS = 3.0      # Increased for tighter tracking
K_ORI = 1.0
K_NULL = 1.0     # Reduced so it doesn't fight the main task
MAX_ANGVEL = 2.0
INTEGRATION_DT = 0.1 # Smaller step for more stability

IMG_W, IMG_H = 640, 480  # Required resolution

def solve_differential_ik(model, data, target_pos, target_rot_matrix, q0, ee_site_id):
    # 1. Get Current Site State (TCP)
    current_pos = data.site_xpos[ee_site_id]
    current_rot_matrix = data.site_xmat[ee_site_id].reshape(3, 3)

    # 2. Calculate Error (Twist)
    error_pos = target_pos - current_pos
    
    # Orientation error
    error_rot_mat = target_rot_matrix @ current_rot_matrix.T
    error_rot_vec = R.from_matrix(error_rot_mat).as_rotvec()

    twist = np.zeros(6)
    twist[:3] = error_pos * K_POS
    twist[3:] = error_rot_vec * K_ORI

    # 3. Compute Site Jacobian
    jac_p = np.zeros((3, model.nv))
    jac_r = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_p, jac_r, ee_site_id)
    J = np.vstack([jac_p, jac_r])[:, :7] 

    # 4. Solve for dq using Damped Least Squares
    vv = J @ J.T
    diag_indices = np.diag_indices_from(vv)
    vv[diag_indices] += DAMPING
    dq = J.T @ np.linalg.solve(vv, twist)

    # 5. Nullspace Control (Bias toward home posture q0)
    current_q = data.qpos[:7]
    null_error = (q0[:7] - current_q)
    J_pinv = np.linalg.pinv(J, rcond=1e-2)
    dq_null = (np.eye(7) - J_pinv @ J) @ (K_NULL * null_error)
    
    return np.clip(dq + dq_null, -MAX_ANGVEL, MAX_ANGVEL)


def is_task_complete(data, ee_site_id, target_pos, threshold=0.02):
    ee_pos = data.site_xpos[ee_site_id]
    return np.linalg.norm(ee_pos - target_pos) < threshold



# Convert rotation matrix to quaternion
def site_quat(model, data, site_id):
    xmat = data.site_xmat[site_id].reshape(9).astype(np.float64)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, xmat)
    return quat.astype(np.float32)   # [w, x, y, z]




def get_freejoint_qpos_adr(model, body_id):
    assert model.body_jntnum[body_id] == 1
    joint_id = model.body_jntadr[body_id]
    assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE
    return model.jnt_qposadr[joint_id]


# Randomly initialize the position of the red cube
def randomize_blocks(model, data, body_ids):
    x_min, x_max = 0.45, 0.75
    y_min, y_max = -0.25, 0.25
    z_table = 0.23
    z_lift = 1.2
    min_distance = 0.1  # minimum distance between block centers

    for body_id in body_ids:
        jadr = get_freejoint_qpos_adr(model, body_id)
        data.qpos[jadr + 2] = z_lift

    mujoco.mj_forward(model, data)

    # Place the first block
    jadr1 = get_freejoint_qpos_adr(model, body_ids[0])
    x1 = random.uniform(x_min, x_max)
    y1 = random.uniform(y_min, y_max)
    data.qpos[jadr1:jadr1 + 7] = [x1, y1, z_table, 1, 0, 0, 0]

    # Place the second block with a minimum distance from the first
    jadr2 = get_freejoint_qpos_adr(model, body_ids[1])
    while True:
        x2 = random.uniform(x_min, x_max)
        y2 = random.uniform(y_min, y_max)
        distance = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if distance >= min_distance:
            break
    data.qpos[jadr2:jadr2 + 7] = [x2, y2, z_table, 1, 0, 0, 0]

    for body_id in body_ids:
        jadr = get_freejoint_qpos_adr(model, body_id)
        data.qvel[jadr:jadr + 6] = 0

    mujoco.mj_forward(model, data)

# TODO have to verify that this is a good way for rotation representation
def matrix_to_6d(matrix):
    """Converts a 3x3 rotation matrix to 6D representation (top two rows)."""
    return matrix[:2, :].flatten() # [R11, R12, R13, R21,

class DataCollector:
    def __init__(self, filename, task_text):
        self.filename = filename
        self.task_text = task_text
        self.file = h5py.File(filename, 'w')
        self.episodes = self.file.create_group('data')
        self.ep_count = 0

    def save_episode(self, images, joints, ee_poses, gripper_actions):
        """
        Saves a full episode stream. 
        images: (T, H, W, 3)
        joints: (T, 8) -> 7 joints + gripper qpos
        ee_poses: (T, 9) -> 3 pos + 6D rotation
        gripper_actions: (T, 1) -> normalized [0, 1]
        """
        ep_grp = self.episodes.create_group(f'episode_{self.ep_count:04d}')
        ep_grp.create_dataset('image_static', data=np.array(images, dtype=np.uint8), compression="gzip")
        ep_grp.create_dataset('proprioception', data=np.array(joints, dtype=np.float32))
        ep_grp.create_dataset('ee_pose_abs', data=np.array(ee_poses, dtype=np.float32))
        ep_grp.create_dataset('gripper_action', data=np.array(gripper_actions, dtype=np.float32))
        ep_grp.attrs['text'] = self.task_text
        self.ep_count += 1
        print(f"Stored Episode {self.ep_count} with {len(images)} frames.")

    def close(self):
        self.file.close()



def main():
    NUM_EPISODES = 10

    model = mujoco.MjModel.from_xml_path("../simulation_assets/model/franka_emika_panda/scene.xml")
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, IMG_H, IMG_W) # 256, 256)

    collector = DataCollector("mimic_video_data.h5", "pick up the red cube")

    # gravity compensation
    model.body_gravcomp[:] = 1.0

    # Ready pose and fixed orientation (downwards)
    init_q = np.array([0, -0.785, 0, -2.356, 0, 1.571, 0.785, 0.04, 0.04])
    # In most MuJoCo Panda models, [pi, 0, 0] or [0, pi, 0] points the gripper down
    target_rot_matrix = R.from_euler('xyz', [np.pi, 0, 0]).as_matrix()

    ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "hand")
    red_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "red")
    green_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "green")
    cam_static = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    cam_ee = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_cam")

    # Define States
    APPROACH = 0  # Move 10cm above block
    DESCEND  = 1  # Move to the block with open gripper
    GRASP    = 2  # Close the gripper
    LIFT     = 3  # Move 15cm up

    print("\n========== VLA DATA COLLECTION ==========\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for ep in range(NUM_EPISODES):
            
            # Reset and Position Arm
            mujoco.mj_resetData(model, data)
            data.qpos[:7] = init_q[:7]
            data.ctrl[:7] = init_q[:7]

            # 2. FORCE GRIPPER OPEN AT START
            data.qpos[7:9] = [0.04, 0.04] 
            data.ctrl[7] = 255.0
            
            randomize_blocks(model, data, [red_body_id, green_body_id])
            mujoco.mj_forward(model, data)

            ep_images, ep_joints, ep_ee_poses, ep_gripper = [], [], [], []

            state = APPROACH
            step_id = 0
            ep_done = False
            lift_start = False
            
            while viewer.is_running() and not ep_done and step_id < 200:
                red_pos = data.xpos[red_body_id].copy()
                ee_pos = data.site_xpos[ee_site_id].copy()
                ee_rot = data.site_xmat[ee_site_id].reshape(3, 3).copy()
                
                # Default Gripper: Open
                target_grip = 255.0 

                if state == APPROACH:
                    target_pos = red_pos + np.array([0, 0, 0.10])
                    target_grip = 255.0  # OPEN
                    if np.linalg.norm(ee_pos - target_pos) < 0.04:
                        state = DESCEND

                elif state == DESCEND:
                    target_pos = red_pos + np.array([0, 0, 0.01])
                    target_grip = 255.0  # KEEP OPEN
                    if np.linalg.norm(ee_pos - target_pos) < 0.02:
                        state = GRASP
                        grasp_timer = 0 

                elif state == GRASP:
                    target_pos = red_pos + np.array([0, 0, 0.01])
                    target_grip = -255.0  # CLOSE
                    grasp_timer += 1
                    if grasp_timer > 30:
                        state = LIFT

                elif state == LIFT:
                    if not lift_start:
                        target_pos = red_pos + np.array([0, 0, 0.01])
                        lift_start = True
                    target_grip = 0.0  # Keep CLOSED
                    if np.linalg.norm(ee_pos - target_pos) < 0.04:
                        ep_done = True
                        print("Episode Success!")

                # --- 3. Solve IK & Apply Control ---
                dq = solve_differential_ik(model, data, target_pos, target_rot_matrix, init_q, ee_site_id)
                
                # Arm Joints
                data.ctrl[:7] = data.qpos[:7] + dq * INTEGRATION_DT
                # Gripper Actuators (Check your XML if it's index 7 or 7 and 8)
                data.ctrl[7] = target_grip 

                for _ in range(10):
                    mujoco.mj_step(model, data)

                # 6. Render and Save
                renderer.update_scene(data, camera=cam_static)
                img = renderer.render()
                #renderer.update_scene(data, camera=cam_ee)
                #img_ee = renderer.render()

                # Proprioception: 7 joints + 1 gripper width
                proprio = np.concatenate([data.qpos[:7], [data.qpos[7]]])
                
                # Absolute EE State: 3 pos + 6D rot
                ee_6d = matrix_to_6d(ee_rot)
                ee_state = np.concatenate([ee_pos, ee_6d])
                
                # Action: Gripper (Normalized 0-1)
                grip_norm = 1.0 if target_grip > 0 else 0.0

                ep_images.append(img)
                ep_joints.append(proprio)
                ep_ee_poses.append(ee_state)
                ep_gripper.append([grip_norm])

                # 7. Visualization
                #try:
                #    data.mocap_pos[0] = target_pos
                #except: pass
                
                viewer.sync()
                step_id += 1

            collector.save_episode(ep_images, ep_joints, ep_ee_poses, ep_gripper)
            print(f"Episode {ep} finished in {step_id} steps.")

            # Get the last image from the episode buffer
            debug_img = ep_images[-1] 

            # Convert RGB (MuJoCo) to BGR (OpenCV)
            debug_img_bgr = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)

            # Optional: Add text overlay to verify resolution
            cv2.putText(debug_img_bgr, f"Res: {debug_img.shape[1]}x{debug_img.shape[0]}", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            cv2.imshow("Data Collection Debug", debug_img_bgr)
            print("Press any key on the image window to continue to next episode...")
            cv2.waitKey(0) # Waits for a key press to continue

        collector.close()

if __name__ == "__main__":
    main()