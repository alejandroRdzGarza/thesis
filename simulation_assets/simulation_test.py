import mujoco
import mujoco.viewer
import numpy as np
import cv2


def main():
    model = mujoco.MjModel.from_xml_path('./simulation_assets/model/franka_emika_panda/scene.xml')
    data = mujoco.MjData(model)

    # Set the initial position
    data.ctrl[:8] = [0, -0.58, 0, -1.68, 0, 1.13, 0.8, 0]

    # Set the camera window size
    width, height = 400, 400
    renderer_static = mujoco.Renderer(model, width, height)
    renderer_ee = mujoco.Renderer(model, width, height)

    static_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam")
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "ee_cam")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():

            mujoco.mj_step(model, data)

            # Update the static camera
            renderer_static.update_scene(data, camera=static_id)
            rgb_static = renderer_static.render()
            print(rgb_static)
            bgr_static = cv2.cvtColor(rgb_static, cv2.COLOR_RGB2BGR)
            cv2.imshow("Static Camera", bgr_static)

            # Update the end-effecotr camera
            renderer_ee.update_scene(data, camera=ee_id)
            rgb_ee = renderer_ee.render()
            bgr_ee = cv2.cvtColor(rgb_ee, cv2.COLOR_RGB2BGR)
            cv2.imshow("End Effector Camera", bgr_ee)

            cv2.waitKey(1)
            viewer.sync() 

if __name__ == '__main__':
    main()
