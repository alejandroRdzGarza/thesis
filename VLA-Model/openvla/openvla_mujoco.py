import mujoco
import mujoco.viewer
import numpy as np
import cv2
import time
import requests
import base64
from PIL import Image
import io

# config
OPENVLA_URL = "http://127.0.0.1:8000/act"
TEXT_GOAL = "grip the red cube"

VLA_SCALE = 5       # meters per step
EMA_ALPHA = 0.85        # smoothing
INFER_EVERY = 10         # reduce VLA calls
MAX_STEPS = 300


# Image preprocess
def preprocess_rgb(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    return img.astype(np.uint8)

def image_to_base64(img_rgb):
    pil_img = Image.fromarray(img_rgb)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

# HTTP
def query_openvla(image_rgb, instruction):
    payload = {
        "image_base64": image_to_base64(image_rgb),
        "instruction": instruction,
    }
    r = requests.post(OPENVLA_URL, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()

    if "action" not in data:
        raise RuntimeError(f"OpenVLA error: {data}")

    return np.array(data["action"], dtype=np.float32)


# Inverse kinematic
def ik_step(model, data, target_pos, ee_site_id,
            damping=1e-3, step_scale=0.4):

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, ee_site_id)

    ee_pos = data.site_xpos[ee_site_id]
    err = target_pos - ee_pos

    J = jacp[:, :7]   # Panda arm
    JT = J.T

    dq = JT @ np.linalg.inv(J @ JT + damping * np.eye(3)) @ err
    dq = np.clip(dq, -0.2, 0.2)

    data.qpos[:7] += step_scale * dq
    mujoco.mj_forward(model, data)

    return np.linalg.norm(err)


def main():

    print("\n=== OpenVLA Mujoco ===")

    # Change your simulation assets path here
    model = mujoco.MjModel.from_xml_path(
        "./simulation_assets/model/franka_emika_panda/scene.xml"
    )
    data = mujoco.MjData(model)

    renderer = mujoco.Renderer(model, 400, 400)

    cam_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "static_cam"
    )
    ee_site_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, "hand"
    )

    target_body_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "target_ghost"
    )
    mocap_id = model.body_mocapid[target_body_id]

    mujoco.mj_forward(model, data)

    ghost_pos = data.site_xpos[ee_site_id].copy()
    data.mocap_pos[mocap_id] = ghost_pos
    mujoco.mj_forward(model, data)

    # cv2.namedWindow("static_cam", cv2.WINDOW_NORMAL)

    delta_ema = np.zeros(3)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        for t in range(MAX_STEPS):

            mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=cam_id)
            img = preprocess_rgb(renderer.render())

            # cv2.imshow("static_cam", img)
            # cv2.waitKey(1)

            # Inference
            if t % INFER_EVERY == 0:
                action = query_openvla(img, TEXT_GOAL)
                delta = np.tanh(action[:3])  
                delta_ema = EMA_ALPHA * delta_ema + (1 - EMA_ALPHA) * delta

            # Update target ball
            ghost_pos = ghost_pos + VLA_SCALE * delta_ema
            data.mocap_pos[mocap_id] = ghost_pos
            mujoco.mj_forward(model, data)

            # Ik
            err = ik_step(model, data, ghost_pos, ee_site_id)

            if t % 10 == 0:
                print(
                    f"[{t:03d}] "
                    f"delta={np.round(delta_ema,3)} "
                    f"ghost={np.round(ghost_pos,3)} "
                    f"err={err:.4f}"
                )

            viewer.sync()
            time.sleep(0.01)

        print("\n[DONE]")

        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
