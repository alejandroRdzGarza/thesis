# openvla_server.py
# Stable OpenVLA HTTP server for MuJoCo / IK directional control

import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
import uvicorn

import os
MODEL_PATH = os.environ.get("OPENVLA_MODEL_PATH", "openvla/openvla-7b")
DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")


UNNORM_KEY = "bridge_orig"

app = FastAPI()

class Request(BaseModel):
    image_base64: str
    instruction: str

# Load OpenVLA
print("Loading OpenVLA processor...")
processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b",
    trust_remote_code=True,
)

print("Loading OpenVLA model (8-bit quantized)...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,
)
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    quantization_config=bnb_config,
    device_map={"": 0},
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)
model.eval()

print("OpenVLA ready.")


# utils
def decode_image(b64: str) -> Image.Image:
    img_bytes = base64.b64decode(b64)
    return Image.open(io.BytesIO(img_bytes)).convert("RGB")

def to_numpy_action(action):
    """
    OpenVLA predict_action may return:
    - torch.Tensor
    - numpy.ndarray

    This function safely converts both to np.ndarray
    """
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy()
    elif isinstance(action, np.ndarray):
        return action
    else:
        raise RuntimeError(f"Unknown action type: {type(action)}")

# API
@app.post("/act")
def act(req: Request):
    try:
        # Decode image
        image = decode_image(req.image_base64)

        prompt = (
            f"In: What action should the robot take to {req.instruction}?\n"
            f"Out:"
        )

        # Preprocess
        inputs = processor(prompt, image, return_tensors="pt")

        inputs["pixel_values"] = inputs["pixel_values"].to(
            DEVICE, dtype=torch.float16
        )
        inputs["input_ids"] = inputs["input_ids"].to(DEVICE)

        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)

        # Inference
        with torch.no_grad():
            action = model.predict_action(
                **inputs,
                unnorm_key=UNNORM_KEY, 
                do_sample=False
            )

        action_np = to_numpy_action(action)

        return {
            "action": action_np.tolist(),
            "unnorm_key": UNNORM_KEY,
        }

    except Exception as e:
        # Error check
        print(" OpenVLA server error:", e)
        return {
            "error": str(e),
            "action": None,
        }

# Run server
if __name__ == "__main__":
    port = int(os.environ.get("OPENVLA_PORT", 8000))
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info",
    )
