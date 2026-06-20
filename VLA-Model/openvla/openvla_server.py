# openvla_server.py
# OpenVLA HTTP server for LIBERO / MuJoCo experiments.
#
# Defaults to the LIBERO-Spatial fine-tuned checkpoint which achieves ~84.7%
# task success. The base openvla-7b with bridge_orig unnorm is a WidowX policy
# and produces nonsensical actions for Franka Panda / LIBERO scenes.
#
# Override via env vars:
#   OPENVLA_MODEL_PATH  -- local path or HF model ID
#   OPENVLA_UNNORM_KEY  -- unnorm key matching the checkpoint
#   OPENVLA_CENTER_CROP -- set "0" to disable 90%-area centre crop

import os
import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForVision2Seq, AutoProcessor
import uvicorn

MODEL_PATH  = os.environ.get("OPENVLA_MODEL_PATH", "openvla/openvla-7b-finetuned-libero-spatial")
UNNORM_KEY  = os.environ.get("OPENVLA_UNNORM_KEY",  "libero_spatial")
CENTER_CROP = os.environ.get("OPENVLA_CENTER_CROP", "true").lower() not in ("0", "false", "no")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()


class Request(BaseModel):
    image_base64: str
    instruction:  str


print(f"Loading processor from {MODEL_PATH} ...")
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)

print(f"Loading model ({MODEL_PATH}) ...")
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.float16,
    trust_remote_code=True,
    low_cpu_mem_usage=True,
).to(DEVICE)
model.eval()

print(f"OpenVLA ready -- unnorm_key={UNNORM_KEY!r}  center_crop={CENTER_CROP}")


def _center_crop_90(img: Image.Image) -> Image.Image:
    """Centre-crop to 90% area then resize back to original size.

    The LIBERO fine-tuned model was trained with random 90%-area crops; at
    inference we use the deterministic centre crop to match training conditions.
    sqrt(0.9) ~= 0.9487, so a 224x224 image is cropped to ~212x212 then resized.
    """
    w, h = img.size
    crop_side = int(min(w, h) * (0.9 ** 0.5))
    left = (w - crop_side) // 2
    top  = (h - crop_side) // 2
    img  = img.crop((left, top, left + crop_side, top + crop_side))
    return img.resize((w, h), Image.BILINEAR)


def _decode_image(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _to_numpy(action) -> np.ndarray:
    if isinstance(action, torch.Tensor):
        return action.detach().cpu().numpy()
    return np.asarray(action)


@app.post("/act")
def act(req: Request):
    try:
        image = _decode_image(req.image_base64)
        if CENTER_CROP:
            image = _center_crop_90(image)

        prompt = (
            f"In: What action should the robot take to {req.instruction}?\n"
            f"Out:"
        )

        inputs = processor(prompt, image, return_tensors="pt")
        inputs["pixel_values"]  = inputs["pixel_values"].to(DEVICE, dtype=torch.float16)
        inputs["input_ids"]     = inputs["input_ids"].to(DEVICE)
        if "attention_mask" in inputs:
            inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)

        with torch.no_grad():
            action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

        return {"action": _to_numpy(action).tolist(), "unnorm_key": UNNORM_KEY}

    except Exception as e:
        print("OpenVLA server error:", e)
        return {"error": str(e), "action": None}


if __name__ == "__main__":
    port = int(os.environ.get("OPENVLA_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
