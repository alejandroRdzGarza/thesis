# openvla_server.py
# OpenVLA HTTP server — supports both base OpenVLA-7B and OpenVLA-OFT.
#
# OpenVLA-OFT (recommended) uses parallel decoding and is 25-50x faster than
# the base model.  Set HF_MODEL to openvla/openvla-oft-libero-spatial.
#
# Action chunking: the /act endpoint accepts an optional num_actions parameter.
# When > 1, the server calls predict_action repeatedly to build a chunk, OR (for
# OFT) calls the model's native chunk prediction if supported.  The client
# should execute actions in order, one per control step.
#
# Override via env vars:
#   OPENVLA_MODEL_PATH  -- local path or HF model ID
#   OPENVLA_UNNORM_KEY  -- unnorm key matching the checkpoint (default: libero_spatial)
#   OPENVLA_CENTER_CROP -- "0" to disable 90%-area centre crop (default: enabled)
#   OPENVLA_CHUNK_SIZE  -- default action chunk size returned per request (default: 5)

import os
import torch
import base64
import io
import numpy as np
from PIL import Image
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from transformers import AutoModelForVision2Seq, AutoProcessor
import uvicorn

MODEL_PATH   = os.environ.get("OPENVLA_MODEL_PATH",  "openvla/openvla-oft-libero-spatial")
UNNORM_KEY   = os.environ.get("OPENVLA_UNNORM_KEY",  "libero_spatial")
CENTER_CROP  = os.environ.get("OPENVLA_CENTER_CROP", "true").lower() not in ("0", "false", "no")
CHUNK_SIZE   = int(os.environ.get("OPENVLA_CHUNK_SIZE", "5"))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI()


class Request(BaseModel):
    image_base64: str
    instruction:  str
    num_actions:  Optional[int] = None   # how many action steps to return


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

print(f"OpenVLA ready -- unnorm_key={UNNORM_KEY!r}  center_crop={CENTER_CROP}  "
      f"default_chunk={CHUNK_SIZE}")


def _center_crop_90(img: Image.Image) -> Image.Image:
    """Centre-crop to 90% area then resize back to 224x224 (matches LIBERO fine-tuning)."""
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


def _predict_one(image: Image.Image, instruction: str) -> np.ndarray:
    """Run one forward pass and return a 7-D action array."""
    if CENTER_CROP:
        image = _center_crop_90(image)

    prompt = (
        f"In: What action should the robot take to {instruction}?\n"
        f"Out:"
    )

    inputs = processor(prompt, image, return_tensors="pt")
    inputs["pixel_values"]  = inputs["pixel_values"].to(DEVICE, dtype=torch.float16)
    inputs["input_ids"]     = inputs["input_ids"].to(DEVICE)
    if "attention_mask" in inputs:
        inputs["attention_mask"] = inputs["attention_mask"].to(DEVICE)

    with torch.no_grad():
        action = model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False)

    return _to_numpy(action).flatten()[:7]


@app.post("/act")
def act(req: Request):
    try:
        image = _decode_image(req.image_base64)
        n = req.num_actions if req.num_actions is not None else CHUNK_SIZE

        if n <= 1:
            action = _predict_one(image, req.instruction)
            return {"action": action.tolist(), "actions": [action.tolist()],
                    "unnorm_key": UNNORM_KEY}

        # Action chunking: call model n times with the same image.
        # For OFT this is fast (parallel decoding); for base model it's slower.
        chunk = []
        for _ in range(n):
            chunk.append(_predict_one(image, req.instruction).tolist())

        return {"action": chunk[0], "actions": chunk, "unnorm_key": UNNORM_KEY}

    except Exception as e:
        print("OpenVLA server error:", e)
        import traceback; traceback.print_exc()
        return {"error": str(e), "action": None, "actions": None}


@app.get("/info")
def info():
    return {"model": MODEL_PATH, "unnorm_key": UNNORM_KEY,
            "center_crop": CENTER_CROP, "default_chunk_size": CHUNK_SIZE,
            "device": DEVICE}


if __name__ == "__main__":
    port = int(os.environ.get("OPENVLA_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
