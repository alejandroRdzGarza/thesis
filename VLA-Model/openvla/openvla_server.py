# openvla_server.py — OpenVLA-OFT HTTP server
#
# Uses the moojink/openvla-oft inference stack (L1-regression action head,
# wrist camera, proprioceptive state).  Install the repo first:
#
#   git clone https://github.com/moojink/openvla-oft /workspace/openvla_oft_repo
#   pip install -e /workspace/openvla_oft_repo
#
# Override via env vars:
#   OPENVLA_MODEL_PATH  -- HF model ID or local path  (default: moojink/openvla-7b-oft-finetuned-libero-spatial)
#   OPENVLA_UNNORM_KEY  -- unnorm key for action space (default: libero_spatial_no_noops)
#   OPENVLA_CHUNK_SIZE  -- actions returned per request (default: 5)
#   OPENVLA_OFT_REPO    -- path to cloned openvla-oft repo (default: /workspace/openvla_oft_repo)
#   OPENVLA_PORT        -- server port (default: 8000)

import os
import sys
import base64
import io
import math

import numpy as np
import torch
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, List
from PIL import Image

# ── OFT repo on path ──────────────────────────────────────────────────────────
OFT_REPO = os.environ.get("OPENVLA_OFT_REPO", "/workspace/openvla_oft_repo")
if OFT_REPO not in sys.path:
    sys.path.insert(0, OFT_REPO)

from dataclasses import dataclass, field

# Define GenerateConfig here to avoid importing run_libero_eval.py,
# which imports libero at module level (not needed for inference).
@dataclass
class GenerateConfig:
    pretrained_checkpoint:        str   = "moojink/openvla-7b-oft-finetuned-libero-spatial"
    use_l1_regression:            bool  = True
    use_diffusion:                bool  = False
    use_film:                     bool  = False
    num_images_in_input:          int   = 2
    use_proprio:                  bool  = True
    load_in_8bit:                 bool  = False
    load_in_4bit:                 bool  = False
    center_crop:                  bool  = True
    num_open_loop_steps:          int   = 5
    unnorm_key:                   str   = "libero_spatial_no_noops"
    lora_rank:                    int   = 32    # OFT default; not used for HF checkpoints
    num_diffusion_steps_train:    int   = 100   # unused (use_diffusion=False)
    num_diffusion_steps_inference: int  = 10    # unused (use_diffusion=False)

# LIBERO-Spatial constants (confirmed from prismatic/vla/constants.py).
# Hardcoded to avoid importing prismatic.vla.constants which triggers the
# training dataset import chain (dlimp → tfds → protobuf crash).
NUM_ACTIONS_CHUNK = 8   # actions generated per inference call
PROPRIO_DIM       = 8   # eef_pos(3) + axis_angle(3) + gripper_qpos(2)

try:
    from experiments.robot.openvla_utils import (
        get_action_head, get_processor, get_proprio_projector,
        get_vla, get_vla_action,
    )
except ImportError as e:
    raise RuntimeError(
        f"Cannot import openvla-oft: {e}\n"
        f"Clone and install: git clone https://github.com/moojink/openvla-oft {OFT_REPO} && "
        f"pip install -e {OFT_REPO}\n"
        f"Then patch: echo '' > {OFT_REPO}/prismatic/__init__.py"
    ) from e

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_PATH  = os.environ.get("OPENVLA_MODEL_PATH", "moojink/openvla-7b-oft-finetuned-libero-spatial")
UNNORM_KEY  = os.environ.get("OPENVLA_UNNORM_KEY",  "libero_spatial_no_noops")
CHUNK_SIZE  = int(os.environ.get("OPENVLA_CHUNK_SIZE", "5"))

print(f"OpenVLA-OFT server config:")
print(f"  model       = {MODEL_PATH}")
print(f"  unnorm_key  = {UNNORM_KEY}")
print(f"  chunk_size  = {CHUNK_SIZE}")
print(f"  PROPRIO_DIM = {PROPRIO_DIM}")
print(f"  NUM_ACTIONS_CHUNK (model default) = {NUM_ACTIONS_CHUNK}")

# ── Load model ────────────────────────────────────────────────────────────────
cfg = GenerateConfig(
    pretrained_checkpoint=MODEL_PATH,
    use_l1_regression=True,
    use_diffusion=False,
    use_film=False,
    num_images_in_input=2,          # agentview + wrist
    use_proprio=True,
    load_in_8bit=False,
    load_in_4bit=False,
    center_crop=True,
    num_open_loop_steps=CHUNK_SIZE,
    unnorm_key=UNNORM_KEY,
)

print("Loading VLA backbone ...")
vla = get_vla(cfg)

print("Loading processor ...")
processor = get_processor(cfg)

print("Loading action head ...")
action_head = get_action_head(cfg, llm_dim=vla.llm_dim)

print("Loading proprio projector ...")
proprio_projector = get_proprio_projector(cfg, llm_dim=vla.llm_dim, proprio_dim=PROPRIO_DIM)

print(f"OpenVLA-OFT ready — chunk={CHUNK_SIZE}  unnorm={UNNORM_KEY}")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI()


class Request(BaseModel):
    image_base64:       str            # agentview image (flipped, 224×224 uint8 RGB, JPEG b64)
    wrist_image_base64: str            # wrist camera image (same preprocessing)
    state:              List[float]    # 8-D proprio: eef_pos(3) + axis_angle(3) + gripper(2)
    instruction:        str
    num_actions:        Optional[int] = None   # if set, overrides CHUNK_SIZE (best-effort)


def _decode(b64: str) -> np.ndarray:
    """Decode base64 JPEG → uint8 HxWx3 numpy array."""
    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    return np.array(img, dtype=np.uint8)


@app.post("/act")
def act(req: Request):
    try:
        full_img  = _decode(req.image_base64)
        wrist_img = _decode(req.wrist_image_base64)
        state     = np.array(req.state, dtype=np.float64)

        observation = {
            "full_image":       full_img,
            "wrist_image":      wrist_img,
            "state":            state,
            "task_description": req.instruction,
        }

        # get_vla_action returns a list of 7-D numpy arrays (the action chunk).
        actions = get_vla_action(
            cfg, vla, processor, observation,
            req.instruction, action_head, proprio_projector,
        )

        # Trim or pad to requested length (OFT always returns NUM_ACTIONS_CHUNK)
        n = req.num_actions if req.num_actions is not None else CHUNK_SIZE
        actions = list(actions)[:n] if len(actions) >= n else actions

        return {
            "action":  actions[0].tolist(),
            "actions": [a.tolist() for a in actions],
            "unnorm_key": UNNORM_KEY,
        }

    except Exception as e:
        print("OpenVLA-OFT server error:", e)
        import traceback; traceback.print_exc()
        return {"error": str(e), "action": None, "actions": None}


@app.get("/info")
def info():
    return {
        "model":              MODEL_PATH,
        "unnorm_key":         UNNORM_KEY,
        "chunk_size":         CHUNK_SIZE,
        "proprio_dim":        PROPRIO_DIM,
        "num_actions_chunk":  NUM_ACTIONS_CHUNK,
    }


if __name__ == "__main__":
    port = int(os.environ.get("OPENVLA_PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
