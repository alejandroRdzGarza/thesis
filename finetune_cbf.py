"""
finetune_cbf.py

Fine-tune OpenVLA-OFT on CBF-corrected LIBERO trajectories stored in our HDF5 format.
Bypasses the RLDS/TFRecord pipeline from finetune.py — reads directly from merged HDF5.

Architecture:
  - Loads existing fine-tuned checkpoint (backbone + action_head + proprio_projector)
  - Applies fresh LoRA (rank=32) to backbone for continued adaptation
  - L1 regression loss on 8-step action chunk (same as original training)
  - Trains to predict safe (CBF-corrected) xyz deltas instead of raw VLA xyz

Action normalization:
  - dims 0-2 (xyz): normalize safe_cartesian using model's q01/q99 stats
  - dims 3-6 (rotation+gripper): use vla_actions directly (CBF doesn't modify these)

Usage (on RunPod from /workspace/VLA_Benchmark-karl/):
    python finetune_cbf.py \\
        --vla-path /workspace/vla_model \\
        --data cbf_train_L1.h5 cbf_train_L2.h5 \\
        --run-dir runs/cbf_finetune \\
        --max-steps 5000

    # Resume from checkpoint:
    python finetune_cbf.py \\
        --vla-path /workspace/vla_model \\
        --data cbf_train_L1.h5 cbf_train_L2.h5 \\
        --run-dir runs/cbf_finetune \\
        --resume --resume-step 2000
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path
from typing import Iterator, List, Optional

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

# ── Path setup: add openvla-oft-main to sys.path ──────────────────────────────
_SCRIPT_DIR = Path(__file__).parent
_OFT_DIR = _SCRIPT_DIR / "openvla-oft-main"
if _OFT_DIR.exists() and str(_OFT_DIR) not in sys.path:
    sys.path.insert(0, str(_OFT_DIR))

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.models.action_heads import L1RegressionActionHead
from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.models.projectors import ProprioProjector
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)

# ── Constants ──────────────────────────────────────────────────────────────────
UNNORM_KEY = "libero_spatial_no_noops"


# ── Action normalization helpers ───────────────────────────────────────────────

def _normalize(values: np.ndarray, q01: np.ndarray, q99: np.ndarray) -> np.ndarray:
    """BOUNDS_Q99 normalization: map [q01, q99] → [-1, 1], clip to [-1, 1]."""
    denom = q99 - q01
    denom = np.where(np.abs(denom) < 1e-8, 1.0, denom)
    return np.clip(2.0 * (values - q01) / denom - 1.0, -1.0, 1.0)


def build_training_target(
    safe_cartesian: np.ndarray,   # (7,) CBF-corrected action in robot space
    vla_action: np.ndarray,       # (7,) raw VLA server output (physical units)
    q01: np.ndarray,              # (7,) action normalization low bound
    q99: np.ndarray,              # (7,) action normalization high bound
) -> np.ndarray:
    """
    Compute normalized training target for one timestep.

    CBF only modifies xyz (dims 0-2).  Rotation and gripper (dims 3-6) are taken
    directly from the raw VLA output so the model is not penalized for rotation or
    gripper choices that the CBF did not affect.
    """
    target = vla_action.copy()
    target[0:3] = safe_cartesian[0:3]   # override xyz with CBF-safe version
    return _normalize(target, q01, q99)


# ── HDF5 Dataset ──────────────────────────────────────────────────────────────

class LiberoHDF5Dataset(IterableDataset):
    """
    Iterable dataset over CBF-corrected LIBERO episodes stored in merged HDF5.

    Each iteration yields one sample: (observation at t, action chunk t:t+N).

    HDF5 schema expected:
        data/demo_<i>/obs/agentview_image         (T, H, W, 3) uint8
        data/demo_<i>/obs/robot0_eye_in_hand_image (T, H, W, 3) uint8
        data/demo_<i>/obs/robot0_eef_pos           (T, 3)  float32
        data/demo_<i>/obs/robot0_eef_quat          (T, 4)  float32
        data/demo_<i>/obs/robot0_gripper_qpos      (T, 2)  float32
        data/demo_<i>/actions                      (T, 7)  float32  ← CBF-corrected
        data/demo_<i>/vla_actions                  (T, 7)  float32  ← raw VLA output
        data/demo_<i>.attrs["language_instruction"]
    """

    def __init__(
        self,
        h5_paths: List[str],
        action_tokenizer: ActionTokenizer,
        base_tokenizer,
        image_transform,
        action_q01: np.ndarray,
        action_q99: np.ndarray,
        proprio_q01: np.ndarray,
        proprio_q99: np.ndarray,
        shuffle: bool = True,
    ):
        self.h5_paths     = [str(p) for p in h5_paths]
        self.action_tok   = action_tokenizer
        self.base_tok     = base_tokenizer
        self.img_tf       = image_transform
        self.action_q01   = action_q01
        self.action_q99   = action_q99
        self.proprio_q01  = proprio_q01
        self.proprio_q99  = proprio_q99
        self.shuffle      = shuffle

        # Build episode index: (file_path, demo_key)
        self.episodes: List[tuple] = []
        for p in self.h5_paths:
            with h5py.File(p, "r") as f:
                for key in f["data"].keys():
                    if f["data"][key].attrs.get("num_steps", 0) >= NUM_ACTIONS_CHUNK:
                        self.episodes.append((p, key))

        print(f"  [dataset] {len(self.episodes)} episodes across {len(self.h5_paths)} file(s)")

    def _make_sample(self, lang: str, img: np.ndarray, wrist: np.ndarray,
                     eef_pos: np.ndarray, eef_quat: np.ndarray, gripper_qpos: np.ndarray,
                     safe_actions: np.ndarray, vla_actions: np.ndarray, t: int) -> dict:
        T = len(safe_actions)
        chunk_end = min(t + NUM_ACTIONS_CHUNK, T)
        chunk_safe = safe_actions[t:chunk_end]
        chunk_vla  = vla_actions[t:chunk_end]

        # Pad to NUM_ACTIONS_CHUNK by repeating last action
        if len(chunk_safe) < NUM_ACTIONS_CHUNK:
            pad = NUM_ACTIONS_CHUNK - len(chunk_safe)
            chunk_safe = np.concatenate([chunk_safe, np.tile(chunk_safe[-1], (pad, 1))])
            chunk_vla  = np.concatenate([chunk_vla,  np.tile(chunk_vla[-1],  (pad, 1))])

        # Normalized action chunk — shape (NUM_ACTIONS_CHUNK, ACTION_DIM)
        norm_chunk = np.stack([
            build_training_target(chunk_safe[i], chunk_vla[i], self.action_q01, self.action_q99)
            for i in range(NUM_ACTIONS_CHUNK)
        ])

        # Tokenize current action for building the prompt labels
        current_norm = norm_chunk[0]
        current_action_string = self.action_tok(current_norm)
        future_action_string  = "".join(self.action_tok(norm_chunk[i]) for i in range(1, NUM_ACTIONS_CHUNK))
        action_chunk_string   = current_action_string + future_action_string
        chunk_tok_len         = len(action_chunk_string)

        # Build prompt
        prompt_builder = PurePromptBuilder("openvla")
        conv = [
            {"from": "human", "value": f"What action should the robot take to {lang.lower()}?"},
            {"from": "gpt",   "value": action_chunk_string},
        ]
        for turn in conv:
            prompt_builder.add_turn(turn["from"], turn["value"])
        raw_ids = self.base_tok(prompt_builder.get_prompt(), add_special_tokens=True).input_ids
        input_ids = torch.tensor(raw_ids)
        labels    = input_ids.clone()
        labels[:-(chunk_tok_len + 1)] = IGNORE_INDEX

        # Images
        pv       = self.img_tf(Image.fromarray(img.astype(np.uint8)))
        pv_wrist = self.img_tf(Image.fromarray(wrist.astype(np.uint8)))

        # Proprio: eef_pos(3) + eef_quat(4) + gripper_scalar(1) = 8D
        proprio_raw = np.concatenate([eef_pos, eef_quat, gripper_qpos[:1]])
        proprio_norm = _normalize(proprio_raw, self.proprio_q01, self.proprio_q99)

        return {
            "pixel_values":       pv,
            "pixel_values_wrist": pv_wrist,
            "input_ids":          input_ids,
            "labels":             labels,
            "actions":            norm_chunk.astype(np.float32),  # numpy — collator wraps in torch
            "proprio":            torch.tensor(proprio_norm, dtype=torch.float32),
            "dataset_name":       "safelibero",
        }

    def __iter__(self) -> Iterator[dict]:
        episodes = list(self.episodes)
        if self.shuffle:
            rng = np.random.default_rng()
            rng.shuffle(episodes)

        for (fpath, dkey) in episodes:
            with h5py.File(fpath, "r") as f:
                demo = f[f"data/{dkey}"]
                lang = demo.attrs.get("language_instruction", "complete the task")

                imgs     = demo["obs"]["agentview_image"][:]
                wrists   = demo["obs"]["robot0_eye_in_hand_image"][:]
                eef_pos  = demo["obs"]["robot0_eef_pos"][:]
                eef_quat = demo["obs"]["robot0_eef_quat"][:]
                grip_qpos = demo["obs"]["robot0_gripper_qpos"][:]
                safe_acts = demo["actions"][:]
                vla_acts  = demo["vla_actions"][:]
                T = len(safe_acts)

            step_order = list(range(T - NUM_ACTIONS_CHUNK + 1))
            if self.shuffle:
                np.random.shuffle(step_order)

            for t in step_order:
                yield self._make_sample(
                    lang, imgs[t], wrists[t],
                    eef_pos[t], eef_quat[t], grip_qpos[t],
                    safe_acts, vla_acts, t,
                )


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def _find_component_ckpt(checkpoint_dir: str, pattern: str) -> Optional[str]:
    matches = glob.glob(os.path.join(checkpoint_dir, f"{pattern}*checkpoint*.pt"))
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        matches.sort()
        print(f"  [warn] multiple {pattern} checkpoints found, using: {matches[-1]}")
        return matches[-1]
    return None


def _load_component(module: nn.Module, path: Optional[str], name: str) -> nn.Module:
    if path and os.path.isfile(path):
        sd = torch.load(path, map_location="cpu", weights_only=True)
        # strip DDP "module." prefix if present
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        module.load_state_dict(sd)
        print(f"  [ckpt] loaded {name} from {path}")
    else:
        print(f"  [ckpt] {name}: no checkpoint found, using random init")
    return module


def save_checkpoint(step: int, run_dir: Path, vla, action_head, proprio_projector, processor):
    ckpt_dir = run_dir / f"checkpoint_step{step:06d}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    # LoRA adapter
    adapter_dir = ckpt_dir / "lora_adapter"
    adapter_dir.mkdir(exist_ok=True)
    vla.save_pretrained(adapter_dir)
    # Action head and proprio projector
    torch.save(action_head.state_dict(),       ckpt_dir / f"action_head--{step}_checkpoint.pt")
    torch.save(proprio_projector.state_dict(), ckpt_dir / f"proprio_projector--{step}_checkpoint.pt")
    # Processor (tokenizer etc.)
    processor.save_pretrained(ckpt_dir)
    print(f"  [save] checkpoint at step {step} → {ckpt_dir}")
    return ckpt_dir


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Fine-tune OpenVLA-OFT on CBF-corrected HDF5 data")
    p.add_argument("--vla-path",        default="/workspace/vla_model",
                   help="Path to base VLA checkpoint (local dir or HF Hub ID)")
    p.add_argument("--data",            nargs="+",
                   default=["cbf_train_L1.h5", "cbf_train_L2.h5"],
                   help="Paths to merged HDF5 files (space-separated)")
    p.add_argument("--run-dir",         default="runs/cbf_finetune")
    p.add_argument("--unnorm-key",      default=UNNORM_KEY,
                   help="Key into model.norm_stats for action normalization")
    p.add_argument("--lora-rank",       type=int,   default=32)
    p.add_argument("--lora-dropout",    type=float, default=0.0)
    p.add_argument("--lr",              type=float, default=2e-4)
    p.add_argument("--lr-decay-step",   type=int,   default=80_000)
    p.add_argument("--max-steps",       type=int,   default=10_000)
    p.add_argument("--grad-accum",      type=int,   default=8,
                   help="Gradient accumulation steps (effective batch = grad_accum)")
    p.add_argument("--save-freq",       type=int,   default=1000)
    p.add_argument("--log-freq",        type=int,   default=50)
    p.add_argument("--resume",          action="store_true")
    p.add_argument("--resume-step",     type=int,   default=None)
    p.add_argument("--no-image-aug",    action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # ── Model loading ──────────────────────────────────────────────────────────
    print(f"\nLoading VLA from: {args.vla_path}")
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    # Copy modeling_prismatic.py to checkpoint dir so HF loads our version
    src_modeling = _OFT_DIR / "prismatic" / "extern" / "hf" / "modeling_prismatic.py"
    dst_modeling  = Path(args.vla_path) / "modeling_prismatic.py"
    if src_modeling.exists() and not dst_modeling.exists():
        import shutil
        shutil.copy(src_modeling, dst_modeling)

    processor = AutoProcessor.from_pretrained(args.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        args.vla_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # Enable gradient checkpointing to reduce VRAM on 24GB GPU
    try:
        vla.language_model.gradient_checkpointing_enable()
    except AttributeError:
        try:
            vla.gradient_checkpointing_enable()
        except Exception:
            print("  [warn] gradient checkpointing not available for this model")

    # Set dual-image input (primary + wrist)
    vla.vision_backbone.set_num_images_in_input(2)

    # Compute NUM_PATCHES before LoRA wrapping
    NUM_PATCHES = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
        + 1  # +1 for proprio token
    )

    # Get LLM dim before LoRA wrapping
    llm_dim = vla.llm_dim

    # Load dataset statistics from checkpoint dir (LIBERO stats are stored separately)
    import json
    stats_path = Path(args.vla_path) / "dataset_statistics.json"
    if stats_path.exists():
        with open(stats_path) as f:
            extra_stats = json.load(f)
        vla.norm_stats.update(extra_stats)
        print(f"  loaded norm_stats from {stats_path}: keys={list(extra_stats.keys())}")

    # Retrieve action normalization stats
    if args.unnorm_key not in vla.norm_stats:
        available = list(vla.norm_stats.keys())
        raise ValueError(
            f"unnorm_key '{args.unnorm_key}' not in model.norm_stats. "
            f"Available: {available}"
        )
    action_stats = vla.norm_stats[args.unnorm_key]["action"]
    action_q01   = np.array(action_stats["q01"], dtype=np.float32)
    action_q99   = np.array(action_stats["q99"], dtype=np.float32)

    # Proprio normalization stats (if available)
    proprio_stats = vla.norm_stats[args.unnorm_key].get("proprio", None)
    if proprio_stats is not None:
        proprio_q01 = np.array(proprio_stats["q01"], dtype=np.float32)
        proprio_q99 = np.array(proprio_stats["q99"], dtype=np.float32)
    else:
        # Fallback: no normalization (identity)
        proprio_q01 = np.full(PROPRIO_DIM, -1.0, dtype=np.float32)
        proprio_q99 = np.full(PROPRIO_DIM,  1.0, dtype=np.float32)

    print(f"  action q01: {action_q01}")
    print(f"  action q99: {action_q99}")

    # ── Apply LoRA (fresh training) or reload adapter (resume) ────────────────
    if args.resume and args.resume_step is not None:
        from peft import PeftModel
        resume_adapter = run_dir / f"checkpoint_step{args.resume_step:06d}" / "lora_adapter"
        vla = PeftModel.from_pretrained(vla, str(resume_adapter), is_trainable=True)
        print(f"  [resume] loaded LoRA adapter from {resume_adapter}")
    else:
        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=min(args.lora_rank, 16),
            lora_dropout=args.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
    vla.print_trainable_parameters()

    # ── Action head & proprio projector ───────────────────────────────────────
    action_head = L1RegressionActionHead(
        input_dim=llm_dim, hidden_dim=llm_dim, action_dim=ACTION_DIM
    ).to(torch.bfloat16).to(device)

    proprio_projector = ProprioProjector(
        llm_dim=llm_dim, proprio_dim=PROPRIO_DIM
    ).to(torch.bfloat16).to(device)

    # Load from base checkpoint (the libero_spatial fine-tuned weights)
    if not args.resume:
        ah_path  = _find_component_ckpt(args.vla_path, "action_head")
        pp_path  = _find_component_ckpt(args.vla_path, "proprio_projector")
        _load_component(action_head,       ah_path,  "action_head")
        _load_component(proprio_projector, pp_path,  "proprio_projector")
    else:
        ckpt_base = run_dir / f"checkpoint_step{args.resume_step:06d}"
        ah_path  = _find_component_ckpt(str(ckpt_base), "action_head")
        pp_path  = _find_component_ckpt(str(ckpt_base), "proprio_projector")
        _load_component(action_head,       ah_path,  "action_head")
        _load_component(proprio_projector, pp_path,  "proprio_projector")

    # ── Optimizer & scheduler ──────────────────────────────────────────────────
    trainable_params = (
        [p for p in vla.parameters()              if p.requires_grad]
        + [p for p in action_head.parameters()    if p.requires_grad]
        + [p for p in proprio_projector.parameters() if p.requires_grad]
    )
    print(f"  total trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = AdamW(trainable_params, lr=args.lr)
    scheduler = MultiStepLR(optimizer, milestones=[args.lr_decay_step], gamma=0.1)

    # ── Dataset & dataloader ───────────────────────────────────────────────────
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    train_ds = LiberoHDF5Dataset(
        h5_paths          = args.data,
        action_tokenizer  = action_tokenizer,
        base_tokenizer    = processor.tokenizer,
        image_transform   = processor.image_processor.apply_transform,
        action_q01        = action_q01,
        action_q99        = action_q99,
        proprio_q01       = proprio_q01,
        proprio_q99       = proprio_q99,
        shuffle           = True,
    )

    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )

    dataloader = DataLoader(
        train_ds,
        batch_size=1,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )

    # ── Training loop ──────────────────────────────────────────────────────────
    print(f"\nStarting training: max_steps={args.max_steps}  grad_accum={args.grad_accum}")
    print(f"  run_dir: {run_dir}")

    vla.train()
    action_head.train()
    proprio_projector.train()
    optimizer.zero_grad()

    start_step = (args.resume_step or 0) if args.resume else 0
    global_step   = start_step
    batch_idx     = 0
    running_loss  = 0.0
    t0            = time.time()

    def _infinite(dl):
        while True:
            for b in dl:
                yield b

    for batch in _infinite(dataloader):
        if global_step >= args.max_steps:
            break

        # ── Forward ──────────────────────────────────────────────────────────
        ground_truth_actions = batch["actions"].to(device).to(torch.bfloat16)
        labels_dev = batch["labels"].to(device)

        # Proprio: collator squeezes batch dim for size-1 batches
        proprio = None
        if "proprio" in batch and batch["proprio"] is not None:
            proprio = batch["proprio"].to(torch.bfloat16).to(device)
            if proprio.dim() == 1:
                proprio = proprio.unsqueeze(0)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = vla(
                input_ids      = batch["input_ids"].to(device),
                attention_mask = batch["attention_mask"].to(device),
                pixel_values   = batch["pixel_values"].to(torch.bfloat16).to(device),
                labels         = labels_dev,
                output_hidden_states = True,
                proprio              = proprio,
                proprio_projector    = proprio_projector,
            )

        # Hidden states → action token positions
        last_hidden = output.hidden_states[-1]            # (B, seq_len, D)
        text_hidden = last_hidden[:, NUM_PATCHES:-1]      # text portion only

        ground_truth_token_ids = labels_dev[:, 1:]
        curr_mask = get_current_action_mask(ground_truth_token_ids)
        next_mask = get_next_actions_mask(ground_truth_token_ids)

        B = batch["input_ids"].shape[0]
        action_hidden = (
            text_hidden[curr_mask | next_mask]
            .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
            .to(torch.bfloat16)
        )

        predicted_actions = action_head.predict_action(action_hidden)
        loss = nn.L1Loss()(ground_truth_actions, predicted_actions)

        # ── Backward (gradient accumulation) ────────────────────────────────
        (loss / args.grad_accum).backward()
        running_loss += loss.item()
        batch_idx    += 1

        if batch_idx % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.log_freq == 0:
                # running_loss accumulated over log_freq * grad_accum batches
                avg_loss = running_loss / (args.log_freq * args.grad_accum)
                lr_now   = optimizer.param_groups[0]["lr"]
                elapsed  = time.time() - t0
                print(
                    f"  step {global_step:6d}/{args.max_steps}"
                    f"  loss={avg_loss:.4f}"
                    f"  lr={lr_now:.2e}"
                    f"  {elapsed:.0f}s elapsed"
                )
                running_loss = 0.0

            if global_step % args.save_freq == 0:
                save_checkpoint(global_step, run_dir, vla, action_head, proprio_projector, processor)

            if global_step >= args.max_steps:
                break

    # ── Final save ────────────────────────────────────────────────────────────
    final_dir = save_checkpoint(global_step, run_dir, vla, action_head, proprio_projector, processor)
    print(f"\nTraining complete.  Final checkpoint: {final_dir}")
    print(f"\nTo merge LoRA weights into a standalone model for inference:")
    print(f"  python openvla-oft-main/vla-scripts/merge_lora_weights_and_save.py \\")
    print(f"      --vla-path {args.vla_path} \\")
    print(f"      --lora-adapter-path {final_dir}/lora_adapter \\")
    print(f"      --output-path runs/cbf_merged")


if __name__ == "__main__":
    main()
