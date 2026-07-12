"""
train_obstacle_projector.py — Fine-tune the obstacle-conditioned proprio projector.

Loads DAgger data from collect_obstacle_data.py and trains ObstacleConditionedProjector
while keeping the VLA backbone and action head frozen.

Gradient flow:
    L1 loss → action_head (frozen) → LLM hidden states → projector token → projector weights

Memory: requires a GPU with ≥ 16 GB VRAM (bfloat16 forward pass through 7B model).
Gradient checkpointing is NOT used here — add if OOM.

Usage (UCL GPU / RunPod):
    python -m experiments.train_obstacle_projector \\
        --dataset data/obs_cond_dataset/safelibero_spatial_t00_LI \\
        --checkpoint /workspace/vla_model \\
        --out runs/obs_cond_projector \\
        --epochs 10 --batch-size 1 --lr 1e-4

Output:
    runs/obs_cond_projector/obstacle_projector--best_checkpoint.pt
    Copy this to the model dir and set OPENVLA_OBS_COND=1 on the server.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, random_split

# ── Import openvla-oft stack ──────────────────────────────────────────────────
OFT_REPO = os.environ.get("OPENVLA_OFT_REPO", "/workspace/openvla_oft_repo")
if OFT_REPO not in sys.path:
    sys.path.insert(0, OFT_REPO)

from experiments.robot.openvla_utils import (
    get_action_head,
    get_obstacle_conditioned_projector,
    get_processor,
    get_vla,
    normalize_proprio,
)
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.constants import (
    ACTION_DIM,
    ACTION_PROPRIO_NORMALIZATION_TYPE,
    IGNORE_INDEX,
    NUM_ACTIONS_CHUNK,
    PROPRIO_DIM,
)
from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask

DEVICE  = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
OBS_DIM = 4  # obs_dir(3) + obs_dist(1)


# ── Action normalization helpers ───────────────────────────────────────────────

def normalize_action(action: np.ndarray, norm_stats: dict) -> np.ndarray:
    """Normalize raw action to [-1, 1] (inverse of VLA unnorm step)."""
    if "q99" in norm_stats:
        hi, lo = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    else:
        hi, lo = np.array(norm_stats["max"]),  np.array(norm_stats["min"])
    return np.clip(2.0 * (action - lo) / (hi - lo + 1e-8) - 1.0, -1.0, 1.0)


# ── Dataset ────────────────────────────────────────────────────────────────────

class ObstacleDataset(Dataset):
    """Loads .npz files from collect_obstacle_data.py.

    Returns: (img, wrist, proprio_f32, obs_feat_f32, target_action_f32)
    target_action is the DAgger label (nom or cbf) in RAW action space.
    Normalization to [-1, 1] happens inside the training loop using VLA norm_stats.
    """

    def __init__(self, data_dir: Path):
        files = sorted(data_dir.glob("ep_*.npz"))
        if not files:
            raise FileNotFoundError(f"No ep_*.npz files found in {data_dir}")

        imgs, wrists, proprios, obs_feats, targets = [], [], [], [], []
        for f in files:
            d     = np.load(f)
            label = d["label"]
            ta    = np.where(label[:, None] == 1, d["cbf_act"], d["nom_act"])
            imgs.append(d["img"])
            wrists.append(d["wrist"])
            proprios.append(d["proprio"])
            obs_feats.append(d["obs_feat"])
            targets.append(ta)

        self.imgs      = np.concatenate(imgs,      axis=0)
        self.wrists    = np.concatenate(wrists,    axis=0)
        self.proprios  = np.concatenate(proprios,  axis=0)
        self.obs_feats = np.concatenate(obs_feats, axis=0)
        self.targets   = np.concatenate(targets,   axis=0)
        print(f"Dataset: {len(self.imgs)} steps from {len(files)} episodes in {data_dir}")

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        return (
            self.imgs[idx],
            self.wrists[idx],
            self.proprios[idx],
            self.obs_feats[idx],
            self.targets[idx],
        )


def _collate(batch):
    imgs, wrists, proprios, obs_feats, targets = zip(*batch)
    return (
        list(imgs),
        list(wrists),
        torch.tensor(np.stack(proprios),  dtype=torch.float32),
        torch.tensor(np.stack(obs_feats), dtype=torch.float32),
        torch.tensor(np.stack(targets),   dtype=torch.float32),
    )


# ── Input construction ─────────────────────────────────────────────────────────

def build_vla_inputs(
    imgs, wrists, processor, action_tokenizer,
    norm_targets: np.ndarray,  # (B, 7) normalized actions
    instruction: str,
):
    """Build input_ids, attention_mask, pixel_values, labels, proprio placeholder.

    norm_targets are used to create the label token IDs for the action mask.
    The sequence is: [prompt tokens] [action tokens × NUM_ACTIONS_CHUNK × ACTION_DIM] [EOS].
    Action tokens are repeated NUM_ACTIONS_CHUNK times (chunk prediction).
    """
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"
    B = len(imgs)

    all_input_ids, all_attn, all_pixel, all_labels = [], [], [], []

    for i in range(B):
        from PIL import Image as _PIL
        img_pil = _PIL.fromarray(imgs[i])

        # Tokenise prompt (includes image placeholder)
        enc = processor(prompt, img_pil)
        p_ids = enc["input_ids"][0]          # (prompt_len,)
        p_px  = enc["pixel_values"]          # (1, C, H, W)

        # Also encode wrist image pixel values
        wrist_pil = _PIL.fromarray(wrists[i])
        wrist_enc = processor(prompt, wrist_pil)
        wrist_px  = wrist_enc["pixel_values"]

        # Tokenise action for current step (repeated to fill chunk)
        act_str  = action_tokenizer(norm_targets[i])  # e.g. "tok0 tok1 ... tok6"
        act_ids  = processor.tokenizer(act_str, add_special_tokens=False).input_ids
        # Repeat the single-step action to fill the full chunk
        act_ids_chunk = act_ids * NUM_ACTIONS_CHUNK   # list of ACTION_DIM * NUM_ACTIONS_CHUNK ids

        eos_id = processor.tokenizer.eos_token_id

        full_ids = list(p_ids) + act_ids_chunk + [eos_id]
        labels   = [IGNORE_INDEX] * len(p_ids) + act_ids_chunk + [eos_id]

        all_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
        all_attn.append(torch.ones(len(full_ids), dtype=torch.long))
        all_pixel.append(torch.cat([p_px, wrist_px], dim=1))  # (1, 2C, H, W)
        all_labels.append(torch.tensor(labels, dtype=torch.long))

    # Pad to equal length
    max_len = max(t.shape[0] for t in all_input_ids)
    for i in range(B):
        pad = max_len - all_input_ids[i].shape[0]
        all_input_ids[i] = F.pad(all_input_ids[i], (0, pad), value=processor.tokenizer.pad_token_id or 0)
        all_attn[i]      = F.pad(all_attn[i],      (0, pad), value=0)
        all_labels[i]    = F.pad(all_labels[i],    (0, pad), value=IGNORE_INDEX)

    return (
        torch.stack(all_input_ids).to(DEVICE),
        torch.stack(all_attn).to(DEVICE),
        torch.cat(all_pixel, dim=0).to(DEVICE, dtype=torch.bfloat16),
        torch.stack(all_labels).to(DEVICE),
    )


# ── Training ───────────────────────────────────────────────────────────────────

def train(args):
    @dataclass
    class Cfg:
        pretrained_checkpoint: str  = args.checkpoint
        use_l1_regression: bool     = True
        use_diffusion: bool         = False
        use_film: bool              = False
        num_images_in_input: int    = 2
        use_proprio: bool           = True
        load_in_8bit: bool          = False
        load_in_4bit: bool          = False
        center_crop: bool           = True
        num_open_loop_steps: int    = NUM_ACTIONS_CHUNK
        unnorm_key: str             = args.unnorm_key
        lora_rank: int              = args.lora_rank
        num_diffusion_steps_train: int       = 100
        num_diffusion_steps_inference: int   = 10

    cfg = Cfg()

    print("Loading VLA (backbone frozen) ...")
    vla = get_vla(cfg)
    vla.eval()
    for p in vla.parameters():
        p.requires_grad_(False)

    print("Loading processor & action tokenizer ...")
    processor       = get_processor(cfg)
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    print("Loading action head (frozen) ...")
    action_head = get_action_head(cfg, llm_dim=vla.llm_dim)
    action_head.eval()
    for p in action_head.parameters():
        p.requires_grad_(False)

    print("Loading obstacle-conditioned projector (warm-start) ...")
    projector = get_obstacle_conditioned_projector(
        cfg, llm_dim=vla.llm_dim,
        proprio_dim=PROPRIO_DIM, obs_dim=OBS_DIM,
        warm_start_from_proprio=True,
    )
    projector.train()
    for p in projector.parameters():
        p.requires_grad_(True)

    # Norm stats
    proprio_norm_stats = vla.norm_stats[cfg.unnorm_key]["proprio"]
    action_norm_stats  = vla.norm_stats[cfg.unnorm_key]["actions"]

    # Compute number of vision patches (needed to slice hidden states correctly)
    num_patches = (
        vla.vision_backbone.get_num_patches()
        * vla.vision_backbone.get_num_images_in_input()
    )
    if cfg.use_proprio:
        num_patches += 1  # proprio token prepended

    # Dataset
    dataset = ObstacleDataset(Path(args.dataset))
    n_val   = max(1, int(0.1 * len(dataset)))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=_collate, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=_collate, num_workers=0)

    optimizer = AdamW(projector.parameters(), lr=args.lr, weight_decay=1e-4)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    instruction = "pick up the object"  # generic; images carry the task context

    for epoch in range(args.epochs):
        projector.train()
        train_losses = []

        for batch_idx, (imgs, wrists, proprios, obs_feats, targets_raw) in enumerate(train_loader):
            B = len(imgs)

            # Normalize proprio (8-dim) and targets (7-dim) to [-1, 1]
            prop_np  = proprios.numpy()
            tgt_np   = targets_raw.numpy()
            prop_norm  = np.stack([normalize_proprio(prop_np[i], proprio_norm_stats) for i in range(B)])
            tgt_norm   = np.stack([normalize_action(tgt_np[i],  action_norm_stats)   for i in range(B)])

            # Full 12-dim proprio: normalized 8-dim + obs_feat (4-dim, already in [-1,1])
            obs_np       = obs_feats.numpy()
            full_prop_np = np.concatenate([prop_norm, obs_np], axis=-1).astype(np.float32)

            # Build tokenized inputs (uses normalized targets for label token IDs)
            input_ids, attn_mask, pixel_values, labels = build_vla_inputs(
                imgs, wrists, processor, action_tokenizer, tgt_norm, instruction,
            )

            # Ground-truth actions: (B, NUM_ACTIONS_CHUNK, ACTION_DIM)
            # Repeat single-step label to fill chunk (simplification for first-iteration training)
            tgt_tensor = (
                torch.tensor(tgt_norm, dtype=torch.bfloat16, device=DEVICE)
                .unsqueeze(1)
                .expand(-1, NUM_ACTIONS_CHUNK, -1)
            )  # (B, 8, 7)

            # VLA forward (projector receives full 12-dim proprio)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = vla(
                    input_ids=input_ids,
                    attention_mask=attn_mask,
                    pixel_values=pixel_values,
                    labels=labels,
                    output_hidden_states=True,
                    proprio=torch.tensor(full_prop_np, dtype=torch.bfloat16, device=DEVICE),
                    proprio_projector=projector,
                )

            # Extract action hidden states
            last_hidden = output.hidden_states[-1]           # (B, seq_len, D)
            text_hidden = last_hidden[:, num_patches:-1]      # skip vision patches and EOS

            ground_truth_token_ids = labels[:, 1:]
            curr_mask = get_current_action_mask(ground_truth_token_ids)
            next_mask = get_next_actions_mask(ground_truth_token_ids)
            action_hidden = (
                text_hidden[curr_mask | next_mask]
                .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )  # (B, 56, D)

            predicted_actions = action_head.predict_action(action_hidden)  # (B, 8, 7) or (B, 56)
            if predicted_actions.dim() == 2:
                predicted_actions = predicted_actions.reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM)

            loss = F.l1_loss(predicted_actions, tgt_tensor)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

            if batch_idx % 20 == 0:
                print(f"  ep {epoch+1}/{args.epochs}  batch {batch_idx}/{len(train_loader)}"
                      f"  loss={loss.item():.4f}")

        # Validation
        projector.eval()
        val_losses = []
        with torch.no_grad():
            for imgs, wrists, proprios, obs_feats, targets_raw in val_loader:
                B = len(imgs)
                prop_np  = proprios.numpy()
                tgt_np   = targets_raw.numpy()
                prop_norm = np.stack([normalize_proprio(prop_np[i], proprio_norm_stats) for i in range(B)])
                tgt_norm  = np.stack([normalize_action(tgt_np[i],  action_norm_stats)   for i in range(B)])
                obs_np    = obs_feats.numpy()
                full_prop = np.concatenate([prop_norm, obs_np], axis=-1).astype(np.float32)

                input_ids, attn_mask, pixel_values, labels = build_vla_inputs(
                    imgs, wrists, processor, action_tokenizer, tgt_norm, instruction,
                )
                tgt_tensor = (
                    torch.tensor(tgt_norm, dtype=torch.bfloat16, device=DEVICE)
                    .unsqueeze(1).expand(-1, NUM_ACTIONS_CHUNK, -1)
                )

                with torch.autocast("cuda", dtype=torch.bfloat16):
                    output = vla(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        pixel_values=pixel_values,
                        labels=labels,
                        output_hidden_states=True,
                        proprio=torch.tensor(full_prop, dtype=torch.bfloat16, device=DEVICE),
                        proprio_projector=projector,
                    )

                last_hidden = output.hidden_states[-1]
                text_hidden = last_hidden[:, num_patches:-1]
                ground_truth_token_ids = labels[:, 1:]
                curr_mask = get_current_action_mask(ground_truth_token_ids)
                next_mask = get_next_actions_mask(ground_truth_token_ids)
                action_hidden = (
                    text_hidden[curr_mask | next_mask]
                    .reshape(B, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                    .to(torch.bfloat16)
                )
                predicted_actions = action_head.predict_action(action_hidden)
                if predicted_actions.dim() == 2:
                    predicted_actions = predicted_actions.reshape(B, NUM_ACTIONS_CHUNK, ACTION_DIM)
                val_losses.append(F.l1_loss(predicted_actions, tgt_tensor).item())

        mean_train = float(np.mean(train_losses))
        mean_val   = float(np.mean(val_losses))
        print(f"Epoch {epoch+1}/{args.epochs}  train={mean_train:.4f}  val={mean_val:.4f}")

        ckpt = out_dir / f"obstacle_projector--ep{epoch+1:03d}_checkpoint.pt"
        torch.save(projector.state_dict(), ckpt)
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            torch.save(projector.state_dict(), out_dir / "obstacle_projector--best_checkpoint.pt")
            print(f"  ↳ new best → saved")

    print(f"\nTraining done. Best val loss: {best_val_loss:.4f}")
    print(f"Copy checkpoint to model dir and restart server with OPENVLA_OBS_COND=1")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",    required=True)
    parser.add_argument("--checkpoint", default=os.environ.get("OPENVLA_MODEL_PATH", "/workspace/vla_model"))
    parser.add_argument("--unnorm-key", default=os.environ.get("OPENVLA_UNNORM_KEY", "libero_spatial_no_noops"))
    parser.add_argument("--lora-rank",  type=int, default=32)
    parser.add_argument("--epochs",     type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--out",        default="runs/obs_cond_projector")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
