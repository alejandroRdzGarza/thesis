"""
load_policy.py — build an openpi Policy that loads a checkpoint into a (LoRA) model via
PARTIAL merge, so the same code path works for both:
  • round 0: a BASE checkpoint (pi05_libero, no LoRA params) loaded into the LoRA config
    (pi05_libero_cbf) — the missing lora_a/lora_b are filled from the fresh init (lora_b=0,
    so behaviorally identical to the base model), and
  • round N: a saved LoRA checkpoint (all params present) loaded strictly.

create_trained_policy uses model.load(), which does a STRICT pytree-equality check and thus
fails to load a base checkpoint into a LoRA model. This mirrors openpi's own training weight
load (init_train_state → CheckpointWeightLoader, missing_regex=".*lora.*") while reusing the
exact transform stack create_trained_policy builds.
"""

from __future__ import annotations

import pathlib


def create_policy_partial(train_cfg, checkpoint_dir, *, default_prompt: str | None = None):
    """Like create_trained_policy but with a base→LoRA-safe partial weight load."""
    import flax.nnx as nnx
    import jax
    import numpy as np

    import openpi.transforms as _transforms
    from openpi.models import model as _model
    from openpi.policies import policy as _policy
    from openpi.shared import download
    from openpi.shared import normalize as _normalize
    from openpi.training import weight_loaders

    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # A round checkpoint from flow_grpo_train stores ONLY the trained LoRA adapter (the frozen
    # backbone never changes, so re-saving it every round is wasteful and OOMs the host). Such a
    # dir has `lora_params/` + `base.txt` pointing at the original base (backbone + norm stats).
    # A base/full checkpoint has `params/` instead. Resolve where the backbone comes from:
    lora_dir = checkpoint_dir / "lora_params"
    is_lora_ckpt = lora_dir.exists()
    if is_lora_ckpt:
        base_dir = pathlib.Path(download.maybe_download((checkpoint_dir / "base.txt").read_text().strip()))
    else:
        base_dir = checkpoint_dir

    # Build the (LoRA) model and merge the BASE checkpoint weights, filling missing LoRA params
    # from the fresh init (openpi CheckpointWeightLoader, missing_regex=".*lora.*").
    print("  [load] building model structure (~1 min) ...", flush=True)
    model = train_cfg.model.create(jax.random.key(0))
    print("  [load] merging base checkpoint weights ...", flush=True)
    graphdef, state = nnx.split(model)
    merged = weight_loaders.CheckpointWeightLoader(
        str(base_dir / "params")
    ).load(state.to_pure_dict())
    state.replace_by_pure_dict(merged)
    if is_lora_ckpt:
        print("  [load] overlaying trained LoRA adapter ...", flush=True)
        lora = _model.restore_params(lora_dir, restore_type=np.ndarray)   # subset: LoRA keys only
        state.replace_by_pure_dict(lora)
    model = nnx.merge(graphdef, state)

    # Norm stats belong to the base π0.5 CHECKPOINT (the LoRA config's data asset_id is a
    # placeholder). Locate norm_stats.json in the BASE assets directly.
    print("  [load] loading norm stats ...", flush=True)
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    _ns_matches = list((base_dir / "assets").rglob("norm_stats.json"))
    if not _ns_matches:
        raise FileNotFoundError(f"No norm_stats.json under {base_dir / 'assets'}")
    norm_stats = _normalize.load(_ns_matches[0].parent)
    print("  [load] policy ready.", flush=True)

    return _policy.Policy(
        model,
        transforms=[
            _transforms.InjectDefaultPrompt(default_prompt),
            *data_config.data_transforms.inputs,
            _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.model_transforms.inputs,
        ],
        output_transforms=[
            *data_config.model_transforms.outputs,
            _transforms.Unnormalize(norm_stats, use_quantiles=data_config.use_quantile_norm),
            *data_config.data_transforms.outputs,
        ],
        metadata=train_cfg.policy_metadata,
    )
