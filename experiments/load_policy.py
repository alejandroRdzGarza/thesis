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

    import openpi.transforms as _transforms
    from openpi.models import model as _model
    from openpi.policies import policy as _policy
    from openpi.shared import download
    from openpi.training import checkpoints as _checkpoints
    from openpi.training import weight_loaders

    checkpoint_dir = pathlib.Path(download.maybe_download(str(checkpoint_dir)))

    # Build the (possibly LoRA) model, then merge checkpoint weights, filling any params the
    # checkpoint lacks (the LoRA ones) from the fresh init — openpi CheckpointWeightLoader.
    model = train_cfg.model.create(jax.random.key(0))
    graphdef, state = nnx.split(model)
    merged = weight_loaders.CheckpointWeightLoader(
        str(checkpoint_dir / "params")
    ).load(state.to_pure_dict())
    state.replace_by_pure_dict(merged)
    model = nnx.merge(graphdef, state)

    # Transforms — identical to create_trained_policy (norm stats from the checkpoint assets).
    data_config = train_cfg.data.create(train_cfg.assets_dirs, train_cfg.model)
    if data_config.asset_id is None:
        raise ValueError("Asset id is required to load norm stats.")
    norm_stats = _checkpoints.load_norm_stats(checkpoint_dir / "assets", data_config.asset_id)

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
