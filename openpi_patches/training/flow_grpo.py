"""flow_grpo.py — flow-SDE GRPO training loss + step for π0.5 (closes the RL loop).

Given a batch of recorded rollout queries (observation, denoising chain, logp_old,
group advantage), this recomputes logp_new under the current policy (Pi0.compute_chain_logp,
HOOK C), forms the Flow-GRPO clipped surrogate, and takes a single optimizer step over the
TRAINABLE (LoRA) params only — the VLM backbone stays frozen via the config's
trainable_filter (= All(Param, Not(freeze_filter))), matching the teacher's guidance to
fine-tune only the action head.

Batch convention (step axis leading, batch axis second — matches flow_sde_sample output):
  observation : Observation, batch size B
  chain       : (num_steps+1, B, action_horizon, action_dim)  model-native latents
  logp_old    : (num_steps, B)   per-step logp under the sampling policy
  advantages  : (B,)             GRPO group-relative advantage per query

Sanity (on-policy at init): current policy == sampling policy ⇒ logp_new == logp_old ⇒
ratio ≡ 1 ⇒ surrogate == advantage, so loss == -mean(advantages).
"""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import optax

from openpi.models import flow_sde as _fsde
from openpi.models import pi0 as _pi0
from openpi.shared import array_typing as at


@at.typecheck
def flow_grpo_loss(
    model: _pi0.Pi0,
    observation,
    chain: at.Float[at.Array, "s b ah ad"],
    logp_old: at.Float[at.Array, "num_steps b"],
    advantages: at.Float[at.Array, "b"],
    *,
    noise_level: float = 0.7,
    sde_type: str = "cps",
    clip: float = 0.2,
) -> at.Float[at.Array, ""]:
    """Flow-GRPO clipped-surrogate loss (scalar, already negated for minimisation)."""
    logp_new = model.compute_chain_logp(
        observation, chain, noise_level=noise_level, sde_type=sde_type)   # (num_steps, B)
    adv = advantages[None, :]                                             # (1, B) broadcast over steps
    surrogate = _fsde.grpo_surrogate(logp_new, logp_old, adv, clip)       # (num_steps, B)
    return -jnp.mean(surrogate)


def grpo_train_step(
    model: _pi0.Pi0,
    tx: optax.GradientTransformation,
    opt_state,
    batch: tuple,
    *,
    trainable_filter,
    noise_level: float = 0.7,
    sde_type: str = "cps",
    clip: float = 0.2,
):
    """One GRPO update over the trainable (LoRA) params. Updates `model` in place.

    `batch` = (observation, chain, logp_old, advantages). `trainable_filter` selects the
    params that receive gradients (config.trainable_filter). Returns (new_opt_state, info).
    Mirrors openpi's train_step: grads only on the DiffState-filtered params, optax update,
    nnx.update in place.
    """
    observation, chain, logp_old, advantages = batch

    def loss_fn(m: _pi0.Pi0):
        return flow_grpo_loss(m, observation, chain, logp_old, advantages,
                              noise_level=noise_level, sde_type=sde_type, clip=clip)

    diff_state = nnx.DiffState(0, trainable_filter)
    loss, grads = nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    params = nnx.state(model, trainable_filter)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    info = {
        "loss": loss,
        "grad_norm": optax.global_norm(grads),
        "mean_advantage": jnp.mean(advantages),
    }
    return new_opt_state, info
