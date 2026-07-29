"""flow_bc.py — supervised flow-matching BC step for shield-as-expert DAgger (thesis Exp 005).

Trains ONLY the trainable (LoRA) params to imitate the CBF-corrected ("shielded") action via
Pi0.compute_loss — the model's native flow-matching objective. Plain behavior cloning: no
advantages, no surrogate. Because the target action *does the task safely*, this structurally
cannot reward-hack to inaction the way the scalar-reward GRPO runs (Exp 002-004) did.

Mirrors flow_grpo.grpo_train_step's JIT + optax pattern (cached grad fn; first call compiles).
"""

from __future__ import annotations

import flax.nnx as nnx
import jax.numpy as jnp
import optax

_GRAD_FN_CACHE: dict = {}


def _make_grad_fn(trainable_filter):
    """JIT-compiled (loss, grads) for the flow-matching BC loss over trainable (LoRA) params."""

    @nnx.jit
    def _loss_and_grad(model, rng, observation, actions):
        def loss_fn(m):
            # compute_loss returns per-(batch, action_horizon) flow-matching MSE.
            return jnp.mean(m.compute_loss(rng, observation, actions, train=True))
        diff_state = nnx.DiffState(0, trainable_filter)
        return nnx.value_and_grad(loss_fn, argnums=diff_state)(model)

    return _loss_and_grad


def bc_train_step(model, tx: optax.GradientTransformation, opt_state, batch: tuple, *,
                  trainable_filter):
    """One BC update over the trainable (LoRA) params. Updates `model` in place.

    `batch` = (rng, observation, actions) where `actions` is the NORMALIZED shielded action
    chunk in model space, shape (b, action_horizon, action_dim). Returns (new_opt_state, info).
    """
    rng, observation, actions = batch

    key = id(trainable_filter)
    grad_fn = _GRAD_FN_CACHE.get(key)
    if grad_fn is None:
        grad_fn = _make_grad_fn(trainable_filter)
        _GRAD_FN_CACHE[key] = grad_fn
    loss, grads = grad_fn(model, rng, observation, actions)

    params = nnx.state(model, trainable_filter)
    updates, new_opt_state = tx.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    nnx.update(model, new_params)

    info = {"loss": loss, "grad_norm": optax.global_norm(grads)}
    return new_opt_state, info
