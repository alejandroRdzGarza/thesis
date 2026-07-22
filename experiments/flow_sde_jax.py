"""
flow_sde_jax.py — JAX port of the flow-SDE step (HOOK A math), validated on CPU
against the tested NumPy reference experiments/flow_sde.py.

This is the exact math to drop into openpi's pi0.py::sample_actions (serving side)
and to recompute log-probs in the GRPO loss (train side). It is a line-for-line JAX
translation of flow_sde.sde_step_with_logprob (both 'sde' and 'cps' branches) +
grpo_surrogate. Run experiments/test_flow_sde_jax.py (openpi/.venv python) to confirm
numerical equivalence to the NumPy version.

Convention (matches pi0.py:271 and flow_sde.py): sigma == time t, sigma runs 1→0,
dt = sigma_prev - sigma < 0.  `v` is the model velocity (action_out_proj output).
"""

from __future__ import annotations

import math
import jax
import jax.numpy as jnp

_SIGMA_ONE_EPS = 1e-9


def sde_step_with_logprob(
    v, sigma, sigma_prev, sample, noise_level, key,
    sde_type: str = "sde", sigma_max: float = 1.0, prev_sample=None,
):
    """One flow-SDE denoising step (JAX). Returns (prev_sample, logp, mean, std).

    logp is the MEAN over action dims (matches Flow-GRPO / flow_sde.py). If prev_sample
    is provided, logp is evaluated for it (recompute path); else a new prev_sample is
    drawn from `key`. sigma/sigma_prev/noise_level are Python/array scalars.
    """
    v = jnp.asarray(v, jnp.float32)
    sample = jnp.asarray(sample, jnp.float32)
    dt = jnp.float32(sigma_prev - sigma)                     # negative

    if sde_type == "sde":
        # Guard sigma==1 exactly (denominator 1-sigma → use sigma_max), matching AEGIS.
        near_one = jnp.abs(sigma - 1.0) < _SIGMA_ONE_EPS
        denom = 1.0 - jnp.where(near_one, sigma_max, sigma)
        std_dev_t = jnp.where(denom > 0, noise_level * jnp.sqrt(sigma / denom), 0.0)
        safe_sigma = jnp.where(sigma > 0, sigma, 1.0)
        mean_full = (sample * (1.0 + std_dev_t**2 / (2.0 * safe_sigma) * dt)
                     + v * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * safe_sigma)) * dt)
        prev_mean = jnp.where(sigma > 0, mean_full, sample + v * dt)
        std = std_dev_t * jnp.sqrt(-dt)
        if prev_sample is None:
            prev_sample = prev_mean + std * jax.random.normal(key, sample.shape, sample.dtype)
        prev_sample = jnp.asarray(prev_sample, jnp.float32)
        s2 = jnp.maximum(std * std, 1e-16)
        logp = (-((prev_sample - prev_mean) ** 2) / (2.0 * s2)
                - jnp.log(jnp.maximum(std, 1e-8)) - 0.5 * jnp.log(2.0 * jnp.pi))

    elif sde_type == "cps":
        std_dev_t = sigma_prev * jnp.sin(noise_level * jnp.pi / 2.0)
        x0_hat = sample - sigma * v
        x1_hat = sample + v * (1.0 - sigma)
        prev_mean = x0_hat * (1.0 - sigma_prev) + x1_hat * jnp.sqrt(
            jnp.maximum(sigma_prev**2 - std_dev_t**2, 0.0))
        std = std_dev_t
        if prev_sample is None:
            prev_sample = prev_mean + std * jax.random.normal(key, sample.shape, sample.dtype)
        prev_sample = jnp.asarray(prev_sample, jnp.float32)
        logp = -((prev_sample - prev_mean) ** 2)            # constants dropped (cancel in ratio)
    else:
        raise ValueError(f"sde_type must be 'sde' or 'cps', got {sde_type!r}")

    # Reduce over the action dims (H, action_dim), keeping any leading batch dim:
    # per-sample logp (Flow-GRPO's "mean over non-batch dims"). For a single (H, ad)
    # chunk this is a scalar; for a (batch, H, ad) training batch it is (batch,).
    return prev_sample, jnp.mean(logp, axis=(-2, -1)), prev_mean, std


def make_sigmas(num_steps: int):
    """π0.5 sigma(=time) schedule 1→0 in num_steps steps (uniform, dt=-1/N)."""
    return jnp.linspace(1.0, 0.0, num_steps + 1)


def flow_sde_sample(velocity_fn, noise, sigmas, noise_level, key, sde_type="sde"):
    """Full flow-SDE sampling loop — the body of pi0.py::sample_actions_with_logprob.

    `velocity_fn(x_t, sigma) -> v` is the model's action-head forward pass (closes over
    the prefix KV cache in the real model). Uses jax.lax.scan over the denoising steps,
    same structure as the deterministic sampler but with the SDE step. Returns:
      action    : x_0
      chain     : (num_steps+1, *action_shape) latent trajectory
      step_logp : (num_steps,) per-step logp under the sampling policy (= logp_old)
    """
    sigma_max = sigmas[1]
    num_steps = sigmas.shape[0] - 1
    keys = jax.random.split(key, num_steps)

    def body(x, inp):
        sig, sig_prev, k = inp
        v = velocity_fn(x, sig)
        x_next, logp, _, _ = sde_step_with_logprob(
            v, sig, sig_prev, x, noise_level, k, sde_type, sigma_max)
        return x_next, (x_next, logp)

    x0, (chain_tail, step_logp) = jax.lax.scan(
        body, noise, (sigmas[:-1], sigmas[1:], keys))
    chain = jnp.concatenate([noise[None], chain_tail], axis=0)
    return {"action": x0, "chain": chain, "step_logp": step_logp}


def flow_sde_recompute_logp(velocity_fn, chain, sigmas, noise_level, sde_type="sde"):
    """Per-step logp of a RECORDED chain under the current policy (for the GRPO ratio).

    With velocity_fn = sampling policy this exactly reproduces step_logp (ratio ≡ 1).
    This is what the training loss calls each update.
    """
    sigma_max = sigmas[1]

    def body(_, inp):
        sig, sig_prev, x, x_next = inp
        v = velocity_fn(x, sig)
        _, logp, _, _ = sde_step_with_logprob(
            v, sig, sig_prev, x, noise_level, None, sde_type, sigma_max, prev_sample=x_next)
        return None, logp

    _, step_logp = jax.lax.scan(
        body, None, (sigmas[:-1], sigmas[1:], chain[:-1], chain[1:]))
    return step_logp


def grpo_surrogate(logp_new, logp_old, advantage, clip: float = 0.2):
    """PPO/GRPO clipped surrogate (JAX; MAXIMISE → loss = -mean(surrogate))."""
    ratio = jnp.exp(logp_new - logp_old)
    return jnp.minimum(ratio * advantage,
                       jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage)
