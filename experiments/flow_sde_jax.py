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

    return prev_sample, jnp.mean(logp), prev_mean, std


def grpo_surrogate(logp_new, logp_old, advantage, clip: float = 0.2):
    """PPO/GRPO clipped surrogate (JAX; MAXIMISE → loss = -mean(surrogate))."""
    ratio = jnp.exp(logp_new - logp_old)
    return jnp.minimum(ratio * advantage,
                       jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * advantage)
