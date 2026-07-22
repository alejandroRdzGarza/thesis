"""
flow_sde.py — Flow-SDE for π0.5, FAITHFUL to Flow-GRPO (no simplifications).

Direct NumPy port of Flow-GRPO's `sde_step_with_logprob`
(github.com/yifan123/flow_grpo, flow_grpo/diffusers_patch/sd3_sde_with_logprob.py;
Liu et al. 2025, arXiv:2505.05470). Both the marginal-preserving `sde` branch and the
newer `cps` (Coefficients-Preserving Sampling) branch are reproduced exactly, adapted to
π0.5's convention.

WHY: π0.5 samples actions by the deterministic flow ODE
     x_{t+dt} = x_t + dt·v_θ(x_t,t)   (dt<0, time 1→0; openpi pi0.py:271)
which has no action log-prob → no policy gradient. Flow-GRPO relaxes each denoising step
into a Gaussian whose mean is a marginal-preserving SDE drift, giving a closed-form
per-step log-prob → real on-policy GRPO/PPO. As noise_level→0 both branches recover the
exact ODE (verified in tests).

CONVENTION MAPPING (Flow-GRPO SD3 → π0.5):
  Flow-GRPO parameterises by `sigma` (the FlowMatchEuler noise level). For π0.5's linear
  flow x_t = t·noise + (1-t)·action, sigma == t. So here `sigma`=current time,
  `sigma_prev`=next time (=t+dt, dt<0), `model_output`=v_θ. sigma runs 1→0.

The `sde` branch (Flow-GRPO §3.1, marginal-preserving):
    std_dev_t        = noise_level · sqrt( sigma / (1 - sigma) )        # sigma=1 guarded
    prev_sample_mean = sample·(1 + std²/(2σ)·dt) + v·(1 + std²(1-σ)/(2σ))·dt
                     = [sample + v·dt]  +  (std²·dt/(2σ))·(sample + (1-σ)·v)   # ODE + score drift
    prev_sample      = prev_sample_mean + std_dev_t·sqrt(-dt)·ε
    logp (per-elem)  = -(x-mean)²/(2·s²) - log s - ½log2π,   s = std_dev_t·sqrt(-dt)

The `cps` branch (README-recommended, noise_level≈0.8):
    std_dev_t        = sigma_prev · sin(noise_level·π/2)
    x0_hat           = sample - sigma·v ;  x1_hat = sample + (1-sigma)·v
    prev_sample_mean = x0_hat·(1-sigma_prev) + x1_hat·sqrt(sigma_prev² - std²)
    prev_sample      = prev_sample_mean + std_dev_t·ε
    logp             = -(x-mean)²          # constants dropped; std is policy-independent → cancels in ratio

Log-prob is reduced by MEAN over action dims (matching Flow-GRPO's `.mean(dim=...)`),
not sum — a deliberate choice that stabilises the clip range across dimensionalities.

This module is the tested reference for the RL machinery + the exact SDE math the openpi
JAX port (FLOW_SDE_OPENPI.md) must mirror.
"""

from __future__ import annotations

import numpy as np

_SIGMA_ONE_EPS = 1e-9


def make_sigmas(num_steps: int) -> np.ndarray:
    """π0.5 sigma(=time) schedule, 1.0 → 0.0 in num_steps steps (uniform, dt=-1/N)."""
    return np.linspace(1.0, 0.0, num_steps + 1)


def sde_step_with_logprob(
    v: np.ndarray, sigma: float, sigma_prev: float, sample: np.ndarray,
    noise_level: float, sde_type: str = "sde", sigma_max: float = 1.0,
    prev_sample: np.ndarray | None = None, rng: np.random.Generator | None = None,
):
    """One faithful Flow-GRPO SDE denoising step. Returns (prev_sample, logp, mean, std).

    If prev_sample is given, logp is evaluated for it (recompute path); otherwise a new
    prev_sample is drawn from `rng`. logp is the mean over action dims (per Flow-GRPO).
    """
    v = np.asarray(v, dtype=np.float64)
    sample = np.asarray(sample, dtype=np.float64)
    dt = float(sigma_prev - sigma)                       # negative

    if sde_type == "sde":
        denom = 1.0 - (sigma_max if abs(sigma - 1.0) < _SIGMA_ONE_EPS else sigma)
        std_dev_t = noise_level * np.sqrt(sigma / denom) if denom > 0 else 0.0
        prev_mean = (sample * (1.0 + std_dev_t**2 / (2.0 * sigma) * dt)
                     + v * (1.0 + std_dev_t**2 * (1.0 - sigma) / (2.0 * sigma)) * dt) \
                    if sigma > 0 else sample + v * dt
        std = std_dev_t * np.sqrt(-dt)
        if prev_sample is None:
            prev_sample = prev_mean + std * rng.standard_normal(sample.shape)
        prev_sample = np.asarray(prev_sample, dtype=np.float64)
        s2 = max(std * std, 1e-16)
        logp = (-((prev_sample - prev_mean) ** 2) / (2.0 * s2)
                - np.log(max(std, 1e-8)) - 0.5 * np.log(2.0 * np.pi))

    elif sde_type == "cps":
        std_dev_t = sigma_prev * np.sin(noise_level * np.pi / 2.0)
        x0_hat = sample - sigma * v
        x1_hat = sample + v * (1.0 - sigma)
        prev_mean = x0_hat * (1.0 - sigma_prev) + x1_hat * np.sqrt(
            max(sigma_prev**2 - std_dev_t**2, 0.0))
        std = std_dev_t
        if prev_sample is None:
            prev_sample = prev_mean + std * rng.standard_normal(sample.shape)
        prev_sample = np.asarray(prev_sample, dtype=np.float64)
        logp = -((prev_sample - prev_mean) ** 2)         # constants dropped (cancel in ratio)
    else:
        raise ValueError(f"sde_type must be 'sde' or 'cps', got {sde_type!r}")

    return prev_sample, float(np.mean(logp)), prev_mean, std


def flow_sde_sample(v_fn, x_init: np.ndarray, sigmas: np.ndarray,
                    noise_level: float, rng: np.random.Generator,
                    sde_type: str = "sde") -> dict:
    """Sample one action chunk via the faithful flow-SDE; record chain + per-step logp.

    v_fn(x, sigma) -> velocity. Returns action (x_0), the latent chain, per-step logp
    (= logp_old under the sampling policy), total_logp, and the schedule/config.
    """
    sigma_max = float(sigmas[1]) if len(sigmas) > 1 else 1.0
    x = np.asarray(x_init, dtype=np.float64)
    chain = [x.copy()]
    step_logp = []
    for k in range(len(sigmas) - 1):
        sig, sig_prev = float(sigmas[k]), float(sigmas[k + 1])
        v = v_fn(x, sig)
        x_next, lp, _, _ = sde_step_with_logprob(
            v, sig, sig_prev, x, noise_level, sde_type, sigma_max, None, rng)
        step_logp.append(lp)
        chain.append(x_next.copy())
        x = x_next
    step_logp = np.array(step_logp)
    return {
        "action": x, "chain": chain, "step_logp": step_logp,
        "total_logp": float(np.sum(step_logp)), "sigmas": sigmas,
        "noise_level": noise_level, "sde_type": sde_type,
    }


def flow_sde_recompute_logp(v_fn, chain, sigmas: np.ndarray, noise_level: float,
                            sde_type: str = "sde") -> np.ndarray:
    """Per-step logp of a RECORDED chain under a (possibly updated) v_fn.

    With v_fn = sampling policy this exactly reproduces step_logp; with the current
    policy it gives logp_new for the importance ratio. Same sde_type as the sampler.
    """
    sigma_max = float(sigmas[1]) if len(sigmas) > 1 else 1.0
    step_logp = []
    for k in range(len(sigmas) - 1):
        sig, sig_prev = float(sigmas[k]), float(sigmas[k + 1])
        x, x_next = chain[k], chain[k + 1]
        v = v_fn(x, sig)
        _, lp, _, _ = sde_step_with_logprob(
            v, sig, sig_prev, x, noise_level, sde_type, sigma_max, x_next, None)
        step_logp.append(lp)
    return np.array(step_logp)


def grpo_surrogate(logp_new, logp_old, advantage, clip: float = 0.2):
    """PPO/GRPO clipped surrogate (MAXIMISE; loss = -mean(surrogate)).

    ratio = exp(logp_new - logp_old); returns min(ratio·A, clip(ratio,1±ε)·A).
    Works per-step or on summed chain log-probs (broadcast the scalar advantage).
    """
    logp_new = np.asarray(logp_new, dtype=np.float64)
    logp_old = np.asarray(logp_old, dtype=np.float64)
    adv = np.asarray(advantage, dtype=np.float64)
    ratio = np.exp(logp_new - logp_old)
    return np.minimum(ratio * adv, np.clip(ratio, 1.0 - clip, 1.0 + clip) * adv)
