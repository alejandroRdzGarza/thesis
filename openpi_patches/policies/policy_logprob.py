"""PolicyWithLogprob — flow-SDE sampling wrapper for on-policy GRPO rollouts (HOOK B).

Wraps a server-built openpi Policy (see create_trained_policy) and, instead of the
deterministic ODE sampler, draws actions with Pi0.sample_actions_with_logprob so every
query yields the denoising CHAIN + per-step logp under the sampling policy (logp_old).

Two action spaces, kept deliberately separate:
  • The env-ready action is produced by the SAME output transform the server uses
    (unnormalize + un-pad to the 7-D env action) — so the shielded rollout behaves
    exactly like a served π0.5 rollout.
  • The chain and logp stay in the model's NATIVE (normalized, action_dim-padded) space
    — that is the space the model samples in, so flow_sde.flow_sde_recompute_logp
    reproduces logp_old exactly at training time (on-policy ratio ≡ 1). Do NOT
    output-transform the chain.

JAX / Pi0 flow-matching only. Reuses the wrapped policy's input/output transforms and
forks its own RNG stream (does not disturb policy.infer's stream).
"""

from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np

from openpi.models import flow_sde as _fsde
from openpi.models import model as _model
from openpi.policies import policy as _policy
from openpi.shared import nnx_utils


class PolicyWithLogprob:
    """Flow-SDE (stochastic) inference wrapper around a trained JAX Policy."""

    def __init__(
        self,
        policy: _policy.Policy,
        *,
        num_steps: int = 10,
        noise_level: float = 0.7,
        sde_type: str = "cps",
        seed: int = 0,
    ):
        if getattr(policy, "_is_pytorch_model", False):
            raise NotImplementedError("PolicyWithLogprob supports JAX Pi0 models only.")
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self.num_steps = int(num_steps)
        self.noise_level = float(noise_level)
        self.sde_type = str(sde_type)
        self._rng = jax.random.key(seed)
        # Compiled flow-SDE sampler. num_steps/sde_type are static (loop length is
        # built from a Python int; sde_type selects a Python branch).
        self._sample_lp = nnx_utils.module_jit(
            policy._model.sample_actions_with_logprob,
            static_argnames=("num_steps", "sde_type"),
        )
        # sigma grid is constant for a given num_steps.
        self._sigmas = np.asarray(_fsde.make_sigmas(self.num_steps))

    def infer_with_logprob(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:
        """Sample one action chunk stochastically and return it + its policy trace.

        Returns a dict compatible with policy_trace.from_flow_sde_roll:
          actions     : (action_horizon, action_dim_env)   env-ready (unnormalized)
          chain       : (num_steps+1, action_horizon, action_dim)  model-space latents
          step_logp   : (num_steps,)                        logp under the sampling policy
          sigmas      : (num_steps+1,)
          noise_level : float
          sde_type    : "sde" | "cps"
        """
        inputs = jax.tree.map(lambda x: x, obs)          # shallow copy (transforms mutate)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)  # add batch
        self._rng, sample_rng = jax.random.split(self._rng)

        n = None
        if noise is not None:
            n = jnp.asarray(noise)
            if n.ndim == 2:
                n = n[None, ...]

        observation = _model.Observation.from_dict(inputs)
        t0 = time.monotonic()
        action, chain, step_logp = self._sample_lp(
            sample_rng, observation,
            num_steps=self.num_steps, noise_level=self.noise_level,
            sde_type=self.sde_type, noise=n,
        )
        infer_ms = (time.monotonic() - t0) * 1000

        # Env-ready action: unbatch then apply the server output transform (action only).
        outputs = {"state": inputs["state"], "actions": action}
        outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)
        outputs = self._output_transform(outputs)

        return {
            "actions": np.asarray(outputs["actions"]),   # (ah, ad_env) env-ready
            "chain": np.asarray(chain[:, 0]),            # (S+1, ah, ad) model space
            "step_logp": np.asarray(step_logp[:, 0]),    # (S,)
            "sigmas": self._sigmas,
            "noise_level": self.noise_level,
            "sde_type": self.sde_type,
            "policy_timing": {"infer_ms": infer_ms},
        }
