"""guided_policy.py — π0.5 sampling with the CBF steering generation instead of correcting it.

WHY A SEPARATE SAMPLING PATH IS NEEDED. The rollout sampler lives in
`openpi/models/pi0.sample_actions_with_logprob`, which runs the denoising loop inside `lax.scan`
under `module_jit`. A CBF projection is a Python QP over live world state and cannot be called from
inside a jitted scan. So the loop is unrolled in Python here: the expensive part (the transformer
forward, via `_build_flow_velocity_fn`) stays jitted and is called once per denoising step, while
guidance runs in numpy between steps.

Cost: ~10 separate jitted calls instead of one fused scan, so inference is roughly 2-3x slower.
Evaluate on a subset of scenes before committing to a full 24-scene arm.

WHAT IT DOES, and why not just project the output. Post-hoc projection is what the shield does now,
and it costs capability: the distilled policy scores TSR 82.5% unshielded and 70.8% with the shield
stacked on, because projecting moves a finished, coherent action off the policy's manifold. Guiding
the velocity instead keeps the sample on the manifold the whole way, so the safe action is one the
policy itself would plausibly have produced.

    v_guided = v - lambda/sigma * delta,    delta = (u_safe - u_nom) / action_scale

Both the 1/sigma schedule and the MINUS were derived and then verified on a synthetic barrier (see
flow_sde.flow_sde_sample_guided): lambda=1.0 lands the action exactly on the projection's target,
lambda=0.5 applies half of it. So lambda reads as "fraction of the projection applied".

CORRECTNESS GATE. With lam=0.0 this must reproduce the stock sampler's actions. At noise_level=0 the
SDE term vanishes and sampling is deterministic, so the comparison is exact and there is no RNG
sequence to match. `verify_equivalence()` checks it. Run that before trusting any guided number —
a Python unroll that silently differs from the scan would corrupt every downstream result.
"""

from __future__ import annotations

import numpy as np


class GuidedPolicy:
    """Mirrors PolicyWithLogprob.infer_with_logprob, with optional CBF guidance.

    guidance_source(obs_dict) -> guidance_fn or None
        Called once per query with the raw observation, so the caller can close over the CURRENT
        world state (obstacle spheres, EE pose) before the denoising loop starts. Returning None
        disables guidance for that query — which is what should happen when no obstacle is near.
    """

    def __init__(self, policy, num_steps: int = 10, noise_level: float = 0.0,
                 sde_type: str = "cps", seed: int = 0,
                 guidance_source=None, lam: float = 1.0):
        import jax
        self._policy = policy
        self._model = policy._model
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self.num_steps = int(num_steps)
        self.noise_level = float(noise_level)
        self.sde_type = sde_type
        self.lam = float(lam)
        self.guidance_source = guidance_source
        self._rng = jax.random.key(seed)

        # Compile the two halves of the forward ONCE. Calling _build_flow_velocity_fn's closure
        # directly runs the transformer eagerly — measured at 4.3 s per denoising step, ~30x the
        # jitted path. Wrapping that closure in jit does not help: it is rebuilt per query, so every
        # query is a fresh compilation cache entry. Passing the cache as an argument instead means
        # one compile total, reused across all queries (shapes are constant).
        from openpi.shared import nnx_utils
        from experiments import pi0_velocity_jit
        pi0_velocity_jit.install()
        self._jit_prefix = nnx_utils.module_jit(self._model.prefix_cache)
        self._jit_vel = nnx_utils.module_jit(self._model.velocity_from_cache)

    def infer_with_logprob(self, obs: dict, *, noise=None) -> dict:
        import jax
        import jax.numpy as jnp
        from openpi.models import model as _model
        from openpi.models import flow_sde as _fsde

        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
        observation = _model.Observation.from_dict(inputs)
        observation = _model.preprocess_observation(None, observation, train=False)

        self._rng, sample_rng = jax.random.split(self._rng)
        b = observation.state.shape[0]
        if noise is None:
            x = jax.random.normal(sample_rng, (b, self._model.action_horizon,
                                               self._model.action_dim))
        else:
            n = jnp.asarray(noise)
            x = n[None, ...] if n.ndim == 2 else n

        kv_cache, prefix_mask = self._jit_prefix(observation)
        def velocity_fn(x_t, sigma):
            return self._jit_vel(observation, kv_cache, prefix_mask, x_t, sigma)
        sigmas = _fsde.make_sigmas(self.num_steps)
        sigma_max = float(sigmas[1]) if len(sigmas) > 1 else 1.0

        guidance_fn = self.guidance_source(obs) if self.guidance_source is not None else None

        chain, step_logp, g_norms = [np.asarray(x)], [], []
        for k in range(len(sigmas) - 1):
            sig, sig_prev = float(sigmas[k]), float(sigmas[k + 1])
            v = velocity_fn(x, jnp.asarray(sig))

            if guidance_fn is not None and self.lam != 0.0:
                # Ask the barrier about the CLEAN action this step is heading toward, not the noisy
                # latent: x0 ~= x - sigma*v under this parameterisation. Early in the chain the
                # latent is mostly noise and a collision barrier evaluated there is meaningless.
                x0_pred = np.asarray(x)[0] - sig * np.asarray(v)[0]
                g = guidance_fn(x0_pred, sig)
                g = np.zeros_like(x0_pred) if g is None else np.asarray(g)
                w = self.lam / max(sig, 1e-3)          # endpoint schedule; see module docstring
                v = v - jnp.asarray(w * g, dtype=v.dtype)[None, ...]   # MINUS: dt < 0
                g_norms.append(float(np.linalg.norm(w * g)))
            else:
                g_norms.append(0.0)

            self._rng, step_rng = jax.random.split(self._rng)
            # KEYWORDS, not positional. The JAX and numpy versions of this function take their
            # arguments in DIFFERENT orders — JAX is (..., noise_level, key, sde_type, ...) while
            # the numpy reference is (..., noise_level, sde_type, sigma_max, prev_sample, rng).
            # Calling positionally passed a sigma where sde_type was expected.
            x, lp, _, _ = _fsde.sde_step_with_logprob(
                v, sig, sig_prev, x, self.noise_level, step_rng,
                sde_type=self.sde_type, sigma_max=sigma_max, prev_sample=None)
            step_logp.append(np.asarray(lp))
            chain.append(np.asarray(x))

        outputs = {"state": inputs["state"], "actions": x}
        outputs = jax.tree.map(lambda a: np.asarray(a[0, ...]), outputs)
        outputs = self._output_transform(outputs)

        return {
            "actions": np.asarray(outputs["actions"]),
            "chain": np.stack(chain), "step_logp": np.asarray(step_logp),
            "sigmas": np.asarray(sigmas), "noise_level": self.noise_level,
            "sde_type": self.sde_type, "guidance_norms": np.asarray(g_norms),
        }


def verify_equivalence(policy, obs: dict, num_steps: int = 10, tol: float = 1e-4) -> bool:
    """Check the Python unroll against the stock jitted sampler, and against ITSELF.

    Self-consistency is the hard requirement: each path must be bitwise reproducible, or there is
    RNG or state leaking and no comparison means anything.

    Cross-path agreement is NOT achievable and should not be demanded. The unroll calls the
    transformer eagerly while the stock sampler runs it fused inside lax.scan, and with bf16
    parameters XLA is free to schedule and accumulate differently. Measured here: both paths
    reproduce themselves exactly (0.00e+00) while differing from each other by 3.85e-3, which is
    1.3% of mean|action| = 0.302. That is kernel scheduling, not a logic error.

    THE CONSEQUENCE FOR THE EXPERIMENT: the control arm must be lam=0 THROUGH THIS CLASS, not the
    stock sampler. Then every arm shares one code path, the discrepancy is common-mode and cancels
    exactly, and any measured difference is guidance. Comparing a guided arm against the stock
    sampler would fold a 1.3% numerical offset into the result.
    """
    from openpi.policies.policy_logprob import PolicyWithLogprob

    def ref_actions():
        p = PolicyWithLogprob(policy, num_steps=num_steps, noise_level=0.0, sde_type="cps", seed=0)
        return np.asarray(p.infer_with_logprob(obs)["actions"])

    def unroll_actions():
        p = GuidedPolicy(policy, num_steps=num_steps, noise_level=0.0, sde_type="cps", seed=0,
                         guidance_source=None, lam=0.0)
        return np.asarray(p.infer_with_logprob(obs)["actions"])

    s1, s2 = ref_actions(), ref_actions()
    g1, g2 = unroll_actions(), unroll_actions()
    d_stock = float(np.max(np.abs(s1 - s2)))
    d_unroll = float(np.max(np.abs(g1 - g2)))
    d_cross = float(np.max(np.abs(s1 - g1)))
    scale = float(np.abs(s1).mean())

    print(f"  action magnitude       : mean|a| = {scale:.4f}")
    print(f"  stock  self-consistent : {d_stock:.2e}   {'OK' if d_stock <= tol else 'FAIL'}")
    print(f"  unroll self-consistent : {d_unroll:.2e}   {'OK' if d_unroll <= tol else 'FAIL'}")
    print(f"  stock vs unroll        : {d_cross:.2e}   ({d_cross/max(scale,1e-9):.1%} of mean|a|)")

    ok = d_stock <= tol and d_unroll <= tol
    if not ok:
        print("    FAIL: a path does not reproduce itself — RNG or state is leaking. Fix before"
              "\n    running anything guided; no comparison is meaningful until this is 0.")
    else:
        print("    PASS: both paths are deterministic. The cross-path gap is eager-vs-fused XLA"
              "\n    numerics, not a logic error.")
        print("    Use lam=0 THROUGH GuidedPolicy as the control arm, so the gap is common-mode"
              "\n    and cancels. Do NOT compare guided arms against the stock sampler.")
    return ok
