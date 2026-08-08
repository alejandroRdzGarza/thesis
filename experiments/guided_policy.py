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

        velocity_fn = self._model._build_flow_velocity_fn(observation)
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
    """With lam=0 the Python unroll must match the stock jitted sampler. Gate every guided run.

    Checked at noise_level=0, where the SDE term vanishes and sampling is deterministic, so the
    comparison is exact and no RNG sequence has to be reproduced. A mismatch means the unroll
    differs from lax.scan, and every guided number downstream would be measuring that difference
    rather than the guidance.
    """
    from openpi.policies.policy_logprob import PolicyWithLogprob

    ref = PolicyWithLogprob(policy, num_steps=num_steps, noise_level=0.0, sde_type="cps", seed=0)
    a_ref = np.asarray(ref.infer_with_logprob(obs)["actions"])

    gp = GuidedPolicy(policy, num_steps=num_steps, noise_level=0.0, sde_type="cps", seed=0,
                      guidance_source=None, lam=0.0)
    a_new = np.asarray(gp.infer_with_logprob(obs)["actions"])

    if a_ref.shape != a_new.shape:
        print(f"  FAIL shape {a_ref.shape} vs {a_new.shape}")
        return False
    d = float(np.max(np.abs(a_ref - a_new)))
    ok = d <= tol
    print(f"  lam=0 vs stock sampler: max|delta| = {d:.2e}  -> {'MATCH' if ok else 'MISMATCH'}")
    if not ok:
        print("    Do NOT trust guided results. The Python unroll differs from the jitted scan;")
        print("    fix that first or every guided number measures the discrepancy, not guidance.")
    return ok
