"""pi0_velocity_jit.py — make the Python-unrolled denoising loop fast enough to run.

THE PROBLEM. `pi0._build_flow_velocity_fn(observation)` returns a CLOSURE over the KV cache. The
stock sampler calls it inside `lax.scan` under `module_jit`, so the transformer forward is compiled.
The guided sampler must unroll the loop in Python (a CBF QP cannot run inside a jitted scan), and
calling that closure directly runs the forward EAGERLY: measured at 4.3 s per denoising step, i.e.
2603 s for a single 300-step episode — about 30x slower than the jitted path, which makes the
experiment infeasible.

Wrapping the closure in jax.jit does not help: it is rebuilt per query, so each new function object
is a fresh compilation cache entry and every query pays a compile.

THE FIX. Split the forward into two methods that take everything as ARGUMENTS rather than closing
over it, so `module_jit` compiles each exactly once and every subsequent call reuses it (shapes are
constant across queries):

    prefix_cache(observation)                        -> kv_cache, prefix_mask     1x per query
    velocity_from_cache(observation, kv_cache, prefix_mask, x_t, time) -> v      10x per query

Attached by monkey-patch rather than edited into pi0.py, because the openpi checkout is git-ignored
and not backed up — a change there would vanish on the next pod and silently take the speed fix
with it.

Correctness is unchanged: both methods are the two halves of _build_flow_velocity_fn's body,
verbatim. `verify_against_closure()` checks them against the original.
"""

from __future__ import annotations


def install():
    """Attach prefix_cache / velocity_from_cache to Pi0. Idempotent."""
    import einops
    import jax.numpy as jnp
    from openpi.models.pi0 import Pi0, make_attn_mask

    if hasattr(Pi0, "velocity_from_cache"):
        return Pi0

    def prefix_cache(self, observation):
        """Prefix forward pass — the expensive half, done once per query."""
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask,
                                         positions=positions)
        return kv_cache, prefix_mask

    def velocity_from_cache(self, observation, kv_cache, prefix_mask, x_t, time):
        """Suffix forward pass — the per-denoising-step half. Body identical to the closure's."""
        batch_size = observation.state.shape[0]
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
            observation, x_t, jnp.broadcast_to(time, batch_size))
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        prefix_attn_mask_s = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate([prefix_attn_mask_s, suffix_attn_mask], axis=-1)
        pos = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [None, suffix_tokens], mask=full_attn_mask, positions=pos,
            kv_cache=kv_cache, adarms_cond=[None, adarms_cond])
        assert prefix_out is None
        return self.action_out_proj(suffix_out[:, -self.action_horizon:])

    Pi0.prefix_cache = prefix_cache
    Pi0.velocity_from_cache = velocity_from_cache
    return Pi0


def verify_against_closure(model, observation, x_t, sigma, tol: float = 1e-5) -> bool:
    """The split methods must reproduce _build_flow_velocity_fn exactly.

    They are the same body, but a transcription slip would change every action while still running
    — so it is checked rather than assumed.
    """
    import numpy as np
    import jax.numpy as jnp

    install()
    ref = np.asarray(model._build_flow_velocity_fn(observation)(x_t, jnp.asarray(sigma)))
    kv, pm = model.prefix_cache(observation)
    new = np.asarray(model.velocity_from_cache(observation, kv, pm, x_t, jnp.asarray(sigma)))
    d = float(np.max(np.abs(ref - new)))
    ok = d <= tol
    print(f"  split velocity vs closure: max|delta| = {d:.2e}  -> {'MATCH' if ok else 'MISMATCH'}")
    return ok
