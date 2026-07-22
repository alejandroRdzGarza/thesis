"""HOOK A end-to-end validation on a tiny (dummy) Pi0 — no GPU/checkpoint needed.

Validates Pi0.sample_actions_with_logprob against the deterministic sample_actions
using openpi's dummy gemma variant + fake_obs, on CPU.
Run:  PYTHONPATH=. JAX_PLATFORMS=cpu openpi/.venv/bin/python -m experiments.test_pi0_flow_sde
"""
import dataclasses
import jax
import jax.numpy as jnp
import openpi.models.pi0_config as pcfg

ok = True
def check(name, cond, extra=""):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}"); ok = ok and cond

# Tiny model in float32 (bf16 default would add ~1e-2 rounding — real serving is bf16).
cfg = dataclasses.replace(
    pcfg.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"), dtype="float32")
model = cfg.create(jax.random.key(0))
rng = jax.random.key(1)

for B in (1, 3):                                   # single-query serving + batched training
    obs = cfg.fake_obs(batch_size=B)
    noise = jax.random.normal(rng, (B, model.action_horizon, model.action_dim))
    a_ode = model.sample_actions(rng, obs, num_steps=10, noise=noise)

    for st in ("cps", "sde"):
        a0, chain, logp = model.sample_actions_with_logprob(
            rng, obs, num_steps=10, noise_level=0.0, sde_type=st, noise=noise)
        d = float(jnp.max(jnp.abs(a0 - a_ode)))
        check(f"B={B} [{st}] noise_level=0 == ODE sampler", d < 1e-3, f"(maxΔ={d:.2e})")
        check(f"B={B} [{st}] chain shape (steps+1,B,ah,ad)",
              chain.shape == (11, B, model.action_horizon, model.action_dim))
        check(f"B={B} [{st}] logp shape (steps,B)", logp.shape == (10, B))

    # stochastic exploration + finite logp
    a_s, _, lp = model.sample_actions_with_logprob(
        rng, obs, num_steps=10, noise_level=0.7, sde_type="cps", noise=noise)
    check(f"B={B} nl=0.7 explores (≠ ODE)", float(jnp.max(jnp.abs(a_s - a_ode))) > 1e-3)
    check(f"B={B} nl=0.7 logp finite", bool(jnp.all(jnp.isfinite(lp))))

print("\nALL PASS — HOOK A validated end-to-end on Pi0" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
