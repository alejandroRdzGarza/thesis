"""Validate HOOK C (compute_chain_logp) + flow_grpo loss/step on a dummy Pi0 (CPU).

The load-bearing check: recomputing logp of a sampled chain under the SAME policy
reproduces the sampling step_logp (on-policy ratio ≡ 1) — this is what makes the GRPO
ratio meaningful. Then: loss == -mean(advantages) at init, and a train step runs and
moves the trainable params.

Run:  PYTHONPATH=. JAX_PLATFORMS=cpu openpi/.venv/bin/python -m experiments.test_flow_grpo
"""
import dataclasses

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import openpi.models.pi0_config as pcfg
from openpi.training import flow_grpo

ok = True
def check(name, cond, extra=""):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}"); ok = ok and cond

cfg = dataclasses.replace(
    pcfg.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"), dtype="float32")
model = cfg.create(jax.random.key(0))
rng = jax.random.key(1)
B, N, NL = 3, 8, 0.7

for ST in ("cps", "sde"):
    obs = cfg.fake_obs(batch_size=B)
    action, chain, step_logp = model.sample_actions_with_logprob(
        rng, obs, num_steps=N, noise_level=NL, sde_type=ST)      # chain (N+1,B,ah,ad), logp (N,B)

    # ── HOOK C: recompute logp of the sampled chain under the same policy ──
    logp_new = model.compute_chain_logp(obs, chain, noise_level=NL, sde_type=ST)
    d = float(jnp.max(jnp.abs(logp_new - step_logp)))
    check(f"[{ST}] compute_chain_logp == step_logp (ratio≡1)", d < 1e-3, f"(maxΔ={d:.2e})")
    check(f"[{ST}] logp_new shape (num_steps, B)", logp_new.shape == (N, B))

    # ── loss at init: ratio≡1 ⇒ surrogate == advantage ⇒ loss == -mean(adv) ──
    adv = jnp.asarray(np.array([1.0, -0.5, 0.25], dtype=np.float32))
    loss = flow_grpo.flow_grpo_loss(model, obs, chain, step_logp, adv,
                                    noise_level=NL, sde_type=ST)
    expected = -float(jnp.mean(adv))
    check(f"[{ST}] loss == -mean(advantages) at init",
          abs(float(loss) - expected) < 1e-3, f"(loss={float(loss):.4f} exp={expected:.4f})")

# ── one GRPO train step actually runs and moves trainable params ──
obs = cfg.fake_obs(batch_size=B)
_, chain, step_logp = model.sample_actions_with_logprob(rng, obs, num_steps=N, noise_level=NL, sde_type="cps")
adv = jnp.asarray(np.array([1.0, -1.0, 0.5], dtype=np.float32))
trainable = nnx.Param                                            # train all params (dummy has no LoRA)
tx = optax.sgd(1e-2)
opt_state = tx.init(nnx.state(model, trainable))
before = jax.tree.map(lambda x: np.array(x), nnx.state(model, trainable).flat_state())
new_opt_state, info = flow_grpo.grpo_train_step(
    model, tx, opt_state, (obs, chain, step_logp, adv),
    trainable_filter=trainable, noise_level=NL, sde_type="cps")
check("train step: loss finite", bool(jnp.isfinite(info["loss"])), f"(loss={float(info['loss']):.4f})")
check("train step: grad_norm finite & >0", bool(jnp.isfinite(info["grad_norm"]) and info["grad_norm"] > 0),
      f"(|g|={float(info['grad_norm']):.3e})")
after = nnx.state(model, trainable).flat_state()
moved = any(not np.allclose(np.array(after[k].value), before[k].value)
            for k in after if hasattr(after[k], "value"))
check("train step: trainable params moved", moved)

# ── driver plumbing: stack_query_batch layout + trace-with-obs round trip ──
import tempfile
from pathlib import Path
from experiments.flow_grpo_train import stack_query_batch
from experiments.policy_trace import QueryTrace, save_episode_trace, load_episode_trace

S, ah, ad = N, model.action_horizon, model.action_dim
qchains = [np.random.randn(S + 1, ah, ad).astype(np.float32) for _ in range(3)]
qlogps = [np.random.randn(S).astype(np.float32) for _ in range(3)]
cb, lo, av = stack_query_batch(qchains, qlogps, [1.0, -0.5, 0.25])
check("stack_query_batch chain layout (S+1,B,ah,ad)", cb.shape == (S + 1, 3, ah, ad)
      and np.allclose(cb[:, 1], qchains[1]))       # batch axis is 1, query order preserved
check("stack_query_batch logp layout (S,B)", lo.shape == (S, 3) and np.allclose(lo[:, 2], qlogps[2]))
check("stack_query_batch advantages (B,)", av.shape == (3,))

# QueryTrace with obs survives the npz round-trip (what the trainer reads back).
sig = np.linspace(1.0, 0.0, S + 1).astype(np.float32)
obs_raw = {"image": np.zeros((224, 224, 3), np.uint8),
           "wrist_image": np.ones((224, 224, 3), np.uint8),
           "state": np.arange(8, dtype=np.float32), "prompt": "pick up the bowl"}
qt = QueryTrace(qchains[0], qlogps[0], sig, 0.7, "cps", obs=obs_raw)
with tempfile.TemporaryDirectory() as d:
    p = save_episode_trace([qt, qt], Path(d) / "t.npz")
    lq = load_episode_trace(p)
    check("trace obs round-trip (image/state/prompt)",
          lq[0].obs is not None
          and np.array_equal(lq[0].obs["wrist_image"], obs_raw["wrist_image"])
          and np.allclose(lq[0].obs["state"], obs_raw["state"])
          and lq[0].obs["prompt"] == "pick up the bowl")

print("\nALL PASS — HOOK C + flow_grpo validated" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
