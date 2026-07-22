"""Validate PolicyWithLogprob + the rl_rollout_local policy_fn contract on CPU.

No GPU / checkpoint needed: wraps a dummy Pi0 in a Policy with identity transforms and
exercises the flow-SDE inference path end-to-end (action + chain + logp), the QueryTrace
build + npz round-trip, and the policy_fn chunk contract.

Run:  PYTHONPATH=.:openpi/src JAX_PLATFORMS=cpu openpi/.venv/bin/python -m experiments.test_rollout_local
"""
import dataclasses
import tempfile
from pathlib import Path

import jax
import numpy as np

import openpi.models.pi0_config as pcfg
from openpi.policies import policy as _policy
from openpi.policies.policy_logprob import PolicyWithLogprob

from experiments.policy_trace import from_flow_sde_roll, save_episode_trace, load_episode_trace
from experiments.rl_rollout_local import build_policy_fn

ok = True
def check(name, cond, extra=""):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}"); ok = ok and cond

# ── tiny real Pi0 (float32) wrapped in a Policy with identity transforms ──
cfg = dataclasses.replace(
    pcfg.Pi0Config(paligemma_variant="dummy", action_expert_variant="dummy"), dtype="float32")
model = cfg.create(jax.random.key(0))
policy = _policy.Policy(model, transforms=[], output_transforms=[])

N, NL, ST = 8, 0.7, "cps"
pol_lp = PolicyWithLogprob(policy, num_steps=N, noise_level=NL, sde_type=ST, seed=0)

# fake_obs(1) → dict (Observation-native keys) → un-batch (infer re-adds batch axis).
obs_batched = cfg.fake_obs(batch_size=1).to_dict()
obs = jax.tree.map(lambda x: np.asarray(x)[0], obs_batched)

roll = pol_lp.infer_with_logprob(obs)
ah, ad = model.action_horizon, model.action_dim
check("actions shape (ah, ad_env)", roll["actions"].shape == (ah, ad))
check("chain shape (num_steps+1, ah, ad)", roll["chain"].shape == (N + 1, ah, ad))
check("step_logp shape (num_steps,)", roll["step_logp"].shape == (N,))
check("sigmas shape (num_steps+1,)", roll["sigmas"].shape == (N + 1,))
check("noise_level/sde_type preserved", roll["noise_level"] == NL and roll["sde_type"] == ST)
check("logp finite", bool(np.all(np.isfinite(roll["step_logp"]))))
check("actions finite", bool(np.all(np.isfinite(roll["actions"]))))

# Two queries explore differently (independent RNG stream).
roll2 = pol_lp.infer_with_logprob(obs)
check("stochastic across queries", float(np.max(np.abs(roll2["chain"] - roll["chain"]))) > 1e-4)

# QueryTrace build + npz round-trip (the on-disk GRPO plumbing).
qt = from_flow_sde_roll(roll)
check("QueryTrace chain/logp/sigmas consistent", qt.chain.shape[0] == len(qt.sigmas)
      and len(qt.logp_old) == len(qt.sigmas) - 1)
with tempfile.TemporaryDirectory() as d:
    p = save_episode_trace([qt, from_flow_sde_roll(roll2)], Path(d) / "trace.npz")
    loaded = load_episode_trace(p)
    check("trace npz round-trip (2 queries)", len(loaded) == 2
          and np.allclose(loaded[0].chain, qt.chain, atol=1e-5)
          and np.allclose(loaded[0].logp_old, qt.logp_old, atol=1e-5))

# policy_fn contract used by run_libero_trial: returns (chunk of 7-D actions, QueryTrace).
# The real transform chain (observation/* -> model keys, prompt -> tokens) only exists
# with a checkpoint on GPU; here we stub infer_with_logprob with a canned roll to test
# build_policy_fn's OWN logic — chunk slicing to num_actions + QueryTrace assembly.
class _StubLP:
    def __init__(self, roll): self._roll = roll
    def infer_with_logprob(self, obs, *, noise=None):
        assert obs["prompt"] == "pick up the bowl"          # instruction plumbed through
        assert obs["observation/image"].shape == (224, 224, 3)
        return self._roll

canned = {
    "actions": np.zeros((ah, 7), dtype=np.float64),
    "chain": roll["chain"], "step_logp": roll["step_logp"],
    "sigmas": roll["sigmas"], "noise_level": NL, "sde_type": ST,
}
policy_fn = build_policy_fn(_StubLP(canned))
img = np.zeros((224, 224, 3), dtype=np.uint8)
state = np.zeros(8)
chunk, qtrace = policy_fn(img, img, state, "pick up the bowl", 5)
check("policy_fn chunk length == num_actions", len(chunk) == 5)
check("policy_fn action is env-width (7-D)", chunk[0].shape == (7,))
check("policy_fn returns a QueryTrace", qtrace.chain.shape == (N + 1, ah, ad))

print("\nALL PASS — PolicyWithLogprob + policy_fn validated" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
