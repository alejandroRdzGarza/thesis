"""Checks for policy_trace.py (run: python -m experiments.test_policy_trace)."""
import tempfile, csv
import numpy as np
from pathlib import Path
from experiments.flow_sde import make_sigmas, flow_sde_sample
from experiments.policy_trace import (
    QueryTrace, from_flow_sde_roll, save_episode_trace, load_episode_trace,
    grpo_training_tuples,
)
from experiments.safe_reward import compute_episode_reward, RewardConfig
from experiments.rl_rollout import Rollout, write_manifest

ok = True
def check(name, cond):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); ok = ok and cond

# Build a synthetic episode: E VLA queries, each an action chunk (H=5, D=7) sampled by flow-SDE.
H, D, E = 5, 7, 4
sigmas = make_sigmas(6)
def v_fn(x, sigma): return np.zeros_like(x)
rng = np.random.default_rng(0)
queries = []
for _ in range(E):
    roll = flow_sde_sample(v_fn, np.zeros((H, D)), sigmas, noise_level=0.7, rng=rng, sde_type="cps")
    queries.append(from_flow_sde_roll(roll))

# 1. QueryTrace shape invariants
q = queries[0]
check("chain length == num_steps+1", q.chain.shape[0] == len(sigmas))
check("logp_old length == num_steps", len(q.logp_old) == len(sigmas) - 1)
check("chain carries action-chunk shape", q.chain.shape[1:] == (H, D))

# 2. save/load roundtrip
with tempfile.TemporaryDirectory() as d:
    p = save_episode_trace(queries, Path(d) / "g0" / "rollout_00_trace.npz")
    back = load_episode_trace(p)
    check("roundtrip preserves query count", len(back) == E)
    check("roundtrip preserves chain values", np.allclose(back[0].chain, q.chain, atol=1e-5))
    check("roundtrip preserves logp_old", np.allclose(back[0].logp_old, q.logp_old, atol=1e-5))
    check("roundtrip preserves config", back[0].sde_type == "cps" and abs(back[0].noise_level - 0.7) < 1e-6)

    # empty trace roundtrip
    pe = save_episode_trace([], Path(d) / "empty.npz")
    check("empty trace roundtrips to []", load_episode_trace(pe) == [])

# 3. GRPO tuples broadcast the trajectory advantage to every query
tuples = list(grpo_training_tuples(queries, advantage=1.7))
check("one training tuple per query", len(tuples) == E)
check("advantage broadcast to all queries", all(abs(t[2] - 1.7) < 1e-9 for t in tuples))
check("tuple carries chain + logp_old + sigmas", tuples[0][0].shape == q.chain.shape
      and len(tuples[0][1]) == len(sigmas) - 1)

# 4. Rollout manifest now carries trace_path
rb = compute_episode_reward(goal_reached=True, collision_detected=False, cfg=RewardConfig())
r = Rollout(0, 0, "g0/rollout_00.h5", rb, trace_path="g0/rollout_00_trace.npz")
with tempfile.TemporaryDirectory() as d:
    mp = write_manifest([r], Path(d) / "manifest.csv")
    row = next(csv.DictReader(open(mp)))
    check("manifest has trace_path column", "trace_path" in row)
    check("manifest trace_path value persisted", row["trace_path"] == "g0/rollout_00_trace.npz")

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
