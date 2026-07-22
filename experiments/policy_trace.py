"""
policy_trace.py — capture the flow-SDE policy trace of a rollout for on-policy GRPO.

For Flow-GRPO-style updates we need, for every VLA query (= one action chunk sampled via
the flow-SDE), the denoising CHAIN and the per-step log-prob under the SAMPLING policy
(logp_old). At update time we recompute logp_new for the same chain under the current
policy (flow_sde.flow_sde_recompute_logp), form the ratio, and apply grpo_surrogate with
the trajectory's group advantage.

An episode is a sequence of QueryTrace (one per action chunk). A rollout = one episode.
This module is the data plumbing: schema + save/load (npz) + GRPO-tuple assembly. It is
pure and unit-tested with synthetic traces.

ON-BOX CONTRACT (the only piece that needs the JAX sampler + server):
  run_libero_trial, when record_policy_trace=True, must append one QueryTrace per π0.5
  query into `metrics.policy_trace` — the server's sample_with_logprob returns the chain
  + logp_old (see FLOW_SDE_OPENPI.md). Everything downstream here is ready.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class QueryTrace:
    """One VLA query: an action chunk sampled by the flow-SDE, with its denoising chain."""
    chain: np.ndarray        # (num_steps+1, *action_shape) latent trajectory
    logp_old: np.ndarray     # (num_steps,) per-step logp under the sampling policy
    sigmas: np.ndarray       # (num_steps+1,) time/sigma grid
    noise_level: float
    sde_type: str            # "sde" | "cps"

    def __post_init__(self):
        self.chain = np.asarray(self.chain, dtype=np.float32)
        self.logp_old = np.asarray(self.logp_old, dtype=np.float32)
        self.sigmas = np.asarray(self.sigmas, dtype=np.float32)
        assert self.chain.shape[0] == len(self.sigmas), "chain/sigmas length mismatch"
        assert len(self.logp_old) == len(self.sigmas) - 1, "logp_old must be num_steps"


def from_flow_sde_roll(roll: dict) -> QueryTrace:
    """Build a QueryTrace from a flow_sde.flow_sde_sample() output dict."""
    return QueryTrace(
        chain=np.stack(roll["chain"]), logp_old=roll["step_logp"],
        sigmas=roll["sigmas"], noise_level=roll["noise_level"], sde_type=roll["sde_type"],
    )


def save_episode_trace(queries: list[QueryTrace], path: str | Path) -> Path:
    """Save a rollout's per-query traces to a single .npz.

    Chains/logps are stacked with a leading query axis; requires uniform shapes across
    queries (true within one config: same num_steps, action_horizon, action_dim).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not queries:
        np.savez_compressed(path, n_queries=0)
        return path
    q0 = queries[0]
    np.savez_compressed(
        path,
        n_queries=len(queries),
        chain=np.stack([q.chain for q in queries]),        # (Q, S+1, *ashape)
        logp_old=np.stack([q.logp_old for q in queries]),  # (Q, S)
        sigmas=q0.sigmas,
        noise_level=q0.noise_level,
        sde_type=q0.sde_type,
    )
    return path


def load_episode_trace(path: str | Path) -> list[QueryTrace]:
    d = np.load(path, allow_pickle=True)
    if int(d["n_queries"]) == 0:
        return []
    chain, logp = d["chain"], d["logp_old"]
    sigmas = d["sigmas"]; nl = float(d["noise_level"]); st = str(d["sde_type"])
    return [QueryTrace(chain[i], logp[i], sigmas, nl, st) for i in range(len(chain))]


def grpo_training_tuples(queries: list[QueryTrace], advantage: float):
    """Yield (chain, logp_old, advantage, sigmas, noise_level, sde_type) per query.

    The trajectory's group advantage is broadcast to every query (action chunk) in the
    episode — standard for trajectory-level reward in DDPO/Flow-GRPO. The trainer then
    recomputes logp_new per chain and applies grpo_surrogate.
    """
    for q in queries:
        yield (q.chain, q.logp_old, float(advantage), q.sigmas, q.noise_level, q.sde_type)
