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
    """One VLA query: an action chunk sampled by the flow-SDE, with its denoising chain.

    For the GRPO update the trainer must recompute logp_new under the current policy, which
    needs the model INPUT at this query. We store the RAW obs (uint8 images + state + prompt)
    so the trainer can re-apply the policy's input transform exactly (small on disk; exact).
    obs is optional — pure-math uses (no env) leave it None.
    """
    chain: np.ndarray        # (num_steps+1, *action_shape) latent trajectory
    logp_old: np.ndarray     # (num_steps,) per-step logp under the sampling policy
    sigmas: np.ndarray       # (num_steps+1,) time/sigma grid
    noise_level: float
    sde_type: str            # "sde" | "cps"
    obs: dict | None = None  # {"image":uint8, "wrist_image":uint8, "state":f32, "prompt":str}
    # Shield-as-expert (DAgger, Exp 005): the CBF-corrected actions ACTUALLY executed after
    # this query — env-space (unnormalized) 7-D actions, one per control step until the next
    # query (≈ replan_steps of them). This is the imitation target: the trainer flattens these
    # across queries into a per-step safe-action sequence and BC-trains the LoRA to reproduce it.
    shielded_actions: np.ndarray | None = None   # (n_exec, action_dim) env-space, or None

    def __post_init__(self):
        self.chain = np.asarray(self.chain, dtype=np.float32)
        self.logp_old = np.asarray(self.logp_old, dtype=np.float32)
        self.sigmas = np.asarray(self.sigmas, dtype=np.float32)
        assert self.chain.shape[0] == len(self.sigmas), "chain/sigmas length mismatch"
        assert len(self.logp_old) == len(self.sigmas) - 1, "logp_old must be num_steps"
        if self.shielded_actions is not None:
            self.shielded_actions = np.asarray(self.shielded_actions, dtype=np.float32)


def from_flow_sde_roll(roll: dict, obs: dict | None = None) -> QueryTrace:
    """Build a QueryTrace from a flow_sde.flow_sde_sample() output dict (+ optional raw obs)."""
    return QueryTrace(
        chain=np.stack(roll["chain"]), logp_old=roll["step_logp"],
        sigmas=roll["sigmas"], noise_level=roll["noise_level"], sde_type=roll["sde_type"],
        obs=obs,
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
    fields = dict(
        n_queries=len(queries),
        chain=np.stack([q.chain for q in queries]),        # (Q, S+1, *ashape)
        logp_old=np.stack([q.logp_old for q in queries]),  # (Q, S)
        sigmas=q0.sigmas,
        noise_level=q0.noise_level,
        sde_type=q0.sde_type,
    )
    # Persist the raw model inputs when captured (prompt is constant per episode).
    if q0.obs is not None:
        fields["has_obs"] = True
        fields["obs_image"] = np.stack([q.obs["image"] for q in queries])          # (Q,H,W,3) uint8
        fields["obs_wrist_image"] = np.stack([q.obs["wrist_image"] for q in queries])
        fields["obs_state"] = np.stack([np.asarray(q.obs["state"], np.float32) for q in queries])
        fields["obs_prompt"] = q0.obs.get("prompt", "")
    else:
        fields["has_obs"] = False
    # Shield-as-expert targets (Exp 005). Per-query executed counts vary at episode end, so pad
    # to the max length with NaN and store the true lengths (pickle-free; sliced back on load).
    if q0.shielded_actions is not None:
        lens = [len(q.shielded_actions) for q in queries]
        ad = int(queries[0].shielded_actions.shape[-1])
        padded = np.full((len(queries), max(lens), ad), np.nan, np.float32)
        for i, q in enumerate(queries):
            padded[i, :len(q.shielded_actions)] = q.shielded_actions
        fields["has_shielded"] = True
        fields["shielded_actions"] = padded
        fields["shielded_lens"] = np.asarray(lens, np.int32)
    else:
        fields["has_shielded"] = False
    np.savez_compressed(path, **fields)
    return path


def load_episode_trace(path: str | Path) -> list[QueryTrace]:
    with np.load(path, allow_pickle=True) as d:
        if int(d["n_queries"]) == 0:
            return []
        chain, logp = d["chain"], d["logp_old"]
        sigmas = d["sigmas"]; nl = float(d["noise_level"]); st = str(d["sde_type"])
        has_obs = bool(d["has_obs"]) if "has_obs" in d else False
        has_shielded = bool(d["has_shielded"]) if "has_shielded" in d else False
        # Hoist EVERY array out of the per-query loop. NpzFile.__getitem__ decompresses the whole
        # array on each access and caches nothing, so `d["obs_image"][i]` inside the loop
        # decompressed the entire (Q,224,224,3) stack once per query: ~110 MB x 728 queries = tens
        # of GB of allocation churn for a single trace. That was the real cause of the silent
        # OOM kills during index building, and of index builds taking half an hour.
        obs_image = d["obs_image"] if has_obs else None
        obs_wrist = d["obs_wrist_image"] if has_obs else None
        obs_state = d["obs_state"] if has_obs else None
        obs_prompt = str(d["obs_prompt"]) if has_obs else ""
        sh_all = d["shielded_actions"] if has_shielded else None
        sh_lens = d["shielded_lens"] if has_shielded else None

    out = []
    for i in range(len(chain)):
        obs = None
        if has_obs:
            obs = {"image": obs_image[i], "wrist_image": obs_wrist[i],
                   "state": obs_state[i], "prompt": obs_prompt}
        sh = sh_all[i, : int(sh_lens[i])] if has_shielded else None
        out.append(QueryTrace(chain[i], logp[i], sigmas, nl, st, obs=obs, shielded_actions=sh))
    return out


def grpo_training_tuples(queries: list[QueryTrace], advantage: float):
    """Yield (chain, logp_old, advantage, sigmas, noise_level, sde_type) per query.

    The trajectory's group advantage is broadcast to every query (action chunk) in the
    episode — standard for trajectory-level reward in DDPO/Flow-GRPO. The trainer then
    recomputes logp_new per chain and applies grpo_surrogate.
    """
    for q in queries:
        yield (q.chain, q.logp_old, float(advantage), q.sigmas, q.noise_level, q.sde_type)
