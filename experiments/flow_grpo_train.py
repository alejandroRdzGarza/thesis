"""
flow_grpo_train.py — GRPO update round for π0.5 (consumes a rollout round, updates LoRA).

Reads a rollout round produced by experiments/rl_rollout_local.py (manifest.csv + per-rollout
*_trace.npz, each carrying the denoising chains, logp_old, and the raw model obs), recomputes
logp_new under the current policy (Pi0.compute_chain_logp, HOOK C), and takes GRPO steps over
the TRAINABLE (LoRA) params only — the VLM backbone stays frozen. Saves a new checkpoint that
rl_rollout_local can load as --checkpoint for the next round.

  manifest.csv (trace_path, advantage)  →  batches of (Observation, chain, logp_old, adv)
      →  grpo_train_step (flow_grpo)  →  save <out>/params (+ copied assets)

Split so the load-bearing data assembly is CPU-testable (stack_query_batch) while the
transform-dependent observation rebuild + optimizer step run on the GPU with the real model.

Run on UCL:
  MUJOCO_GL=egl PYTHONPATH=. python -m experiments.flow_grpo_train \
      --config pi05_libero_cbf --checkpoint /path/to/pi05_libero \
      --round results_grpo/round0 --out results_grpo/round1_ckpt \
      --noise-level 0.7 --sde-type cps --epochs 1 --minibatch 8
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np


def read_manifest(round_dir: str | Path) -> list[dict]:
    """Rows of manifest.csv that have a policy trace, with their group advantage."""
    round_dir = Path(round_dir)
    rows = []
    with open(round_dir / "manifest.csv") as f:
        for r in csv.DictReader(f):
            if r.get("trace_path"):
                rows.append({"trace_path": r["trace_path"], "advantage": float(r["advantage"])})
    return rows


def stack_query_batch(chains: list[np.ndarray], logps: list[np.ndarray], advs: list[float]):
    """Stack per-query arrays into the (step-leading, batch-second) GRPO batch layout.

    chains[i] : (S+1, ah, ad)   logps[i] : (S,)   advs[i] : scalar
    returns   : chain (S+1, B, ah, ad),  logp_old (S, B),  advantages (B,)
    """
    chain = np.stack(chains, axis=1).astype(np.float32)      # (S+1, B, ah, ad)
    logp_old = np.stack(logps, axis=1).astype(np.float32)    # (S, B)
    advantages = np.asarray(advs, dtype=np.float32)          # (B,)
    return chain, logp_old, advantages


def _obs_dict_for_transform(q_obs: dict) -> dict:
    """Map a stored raw obs back to the server input keys the policy transform expects."""
    return {
        "observation/image": np.asarray(q_obs["image"]),
        "observation/wrist_image": np.asarray(q_obs["wrist_image"]),
        "observation/state": np.asarray(q_obs["state"], dtype=np.float64),
        "prompt": q_obs["prompt"],
    }


def build_observation(policy, q_obs_list: list[dict]):
    """Rebuild a batched model Observation from stored raw obs via the policy's input transform.

    Mirrors PolicyWithLogprob.infer_with_logprob's input handling, batched over queries.
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    transformed = [policy._input_transform(_obs_dict_for_transform(o)) for o in q_obs_list]
    # Stack each leaf across the query batch (leaves are numeric after the transform).
    batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *transformed)
    return _model.Observation.from_dict(batched)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf",
                    help="TrainConfig (must define the LoRA freeze_filter/trainable_filter)")
    ap.add_argument("--checkpoint", required=True, help="current-policy checkpoint dir (params/ + assets/)")
    ap.add_argument("--round", required=True, help="rollout round dir (manifest.csv + *_trace.npz)")
    ap.add_argument("--out", required=True, help="output checkpoint dir for the updated policy")
    ap.add_argument("--noise-level", type=float, default=0.7)
    ap.add_argument("--sde-type", default="cps", choices=["cps", "sde"])
    ap.add_argument("--clip", type=float, default=0.2, help="GRPO surrogate clip ε")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--minibatch", type=int, default=8, help="queries per optimizer step")
    args = ap.parse_args()

    import pathlib

    import flax.nnx as nnx
    import jax
    import optax

    from openpi.training import config as _config
    from openpi.training import flow_grpo

    from experiments.load_policy import create_policy_partial
    from experiments.policy_trace import load_episode_trace

    train_cfg = _config.get_config(args.config)
    trainable_filter = train_cfg.trainable_filter
    print(f"Loading policy: config={args.config}  checkpoint={args.checkpoint}")
    # Partial load: a BASE checkpoint initializes the LoRA config (missing lora params filled
    # fresh, lora_b=0 → identical to base); a saved LoRA checkpoint loads fully.
    policy = create_policy_partial(train_cfg, args.checkpoint)
    model = policy._model

    import time as _time

    # Flatten all rollout queries in the round into (obs, chain, logp_old, advantage).
    print(f"  loading traces from {args.round} ...", flush=True)
    rows = read_manifest(args.round)
    queries: list[tuple] = []
    for r in rows:
        for q in load_episode_trace(r["trace_path"]):
            if q.obs is None:
                raise ValueError(f"trace {r['trace_path']} has no stored obs — re-run rollout "
                                 "with the obs-capturing policy_fn (rl_rollout_local).")
            queries.append((q.obs, q.chain, q.logp_old, r["advantage"]))
    if not queries:
        raise SystemExit("no trace queries found in round")

    # Only process FULL minibatches: a trailing partial batch has a different shape, which
    # forces XLA to recompile the step (a second ~15GB program) whose host-memory compile
    # spike OOM-kills the process. Dropping the <minibatch remainder keeps one shape.
    n_full = (len(queries) // args.minibatch) * args.minibatch
    n_steps = args.epochs * (n_full // args.minibatch)
    print(f"  {len(rows)} rollouts → {len(queries)} queries  |  {n_steps} GRPO steps "
          f"(epochs={args.epochs}, minibatch={args.minibatch}, "
          f"dropping {len(queries) - n_full} trailing)", flush=True)
    print("  NOTE: the FIRST step compiles the full-remat backward (~1-3 min, silent); "
          "every step after is fast.", flush=True)

    tx = optax.adamw(args.lr)
    opt_state = tx.init(nnx.state(model, trainable_filter))
    rng = np.random.default_rng(0)

    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(queries))
        for s in range(0, n_full, args.minibatch):
            idx = order[s:s + args.minibatch]
            obs_list = [queries[i][0] for i in idx]
            chain, logp_old, adv = stack_query_batch(
                [queries[i][1] for i in idx], [queries[i][2] for i in idx],
                [queries[i][3] for i in idx])
            observation = build_observation(policy, obs_list)
            batch = (observation, jax.numpy.asarray(chain),
                     jax.numpy.asarray(logp_old), jax.numpy.asarray(adv))
            if step == 0:
                print("  compiling + running first step ...", flush=True)
            _t0 = _time.monotonic()
            opt_state, info = flow_grpo.grpo_train_step(
                model, tx, opt_state, batch, trainable_filter=trainable_filter,
                noise_level=args.noise_level, sde_type=args.sde_type, clip=args.clip)
            float(info["loss"])   # block until the step actually finishes (for honest timing)
            step += 1
            print(f"  step {step:04d}/{n_steps}  loss={float(info['loss']):+.4f}  "
                  f"|g|={float(info['grad_norm']):.3e}  mean_adv={float(info['mean_advantage']):+.3f}  "
                  f"({_time.monotonic() - _t0:.1f}s)", flush=True)

    # ── Save updated params as a create_trained_policy-loadable checkpoint ──
    import orbax.checkpoint as ocp

    del queries          # free the in-memory traces (obs images) before the host-side param pull
    print("  saving updated checkpoint ...", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    params_pure = nnx.state(model, nnx.Param).to_pure_dict()
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str((out / "params").resolve()), {"params": params_pure})
    # Norm stats live under assets/ — copy from the source checkpoint so the next round loads.
    src_assets = Path(args.checkpoint) / "assets"
    if src_assets.exists():
        shutil.copytree(src_assets, out / "assets", dirs_exist_ok=True)
    print(f"\nSaved updated policy → {out}  (load next round with --checkpoint {out})")


if __name__ == "__main__":
    main()
