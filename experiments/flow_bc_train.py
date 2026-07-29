"""
flow_bc_train.py — shield-as-expert BC round for π0.5 (thesis Exp 005 pivot).

Consumes a SHIELDED rollout round (rl_rollout_local --shield-prob 1.0): each *_trace.npz carries,
per VLA query, the raw model obs AND the CBF-corrected actions actually executed
(QueryTrace.shielded_actions). We behavior-clone the LoRA action head to reproduce those safe
actions via the model's native flow-matching loss (Pi0.compute_loss) — dense per-step supervision
that structurally can't collapse to inaction (the target action does the task safely).

  traces (obs, shielded_actions)  →  (Observation, normalized action chunk)  →  bc_train_step
      →  save <out>/lora_params (+ base.txt), reloadable by create_policy_partial next round.

The shielded actions are ENV-space; we normalize them to model space by running them THROUGH the
policy's own input transform (LiberoInputs passes `actions` → Normalize normalizes them), so the
target matches exactly what π0.5 was trained on — no hand-rolled normalization.

Run (on the pod), validate the data path first:
  python -m experiments.flow_bc_train --config pi05_libero_cbf --checkpoint $CKPT \
      --round results_dagger/round0 --out results_dagger/round1_ckpt --dry-run
Then the real update (drop --dry-run).
"""

from __future__ import annotations

import argparse
import glob
import shutil
import time as _time
from pathlib import Path

import numpy as np


def _episode_targets(queries, action_horizon: int):
    """Build (obs, env-space target chunk) pairs from one episode's queries.

    Each query i predicts `action_horizon` actions from its obs; the target is the executed
    shield-corrected trajectory starting at that query — i.e. the concatenation of this and
    subsequent queries' shielded_actions, truncated to action_horizon (tail-padded by repeating
    the last executed action, which near episode end is a settled hold pose).
    """
    usable = [q for q in queries if q.obs is not None and q.shielded_actions is not None
              and len(q.shielded_actions) > 0]
    if not usable:
        return []
    # Per-episode flattened executed sequence + each query's start offset into it.
    seq = np.concatenate([q.shielded_actions for q in usable], axis=0)   # (T, 7)
    offsets, o = [], 0
    for q in usable:
        offsets.append(o)
        o += len(q.shielded_actions)
    out = []
    for q, off in zip(usable, offsets):
        chunk = seq[off: off + action_horizon]
        if len(chunk) < action_horizon:                       # tail-pad by holding the last action
            pad = np.repeat(chunk[-1:], action_horizon - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        out.append((q.obs, chunk.astype(np.float32)))
    return out


def build_bc_batch(policy, obs_list, act_list, action_dim: int):
    """Transform (raw obs, env-space action chunk) pairs → (Observation, normalized actions).

    Runs each sample through the policy's input transform (same stack π0.5 trains with), which
    normalizes `actions` alongside the obs. Actions are zero-padded to the model action_dim
    (normalized zero = the mean, matching openpi's LIBERO 7→action_dim padding).
    """
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    samples = []
    for o, a in zip(obs_list, act_list):
        d = {
            "observation/image": np.asarray(o["image"]),
            "observation/wrist_image": np.asarray(o["wrist_image"]),
            "observation/state": np.asarray(o["state"], dtype=np.float64),
            "prompt": o["prompt"] if isinstance(o["prompt"], str) else str(o["prompt"]),
            "actions": np.asarray(a, dtype=np.float32),          # (ah, 7) env-space
        }
        samples.append(policy._input_transform(d))
    batched = jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *samples)
    actions = jnp.asarray(batched.pop("actions"))               # normalized by the transform
    if actions.shape[-1] < action_dim:                          # pad 7 → action_dim with (normalized) zeros
        pad = jnp.zeros((*actions.shape[:-1], action_dim - actions.shape[-1]), actions.dtype)
        actions = jnp.concatenate([actions, pad], axis=-1)
    observation = _model.Observation.from_dict(batched)
    return observation, actions


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="pi05_libero_cbf")
    ap.add_argument("--checkpoint", required=True, help="current-policy checkpoint dir")
    ap.add_argument("--round", required=True, help="SHIELDED rollout round dir (*_trace.npz)")
    ap.add_argument("--out", required=True, help="output checkpoint dir for the BC-updated policy")
    ap.add_argument("--lr", type=float, default=1e-4, help="BC LR (imitation is stable; higher than RL)")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--dry-run", action="store_true",
                    help="assemble one minibatch, print shapes + one compute_loss, and exit "
                         "(validate the data path before a full run)")
    args = ap.parse_args()

    import flax.nnx as nnx
    import jax
    import optax

    from openpi.training import config as _config
    from openpi.training import flow_bc

    from experiments.load_policy import create_policy_partial
    from experiments.policy_trace import load_episode_trace

    train_cfg = _config.get_config(args.config)
    trainable_filter = train_cfg.trainable_filter
    print(f"Loading policy: config={args.config}  checkpoint={args.checkpoint}", flush=True)
    policy = create_policy_partial(train_cfg, args.checkpoint)
    model = policy._model
    action_horizon = int(model.action_horizon)
    action_dim = int(model.action_dim)
    import gc as _gc
    _gc.collect()

    # Flatten every shielded rollout in the round into (obs, env-space target chunk) examples.
    trace_paths = sorted(glob.glob(str(Path(args.round) / "**" / "*_trace.npz"), recursive=True))
    if not trace_paths:
        raise SystemExit(f"no *_trace.npz under {args.round} — run rl_rollout_local --shield-prob 1.0 first")
    examples: list[tuple] = []
    n_no_shield = 0
    for tp in trace_paths:
        qs = load_episode_trace(tp)
        if qs and all(q.shielded_actions is None for q in qs):
            n_no_shield += 1
        examples.extend(_episode_targets(qs, action_horizon))
    if not examples:
        raise SystemExit(
            f"no shielded targets found in {args.round} ({n_no_shield} traces had no shielded_actions "
            "— the rollout must be run with the updated libero_runner that records them).")
    print(f"  {len(trace_paths)} traces → {len(examples)} BC examples  "
          f"(action_horizon={action_horizon}, action_dim={action_dim})", flush=True)

    n_full = (len(examples) // args.minibatch) * args.minibatch
    rng = np.random.default_rng(0)
    jax_rng = jax.random.key(0)

    if args.dry_run:
        idx = list(range(min(args.minibatch, len(examples))))
        obs, acts = build_bc_batch(policy, [examples[i][0] for i in idx],
                                   [examples[i][1] for i in idx], action_dim)
        print(f"  [dry-run] Observation.state {obs.state.shape}  actions {acts.shape} "
              f"(want (b, {action_horizon}, {action_dim}))", flush=True)
        per = model.compute_loss(jax_rng, obs, acts, train=False)
        print(f"  [dry-run] compute_loss mean = {float(per.mean()):.4f}  — data path OK.", flush=True)
        return

    tx = optax.adamw(args.lr)
    opt_state = tx.init(nnx.state(model, trainable_filter))
    n_steps = args.epochs * (n_full // args.minibatch)
    print(f"  {n_steps} BC steps (epochs={args.epochs}, minibatch={args.minibatch}, "
          f"dropping {len(examples) - n_full} trailing). First step compiles (~1-3 min).", flush=True)

    step = 0
    for epoch in range(args.epochs):
        order = rng.permutation(len(examples))
        for s in range(0, n_full, args.minibatch):
            idx = order[s:s + args.minibatch]
            obs, acts = build_bc_batch(policy, [examples[i][0] for i in idx],
                                       [examples[i][1] for i in idx], action_dim)
            jax_rng, step_rng = jax.random.split(jax_rng)
            _t0 = _time.monotonic()
            opt_state, info = flow_bc.bc_train_step(
                model, tx, opt_state, (step_rng, obs, acts), trainable_filter=trainable_filter)
            float(info["loss"])
            step += 1
            print(f"  step {step:04d}/{n_steps}  loss={float(info['loss']):.4f}  "
                  f"|g|={float(info['grad_norm']):.3e}  ({_time.monotonic() - _t0:.1f}s)", flush=True)

    # Save ONLY the trained LoRA adapter (+ base.txt), like flow_grpo_train.
    import gc
    import orbax.checkpoint as ocp
    del examples
    gc.collect()
    print("  saving LoRA adapter (frozen backbone not re-saved) ...", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    lora_pure = nnx.state(model, trainable_filter).to_pure_dict()
    lora_path = (out / "lora_params").resolve()
    if lora_path.exists():
        shutil.rmtree(lora_path)
    with ocp.PyTreeCheckpointer() as ckptr:
        ckptr.save(str(lora_path), {"params": lora_pure})
    _src = Path(args.checkpoint)
    resolved_base = ((_src / "base.txt").read_text().strip()
                     if (_src / "base.txt").exists() else str(_src.resolve()))
    (out / "base.txt").write_text(resolved_base)
    print(f"\nSaved LoRA adapter → {out}  (backbone from {resolved_base})\n"
          f"  load next round with --checkpoint {out}", flush=True)


if __name__ == "__main__":
    main()
