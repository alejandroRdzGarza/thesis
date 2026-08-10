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


def _successful_traces(round_dir) -> set | None:
    """Absolute paths of traces whose rollout SUCCEEDED and stayed collision-free.

    Read from the round's manifest.csv (r_success>0 and robot_caused_collision==0). Returns None
    if there's no manifest (→ caller keeps all traces). This is the automated "expert filter":
    imitate only safe, task-completing shielded trajectories, so BC can't drift toward the
    shield's over-cautious *failures* (the success erosion seen across DAgger rounds in Exp 005).
    """
    import csv
    mpath = Path(round_dir) / "manifest.csv"
    if not mpath.exists():
        return None
    keep = set()
    with open(mpath) as f:
        for r in csv.DictReader(f):
            tp = r.get("trace_path")
            if not tp:
                continue
            succeeded = float(r.get("r_success", 0) or 0) > 0
            # Prefer the RAW collision flag when the manifest carries it. `robot_caused_collision`
            # is the contact-graph attribution, a documented LOWER BOUND that misses delayed and
            # indirect pushes — filtering a SAFETY demo set on it admits demos that did displace
            # the obstacle. Older manifests without the column fall back to the attributed flag.
            _coll = r.get("collision_raw", r.get("robot_caused_collision", 0))
            safe = int(_coll or 0) == 0
            if succeeded and safe:
                keep.add(str(Path(tp).resolve()))
    return keep


def _usable_queries(queries):
    """Queries that carry both an obs and executed target actions (BC-trainable)."""
    return [q for q in queries if q.obs is not None and q.shielded_actions is not None
            and len(q.shielded_actions) > 0]


def _episode_target_chunks(queries, action_horizon: int):
    """(usable_index, env-space target chunk) pairs from one episode's queries.

    Each usable query i predicts `action_horizon` actions from its obs; the target is the executed
    shield-corrected trajectory starting at that query — i.e. the concatenation of this and
    subsequent queries' shielded_actions, truncated to action_horizon (tail-padded by repeating
    the last executed action, which near episode end is a settled hold pose). Returns only the
    index into _usable_queries(queries) + the (small) chunk — NOT the image — so the caller can
    build a memory-light index and stream obs from disk per minibatch.
    """
    usable = _usable_queries(queries)
    if not usable:
        return []
    # Per-episode flattened executed sequence + each query's start offset into it.
    seq = np.concatenate([q.shielded_actions for q in usable], axis=0)   # (T, 7)
    offsets, o = [], 0
    for q in usable:
        offsets.append(o)
        o += len(q.shielded_actions)
    out = []
    for i, off in enumerate(offsets):
        chunk = seq[off: off + action_horizon]
        if len(chunk) < action_horizon:                       # tail-pad by holding the last action
            pad = np.repeat(chunk[-1:], action_horizon - len(chunk), axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        out.append((i, chunk.astype(np.float32)))
    return out


def _episode_targets(queries, action_horizon: int):
    """(obs, target chunk) pairs — the in-RAM convenience form (dry-run / small use)."""
    usable = _usable_queries(queries)
    return [(usable[i].obs, chunk) for i, chunk in _episode_target_chunks(queries, action_horizon)]


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
    ap.add_argument("--round", required=True, nargs="+",
                    help="one or more round dirs with *_trace.npz (aggregate them — e.g. round-0 "
                         "offline demos + all DAgger rounds)")
    ap.add_argument("--out", required=True, help="output checkpoint dir for the BC-updated policy")
    ap.add_argument("--lr", type=float, default=1e-4, help="BC LR (imitation is stable; higher than RL)")
    ap.add_argument("--epochs", type=int, default=20,
                    help="20+. NOT 2: the long-standing 'BC gives 0%% success' result was a "
                         "2-epoch underfit, not a property of BC. A default that silently "
                         "underfits reads as 'this data teaches nothing' whatever the data is, "
                         "which is how a bad default fakes an experimental conclusion.")
    ap.add_argument("--minibatch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0,
                    help="seeds BOTH the data shuffle and the JAX key. Both were hardcoded to 0, so "
                         "every arm in this project is a single training run — the softest "
                         "methodological point in the results. Vary this to measure how much of a "
                         "reported difference is training-seed variance rather than the treatment.")
    ap.add_argument("--success-only", action="store_true",
                    help="imitate ONLY traces whose rollout succeeded AND stayed collision-free "
                         "(via manifest.csv) — the automated expert filter; guards against BC "
                         "drifting toward the shield's over-cautious failures (success erosion).")
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

    # Flatten every recorded rollout across ALL round dirs into (obs, env-space target chunk)
    # examples. DAgger aggregates round-0 offline demos + every relabelled DAgger round here.
    round_dirs = list(args.round)
    trace_paths = []
    for rd in round_dirs:
        trace_paths.extend(glob.glob(str(Path(rd) / "**" / "*_trace.npz"), recursive=True))
    trace_paths = sorted(set(trace_paths))
    if not trace_paths:
        raise SystemExit(f"no *_trace.npz under {round_dirs} — collect demos / rollouts first")
    print(f"  aggregating {len(round_dirs)} round dir(s): {round_dirs}", flush=True)
    if args.success_only:
        # Union the per-round success filters (a trace kept if ITS round's manifest marks it clean).
        keep = set()
        any_manifest = False
        for rd in round_dirs:
            k = _successful_traces(rd)
            if k is not None:
                any_manifest = True
                keep |= k
        if not any_manifest:
            print("  [success-only] no manifest.csv in any round → keeping all traces", flush=True)
        else:
            before = len(trace_paths)
            trace_paths = [tp for tp in trace_paths if str(Path(tp).resolve()) in keep]
            print(f"  [success-only] kept {len(trace_paths)}/{before} traces "
                  "(succeeded + collision-free)", flush=True)
            if not trace_paths:
                raise SystemExit("no successful+safe traces — nothing to imitate "
                                 "(policy may have regressed; check the round summaries).")
    # Build a memory-LIGHT index only (trace path + per-query target chunk, NO images). Images are
    # streamed from disk one trace at a time during training, so peak RAM is O(one trace), not
    # O(dataset) — essential because DAgger grows the dataset every round (all-in-RAM OOM-killed the
    # host at round 1: ~2.8k 224² images ≈ 0.9 GB decoded at once).
    index: list[tuple] = []   # (trace_path, usable_idx, chunk)
    n_no_shield = 0
    # Narrate it. This loop decompresses every trace and can run for tens of minutes on a large
    # round; when it was silent, an OOM kill here (SIGKILL, so no traceback) was indistinguishable
    # from normal progress, and cost two 30-minute runs before it was diagnosed.
    _n_tp = len(trace_paths)
    print(f"  building index from {_n_tp} traces (decompressing; this is the slow part) ...",
          flush=True)
    import time as _t
    _t0 = _t.time()
    for _i, tp in enumerate(trace_paths, 1):
        qs = load_episode_trace(tp)
        if qs and all(q.shielded_actions is None for q in qs):
            n_no_shield += 1
        for uidx, chunk in _episode_target_chunks(qs, action_horizon):
            index.append((tp, uidx, chunk))
        del qs
        _gc.collect()
        if _i % 10 == 0 or _i == _n_tp:
            _el = _t.time() - _t0
            _eta = _el / _i * (_n_tp - _i)
            print(f"    [{_i}/{_n_tp}] {len(index)} examples  "
                  f"{_el/60:.1f}m elapsed  ETA {_eta/60:.1f}m", flush=True)
    if not index:
        raise SystemExit(
            f"no shielded targets found in {round_dirs} ({n_no_shield} traces had no shielded_actions "
            "— the rollout must be run with the updated libero_runner that records them).")
    n_examples = len(index)
    print(f"  {len(trace_paths)} traces → {n_examples} BC examples  "
          f"(action_horizon={action_horizon}, action_dim={action_dim})  [streamed from disk]",
          flush=True)

    n_full = (n_examples // args.minibatch) * args.minibatch
    rng = np.random.default_rng(args.seed)
    jax_rng = jax.random.key(args.seed)
    print(f"  seed = {args.seed} (data shuffle + JAX key)", flush=True)

    # Group example rows by trace so an epoch loads each trace's images exactly once (streaming),
    # while still shuffling trace order + within-trace rows. build_bc_batch needs the obs, which we
    # fetch from the currently-loaded trace via its usable_idx.
    from collections import defaultdict
    rows_by_trace: dict[str, list[int]] = defaultdict(list)
    for ri, (tp, _uidx, _chunk) in enumerate(index):
        rows_by_trace[tp].append(ri)
    trace_list = list(rows_by_trace.keys())

    def _obs_for_trace(tp):
        return [q.obs for q in _usable_queries(load_episode_trace(tp))]

    if args.dry_run:
        tp0 = trace_list[0]
        obs_pool = _obs_for_trace(tp0)
        rows = rows_by_trace[tp0][:args.minibatch]
        obs, acts = build_bc_batch(policy, [obs_pool[index[i][1]] for i in rows],
                                   [index[i][2] for i in rows], action_dim)
        print(f"  [dry-run] Observation.state {obs.state.shape}  actions {acts.shape} "
              f"(want (b, {action_horizon}, {action_dim}))", flush=True)
        per = model.compute_loss(jax_rng, obs, acts, train=False)
        print(f"  [dry-run] compute_loss mean = {float(per.mean()):.4f}  — data path OK.", flush=True)
        return

    tx = optax.adamw(args.lr)
    opt_state = tx.init(nnx.state(model, trainable_filter))
    n_steps = args.epochs * (n_full // args.minibatch)
    print(f"  {n_steps} BC steps (epochs={args.epochs}, minibatch={args.minibatch}, "
          f"dropping {n_examples - n_full} trailing). First step compiles (~1-3 min).", flush=True)

    def _run_step(nonlocal_rng, obs_list, act_list):
        obs, acts = build_bc_batch(policy, obs_list, act_list, action_dim)
        step_rng, nxt = jax.random.split(nonlocal_rng)
        _t0 = _time.monotonic()
        os_, info = flow_bc.bc_train_step(
            model, tx, opt_state, (step_rng, obs, acts), trainable_filter=trainable_filter)
        return os_, nxt, float(info["loss"]), float(info["grad_norm"]), _time.monotonic() - _t0

    step = 0
    for epoch in range(args.epochs):
        # Trace-blocked shuffle: shuffle trace order + within-trace rows; carry a minibatch buffer
        # across trace boundaries so we only drop ONE trailing partial batch per epoch. BC tolerates
        # this mild (non-global) shuffle; peak RAM stays at one trace's images.
        t_order = rng.permutation(len(trace_list))
        buf_obs: list = []
        buf_act: list = []
        emitted = 0
        for tii in t_order:
            tp = trace_list[tii]
            obs_pool = _obs_for_trace(tp)
            rows = rows_by_trace[tp]
            for li in rng.permutation(len(rows)):
                ri = rows[li]
                buf_obs.append(obs_pool[index[ri][1]])
                buf_act.append(index[ri][2])
                if len(buf_obs) == args.minibatch:
                    opt_state, jax_rng, loss, gnorm, dt = _run_step(jax_rng, buf_obs, buf_act)
                    step += 1; emitted += 1
                    print(f"  step {step:04d}/{n_steps}  loss={loss:.4f}  "
                          f"|g|={gnorm:.3e}  ({dt:.1f}s)", flush=True)
                    buf_obs, buf_act = [], []
                    if emitted >= n_full // args.minibatch:
                        break
            del obs_pool
            _gc.collect()
            if emitted >= n_full // args.minibatch:
                break

    # Save ONLY the trained LoRA adapter (+ base.txt), like flow_grpo_train.
    import gc
    import orbax.checkpoint as ocp
    del index, rows_by_trace
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
