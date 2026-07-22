# RWFM trainer hook (step 3) — apply on UCL when the GPU is back

Reward-Weighted Flow-Matching needs exactly two changes to openpi: (1) each
training sample carries a scalar `weight` (its trajectory's RWFM weight), and
(2) the flow-matching loss is reduced with a **weighted** mean instead of a plain
mean. Nothing about the model or sampling changes.

## Pipeline

```
rl_rollout.run_collection(...)                     # K rollouts/group + CBF shield
  → per-group .h5 trajectories + manifest.csv      # traj_path, weight (=exp(A/τ))
convert trajectories → LeRobot dataset (+ weight)  # reuse convert_cbf_npz_to_lerobot.py
  → add a per-sample `weight` field from manifest  # all timesteps of a traj share it
openpi train (patched loss)                        # weighted flow-matching fine-tune
  → new checkpoint → serve → eval → next round     # cbf_activation_rate should fall
```

## Change 1 — the loss reduction (openpi/scripts/train.py, ~line 150)

Current:
```python
def loss_fn(model, rng, observation, actions):
    chunked_loss = model.compute_loss(rng, observation, actions, train=True)  # [*b, ah]
    return jnp.mean(chunked_loss)
```

RWFM (weighted mean; `weights` is [*b], one per sample, from the batch):
```python
def loss_fn(model, rng, observation, actions, weights):
    chunked_loss = model.compute_loss(rng, observation, actions, train=True)  # [*b, ah]
    per_sample = jnp.mean(chunked_loss, axis=-1)                              # [*b]
    # weighted mean; falls back to plain mean if all weights are 1.
    return jnp.sum(per_sample * weights) / (jnp.sum(weights) + 1e-8)
```
`compute_loss` itself (pi0.py:214, `jnp.mean((v_t-u_t)^2, axis=-1)`) is unchanged —
we only change how the batch is reduced. Thread `weights = batch["weight"]` from the
train_step batch into `loss_fn`.

## Change 2 — carry `weight` through the data pipeline (exact injection points)

The batch is a plain dict; the final handoff is `DataLoaderImpl.__iter__`
(data_loader.py:540):
```python
yield _model.Observation.from_dict(batch), batch["actions"]
```
Three edits, in order:

1. **Dataset**: add a scalar `weight` column when converting (per-trajectory, from
   manifest). Use `convert_rwfm_to_lerobot.py --weight-mode column` (add this mode:
   write `"weight"` into each `add_frame`).

2. **Survive the input transform**: `LiberoInputs` (the model input transform) returns
   a fixed dict, so a raw `weight` column is dropped. Add a one-line passthrough — in
   `LeRobotLiberoDataConfig`'s transform stack (or a tiny custom transform that runs
   last), copy `data["weight"]` into the output dict. This is the only edit that needs
   care; verify `batch` contains `"weight"` at line 540 with a quick print.

3. **Carry + use it**:
   - data_loader.py:540 →
     `yield _model.Observation.from_dict(batch), batch["actions"], batch.get("weight")`
   - train.py `train_step`: unpack `observation, actions, weights = batch`; thread
     `weights` into `loss_fn` and use the weighted mean from Change 1. Default weights
     to `jnp.ones(batch_shape)` when the column is absent, so non-RWFM training is
     unchanged.

Realistic effort: ~2–4 small edits, ~1–2 h to wire + verify ON the GPU box (untestable
locally). Not a research problem. Start from `--weight-mode filter` (no patch) to get a
first result, then land this for the faithful RWR and the filter→soft ablation.

## Notes / choices

- **What to train:** freeze the PaliGemma backbone; LoRA on the action head +
  projector. Set via the existing LoRA config knobs (`pi05_libero` fine-tune config
  as the base). This keeps the RWFM update lightweight.
- **positive_only:** starting with `advantage_weights(..., positive_only=True)`
  (filtered BC on the above-average rollouts) is the most stable first run; switch to
  the soft `exp(A/τ)` weighting once that trains cleanly.
- **Round loop (step 4):** collect → convert → train → serve new ckpt → eval,
  logging `mean_cbf_activation_rate` and robot-caused collision rate per round. The
  headline result is these falling across rounds while TSR holds.
- **compute_norm_stats:** rerun `scripts/compute_norm_stats.py --config-name <cfg>`
  on the RWFM dataset before the first training run.
