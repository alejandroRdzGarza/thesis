# RWFM Runbook — Reward-Weighted Flow-Matching, one round at a time

Everything needed to run the RL loop once the GPU is back. Rollouts run **locally**
(Mac + tunnel, exactly like the benchmark); conversion + training run **on UCL**.
Each round = collect → convert → train → serve → (eval) → repeat. `cbf_activation_rate`
and robot-caused collision rate falling across rounds while TSR holds is the result.

Files (all built + unit-tested; GPU/server parts untested but wired):
- `run_rwfm_round.py`            — collect one round of rollouts → manifest.csv (local)
- `experiments/safe_reward.py`   — reward + GRPO advantage + RWFM weights
- `experiments/rl_rollout.py`    — K-rollout groups, scoring, manifest
- `experiments/convert_rwfm_to_lerobot.py` — manifest+HDF5 → weighted LeRobot dataset
- `experiments/RWFM_TRAINING.md` — the (optional) soft-loss-weight openpi patch

---

## 0. Prereqs (each session)

- Serve base π0.5 (round 0) on UCL + tunnel up — see the `reference-ucl-server` memory.
- Round N>0: serve the checkpoint trained in round N−1 instead.

---

## 1. Collect a round (LOCAL, needs tunnel)

```bash
conda activate libero
python run_rwfm_round.py \
    --suite safelibero_spatial --level I --task 0 \
    --groups 8 --K 6 --horizon 400 \
    --positive-only \                 # start with filtered-BC; drop this later for soft weights
    --out results_rwfm/round0 2>&1 | tee results_rwfm/round0.log
```
Writes `results_rwfm/round0/manifest.csv`, per-rollout HDF5, `round_summary.json`.

**CRITICAL — check rollout diversity.** GRPO needs the K rollouts of a group to
DIFFER (flow-matching sampling noise). Inspect `round_summary.json`: if `mean_weight`
≈ 1 and rewards have ~zero spread, the server is sampling deterministically → all
advantages are 0 → no learning. Fix by ensuring the π0.5 server samples with a fresh
RNG per request (or add small action noise). Verify with:
```bash
python -c "import csv,collections; r=list(csv.DictReader(open('results_rwfm/round0/manifest.csv')));
import statistics as s;
g=collections.defaultdict(list); [g[x['group_id']].append(float(x['reward'])) for x in r];
print('per-group reward stdev:', {k: round(s.pstdev(v),3) for k,v in g.items()})"
```
Non-zero stdevs → good.

---

## 2. Send to UCL

```bash
rsync -avz --exclude='__pycache__' results_rwfm/round0 \
    <user>@<gateway>.cs.ucl.ac.uk:$BASE/thesis/results_rwfm/
# also sync the experiments/ code if changed
rsync -avz --exclude='__pycache__' experiments \
    <user>@<gateway>.cs.ucl.ac.uk:$BASE/thesis/
```

---

## 3. Convert → weighted LeRobot dataset (UCL, openpi env)

```bash
ssh <user>@<gateway>.cs.ucl.ac.uk ; ssh <gpu-node> ; bash
cd $BASE/openpi

# filter (DEFAULT) = group-relative filtered BC (RAFT/hard-RWR); no openpi change.
uv run python ../thesis/experiments/convert_rwfm_to_lerobot.py \
    --manifest ../thesis/results_rwfm/round0/manifest.csv \
    --repo-name safelibero_rwfm_round0 \
    --weight-mode filter --adv-threshold 0.0
```
> `filter` is the legitimate v1. The faithful RWR is soft per-sample loss weighting —
> land the `RWFM_TRAINING.md` patch on-box for the main result, then report the
> filter→soft-weight ablation.

---

## 4. Train (UCL GPU) — STOCK openpi, LoRA + frozen backbone

Reuse the SAME config you used for DAgger R0 (it already points a LoRA/frozen-backbone
fine-tune at a LeRobot dataset); just point it at the RWFM dataset and a new exp name.

```bash
# 1) norm stats for the new dataset
uv run scripts/compute_norm_stats.py --config-name pi05_libero_cbf

# 2) train (freeze backbone / LoRA is set by the config's trainable_filter)
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run scripts/train.py pi05_libero_cbf --exp-name rwfm_r0 --overwrite
```
Checkpoints land in `checkpoints/pi05_libero_cbf/rwfm_r0/<step>`.

> Soft per-sample loss weighting (instead of duplicate/filter) is optional and needs
> the two-line patch in `RWFM_TRAINING.md`. Not required for a first result.

---

## 5. Serve the new checkpoint + evaluate

```bash
OPENPI_DATA_HOME=$BASE/openpi_cache \
uv run scripts/serve_policy.py policy:checkpoint \
    --policy.config=pi05_libero_cbf \
    --policy.dir=$BASE/openpi/checkpoints/pi05_libero_cbf/rwfm_r0/<step>
```
Then LOCALLY, eval this checkpoint and read the headline metrics:
```bash
python run_libero_benchmark.py --suite safelibero_spatial --safety-level I \
    --all --mode both --episodes 10 --vla pi05 --replan-steps 5 --horizon 400 \
    --results-dir results_rwfm/eval_r0 2>&1 | tee results_rwfm/eval_r0.log
# cbf_activation_rate (mean_cbf_activation_rate) should be dropping vs base; TSR holding.
```

---

## 6. Next round

Serve the round-0 checkpoint (step 0), then repeat 1–5 with `--out results_rwfm/round1`,
`--repo-name safelibero_rwfm_round1`, `--exp-name rwfm_r1`. Stop when
`cbf_activation_rate` plateaus.

---

## Hyperparameters worth tuning

| knob | where | default | effect |
|---|---|---|---|
| K (rollouts/group) | run_rwfm_round | 6 | GRPO group size; ≥4 for stable advantages |
| groups | run_rwfm_round | 8 | initial-state coverage |
| reward weights | run_rwfm_round `--w-*` | 1.5/1.0/0.5/0.3 | success / collision / cbf-rate / progress |
| positive_only | run_rwfm_round | off | filtered-BC vs soft weights |
| dup-scale / dup-cap | converter | 2 / 8 | oversampling strength (duplicate mode) |
| adv-threshold | converter | 0.0 | filter cutoff |

## Known limitations (document, don't block on)
- Trajectory HDF5 (`collect_cbf_data`) saves steps where the CBF was active
  (safe_cartesian set); far-from-obstacle free-motion steps may be dropped. Fine for
  safety-behaviour learning; note it.
- Reward is episode-level (trajectory weight). Per-step credit assignment (DPPO) is
  the stretch upgrade.
