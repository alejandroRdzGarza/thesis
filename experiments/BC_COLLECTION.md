# BC demo collection → fine-tune → benchmark

## Where to run what

| stage | needs | why |
|---|---|---|
| demo collection | **CPU only** (MuJoCo + cvxpy) | the classical expert never touches the GPU — only vCPU count matters |
| `flow_bc_train` | A40 | LoRA fine-tune of π0.5 |
| benchmark eval | A40 | serving π0.5 |

Collection is CPU-bound but should still run **on the pod**: the traces are the bulk artifact
(~4 GB clean-only for the full grid) and they have to be wherever training runs. Collecting on the
Mac means rsyncing gigabytes afterwards. At $0.44/hr an extra ~4 h of A40 idle during collection
costs under $2 — far cheaper than the transfer.

Set `--workers` from the pod's vCPU count, not its GPU:

```bash
nproc            # workers = nproc - 2
```

## Collect

```bash
source /workspace/thesis/experiments/rl_env_runpod.sh    # PYOPENGL_PLATFORM=egl for headless render
cd /workspace/thesis
PYTHONPATH=. $PY -m experiments.collect_bc_all \
    --out results_distill/bc_all --workers $(( $(nproc) - 2 ))
```

Defaults: 3 suites × 2 levels × 4 tasks, inits **0–34**. Inits **35–49 are deliberately left
out** — benchmarking the fine-tuned VLA on inits it trained on measures memorisation, not
learning.

Shards are per-(suite, level, task) and resumable: re-running skips any shard that already has a
`manifest.csv`, so an interrupted collection picks up where it stopped. Traces for failed or
colliding episodes are dropped by default (`--keep-all` to retain them) — BC discards them anyway.

The run ends with per-scene clean-demo counts. That table is the thing to read: it says which
scenes contributed training signal and which produced nothing.

## Fine-tune

```bash
PYTHONPATH=. $PY -m experiments.flow_bc_train \
    --round results_distill/bc_all --success-only --epochs 20 ...
```

`--success-only` filters on the merged manifest: `r_success > 0` **and**
`robot_caused_collision == 0`. Only demos that completed the task *and* stayed clean are imitated.

20+ epochs matters — the earlier "distillation doesn't work" result (0% success, 44% collision)
was 2-epoch underfitting, not a barrier.

## Benchmark — four arms, all on held-out inits 35–49

| arm | expectation |
|---|---|
| base π0.5, no CBF | high collision — the problem statement |
| base π0.5 + CBF | the AEGIS-matched baseline |
| **distilled, no CBF** | **the headline: collision drops at matched success** |
| distilled + CBF | success may *drop* vs base+CBF — see below |

### Expect the with-CBF arm to disappoint, and don't read it as failure

The prior full-suite result (spatial, n=60/policy) came out:

- base+noCBF: 51.7% success, 90.0% collision
- distilled+noCBF: 51.7% success, **15.0% collision** ← non-overlapping CIs, the real result
- base+CBF: 80.0% success, 11.7% collision, cbf|r| 0.589
- distilled+CBF: 38.3% success, 0.0% collision, cbf|r| **0.142**

Distillation's value is **shield-free operation**, not stacking with the shield. Once the policy
already avoids the obstacle, the shield over-corrects it off the grasp and success falls. So the
claim to build the write-up around is:

1. collision at matched success, without any filter (90% → 15%), and
2. **CBF activation rate** (0.589 → 0.142) — the model needs the shield ~4× less.

If you frame the target as "distilled+CBF beats base+CBF on success", the numbers will likely not
support it, and that's a property of the method, not a bug in the run.
