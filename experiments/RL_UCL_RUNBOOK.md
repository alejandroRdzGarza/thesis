# Flow-SDE GRPO — UCL run-book

Everything is code-complete and CPU-validated. This is the checklist to run it on the UCL GPU.
The loop: `rl_rollout_local` (CBF-shielded stochastic rollouts + scored traces) →
`flow_grpo_train` (GRPO update over the LoRA action head → next checkpoint) → repeat.
`run_grpo_round.sh` chains one full round.

## 0. The one hard prerequisite: a SINGLE unified environment
The co-located rollout (Option B) runs env + model + CBF in one process, so **one Python must
import all of**:
- `jax` + `openpi` (model, flow-SDE sampler, GRPO loss)
- `libero`, `robosuite`, `mujoco` (SafeLIBERO env)  — headless via `MUJOCO_GL=egl`
- `cvxpy` (CBF QP; without it the CBF silently drops to a weak SLSQP fallback — we warn, but it must be installed for faithful numbers)
- `orbax` (checkpoint save — already an openpi dep)

On the Mac these are split across two venvs, which is why only the JAX half was CPU-tested here.
On UCL, verify in ONE interpreter before anything else:
```bash
MUJOCO_GL=egl python -c "import jax, openpi, libero, robosuite, mujoco, cvxpy, orbax; \
print('unified env OK', jax.devices())"
```
If that prints a GPU device and no ImportError, you're good. If `libero`/`robosuite` live in a
different venv than `openpi`, install openpi (`uv pip install -e openpi` + the patch) into the
libero env, or install libero/robosuite into the openpi `.venv`.

## 1. Sync the code
```bash
# thesis repo (experiments/, run_grpo_round.sh)
git pull

# openpi changes are NOT on GitHub (upstream remote) — apply the patch to the UCL openpi:
cd $BASE/openpi
git apply /path/to/thesis/openpi_patches/flow_sde_grpo.patch
python -c "import openpi.models.flow_sde, openpi.policies.policy_logprob, openpi.training.flow_grpo"
```
(Applies cleanly to a recent openpi — verified with `git apply --check`.)

## 2. Base checkpoint
`--checkpoint` needs a dir with `params/` + `assets/` (norm stats). The base π0.5 LIBERO
checkpoint (`gs://openpi-assets/checkpoints/pi05_libero`) — point `--checkpoint` at the local
copy (e.g. under `$BASE/openpi_cache`). `create_trained_policy` loads `params/` and reads norm
stats from `assets/`.

## 3. Smoke test (do this FIRST — tiny + fast)
Confirms the real model forward, headless render, CBF shield, trace save, and one GRPO step all
work together before spending a full round:
```bash
cd /path/to/thesis
export MUJOCO_GL=egl PYTHONPATH=.
# 1 episode, 2 rollouts, short horizon
python -m experiments.rl_rollout_local \
    --config pi05_libero --checkpoint $CKPT \
    --suite safelibero_object --level II --task 0 \
    --episodes 0 --K 2 --horizon 120 --out results_grpo/smoke
# then one update:
python -m experiments.flow_grpo_train \
    --config pi05_libero_cbf --checkpoint $CKPT \
    --round results_grpo/smoke --out results_grpo/smoke_ckpt --minibatch 2
```
Expect: `results_grpo/smoke/manifest.csv` + `*_trace.npz` (with obs), then a training log with
`loss` / `|g|` and `results_grpo/smoke_ckpt/params` written. Sanity: at step 0 the loss should be
≈ `-mean(advantage)` (on-policy ratio ≈ 1).

## 4. A full round
```bash
CKPT=$CKPT ROUND=0 SUITE=safelibero_object LEVEL=II TASK=0 \
  EPISODES="0 1 2 3" K=8 ./run_grpo_round.sh
# → results_grpo/round1_ckpt ; then feed it back:
CKPT=results_grpo/round1_ckpt ROUND=1 ... ./run_grpo_round.sh
```
Step 3 of the script runs a **no-CBF** eval of the updated policy — that's the headline safety
number.

## 5. What to watch (the thesis metrics)
- **`cbf_activation_rate`** (in the WITH-CBF rollout `round_summary.json`): should DROP across
  rounds — the model needs the shield less as it internalizes safety.
- **collision rate** (in the `*_eval_nocbf/round_summary.json`, `robot_caused_collision_rate`):
  should stay LOW without the shield — proof the safety is internalized, not just filtered.
- **success_rate**: must not collapse — reward is tuned so clean success beats collided success.

## 6. Knobs (Flow-GRPO-Fast)
- `NUM_STEPS` 10→ try 4 then 1-2 (Flow-GRPO-Fast: train on few denoising steps for compute).
- `NOISE_LEVEL` 0.7 (cps recommends ~0.8) — exploration; eval uses 0 (ODE).
- `K` group size (more = better advantage estimate), `MINIBATCH`, `LR` 2e-4.
- `SDE_TYPE` cps (recommended) | sde.

## 7. Gotchas
- **Success criterion**: standard LIBERO by default now. `LIBERO_LENIENT_ONTOP=1` re-enables the
  old 6cm placement tolerance (NOT AEGIS-comparable — leave unset for thesis numbers).
- **CBF solver**: ensure `cvxpy` imports, else the shield is weak (watch for the SLSQP warning).
- **OOM**: π0.5 + rollout + GRPO on one GPU. LoRA keeps trainable params tiny; if the update
  OOMs, lower `MINIBATCH`. If rollout OOMs, lower `K` (rollouts are sequential anyway).
- **Headless**: `MUJOCO_GL=egl` must be exported in every shell that renders.
