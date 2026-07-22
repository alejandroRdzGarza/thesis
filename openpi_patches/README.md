# openpi patches — flow-SDE GRPO (thesis)

The nested `openpi/` checkout is a **separate git repo** whose remote is upstream
`Physical-Intelligence/openpi` (no push access) and it is **git-ignored** by this thesis
repo. So the thesis-specific openpi changes live only in that local checkout and are **not
backed up on GitHub**. This directory is that backup: everything here is version-controlled
and pushed with the thesis repo.

## What our changes add
Flow-SDE GRPO support for making π0.5 collision-safe via on-policy RL (HOOK A + HOOK B):

| File | Change | Purpose |
|------|--------|---------|
| `src/openpi/models/flow_sde.py` | **new** | Validated Flow-GRPO SDE step / sample / logp-recompute + `grpo_surrogate` (JAX) |
| `src/openpi/models/pi0.py` | +57 | `Pi0.sample_actions_with_logprob` (HOOK A) — stochastic sampler with per-step logp; `noise_level=0` reproduces the ODE sampler |
| `src/openpi/policies/policy_logprob.py` | **new** | `PolicyWithLogprob` (HOOK B) — wraps a trained Policy to sample with logp; env-ready action via output transform, chain+logp in model space |
| `src/openpi/training/config.py` | +33 | `pi05_libero_cbf` TrainConfig — LoRA action head + frozen VLM backbone via `get_freeze_filter()` |

The two **new** files are also copied verbatim under `models/` and `policies/` here for
quick reference without applying the patch.

## Base commit
The patch is `git diff a68ede1..HEAD` in the openpi checkout, where `a68ede1` is the last
commit before the GRPO work (upstream `15a9616` + a local dep tweak). It only adds new files
and appends to existing ones, so it applies cleanly to any recent openpi.

## Apply to a fresh openpi checkout (e.g. on UCL)
```bash
cd openpi
git apply /path/to/thesis/openpi_patches/flow_sde_grpo.patch
# verify:
PYTHONPATH=. JAX_PLATFORMS=cpu .venv/bin/python -c "import openpi.models.flow_sde, openpi.policies.policy_logprob"
```

## Dependency note (not in the patch)
`a68ede1` also relaxed the JAX pin in `pyproject.toml` so it installs on macOS CPU too:
```
"jax[cuda12]==0.5.3 ; sys_platform == 'linux'"
"jax[cpu]==0.5.3   ; sys_platform == 'darwin'"
```
Not required on UCL (Linux/CUDA already matches the original pin).

## Regenerate this backup after further openpi edits
```bash
cd openpi
git diff a68ede1..HEAD -- src/ > ../openpi_patches/flow_sde_grpo.patch
cp src/openpi/models/flow_sde.py        ../openpi_patches/models/
cp src/openpi/policies/policy_logprob.py ../openpi_patches/policies/
```
