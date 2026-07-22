# Flow-SDE + GRPO on π0.5 — openpi port spec

Turns π0.5's flow-matching head into a stochastic policy with a tractable log-prob so
we can run real on-policy GRPO with our safety reward. Method = Flow-GRPO
(Liu et al. 2025, arXiv:2505.05470, github.com/yifan123/flow_grpo), which πRL adapted
to VLAs. `experiments/flow_sde.py` is the tested NumPy reference for the RL MACHINERY;
this doc is the JAX/openpi port. Runs on the free 24 GB 3090 (see speedups below).

## The 4 pieces to port

### 1. Marginal-preserving ODE→SDE  (the correctness piece)
π0.5 samples via the deterministic ODE `x_{t+dt} = x_t + dt·v_θ` (pi0.py:271). Replace
with Flow-GRPO's SDE. **`experiments/flow_sde.py` is now a FAITHFUL NumPy port of their
`flow_grpo/diffusers_patch/sd3_sde_with_logprob.py`** (both the `sde` and `cps` branches),
mapped to π0.5's convention (sigma=t). The JAX port just mirrors `sde_step_with_logprob`
from that file, exactly. Use **`cps` with noise_level≈0.8** (README-recommended); keep
`sde` as the alternative. Both recover the exact ODE at noise_level=0 (tested). Do NOT
re-derive — copy the two branches verbatim from flow_sde.py.

### 2. sample_actions returns the chain + logp  (inference/serving)
`pi0.py::sample_actions` must return, alongside `x_0`: the latent chain `[x_k]`, the
sampling noises, and per-step `logp_old`. The **serving path** records these per action
chunk so training can form ratios. (Add a "sample_with_logprob" mode; keep the plain
sampler for eval.)

### 3. GRPO loss  (training)
Replace the flow-matching `loss_fn` (train.py:150) with the GRPO surrogate:
```
logp_new = recompute per-step logp of the recorded chain under the CURRENT policy
ratio    = exp(logp_new - logp_old)
loss     = -mean( min(ratio·A, clip(ratio,1±ε)·A) )   # A = group advantage
```
This is `flow_sde.grpo_surrogate` in JAX. `A` = `safe_reward.group_advantages` over the
K rollouts of each state. Broadcast the scalar advantage across denoising steps.

### 4. GRPO-Guard  (stability — add only if over-optimization appears)
Flow-GRPO reports the importance ratio is biased (mean <1, worse at low-noise steps),
so clipping is imbalanced → reward goes up while quality drops. GRPO-Guard
(arXiv:2510.22319) fixes it with RatioNorm + per-step gradient reweight. Skip initially;
add if you see reward rising while TSR/quality falls.

## Single-GPU speedups (make on-policy feasible on the 3090)

- **Denoising Reduction (Flow-GRPO §3.2):** infer with the usual N steps but TRAIN on
  far fewer denoising steps. Big cost cut, no quality loss reported.
- **Flow-GRPO-Fast:** generate a deterministic ODE trajectory; at ONE random step inject
  SDE noise to branch a group; ODE elsewhere. Stochasticity (and training) confined to
  1–2 steps → each trajectory trained 1–2× instead of N. This is the key enabler for a
  single GPU. Use a small `clip_range` with Fast or it can crash (their FAQ).
- fp16 (not bf16) for smaller logp error between collection and training.

## Wiring into OUR pipeline (glue BUILT; 2 on-box hooks remain)

```
serve π0.5 (sample_with_logprob) ──► rl_rollout.collect_group (CBF shield)
   per query: action chunk, chain, logp_old  ──► metrics.policy_trace (list[QueryTrace])
   collect_group saves it → rollout_kk_trace.npz, records trace_path in manifest   [BUILT]
      └─► safe_reward.reward_from_metrics  (robot_caused collision + cbf_activation_rate)  [BUILT]
      └─► safe_reward.group_advantages     (GRPO advantage over K rollouts)                [BUILT]
train: for each manifest row → policy_trace.load_episode_trace → grpo_training_tuples(adv) [BUILT]
        → flow_sde.flow_sde_recompute_logp(current v_fn) = logp_new                        [needs JAX]
        → flow_sde.grpo_surrogate(logp_new, logp_old, adv) → LoRA update (frozen backbone)  [needs JAX]
   repeat.  Log cbf_activation_rate per round — the headline curve.
```
Only the OPTIMIZER changes vs RWFM; reward, shield, attribution, harness, trace plumbing
are all built + tested (safe_reward, rl_rollout, policy_trace, flow_sde). The 2 remaining
pieces are JAX/on-box:

HOOK A — serving `sample_with_logprob`: π0.5 server returns, per query, (action_chunk,
   chain, logp_old, sigmas, noise_level, sde_type) using the flow_sde SDE (mirror
   flow_sde.py). Plain sampler stays for eval.
HOOK B — run_libero_trial `record_policy_trace=True`: after each π0.5 query, append a
   `policy_trace.QueryTrace(chain, logp_old, sigmas, noise_level, sde_type)` to
   `metrics.policy_trace`. collect_group already persists it (contract met).
The GRPO loss then recomputes logp_new per chain (flow_sde_recompute_logp in JAX) and
applies grpo_surrogate — replacing train.py:150. Flow-GRPO-Fast: only 1–2 chain steps
need recording/training, so QueryTrace can hold just the branched segment.

## Validation ladder (do in order, on-box)
1. `noise_level=0` → SDE output == deterministic ODE output (SDE impl correct).
2. Purely on-policy (collect == train weights, 1 batch, no grad-accum) → **ratio ≡ 1**
   (sampling and training code paths agree; Flow-GRPO README step 3).
3. `beta=0`, tiny run → reward increases (learning signal present).
4. Full loop with the safety reward → cbf_activation_rate falls, TSR holds.

## Compute note
Rollouts dominate on one GPU. Keep K and horizon small for the PoC; Flow-GRPO-Fast +
Denoising Reduction are what make it tractable. A short bounded paid-A100 burst is only
for a final scaled run, not development.
