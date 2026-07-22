"""Validate flow_sde_jax.py against the NumPy reference flow_sde.py.
Run with the openpi venv (has jax + numpy):
    PYTHONPATH=. openpi/.venv/bin/python -m experiments.test_flow_sde_jax
"""
import numpy as np
import jax.numpy as jnp
from experiments.flow_sde import sde_step_with_logprob as np_step, grpo_surrogate as np_surr
from experiments.flow_sde_jax import sde_step_with_logprob as jx_step, grpo_surrogate as jx_surr

ok = True
def check(name, cond, extra=""):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name} {extra}"); ok = ok and cond

rng = np.random.default_rng(0)
H, D = 5, 7   # action chunk shape

for sde_type in ("sde", "cps"):
    for trial in range(3):
        sigma = float(rng.uniform(0.15, 0.95))
        sigma_prev = float(max(0.0, sigma - rng.uniform(0.05, 0.15)))
        nl = float(rng.uniform(0.3, 0.9))
        v = rng.standard_normal((H, D))
        sample = rng.standard_normal((H, D))
        prev = rng.standard_normal((H, D))          # recompute path → deterministic, no RNG

        ps_n, lp_n, mean_n, std_n = np_step(
            v, sigma, sigma_prev, sample, nl, sde_type=sde_type, sigma_max=1.0, prev_sample=prev)
        ps_j, lp_j, mean_j, std_j = jx_step(
            v, sigma, sigma_prev, sample, nl, key=None, sde_type=sde_type,
            sigma_max=1.0, prev_sample=prev)

        dlp = abs(float(lp_j) - float(lp_n))
        dmean = float(np.max(np.abs(np.asarray(mean_j) - mean_n)))
        dstd = abs(float(std_j) - float(std_n))
        # logp is float32 (π0.5's precision) vs float64 ref → accumulates rounding
        # over the action dims; 1e-3 is generous for float32, mean/std match to ~1e-7.
        check(f"[{sde_type}] trial{trial} logp match (float32)", dlp < 1e-3, f"(Δ={dlp:.2e})")
        check(f"[{sde_type}] trial{trial} mean match", dmean < 1e-4, f"(Δ={dmean:.2e})")
        check(f"[{sde_type}] trial{trial} std match",  dstd < 1e-5, f"(Δ={dstd:.2e})")

# sigma==1 first-step guard (no NaN/inf)
ps_j, lp_j, *_ = jx_step(np.zeros((H, D)), 1.0, 0.9, np.ones((H, D)), 0.7, key=None,
                         sde_type="sde", sigma_max=0.9, prev_sample=np.ones((H, D))*0.5)
check("sigma=1 guard finite (jax)", bool(np.isfinite(float(lp_j))))

# grpo_surrogate parity
for ln, lo, a in [(0.1, 0.0, 1.0), (np.log(2.0), 0.0, 1.0), (np.log(0.5), 0.0, -1.0)]:
    sn = float(np_surr(np.array([ln]), np.array([lo]), np.array([a])))
    sj = float(jx_surr(jnp.array([ln]), jnp.array([lo]), jnp.array([a]))[0])
    check(f"surrogate parity (ln={ln:.2f},a={a})", abs(sn - sj) < 1e-5, f"(np={sn:.3f} jax={sj:.3f})")

print("\nALL PASS — JAX port matches NumPy reference" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
