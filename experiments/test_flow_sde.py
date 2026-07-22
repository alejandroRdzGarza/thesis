"""Checks for flow_sde.py — faithful Flow-GRPO port.
Run: python -m experiments.test_flow_sde"""
import numpy as np
from experiments.flow_sde import (
    make_sigmas, sde_step_with_logprob,
    flow_sde_sample, flow_sde_recompute_logp, grpo_surrogate,
)

ok = True
def check(name, cond):
    global ok; print(f"  [{'PASS' if cond else 'FAIL'}] {name}"); ok = ok and cond

D = 7
sigmas = make_sigmas(10)
c = (np.arange(D, dtype=float) - 3) * 0.1
def v_const(x, sigma): return c
x_init = np.ones(D) * 0.5

def ode_sample(v_fn, x0, sigmas):
    x = np.array(x0, float)
    for k in range(len(sigmas) - 1):
        x = x + (sigmas[k + 1] - sigmas[k]) * v_fn(x, sigmas[k])
    return x

ode = ode_sample(v_const, x_init, sigmas)          # = x_init - c for constant c

# ── Flow-GRPO validation #1: noise_level=0 recovers the deterministic ODE ──
for mode in ("sde", "cps"):
    r0 = flow_sde_sample(v_const, x_init, sigmas, noise_level=0.0,
                         rng=np.random.default_rng(0), sde_type=mode)
    check(f"[{mode}] noise_level=0 == ODE output", np.allclose(r0["action"], ode, atol=1e-9))
check("ODE constant-velocity sanity (x0 = x_init - c)", np.allclose(ode, x_init - c))

# ── stochastic exploration when noise_level>0 ──
for mode in ("sde", "cps"):
    a = flow_sde_sample(v_const, x_init, sigmas, 0.7, np.random.default_rng(1), mode)
    b = flow_sde_sample(v_const, x_init, sigmas, 0.7, np.random.default_rng(2), mode)
    check(f"[{mode}] different seeds → different actions", not np.allclose(a["action"], b["action"]))

# ── Flow-GRPO validation #2: on-policy → recompute logp == sampling logp → ratio≡1 ──
for mode in ("sde", "cps"):
    roll = flow_sde_sample(v_const, x_init, sigmas, 0.8, np.random.default_rng(3), mode)
    lp_new = flow_sde_recompute_logp(v_const, roll["chain"], sigmas, 0.8, mode)
    check(f"[{mode}] recompute logp == sampling logp (on-policy)",
          np.allclose(lp_new, roll["step_logp"], atol=1e-9))
    surr = grpo_surrogate(lp_new, roll["step_logp"], advantage=2.5)
    check(f"[{mode}] on-policy ratio≡1 → surrogate == advantage", np.allclose(surr, 2.5))

# ── a changed policy actually shifts logp (ratio ≠ 1) ──
def v_shift(x, sigma): return c + 0.05
roll = flow_sde_sample(v_const, x_init, sigmas, 0.7, np.random.default_rng(4), "sde")
lp_shift = flow_sde_recompute_logp(v_shift, roll["chain"], sigmas, 0.7, "sde")
check("changed policy → logp differs", not np.allclose(lp_shift, roll["step_logp"]))

# ── GRPO surrogate mechanics ──
check("ratio=1 → surrogate = advantage",
      np.allclose(grpo_surrogate(np.array([1.0]), np.array([1.0]), np.array([2.5])), 2.5))
check("positive adv clipped at (1+clip)*A",
      np.allclose(grpo_surrogate(np.array([np.log(2.0)]), np.array([0.0]), np.array([1.0]), 0.2), 1.2))
check("negative adv uses pessimistic (min) branch",
      np.allclose(grpo_surrogate(np.array([np.log(0.5)]), np.array([0.0]), np.array([-1.0]), 0.2), -0.8))

# ── sigma=1 first step is guarded (no div-by-zero) ──
v = np.zeros(D)
ps, lp, mean, std = sde_step_with_logprob(v, 1.0, 0.9, x_init, 0.7, "sde",
                                          sigma_max=0.9, rng=np.random.default_rng(0))
check("sigma=1 step finite (guarded)", np.all(np.isfinite(ps)) and np.isfinite(lp))

print("\nALL PASS" if ok else "\nSOME FAILED")
raise SystemExit(0 if ok else 1)
