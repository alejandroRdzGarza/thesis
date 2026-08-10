"""diag_restore.py — find WHERE simulator rewind loses information.

Three rounds of guessing (controller internals, then warm-start/ctrl/deques) moved the drift from
1.9e-2 to 3.8e-3 and then not at all, so the remaining state is somewhere not yet considered.
This stops hypothesising and measures two things directly:

  1. WHEN the divergence appears — after 1 step or only after several. Immediate divergence means
     unrestored state is read on the very first step; gradual means something accumulates.
  2. WHICH fields of sim.data differ between the two runs at the moment of divergence.

  python -m experiments.diag_restore
"""
from __future__ import annotations

import numpy as np


def main():
    from experiments.libero_runner import make_libero_env
    from experiments.best_of_n import save_full_state, restore_full_state, _sim_of

    env, lang, inits = make_libero_env(task_suite="safelibero_spatial", task_idx=0,
                                       safety_level="II", horizon=300)
    env.reset(); env.set_init_state(inits[35])
    sim = _sim_of(env)

    rng = np.random.default_rng(0)
    acts = [np.concatenate([rng.uniform(-0.3, 0.3, 6), [-1.0]]) for _ in range(5)]

    # --- when does it diverge? ---
    snap = save_full_state(env)
    run1 = []
    for a in acts:
        env.step(a); run1.append(np.array(sim.get_state().qpos, copy=True))

    restore_full_state(env, snap)
    run2 = []
    for a in acts:
        env.step(a); run2.append(np.array(sim.get_state().qpos, copy=True))

    print("\n  divergence by step:")
    for i, (x, y) in enumerate(zip(run1, run2), 1):
        print(f"    after {i} step(s): max|dq| = {np.max(np.abs(x - y)):.3e}")

    # --- what differs in sim.data right after restore vs the original start? ---
    restore_full_state(env, snap)
    fields = [f for f in dir(sim.data)
              if not f.startswith("_") and isinstance(getattr(sim.data, f, None), np.ndarray)]
    before = {f: np.array(getattr(sim.data, f), copy=True) for f in fields}

    restore_full_state(env, snap)          # restore twice: must be idempotent
    print("\n  sim.data fields that differ across two identical restores:")
    any_diff = False
    for f in fields:
        now = np.array(getattr(sim.data, f), copy=True)
        if now.shape == before[f].shape and not np.array_equal(now, before[f]):
            print(f"    {f:<24} max|delta| = {np.max(np.abs(now - before[f])):.3e}")
            any_diff = True
    if not any_diff:
        print("    none — the restore itself is idempotent, so the loss is in env/robot state")
        print("    outside sim.data (robosuite wrapper, gripper, or observation caches).")

    print("\n  READ IT LIKE THIS")
    print("   step 1 already large  -> unrestored state is READ on the first step")
    print("   grows with steps      -> something accumulates; check the controller's ramp/filter")
    print("   restore idempotent    -> the missing state is NOT in sim.data\n")


if __name__ == "__main__":
    main()
