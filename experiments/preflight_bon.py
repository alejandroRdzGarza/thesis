"""preflight_bon.py — validate EVERY prerequisite for best-of-N in one cheap run.

Written after six separate debugging rounds, each costing a ~60 s policy load to discover one more
SafeLIBERO quirk: the robot's own joints counted as obstacle displacement, a parked obstacle
free-falling off-scene, the "_main" body-name suffix, the two competing venvs, swallowed
diagnostics. Every one of those produced a CONFIDENT WRONG ANSWER rather than an error — "0/4
candidates safe" reads as a capability finding, not a broken measurement.

So this checks all of them at once, up front, and none of it needs the VLA:

  1. environment    — the sim stack imports (this is the venv check)
  2. suites         — SafeLIBERO registers, so the scene can be built at all
  3. rewind         — save/restore is exact, or candidates are scored from different worlds
  4. obstacle       — detected after settling, name resolves in MuJoCo, and it is ON-SCENE
  5. measurement    — the two directions that matter, and the ones previously got wrong:
                        (a) moving the ROBOT must NOT register obstacle displacement
                        (b) moving the OBSTACLE must register it
                      Check (a) is the one that failed silently and produced "nothing is safe".

Only after all five pass is it worth loading a 3 B-parameter policy.

    python -m experiments.preflight_bon --suite safelibero_spatial --level II --task 0 --episode 35
"""

from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("PYTHONWARNINGS", "ignore")
warnings.filterwarnings("ignore")

import numpy as np

OK, BAD = "  [ok]  ", "  [FAIL]"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episode", type=int, default=35)
    ap.add_argument("--settle", type=int, default=60)
    args = ap.parse_args()
    fails = []

    # 1 ── environment ────────────────────────────────────────────────────────────────────
    try:
        import gym, robosuite, mujoco, bddl              # noqa: F401
        import sys
        print(f"{OK} sim stack imports (python = {sys.executable})")
    except Exception as e:
        print(f"{BAD} sim stack: {type(e).__name__}: {e}")
        print("         wrong venv — run: source experiments/rl_env_runpod.sh")
        return 1

    # 2 ── suites ─────────────────────────────────────────────────────────────────────────
    from libero.libero import benchmark
    suites = sorted(k for k in benchmark.get_benchmark_dict() if k.startswith("safelibero_"))
    if args.suite in suites:
        print(f"{OK} SafeLIBERO suites registered: {suites}")
    else:
        print(f"{BAD} {args.suite} not registered (found {suites})")
        return 1

    from experiments.libero_runner import make_libero_env, detect_safelibero_obstacle
    from experiments.best_of_n import save_full_state, restore_full_state, _sim_of

    env, lang, inits = make_libero_env(task_suite=args.suite, task_idx=args.task,
                                       safety_level=args.level, horizon=300)
    env.reset(); obs0 = env.set_init_state(inits[args.episode])
    sim = _sim_of(env)

    # 3 ── rewind fidelity ────────────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    acts = [np.concatenate([rng.uniform(-0.3, 0.3, 6), [-1.0]]) for _ in range(5)]
    snap = save_full_state(env)
    for a in acts: env.step(a)
    q1 = np.array(sim.get_state().qpos, copy=True)
    restore_full_state(env, snap)
    for a in acts: env.step(a)
    q2 = np.array(sim.get_state().qpos, copy=True)
    restore_full_state(env, snap)
    d = float(np.max(np.abs(q1 - q2)))
    if d <= 1e-9:
        print(f"{OK} rewind exact (max|dq| = {d:.1e})")
    else:
        print(f"{BAD} rewind LOSSY (max|dq| = {d:.1e}) — candidates would be scored from "
              f"different worlds"); fails.append("rewind")

    # 4 ── obstacle: settle first, resolve the name, confirm on-scene ─────────────────────
    _z = np.zeros(7); _z[6] = -1.0
    for _ in range(args.settle):
        env.step(_z)
    ob = detect_safelibero_obstacle(env, obs0)
    name = (getattr(ob, "body_name", None) or getattr(ob, "name", None)) if ob else None
    if not name:
        print(f"{BAD} no obstacle detected after {args.settle} settle steps"); return 1

    bid = None
    for cand in (f"{name}_main", name):                 # LIBERO suffixes the MuJoCo body
        try:
            bid = sim.model.body_name2id(cand); resolved = cand; break
        except Exception:
            pass
    if bid is None:
        for i in range(sim.model.nbody):
            if name in (sim.model.body_id2name(i) or ""):
                bid, resolved = i, sim.model.body_id2name(i); break
    if bid is None:
        print(f"{BAD} '{name}' does not resolve to any MuJoCo body"); return 1

    z = float(sim.data.body_xpos[bid][2])
    if z > 0.3:
        print(f"{OK} obstacle '{resolved}' on-scene (z = {z:.3f})")
    else:
        print(f"{BAD} obstacle '{resolved}' is PARKED (z = {z:.1f}) — SafeLIBERO dumps unused "
              f"obstacles off-scene, where they free-fall and make every candidate look unsafe")
        fails.append("parked")

    # 5 ── the measurement itself, both directions ───────────────────────────────────────
    def obs_xpos():
        return np.array(sim.data.body_xpos[bid], copy=True)

    # (a) the robot moving must NOT count as obstacle displacement. This is the check that would
    #     have caught scoring against whole-qpos, which made every candidate unsafe.
    snap = save_full_state(env)
    p0 = obs_xpos()
    for a in acts: env.step(a)
    moved_by_robot = float(np.linalg.norm(obs_xpos() - p0))
    restore_full_state(env, snap)
    if moved_by_robot < 1e-3:
        print(f"{OK} robot motion alone does not displace the obstacle ({moved_by_robot*1000:.2f} mm)")
    else:
        print(f"{BAD} robot motion registers {moved_by_robot*1000:.1f} mm of obstacle "
              f"displacement — the measurement is picking up something other than the obstacle")
        fails.append("measurement-a")

    # (b) actually moving the obstacle MUST register, or the metric is dead and everything
    #     would look safe — the opposite failure, equally invisible.
    snap = save_full_state(env)
    st = sim.get_state()
    jadr = sim.model.body_jntadr[bid]
    if jadr >= 0:
        qadr = sim.model.jnt_qposadr[jadr]
        st.qpos[qadr + 2] += 0.05                        # lift it 5 cm
        sim.set_state(st); sim.forward()
        moved_by_hand = float(np.linalg.norm(obs_xpos() - p0))
        restore_full_state(env, snap)
        if moved_by_hand > 1e-3:
            print(f"{OK} displacing the obstacle registers ({moved_by_hand*1000:.0f} mm)")
        else:
            print(f"{BAD} moving the obstacle registers nothing — metric is dead, everything "
                  f"would score as safe"); fails.append("measurement-b")
    else:
        print(f"{OK} obstacle is static (no free joint) — nothing to displace-test")

    print()
    if fails:
        print(f"  {len(fails)} CHECK(S) FAILED: {fails}")
        print("  Do not run best-of-N until these pass — each of them produces a confident wrong")
        print("  answer rather than an error.")
        return 1
    print("  ALL CHECKS PASSED — best-of-N will measure action safety, not an artefact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
