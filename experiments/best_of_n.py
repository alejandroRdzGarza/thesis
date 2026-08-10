"""best_of_n.py — sample K action chunks, simulate each, execute the safest.

WHY THIS AND NOT A BARRIER. Every safety mechanism in this project so far is bounded by what the
safety signal can perceive or express, and that bound has now been measured four ways: the CBF
shield drives gripper collisions to zero but leaves 15 of 16 residual collisions on arm links;
guided sampling inherits the same blindness (the EE stayed 0.231 m clear on episodes that still
collided); outcome selection reaches constraints the barrier cannot express (arm_link 17 -> 8);
and a safety clause in the prompt produced 27% slower behaviour with no safety benefit.

A barrier is needed to PROJECT an action. It is not needed to EVALUATE one. Rolling a candidate
forward in the simulator and checking for contact sees every body — gripper, arm links, carried
object — with no geometry to inflate, nothing to differentiate, and no QP. And because every
candidate is a genuine policy sample, nothing is ever pushed off the policy's manifold, which is
what costs projection 12 points of success (TSR 82.5% unshielded vs 70.8% with the shield stacked).

CHUNK-LEVEL LOOKAHEAD. The policy emits H actions at once and executes a prefix of them before
replanning. Scoring only the first action accepts candidates that are safe now and doom the
episode two steps later. Scoring the chunk rejects the trajectory rather than the step — and it is
nearly free, because a MuJoCo step costs under a millisecond against ~800 ms for one VLA forward.
The chunk length also supplies the lookahead horizon, so no arbitrary depth has to be justified:
action chunking, introduced for smoothness and inference cost, doubles as a safety primitive.

SELECTION. Safety is a FILTER, progress is the tiebreak — never a weighted sum. If the criterion
rewards safety alone the optimum is to stop moving, and iterating on that collapses the policy to
inaction. Among collision-free candidates the one that most reduces distance to the goal is taken;
if none is collision-free the least-bad is executed and recorded, because that failure rate is
itself the measurement that says whether safety here is a SAMPLING problem (the policy can behave
safely but does not reliably pick it) or a CAPABILITY problem (no safe behaviour exists to select).

    from experiments.best_of_n import BestOfNSelector, verify_state_restore
    verify_state_restore(env)                 # GATE — see below
    sel = BestOfNSelector(env, policy_fn, k=4)
    metrics = run_libero_trial(..., policy_fn=sel)

CORRECTNESS GATE. The whole method rests on being able to rewind the simulator exactly. If
save/restore is lossy — controller internals, contact state, warm-start caches — then each
candidate is evaluated from a slightly different world and the selection is noise.
`verify_state_restore()` checks it and must pass before any result is trusted.
"""

from __future__ import annotations

import numpy as np


def _sim_of(env):
    for attr in ("sim", "_sim"):
        s = getattr(env, attr, None)
        if s is not None:
            return s
    e = getattr(env, "env", None)
    return _sim_of(e) if e is not None else None


def _controllers(env):
    """Every OSC controller on the env, in a version-tolerant way."""
    out = []
    for r in getattr(env, "robots", []) or []:
        c = getattr(r, "controller", None)
        if c is not None:
            out.append(c)
    return out


def save_full_state(env):
    """Sim state AND controller internals.

    sim.set_state() restores physics but NOT the OSC controller, which holds its own goal pose and
    orientation reference. Rewinding physics alone leaves the controller servoing toward stale
    targets, so the next step applies a different torque than it did the first time — measured at
    1.9e-2 of qpos drift over five steps, which is three orders of magnitude above float noise and
    would make every best-of-N candidate incomparable.
    """
    import copy as _copy
    sim = _sim_of(env)
    ctrl_snap = []
    for c in _controllers(env):
        ctrl_snap.append({k: (v.copy() if isinstance(v, np.ndarray) else v)
                          for k, v in c.__dict__.items()
                          if not k.startswith("__") and not callable(v)})
    # MjSimState carries time/qpos/qvel/act but NOT these. qacc_warmstart seeds the constraint
    # solver, so restoring without it makes the next solve start from a different guess and
    # converge slightly differently; ctrl is the actuator command still latched from the last step.
    d = sim.data
    extra = {}
    for k in ("ctrl", "qacc_warmstart", "act", "qfrc_applied", "xfrc_applied"):
        v = getattr(d, k, None)
        if v is not None:
            extra[k] = np.array(v, copy=True)
    # Snapshot the ROBOT and GRIPPER objects wholesale, not just recent_* deques. Measured: the
    # divergence is largest at the FIRST step (5.96e-3) and then SHRINKS (3.77e-3 by step five),
    # i.e. a transient the controller damps out — so unrestored state is read immediately rather
    # than accumulating. The sim.data diffs were all efc_*/island_* constraint-solver scratch with
    # magnitudes like 6e306, i.e. uninitialised memory, so the loss is outside sim.data. The
    # gripper holds `current_action`, which the first step reads.
    objs = []
    for r in getattr(env, "robots", []) or []:
        for o in (r, getattr(r, "gripper", None)):
            if o is None:
                continue
            snap_o = {}
            for k, v in o.__dict__.items():
                if k.startswith("__") or callable(v):
                    continue
                try:
                    snap_o[k] = v.copy() if isinstance(v, np.ndarray) else _copy.deepcopy(v)
                except Exception:
                    pass          # unpicklable handles (sim, model refs) are shared, not state
            objs.append((o, snap_o))
    return {"sim": sim.get_state(), "ctrl": ctrl_snap, "extra": extra, "objs": objs,
            "timestep": getattr(env, "timestep", None)}


def restore_full_state(env, snap):
    import copy as _copy
    sim = _sim_of(env)
    sim.set_state(snap["sim"])
    for k, v in snap.get("extra", {}).items():
        tgt = getattr(sim.data, k, None)
        if tgt is not None:
            tgt[:] = v
    sim.forward()
    for c, d in zip(_controllers(env), snap["ctrl"]):
        for k, v in d.items():
            setattr(c, k, v.copy() if isinstance(v, np.ndarray) else v)
    for o, d in snap.get("objs", []):
        for k, v in d.items():
            try:
                setattr(o, k, v.copy() if isinstance(v, np.ndarray) else _copy.deepcopy(v))
            except Exception:
                pass
    if snap["timestep"] is not None:
        env.timestep = snap["timestep"]


def verify_state_restore(env, n_steps: int = 5, tol: float = 1e-9) -> bool:
    """Rewind must be exact, or every candidate is scored from a different world.

    Executes the same action sequence twice from the same saved state. If the resulting qpos
    differs, save/restore is lossy and best-of-N selection is measuring simulator drift rather
    than action safety.
    """
    sim = _sim_of(env)
    if sim is None:
        print("  FAIL: could not locate the MuJoCo sim on this env")
        return False

    rng = np.random.default_rng(0)
    acts = [rng.uniform(-0.3, 0.3, 7) for _ in range(n_steps)]
    acts = [np.concatenate([a[:6], [-1.0]]) for a in acts]      # keep the gripper open

    st = save_full_state(env)
    for a in acts:
        env.step(a)
    q1 = np.array(sim.get_state().qpos, copy=True)

    restore_full_state(env, st)
    for a in acts:
        env.step(a)
    q2 = np.array(sim.get_state().qpos, copy=True)

    restore_full_state(env, st)
    d = float(np.max(np.abs(q1 - q2)))
    ok = d <= tol
    print(f"  state restore: max|qpos delta| over {n_steps} steps = {d:.3e} -> "
          f"{'EXACT' if ok else 'LOSSY'}")
    if not ok:
        print("    Do NOT trust best-of-N results. Candidates would be scored from different\n"
              "    worlds, so the selection measures drift, not safety. Likely causes: controller\n"
              "    internal state (OSC targets) or contact warm-start caches not covered by\n"
              "    sim.set_state(). Reset the controller alongside the sim state.")
    return ok


class BestOfNSelector:
    """policy_fn wrapper: sample K chunks, simulate, execute the safest.

    Wraps a policy_fn with the same signature the runner expects, so it is a drop-in.
    `stats` accumulates the numbers that make a null result interpretable — above all
    `no_safe_rate`, the fraction of queries where NO candidate was collision-free.
    """

    def __init__(self, env, policy_fn, k: int = 4, exec_steps: int = 5,
                 score_full_chunk: bool = False, obstacle_pos=None, goal_pos=None):
        self.env, self.policy_fn, self.k = env, policy_fn, int(k)
        self.exec_steps = int(exec_steps)
        self.score_full_chunk = bool(score_full_chunk)
        self.obstacle_pos = None if obstacle_pos is None else np.asarray(obstacle_pos, float)
        self.goal_pos = None if goal_pos is None else np.asarray(goal_pos, float)
        self.stats = {"queries": 0, "no_safe": 0, "k_safe": [], "picked_rank": []}

    def _obstacle_moved(self, base_qpos) -> float:
        sim = _sim_of(self.env)
        return float(np.max(np.abs(np.array(sim.get_state().qpos) - base_qpos)))

    def _simulate(self, chunk, n_exec):
        """Execute a candidate and report (max obstacle displacement, final EE position)."""
        sim = _sim_of(self.env)
        base = np.array(sim.get_state().qpos, copy=True)
        worst = 0.0
        for a in chunk[:n_exec]:
            self.env.step(np.asarray(a, float))
            worst = max(worst, self._obstacle_moved(base))
        ee = None
        try:
            ee = np.array(self.env._eef_xpos, copy=True)
        except Exception:
            pass
        return worst, ee

    def __call__(self, img, wrist, state, instruction, num_actions):
        saved = save_full_state(self.env)
        n_exec = None if self.score_full_chunk else self.exec_steps

        cands = []
        for _ in range(self.k):
            chunk, trace = self.policy_fn(img, wrist, state, instruction, num_actions)
            arr = np.asarray(chunk, float)
            ne = len(arr) if n_exec is None else min(n_exec, len(arr))
            disp, ee = self._simulate(arr, ne)
            restore_full_state(self.env, saved)          # rewind physics AND controller
            prog = (float(np.linalg.norm(ee - self.goal_pos))
                    if (ee is not None and self.goal_pos is not None) else 0.0)
            cands.append({"chunk": chunk, "trace": trace, "disp": disp, "prog": prog})

        restore_full_state(self.env, saved)

        # Safety FILTERS, progress only breaks ties. A weighted sum would let a large progress
        # gain buy a collision, and rewarding safety alone makes standing still optimal.
        safe = [c for c in cands if c["disp"] <= 1e-3]     # the benchmark's own 1 mm threshold
        self.stats["queries"] += 1
        self.stats["k_safe"].append(len(safe))
        if safe:
            best = min(safe, key=lambda c: c["prog"])
        else:
            self.stats["no_safe"] += 1
            best = min(cands, key=lambda c: c["disp"])     # least-bad; recorded, not hidden
        self.stats["picked_rank"].append(cands.index(best))
        return best["chunk"], best["trace"]

    def summary(self) -> dict:
        q = max(1, self.stats["queries"])
        ks = self.stats["k_safe"]
        return {
            "queries": self.stats["queries"],
            "no_safe_rate": self.stats["no_safe"] / q,
            "mean_safe_candidates": float(np.mean(ks)) if ks else 0.0,
            "all_k_safe_rate": (sum(1 for n in ks if n == self.k) / q) if ks else 0.0,
        }
