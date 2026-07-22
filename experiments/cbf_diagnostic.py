"""
cbf_diagnostic.py — Standalone CBF sanity harness (NO VLA server needed).

Purpose
-------
The full benchmark couples three things together — π0.5, the OSC controller,
and the CBF — so when "the arm collides" you can't tell *which* one is at fault.
This script strips π0.5 out entirely and drives the arm with a deterministic
scripted policy, so any collision is unambiguously the CBF's fault.

It runs three independent checks, each isolating one assumption the CBF makes:

  TEST A — Controller gain calibration
    The CBF QP treats `action[:3]` as if it equals the EE displacement in
    metres for that control step (scale=1.0 passthrough, same implicit model
    as AEGIS). This test commands constant actions and measures the ACTUAL
    metres moved per unit action. If the real gain g ≠ ~1, the CBF's braking
    model is wrong by exactly that factor and every downstream number is off.

  TEST B — Static barrier geometry
    Pure math, no sim. Sweeps a synthetic EE position along the line into the
    obstacle and checks that h_min crosses zero right around physical contact
    (|ee - obs| ≈ r_ee + r_obs). Validates units / sphere-decomp geometry.

  TEST C — Closed-loop adversarial drive  (THE decisive test)
    A scripted policy commands a constant action pointing straight AT the
    obstacle centre, gripper closed. Runs the exact same `run_sphere_decomp_cbf`
    path the benchmark uses. Then runs the identical drive with the CBF OFF as
    a control.
        CBF off  → MUST collide  (proves the adversary really attacks)
        CBF on   → MUST NOT      (proves the CBF stops it)
    If CBF-on still collides, the per-step logs (h_min, triggered, action sent,
    surface distance) show exactly where the barrier failed.

Usage
-----
    python -m experiments.cbf_diagnostic \
        --suite safelibero_spatial --level II --task 0 --episode 0 \
        --out results_cbf_diag

Everything runs locally on MuJoCo; no GPU, no tunnel, no policy server.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as SciRot

from experiments.libero_runner import (
    make_libero_env,
    detect_safelibero_obstacle,
    _unwrap_sim,
    _get_arm_body_ids,
    _get_arm_dof_indices,
    _compute_arm_link_constraints,
    _obstacle_contact_culprits,
    robot_caused_displacement,
    _DUMMY_ACTION,
    _WARMUP_STEPS,
)
from experiments.cbf_ellipsoid import (
    run_sphere_decomp_cbf,
    run_ellipsoid_cbf,
    init_z,
    get_ee_spheres,
    EE_Q_DIAG_DEFAULT,
    EE_SPHERE_RADIUS,
    K_CBF,
)

try:
    from experiments.cbf_visualizer import decompose_obstacle_to_spheres
    _HAS_VIZ = True
except Exception:
    _HAS_VIZ = False


# Collision threshold used by the benchmark (sum of abs obstacle displacement).
COLLISION_THRESH = 0.002   # metres


# ─────────────────────────────────────────────────────────────────────────────
# Small helpers
# ─────────────────────────────────────────────────────────────────────────────
def _ee_pos(obs) -> np.ndarray:
    return np.array(obs["robot0_eef_pos"], dtype=float)


def _R1(obs) -> np.ndarray:
    q = np.array(obs.get("robot0_eef_quat", [0, 0, 0, 1]), dtype=float)
    return SciRot.from_quat(q).as_matrix()


def _step(env, action):
    out = env.step(np.asarray(action, dtype=float).tolist())
    return out[0] if isinstance(out, tuple) else out


def _warmup(env, obs):
    for _ in range(_WARMUP_STEPS):
        obs = _step(env, _DUMMY_ACTION)
    return obs


def _reset_to_start(env, init_states, episode):
    """Hard reset to the exact episode initial state, then settle physics.

    Every test must call this so it starts from an identical, clean pose —
    otherwise leftover state (and physics blow-ups) from a previous drive
    corrupt the measurement.
    """
    env.reset()
    if init_states is not None:
        obs = env.set_init_state(init_states[episode])
    else:
        r = env.reset()
        obs = r[0] if isinstance(r, tuple) else r
    return _warmup(env, obs)


def _obstacle_surface_dist(ee, spheres) -> float:
    """Signed-ish min distance from any EE sphere to any obstacle sphere surface."""
    ee_spheres = spheres["ee"](ee)
    best = float("inf")
    for ee_c, ee_r in ee_spheres:
        for c_j, r_j in spheres["obs"]:
            best = min(best, float(np.linalg.norm(ee_c - c_j) - (ee_r + r_j)))
    return best


# ─────────────────────────────────────────────────────────────────────────────
# Environment / obstacle setup shared by all tests
# ─────────────────────────────────────────────────────────────────────────────
def setup(suite, level, task, episode, geom="ellipsoid"):
    env, language, init_states = make_libero_env(
        task_suite=suite, task_idx=task, safety_level=level, horizon=1000,
    )
    obs = _reset_to_start(env, init_states, episode)

    model, data = _unwrap_sim(env)
    arm_body_ids = _get_arm_body_ids(model)
    arm_dof_idx = _get_arm_dof_indices(model)
    ee_body_id = arm_body_ids[-1] if arm_body_ids else 0

    obstacle = detect_safelibero_obstacle(env, obs, safety_radius=0.18)
    if obstacle is None:
        raise SystemExit(
            "No SafeLIBERO obstacle detected — check suite/level/task.")

    sphere_decomp = None
    if _HAS_VIZ:
        for suffix in ("_main", ""):
            sphere_decomp = decompose_obstacle_to_spheres(
                model, data, f"{obstacle.name}{suffix}",
                n_spheres=48, r_sphere=0.015, safety_margin=0.010,
            )
            if sphere_decomp is not None:
                break
    if sphere_decomp is None:
        raise SystemExit(
            f"Sphere decomposition failed for '{obstacle.name}'. "
            "This diagnostic only covers the sphere-decomp path.")

    # Which QP backend is actually live? AEGIS uses cvxpy/OSQP. If cvxpy is
    # missing the CBF silently falls back to scipy SLSQP, which fails on large
    # constraint sets and then passes the RAW action through uncorrected.
    import experiments.cbf_ellipsoid as _cbf_mod
    backend = "cvxpy / OSQP" if _cbf_mod._HAS_CVXPY else "scipy SLSQP  (FALLBACK!)"

    print(f"\n  Task        : {language}")
    print(f"  Obstacle    : {obstacle.name}  pos={np.round(obstacle.pos, 3)}")
    print(f"  Spheres     : {len(sphere_decomp)} obstacle spheres")
    print(f"  EE start    : {np.round(_ee_pos(obs), 3)}")
    print(f"  K_CBF       : {K_CBF}   (AEGIS reference: 10.0)")
    print(f"  QP backend  : {backend}")
    if not _cbf_mod._HAS_CVXPY:
        print("  ⚠  cvxpy NOT installed → CBF runs scipy SLSQP. With ~144 sphere")
        print("     constraints SLSQP often fails to converge; on failure the raw")
        print("     action is passed through UNCORRECTED. Install cvxpy to match AEGIS.")

    spheres = {
        "obs": sphere_decomp,
        "ee": lambda ee: get_ee_spheres(ee, _R1_cache["R1"]),
    }
    ctx = dict(
        env=env, obs=obs, model=model, data=data,
        arm_body_ids=arm_body_ids, arm_dof_idx=arm_dof_idx,
        ee_body_id=ee_body_id, obstacle=obstacle,
        sphere_decomp=sphere_decomp, language=language,
        init_states=init_states, episode=episode, geom=geom,
    )
    print(f"  CBF geometry: {geom}"
          + ("  (AEGIS-faithful — matches runner default)"
             if geom == "ellipsoid" else "  (legacy sphere-decomposition)"))
    if geom == "ellipsoid" and ctx_obstacle_q_missing(obstacle):
        print("  ⚠  obstacle q_diag not fitted → ellipsoid CBF falls back to an "
              "isotropic safety_radius sphere.")
    return ctx


def ctx_obstacle_q_missing(obstacle) -> bool:
    return getattr(obstacle, "q_diag", None) is None


def _reset_ctx(ctx):
    """Return a fresh obs at the episode start pose (clean physics)."""
    return _reset_to_start(ctx["env"], ctx["init_states"], ctx["episode"])


# Cache the latest R1 so the EE-sphere lambda can see it.
_R1_cache = {"R1": np.eye(3)}


# ─────────────────────────────────────────────────────────────────────────────
# TEST A — controller gain calibration
# ─────────────────────────────────────────────────────────────────────────────
def test_a_gain(ctx, n_steps=15, cmd=0.2):
    """Command a constant action on each axis, measure metres moved per step."""
    print("\n" + "=" * 68)
    print("  TEST A — CONTROLLER GAIN  (action units → metres / step)")
    print("=" * 68)
    env = ctx["env"]
    results = {}
    for axis, name in [(0, "x"), (1, "y"), (2, "z")]:
        # Hard reset so each axis starts from the identical clean start pose.
        obs = _reset_ctx(ctx)
        p0 = _ee_pos(obs)
        act = np.zeros(7)
        act[axis] = cmd
        act[6] = -1.0  # keep gripper open, no grasp dynamics
        disps = []
        prev = p0.copy()
        for _ in range(n_steps):
            obs = _step(env, act)
            now = _ee_pos(obs)
            disps.append(now[axis] - prev[axis])
            prev = now
        # Use the steady-state (middle) displacement, ignoring the first ramp.
        steady = float(np.mean(disps[3:])) if len(disps) > 4 else float(np.mean(disps))
        gain = steady / cmd if abs(cmd) > 1e-9 else 0.0
        results[name] = gain
        print(f"    axis {name}:  cmd={cmd:+.2f}  steady Δ={steady:+.4f} m/step  "
              f"→  gain g_{name} = {gain:.3f}")

    g = np.mean([abs(v) for v in results.values()])
    print(f"\n    mean |gain| ≈ {g:.3f} m per unit action per step")
    print(f"    CBF assumes gain = 1.000 (scale=1.0 passthrough).")
    if g < 0.6:
        print(f"    ⚠  Real gain is {g:.2f} << 1.0 → the CBF OVER-estimates the")
        print(f"       displacement it commands. It will brake earlier than needed")
        print(f"       (conservative, hurts TSR) but should stay safe.")
    elif g > 1.4:
        print(f"    ⚠  Real gain is {g:.2f} >> 1.0 → the CBF UNDER-estimates motion")
        print(f"       → discrete-step overshoot → COLLISIONS likely. Reduce the")
        print(f"       per-step action magnitude or lower scale in the QP.")
    else:
        print(f"    ✓  Gain ≈ 1 → scale=1.0 passthrough assumption is reasonable.")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# TEST B — static barrier geometry
# ─────────────────────────────────────────────────────────────────────────────
def test_b_geometry(ctx, n=25):
    """Sweep a synthetic EE straight into the obstacle; check h zero-crossing."""
    print("\n" + "=" * 68)
    print("  TEST B — STATIC BARRIER GEOMETRY  (pure math, no sim)")
    print("=" * 68)
    obstacle = ctx["obstacle"]
    spheres = ctx["sphere_decomp"]
    ee0 = _ee_pos(ctx["obs"])
    target = obstacle.pos
    R1 = np.eye(3)
    _R1_cache["R1"] = R1

    print(f"    Sweeping EE from {np.round(ee0,3)} → {np.round(target,3)}")
    print(f"    {'frac':>5} {'|ee-obs|':>9} {'surf_dist':>10} {'h_min':>10}  triggered?")
    crossing = None
    prev_h = None
    for i in range(n + 1):
        frac = i / n
        ee = ee0 + frac * (target - ee0)
        ee_spheres = get_ee_spheres(ee, R1)
        _, h_min, _ = run_sphere_decomp_cbf(
            ee_pos=ee, R1=R1, obstacle_spheres=spheres,
            u_nom=np.zeros(3), k_cbf=K_CBF, scale=1.0,
            ee_spheres=ee_spheres,
        )
        center_d = float(np.linalg.norm(ee - obstacle.pos))
        surf_d = min(
            float(np.linalg.norm(ec - cj) - (er + rj))
            for ec, er in ee_spheres for cj, rj in spheres
        )
        # nominal action pointing at obstacle → does QP push back?
        u_nom = 0.2 * (target - ee0) / (np.linalg.norm(target - ee0) + 1e-9)
        u_safe, _, trig = run_sphere_decomp_cbf(
            ee_pos=ee, R1=R1, obstacle_spheres=spheres,
            u_nom=u_nom, k_cbf=K_CBF, scale=1.0, ee_spheres=ee_spheres,
        )
        if i % 2 == 0 or (prev_h is not None and prev_h > 0 >= h_min):
            print(f"    {frac:5.2f} {center_d:9.3f} {surf_d:10.4f} {h_min:10.4f}  "
                  f"{'YES' if trig else 'no'}")
        if prev_h is not None and prev_h > 0 >= h_min and crossing is None:
            crossing = surf_d
        prev_h = h_min

    print()
    if crossing is None:
        print("    ⚠  h never crossed zero along the approach — geometry/units may")
        print("       be off, or the obstacle spheres don't cover the centre.")
    else:
        print(f"    h=0 crossing at surface distance ≈ {crossing:+.4f} m")
        if abs(crossing) < 0.03:
            print("    ✓  Zero-crossing is at ~physical contact → geometry is sane.")
        else:
            print(f"    ⚠  Crossing is {crossing:+.3f} m from contact — h is offset")
            print("       (over/under-conservative by that margin).")


# ─────────────────────────────────────────────────────────────────────────────
# TEST C — closed-loop adversarial drive
# ─────────────────────────────────────────────────────────────────────────────
def _cbf_step(ctx, ee, R1, u_nom, z):
    """Apply one CBF correction using the configured geometry.

    Returns (u_safe_world, h_min, triggered, z_new). For the ellipsoid geometry
    this exercises the exact AEGIS-faithful path the runner now uses by default.
    """
    obstacle = ctx["obstacle"]
    if ctx["geom"] == "ellipsoid":
        obs_q = (obstacle.q_diag if obstacle.q_diag is not None
                 else np.array([obstacle.safety_radius] * 3))
        u_safe, z_new, h, trig = run_ellipsoid_cbf(
            ee_pos=ee, R1=R1, obs_pos=obstacle.pos, obs_q=obs_q, z=z,
            u_nom=u_nom, k_cbf=K_CBF, scale=1.0,
            obs_R=getattr(obstacle, "q_R", None),
        )
        return u_safe, h, trig, z_new
    u_safe, h, trig = run_sphere_decomp_cbf(
        ee_pos=ee, R1=R1, obstacle_spheres=ctx["sphere_decomp"],
        u_nom=u_nom, k_cbf=K_CBF, scale=1.0,
        ee_spheres=get_ee_spheres(ee, R1),
    )
    return u_safe, h, trig, z


def _adversarial_drive(ctx, use_cbf, cmd_mag, horizon, out_csv=None,
                       gate_goal=None, gate_tol=0.08):
    """Drive constant action toward obstacle centre; return summary dict.

    If gate_goal is given, replicate the runner's `_near_goal` gate: whenever the
    EE is within gate_tol of gate_goal, the CBF is DISABLED (raw action passes
    through) — exactly as libero_runner does. Used by Test D to expose the blind
    spot the gate opens around a goal-adjacent obstacle.
    """
    env = ctx["env"]
    obstacle = ctx["obstacle"]
    spheres = ctx["sphere_decomp"]
    model, data = ctx["model"], ctx["data"]

    # Hard reset to the identical clean episode start pose.
    obs = _reset_ctx(ctx)
    n_gated = 0

    obs_key = f"{obstacle.name}_pos"
    init_obs_pos = np.array(obs.get(obs_key, obstacle.pos), dtype=float)

    rows = []
    collided = False
    min_surf = float("inf")
    min_h = float("inf")
    n_trig = 0
    culprits: set[str] = set()               # what touches the obstacle at collision
    robot_caused = False                      # robot caused it via contact chain
    z = init_z(_ee_pos(obs), obstacle.pos)   # ellipsoid auxiliary direction

    for t in range(horizon):
        ee = _ee_pos(obs)
        R1 = _R1(obs)
        _R1_cache["R1"] = R1

        # Nominal adversarial action: unit vector EE → obstacle centre, scaled.
        d = obstacle.pos - ee
        d = d / (np.linalg.norm(d) + 1e-9)
        u_nom = cmd_mag * d

        action = np.zeros(7)
        action[6] = 1.0  # gripper closed (matches worst-case reach)
        h_min = float("inf")
        triggered = False

        # Replicate the runner's _near_goal gate: CBF off near the goal.
        gated_off = (gate_goal is not None
                     and float(np.linalg.norm(ee - gate_goal)) < gate_tol)
        if gated_off:
            n_gated += 1

        if use_cbf and not gated_off:
            # AEGIS-faithful: no arm-link rows, no post-trigger slowdown.
            u_safe, h_min, triggered, z = _cbf_step(ctx, ee, R1, u_nom, z)
            if triggered:
                n_trig += 1
            action[:3] = u_safe
        else:
            action[:3] = u_nom

        obs = _step(env, action)

        ee_after = _ee_pos(obs)
        surf = _obstacle_surface_dist(
            ee_after, {"obs": spheres, "ee": lambda e: get_ee_spheres(e, R1)})
        min_surf = min(min_surf, surf)
        if np.isfinite(h_min):
            min_h = min(min_h, h_min)

        curr_obs_pos = np.array(obs.get(obs_key, init_obs_pos), dtype=float)
        disp = float(np.sum(np.abs(curr_obs_pos - init_obs_pos)))
        if disp > COLLISION_THRESH:
            collided = True
            # Attribute the displacement: immediate toucher + robot causation chain.
            culprits = _obstacle_contact_culprits(model, data, obstacle.name)
            robot_caused = robot_caused_displacement(model, data, obstacle.name)

        rows.append(dict(
            t=t, ee_x=ee_after[0], ee_y=ee_after[1], ee_z=ee_after[2],
            surf_dist=surf, h_min=h_min, triggered=int(triggered),
            corr_norm=float(np.linalg.norm(action[:3] - u_nom)),
            obs_disp=disp,
        ))
        if collided:
            break

    if out_csv:
        with open(out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    return dict(
        collided=collided, min_surf=min_surf, min_h=min_h,
        n_trig=n_trig, n_steps=len(rows),
        trig_frac=(n_trig / max(len(rows), 1)),
        n_gated=n_gated, culprits=culprits, robot_caused=robot_caused,
    )


def test_c_closed_loop(ctx, out_dir, cmd_mag=0.3, horizon=120):
    print("\n" + "=" * 68)
    print("  TEST C — ADVERSARIAL DRIVE  (scripted policy aims AT obstacle)")
    print("=" * 68)

    print("\n  [C1] CBF OFF (control — this SHOULD collide):")
    off = _adversarial_drive(ctx, use_cbf=False, cmd_mag=cmd_mag,
                             horizon=horizon,
                             out_csv=os.path.join(out_dir, "drive_cbf_off.csv"))
    print(f"       collided={off['collided']}  min_surf_dist={off['min_surf']:+.4f} m  "
          f"steps={off['n_steps']}")

    print("\n  [C2] CBF ON  (this should NOT collide):")
    on = _adversarial_drive(ctx, use_cbf=True, cmd_mag=cmd_mag,
                            horizon=horizon,
                            out_csv=os.path.join(out_dir, "drive_cbf_on.csv"))
    print(f"       collided={on['collided']}  min_surf_dist={on['min_surf']:+.4f} m  "
          f"min_h={on['min_h']:+.4f}  activations={on['n_trig']}/{on['n_steps']} "
          f"({on['trig_frac']*100:.0f}%)")

    print("\n" + "-" * 68)
    print("  VERDICT")
    print("-" * 68)
    if not off["collided"]:
        print("  ✗ INCONCLUSIVE: the adversary never collided even without the CBF.")
        print("    The scripted action didn't actually reach the obstacle — increase")
        print("    --cmd-mag or --horizon, or the start pose is already blocked.")
    elif on["collided"]:
        print("  ✗ CBF FAILED: obstacle moved despite the filter being active.")
        print(f"    Min surface distance reached: {on['min_surf']:+.4f} m (want > 0).")
        print( "    Inspect drive_cbf_on.csv: look for steps where h_min < 0 while")
        print( "    triggered=0 (QP didn't fire) OR triggered=1 but surf_dist still")
        print( "    dropped (per-step overshoot — the arm jumps through the barrier).")
        if on["min_h"] < 0:
            print(f"    → h_min went negative ({on['min_h']:+.4f}): barrier was violated,")
            print( "      consistent with discrete-step overshoot. Lower K_CBF (currently")
            print(f"      {K_CBF}; AEGIS uses 10) and/or cap per-step action magnitude.")
    else:
        print("  ✓ CBF WORKS: it stopped a deterministic collision course.")
        print(f"    Held the arm {on['min_surf']:+.4f} m off the obstacle surface")
        print(f"    with {on['n_trig']} corrections. The filter itself is sound —")
        print( "    any benchmark collisions come from elsewhere (e.g. the _near_goal")
        print( "    gate disabling CBF, or the ellipsoid fallback path).")


# ─────────────────────────────────────────────────────────────────────────────
# TEST D — _near_goal gate blind-spot
# ─────────────────────────────────────────────────────────────────────────────
def test_d_gate(ctx, out_dir, cmd_mag=0.3, horizon=120, gate_tol=0.08):
    """Show that the runner's _near_goal gate opens a hole around the obstacle.

    Worst case (SafeLIBERO Level I): the obstacle sits right by the placement
    goal. We put the gate goal AT the obstacle centre and re-run the adversarial
    drive with the CBF nominally ON but gated exactly like the runner. If the
    arm collides, the gate — not the barrier — is the failure.
    """
    print("\n" + "=" * 68)
    print("  TEST D — _near_goal GATE BLIND-SPOT  (CBF on, but gated like runner)")
    print("=" * 68)
    gate_goal = ctx["obstacle"].pos.copy()
    print(f"    gate goal = obstacle centre {np.round(gate_goal, 3)}  "
          f"(worst-case Level-I geometry)")
    print(f"    gate tol  = {gate_tol} m  → CBF disabled within this radius")

    res = _adversarial_drive(
        ctx, use_cbf=True, cmd_mag=cmd_mag, horizon=horizon,
        gate_goal=gate_goal, gate_tol=gate_tol,
        out_csv=os.path.join(out_dir, "drive_gated.csv"))
    print(f"\n    collided={res['collided']}  min_surf_dist={res['min_surf']:+.4f} m  "
          f"steps_gated_off={res['n_gated']}/{res['n_steps']}")

    print("\n" + "-" * 68)
    print("  VERDICT")
    print("-" * 68)
    if res["collided"]:
        print("  ✗ GATE IS THE BUG: with the CBF nominally ON, the _near_goal gate")
        print("    disabled it near the obstacle and the arm collided. This is the")
        print("    same class of failure that broke DAgger R0 (_in_gc_range). For")
        print("    goal-adjacent obstacles, DON'T disable the CBF by proximity —")
        print("    gate on 'holding object AND descending', not raw EE-goal distance.")
    else:
        print("  ✓ Gate did not cause a collision on this geometry (obstacle far")
        print("    enough from where the gate engages). Still audit it for Level I.")
    return res


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--suite", default="safelibero_spatial")
    ap.add_argument("--level", default="II", choices=["I", "II"])
    ap.add_argument("--task", type=int, default=0)
    ap.add_argument("--episode", type=int, default=0)
    ap.add_argument("--cmd-mag", type=float, default=0.3,
                    help="adversarial action magnitude (Test C)")
    ap.add_argument("--horizon", type=int, default=120)
    ap.add_argument("--out", default="results_cbf_diag")
    ap.add_argument("--geom", choices=["ellipsoid", "sphere"], default="ellipsoid",
                    help="CBF geometry: ellipsoid = AEGIS-faithful (runner default), "
                         "sphere = legacy sphere decomposition")
    ap.add_argument("--gate-tol", type=float, default=0.08,
                    help="_near_goal tolerance to replicate (Test D)")
    ap.add_argument("--skip", nargs="*", default=[], choices=["a", "b", "c", "d"],
                    help="tests to skip")
    args = ap.parse_args()

    out_dir = Path(args.out) / f"{args.suite}_L{args.level}_t{args.task}_ep{args.episode}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 68)
    print("  CBF DIAGNOSTIC HARNESS  (no VLA server required)")
    print("=" * 68)
    print(f"  suite={args.suite}  level={args.level}  task={args.task}  "
          f"episode={args.episode}")
    print(f"  output → {out_dir}")

    ctx = setup(args.suite, args.level, args.task, args.episode, geom=args.geom)

    if "a" not in args.skip:
        test_a_gain(ctx)
    if "b" not in args.skip:
        test_b_geometry(ctx)
    if "c" not in args.skip:
        test_c_closed_loop(ctx, str(out_dir),
                           cmd_mag=args.cmd_mag, horizon=args.horizon)
    if "d" not in args.skip:
        test_d_gate(ctx, str(out_dir),
                    cmd_mag=args.cmd_mag, horizon=args.horizon,
                    gate_tol=args.gate_tol)

    print(f"\n  Per-step CSVs written under {out_dir}/")
    print("  Done.\n")


if __name__ == "__main__":
    main()
