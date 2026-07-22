"""
Ellipsoid-based CBF — ported from AEGIS/VLSA (Funada et al. 2024).

Barrier function h(x) measures separation between two oriented ellipsoids:
  h ≥ 0  ⟺  ellipsoids are disjoint (safe).

Compared to our previous squared-distance CBF, this properly accounts for:
  • EE shape via Q_ef = diag(0.06, 0.12, 0.11) and its current orientation R1
  • Obstacle shape via Q_obs (sphere or MVEE-fitted)
  • Virtual auxiliary state z on the unit sphere (tracks worst-case separation direction)
  • α(h) = k·h with k=10, 5× more aggressive than the previous γ=1.8

Reference: Funada et al. 2024, Eq. 9 (rotating supporting hyperplane between ellipsoids)

API
---
  z = init_z(ee_pos, obs_pos)                     # unit vector EE→obstacle
  h, a_v, a_uz = compute_cbf(ee_pos, R1, obs_pos) # for single obstacle, given z
  u_safe, z = run_ellipsoid_cbf(...)               # full QP + z update
"""

from __future__ import annotations

import numpy as np

try:
    import cvxpy as cp
    _HAS_CVXPY = True
except ImportError:
    _HAS_CVXPY = False

# EE ellipsoid semi-axes (meters) — MATCHED TO AEGIS/VLSA
# (main_aegis_translational.py: Q1_diag = diag(0.06, 0.12, 0.11)). AEGIS deliberately
# uses an INFLATED end-effector ellipsoid — much larger than the true gripper AABB
# (~0.032, 0.102, 0.066) — to create standoff so the barrier engages EARLY and keeps
# grazes out. Our earlier tight [0.040, 0.115, 0.075] fired late (cbf_activation ~0.11)
# and gave CAR ~31% vs AEGIS ~77%. Matching their ellipsoid is the primary CAR fix.
# For tall cartons (orange juice / milk / alphabet soup) AEGIS uses z=0.20; the runner
# passes that per-instruction via EE_Q_DIAG_TALL.
EE_Q_DIAG_DEFAULT = np.array([0.06, 0.12, 0.11])
EE_Q_DIAG_TALL    = np.array([0.06, 0.12, 0.20])
_TALL_OBJECT_KEYS = ("orange juice", "milk", "alphabet soup")

# Offset from robot0_eef_pos to the EE ellipsoid centre — MATCHED TO AEGIS ([0,0,-0.08]).
EE_OFFSET_LOCAL = np.array([0.0, 0.0, -0.08])

# CBF class-K coefficient: α(h) = K_CBF · h.  AEGIS uses 10.
# Larger k is MORE permissive (allows the barrier to approach 0 faster → closer
# approach). We match AEGIS's 10 for a like-for-like comparison; the diagnostic
# (experiments/cbf_diagnostic.py, Test C) confirms the barrier holds at 10.
K_CBF = 10.0

# One-shot warning flag: the scipy SLSQP fallback silently keeps the raw action
# on solver failure. Warn loudly the first time so a missing cvxpy install can't
# hide a CBF bypass.
_SLSQP_FAIL_WARNED = False


def _warn_slsqp_failure(where: str) -> None:
    global _SLSQP_FAIL_WARNED
    if not _SLSQP_FAIL_WARNED:
        print(f"[CBF][WARN] scipy SLSQP failed to converge in {where}; the RAW "
              f"(uncorrected) action is being passed through — the CBF is NOT "
              f"protecting this step. Install cvxpy to use the OSQP QP instead. "
              f"(further failures suppressed)")
        _SLSQP_FAIL_WARNED = True

# Integration step for z-update (should match sim dt ≈ 0.05 s at 20 Hz).
Z_UPDATE_DT = 0.05


def _vector_hat(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric (hat) matrix of a 3-vector."""
    return np.array([
        [0,    -v[2],  v[1]],
        [v[2],  0,    -v[0]],
        [-v[1], v[0],  0   ],
    ])


def _project_matrix(z: np.ndarray) -> np.ndarray:
    """Tangent-space projection: (I - z z^T), z must be unit."""
    z = z / (np.linalg.norm(z) + 1e-12)
    return np.eye(3) - np.outer(z, z)


def init_z(ee_pos: np.ndarray, obs_pos: np.ndarray) -> np.ndarray:
    """Initialise auxiliary direction state z as unit vector EE→obstacle."""
    d = obs_pos - ee_pos
    n = np.linalg.norm(d)
    return d / n if n > 1e-6 else np.array([0.0, 0.0, 1.0])


def ee_ellipsoid_center(ee_pos: np.ndarray, R1: np.ndarray) -> np.ndarray:
    """Compute ellipsoid center = EE flange pos + 8 cm offset along EE z-axis."""
    return ee_pos + R1 @ EE_OFFSET_LOCAL


def compute_h(
    p1: np.ndarray, Q1_diag: np.ndarray, R1: np.ndarray,
    p2: np.ndarray, Q2_diag: np.ndarray, R2: np.ndarray,
    z: np.ndarray,
) -> float:
    """
    Evaluate barrier h between ellipsoid i (EE) and ellipsoid j (obstacle).

    h ≥ 0  ⟺  ellipsoids are separated.

    From Funada et al. 2024, Eq. 9:
      h = (-(|Q̄_j Q̄_i^{-1} z|) + (p_j - p_i)^T Q̄_i^{-1} z  - 1)
          / |Q̄_i^{-1} z|

    Parameters
    ----------
    p1, Q1_diag, R1 : EE ellipsoid (center, semi-axes, orientation)
    p2, Q2_diag, R2 : obstacle ellipsoid
    z               : current unit auxiliary direction
    """
    Qbar1     = R1 @ np.diag(Q1_diag) @ R1.T
    Qbar2     = R2 @ np.diag(Q2_diag) @ R2.T
    Qbar1_inv = np.linalg.inv(Qbar1)

    z = z / (np.linalg.norm(z) + 1e-12)

    term1 = np.linalg.norm(Qbar2 @ Qbar1_inv @ z)
    term2 = (p2 - p1) @ Qbar1_inv @ z
    denom = np.linalg.norm(Qbar1_inv @ z) + 1e-12

    return float((-term1 + term2 - 1.0) / denom)


def compute_h_coeffs(
    p1: np.ndarray, Q1_diag: np.ndarray, R1: np.ndarray,
    p2: np.ndarray, Q2_diag: np.ndarray, R2: np.ndarray,
    z: np.ndarray,
    eps: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Compute CBF constraint coefficients for the 6-D QP.

    Returns (a_v, a_uz, h) where the CBF constraint is:
        a_v @ (0.2 * u_v) + a_uz @ u_z + K_CBF * h  ≥  0

    a_v  (3,) : coefficient on translational control u_v (world frame)
    a_uz (3,) : coefficient on auxiliary-state control u_z
    h    float: current barrier value

    This matches AEGIS utils.py::compute_h_coeffs_3d, simplified to only
    return the translational gradient (a_omega terms dropped because we
    lock orientation in the CBF, matching AEGIS's translational-only policy).
    """
    Q1 = np.diag(Q1_diag); Q2 = np.diag(Q2_diag)
    Qbar1 = R1 @ Q1 @ R1.T
    Qbar2 = R2 @ Q2 @ R2.T
    Qbar1_inv  = np.linalg.inv(Qbar1)
    Qbar1_inv2 = Qbar1_inv @ Qbar1_inv
    Qbar2_sq   = Qbar2 @ Qbar2

    z = z / (np.linalg.norm(z) + eps)
    a_vec  = Qbar1_inv @ z
    denom  = np.linalg.norm(a_vec) + eps
    b_vec  = Qbar2 @ a_vec
    term1  = np.linalg.norm(b_vec) + eps
    sigma  = term1 * denom + eps
    rho    = 1.0 - (p2 - p1) @ a_vec + term1

    # η row: ∂h/∂p1 · direction (translational gradient of h w.r.t. EE position)
    eta_row = -(1.0 / denom) * (z @ Qbar1_inv)

    # μ row: ∂h/∂z (gradient w.r.t. auxiliary direction z)
    mu_row = (
        (rho / (denom**3 + eps)) * (z @ Qbar1_inv2)
        + (1.0 / denom) * ((p2 - p1) @ Qbar1_inv)
        - (1.0 / sigma) * (z @ Qbar1_inv @ Qbar2_sq @ Qbar1_inv)
    )

    # a_v: coefficient on world-frame EE velocity  (η @ R1 gives body→world)
    # AEGIS applies ẋ1 = R1 · u_v (body-frame input → world frame)
    a_v = (eta_row @ R1).ravel()

    # a_uz: coefficient on z-update via tangent-space projection
    a_uz = (mu_row @ _project_matrix(z)).ravel()

    h = compute_h(p1, Q1_diag, R1, p2, Q2_diag, R2, z)
    return a_v, a_uz, h, np.asarray(mu_row).ravel()


# ── Sphere-decomposition CBF (replaces ellipsoid for obstacle geometry) ────────
#
# Rather than fitting one coarse ellipsoid to each obstacle, we represent it as
# a cloud of N small spheres tracing the actual surface (computed by
# cbf_visualizer.decompose_obstacle_to_spheres).  Each sphere gives one linear
# CBF constraint in the same QP, with no auxiliary z state needed:
#
#   h_j = ||p_ee - c_j||² - (r_ee + r_j)²      [sphere-sphere barrier]
#   ḣ_j = 2(p_ee - c_j)·ẋ_ee  ≥  -k·h_j
#   with ẋ_ee = scale · R1 @ u_v  →  constraint: a_body @ u_v + b ≥ 0
#
# Advantages over the Funada et al. ellipsoid barrier:
#   • Captures concave geometry (moka pot handle, bottle neck) that a single
#     ellipsoid must over-conservatively enclose
#   • Simpler QP (3 variables vs 6) and no per-obstacle z state to integrate
#   • Arm-link sphere constraints drop into the same QP unchanged

# Effective EE bounding-sphere radius for sphere–sphere CBF (single-sphere fallback).
EE_SPHERE_RADIUS = 0.06   # metres

# Multi-sphere gripper model in the EE local frame (z = approach / finger direction).
# Three spheres approximate the Franka Panda hand body + two finger pads.
# Coordinates measured from the grasp-site (robot0_eef_pos):
#   x = gripper width  (±4 cm)
#   y = finger spread  (±10.5 cm)
#   z = depth, wrist→tip  (negative = toward wrist)
_EE_SPHERES_LOCAL: list[tuple[np.ndarray, float]] = [
    (np.array([0.0,    0.0,   -0.056]),  0.048),   # hand body (4.8 cm radius covers the ±4 cm wide palm)
    (np.array([0.0,  +0.036,  -0.105]),  0.020),   # left finger pad
    (np.array([0.0,  -0.036,  -0.105]),  0.020),   # right finger pad
]


def get_ee_spheres(
    ee_pos: np.ndarray, R1: np.ndarray
) -> list[tuple[np.ndarray, float]]:
    """Return the 3-sphere EE model in world frame (hand body + 2 finger pads)."""
    return [(ee_pos + R1 @ offset, r) for offset, r in _EE_SPHERES_LOCAL]


def compute_sphere_decomp_constraints(
    ee_pos:     np.ndarray,
    R1:         np.ndarray,
    spheres:    list[tuple[np.ndarray, float]],
    r_ee:       float = EE_SPHERE_RADIUS,
    k_cbf:      float = K_CBF,
    scale:      float = 0.2,
    ee_spheres: list[tuple[np.ndarray, float]] | None = None,
) -> list[tuple[np.ndarray, float]]:
    """
    Build CBF QP rows for a sphere-decomposition obstacle model.

    If ee_spheres is provided, generates N_ee × N_obs constraint rows (one per
    EE-sphere / obstacle-sphere pair) giving proper coverage of the elongated
    gripper shape.  Otherwise falls back to a single sphere at ee_pos with r_ee.

    For each (EE sphere i, obstacle sphere j) pair:
        h_ij   = ||ee_c_i − c_j||² − (r_i + r_j)²
        ḣ_ij   = 2(ee_c_i − c_j) · (scale · R1 @ u_v)      [rigid-body kinematics]
        CBF:    [2(ee_c_i − c_j) @ (scale · R1)] @ u_v + k·h_ij  ≥  0

    Returns list of (a_body, b) in extra_lin_constraints format.
    """
    rows: list[tuple[np.ndarray, float]] = []
    if ee_spheres is not None:
        for ee_c, ee_r in ee_spheres:
            for c_j, r_j in spheres:
                diff   = ee_c - c_j
                h_ij   = float(np.dot(diff, diff) - (ee_r + r_j) ** 2)
                a_body = 2.0 * scale * (diff @ R1)
                rows.append((a_body, k_cbf * h_ij))
    else:
        for c_j, r_j in spheres:
            diff   = ee_pos - c_j
            h_j    = float(np.dot(diff, diff) - (r_ee + r_j) ** 2)
            a_body = 2.0 * scale * (diff @ R1)
            rows.append((a_body, k_cbf * h_j))
    return rows


def run_sphere_decomp_cbf(
    ee_pos:            np.ndarray,
    R1:                np.ndarray,
    obstacle_spheres:  list[tuple[np.ndarray, float]],
    u_nom:             np.ndarray,
    r_ee:              float = EE_SPHERE_RADIUS,
    k_cbf:             float = K_CBF,
    scale:             float = 0.2,
    extra_constraints: list | None = None,
    ee_spheres:        list[tuple[np.ndarray, float]] | None = None,
) -> tuple[np.ndarray, float, bool]:
    """
    CBF-QP using sphere-decomposition obstacle geometry.

    Simpler than run_ellipsoid_cbf: 3-variable QP, no auxiliary z state.

    Decision variable: u_v ∈ R³  (body-frame translational velocity)
    Objective:  min  ||u_v − u_v_ref||²
    Constraints: N sphere rows  +  optional arm-link rows

    Returns
    -------
    u_safe_world : (3,) safe EE position delta, world frame (scaled by 0.2)
    h_min        : float — minimum barrier value across all obstacle spheres
    triggered    : bool  — QP corrected the nominal action
    """
    u_v_ref = R1.T @ u_nom   # world → body frame reference

    sphere_rows = compute_sphere_decomp_constraints(
        ee_pos, R1, obstacle_spheres, r_ee, k_cbf, scale, ee_spheres=ee_spheres,
    )
    all_rows = sphere_rows + (extra_constraints or [])

    if ee_spheres is not None:
        h_values = [
            float(np.dot(ee_c - c_j, ee_c - c_j) - (ee_r + r_j) ** 2)
            for ee_c, ee_r in ee_spheres
            for c_j, r_j in obstacle_spheres
        ]
    else:
        h_values = [
            float(np.dot(ee_pos - c_j, ee_pos - c_j) - (r_ee + r_j) ** 2)
            for c_j, r_j in obstacle_spheres
        ]
    h_min = float(min(h_values)) if h_values else float("inf")

    u_v_out   = u_v_ref.copy()
    triggered = False

    if not all_rows:
        return scale * R1 @ u_v_out, h_min, triggered

    if not _HAS_CVXPY:
        from scipy.optimize import minimize
        def _obj(u): return 0.5 * float(np.dot(u - u_v_ref, u - u_v_ref))
        cons = [{"type": "ineq", "fun": lambda u, a=a, b=b: float(a @ u) + b}
                for a, b in all_rows]
        res = minimize(_obj, x0=u_v_ref.copy(), method="SLSQP", constraints=cons)
        if res.success:
            u_v_out = res.x
        else:
            _warn_slsqp_failure("run_sphere_decomp_cbf")
        triggered = bool(np.linalg.norm(u_v_out - u_v_ref) > 1e-4)
    else:
        import cvxpy as cp
        u           = cp.Variable(3)
        objective   = cp.Minimize(cp.sum_squares(u - u_v_ref))
        constraints = [a @ u + b >= 0 for a, b in all_rows]
        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.OSQP, verbose=False,
                   eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
        if u.value is not None:
            u_v_out   = u.value
            triggered = bool(np.linalg.norm(u_v_out - u_v_ref) > 1e-4)

    return scale * R1 @ u_v_out, h_min, triggered


def run_ellipsoid_cbf(
    ee_pos:   np.ndarray,
    R1:       np.ndarray,
    obs_pos:  np.ndarray,
    obs_q:    np.ndarray,           # obstacle semi-axes (Q2_diag)
    z:        np.ndarray,           # current auxiliary direction (unit)
    u_nom:    np.ndarray,           # nominal VLA action[:3] in world frame
    Q1_diag:  np.ndarray | None = None,
    k_cbf:    float = K_CBF,
    scale:    float = 0.2,          # AEGIS uses ẋ = 0.2·u_v
    dt:       float = Z_UPDATE_DT,
    extra_lin_constraints: list | None = None,
    obs_R:    np.ndarray | None = None,  # obstacle rotation (3,3); None → identity
    # extra_lin_constraints: list of (a_row_3d, b_scalar) where the constraint
    # is  a_row @ u_v_body + b >= 0  (arm-link sphere CBF rows, body frame)
) -> tuple[np.ndarray, np.ndarray, float, bool]:
    """
    CBF-QP with ellipsoidal barrier (AEGIS formulation).

    Decision variable u ∈ R^6 = [u_v (3D translational velocity), u_z (3D aux)]
    Objective : min_u  ||u[:3] - u_v_ref||² / 25 + ||u[3:] - u_z_nom||²
    Constraint: scale · a_v @ u[:3]  +  a_uz @ u[3:]  +  k_cbf · h  ≥  0

    Returns
    -------
    u_safe_world : (3,) safe EE position delta (world frame, already scaled by 0.2)
    z_new        : (3,) updated unit auxiliary direction
    h            : float  current barrier value
    triggered    : bool   CBF was binding
    """
    if Q1_diag is None:
        Q1_diag = EE_Q_DIAG_DEFAULT.copy()

    p1 = ee_ellipsoid_center(ee_pos, R1)
    Q2_diag = obs_q
    R2 = obs_R if obs_R is not None else np.eye(3)

    a_v, a_uz, h, mu_row = compute_h_coeffs(p1, Q1_diag, R1, obs_pos, Q2_diag, R2, z)

    # Nominal reference: convert world-frame u_nom to body-frame.
    #
    # AEGIS blows the QP variable up by 5× (u_v_ref = 5·v_ref) purely for solver
    # conditioning, then cancels it at the output with `0.2 · R1 @ u_v` (5×0.2=1).
    # The blow-up factor MUST equal 1/scale so the passthrough is unit-gain:
    #   output = scale · R1 @ (BETA · R1ᵀ u_nom) = scale·BETA · u_nom = u_nom.
    # Previously this was hardcoded to 5.0 while scale=1.0 was passed, so the
    # unconstrained (non-triggered) action came out 5× too large — a real bug.
    beta       = 1.0 / scale if scale > 1e-9 else 1.0   # = 5.0 when scale=0.2 (AEGIS)
    v_ref_body = R1.T @ u_nom
    u_v_ref    = beta * v_ref_body
    # AEGIS z-nominal: u_z_nom = 10·mu_row (raw ∂h/∂z), NOT the normalized a_uz. This
    # drives the auxiliary direction toward the tightest separation, making the barrier
    # less conservative in the *task* direction while still engaging (matches main_aegis).
    u_z_nom    = 10.0 * mu_row

    triggered = False
    u_v_out   = v_ref_body.copy()

    if not _HAS_CVXPY:
        # Fallback: scipy SLSQP (no auxiliary z update, no arm-link constraints)
        from scipy.optimize import minimize
        a_u_v = scale * a_v
        def obj(u): return 0.5 * np.dot(u - u_v_ref, u - u_v_ref)
        cons = [{"type": "ineq", "fun": lambda u: float(a_u_v @ u) + k_cbf * h}]
        if extra_lin_constraints:
            for a_row, b in extra_lin_constraints:
                cons.append({"type": "ineq",
                              "fun": lambda u, a=a_row, bv=b: float(a @ u) + bv})
        res = minimize(obj, x0=u_v_ref.copy(), method="SLSQP", constraints=cons)
        if not res.success:
            _warn_slsqp_failure("run_ellipsoid_cbf")
        u_v_out  = res.x if res.success else u_v_ref.copy()
        triggered = np.linalg.norm(u_v_out - u_v_ref) > 1e-4
        u_z_out   = u_z_nom.copy()
    else:
        u = cp.Variable(6)
        W = np.diag([1.0/25, 1.0/25, 1.0/25, 1.0, 1.0, 1.0])
        u_ref_vec = np.hstack([u_v_ref, u_z_nom])
        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
        a_u_v = scale * a_v   # effective translation coefficient
        # Primary ellipsoid constraint (EE ↔ obstacle)
        qp_constraints = [a_u_v @ u[:3] + a_uz @ u[3:] + k_cbf * h >= 0]
        # Secondary arm-link sphere constraints (body frame, translation only)
        if extra_lin_constraints:
            for a_row, b in extra_lin_constraints:
                qp_constraints.append(a_row @ u[:3] + b >= 0)
        prob = cp.Problem(objective, qp_constraints)
        prob.solve(solver=cp.OSQP, verbose=False,
                   eps_abs=1e-5, eps_rel=1e-5, max_iter=10000)
        if u.value is not None:
            u_v_out   = u.value[:3]
            u_z_out   = u.value[3:]
        else:
            u_v_out   = u_v_ref.copy()
            u_z_out   = u_z_nom.copy()
        triggered = np.linalg.norm(u_v_out - u_v_ref) > 1e-4

    # Update auxiliary direction z (sphere tangent dynamics)
    dz    = _project_matrix(z) @ u_z_out
    z_new = z + dz * dt
    n     = np.linalg.norm(z_new)
    z_new = z_new / n if n > 1e-6 else z.copy()

    # Convert body-frame u_v back to world-frame delta (scaled by 0.2)
    u_safe_world = scale * R1 @ u_v_out

    return u_safe_world, z_new, h, triggered
