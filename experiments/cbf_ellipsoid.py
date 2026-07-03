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

# EE ellipsoid semi-axes (meters) — calibrated from the actual Franka Emika
# Panda gripper collision geometry in LIBERO/MuJoCo (gripper0_hand_collision +
# both finger + finger-pad collision meshes, measured as AABB half-extents in
# the EE local frame, + 5 mm safety margin on each axis):
#   x (gripper width):       measured 0.032 m  → 0.040 m with margin
#   y (finger spread ±):     measured 0.105 m  → 0.115 m with margin
#   z (hand depth, wrist→tip):measured 0.069 m  → 0.075 m with margin
EE_Q_DIAG_DEFAULT = np.array([0.040, 0.115, 0.075])

# Offset from robot0_eef_pos (grasp-site) to the geometric centre of the
# full gripper AABB in the EE local frame.
#
# robot0_eef_pos in LIBERO/robosuite sits at the fingertip grasp site
# (gripper0_eef body).  The full gripper assembly (hand body + two fingers
# + finger-pad geoms) spans z ∈ [-0.124, +0.013] in the EE local frame,
# so its centre is at z = -0.056 m (56 mm toward the wrist).
# Using zero offset (pure fingertip reference) leaves the hand-body portion
# ~14 mm unprotected on the wrist side; centering correctly fixes this.
EE_OFFSET_LOCAL = np.array([0.0, 0.0, -0.056])

# CBF class-K coefficient: α(h) = K_CBF · h.  AEGIS uses 10.
K_CBF = 10.0

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
    return a_v, a_uz, h


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

# Effective EE bounding-sphere radius for sphere–sphere CBF.
# Chosen as the gripper half-width (narrowest cross-section in approach direction)
# plus a small margin; less conservative than EE_Q_DIAG_DEFAULT.max = 0.115 m.
EE_SPHERE_RADIUS = 0.06   # metres


def compute_sphere_decomp_constraints(
    ee_pos:  np.ndarray,
    R1:      np.ndarray,
    spheres: list[tuple[np.ndarray, float]],
    r_ee:    float = EE_SPHERE_RADIUS,
    k_cbf:   float = K_CBF,
    scale:   float = 0.2,
) -> list[tuple[np.ndarray, float]]:
    """
    Build CBF QP rows for a sphere-decomposition obstacle model.

    For each obstacle sphere (c_j, r_j):
        h_j   = ||p_ee - c_j||² − (r_ee + r_j)²
        ḣ_j   = 2(p_ee − c_j) · (scale · R1 @ u_v)
        CBF:    [2(p_ee − c_j) @ (scale · R1)] @ u_v  +  k·h_j  ≥  0

    Returns list of (a_body, b) pairs in the extra_lin_constraints format.
    a_body is the (3,) coefficient vector on the body-frame QP variable u_v;
    b is the scalar offset so the constraint is  a_body @ u_v + b >= 0.
    """
    rows: list[tuple[np.ndarray, float]] = []
    for c_j, r_j in spheres:
        diff   = ee_pos - c_j
        h_j    = float(np.dot(diff, diff) - (r_ee + r_j) ** 2)
        a_body = 2.0 * scale * (diff @ R1)   # (3,) body-frame coefficient
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
        ee_pos, R1, obstacle_spheres, r_ee, k_cbf, scale,
    )
    all_rows = sphere_rows + (extra_constraints or [])

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

    a_v, a_uz, h = compute_h_coeffs(p1, Q1_diag, R1, obs_pos, Q2_diag, R2, z)

    # Nominal reference: convert world-frame u_nom to body-frame, scale × 5
    v_ref_body = R1.T @ u_nom
    u_v_ref    = 5.0 * v_ref_body     # AEGIS: u_v_ref = 5 * v_ref
    u_z_nom    = 10.0 * (a_uz / (np.linalg.norm(a_uz) + 1e-12))

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
