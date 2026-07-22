"""
3D CBF region visualization — injected into MuJoCo's offscreen render pipeline.

Hooks into robosuite's MjSim render context to add translucent safety-zone
geoms (EE ellipsoid, obstacle ellipsoid, arm-link spheres) directly into the
3D scene BEFORE mjr_render.  They appear in the agentview camera image and
any saved video without any 2D pixel projection — MuJoCo renders them in
correct 3D perspective with depth.

Public API
----------
  fit_obstacle_ellipsoid(model, data, body_name, safety_margin=0.02)
      → (center_world, q_diag) or None
      Compute the AABB of all collision geoms in a body.

  install_scene_hook(sim)
      Call once after the env is created + first step has run (render context
      is initialised lazily on first render call).  Patches the render context
      to accept custom geoms.

  push_cbf_geoms(sim, ee_center, ee_q, R1, obstacles, h_values,
                 arm_body_ids, arm_radii, model, data, cbf_triggered)
      Call every step to update what geoms the hook will inject.
"""

from __future__ import annotations
import types
import numpy as np
import mujoco

_ELLIPSOID = mujoco.mjtGeom.mjGEOM_ELLIPSOID
_SPHERE    = mujoco.mjtGeom.mjGEOM_SPHERE

# RGBA palettes (BGR in OpenCV but RGBA here for MuJoCo)
_RGBA_EE_SAFE     = np.array([0.10, 0.85, 0.20, 0.12], np.float32)   # green
_RGBA_EE_ACTIVE   = np.array([0.95, 0.85, 0.05, 0.18], np.float32)   # yellow
_RGBA_OBS_SAFE    = np.array([0.30, 0.55, 1.00, 0.10], np.float32)   # blue
_RGBA_OBS_CLOSE   = np.array([1.00, 0.50, 0.10, 0.15], np.float32)   # orange
_RGBA_OBS_HOT     = np.array([1.00, 0.12, 0.12, 0.22], np.float32)   # red
_RGBA_LINK        = np.array([0.20, 0.90, 0.90, 0.08], np.float32)   # cyan


# Per-obstacle-class ellipsoid semi-axes (x, y, z) measured from MuJoCo geoms.
# These are the PHYSICAL half-extents of each object's main body (excluding
# thin appendages like handles that don't meaningfully obstruct the arm path).
# Safety margin is added on top in detect_safelibero_obstacle().
_KNOWN_OBSTACLE_Q: dict[str, np.ndarray] = {
    "moka_pot_obstacle":          np.array([0.115, 0.085, 0.155]),  # spout tip extends ~15.5 cm above body center
    "white_storage_box_obstacle": np.array([0.070, 0.070, 0.066]),  # measured from 4 wall geoms (outer faces ±0.070 xy, ±0.066 z)
    "wine_bottle_obstacle":       np.array([0.038, 0.038, 0.170]),  # tall thin cylinder
    "milk_obstacle":              np.array([0.048, 0.038, 0.110]),  # carton
    "red_coffee_mug_obstacle":    np.array([0.048, 0.048, 0.060]),  # mug
    "yellow_book_obstacle":       np.array([0.095, 0.135, 0.018]),  # flat book
}


# ── Shared obstacle point cloud (single geometry source) ─────────────────────
#
# Both the ellipsoid (MVEE) and the sphere-decomposition CBF now derive from ONE
# real point cloud read directly from MuJoCo — mesh vertices AND primitive-geom
# surface samples — instead of the hand-measured _KNOWN_OBSTACLE_Q table. This
# mirrors AEGIS's method (they MVEE-fit a point cloud from VLM+depth); the only
# difference is the cloud's source (sim ground-truth vs perception).

def get_obstacle_point_cloud(
    model, data, body_name: str, max_points: int = 2000,
) -> np.ndarray | None:
    """World-frame point cloud of an obstacle body's true collision geometry.

    Reads actual mesh vertices (via geom_dataid → mesh_vert) and samples the
    surfaces of all primitive geoms (box/sphere/cylinder/capsule/ellipsoid).
    Returns an (N,3) array, or None if the body has no usable geometry.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None

    _MESH     = int(mujoco.mjtGeom.mjGEOM_MESH)
    _BOX      = int(mujoco.mjtGeom.mjGEOM_BOX)
    _SPHERE   = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    _CYLINDER = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    _CAPSULE  = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    _ELLIP    = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)

    pts: list[np.ndarray] = []
    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id:
            continue
        gtype = int(model.geom_type[gid])
        gpos  = data.geom_xpos[gid].copy()
        gmat  = data.geom_xmat[gid].reshape(3, 3).copy()
        gs    = model.geom_size[gid].copy()

        def _w(lp, _p=gpos, _m=gmat):
            return _p + _m @ np.asarray(lp, float)

        if gtype == _MESH:
            mid = int(model.geom_dataid[gid])
            if mid < 0:
                continue
            va = int(model.mesh_vertadr[mid]); vn = int(model.mesh_vertnum[mid])
            V = np.asarray(model.mesh_vert[va:va + vn]).reshape(-1, 3)
            pts.append((gmat @ V.T).T + gpos)

        elif gtype == _BOX:
            hx, hy, hz = gs[0], gs[1], gs[2]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        pts.append(_w([sx*hx, sy*hy, sz*hz])[None])
            for ax, sign in ((0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)):
                p = [0.0, 0.0, 0.0]; p[ax] = sign * gs[ax]
                pts.append(_w(p)[None])

        elif gtype in (_SPHERE, _ELLIP):
            rx, ry, rz = (gs[0], gs[0], gs[0]) if gtype == _SPHERE else (gs[0], gs[1], gs[2])
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for phi in np.linspace(0.3, np.pi - 0.3, 4):
                    pts.append(_w([rx*np.sin(phi)*np.cos(theta),
                                   ry*np.sin(phi)*np.sin(theta),
                                   rz*np.cos(phi)])[None])
            pts.append(_w([0, 0, rz])[None]); pts.append(_w([0, 0, -rz])[None])

        elif gtype in (_CYLINDER, _CAPSULE):
            r, hh = gs[0], gs[1]
            for zz in np.linspace(-hh, hh, 3):
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    pts.append(_w([r*np.cos(theta), r*np.sin(theta), zz])[None])
            pts.append(_w([0, 0, hh + (r if gtype == _CAPSULE else 0)])[None])
            pts.append(_w([0, 0, -hh - (r if gtype == _CAPSULE else 0)])[None])

    if not pts:
        return None
    P = np.vstack(pts)
    if len(P) > max_points:
        idx = np.random.default_rng(0).choice(len(P), max_points, replace=False)
        P = P[idx]
    return P


def _mvee_khachiyan(P: np.ndarray, tol: float = 1e-3, max_iter: int = 2000):
    """Minimum-volume enclosing ellipsoid (Khachiyan). No external deps.

    Returns (center (3,), R (3,3) principal axes as columns, semi_axes (3,)),
    or None if the cloud is degenerate (too few / coplanar points).
    """
    P = np.asarray(P, float)
    N, d = P.shape
    if N < d + 1:
        return None
    Q = np.vstack([P.T, np.ones(N)])           # (d+1, N)
    u = np.full(N, 1.0 / N)
    for _ in range(max_iter):
        X = Q @ (u[:, None] * Q.T)             # (d+1, d+1)
        try:
            G = Q.T @ np.linalg.inv(X)         # (N, d+1)
        except np.linalg.LinAlgError:
            return None
        M = np.einsum("ij,ij->i", G, Q.T)      # leverage scores (N,)
        j = int(np.argmax(M)); step = (M[j] - d - 1) / ((d + 1) * (M[j] - 1))
        if not np.isfinite(step) or step <= 0:
            break
        new_u = (1 - step) * u; new_u[j] += step
        if np.linalg.norm(new_u - u) < tol:
            u = new_u; break
        u = new_u
    c = P.T @ u
    try:
        A = np.linalg.inv(P.T @ (u[:, None] * P) - np.outer(c, c)) / d
    except np.linalg.LinAlgError:
        return None
    vals, vecs = np.linalg.eigh(A)
    vals = np.clip(vals, 1e-9, None)
    semi_axes = 1.0 / np.sqrt(vals)            # along principal axes (columns of vecs)
    return c, vecs, semi_axes


def fit_obstacle_mvee(
    model, data, body_name: str, safety_margin: float = 0.010,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Oriented MVEE ellipsoid for an obstacle: (center, R, q_diag).

    q_diag are the semi-axes (metres) along the ellipsoid principal axes (the
    columns of R). Falls back to the _KNOWN_OBSTACLE_Q table (axis-aligned) only
    if the point cloud is unavailable.
    """
    P = get_obstacle_point_cloud(model, data, body_name)
    if P is not None:
        res = _mvee_khachiyan(P)
        if res is not None:
            c, R, semi = res
            return c, R, semi + safety_margin

    # Fallback: hand-measured table, axis-aligned.
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    center = data.xpos[body_id].copy()
    for key, q in _KNOWN_OBSTACLE_Q.items():
        if key in body_name:
            return center, np.eye(3), q + safety_margin
    return None


# ── Obstacle geometry from MuJoCo ────────────────────────────────────────────

def decompose_obstacle_to_spheres(
    model,
    data,
    body_name:     str,
    n_spheres:     int   = 16,
    r_sphere:      float = 0.010,
    safety_margin: float = 0.010,
) -> list[tuple[np.ndarray, float]] | None:
    """
    Decompose an obstacle body into a cloud of spheres tracing its surface.

    Uses the shared get_obstacle_point_cloud() (mesh vertices + primitive-geom
    surface samples — the SAME source the MVEE ellipsoid uses), then thins it to
    n_spheres via farthest-point sampling for uniform surface coverage.

    Returns list of (center_world, radius) or None if body not found.
    Every sphere has radius = r_sphere + safety_margin.
    """
    pts = get_obstacle_point_cloud(model, data, body_name)

    # Fallback: mesh-less / empty body → sample the known analytic ellipsoid.
    if pts is None:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return None
        body_pos = data.xpos[body_id].copy()
        surface_pts: list[np.ndarray] = []
        for key, q in _KNOWN_OBSTACLE_Q.items():
            if key in body_name:
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    for phi in np.linspace(0.3, np.pi - 0.3, 4):
                        surface_pts.append(body_pos + np.array([
                            q[0]*np.sin(phi)*np.cos(theta),
                            q[1]*np.sin(phi)*np.sin(theta),
                            q[2]*np.cos(phi),
                        ]))
                break
        if not surface_pts:
            return None
        pts = np.array(surface_pts)

    # Farthest-point sampling: greedy selection of n_spheres maximally spread points
    if len(pts) > n_spheres:
        chosen    = [0]
        min_dists = np.full(len(pts), np.inf)
        for _ in range(n_spheres - 1):
            min_dists = np.minimum(
                min_dists,
                np.linalg.norm(pts - pts[chosen[-1]], axis=1),
            )
            chosen.append(int(np.argmax(min_dists)))
        pts = pts[chosen]

    r_total = r_sphere + safety_margin
    return [(c.copy(), r_total) for c in pts]


def fit_obstacle_ellipsoid(
    model,
    data,
    body_name: str,
    safety_margin: float = 0.025,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Axis-aligned (center, q_diag) for visualization/back-compat.

    Thin wrapper over fit_obstacle_mvee(): returns the MVEE centre and its
    principal semi-axes as an axis-aligned size. The CBF itself uses the fully
    oriented MVEE (center, R, q_diag) via detect_safelibero_obstacle; this 2-tuple
    exists only for the render hook, which draws a rough translucent blob.
    """
    res = fit_obstacle_mvee(model, data, body_name, safety_margin=safety_margin)
    if res is None:
        return None
    center, _R, q_diag = res
    return center, q_diag


# ── Render-context hook ───────────────────────────────────────────────────────

def _make_patched_render(original_render):
    """Return a patched render method that injects _cbf_geom_specs into scn."""

    def patched_render(self, width, height, camera_id=None, segmentation=False):
        import mujoco as _muj

        # Run the standard pipeline up to updateScene
        viewport = _muj.MjrRect(0, 0, width, height)
        if width > self.con.offWidth or height > self.con.offHeight:
            new_w = max(width,  self.model.vis.global_.offwidth)
            new_h = max(height, self.model.vis.global_.offheight)
            self.update_offscreen_size(new_w, new_h)
        if camera_id is not None:
            if camera_id == -1:
                self.cam.type = _muj.mjtCamera.mjCAMERA_FREE
            else:
                self.cam.type = _muj.mjtCamera.mjCAMERA_FIXED
            self.cam.fixedcamid = camera_id

        _muj.mjv_updateScene(
            self.model._model, self.data._data,
            self.vopt, self.pert, self.cam,
            _muj.mjtCatBit.mjCAT_ALL, self.scn,
        )

        if segmentation:
            self.scn.flags[_muj.mjtRndFlag.mjRND_SEGMENT] = 1
            self.scn.flags[_muj.mjtRndFlag.mjRND_IDCOLOR] = 1

        # ── Inject custom CBF geoms ───────────────────────────────────────
        specs = getattr(self, "_cbf_geom_specs", [])
        for gtype, size, pos, mat, rgba in specs:
            if self.scn.ngeom >= self.scn.maxgeom:
                break
            g = self.scn.geoms[self.scn.ngeom]
            # mjv_initGeom sets ALL fields (incl. texid, emission, specular …)
            # to safe defaults before we override our custom fields.  Without
            # this, slots reused from a prior mjv_updateScene frame retain
            # stale texid pointers that renderGeom dereferences → SIGSEGV.
            _muj.mjv_initGeom(
                g,
                gtype,
                np.asarray(size, dtype=np.float64),
                np.asarray(pos,  dtype=np.float64),
                np.asarray(mat,  dtype=np.float64).ravel(),
                np.asarray(rgba, dtype=np.float32),
            )
            g.dataid   = -1
            g.objtype  = _muj.mjtObj.mjOBJ_UNKNOWN
            g.objid    = -1
            g.category = _muj.mjtCatBit.mjCAT_DECOR
            g.label    = ""
            self.scn.ngeom += 1

        _muj.mjr_render(viewport=viewport, scn=self.scn, con=self.con)

        if segmentation:
            self.scn.flags[_muj.mjtRndFlag.mjRND_SEGMENT] = 0
            self.scn.flags[_muj.mjtRndFlag.mjRND_IDCOLOR] = 0

    return patched_render


def install_scene_hook(sim) -> bool:
    """
    Patch sim._render_context_offscreen to inject CBF geoms each frame.

    Must be called AFTER the first env.step() so the render context is
    initialised.  Safe to call multiple times (idempotent).

    Returns True on success, False if context not yet initialised.
    """
    rc = sim._render_context_offscreen
    if rc is None:
        return False
    if getattr(rc, "_cbf_hook_installed", False):
        return True
    rc._cbf_geom_specs = []
    rc.render = types.MethodType(_make_patched_render(rc.render), rc)
    rc._cbf_hook_installed = True
    return True


def push_cbf_geoms(
    sim,
    ee_center:           np.ndarray,        # world-frame EE ellipsoid centre
    ee_q:                np.ndarray,        # EE semi-axes (3,)
    R1:                  np.ndarray,        # EE rotation matrix (3,3)
    obstacles:           list,              # list of ObstacleConfig
    h_values:            list[float],
    arm_body_ids:        list[int],
    arm_radii:           dict[str, float],  # body_name → sphere radius
    model,
    data,
    cbf_triggered:       bool,
    obstacle_spheres:    list | None = None,  # (center, radius) pairs from decompose_obstacle_to_spheres
    arm_sample_positions: dict | None = None, # {body_id: [pts]} from _link_sample_positions
) -> None:
    """
    Update the geom injection list for the next render frame.

    Geoms drawn:
      • EE sphere             (green/yellow based on CBF active)
      • Obstacle surface spheres (blue→orange→red per sphere based on EE proximity)
        OR obstacle ellipsoid  (fallback when sphere decomp unavailable)
      • Arm-link spheres      (cyan, at sample positions along each link axis)
    """
    rc = sim._render_context_offscreen
    if rc is None or not getattr(rc, "_cbf_hook_installed", False):
        return

    specs: list = []

    # ── EE ellipsoid (oriented, matches CBF geometry) ─────────────────────
    # EE_Q_DIAG_DEFAULT already holds semi-axes directly — do NOT take sqrt.
    ee_rgba = _RGBA_EE_ACTIVE if cbf_triggered else _RGBA_EE_SAFE
    ee_size = np.asarray(ee_q, float)   # direct semi-axes [0.040, 0.115, 0.075] m
    specs.append((_ELLIPSOID, ee_size, ee_center.copy(), R1.copy(), ee_rgba.copy()))

    # ── Obstacle geometry ─────────────────────────────────────────────────
    if obstacle_spheres is not None:
        # Sphere decomposition: draw each surface sphere, coloured by proximity
        ee_r = float(np.mean(ee_size))
        for c_j, r_j in obstacle_spheres:
            dist = float(np.linalg.norm(ee_center - c_j))
            clearance = dist - (ee_r + r_j)
            if clearance < 0:
                rgba = _RGBA_OBS_HOT.copy()
            elif clearance < 0.08:
                rgba = _RGBA_OBS_CLOSE.copy()
            else:
                rgba = _RGBA_OBS_SAFE.copy()
            specs.append((_SPHERE, np.array([r_j, r_j, r_j]),
                          c_j.copy(), np.eye(3), rgba))
    else:
        # Single ellipsoid per obstacle using known dimensions or MuJoCo AABB
        for i, ob in enumerate(obstacles):
            h = h_values[i] if i < len(h_values) else float("inf")
            if h < 0:
                obs_rgba = _RGBA_OBS_HOT.copy()
            elif h < 0.5:
                obs_rgba = _RGBA_OBS_CLOSE.copy()
            else:
                obs_rgba = _RGBA_OBS_SAFE.copy()

            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob.name)
            obs_mat = data.xmat[body_id].reshape(3, 3).copy() if body_id >= 0 else np.eye(3)

            # Priority: known table → AABB from MuJoCo → safety_radius sphere
            if ob.name in _KNOWN_OBSTACLE_Q:
                obs_size = _KNOWN_OBSTACLE_Q[ob.name].copy()
            else:
                aabb = fit_obstacle_ellipsoid(model, data, ob.name)
                if aabb is not None:
                    _, obs_size = aabb
                else:
                    r = getattr(ob, "safety_radius", 0.10)
                    obs_size = np.array([r, r, r])

            specs.append((_ELLIPSOID, obs_size, ob.pos.copy(), obs_mat, obs_rgba))

    # ── Arm-link spheres ──────────────────────────────────────────────────
    if arm_sample_positions is not None:
        # Multi-sample: draw one sphere per sample point along each link axis
        for bid, pts in arm_sample_positions.items():
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            r = arm_radii.get(bname)
            if r is None:
                continue
            for p in pts:
                specs.append((_SPHERE, np.array([r, r, r]),
                              np.asarray(p, float), np.eye(3), _RGBA_LINK.copy()))
    else:
        # Fallback: single sphere at body origin (original behaviour)
        for bid in arm_body_ids:
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            r = arm_radii.get(bname)
            if r is None:
                continue
            p = data.xpos[bid].copy()
            specs.append((_SPHERE, np.array([r, r, r]), p, np.eye(3), _RGBA_LINK.copy()))

    rc._cbf_geom_specs = specs
