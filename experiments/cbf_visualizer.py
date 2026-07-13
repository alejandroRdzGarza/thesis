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
    "moka_pot_obstacle":          np.array([0.115, 0.085, 0.130]),  # conservative bound: spout extends ~5cm laterally
    "white_storage_box_obstacle": np.array([0.070, 0.070, 0.066]),  # measured from 4 wall geoms (outer faces ±0.070 xy, ±0.066 z)
    "wine_bottle_obstacle":       np.array([0.038, 0.038, 0.170]),  # tall thin cylinder
    "milk_obstacle":              np.array([0.048, 0.038, 0.110]),  # carton
    "red_coffee_mug_obstacle":    np.array([0.048, 0.048, 0.060]),  # mug
    "yellow_book_obstacle":       np.array([0.095, 0.135, 0.018]),  # flat book
}


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

    Samples surface points from all primitive (non-mesh) collision geoms
    on the body, then thins them to n_spheres via farthest-point sampling
    so the cloud gives uniform coverage of the object surface.

    If no primitive geoms exist (mesh-only bodies), falls back to sampling
    the surface of the known analytic ellipsoid from _KNOWN_OBSTACLE_Q.

    Returns list of (center_world, radius) or None if body not found.
    Every sphere has radius = r_sphere + safety_margin.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None

    body_pos = data.xpos[body_id].copy()

    _MESH      = int(mujoco.mjtGeom.mjGEOM_MESH)
    _BOX       = int(mujoco.mjtGeom.mjGEOM_BOX)
    _SPHERE    = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    _CYLINDER  = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    _CAPSULE   = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    _ELLIPSOID = int(mujoco.mjtGeom.mjGEOM_ELLIPSOID)

    surface_pts: list[np.ndarray] = []

    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id:
            continue
        gtype = int(model.geom_type[gid])
        gpos = data.geom_xpos[gid].copy()
        gmat = data.geom_xmat[gid].reshape(3, 3).copy()
        gs   = model.geom_size[gid].copy()

        def _w(lp, _p=gpos, _m=gmat):
            return _p + _m @ np.asarray(lp, float)

        if gtype == _MESH:
            continue  # mesh geom_xpos is at object base; geom_rbound from base spreads spheres on the floor

        if gtype == _BOX:
            hx, hy, hz = gs[0], gs[1], gs[2]
            for sx in (-1, 1):
                for sy in (-1, 1):
                    for sz in (-1, 1):
                        surface_pts.append(_w([sx*hx, sy*hy, sz*hz]))
            for ax, sign in ((0,1),(0,-1),(1,1),(1,-1),(2,1),(2,-1)):
                p = [0.0, 0.0, 0.0]; p[ax] = sign * gs[ax]
                surface_pts.append(_w(p))

        elif gtype == _SPHERE:
            r = gs[0]
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for phi in np.linspace(0.3, np.pi - 0.3, 4):
                    surface_pts.append(_w([
                        r*np.sin(phi)*np.cos(theta),
                        r*np.sin(phi)*np.sin(theta),
                        r*np.cos(phi),
                    ]))
            surface_pts.extend([_w([0, 0, r]), _w([0, 0, -r])])

        elif gtype == _CYLINDER:
            r, hh = gs[0], gs[1]
            for zz in (-hh, 0.0, hh):
                n = 8 if zz == 0.0 else 6
                for theta in np.linspace(0, 2*np.pi, n, endpoint=False):
                    surface_pts.append(_w([r*np.cos(theta), r*np.sin(theta), zz]))
            surface_pts.extend([_w([0, 0, hh]), _w([0, 0, -hh])])

        elif gtype == _CAPSULE:
            r, hh = gs[0], gs[1]
            for zz in np.linspace(-hh, hh, 3):
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    surface_pts.append(_w([r*np.cos(theta), r*np.sin(theta), zz]))
            for sign in (-1, 1):
                z0 = sign * hh
                for theta in np.linspace(0, 2*np.pi, 6, endpoint=False):
                    for phi in (np.pi/4, np.pi/2):
                        surface_pts.append(_w([
                            r*np.sin(phi)*np.cos(theta),
                            r*np.sin(phi)*np.sin(theta),
                            z0 + sign*r*np.cos(phi),
                        ]))

        elif gtype == _ELLIPSOID:
            rx, ry, rz = gs[0], gs[1], gs[2]
            for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                for phi in np.linspace(0.3, np.pi - 0.3, 4):
                    surface_pts.append(_w([
                        rx*np.sin(phi)*np.cos(theta),
                        ry*np.sin(phi)*np.sin(theta),
                        rz*np.cos(phi),
                    ]))

    # Fallback: if only mesh geoms, sample on the known analytic ellipsoid
    if not surface_pts:
        for key, q in _KNOWN_OBSTACLE_Q.items():
            if key in body_name:
                for theta in np.linspace(0, 2*np.pi, 8, endpoint=False):
                    for phi in np.linspace(0.3, np.pi - 0.3, 4):
                        lp = np.array([
                            q[0]*np.sin(phi)*np.cos(theta),
                            q[1]*np.sin(phi)*np.sin(theta),
                            q[2]*np.cos(phi),
                        ])
                        surface_pts.append(body_pos + lp)
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
    """
    Compute the CBF ellipsoid for an obstacle body.

    Always fits the geometric center from non-mesh geom centers so the
    returned center matches the actual collision geometry (not the body
    origin, which can differ from the geom center for compound objects).

    q_diag is taken from _KNOWN_OBSTACLE_Q when available (hand-measured,
    more accurate than auto-fit); otherwise auto-fitted from geom AABB.

    Returns (center_world, q_diag) or None if body not found.
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        return None
    body_pos = data.xpos[body_id].copy()

    # ── Fit geom center from primitive (non-mesh) geom positions ──────────
    _MESH = int(mujoco.mjtGeom.mjGEOM_MESH)

    centers: list[np.ndarray] = []
    half_sizes: list[float]   = []

    for gid in range(model.ngeom):
        if model.geom_bodyid[gid] != body_id:
            continue
        if model.geom_type[gid] == _MESH:
            continue  # mesh size field is not a bounding box
        gpos  = data.geom_xpos[gid].copy()
        gsize = model.geom_size[gid].copy()
        centers.append(gpos - body_pos)
        half_sizes.append(float(np.max(gsize[:3])) if np.any(gsize > 0) else 0.02)

    if centers:
        centers_arr   = np.array(centers)
        center_offset = (centers_arr.min(axis=0) + centers_arr.max(axis=0)) / 2.0
        geom_center   = body_pos + center_offset
    else:
        geom_center = body_pos.copy()

    # ── q_diag: use pre-measured table if available ────────────────────────
    for key, q in _KNOWN_OBSTACLE_Q.items():
        if key in body_name:
            return geom_center, q + safety_margin

    # ── Auto-fit q_diag from geom AABB ────────────────────────────────────
    if not centers:
        return None

    centers_arr = np.array(centers)
    mins = centers_arr.min(axis=0)
    maxs = centers_arr.max(axis=0)
    median_hs = float(np.median(half_sizes))

    half_extents = (maxs - mins) / 2.0 + median_hs
    q_diag = np.maximum(half_extents + safety_margin, 0.02)

    return geom_center, q_diag


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

    # ── EE sphere ─────────────────────────────────────────────────────────
    ee_rgba = _RGBA_EE_ACTIVE if cbf_triggered else _RGBA_EE_SAFE
    ee_r = 0.06   # matches EE_SPHERE_RADIUS used in the CBF barrier
    specs.append((_SPHERE, np.array([ee_r, ee_r, ee_r]), ee_center.copy(), np.eye(3), ee_rgba.copy()))

    # ── Obstacle geometry ─────────────────────────────────────────────────
    if obstacle_spheres is not None:
        # Sphere decomposition: draw each surface sphere, coloured by proximity
        for c_j, r_j in obstacle_spheres:
            dist = float(np.linalg.norm(ee_center - c_j))
            clearance = dist - (ee_r + r_j)   # positive = safe
            if clearance < 0:
                rgba = _RGBA_OBS_HOT.copy()
            elif clearance < 0.08:
                rgba = _RGBA_OBS_CLOSE.copy()
            else:
                rgba = _RGBA_OBS_SAFE.copy()
            specs.append((_SPHERE, np.array([r_j, r_j, r_j]),
                          c_j.copy(), np.eye(3), rgba))
    else:
        # Fallback: single ellipsoid per obstacle (original behaviour)
        for i, ob in enumerate(obstacles):
            h = h_values[i] if i < len(h_values) else float("inf")
            if h < 0:
                obs_rgba = _RGBA_OBS_HOT.copy()
            elif h < 0.5:
                obs_rgba = _RGBA_OBS_CLOSE.copy()
            else:
                obs_rgba = _RGBA_OBS_SAFE.copy()

            if hasattr(ob, "q_diag") and ob.q_diag is not None:
                obs_size = ob.q_diag.copy()
            else:
                r = getattr(ob, "safety_radius", 0.10)
                obs_size = np.array([r, r, r])

            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ob.name)
            obs_mat = data.xmat[body_id].reshape(3, 3).copy() if body_id >= 0 else np.eye(3)
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
