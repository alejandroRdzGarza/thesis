"""cbf_guidance.py — bridge the CBF to the flow sampler, so safety STEERS generation.

THE PROBLEM THIS ATTACKS, measured not assumed. The shield currently runs as a post-hoc
projection: pi0.5 emits an action, the QP pushes it onto the safe set, the robot executes the
result. That costs capability — the distilled policy scores TSR 82.5% unshielded and 70.8% with the
shield stacked on it. Projection takes a coherent action and moves it off the policy's manifold:
safe, but no longer a competent grasp.

Flow matching gives an intervention point projection does not have. The action is integrated over
~10 denoising steps, so the safe direction can be applied DURING generation, keeping the sample on
the manifold the whole way. `flow_sde.flow_sde_sample_guided` does the integration; this module
supplies the `guidance_fn` it calls.

    latent x0_pred --(unnormalise)--> env action --(CBF QP)--> u_safe
    guidance = (u_safe - u_nom) / action_scale        # back to latent space

Normalisation is affine, so a DELTA maps by dividing by the scale alone — the mean cancels. That is
why this needs no gradient through the model and can reuse the existing, already-validated QP.

WHAT IS GUIDED. Only the FIRST action of the chunk has a well-defined safety value: the barrier is
evaluated against the CURRENT world state, and later actions in the chunk happen after the world has
moved. The correction is broadcast across `n_guide` leading actions with a decaying weight, since
consecutive actions in a chunk point in similar directions — set n_guide=1 for the strictly
defensible version, higher to strengthen the effect at the cost of an approximation.

DIAGNOSTIC, NOT ONLY A KNOB: `GuidanceStats` records how often the QP actually moved the action and
by how much. If guidance never fires, a null result means the wiring is inert, not that the idea
failed — a distinction that is invisible from the success/collision numbers alone.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class GuidanceStats:
    """So a null result can be told apart from inert wiring."""
    calls: int = 0
    fired: int = 0
    norms: list = field(default_factory=list)

    def note(self, delta_norm: float):
        self.calls += 1
        if delta_norm > 1e-9:
            self.fired += 1
            self.norms.append(float(delta_norm))

    def summary(self) -> dict:
        return {
            "calls": self.calls,
            "fired": self.fired,
            "fire_rate": (self.fired / self.calls) if self.calls else 0.0,
            "mean_norm": float(np.mean(self.norms)) if self.norms else 0.0,
            "max_norm": float(np.max(self.norms)) if self.norms else 0.0,
        }


def make_cbf_guidance_fn(cbf_project, action_scale, n_guide: int = 3,
                         decay: float = 0.5, stats: GuidanceStats | None = None):
    """Build a `guidance_fn(x0_pred, sigma)` for flow_sde_sample_guided.

    cbf_project(u_env) -> u_safe_env
        The existing safety projection for a single env-space action, closed over the current
        world state (obstacle spheres, EE pose, ...). Return u_env unchanged when the action is
        already safe; guidance is then exactly zero and the step is untouched.

    action_scale : (action_dim,) or scalar
        The normalisation scale for actions. A delta maps to latent space by dividing by this;
        the mean cancels because the transform is affine. Get it from the policy's norm stats —
        pass the same std/scale the input transform applies to `actions`.

    n_guide : how many leading actions of the chunk receive the correction (1 = only the action
        whose safety is actually defined; more approximates, with `decay` per position).
    """
    scale = np.asarray(action_scale, dtype=np.float64)
    scale = np.where(np.abs(scale) < 1e-8, 1.0, scale)     # never divide by a degenerate scale

    def guidance_fn(x0_pred, sigma):
        x0 = np.asarray(x0_pred, dtype=np.float64)
        flat = x0.reshape(-1, x0.shape[-1]) if x0.ndim > 1 else x0[None, :]
        g = np.zeros_like(flat)

        # Only the first action's safety is defined against the CURRENT state.
        u_nom_env = flat[0] * scale
        u_safe_env = np.asarray(cbf_project(u_nom_env), dtype=np.float64)
        delta_latent = (u_safe_env - u_nom_env) / scale
        n = float(np.linalg.norm(delta_latent))
        if stats is not None:
            stats.note(n)
        if n <= 1e-9:
            return np.zeros_like(x0)                        # already safe: leave the step alone

        w = 1.0
        for i in range(min(n_guide, len(flat))):
            g[i] = w * delta_latent
            w *= decay
        return g.reshape(x0.shape)

    return guidance_fn


def make_sphere_guidance_fn(ee_pos, obstacle_spheres, r_ee: float, action_scale,
                            dt: float = 1.0, margin: float = 0.0, **kw):
    """Convenience wrapper: guidance from the sphere barrier alone, no QP.

    Treats the action's XYZ component as an EE displacement, and pushes it out along the
    obstacle-to-EE direction only when the resulting position would breach (r_ee + r_j + margin).
    Cheaper than a QP per denoising step and needs no solver, so it is the right thing to try
    first — if guidance does not move the safety/capability frontier even in this crude form,
    the more expensive QP version is unlikely to rescue it.
    """
    ee = np.asarray(ee_pos, dtype=np.float64)

    def project(u_env):
        u = np.asarray(u_env, dtype=np.float64).copy()
        nxt = ee + dt * u[:3]
        for c_j, r_j in obstacle_spheres:
            d = nxt - np.asarray(c_j, dtype=np.float64)
            dist = float(np.linalg.norm(d))
            need = r_ee + float(r_j) + margin
            if dist < need and dist > 1e-9:
                nxt = np.asarray(c_j, float) + d / dist * need   # push out to the boundary
                u[:3] = (nxt - ee) / dt
        return u

    return make_cbf_guidance_fn(project, action_scale, **kw)
