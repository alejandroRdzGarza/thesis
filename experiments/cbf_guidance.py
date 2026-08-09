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
    # Closest the PREDICTED next EE position came to the obstacle surface. Without this, a 0%
    # fire rate cannot be told apart from "the threshold is too tight" — both look identical in
    # the headline numbers, and only one of them is a reason to change a flag.
    min_gap: float = float("inf")

    def note(self, delta_norm: float, gap: float | None = None):
        self.calls += 1
        if gap is not None:
            self.min_gap = min(self.min_gap, float(gap))
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
            "min_gap": (float(self.min_gap) if self.min_gap < 1e8 else None),
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

    nd = len(scale)

    def guidance_fn(x0_pred, sigma):
        x0 = np.asarray(x0_pred, dtype=np.float64)
        flat = x0.reshape(-1, x0.shape[-1]) if x0.ndim > 1 else x0[None, :]
        g = np.zeros_like(flat)

        # The LATENT action is zero-padded to the model's action_dim (32 for pi0.5) while the env
        # action — and therefore the barrier and the scale — are 7-D. Slice to the real dims;
        # broadcasting (32,) against (7,) raised inside the policy, and the runner caught it and
        # silently held the last action, so guidance appeared to run while doing nothing at all.
        # Only the first action's safety is defined against the CURRENT state.
        u_nom_env = flat[0, :nd] * scale
        u_safe_env = np.asarray(cbf_project(u_nom_env), dtype=np.float64)[:nd]
        delta_latent = np.zeros(flat.shape[1], dtype=np.float64)
        delta_latent[:nd] = (u_safe_env - u_nom_env) / scale
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
        if kw.get("stats") is not None:
            g = min(float(np.linalg.norm(nxt - np.asarray(c, float))) - (r_ee + float(r) + margin)
                    for c, r in obstacle_spheres)
            kw["stats"].min_gap = min(kw["stats"].min_gap, g)
        for c_j, r_j in obstacle_spheres:
            d = nxt - np.asarray(c_j, dtype=np.float64)
            dist = float(np.linalg.norm(d))
            need = r_ee + float(r_j) + margin
            if dist < need and dist > 1e-9:
                nxt = np.asarray(c_j, float) + d / dist * need   # push out to the boundary
                u[:3] = (nxt - ee) / dt
        return u

    return make_cbf_guidance_fn(project, action_scale, **kw)


def extract_action_scale(policy, action_dim: int = 7, verbose: bool = True):
    """The scale a guidance delta must be divided by to reach latent space — MEASURED, not guessed.

    Reading a named field is fragile: the stats live on an inner Normalize transform inside a
    CompositeTransform (not on the composite), and openpi's Normalize has both mean/std and
    quantile paths, so which field is authoritative depends on config. Guessing wrong does not
    crash — it silently mis-scales every correction, which presents as "guidance is too weak"
    and sends you tuning lambda against a bug.

    So this probes the real transform instead. Normalisation is affine, so pushing two known
    action vectors through it and taking the difference recovers the scale exactly:

        latent = (env - mean) / scale   =>   d(latent)/d(env) = 1 / scale

    The named fields are read too and compared. On pi05_libero the measured slope matches
    (q99-q01)/2 exactly — the config normalises by quantiles onto [-1, 1] — while `std` is 2.5x
    smaller. Reading `std` would therefore have applied every correction at 40% of its intended
    size: no crash, just guidance that looks too weak to work.
    """
    import numpy as np

    ah = getattr(getattr(policy, "_model", None), "action_horizon", 10)
    z = np.zeros((224, 224, 3), dtype=np.uint8)

    def _norm(a):
        d = {"observation/image": z, "observation/wrist_image": z,
             "observation/state": np.zeros(8), "prompt": "probe",
             "actions": np.asarray(a, dtype=np.float32)}
        return np.asarray(policy._input_transform(d)["actions"], dtype=np.float64)

    n0 = _norm(np.zeros((ah, action_dim)))
    n1 = _norm(np.ones((ah, action_dim)))
    slope = (n1 - n0)[0, :action_dim]                 # d(latent)/d(env) for a unit env delta
    slope = np.where(np.abs(slope) < 1e-12, 1.0, slope)
    scale = 1.0 / slope

    if verbose:
        print(f"  action_scale (measured through the transform): {np.round(scale, 5)}")
        # Cross-check against the declared fields; disagreement usually means quantile norm.
        ns = None
        for attr in ("_input_transform", "_output_transform"):
            t = getattr(policy, attr, None)
            for f in (getattr(t, "transforms", None) or []):
                if getattr(f, "norm_stats", None) and "actions" in f.norm_stats:
                    ns = f.norm_stats["actions"]
                    break
            if ns is not None:
                break
        if ns is not None:
            for name in ("std", "q99"):
                v = getattr(ns, name, None)
                if v is None:
                    continue
                v = np.asarray(v, float).ravel()[:action_dim]
                if name == "q99":
                    # Quantile normalisation maps [q01, q99] -> [-1, 1], i.e.
                    # x_norm = 2(x - q01)/(q99 - q01) - 1, so the scale is HALF the range.
                    # Confirmed on pi05_libero: the measured slope equals (q99-q01)/2 exactly.
                    q01 = np.asarray(getattr(ns, "q01"), float).ravel()[:action_dim]
                    v = (v - q01) / 2.0
                    name = "(q99-q01)/2"
                agree = np.allclose(v, scale, rtol=0.02)
                print(f"    vs norm_stats {name:<8}: {np.round(v, 5)}  "
                      f"{'MATCHES' if agree else 'differs'}")
        else:
            print("    (no norm_stats found to cross-check against)")
    return scale


def make_guidance_source(obstacle_spheres, r_ee: float, action_scale, dt: float = 1.0,
                         margin: float = 0.0, n_guide: int = 3, decay: float = 0.5,
                         enable_radius: float = 0.30,
                         stats: GuidanceStats | None = None):
    """A `guidance_source(obs_dict) -> guidance_fn` for GuidedPolicy.

    Needs no hook into the runner: the END-EFFECTOR POSITION IS ALREADY IN THE OBSERVATION.
    pi0.5's 8-D proprio state is eef_pos(3) + axis_angle(3) + gripper(2), so state[:3] is the
    current EE position and the barrier can be built per query from the observation alone. The
    obstacle geometry is static within an episode and is closed over here.

    Returns None when no obstacle is within reach of one action step, so unguided steps cost
    nothing and `GuidanceStats.fire_rate` stays interpretable.
    """
    spheres = [(np.asarray(c, dtype=np.float64), float(r)) for c, r in obstacle_spheres]

    def guidance_source(obs_dict):
        state = np.asarray(obs_dict["observation/state"], dtype=np.float64).ravel()
        ee = state[:3]
        # Gate on `enable_radius`, NOT on dt. These are different quantities: dt scales the
        # DISPLACEMENT ESTIMATE (how far one action moves the EE), while the gate only asks
        # whether the obstacle is near enough to be worth consulting. Deriving the gate from dt
        # meant that correcting dt from 1.0 to 0.05 shrank it from 1.05 m to 0.10 m and guidance
        # stopped being called at all — 0/0, which reads as "inert wiring", not "nothing to do".
        if not any(np.linalg.norm(ee - c) - r <= enable_radius for c, r in spheres):
            return None
        return make_sphere_guidance_fn(ee, spheres, r_ee=r_ee, action_scale=action_scale,
                                       dt=dt, margin=margin, n_guide=n_guide, decay=decay,
                                       stats=stats)

    return guidance_source
