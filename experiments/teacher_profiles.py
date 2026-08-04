"""teacher_profiles.py — per-scene specialisation of the classical expert.

WHY: a SINGLE universal pick-and-place controller cannot cover every SafeLIBERO scene. The
sweep (`sweep_final/per_config.csv`) shows it plainly — object suite ~87% success, but
goal level-I 0/16 with EVERY failure frozen in APPROACH, and several spatial tasks stalled in
DESCEND. The failures are not random: they cluster by SCENE GEOMETRY (where the obstacle sits
relative to the grasp, how high the object is, container-vs-surface placement).

So instead of one config, the teacher is a small library of strategies whose parameters are
RESOLVED FROM THE SCENE. Two layers:

  1. `derive_profile(...)` — automatic, geometry-driven. The main lever is the GRASP SIDE: a wide
     bowl is grasped by pinching its rim, which offsets the grip point ~5 cm along the gripper's
     closing axis (world Y). That offset was hard-coded to +Y, so on any scene whose obstacle sits
     on the +Y side the teacher was told to reach 5 cm TOWARDS the obstacle — the barrier then
     (correctly) refuses, and the arm freezes in APPROACH. Choosing the side that maximises
     obstacle clearance is a per-scene decision that costs nothing and is what a human would do.

  2. `PROFILE_OVERRIDES` / `teacher_profiles.json` — explicit per-(suite, level, task) overrides
     for scenes the automatic rules do not cover, written by `tune_teacher.py` (which searches a
     candidate set per task and keeps the winner). Hand edits are allowed but the tuner is the
     intended author.

The result is still ONE demo format and ONE student: the VLA sees (obs → safe action) and never
knows the teacher differed per scene. Specialising the teacher only raises the ceiling on how
many clean, safe demos exist to distil.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

_PROFILE_JSON = Path(__file__).with_name("teacher_profiles.json")

# Object z above this counts as ELEVATED (on a stove / cabinet shelf rather than the table).
# SafeLIBERO tables sit at z≈0.90; stove/cabinet surfaces at z≈1.13.
ELEVATED_Z = 1.00

# Panda finger separation when fully open. An object wider than this (leaving clearance for the
# finger thickness) cannot be straddled top-down and has to be pinched at an EDGE instead.
GRIPPER_MAX_OPEN = 0.08
STRADDLE_MAX_WIDTH = 0.068


@dataclass
class SceneFeatures:
    """Geometry of one episode, from privileged sim state — the input to strategy selection."""
    obj_pos: np.ndarray
    goal_pos: np.ndarray
    place_mode: str                 # 'in' (drop into a container) | 'on' (set down on a surface)
    grasp_mode: str                 # 'top' | 'rim'
    obstacle_pos: np.ndarray | None  # the ACTIVE obstacle (the one scored by CAR), if any
    obstacle_radius: float
    clutter: list = field(default_factory=list)   # other scene objects to route around (pos, r)
    # Height of the object's TOP above its reported centre, measured from the MuJoCo geoms.
    # LIBERO reports object poses at the body CENTRE, but a top-down grasp has to meet the object
    # at its TOP — 4 cm above centre for a pudding box, 8.5 cm for an orange-juice carton. None
    # when the geometry couldn't be read (rules then fall back to the centre-relative defaults).
    obj_top_dz: float | None = None
    # Object extent along the gripper's closing axis (world Y). The Panda's fingers open to 8 cm,
    # so anything wider CANNOT be straddled top-down — the fingers land on the lid and close on
    # air. None when the geometry couldn't be read.
    obj_width_y: float | None = None

    @property
    def obj_elevated(self) -> bool:
        return float(self.obj_pos[2]) > ELEVATED_Z

    @property
    def goal_elevated(self) -> bool:
        return float(self.goal_pos[2]) > ELEVATED_Z

    @property
    def transport_dist(self) -> float:
        return float(np.linalg.norm(self.goal_pos[:2] - self.obj_pos[:2]))

    def obstacle_gap(self, point) -> float:
        """Clearance from `point` to the nearest keep-out centre (active obstacle + clutter)."""
        pts = ([(self.obstacle_pos, self.obstacle_radius)] if self.obstacle_pos is not None else [])
        pts += list(self.clutter)
        if not pts:
            return float("inf")
        p = np.asarray(point, float)
        return min(float(np.linalg.norm(p - np.asarray(c, float)) - float(r)) for c, r in pts)


@dataclass
class TeacherProfile:
    """The resolved strategy for one scene: how to grasp, and which controller knobs to move."""
    name: str = "default"
    grasp_mode: str | None = None        # 'top' | 'rim'; None = leave the caller's choice alone
    grasp_offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    cfg_overrides: dict = field(default_factory=dict)     # → ControllerConfig fields
    mpc_overrides: dict = field(default_factory=dict)     # → MPCConfig fields
    notes: list = field(default_factory=list)             # why each decision was taken (for logs)

    def apply(self, controller) -> "TeacherProfile":
        """Configure a PickPlaceController in place.

        Controllers are reused across episodes, so the caller's ORIGINAL config is snapshotted on
        first use and restored before each apply — otherwise one scene's overrides would leak into
        the next scene, which is exactly the kind of silent cross-task coupling this module exists
        to remove.
        """
        if not hasattr(controller, "_profile_base"):
            controller._profile_base = (replace(controller.cfg), replace(controller.mpc_cfg))
        base_cfg, base_mpc = controller._profile_base
        controller.cfg = replace(base_cfg, **self.cfg_overrides)
        controller.mpc_cfg = replace(base_mpc, **self.mpc_overrides)
        controller.grasp_offset_xy = np.asarray(self.grasp_offset, float)
        if self.grasp_mode is not None:
            controller.grasp_mode = self.grasp_mode
        controller.profile_name = self.name
        return self

    def describe(self) -> str:
        off = np.round(self.grasp_offset, 3)
        bits = [f"grasp={self.grasp_mode or '-'}", f"offset={off}"]
        if self.cfg_overrides:
            bits.append("cfg=" + ",".join(f"{k}={v}" for k, v in sorted(self.cfg_overrides.items())))
        if self.mpc_overrides:
            bits.append("mpc=" + ",".join(f"{k}={v}" for k, v in sorted(self.mpc_overrides.items())))
        return f"{self.name} [{'  '.join(bits)}]" + (f"  ({'; '.join(self.notes)})" if self.notes else "")


# ── Layer 1: automatic, geometry-driven strategy ─────────────────────────────────────────

def choose_grasp_offset(feat: SceneFeatures, rim_offset: float, approach_h: float) -> tuple[np.ndarray, str]:
    """Pick the grip point offset for a rim grasp: which SIDE of the bowl to pinch.

    The gripper keeps its reset orientation (the controller commands zero rotation), so its
    fingers close along world Y — the rim can only be pinched at obj ± rim_offset·ŷ. Score both
    sides by the clearance of the grip point AND of the hover point above it (the approach also
    has to be safe), and take the roomier one. Ties keep +Y so scenes that were already fine do
    not change.
    """
    if feat.grasp_mode != "rim" or rim_offset == 0.0:
        return np.zeros(3), "top grasp — no lateral offset"
    best, best_score, best_sign = None, -np.inf, None
    for sign in (+1.0, -1.0):
        off = np.array([0.0, sign * abs(rim_offset), 0.0])
        grip = np.asarray(feat.obj_pos, float) + off
        hover = grip + np.array([0.0, 0.0, approach_h])
        score = min(feat.obstacle_gap(grip), feat.obstacle_gap(hover))
        # Prefer +Y on a tie (keeps previously-working scenes byte-identical).
        if score > best_score + 1e-6 or (abs(score - best_score) <= 1e-6 and sign > 0):
            best, best_score, best_sign = off, score, sign
    side = "+Y" if best_sign > 0 else "−Y"
    return best, f"rim grasp on {side} (clearance {best_score:.3f} m)"


def derive_profile(feat: SceneFeatures, base_cfg, base_mpc) -> TeacherProfile:
    """Geometry → strategy. Every rule keys off a measurable scene property, not a task index."""
    cfg_over: dict = {}
    mpc_over: dict = {}
    notes: list[str] = []

    # ── Grasp PRIMITIVE, from measured width ─────────────────────────────────
    # resolve_pick_and_place picks the rim grasp by NAME ("bowl" in the object name). That's a
    # heuristic about SHAPE, so check the actual shape: anything too wide for the 8 cm gripper to
    # straddle gets pinched at an edge instead, like a bowl rim, offset by its own half-width.
    #
    # NB this does NOT currently fire anywhere on SafeLIBERO — every target measures 0.049-0.054 m
    # along the closing axis (measured from the true point cloud, not the bounding sphere). It is a
    # guard against a name heuristic silently mis-classifying a future object, not a fix for the
    # object-suite grasp failure, whose cause is still open.
    grasp_mode = feat.grasp_mode
    rim_offset = base_cfg.rim_offset
    if (feat.obj_width_y is not None and feat.obj_width_y > STRADDLE_MAX_WIDTH
            and grasp_mode != "rim"):
        grasp_mode = "rim"
        rim_offset = round(feat.obj_width_y / 2.0, 4)
        cfg_over["rim_offset"] = rim_offset
        notes.append(f"too wide to straddle ({feat.obj_width_y:.3f} m > {STRADDLE_MAX_WIDTH}) "
                     f"→ edge pinch at ±{rim_offset:.3f}")

    feat = replace(feat, grasp_mode=grasp_mode)
    offset, why = choose_grasp_offset(feat, rim_offset, base_cfg.approach_h)
    notes.append(why)

    grip = np.asarray(feat.obj_pos, float) + offset
    gap = feat.obstacle_gap(grip)

    # ── Obstacle crowding the grasp ──────────────────────────────────────────
    # Even on the roomier side the obstacle can sit close to the grip point. The MPC's keep-out
    # (safety_radius + radius_buffer, engaged from activate_margin away) then covers the approach
    # corridor and the arm hovers instead of descending. Shrink the MPC's EXTRA buffer — the
    # reactive CBF is untouched, so hard safety is unchanged; we are only asking the planner to
    # stop refusing a corridor the barrier itself permits.
    #
    # The threshold is the MPC's own engagement distance measured from the keep-out SURFACE
    # (`obstacle_gap` already subtracts the radius), so this fires exactly when the planner would
    # otherwise be constraining the grasp — and stays quiet on the roomy scenes that already work.
    crowd_tol = base_mpc.radius_buffer + base_mpc.activate_margin
    if gap < crowd_tol:
        mpc_over["radius_buffer"] = 0.0
        mpc_over["activate_margin"] = 0.05
        notes.append(f"tight grasp corridor (gap {gap:.3f} m) → MPC buffer relaxed, CBF unchanged")
        # Come down more vertically so the lateral swing near the obstacle is short.
        cfg_over["approach_h"] = 0.09

    # ── Tall object, top-down grasp ──────────────────────────────────────────
    # The controller aims `grasp_dz` above the object's reported CENTRE, but a top grasp meets the
    # object at its TOP. On the object suite (cartons, boxes) that gap is 4-8.5 cm: the gripper
    # bottoms out on the lid, the geometric grasp trigger never fires, and `_holding` then rejects
    # the grasp because the EE sits further above the centre than `hold_z` allows — the controller
    # deadlocks with the object untouched. Aim just below the lid instead, and widen the hold test
    # to the offset a successful grasp actually produces.
    #
    # Top grasps ONLY. A wide bowl is pinched by the rim: it needs the deep centre-relative target
    # plus the contact-stall detector to seat the fingers on the rim wall, and raising that target
    # to just under the rim would stop the descent before the pinch.
    if feat.grasp_mode == "top" and feat.obj_top_dz is not None and feat.obj_top_dz > 0.02:
        grip_depth = min(0.03, feat.obj_top_dz)      # how far below the lid to place the grip site
        cfg_over["grasp_dz"] = round(feat.obj_top_dz - grip_depth, 4)
        cfg_over["hold_z"] = round(feat.obj_top_dz + 0.04, 4)
        notes.append(f"tall top grasp (top {feat.obj_top_dz:+.3f} above centre) → "
                     f"grip at centre{cfg_over['grasp_dz']:+.3f}, hold_z {cfg_over['hold_z']:.3f}")

    # ── Elevated pick (bowl on a stove / cabinet shelf) ──────────────────────
    if feat.obj_elevated:
        cfg_over["lift_h"] = 0.08          # a full lift drives the EE into its upward reach limit
        cfg_over["descend_z_cap"] = 0.20   # slower descent: extended reaches drift in XY
        cfg_over["stall_patience"] = 16    # extended-reach descents bounce; be surer before grasping
        notes.append(f"elevated pick (obj_z {feat.obj_pos[2]:.2f}) → short lift, gentle descent")

    # ── Elevated placement (set down on a raised surface) ────────────────────
    if feat.goal_elevated:
        cfg_over["goal_clear_h"] = 0.14    # measured from the raised goal, not the table
        cfg_over["setdown_reach"] = 0.04
        notes.append(f"elevated placement (goal_z {feat.goal_pos[2]:.2f}) → lower carry height")

    # ── Long transport across the workspace ──────────────────────────────────
    if feat.transport_dist > 0.35:
        cfg_over["carry_margin"] = 0.06    # more room for the held object on a long swing
        notes.append(f"long transport ({feat.transport_dist:.2f} m) → wider carry margin")

    # ── Placement onto a surface with the obstacle nearby ────────────────────
    if feat.place_mode == "on" and feat.obstacle_gap(feat.goal_pos) < crowd_tol:
        cfg_over["place_xy_tol"] = 0.025   # be precise: a sloppy set-down nudges the obstacle
        cfg_over["setdown_reach"] = 0.04
        notes.append("obstacle near the goal surface → precise, gentle set-down")

    name = "auto"
    if feat.obj_elevated:
        name += "+elevated"
    if gap < crowd_tol:
        name += "+tight"
    return TeacherProfile(name=name, grasp_mode=grasp_mode, grasp_offset=offset,
                          cfg_overrides=cfg_over, mpc_overrides=mpc_over, notes=notes)


# ── Layer 2: explicit per-task overrides ─────────────────────────────────────────────────
# Written by tune_teacher.py; hand edits allowed. Keys are "<suite>|<level>|<task>".
PROFILE_OVERRIDES: dict[str, dict] = {}


def _load_overrides() -> dict:
    merged = dict(PROFILE_OVERRIDES)
    if _PROFILE_JSON.exists():
        try:
            merged.update(json.loads(_PROFILE_JSON.read_text()))
        except Exception as e:                       # a malformed file must not kill a sweep
            print(f"  [teacher_profiles] ignoring {_PROFILE_JSON.name}: {e}")
    return merged


def profile_key(suite: str, level: str, task: int) -> str:
    return f"{suite}|{level}|{task}"


def resolve_profile(feat: SceneFeatures, base_cfg, base_mpc, *,
                    suite: str | None = None, level: str | None = None,
                    task: int | None = None) -> TeacherProfile:
    """The full resolution: geometry-derived strategy, then any per-task override on top."""
    prof = derive_profile(feat, base_cfg, base_mpc)
    if suite is None or level is None or task is None:
        return prof
    over = _load_overrides().get(profile_key(suite, level, task))
    if not over:
        return prof
    if "grasp_offset" in over:
        prof = replace(prof, grasp_offset=np.asarray(over["grasp_offset"], float))
        prof.notes.append("grasp_offset from per-task override")
    elif over.get("flip_side"):
        # "Take the other rim side" — expressed relative to the geometric choice so the override
        # stays valid if the scene (or the clearance rule) changes.
        prof = replace(prof, grasp_offset=-np.asarray(prof.grasp_offset, float))
        prof.notes.append("grasp side flipped by per-task override")
    prof.cfg_overrides.update(over.get("cfg", {}))
    prof.mpc_overrides.update(over.get("mpc", {}))
    prof.name = over.get("name", prof.name + "+override")
    if over.get("cfg") or over.get("mpc"):
        prof.notes.append(f"per-task override {profile_key(suite, level, task)}")
    return prof


def save_overrides(entries: dict, path: Path | None = None) -> Path:
    """Persist tuned per-task profiles (tune_teacher.py writes this)."""
    path = path or _PROFILE_JSON
    existing = {}
    if path.exists():
        try:
            existing = json.loads(path.read_text())
        except Exception:
            existing = {}
    existing.update(entries)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
    return path


# ── CPU self-test: the grasp-side rule on the real goal-LI geometry ──────────────────────
if __name__ == "__main__":
    from experiments.classical_expert import ControllerConfig, MPCConfig

    cfg, mpc = ControllerConfig(), MPCConfig()
    ok = True

    # safelibero_goal LI t0 (from sweep_final logs): obstacle 0.23 m away on +Y — the hard-coded
    # +Y rim offset reached TOWARDS it and the run froze in APPROACH for all 4 episodes.
    feat = SceneFeatures(obj_pos=np.array([-0.098, -0.009, 0.898]),
                         goal_pos=np.array([0.092, 0.021, 0.902]),
                         place_mode="on", grasp_mode="rim",
                         obstacle_pos=np.array([-0.07, 0.225, 1.008]), obstacle_radius=0.10)
    p = resolve_profile(feat, cfg, mpc)
    print("goal LI t0 →", p.describe())
    ok &= p.grasp_offset[1] < 0        # must pick the −Y side, away from the obstacle
    ok &= not p.mpc_overrides           # …but this scene is roomy enough not to touch the planner

    # Same layout with the obstacle pulled in until it crowds the grasp corridor → relax the MPC.
    feat_tight = replace(feat, obstacle_pos=np.array([-0.07, 0.06, 0.95]))
    p_tight = resolve_profile(feat_tight, cfg, mpc)
    print("crowded grasp →", p_tight.describe())
    ok &= p_tight.mpc_overrides.get("radius_buffer") == 0.0

    # safelibero_spatial LI t0: obstacle on the −Y side → keep the +Y rim (this task already works).
    feat2 = SceneFeatures(obj_pos=np.array([-0.063, 0.202, 0.898]),
                          goal_pos=np.array([0.053, 0.205, 0.902]),
                          place_mode="on", grasp_mode="rim",
                          obstacle_pos=np.array([-0.052, 0.005, 1.008]), obstacle_radius=0.10)
    p2 = resolve_profile(feat2, cfg, mpc)
    print("spatial LI t0 →", p2.describe())
    ok &= p2.grasp_offset[1] > 0

    # An elevated pick must shorten the lift and slow the descent.
    feat3 = SceneFeatures(obj_pos=np.array([-0.20, 0.20, 1.13]),
                          goal_pos=np.array([0.05, 0.20, 0.90]),
                          place_mode="on", grasp_mode="rim",
                          obstacle_pos=np.array([-0.05, 0.0, 1.01]), obstacle_radius=0.10)
    p3 = resolve_profile(feat3, cfg, mpc)
    print("elevated pick →", p3.describe())
    ok &= p3.cfg_overrides.get("lift_h") == 0.08

    # A top grasp (carton) takes no lateral offset.
    feat4 = SceneFeatures(obj_pos=np.array([-0.10, 0.0, 0.92]),
                          goal_pos=np.array([0.10, 0.0, 0.92]),
                          place_mode="in", grasp_mode="top",
                          obstacle_pos=np.array([0.0, 0.20, 1.0]), obstacle_radius=0.10)
    p4 = resolve_profile(feat4, cfg, mpc)
    print("top grasp →", p4.describe())
    ok &= float(np.linalg.norm(p4.grasp_offset)) == 0.0

    # A per-task override must layer on top of (not replace) the geometric strategy.
    PROFILE_OVERRIDES["safelibero_goal|I|0"] = {"flip_side": True, "cfg": {"approach_h": 0.16}}
    p5 = resolve_profile(feat, cfg, mpc, suite="safelibero_goal", level="I", task=0)
    print("override →", p5.describe())
    ok &= p5.grasp_offset[1] > 0                       # flipped back off the geometric −Y choice
    ok &= p5.cfg_overrides["approach_h"] == 0.16
    PROFILE_OVERRIDES.pop("safelibero_goal|I|0")

    # apply() must not leak one scene's overrides into the next controller episode.
    from experiments.classical_expert import PickPlaceController
    ctrl = PickPlaceController()
    p3.apply(ctrl)
    ok &= ctrl.cfg.lift_h == 0.08
    p.apply(ctrl)
    ok &= ctrl.cfg.lift_h == ControllerConfig().lift_h   # restored, not stuck at the elevated value
    ok &= float(ctrl.grip_point(feat.obj_pos)[1] - feat.obj_pos[1]) < 0   # profile drives the grip point

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
