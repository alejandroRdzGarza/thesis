"""
Generates a MuJoCo XML string from a SceneConfig.

Uses the existing panda.xml as a base (via <include>) and injects the task
geometry at runtime.

Visual design
-------------
- Dark optical-table surface with fine checker texture (mimics breadboard)
- Tall green block (freejoint, moveable) at start_pos — the object to move
- Flat red circular pad at goal_pos — the target marker
- Obstacles rendered as cylinders rising from the table surface, with a faint
  transparent sphere showing the CBF safety-exclusion boundary
- Two cameras:
    static_cam   — Bridge V2 side-front view, used for VLA inference (do not move)
    display_cam  — angled top-down, default viewer camera for qualitative assessment
"""

from __future__ import annotations
import textwrap
from pathlib import Path
from experiments.scene_config import SceneConfig, ObstacleConfig

# Directory that contains panda.xml and its assets/ subfolder.
# The generated XML is written here so that meshdir="assets" resolves correctly.
PANDA_DIR = Path("simulation_assets/model/franka_emika_panda")
_PANDA_XML = "panda.xml"


# Table surface z-height — green block and red pad sit on this plane.
_TABLE_TOP_Z = 0.20


def _obstacle_xml(obs: ObstacleConfig, idx: int) -> str:
    x, y, z = obs.pos
    r, g, b, a = obs.color
    # Visual cylinder rises from table surface to obs.pos[2] + obs.radius,
    # making it look like a bottle / can sitting on the table.
    # The CBF constraint still uses obs.pos as the sphere centre — the visual
    # shape is an approximation for rendering only.
    cyl_top = z + obs.radius
    return textwrap.dedent(f"""\
        <!-- OBSTACLE {idx}: {obs.name}  r_safe={obs.safety_radius:.2f}m -->
        <body name="{obs.name}" pos="0 0 0">
          <!-- Cylinder from table surface upward — bottle / can aesthetic -->
          <geom name="{obs.name}_cyl" type="cylinder"
                fromto="{x:.4f} {y:.4f} {_TABLE_TOP_Z:.4f}  {x:.4f} {y:.4f} {cyl_top:.4f}"
                size="{obs.radius:.4f}"
                rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"
                contype="0" conaffinity="0"/>
          <!-- Safety-exclusion sphere (very transparent) — CBF boundary -->
          <geom name="{obs.name}_safe" type="sphere"
                pos="{x:.4f} {y:.4f} {z:.4f}" size="{obs.safety_radius:.4f}"
                rgba="{r:.2f} {g:.2f} {b:.2f} 0.07"
                contype="0" conaffinity="0"/>
        </body>""")


def build_scene_xml(cfg: SceneConfig) -> str:
    sx, sy, sz = cfg.start_pos
    gx, gy, gz = cfg.goal_pos

    table_x = (sx + gx) / 2
    table_y = (sy + gy) / 2

    obstacle_blocks = "\n\n    ".join(
        _obstacle_xml(obs, i) for i, obs in enumerate(cfg.obstacles)
    )

    # Green block: centre = table_top + half_height = 0.20 + 0.05 = 0.25
    green_z = _TABLE_TOP_Z + 0.05
    # Red pad:   centre = table_top + half_height = 0.20 + 0.003 = 0.203
    red_z   = _TABLE_TOP_Z + 0.003

    return textwrap.dedent(f"""\
        <mujoco model="panda safety scene — {cfg.name}">
          <include file="{_PANDA_XML}"/>

          <statistic center="0.5 0 0.4" extent="1.1"/>

          <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.25 0.25 0.25" specular="0 0 0"/>
            <rgba haze="0.05 0.05 0.08 1"/>
            <global azimuth="140" elevation="-25"/>
          </visual>

          <asset>
            <texture type="skybox" builtin="gradient"
                     rgb1="0.18 0.20 0.25" rgb2="0.04 0.04 0.06"
                     width="512" height="3072"/>
            <!-- Fine checker floor — approximates optical breadboard dot pattern -->
            <texture type="2d" name="floor_tex" builtin="checker"
                     rgb1="0.16 0.16 0.16" rgb2="0.10 0.10 0.10"
                     width="512" height="512"/>
            <material name="floor_mat" texture="floor_tex"
                      texuniform="false" texrepeat="60 60" reflectance="0.02"/>
            <!-- Table top: same dark optical-table aesthetic, coarser grid -->
            <texture type="2d" name="table_tex" builtin="checker"
                     rgb1="0.15 0.15 0.15" rgb2="0.09 0.09 0.09"
                     width="512" height="512"/>
            <material name="table_mat" texture="table_tex"
                      texuniform="false" texrepeat="22 20" reflectance="0.04"/>
          </asset>

          <worldbody>
            <!-- Three-point lighting -->
            <light name="key"  pos="0.5 -1.0 2.0" dir="0  0.4 -1"
                   directional="true" diffuse="0.80 0.80 0.80" specular="0.1 0.1 0.1"/>
            <light name="fill" pos="-0.5 0.6 1.5"  dir="0.2 -0.2 -1"
                   directional="true" diffuse="0.35 0.35 0.35" specular="0 0 0"/>
            <light name="rim"  pos="0.5  1.0 1.5"  dir="0  -0.4 -1"
                   directional="true" diffuse="0.15 0.15 0.15" specular="0 0 0"/>

            <geom name="floor" type="plane" size="0 0 0.05" material="floor_mat"/>

            <!-- Ghost target: subtle yellow dot tracking the EE goal -->
            <body name="target_ghost" mocap="true" pos="{sx:.4f} {sy:.4f} {sz:.4f}">
              <geom type="sphere" size="0.010" rgba="0.95 0.85 0.10 0.45"
                    contype="0" conaffinity="0"/>
            </body>

            <!-- VLA camera: side-front matching Bridge V2 training distribution.
                 Used by Renderer for inference — do not change position. -->
            <camera name="static_cam"
              pos="0.15 -0.78 0.55"
              euler="1.15 0 -0.18"
              fovy="50"/>

            <!-- Display camera: angled top-down for qualitative assessment.
                 Runner sets this as the default viewer camera (press F to cycle). -->
            <camera name="display_cam"
              pos="0.25 -0.88 1.10"
              euler="0.82 0 -0.10"
              fovy="52"/>

            <!-- TABLE: dark optical-table surface, 96×84 cm -->
            <body name="table" pos="{table_x:.4f} {table_y:.4f} 0.10">
              <geom name="table_top" type="box" size="0.48 0.42 0.10"
                    material="table_mat" contype="1" conaffinity="1"/>
            </body>

            <!-- GREEN BLOCK: freejoint so runner.py can move it for virtual grasp.
                 contype/conaffinity=0 so the arm cannot physically knock it over;
                 runner.py anchors it at spawn when not grasped. -->
            <body name="green_block" pos="{sx:.4f} {sy:.4f} {green_z:.4f}">
              <freejoint name="green_freejoint"/>
              <geom name="green_geom" type="box" size="0.025 0.025 0.050"
                    rgba="0.15 0.78 0.15 1.0" contype="0" conaffinity="0"/>
            </body>

            <!-- RED TARGET: flat circular pad on table surface, visual only -->
            <body name="red_target" pos="{gx:.4f} {gy:.4f} {red_z:.4f}">
              <geom name="red_geom" type="cylinder" size="0.055 0.003"
                    rgba="0.85 0.10 0.10 0.90" contype="0" conaffinity="0"/>
            </body>

            {obstacle_blocks}

          </worldbody>
        </mujoco>""")
