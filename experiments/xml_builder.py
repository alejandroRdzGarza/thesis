"""
Generates a MuJoCo XML string from a SceneConfig.

Uses the existing panda.xml as a base (via <include>) and injects the task
geometry (desk, cubes, obstacles, ghost target, camera) at runtime. This
avoids maintaining a separate XML file per scene.
"""

from __future__ import annotations
import textwrap
from experiments.scene_config import SceneConfig, ObstacleConfig

# Absolute path to panda.xml so MuJoCo can resolve it regardless of cwd.
_PANDA_XML = "./simulation_assets/model/franka_emika_panda/panda.xml"


def _obstacle_xml(obs: ObstacleConfig, idx: int) -> str:
    x, y, z = obs.pos
    r, g, b, a = obs.color
    return textwrap.dedent(f"""\
        <!-- OBSTACLE {idx}: {obs.name} (safety_radius={obs.safety_radius}m) -->
        <body name="{obs.name}" pos="{x:.4f} {y:.4f} {z:.4f}">
          <geom name="{obs.name}_geom" type="sphere" size="{obs.radius:.4f}"
                rgba="{r:.2f} {g:.2f} {b:.2f} {a:.2f}"
                contype="0" conaffinity="0"/>
        </body>""")


def build_scene_xml(cfg: SceneConfig) -> str:
    sx, sy, sz = cfg.start_pos
    gx, gy, gz = cfg.goal_pos

    # Desk centred roughly between start and goal at table height
    desk_x = (sx + gx) / 2
    desk_y = (sy + gy) / 2

    obstacle_blocks = "\n\n    ".join(
        _obstacle_xml(obs, i) for i, obs in enumerate(cfg.obstacles)
    )

    return textwrap.dedent(f"""\
        <mujoco model="panda safety scene — {cfg.name}">
          <include file="{_PANDA_XML}"/>

          <statistic center="0.3 0 0.4" extent="1"/>

          <visual>
            <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
            <rgba haze="0.15 0.25 0.35 1"/>
            <global azimuth="120" elevation="-20"/>
          </visual>

          <asset>
            <texture type="skybox" builtin="gradient"
                     rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
            <texture type="2d" name="groundplane" builtin="checker" mark="edge"
                     rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
                     markrgb="0.8 0.8 0.8" width="300" height="300"/>
            <material name="groundplane" texture="groundplane"
                      texuniform="true" texrepeat="5 5" reflectance="0.2"/>
          </asset>

          <worldbody>
            <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
            <geom name="floor" type="plane" size="0 0 0.05" material="groundplane"/>

            <!-- Coordinate-frame helper arrows -->
            <body name="world_frame" pos="0 0 0">
              <geom name="X" type="cylinder" size="0.005" fromto="0 0 0 0.3 0 0"
                    rgba="1 0 0 1" contype="0" conaffinity="0"/>
              <geom name="Y" type="cylinder" size="0.005" fromto="0 0 0 0 0.3 0"
                    rgba="0 1 0 1" contype="0" conaffinity="0"/>
              <geom name="Z" type="cylinder" size="0.005" fromto="0 0 0 0 0 0.3"
                    rgba="1 0 1 1" contype="0" conaffinity="0"/>
            </body>

            <!-- Ghost target for IK — driven by Python at runtime -->
            <body name="target_ghost" mocap="true" pos="{sx:.4f} {sy:.4f} {sz:.4f}">
              <geom type="sphere" size="0.02" rgba="0.2 0.4 0.8 0.5"
                    contype="0" conaffinity="0"/>
            </body>

            <camera name="static_cam"
              pos="0.05 -0.75 1.3"
              euler="0.8 -0.3236 -0.26"
              fovy="45"/>

            <!-- DESK -->
            <body name="desk" pos="{desk_x:.4f} {desk_y:.4f} 0.1">
              <geom name="desk_geom" type="box" size="0.3 0.3 0.1"
                    rgba="0.7 0.5 0.3 1" contype="1" conaffinity="1"/>
            </body>

            <!-- START MARKER: red cube -->
            <body name="red" pos="{sx:.4f} {sy:.4f} {sz + 0.1:.4f}">
              <freejoint/>
              <geom name="block_red" type="box" size="0.03 0.03 0.03"
                    rgba="1 0 0 1" density="500"/>
            </body>

            <!-- GOAL MARKER: green cube (ghost, no collision) -->
            <body name="green" pos="{gx:.4f} {gy:.4f} {gz - 0.17:.4f}">
              <geom name="block_green" type="box" size="0.03 0.03 0.03"
                    rgba="0.2 0.8 0.2 0.4" contype="0" conaffinity="0"/>
            </body>

            {obstacle_blocks}

          </worldbody>
        </mujoco>""")
