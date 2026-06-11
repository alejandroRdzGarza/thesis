### Introduction

This branch has stored codes for VLA implementation, data collection, mujoco simulation. For any detail instructions, you can follow the **README.md** under the corresponding file.

## Data Collection
This folder contains code to generate trajectories datas, for later fine-tuning stage.

## VLA Model
This folder contains code of different VLA we have tested on. 


## Simulation Assets
This folder contains all the assets used in our mujoco environment.

There is a blue ball objects to visualize the desired position of the manipulator, and some of the code may not using it. It can be unvisualize simply by comment it out in the **scene.xml**

To modify the scene, please check the **scene.xml** and **panda.xml** (where the end-effector camera been add) in **franka_emika_panda** folder

To inspect the simulation scene, run **simulation.py**. Three windows (Mujoco simulation, static camera, end-effector camera) should pop out. 

