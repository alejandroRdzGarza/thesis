### Data Collection

Code to generate trajectory datas for later fine-tuning. There is only move to the cube trajectories has built. 

## Instruction

### Trajectories data collection

```
# Create environment
conda create -n data_collect python=3.10 -y
conda activate data_collect

# install library for simulation
cd data_collection
pip install -r requirements.txt
```

To generate trajectory datas, run **collect_data.py**. A file called **dataset** will created, storing every episode (10 in default) inside, and theres further datas for each steps under the episode folder. In default, it saved:

- Static camera image
- Gripper camera image
- End effector position
- End effector quaternion
- Target position (Red cube position)
- Delta position
- Episode ID
- Step ID
- Success (True/False)
- Task instruction prompt

These config can simply change in the code.

To inspect the step's data, run **inspect_data.py** with the desired path.

