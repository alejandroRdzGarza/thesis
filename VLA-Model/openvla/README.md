### Mujoco Simulation (OpenVLA)

A vla inference server (**openvla_server.py**) and a mujoco simulation code (**openvla_server.py**). The vla inference server code will load the model, receive image and text instruction from the simulation code, output the action sequence, and pass it back to the simulation code to do the control part. 

## Instruction

### Single Robot Simulation

```
# Create environment
conda create -n openvla python=3.10 -y
conda activate openvla

# install library for simulation
cd openvla
pip install -r requirements.txt
```


First, you have to download the pre-trained openvla checkpoint on Huggin Face (https://huggingface.co/openvla/openvla-7b)

After everything has set, you can run the **openvla_server.py**. Once the model is loaded and the server has started, run the **openvla_server.py**, and the mujoco should be visualized. 

The **simulation.py** in the file simulation assets may help a bit if you want to modify the scene with some visualization.

In the result, without any fine tuning, the OpenVLA cannot perform anything make sense. It may due to the camera setting, domain gap, or just the inference code issue, and the problem will be studied.

