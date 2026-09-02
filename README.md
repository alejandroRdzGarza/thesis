# Distilling Runtime Safety into VLA Robot Policies

MSc Robotics and Computation thesis, UCL. **[Project page →](https://alejandroRdzGarza.github.io/thesis/)**

Vision-language-action policies are competent manipulators and unsafe ones: on the
SafeLIBERO benchmark, π<sub>0.5</sub> displaces a non-target obstacle in 82.5% of episodes.
The usual remedy is a control-barrier-function shield that projects every proposed action
onto a safe set — which works, but stays attached for the life of the deployment.

This work asks whether that corrective behaviour can be amortised into the weights instead.
Distilling π<sub>0.5</sub> on its own shielded rollouts cuts collisions to **19.2% with no
shield at inference**, while raising task success from 58.3% to **82.5%** on held-out
initial states. The transfer has a boundary, and mapping it is the contribution: a
geometry-privileged motion planner, supplying corrections of equal density under the same
shield, transferred nothing measurable. What transfers is correction at states the policy
itself reaches.

## Layout

| Path | What it is |
| --- | --- |
| [`experiments/`](experiments/) | The working code — CBF shield, barrier geometry, LIBERO runner, teachers, distillation, figure scripts |
| [`docs/`](docs/) | The project page (GitHub Pages), including 68 side-by-side demo clips |
| [`figures/`](figures/) | Figures in the thesis, and the scripts' output |
| [`thesis/`](thesis/), `official_thesis_draft.tex` | The written thesis |
| [`openpi_patches/`](openpi_patches/) | Patches applied to openpi for flow-SDE GRPO |
| [`runpod/`](runpod/), [`cs_timeshare/`](cs_timeshare/), [`myriad/`](myriad/) | Cluster setup and tunnelling scripts |
| [`simulation_assets/`](simulation_assets/) | Franka Panda MuJoCo model |
| `VLA-Model/`, `data_collection/` | Early OpenVLA-7B work, superseded by the π<sub>0.5</sub> pipeline |

Third-party dependencies are **not** vendored here — see [THIRD_PARTY.md](THIRD_PARTY.md)
for openvla-oft, SafeLIBERO/VLSA, and openpi.

## Running things

The cluster scripts read site-specific settings from the environment rather than hardcoding
them. Set what applies to your machine:

```bash
export UCL_USER=<your-username>      # or CS_USER for cs_timeshare/
export UCL_GATEWAY=<gateway-host>
export UCL_BASE=/path/to/your/project/dir
export GPU_HOST=<gpu-machine>
```

Key entry points:

```bash
# Evaluate a policy arm on SafeLIBERO
bash run_eval_arm.sh

# Collect shielded demonstrations, then distil on them
bash run_shielded_distill.sh

# The matched control (shield off during collection)
bash run_shield_control.sh

# Rebuild the thesis figures
PYTHONPATH=. python -m experiments.make_fig_internalisation
```

## Citing

The benchmark and the runtime-shielding baseline are due to Hu et al.,
*[VLSA: Vision-Language-Action Models with Plug-and-Play Safety Constraint Layer](https://arxiv.org/abs/2512.11891)*.
The base policy is π<sub>0.5</sub> (Physical Intelligence); the environment is LIBERO on
robosuite / MuJoCo.
