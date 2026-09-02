# Distilling Runtime Safety into Vision-Language-Action Policies

Code accompanying the MSc Robotics and AI thesis of the same name, University College London.

**[Project page with results and demo videos →](https://alejandroRdzGarza.github.io/thesis/)**

The project asks whether the collision avoidance supplied by an inference-time control barrier
function (CBF) shield can be amortised into the weights of a vision-language-action policy, so that
the shield can be removed at deployment. Everything here operates on **SafeLIBERO**, a MuJoCo
manipulation benchmark with obstacles placed in the robot's path.

---

## Where the report's claims live in this code

The report is [`docs/assets/thesis.pdf`](docs/assets/thesis.pdf). Each row below maps a part of it
to the code that produced it.

| Report section | Component | Files |
|---|---|---|
| §3.5 Obstacle geometry and the barrier | Sphere decomposition, `h_ij` | `experiments/cbf_ellipsoid.py` |
| §3.6 The CBF shield | The QP, arm-link constraints, discrete-time margins | `experiments/libero_runner.py` |
| §3.3 Metrics | TSR, collision, CAR, ETS, activation rate | `experiments/metrics.py` |
| §3.7.1 Shielded self-teacher | Rollout + retention filter | `experiments/collect_shielded_demos.py` |
| §3.7.2 Privileged planner | RRT-Connect, IK, playback | `experiments/rrt_planner.py`, `planner_expert.py`, `classical_expert.py` |
| §3.9 Distillation | Flow-matching behaviour cloning, LoRA on the action head | `experiments/flow_bc_train.py` |
| §4.5 State coverage | Matched-sample coverage analysis | `experiments/make_fig_state_coverage.py` |
| §4.5 Observation aliasing | The prediction the data refuted | `experiments/aliasing_diagnostic.py` |
| §4.6 Per-body attribution | Culprit decomposition | `experiments/culprits_from_log.py` |
| Figures 3.1–5.1 | Every figure in the report | `experiments/make_fig_*.py` |

The central experiment is the comparison between two demonstration sources
(`collect_shielded_demos.py` against `planner_expert.py`), distilled identically by
`flow_bc_train.py` and evaluated with the shield detached.

---

## Dependencies not included here

Three third-party repositories are required at runtime and are **deliberately not vendored**, to
keep this submission to the work that is my own:

- **openpi** — serves the `π₀.₅` policy. Modifications needed for this project are supplied as
  patches in `openpi_patches/`; apply with `git apply`.
- **LIBERO / SafeLIBERO** — the benchmark suites, task definitions and scene assets.
- **robosuite / MuJoCo** — simulation backend.

Python packages: `numpy`, `scipy`, `mujoco`, `torch`, `jax`, `flax`, `optax`, `cvxpy`, `osqp`,
`matplotlib`, `opencv-python`, `pillow`. See `requirements_libero.txt`.

The QP is solved with **OSQP** through **CVXPY**, at roughly 0.36 ms per control step.

---

## Running the pipeline

The policy is served over a websocket; scripts assume it is reachable on `localhost:8000`.

```bash
# 1. collect demonstrations with the shield active
bash run_collect_all.sh

# 2. distil them into the action head
bash run_shielded_distill.sh

# 3. evaluate one arm, with or without the shield at inference
bash run_eval_arm.sh

# 4. the matched control of §4.4 — identical filter, shield off during collection
bash run_shield_control.sh
```

Figures are regenerated from evaluation output:

```bash
PYTHONPATH=. python -m experiments.make_fig_internalisation
PYTHONPATH=. python -m experiments.make_fig_state_coverage
PYTHONPATH=. python -m experiments.make_fig_filmstrip --list   # inventory paired episodes
```

## Tests

```bash
PYTHONPATH=. python -m pytest experiments/test_*.py
```

Nine test files cover the flow-SDE conversion (`test_flow_sde.py`, `test_pi0_flow_sde.py`), the
reward decomposition (`test_safe_reward.py`), the RRT transfer (`test_rrt_transfer.py`), the
sphere-decomposition barrier (`test_cbf_spheres.py`), and the rollout and trace machinery.

---

## The project page

`docs/` is a static site published with GitHub Pages: the argument, the figures, the results
tables, and 68 paired demo clips showing the base policy and the distilled policy on the same
scene from the same held-out initial state. It has no build step — see `docs/README.md`.

---

## What is not included, and why

- **Model checkpoints and evaluation traces** (~30 GB): distilled adapters, the six evaluation arms
  at 120 rollouts each, and the demonstration sets. Available on request.
- **Vendored third-party repositories** (~5 GB): see *Dependencies* above.
- **The abandoned OpenVLA-7B phase** (§3.7.3): the first distillation attempt, which produced no
  usable learning signal and was replaced by `π₀.₅`. Reported in the text; code omitted as it is
  superseded.
- **Literature PDFs**: not redistributed.

## Note on AI assistance

See the *Implementation notes* appendix of the report.
