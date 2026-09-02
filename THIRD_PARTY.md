# Third-party code

Two upstream repositories are used by this project and are **not** included here.
Clone them yourself; nothing in this repo redistributes them.

## openvla-oft

[moojink/openvla-oft](https://github.com/moojink/openvla-oft) — MIT, © 2025 Moo Jin Kim,
Chelsea Finn, Percy Liang. Used by the early OpenVLA-7B experiments, notably
`finetune_cbf.py`, which imports `prismatic.*` from it.

```
git clone https://github.com/moojink/openvla-oft.git ../openvla-oft
export OPENVLA_OFT_DIR=../openvla-oft
```

`runpod/setup_ucl.sh` and `runpod/setup_oft.sh` clone and patch it automatically on a
remote machine.

## VLSA / AEGIS and SafeLIBERO

[vlsa-aegis](https://vlsa-aegis.github.io/) — the SafeLIBERO benchmark and the
runtime-shielding baseline this work compares against
([arXiv:2512.11891](https://arxiv.org/abs/2512.11891)). The benchmark suite definitions
(`safelibero_spatial`, `safelibero_object`, `safelibero_goal`) come from it; this repo
depends on having it available but does not vendor it.

## Everything else

`openpi` (π0.5, Physical Intelligence), LIBERO, robosuite and MuJoCo are installed as
normal dependencies — see the `requirements_*.txt` files and `runpod/setup_*.sh`.
Patches this project applies to openpi live in `openpi_patches/`.
