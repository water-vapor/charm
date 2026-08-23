# CHARM: Structured Sparse Memory for Recurrent Reasoning

Code release for the paper [Structured Sparse Memory for Recurrent Reasoning](https://arxiv.org/abs/TODO).
Checkpoints and data are hosted at [water-vapor-vx/charm-arc-agi](https://huggingface.co/datasets/water-vapor-vx/charm-arc-agi).
The generators and the dataset for the synthetic Re-ARC2 are at [synth-rearc](https://github.com/water-vapor/synth-rearc).

## Highlights

CHARM is an ARC-AGI system trained from scratch. It replaces the dominant per-instance task embedding table of recurrent ARC models with CoSE, a compositional sparse embedding using ~1/15 of the task-memory parameters. With matched synthetic data (Re-ARC for ARC-AGI-1, our Re-ARC2 for ARC-AGI-2), it achieves the best reported public-evaluation accuracy among train-from-scratch systems:

|           | Synthetic data | Backbone params | Total params (ARC-1) | ARC-AGI-1 pass@2 | ARC-AGI-2 pass@2 |
|-----------|----------------|-----------------|----------------------|------------------|------------------|
| HRM       | –              | 27M             | 475M                 | 40.3             | 5.0              |
| TRM       | –              | 7M              | 455M                 | 44.6             | 7.8              |
| URM       | –              | 14M             | 462M                 | 61.8             | 18.2             |
| VARC      | Re-ARC         | 73M             | 73M                  | 60.4             | 11.1             |
| **CHARM** | Re-ARC(2)      | 14M             | **44M**              | **84.0**         | **46.7**         |

## Getting started

```bash
pip install -r requirements.txt
pip install -e .
hf download water-vapor-vx/charm-arc-agi --repo-type dataset --local-dir . --include "data/*" "checkpoints/*"
```

The full checkpoint set is ~140 GB: we release checkpoints every 10k steps to enable further research. The evaluations below only need `data/` plus checkpoints 1510000–1590000 (ARC-AGI-1) and 2290000–2380000 (ARC-AGI-2), under 10 GB total; narrow the `--include` patterns to save disk space.

Scripts run on all detected GPUs via torchrun; override with `NUM_GPUS=...`.

## Evaluate released checkpoints

Reproduces the final results (84.0 / 46.7 pass@2):

```bash
scripts/cose/eval_arc_agi_1_extended.sh
scripts/cose/eval_arc_agi_2_extended.sh
```

## Train from scratch

```bash
scripts/cose/train_arc_agi_1_charm.sh   # 600k steps
scripts/cose/train_arc_agi_2_charm.sh   # 1M steps
```

These use the paper's standard budgets and reach 79.6 / 38.5 pass@2. The released checkpoints come from extending these same runs; to continue one, append `+load_checkpoint=latest train_steps_override=1600000`.

## Baselines and ablations

`scripts/baselines/` retrains the TRM and URM baselines. `scripts/ablations/` covers the paper's main ablations: `task_memory/` (CoSE variants), `color_pool/` (color-permutation pooling), `data_mix/` (synthetic-data ablations), `backbone/` (representative reasoning-depth/learning-horizon configurations), and `extra_ttc/` (replay-decay evaluator). The appendix sweeps are flag variations of these scripts.

## OOD generalization and continual learning

`scripts/cose_cl/` contains the ARC-2 OOD evaluation, per-puzzle test-time training, and continual-learning experiments on ARC-1 checkpoints. See `scripts/cose_cl/README.md` for more details.

## Citation

```bibtex
@misc{zhao2026structuredsparsememory,
  title = {Structured Sparse Memory for Recurrent Reasoning},
  author = {Zhao, Zixuan and Wheeler, Samuel and Getty, Neil and Duan, Xiaotian and Stevens, Rick and Xia, Fangfang},
  year = {2026}
}
```
