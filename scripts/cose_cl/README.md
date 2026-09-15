# OOD generalization and continual learning

These experiments evaluate various ARC-1 checkpoints on unseen ARC-2 puzzles, either independently or as a sequential stream. The ARC-2 training set contains 233 unseen puzzles, and the public evaluation set contains 116.

## Setup

```bash
hf download water-vapor-vx/charm-arc-agi --repo-type dataset --local-dir . \
  --include "arc1-cl-base-muon-600k/*" "data/augmented/v2/*"
```

The eight ARC-1 checkpoints use Muon and were taken at step 600000.
All launchers default to `arc1-cl-base-muon-600k/` in the project root and
`CHECKPOINT_STEP=600000`. Set `CHECKPOINT_DIR` to use this folder from another
location. The default `cose_lowrank` checkpoint does not use gate.

| Variant | Task memory |
|---|---|
| `cose_lowrank` | The default setup used in the paper: CoSE with a rank-32 per-instance residual. |
| `cose_fulltable` | CoSE with a full-width (512-dimensional) per-instance residual. |
| `table_lowrank` | No CoSE; a rank-32 per-instance task table. |
| `table_fullrank` | No CoSE; the standard 512-dimensional per-instance task table. |
| `compo_only` | The CoSE compositional branch without a per-instance residual. |
| `cose_lowrank_gate` | CoSE with a gated rank-32 per-instance residual. |
| `cose_lowrank_t32` | CoSE with 32-dimensional task-ID rows and a rank-32 per-instance residual. |
| `compo_only_t32` | The compositional branch with 32-dimensional task-ID rows and no per-instance residual. |

To disable halt loss, append `--halt_loss_weight 0.0` to any CoSE-CL launcher or baseline to use
`lm_loss + 0.0 * (q_halt_loss + q_continue_loss)`. The default coefficient is
`0.5`.

The folder includes representative launchers for the four frozen CL variants,
the five reset baselines, a full-model EWC run, and a frozen CoSE low-rank run
on the ARC-2 evaluation stream. For example:

```bash
scripts/cose_cl/arc2_training_cose_lowrank_frozen.sh
SEED=2 scripts/cose_cl/arc2_training_table_fullrank_reset.sh
scripts/cose_cl/arc2_eval_cose_lowrank_frozen.sh
```

## Terminology

| Name | Meaning |
|---|---|
| Zero-shot | Evaluation of an ARC-1 checkpoint on unseen ARC-2 puzzles without learning. |
| Reset | Per-puzzle test-time training: the method resets the weights before each puzzle and then fine-tunes the full model on that puzzle's demonstrations. This is not continual learning. |
| Frozen CL | Trains and retains only the new puzzle-specific embeddings while keeping the shared backbone frozen. |
| Naive CL | Fine-tunes the full model sequentially without protection against forgetting. |
| Joint / joint-all | Both train on all stream puzzles together; `joint` updates new rows only, while `joint_all` also updates shared weights. Neither is continual learning. |
| EWC / L2-SP / rehearsal | Continual-learning baselines that regularize shared weights or replay ARC-AGI-1 data. `all` and `comp` select all shared weights or only CoSE composition parameters. |

## Other configurations

`run.sh` launches one configuration at a time:

```bash
SEED=0 scripts/cose_cl/run.sh STREAM VARIANT METHOD [trainer args...]
```

Replace `VARIANT` with one of the variants in the table above.
Replace `SCOPE` with `all` or `comp`.

| Stream | Configuration | CLI |
|---|---|---|
| ARC-2 training | Naive CL | `scripts/cose_cl/run.sh arc2_training cose_lowrank naive` |
| ARC-2 training | Joint rows, four main variants | `scripts/cose_cl/run.sh arc2_training VARIANT joint` |
| ARC-2 training | Joint full model, all eight variants | `scripts/cose_cl/run.sh arc2_training VARIANT joint_all` |
| ARC-2 training | EWC, composition scope | `scripts/cose_cl/run.sh arc2_training cose_lowrank ewc_comp --reg_lambda 1e-2 --fisher_batches 64` |
| ARC-2 training | L2-SP, either scope | `scripts/cose_cl/run.sh arc2_training cose_lowrank l2sp_SCOPE --reg_lambda 1e-1` |
| ARC-2 training | Rehearsal, either scope | `scripts/cose_cl/run.sh arc2_training cose_lowrank rehearsal_SCOPE --rehearsal_every 2` |
| ARC-2 eval | Frozen CL, remaining main variants | `scripts/cose_cl/run.sh arc2_eval VARIANT frozen` |
| ARC-2 eval | Reset, all eight variants | `scripts/cose_cl/run.sh arc2_eval VARIANT reset` |
| ARC-2 eval | Joint full model, all eight variants | `scripts/cose_cl/run.sh arc2_eval VARIANT joint_all` |
