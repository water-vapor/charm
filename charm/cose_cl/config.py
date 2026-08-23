"""CoSE-CL: continual learning of new puzzles on a frozen ARC1 donor checkpoint.

Each stream puzzle gets fresh rows appended to the donor's embedding tables; what
else trains depends on the arm:
  frozen     new rows only (the method); donor weights provably untouched
  naive      everything, sequentially (catastrophic-forgetting lower bound)
  reset      everything, but the donor is restored before every puzzle
             (per-puzzle fine-tuning oracle; no continual learning)
  joint      new rows only, all stream puzzles trained at once (order-free reference)
  joint_all  everything, all stream puzzles at once (multi-task upper bound)

The "pure composition" baseline is the zero-shot sweep every run performs before
training: the expanded model with blank rows.

Defaults are the validated recipe: 1000 steps/puzzle, global batch 128, no early
stopping, 128-augmentation voting eval. Runs single-GPU (python -m) or DDP
(torchrun); batch_size is the global batch, so the recipe is world-size invariant.
"""

import argparse
from dataclasses import dataclass


@dataclass
class Config:
    ckpt: str
    ckpt_config: str
    stream_parquet: str
    out_dir: str
    data_dir: str
    arm: str
    n_puzzles: int
    order_seed: int
    seed: int

    stage_steps: int
    batch_size: int
    warmup_steps: int
    embed_lr: float
    sparse_lr: float
    sparse_weight_decay: float
    ft_lr: float
    ft_weight_decay: float
    beta1: float
    beta2: float
    no_translation_ratio: float

    eval_batch_size: int
    eval_augs: int
    eval_every: int
    arc1_eval_puzzles: int
    arc1_eval_augs: int

    project_name: str
    run_name: str
    resume: bool
    device: str


def add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--ckpt", required=True, help="donor checkpoint .pt")
    p.add_argument("--ckpt_config", required=True, help="donor training config yaml")
    p.add_argument("--stream_parquet", required=True, help="augmented v2 parquet of the CL stream")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--data_dir", default="data/augmented/v2",
                   help="local dir holding the donor's training parquets (basename-matched)")
    p.add_argument("--n_puzzles", type=int, default=0, help="0 = all donor-unseen stream puzzles")
    p.add_argument("--order_seed", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--stage_steps", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=128,
                   help="global batch; must be divisible by the distributed world size")
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--embed_lr", type=float, default=1e-2, help="new semantic rows (row-local AdamW)")
    p.add_argument("--sparse_lr", type=float, default=1e-2, help="instance rows (row-local SignSGD)")
    p.add_argument("--sparse_weight_decay", type=float, default=0.1)
    p.add_argument("--ft_lr", type=float, default=1e-4, help="shared weights, when the arm trains them")
    p.add_argument("--ft_weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--no_translation_ratio", type=float, default=0.2)

    p.add_argument("--eval_batch_size", type=int, default=256)
    p.add_argument("--eval_augs", type=int, default=128,
                   help="offline augs per puzzle for voting eval (identity first)")
    p.add_argument("--eval_every", type=int, default=15,
                   help="full-stream re-check sweep every K stages (plus start/end)")
    p.add_argument("--arc1_eval_puzzles", type=int, default=60,
                   help="ARC1 retention probe size (before and after the stream); 0 disables")
    p.add_argument("--arc1_eval_augs", type=int, default=64)

    p.add_argument("--project_name", default="cose-cl")
    p.add_argument("--run_name", default="")
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--device", default="cuda", choices=["cuda"])


def parse_args() -> Config:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arm", default="frozen",
                   choices=["frozen", "naive", "reset", "joint", "joint_all"])
    add_shared_args(p)
    return Config(**vars(p.parse_args()))
