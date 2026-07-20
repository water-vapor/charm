from dataclasses import dataclass
from typing import Optional, Any, Sequence, List, Literal

import pydantic
import torch
from torch import nn


class LossConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="allow")
    name: str
    loss: LossConfig


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    # This release supports the ARC-AGI setup used in the paper.
    task_type: str = "arc"
    # Data
    data_paths: List[str]
    data_paths_test: List[str] = []
    # pair_types for each data_path ("both" or "train") - ARC only
    # if not specified, defaults to "both" for all
    pair_types: Optional[List[str]] = None
    # ARC only: repeat factor per path in data_paths (train dataloader only)
    data_path_multiplicities: Optional[List[int]] = None
    # Dataloader worker processes per rank. Keep configurable because large cached
    # ARC tables can otherwise be multiplied by worker-process memory overhead.
    dataloader_num_workers: int = 4
    # fraction of samples placed at origin (0,0) instead of random translation - ARC only
    no_translation_ratio: float = 0.0
    # Number of eval passes over the offline-augmented public-evaluation set.
    eval_passes: int = 1

    # Hyperparams
    global_batch_size: int
    grad_accum_steps: int = 1
    epochs: int  # effective epochs (1 epoch = num_original_puzzles * mean_pairs_per_puzzle samples)

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int
    optimizer: Literal["adamw", "muon"] = "adamw"

    weight_decay: float
    beta1: float
    beta2: float

    # Puzzle embedding
    puzzle_emb_lr: float
    puzzle_emb_weight_decay: float

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    load_checkpoint: Optional[str] = None
    load_optimizer_state: bool = True
    checkpoint_path: Optional[str] = None

    # Extras
    seed: int = 0
    checkpoint_every_eval: bool = False
    eval_interval: Optional[int] = None  # in effective epochs
    # Optional step-based overrides for cross-dataset ablations.
    # Must be specified together (or both unset).
    train_steps_override: Optional[int] = None
    eval_interval_steps_override: Optional[int] = None
    eval_recent_window_size: Optional[int] = None
    # ARC evaluator: optionally apply exponential decay over eval events/checkpoints.
    eval_exp_decay: bool = False
    eval_decay_half_life: float = 10.0

    ema: bool = False
    ema_rate: float = 0.999
    freeze_weights: bool = False

@dataclass
class DatasetMetadata:
    """Metadata matching TRM's PuzzleDatasetMetadata interface."""
    vocab_size: int
    seq_len: int
    puzzle_embed_vocab_size: int
    num_unique_puzzles: int = 0
    per_aug_vocab_sizes: dict = None  # for separated mode: {"dih": 9, "colorperm": 1001}


@dataclass
class TrainState:
    model: nn.Module
    optimizers: Sequence[torch.optim.Optimizer]
    optimizer_lrs: Sequence[float]
    carry: Any

    step: int
    total_steps: int
    accum_step: int = 0
