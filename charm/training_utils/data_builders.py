from typing import Optional, Any
import os

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from charm.datasets import (
    ARCPerPairDataset,
    create_puzzle_dataloader,
)
from charm.transforms import (
    CanvasEOSTransform,
    FlattenTransform,
    VOCAB_SIZE as ARC_VOCAB_SIZE,
    SEQ_LEN as ARC_SEQ_LEN,
)

from charm.training_utils.config_types import PretrainConfig, DatasetMetadata


def create_train_dataloader(
    config: PretrainConfig,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler, DatasetMetadata]:
    """Create the ARC training dataloader used by the CHARM paper."""
    if config.task_type != "arc":
        raise ValueError("The CHARM release supports only task_type='arc'.")

    # Check if using separated mode (requires per-aug tracking)
    use_smart_embed = config.arch.__pydantic_extra__.get("use_smart_embed", False)
    embed_source = config.arch.__pydantic_extra__.get("smart_embed_source", "slotperm")
    track_per_aug = use_smart_embed and embed_source == "separated"

    # Determine pair_types:
    # - If explicitly provided, use it
    # - Otherwise: "train" for files in data_paths_test (reserve test pairs), "both" for others
    if config.pair_types is not None:
        pair_types = config.pair_types
    else:
        test_set = set(config.data_paths_test) if config.data_paths_test else set()
        pair_types = ["train" if p in test_set else "both" for p in config.data_paths]

    train_transforms = [
        CanvasEOSTransform(canvas_size=30, p_translate=1.0 - config.no_translation_ratio),
        FlattenTransform(),
    ]
    dataset = ARCPerPairDataset(
        parquet_paths=config.data_paths,
        pair_types=pair_types,
        path_multiplicities=config.data_path_multiplicities,
        eval_mode=False,
        online_transforms=train_transforms,
        track_per_aug_embeddings=track_per_aug,
    )
    vocab_size = ARC_VOCAB_SIZE
    seq_len = ARC_SEQ_LEN

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )

    local_batch_size = config.global_batch_size // world_size
    dataloader = create_puzzle_dataloader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        drop_last=True,
        set_name="train",
    )

    per_aug_vocab_sizes = None
    if hasattr(dataset, "per_aug_embed_vocab_sizes"):
        per_aug_vocab_sizes = dataset.per_aug_embed_vocab_sizes

    metadata = DatasetMetadata(
        vocab_size=vocab_size,
        seq_len=seq_len,
        puzzle_embed_vocab_size=dataset.puzzle_embed_vocab_size,
        num_unique_puzzles=dataset.num_unique_puzzles,
        per_aug_vocab_sizes=per_aug_vocab_sizes,
    )

    return dataloader, sampler, metadata


def create_eval_dataloader(
    config: PretrainConfig,
    rank: int,
    world_size: int,
    solution_output_path: Optional[str] = None,
    unique_str_to_int: Optional[dict] = None,
    puzzle_id_to_int: Optional[dict] = None,
    track_per_aug_embeddings: bool = False,
    per_aug_param_to_int: Optional[dict] = None,
) -> tuple[DataLoader, DistributedSampler, DatasetMetadata, Any]:
    """Create the ARC public-evaluation dataloader."""
    import json

    if config.task_type != "arc":
        raise ValueError("The CHARM release supports only task_type='arc'.")

    eval_paths = config.data_paths_test if config.data_paths_test else config.data_paths

    eval_transforms = [
        CanvasEOSTransform(canvas_size=30, p_translate=0.0),
        FlattenTransform(),
    ]
    dataset = ARCPerPairDataset(
        parquet_paths=eval_paths,
        eval_mode=True,
        online_transforms=eval_transforms,
        unique_str_to_int=unique_str_to_int,
        puzzle_id_to_int=puzzle_id_to_int,
        track_per_aug_embeddings=track_per_aug_embeddings,
        per_aug_param_to_int=per_aug_param_to_int,
    )
    vocab_size = ARC_VOCAB_SIZE
    seq_len = ARC_SEQ_LEN

    if solution_output_path is not None:
        if rank == 0:
            os.makedirs(os.path.dirname(solution_output_path), exist_ok=True)
            solution = dataset.get_noaug_test_outputs()
            with open(solution_output_path, "w") as f:
                json.dump(solution, f)
        if dist.is_initialized():
            dist.barrier()

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
    )

    local_batch_size = config.global_batch_size // world_size
    dataloader = create_puzzle_dataloader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        drop_last=False,
        set_name="eval",
    )

    metadata = DatasetMetadata(
        vocab_size=vocab_size,
        seq_len=seq_len,
        puzzle_embed_vocab_size=dataset.puzzle_embed_vocab_size,
        num_unique_puzzles=dataset.num_unique_puzzles,
    )

    return dataloader, sampler, metadata, dataset
