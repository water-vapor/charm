"""CAR-specific dataset metadata and D4 evaluation loaders."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from charm.car.metadata import load_car_id_maps, parse_car_puzzle_id
from charm.datasets.arc_per_pair_dataset import ARCPerPairDataset
from charm.datasets.puzzle_dataloader import puzzle_collate_fn
from charm.training_utils.config_types import DatasetMetadata
from charm.transforms import (
    CanvasEOSTransform,
    DihedralTransform,
    FlattenTransform,
    VOCAB_SIZE as ARC_VOCAB_SIZE,
    SEQ_LEN as ARC_SEQ_LEN,
    apply_chain_paired,
)


class CARPairDataset(ARCPerPairDataset):
    """Attach rule, horizon, and observed-task indices to ARC-v2 pairs."""

    def __init__(
        self,
        *args,
        car_task_id_to_int: dict[str, int],
        car_rule_id_to_int: dict[str, int],
        **kwargs,
    ) -> None:
        self.car_task_id_to_int = dict(car_task_id_to_int)
        self.car_rule_id_to_int = dict(car_rule_id_to_int)
        super().__init__(*args, **kwargs)

    def _car_metadata(self, puzzle_id: str) -> dict[str, int]:
        factors = parse_car_puzzle_id(puzzle_id)
        if factors is None:
            raise ValueError(f"invalid CAR puzzle identifier: {puzzle_id!r}")
        return {
            "car_task_idx": self.car_task_id_to_int.get(puzzle_id, -1),
            "car_rule_idx": self.car_rule_id_to_int[factors.rule_id],
            "car_horizon_idx": factors.horizon_idx,
        }

    def __getitem__(self, index: int) -> dict:
        result = super().__getitem__(index)
        result.update(self._car_metadata(result["puzzle_id"]))
        return result

    def get_raw_pair(self, index: int) -> dict:
        result = super().get_raw_pair(index)
        result.update(self._car_metadata(result["puzzle_id"]))
        return result


class ExhaustiveDihedralDataset(Dataset):
    """Evaluate each base pair once under every element of D4."""

    def __init__(self, base_dataset: Dataset):
        self.base_dataset = base_dataset
        self.transforms = list(base_dataset.online_transforms)

    def __len__(self) -> int:
        return 8 * len(self.base_dataset)

    def __getattr__(self, name):
        if name == "base_dataset":
            raise AttributeError(name)
        return getattr(self.base_dataset, name)

    def __getitem__(self, index: int) -> dict:
        pair_index, dihedral_index = divmod(int(index), 8)
        raw = self.base_dataset.get_raw_pair(pair_index)
        transforms = [
            DihedralTransform(p=1.0, fixed_id=dihedral_index),
            *self.transforms[1:],
        ]
        inp, out, online_aug = apply_chain_paired(
            raw["input"],
            raw["output"],
            transforms,
            np.random.RandomState(pair_index),
        )
        result = {
            **raw,
            "input": inp,
            "online_aug": online_aug,
        }
        if out is not None:
            result["output"] = out
        return result


def car_collate_fn(
    batch: list[dict],
    set_name: str = "all",
) -> tuple[str, dict[str, Any], int]:
    """Extend CHARM's standard ARC batch with CAR factor tensors."""

    name, batch_dict, batch_size = puzzle_collate_fn(
        batch,
        set_name=set_name,
    )
    required = ("car_task_idx", "car_rule_idx", "car_horizon_idx")
    missing = [key for key in required if key not in batch[0]]
    if missing:
        raise KeyError(f"CAR batch is missing metadata: {', '.join(missing)}")
    batch_dict.update(
        {
            "car_task_idxs": torch.tensor(
                [item["car_task_idx"] for item in batch],
                dtype=torch.int32,
            ),
            "car_rule_idxs": torch.tensor(
                [item["car_rule_idx"] for item in batch],
                dtype=torch.int32,
            ),
            "car_horizon_idxs": torch.tensor(
                [item["car_horizon_idx"] for item in batch],
                dtype=torch.int32,
            ),
        }
    )
    return name, batch_dict, batch_size


def create_car_dataloader(
    dataset: Dataset,
    *,
    batch_size: int,
    sampler: DistributedSampler,
    num_workers: int,
    drop_last: bool,
    set_name: str,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=partial(car_collate_fn, set_name=set_name),
    )


@dataclass
class CARDatasetMetadata(DatasetMetadata):
    car_num_tasks: int = 0
    car_num_rules: int = 0


def _local_batch_size(global_batch_size: int, world_size: int) -> int:
    if global_batch_size % world_size:
        raise ValueError(
            f"global batch {global_batch_size} must be divisible by "
            f"world size {world_size}"
        )
    return global_batch_size // world_size


def create_car_train_dataloader(
    config: Any,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler, CARDatasetMetadata]:
    task_ids, rule_ids = load_car_id_maps(config.data_paths)
    if config.pair_types is not None:
        pair_types = config.pair_types
    else:
        test_paths = set(config.data_paths_test or [])
        pair_types = [
            "train" if path in test_paths else "both" for path in config.data_paths
        ]

    dataset = CARPairDataset(
        parquet_paths=config.data_paths,
        pair_types=pair_types,
        path_multiplicities=config.data_path_multiplicities,
        eval_mode=False,
        online_transforms=[
            DihedralTransform(p=1.0),
            CanvasEOSTransform(
                canvas_size=30,
                p_translate=1.0 - config.no_translation_ratio,
            ),
            FlattenTransform(),
        ],
        car_task_id_to_int=task_ids,
        car_rule_id_to_int=rule_ids,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=config.seed,
    )
    dataloader = create_car_dataloader(
        dataset,
        batch_size=_local_batch_size(config.global_batch_size, world_size),
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        drop_last=True,
        set_name="train",
    )
    metadata = CARDatasetMetadata(
        vocab_size=ARC_VOCAB_SIZE,
        seq_len=ARC_SEQ_LEN,
        puzzle_embed_vocab_size=dataset.puzzle_embed_vocab_size,
        num_unique_puzzles=dataset.num_unique_puzzles,
        car_num_tasks=len(task_ids),
        car_num_rules=len(rule_ids),
    )
    return dataloader, sampler, metadata


def create_car_eval_dataloader(
    config: Any,
    train_dataset: CARPairDataset,
    rank: int,
    world_size: int,
) -> tuple[DataLoader, DistributedSampler, ExhaustiveDihedralDataset]:
    eval_paths = config.data_paths_test or config.data_paths
    base_dataset = CARPairDataset(
        parquet_paths=eval_paths,
        eval_mode=True,
        online_transforms=[
            DihedralTransform(p=1.0),
            CanvasEOSTransform(canvas_size=30, p_translate=0.0),
            FlattenTransform(),
        ],
        unique_str_to_int=train_dataset.unique_str_to_int,
        puzzle_id_to_int=train_dataset.puzzle_id_to_int,
        car_task_id_to_int=train_dataset.car_task_id_to_int,
        car_rule_id_to_int=train_dataset.car_rule_id_to_int,
    )
    dataset = ExhaustiveDihedralDataset(base_dataset)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
        seed=config.seed,
    )
    dataloader = create_car_dataloader(
        dataset,
        batch_size=_local_batch_size(config.global_batch_size, world_size),
        sampler=sampler,
        num_workers=config.dataloader_num_workers,
        drop_last=False,
        set_name="eval",
    )
    return dataloader, sampler, dataset
