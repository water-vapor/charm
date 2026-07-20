from functools import partial
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from charm.transforms.utils import parse_aug_str

PAD_TOKEN = 0
IGNORE_LABEL_ID = -100

# Identity colorperm slot order (colors 1-9 mapped to slots 0-8)
IDENTITY_COLORPERM_SLOTS = [0, 1, 2, 3, 4, 5, 6, 7, 8]


def parse_offline_aug_to_tensors(offline_augs: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse offline_aug strings to produce tensors for smart embedding.

    Args:
        offline_augs: list of augmentation strings like "dih|3||colorperm|0312456789"

    Returns:
        dih_idxs: (batch,) int array of dihedral indices (0-7, default 0)
        colorperm_slots: (batch, 9) int array of slot indices (default identity)
    """
    batch_size = len(offline_augs)
    dih_idxs = np.zeros(batch_size, dtype=np.int32)
    colorperm_slots = np.tile(IDENTITY_COLORPERM_SLOTS, (batch_size, 1)).astype(np.int32)

    for i, aug_str in enumerate(offline_augs):
        if not aug_str:
            continue
        for aug_name, aug_param in parse_aug_str(aug_str):
            if aug_name == "dih":
                dih_idxs[i] = int(aug_param)
            elif aug_name == "colorperm":
                # perm_str format: "0XXXXXXXXX" where positions 1-9 indicate color mapping
                # slot_order[j] = int(perm_str[j+1]) - 1 for j in 0..8
                if aug_param and aug_param != "0123456789":
                    colorperm_slots[i] = [int(aug_param[j + 1]) - 1 for j in range(9)]

    return dih_idxs, colorperm_slots


def puzzle_collate_fn(
    batch: list[dict],
    set_name: str = "all",
) -> tuple[str, dict[str, torch.Tensor], int]:
    """
    Collate function for puzzle datasets.

    Input: list of dicts from Dataset.__getitem__ with keys:
        - input: np.ndarray (1D sequence or 2D grid)
        - output: np.ndarray or None (eval mode)
        - offline_aug: str (augmentation string from parquet)
        - online_aug: str (augmentation string from runtime transforms)
        - puzzle_id: str
        - unique_str: str
        - puzzle_embed_idx: int
        - per_aug_embed_idxs: dict[str, int] (optional, when track_per_aug_embeddings=True)

    Output: (set_name, batch_dict, batch_size) tuple
        batch_dict includes per_aug_embed_idxs: dict[str, Tensor] when tracking enabled
    """
    inputs = np.stack([item["input"] for item in batch], axis=0)
    puzzle_embed_idxs = np.array([item["puzzle_embed_idx"] for item in batch], dtype=np.int32)
    puzzle_idxs = np.array([item["puzzle_idx"] for item in batch], dtype=np.int32)
    offline_augs = [item["offline_aug"] for item in batch]
    online_augs = [item["online_aug"] for item in batch]

    # Parse aug strings to tensors for smart embedding (slotperm mode)
    # Combine offline and online: supports both offline mode (dih/colorperm in offline_aug)
    # and online mode (dih/colorperm in online_aug). Parser only extracts dih/colorperm.
    combined_augs = [f"{off}||{on}" if off else on for off, on in zip(offline_augs, online_augs)]
    dih_idxs, colorperm_slots = parse_offline_aug_to_tensors(combined_augs)

    inputs = torch.from_numpy(inputs.astype(np.int32))
    puzzle_embed_idxs = torch.from_numpy(puzzle_embed_idxs)
    puzzle_idxs = torch.from_numpy(puzzle_idxs)
    dih_idxs = torch.from_numpy(dih_idxs)
    colorperm_slots = torch.from_numpy(colorperm_slots)

    batch_dict = {
        "inputs": inputs,
        "puzzle_embed_idxs": puzzle_embed_idxs,
        "puzzle_idxs": puzzle_idxs,  # puzzle_id index tensor (for smart embed)
        "dih_idxs": dih_idxs,  # dihedral indices tensor (for smart embed slotperm)
        "colorperm_slots": colorperm_slots,  # colorperm slot indices (for smart embed slotperm)
        "unique_strs": [item["unique_str"] for item in batch],
        "puzzle_ids": [item["puzzle_id"] for item in batch],
        "offline_aug": offline_augs,
        "online_aug": online_augs,
    }

    # Handle per-aug embedding indices when tracking is enabled (separated mode)
    if "per_aug_embed_idxs" in batch[0]:
        all_aug_types = set()
        for item in batch:
            all_aug_types.update(item["per_aug_embed_idxs"].keys())
        per_aug_tensors = {}
        for aug_type in all_aug_types:
            idxs = [item["per_aug_embed_idxs"].get(aug_type, 0) for item in batch]
            per_aug_tensors[aug_type] = torch.tensor(idxs, dtype=torch.int32)
        batch_dict["per_aug_embed_idxs"] = per_aug_tensors

    if batch[0].get("output") is not None:
        labels = np.stack([item["output"] for item in batch], axis=0)
        labels = torch.from_numpy(labels.astype(np.int32))
        labels = torch.where(labels == PAD_TOKEN, IGNORE_LABEL_ID, labels)
        batch_dict["labels"] = labels

    return set_name, batch_dict, len(batch)


def create_puzzle_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    set_name: str = "all",
    sampler=None,
) -> DataLoader:
    collate_fn = partial(puzzle_collate_fn, set_name=set_name)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        sampler=sampler,
        num_workers=num_workers,
        drop_last=drop_last,
        collate_fn=collate_fn,
    )
