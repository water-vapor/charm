"""Donor id mapping, the new-puzzle stream, and evaluation batching.

The donor mapping is rebuilt from the same parquets the checkpoint was trained on
(this reproduces its embedding-table row order exactly; verified against the
checkpoints' table sizes). The stream parquet is registered on top, so stream
puzzles get fresh semantic/instance ids appended after the donor's. Puzzles the
donor has already seen are excluded from the stream.
"""

import os

import numpy as np

from charm.datasets import ARCPerPairDataset
from charm.datasets.arc_per_pair_dataset import _mapping_fingerprint
from charm.datasets.puzzle_dataloader import puzzle_collate_fn
from charm.transforms import CanvasEOSTransform, FlattenTransform


def train_transforms(no_translation_ratio: float):
    return [CanvasEOSTransform(canvas_size=30, p_translate=1.0 - no_translation_ratio),
            FlattenTransform()]


def eval_transforms():
    return [CanvasEOSTransform(canvas_size=30, p_translate=0.0), FlattenTransform()]


def pair_indices_by_puzzle(ds: ARCPerPairDataset, max_augs: int | None = None) -> dict[int, np.ndarray]:
    """puzzle_idx -> raw pair indices, keeping the first max_augs offline augs (identity first)."""
    table = ds.pairs
    row_puzzle = np.asarray(table.unique_puzzle_idx)[np.asarray(table.row_unique_idx)]
    result = {}
    for p in np.unique(row_puzzle):
        rows = np.nonzero(row_puzzle == p)[0]
        if max_augs is not None:
            rows = rows[:max_augs]
        chunks = [np.arange(table.row_offsets[r], table.row_offsets[r] + table.row_total_count[r])
                  for r in rows]
        result[int(p)] = np.concatenate(chunks) if chunks else np.empty(0, dtype=np.int64)
    return result


class Stream:
    def __init__(self, donor_cfg: dict, data_dir: str, stream_parquet: str,
                 no_translation_ratio: float, eval_augs: int):
        test_names = {os.path.basename(p) for p in donor_cfg.get("data_paths_test") or []}
        self.donor_paths = [os.path.join(data_dir, os.path.basename(p))
                            for p in donor_cfg["data_paths"]]
        self.donor_pair_types = donor_cfg.get("pair_types") or [
            "train" if os.path.basename(p) in test_names else "both" for p in self.donor_paths]
        self.donor_multiplicities = donor_cfg.get("data_path_multiplicities")
        donor = ARCPerPairDataset(self.donor_paths, pair_types=self.donor_pair_types,
                                  eval_mode=False, online_transforms=None)
        self.donor_vocab = donor.puzzle_embed_vocab_size
        self.donor_puzzles = donor.num_unique_puzzles

        self.train_set = ARCPerPairDataset(
            [stream_parquet], pair_types=["train"], eval_mode=False,
            online_transforms=train_transforms(no_translation_ratio),
            unique_str_to_int=donor.unique_str_to_int, puzzle_id_to_int=donor.puzzle_id_to_int)
        self.eval_set = ARCPerPairDataset(
            [stream_parquet], eval_mode=True, online_transforms=eval_transforms(),
            unique_str_to_int=self.train_set.unique_str_to_int,
            puzzle_id_to_int=self.train_set.puzzle_id_to_int)
        assert len(self.train_set._idx_map) == len(self.train_set.pairs)
        assert len(self.eval_set._idx_map) == len(self.eval_set.pairs)
        self.vocab = max(self.train_set.puzzle_embed_vocab_size,
                         self.eval_set.puzzle_embed_vocab_size)
        self.num_puzzles = max(self.train_set.num_unique_puzzles,
                               self.eval_set.num_unique_puzzles)

        by_idx = sorted(self.train_set.puzzle_id_to_int.items(), key=lambda kv: kv[1])
        self.puzzle_ids = [pid for pid, idx in by_idx if idx >= self.donor_puzzles]
        self.train_pairs = pair_indices_by_puzzle(self.train_set)
        self.eval_pairs = pair_indices_by_puzzle(self.eval_set, max_augs=eval_augs)
        self.gt_inputs = self.eval_set.get_noaug_test_inputs()
        self.gt_outputs = self.eval_set.get_noaug_test_outputs()
        # audit trail: donor-mapping digests (must be stable across every run of a campaign)
        self.fingerprints = {
            "instance": _mapping_fingerprint(donor.unique_str_to_int)["sha256"],
            "semantic": _mapping_fingerprint(donor.puzzle_id_to_int)["sha256"]}
        # instance-id span per puzzle (registration is contiguous per parquet row)
        table = self.train_set.pairs
        row_puzzle = np.asarray(table.unique_puzzle_idx)[np.asarray(table.row_unique_idx)]
        row_unique = np.asarray(table.row_unique_idx)
        self.instance_ranges = {
            int(p): (int(row_unique[row_puzzle == p].min()), int(row_unique[row_puzzle == p].max()) + 1)
            for p in np.unique(row_puzzle)}

    def identity_view(self) -> ARCPerPairDataset:
        """Raw pair index == dataset index; joint batches index puzzles directly."""
        self.train_set._idx_map = np.arange(len(self.train_set.pairs))
        return self.train_set

    def order(self, order_seed: int, n_puzzles: int) -> list[str]:
        order = list(self.puzzle_ids)
        np.random.RandomState(order_seed).shuffle(order)
        return order[:n_puzzles] if n_puzzles > 0 else order

    def semantic_idx(self, puzzle_id: str) -> int:
        return self.train_set.puzzle_id_to_int[puzzle_id]

    def train_view(self, puzzle_ids: list[str]) -> ARCPerPairDataset:
        """Restrict the train set to the given puzzles (single puzzle, or all for joint arms)."""
        idxs = np.concatenate([self.train_pairs[self.semantic_idx(p)] for p in puzzle_ids])
        self.train_set._idx_map = idxs
        return self.train_set

    def eval_pair_idxs(self, puzzle_ids: list[str]) -> np.ndarray:
        return np.concatenate([self.eval_pairs[self.semantic_idx(p)] for p in puzzle_ids])


def arc1_eval_set(data_dir: str, stream: Stream, n_puzzles: int, max_augs: int):
    """ARC1 eval-split dataset under the donor mapping, for the retention probe."""
    ds = ARCPerPairDataset(
        [os.path.join(data_dir, "arc1_eval_aug1000.parquet")], eval_mode=True,
        online_transforms=eval_transforms(),
        unique_str_to_int=stream.train_set.unique_str_to_int,
        puzzle_id_to_int=stream.train_set.puzzle_id_to_int)
    assert len(ds._idx_map) == len(ds.pairs)
    pairs = pair_indices_by_puzzle(ds, max_augs=max_augs)
    selected = set(sorted(pairs.keys())[:n_puzzles])
    puzzle_ids = sorted(pid for pid, idx in ds.puzzle_id_to_int.items() if idx in selected)
    pair_idxs = np.concatenate([pairs[i] for i in sorted(selected)])
    return ds, pair_idxs, puzzle_ids


def eval_batches(ds: ARCPerPairDataset, pair_idxs: np.ndarray, batch_size: int):
    """Yield (batch_dict, n_real); the last batch is padded to a fixed shape."""
    n = len(pair_idxs)
    for start in range(0, n, batch_size):
        idxs = [int(pair_idxs[min(start + i, n - 1)]) for i in range(batch_size)]
        _, batch, _ = puzzle_collate_fn([ds[i] for i in idxs], set_name="eval")
        yield batch, min(batch_size, n - start)
