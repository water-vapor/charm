from collections.abc import Callable
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from torch.utils.data import Dataset


class BasePuzzleDataset(Dataset):
    """Base class for puzzle datasets."""

    def __init__(
        self,
        parquet_paths: list[str | Path],
        eval_mode: bool = False,
        unique_str_to_int: dict | None = None,
        puzzle_id_to_int: dict | None = None,
    ):
        self.parquet_paths = [str(p) for p in parquet_paths]
        self.eval_mode = eval_mode
        self.pairs = []
        self.unique_str_to_int = dict(unique_str_to_int) if unique_str_to_int else {}
        self._next_id = max(self.unique_str_to_int.values(), default=0) + 1
        self.puzzle_id_to_int = dict(puzzle_id_to_int) if puzzle_id_to_int else {}
        self._puzzle_id_next_id = max(self.puzzle_id_to_int.values(), default=-1) + 1
        self._idx_map = []
        self._load_data()

    def _load_data(self):
        """Subclass implements parsing logic."""
        raise NotImplementedError

    def _add_pair(self, puzzle_id, offline_aug, inp, out):
        """Add a pair with automatic unique_str and puzzle_id tracking."""
        unique_str = f"{puzzle_id}||{offline_aug}"
        if unique_str not in self.unique_str_to_int:
            self.unique_str_to_int[unique_str] = self._next_id
            self._next_id += 1
        if puzzle_id not in self.puzzle_id_to_int:
            self.puzzle_id_to_int[puzzle_id] = self._puzzle_id_next_id
            self._puzzle_id_next_id += 1
        self.pairs.append({
            "puzzle_id": puzzle_id,
            "offline_aug": offline_aug,
            "unique_str": unique_str,
            "input": inp,
            "output": out,
        })
        self._idx_map.append(len(self.pairs) - 1)

    @property
    def puzzle_embed_vocab_size(self):
        """Size of puzzle embedding table (max_id + 1, includes reserved 0)."""
        if not self.unique_str_to_int:
            return 0
        return max(self.unique_str_to_int.values()) + 1

    @property
    def num_unique_puzzles(self):
        """Number of unique puzzle IDs."""
        return len(self.puzzle_id_to_int)

    @property
    def num_original_puzzles(self):
        return len(set(p["puzzle_id"] for p in self.pairs))

    @property
    def num_augmented_puzzles(self):
        return len(set(p["unique_str"] for p in self.pairs))

    @property
    def mean_pairs_per_puzzle(self):
        if self.num_augmented_puzzles == 0:
            return 0.0
        return len(self) / self.num_augmented_puzzles

    def __len__(self):
        return len(self._idx_map)

    def __getitem__(self, idx):
        pair = self.pairs[self._idx_map[idx]]
        result = {
            "input": pair["input"],
            "puzzle_id": pair["puzzle_id"],
            "puzzle_idx": self.puzzle_id_to_int[pair["puzzle_id"]],  # puzzle_id index (for smart embed)
            "puzzle_embed_idx": self.unique_str_to_int[pair["unique_str"]],
            "offline_aug": pair["offline_aug"],
            "online_aug": "",
            "unique_str": pair["unique_str"],
        }
        if pair["output"] is not None:
            result["output"] = pair["output"]
        return result

    def filter(self, filter_fn: Callable[[dict], bool]) -> "BasePuzzleDataset":
        self._idx_map = [i for i, p in enumerate(self.pairs) if filter_fn(p)]
        return self

    def reset_filter(self) -> "BasePuzzleDataset":
        self._idx_map = list(range(len(self.pairs)))
        return self

    def rebuild_id_maps(self):
        """Rebuild unique_str_to_int and puzzle_id_to_int based on current _idx_map."""
        unique_strs_in_use = set(self.pairs[i]["unique_str"] for i in self._idx_map)
        puzzle_ids_in_use = set(self.pairs[i]["puzzle_id"] for i in self._idx_map)

        self.unique_str_to_int = {uid: idx + 1 for idx, uid in enumerate(sorted(unique_strs_in_use))}
        self._next_id = len(self.unique_str_to_int) + 1

        self.puzzle_id_to_int = {pid: idx for idx, pid in enumerate(sorted(puzzle_ids_in_use))}
        self._puzzle_id_next_id = len(self.puzzle_id_to_int)

    def extend_id_maps(self):
        """Add missing entries from current _idx_map without rebuilding."""
        for i in self._idx_map:
            unique_str = self.pairs[i]["unique_str"]
            if unique_str not in self.unique_str_to_int:
                self.unique_str_to_int[unique_str] = self._next_id
                self._next_id += 1
            puzzle_id = self.pairs[i]["puzzle_id"]
            if puzzle_id not in self.puzzle_id_to_int:
                self.puzzle_id_to_int[puzzle_id] = self._puzzle_id_next_id
                self._puzzle_id_next_id += 1
