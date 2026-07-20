"""
ARC Per-Pair Dataset with composable transforms.

Each sample is a single input-output pair from an ARC puzzle.
Supports both offline augmentations (from parquet) and online augmentations (runtime transforms).
"""

from pathlib import Path
from array import array
import json
import hashlib
import tempfile
import numpy as np
import pyarrow.parquet as pq
from tqdm.auto import tqdm

from charm.preprocessing.arc import ARC_AUGMENTED_FORMAT_VERSION
from charm.utils.distributed import get_rank, print0
from charm.transforms import Transform, apply_chain_paired, apply_with_aug_str, parse_aug_str

from .base import BasePuzzleDataset
from .cache import ARCGeneratedCacheManager


def _smallest_uint_dtype(max_value: int) -> np.dtype:
    if max_value <= np.iinfo(np.uint8).max:
        return np.dtype(np.uint8)
    if max_value <= np.iinfo(np.uint16).max:
        return np.dtype(np.uint16)
    if max_value <= np.iinfo(np.uint32).max:
        return np.dtype(np.uint32)
    return np.dtype(np.uint64)


def _mapping_fingerprint(mapping: dict | None) -> dict[str, object]:
    if not mapping:
        return {"len": 0, "sha256": ""}
    h = hashlib.sha256()
    for key, value in sorted(mapping.items()):
        h.update(str(key).encode("utf-8"))
        h.update(b"\0")
        h.update(str(int(value)).encode("ascii"))
        h.update(b"\n")
    return {"len": len(mapping), "sha256": h.hexdigest()[:16]}


def _nested_mapping_fingerprint(mapping: dict[str, dict[str, int]] | None) -> dict[str, object]:
    if not mapping:
        return {"len": 0, "sha256": ""}
    h = hashlib.sha256()
    entry_count = 0
    for outer_key, inner in sorted(mapping.items()):
        h.update(str(outer_key).encode("utf-8"))
        h.update(b"\0")
        for inner_key, value in sorted((inner or {}).items()):
            entry_count += 1
            h.update(str(inner_key).encode("utf-8"))
            h.update(b"\0")
            h.update(str(int(value)).encode("ascii"))
            h.update(b"\n")
        h.update(b"\xff")
    return {"len": len(mapping), "entries": entry_count, "sha256": h.hexdigest()[:16]}


class GeneratedPairTable:
    """Sequence adapter that lazily generates offline-augmented grids from base grids."""

    def __init__(
        self,
        *,
        rows: dict[str, np.ndarray],
        base: dict[str, np.ndarray],
        input_flat: np.ndarray,
        output_flat: np.ndarray,
        unique_strs: list[str],
        puzzle_ids: list[str],
        offline_aug_strings: list[str],
        unique_puzzle_idx: np.ndarray,
        unique_offline_aug_idx: np.ndarray,
        per_aug_param_to_int: dict[str, dict[str, int]] | None = None,
        eval_mode: bool = False,
    ):
        self.row_unique_idx = rows["row_unique_idx"]
        self.row_base_idx = rows["row_base_idx"]
        self.row_train_count = rows["row_train_count"]
        self.row_test_count = rows["row_test_count"]
        self.row_total_count = (
            self.row_train_count.astype(np.int64)
            + self.row_test_count.astype(np.int64)
        )
        self.row_offsets = self._compute_offsets(self.row_total_count)
        self.total_pairs = int(self.row_total_count.sum(dtype=np.int64))

        self.train_input_shape_hw = base["train_input_shape_hw"]
        self.train_output_shape_hw = base["train_output_shape_hw"]
        self.train_original_pair_index = base["train_original_pair_index"]
        self.test_input_shape_hw = base["test_input_shape_hw"]
        self.test_output_shape_hw = base["test_output_shape_hw"]
        self.test_output_present = base["test_output_present"].astype(np.bool_, copy=False)
        self.test_original_pair_index = base["test_original_pair_index"]
        self.train_pair_start = base["train_pair_start"]
        self.test_pair_start = base["test_pair_start"]

        self.input_flat = input_flat
        self.output_flat = output_flat
        self.unique_strs = unique_strs
        self.puzzle_ids = puzzle_ids
        self.offline_aug_strings = offline_aug_strings
        self.unique_puzzle_idx = unique_puzzle_idx
        self.unique_offline_aug_idx = unique_offline_aug_idx
        self.per_aug_param_to_int = per_aug_param_to_int or {}
        self.eval_mode = eval_mode

        self.train_input_offsets = self._compute_offsets(
            self.train_input_shape_hw[:, 0].astype(np.int64)
            * self.train_input_shape_hw[:, 1].astype(np.int64)
        )
        train_input_cells = int(self.train_input_offsets[-1]) + self._last_size(
            self.train_input_shape_hw
        ) if len(self.train_input_shape_hw) else 0
        self.test_input_offsets = train_input_cells + self._compute_offsets(
            self.test_input_shape_hw[:, 0].astype(np.int64)
            * self.test_input_shape_hw[:, 1].astype(np.int64)
        )

        self.train_output_offsets = self._compute_offsets(
            self.train_output_shape_hw[:, 0].astype(np.int64)
            * self.train_output_shape_hw[:, 1].astype(np.int64)
        )
        train_output_cells = int(self.train_output_offsets[-1]) + self._last_size(
            self.train_output_shape_hw
        ) if len(self.train_output_shape_hw) else 0
        test_output_sizes = (
            self.test_output_shape_hw[:, 0].astype(np.int64)
            * self.test_output_shape_hw[:, 1].astype(np.int64)
        )
        test_output_sizes = np.where(self.test_output_present, test_output_sizes, 0)
        self.test_output_offsets = train_output_cells + self._compute_offsets(test_output_sizes)

    @staticmethod
    def _compute_offsets(sizes: np.ndarray) -> np.ndarray:
        offsets = np.empty(len(sizes), dtype=np.int64)
        if len(sizes):
            offsets[0] = 0
            if len(sizes) > 1:
                offsets[1:] = np.cumsum(sizes[:-1], dtype=np.int64)
        return offsets

    @staticmethod
    def _last_size(shapes: np.ndarray) -> int:
        if not len(shapes):
            return 0
        return int(shapes[-1, 0]) * int(shapes[-1, 1])

    def __len__(self) -> int:
        return self.total_pairs

    def _row_for_pair(self, pair_idx: int) -> tuple[int, int]:
        pair_idx = int(pair_idx)
        if pair_idx < 0 or pair_idx >= self.total_pairs:
            raise IndexError(pair_idx)
        row_idx = int(np.searchsorted(self.row_offsets, pair_idx, side="right") - 1)
        return row_idx, pair_idx - int(self.row_offsets[row_idx])

    def _strings_for_row(self, row_idx: int) -> tuple[int, int, str, str, str]:
        unique_idx = int(self.row_unique_idx[row_idx])
        puzzle_idx = int(self.unique_puzzle_idx[unique_idx])
        offline_aug_idx = int(self.unique_offline_aug_idx[unique_idx])
        return (
            unique_idx,
            puzzle_idx,
            self.puzzle_ids[puzzle_idx],
            self.offline_aug_strings[offline_aug_idx],
            self.unique_strs[unique_idx],
        )

    def _pair_location(self, pair_idx: int) -> tuple[int, int, int, str, int]:
        row_idx, local_idx = self._row_for_pair(pair_idx)
        unique_idx, puzzle_idx, _, _, _ = self._strings_for_row(row_idx)
        base_idx = int(self.row_base_idx[row_idx])
        train_count = int(self.row_train_count[row_idx])
        if local_idx < train_count:
            base_pair_idx = int(self.train_pair_start[base_idx]) + local_idx
            original_pair_index = int(self.train_original_pair_index[base_pair_idx])
            return row_idx, unique_idx, base_pair_idx, "train", original_pair_index

        test_local_idx = local_idx - train_count
        base_pair_idx = int(self.test_pair_start[base_idx]) + test_local_idx
        original_pair_index = int(self.test_original_pair_index[base_pair_idx])
        return row_idx, unique_idx, base_pair_idx, "test", original_pair_index

    def get_per_aug_embed_idxs(self, offline_aug: str) -> dict[str, int]:
        if not self.per_aug_param_to_int or not offline_aug:
            return {}
        result = {}
        for aug_name, aug_param in parse_aug_str(offline_aug):
            value = self.per_aug_param_to_int.get(aug_name, {}).get(aug_param, 0)
            if value:
                result[aug_name] = value
        return result

    def get_metadata(self, pair_idx: int) -> dict:
        row_idx, _, _, pair_type, original_pair_index = self._pair_location(pair_idx)
        _, _, puzzle_id, offline_aug, unique_str = self._strings_for_row(row_idx)
        return {
            "puzzle_id": puzzle_id,
            "offline_aug": offline_aug,
            "unique_str": unique_str,
            "pair_type": pair_type,
            "pair_index": original_pair_index,
            "per_aug_embed_idxs": self.get_per_aug_embed_idxs(offline_aug),
        }

    def _base_input(self, base_pair_idx: int, pair_type: str) -> np.ndarray:
        if pair_type == "train":
            shape = self.train_input_shape_hw[base_pair_idx]
            offset = int(self.train_input_offsets[base_pair_idx])
        else:
            shape = self.test_input_shape_hw[base_pair_idx]
            offset = int(self.test_input_offsets[base_pair_idx])
        size = int(shape[0]) * int(shape[1])
        return self.input_flat[offset:offset + size].reshape(int(shape[0]), int(shape[1]))

    def _base_output(self, base_pair_idx: int, pair_type: str) -> np.ndarray | None:
        if self.eval_mode:
            return None
        if pair_type == "train":
            shape = self.train_output_shape_hw[base_pair_idx]
            offset = int(self.train_output_offsets[base_pair_idx])
        else:
            if not bool(self.test_output_present[base_pair_idx]):
                return None
            shape = self.test_output_shape_hw[base_pair_idx]
            offset = int(self.test_output_offsets[base_pair_idx])
        size = int(shape[0]) * int(shape[1])
        return self.output_flat[offset:offset + size].reshape(int(shape[0]), int(shape[1]))

    def get_arrays(self, pair_idx: int, copy: bool = False):
        row_idx, _, base_pair_idx, pair_type, _ = self._pair_location(pair_idx)
        _, _, _, offline_aug, _ = self._strings_for_row(row_idx)

        inp = self._base_input(base_pair_idx, pair_type)
        out = self._base_output(base_pair_idx, pair_type)
        if offline_aug:
            inp = apply_with_aug_str(inp, offline_aug)
            if out is not None:
                out = apply_with_aug_str(out, offline_aug)

        inp = np.asarray(inp, dtype=np.uint8)
        if out is not None:
            out = np.asarray(out, dtype=np.uint8)
        if copy:
            inp = inp.copy()
            if out is not None:
                out = out.copy()
        return inp, out

    def __getitem__(self, pair_idx: int) -> dict:
        pair_idx = int(pair_idx)
        result = self.get_metadata(pair_idx)
        inp, out = self.get_arrays(pair_idx)
        result["input"] = inp
        result["output"] = out
        return result


class ARCPerPairDataset(BasePuzzleDataset):
    """
    PyTorch Dataset for ARC puzzles at the per-pair level.

    Each sample is a single input-output pair from a puzzle.

    Two modes:
        - Train mode (eval_mode=False): reads pairs according to pair_types
        - Eval mode (eval_mode=True): only reads "test" pair inputs, no labels
    """

    def __init__(
        self,
        parquet_paths: list[str | Path],
        pair_types: list[str] | None = None,
        path_multiplicities: list[int] | None = None,
        eval_mode: bool = False,
        online_transforms: list[Transform] | None = None,
        unique_str_to_int: dict | None = None,
        puzzle_id_to_int: dict[str, int] | None = None,
        cache_dir: str | Path | bool | None = None,
        deterministic_online_aug: bool = False,
        max_grid_size: int = 30,
        track_per_aug_embeddings: bool = False,
        per_aug_param_to_int: dict[str, dict[str, int]] | None = None,
    ):
        """
        Args:
            parquet_paths: List of paths to augmented parquet files
            pair_types: (train mode only) list matching parquet_paths length
                - "both" = use both "train" and "test" pairs from that parquet
                - "train" = use only "train" pairs (test pairs reserved for eval)
            path_multiplicities: (train mode only) list matching parquet_paths length.
                Repeats each path's loaded pairs this many times in epoch sampling.
                If None, uses 1 for all paths.
            eval_mode: If True, only read "test" pair inputs, no labels
            online_transforms: List of transforms to apply at runtime
            unique_str_to_int: Existing mapping to reuse (new keys added, existing preserved)
            cache_dir: Directory for caching (None=default, False=disable)
            deterministic_online_aug: If True, seed RNG with sample index for reproducible
                augmentation. If False (default), truly random augmentation each call.
            max_grid_size: Maximum grid dimension (pairs with larger grids are dropped)
            track_per_aug_embeddings: If True, build per-augmentation-type embedding indices
            per_aug_param_to_int: Existing per-aug mappings to reuse (for eval dataset sharing)
        """
        # ARC-specific attributes (must be set before super().__init__ calls _load_data)
        self.max_grid_size = max_grid_size
        self.online_transforms = online_transforms or []
        self.deterministic_online_aug = deterministic_online_aug
        self._noaug_test_data = {}
        self._path_pair_ranges: list[tuple[int, int]] = []
        self._cache_tempdir = tempfile.TemporaryDirectory(prefix="arc_pair_cache_") if cache_dir is False else None
        cache_root = self._cache_tempdir.name if self._cache_tempdir is not None else cache_dir
        self.generated_cache_manager = ARCGeneratedCacheManager(cache_root)
        # Key used for this instance's cache lookup/save during initial load.
        self._cache_key_used: str | None = None
        self._num_original_puzzles_override: int | None = None
        self._num_augmented_puzzles_override: int | None = None

        # Per-augmentation embedding tracking (optional)
        self.track_per_aug_embeddings = track_per_aug_embeddings
        if track_per_aug_embeddings:
            self._per_aug_param_to_int: dict[str, dict[str, int]] = (
                {k: dict(v) for k, v in per_aug_param_to_int.items()}
                if per_aug_param_to_int else {}
            )
            self._per_aug_next_id: dict[str, int] = {
                k: max(v.values(), default=0) + 1
                for k, v in self._per_aug_param_to_int.items()
            }
        else:
            self._per_aug_param_to_int = {}
            self._per_aug_next_id = {}

        if eval_mode:
            self.pair_types = None
            if path_multiplicities is not None:
                assert len(path_multiplicities) == len(parquet_paths), \
                    f"path_multiplicities length ({len(path_multiplicities)}) must match parquet_paths length ({len(parquet_paths)})"
                assert all(isinstance(m, int) and m == 1 for m in path_multiplicities), \
                    "path_multiplicities must be all 1 in eval mode"
            self.path_multiplicities = [1] * len(parquet_paths)
        else:
            assert pair_types is not None, "pair_types required for train mode"
            assert len(pair_types) == len(parquet_paths), \
                f"pair_types length ({len(pair_types)}) must match parquet_paths length ({len(parquet_paths)})"
            assert all(pt in ("both", "train") for pt in pair_types), \
                "pair_types must be 'both' or 'train'"
            if path_multiplicities is None:
                path_multiplicities = [1] * len(parquet_paths)
            assert len(path_multiplicities) == len(parquet_paths), \
                f"path_multiplicities length ({len(path_multiplicities)}) must match parquet_paths length ({len(parquet_paths)})"
            assert all(isinstance(m, int) and m >= 1 for m in path_multiplicities), \
                "path_multiplicities must contain integers >= 1"
            self.pair_types = pair_types
            self.path_multiplicities = list(path_multiplicities)

        super().__init__(parquet_paths, eval_mode, unique_str_to_int, puzzle_id_to_int)

    def _load_data(self):
        """Load the v5 generated-grid cache, building it from parquet if needed."""
        cache_key = self._compute_cache_key()
        self._cache_key_used = cache_key
        if self._try_load_v5_cache(cache_key):
            return
        with self.generated_cache_manager.lock(cache_key):
            if not self.generated_cache_manager.exists(cache_key):
                print0("Generated ARC dataset not cached, building v5 cache from parquet...")
                self._build_v5_cache_from_parquet(cache_key)
        if self._try_load_v5_cache(cache_key):
            return
        raise RuntimeError(f"Failed to load v5 cache after building: {cache_key}")

    def _grid_fits(self, grid) -> bool:
        """Check if grid fits within max_grid_size."""
        h, w = len(grid), len(grid[0]) if grid else 0
        return h <= self.max_grid_size and w <= self.max_grid_size

    @property
    def per_aug_embed_vocab_sizes(self) -> dict[str, int]:
        """Get vocab size per aug type for network init. Index 0 reserved for each."""
        if not self.track_per_aug_embeddings:
            return {}
        return {
            aug_name: max(param_to_int.values(), default=0) + 1
            for aug_name, param_to_int in self._per_aug_param_to_int.items()
        }

    def _compute_cache_key(self):
        """Compute cache key from configuration."""
        parquet_files = []
        for path_str in self.parquet_paths:
            path = Path(path_str)
            try:
                size = path.stat().st_size
            except OSError:
                # Keep key computation robust; loading will still fail later if unreadable.
                size = None
            parquet_files.append({
                "name": path.name,
                "size": size,
            })

        config = {
            "cache_key_version": 5,
            "source_format_version": ARC_AUGMENTED_FORMAT_VERSION,
            "base_scope": "path_puzzle",
            "parquet_files": parquet_files,
            "pair_types": self.pair_types,
            "eval_mode": self.eval_mode,
            "unique_str_to_int_fingerprint": _mapping_fingerprint(self.unique_str_to_int),
            "puzzle_id_to_int_fingerprint": _mapping_fingerprint(self.puzzle_id_to_int),
            "max_grid_size": self.max_grid_size,
            "track_per_aug_embeddings": self.track_per_aug_embeddings,
        }
        if self.track_per_aug_embeddings and self._per_aug_param_to_int:
            config["per_aug_param_to_int_fingerprint"] = _nested_mapping_fingerprint(
                self._per_aug_param_to_int
            )
        return self.generated_cache_manager.compute_key(config)

    @property
    def cache_key_used(self) -> str | None:
        """Cache key used for disk load/save by this dataset instance."""
        return self._cache_key_used

    def _register_pair_ids(self, puzzle_id: str, offline_aug: str) -> tuple[str, int, int]:
        unique_str = f"{puzzle_id}||{offline_aug}"
        if unique_str not in self.unique_str_to_int:
            self.unique_str_to_int[unique_str] = self._next_id
            self._next_id += 1
        if puzzle_id not in self.puzzle_id_to_int:
            self.puzzle_id_to_int[puzzle_id] = self._puzzle_id_next_id
            self._puzzle_id_next_id += 1
        return unique_str, self.unique_str_to_int[unique_str], self.puzzle_id_to_int[puzzle_id]

    def _register_per_aug_ids(self, offline_aug: str) -> dict[str, int]:
        per_aug_embed_idxs = {}
        if not self.track_per_aug_embeddings or not offline_aug:
            return per_aug_embed_idxs
        for aug_name, aug_param in parse_aug_str(offline_aug):
            if aug_name not in self._per_aug_param_to_int:
                self._per_aug_param_to_int[aug_name] = {}
                self._per_aug_next_id[aug_name] = 1  # 0 reserved
            if aug_param not in self._per_aug_param_to_int[aug_name]:
                self._per_aug_param_to_int[aug_name][aug_param] = self._per_aug_next_id[aug_name]
                self._per_aug_next_id[aug_name] += 1
            per_aug_embed_idxs[aug_name] = self._per_aug_param_to_int[aug_name][aug_param]
        return per_aug_embed_idxs

    def _build_unique_tables(self) -> tuple[list[str], list[str], np.ndarray, np.ndarray, list[str]]:
        max_unique_idx = max(self.unique_str_to_int.values(), default=0)
        unique_strs = [""] * (max_unique_idx + 1)
        unique_puzzle_idx = np.zeros(max_unique_idx + 1, dtype=_smallest_uint_dtype(max(self.puzzle_id_to_int.values(), default=0)))
        offline_aug_to_idx = {"": 0}
        offline_aug_strings = [""]
        unique_offline_aug_idx_tmp = np.zeros(max_unique_idx + 1, dtype=np.uint32)

        for unique_str, unique_idx in self.unique_str_to_int.items():
            puzzle_id, offline_aug = unique_str.split("||", 1)
            unique_strs[unique_idx] = unique_str
            unique_puzzle_idx[unique_idx] = self.puzzle_id_to_int[puzzle_id]
            if offline_aug not in offline_aug_to_idx:
                offline_aug_to_idx[offline_aug] = len(offline_aug_strings)
                offline_aug_strings.append(offline_aug)
            unique_offline_aug_idx_tmp[unique_idx] = offline_aug_to_idx[offline_aug]

        unique_offline_aug_idx = unique_offline_aug_idx_tmp.astype(
            _smallest_uint_dtype(len(offline_aug_strings) - 1),
            copy=False,
        )
        puzzle_ids = [""] * (max(self.puzzle_id_to_int.values(), default=-1) + 1)
        for puzzle_id, puzzle_idx in self.puzzle_id_to_int.items():
            puzzle_ids[puzzle_idx] = puzzle_id
        return unique_strs, puzzle_ids, unique_puzzle_idx, unique_offline_aug_idx, offline_aug_strings

    def _iter_parquet_batches(self, parquet_file: pq.ParquetFile):
        return parquet_file.iter_batches(
            columns=["format_version", "puzzle_id", "train", "test", "offline_augs"],
            # REARC v2 rows carry large base train lists. Keep batches modest so
            # as_py() conversion never spikes memory.
            batch_size=64,
            use_threads=True,
        )

    @staticmethod
    def _validate_v2_parquet(parquet_file: pq.ParquetFile, path: str | Path) -> None:
        required = {"format_version", "puzzle_id", "train", "test", "offline_augs"}
        present = set(parquet_file.schema_arrow.names)
        missing = sorted(required - present)
        if missing:
            raise ValueError(
                f"{Path(path).name} is not an ARC augmented v2 parquet; "
                f"missing columns: {', '.join(missing)}"
            )

    def _validate_v5_offline_aug(self, offline_aug: str) -> None:
        for aug_name, _ in parse_aug_str(offline_aug):
            if aug_name not in {"dih", "colorperm"}:
                raise ValueError(
                    "ARC pair cache currently supports shape-preserving ARC "
                    f"offline augmentations only, got {aug_name!r} in {offline_aug!r}"
                )

    def _store_v5_base_row(
        self,
        base_by_key: dict[tuple[int, str], dict[str, list]],
        noaug_test_data: dict[str, list],
        path_idx: int,
        puzzle_id: str,
        train_data: list,
        test_data: list,
    ) -> None:
        base_key = (path_idx, puzzle_id)
        if base_key in base_by_key:
            return

        train_data = train_data or []
        test_data = test_data or []

        selected_train = []
        selected_train_indices = []
        for pair_idx, pair in enumerate(train_data):
            output = pair.get("output")
            if output is None:
                continue
            if self._grid_fits(pair["input"]) and self._grid_fits(output):
                selected_train.append(pair)
                selected_train_indices.append(pair_idx)

        selected_test = []
        selected_test_indices = []
        for pair_idx, pair in enumerate(test_data):
            if not self._grid_fits(pair["input"]):
                continue
            output = pair.get("output")
            if not self.eval_mode and output is not None and not self._grid_fits(output):
                continue
            selected_test.append(pair)
            selected_test_indices.append(pair_idx)

        if test_data:
            filtered = [p for p in test_data if self._grid_fits(p["input"])]
            if filtered:
                noaug_test_data[puzzle_id] = filtered

        base_by_key[base_key] = {
            "train": selected_train,
            "train_indices": selected_train_indices,
            "test": selected_test,
            "test_indices": selected_test_indices,
        }

    def _build_v5_base_arrays(
        self,
        paths: dict[str, Path],
        base_keys: list[tuple[int, str]],
        base_by_key: dict[tuple[int, str], dict[str, list]],
        shape_dtype: np.dtype,
        shape_typecode: str,
    ) -> dict[str, np.ndarray]:
        input_raw_path = paths["dir"] / "inputs.raw"
        output_raw_path = paths["dir"] / "outputs.raw"

        train_input_shape_buffer = array(shape_typecode)
        train_output_shape_buffer = array(shape_typecode)
        test_input_shape_buffer = array(shape_typecode)
        test_output_shape_buffer = array(shape_typecode)
        test_output_present_buffer = bytearray()
        train_original_pair_index_buffer = array("I")
        test_original_pair_index_buffer = array("I")
        if train_original_pair_index_buffer.itemsize != np.dtype(np.uint32).itemsize:
            raise RuntimeError("array('I') is not uint32-sized on this platform")

        train_pair_start = [0]
        test_pair_start = [0]
        input_cell_count = 0
        output_cell_count = 0
        max_original_pair_index = 0

        def write_grid(raw_file, grid) -> tuple[int, int]:
            arr = np.asarray(grid, dtype=np.uint8)
            arr.ravel().tofile(raw_file)
            return int(arr.shape[0]), int(arr.shape[1])

        try:
            with open(input_raw_path, "wb") as input_raw, open(output_raw_path, "wb") as output_raw:
                for base_key in base_keys:
                    base = base_by_key[base_key]

                    for pair, original_idx in zip(base["train"], base["train_indices"]):
                        h, w = write_grid(input_raw, pair["input"])
                        train_input_shape_buffer.extend((h, w))
                        input_cell_count += h * w

                        h, w = write_grid(output_raw, pair["output"])
                        train_output_shape_buffer.extend((h, w))
                        output_cell_count += h * w

                        train_original_pair_index_buffer.append(int(original_idx))
                        max_original_pair_index = max(max_original_pair_index, int(original_idx))
                    train_pair_start.append(len(train_original_pair_index_buffer))

                for base_key in base_keys:
                    base = base_by_key[base_key]

                    for pair, original_idx in zip(base["test"], base["test_indices"]):
                        h, w = write_grid(input_raw, pair["input"])
                        test_input_shape_buffer.extend((h, w))
                        input_cell_count += h * w

                        output = pair.get("output")
                        if output is None:
                            test_output_present_buffer.append(0)
                            test_output_shape_buffer.extend((0, 0))
                        else:
                            h, w = write_grid(output_raw, output)
                            test_output_present_buffer.append(1)
                            test_output_shape_buffer.extend((h, w))
                            output_cell_count += h * w

                        test_original_pair_index_buffer.append(int(original_idx))
                        max_original_pair_index = max(max_original_pair_index, int(original_idx))
                    test_pair_start.append(len(test_original_pair_index_buffer))

            def save_raw_uint8(raw_path: Path, npy_path: Path, count: int) -> None:
                if count:
                    raw = np.fromfile(raw_path, dtype=np.uint8, count=count)
                    np.save(npy_path, raw)
                else:
                    np.save(npy_path, np.empty(0, dtype=np.uint8))
                raw_path.unlink(missing_ok=True)

            save_raw_uint8(input_raw_path, paths["inputs"], input_cell_count)
            save_raw_uint8(output_raw_path, paths["outputs"], output_cell_count)
        except Exception:
            input_raw_path.unlink(missing_ok=True)
            output_raw_path.unlink(missing_ok=True)
            raise

        train_pair_count = len(train_original_pair_index_buffer)
        test_pair_count = len(test_original_pair_index_buffer)
        return {
            "train_input_shape_hw": np.frombuffer(
                train_input_shape_buffer,
                dtype=shape_dtype,
            ).reshape(train_pair_count, 2),
            "train_output_shape_hw": np.frombuffer(
                train_output_shape_buffer,
                dtype=shape_dtype,
            ).reshape(train_pair_count, 2),
            "train_original_pair_index": np.frombuffer(
                train_original_pair_index_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(max_original_pair_index), copy=False),
            "test_input_shape_hw": np.frombuffer(
                test_input_shape_buffer,
                dtype=shape_dtype,
            ).reshape(test_pair_count, 2),
            "test_output_shape_hw": np.frombuffer(
                test_output_shape_buffer,
                dtype=shape_dtype,
            ).reshape(test_pair_count, 2),
            "test_output_present": np.frombuffer(test_output_present_buffer, dtype=np.bool_),
            "test_original_pair_index": np.frombuffer(
                test_original_pair_index_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(max_original_pair_index), copy=False),
            "train_pair_start": np.asarray(train_pair_start, dtype=np.int64),
            "test_pair_start": np.asarray(test_pair_start, dtype=np.int64),
        }

    def _build_v5_cache_from_parquet(self, cache_key: str) -> None:
        assert self.generated_cache_manager is not None
        paths = self.generated_cache_manager.prepare_write(cache_key)
        parquet_files = [pq.ParquetFile(path) for path in self.parquet_paths]
        for path, parquet_file in zip(self.parquet_paths, parquet_files):
            self._validate_v2_parquet(parquet_file, path)

        base_by_key: dict[tuple[int, str], dict[str, list]] = {}
        self._noaug_test_data = {}
        shape_typecode = "B" if self.max_grid_size <= np.iinfo(np.uint8).max else "H"
        shape_dtype = np.dtype(np.uint8 if shape_typecode == "B" else np.uint16)

        row_unique_idx_buffer = array("I")
        row_base_idx_buffer = array("I")
        row_train_count_buffer = array("I")
        row_test_count_buffer = array("I")
        uint32_buffers = [
            row_unique_idx_buffer,
            row_base_idx_buffer,
            row_train_count_buffer,
            row_test_count_buffer,
        ]
        if any(buffer.itemsize != np.dtype(np.uint32).itemsize for buffer in uint32_buffers):
            raise RuntimeError("array('I') is not uint32-sized on this platform")

        pair_count = 0
        max_row_count = 0
        used_puzzle_ids: set[str] = set()
        used_unique_strs: set[str] = set()
        base_key_to_idx: dict[tuple[int, str], int] = {}
        base_keys: list[tuple[int, str]] = []
        progress_desc = "Building ARC eval v5 cache" if self.eval_mode else "Building ARC train v5 cache"
        self._path_pair_ranges = []

        for path_idx, path in enumerate(self.parquet_paths):
            path_start_idx = pair_count
            pair_type = None if self.eval_mode else self.pair_types[path_idx]
            parquet_file = parquet_files[path_idx]
            path_name = Path(path).name
            path_rows = parquet_file.metadata.num_rows
            path_candidate_pairs = 0

            with tqdm(
                total=path_rows,
                desc=f"{progress_desc} [{path_name}]",
                unit="row",
                disable=get_rank() != 0,
                leave=False,
            ) as progress:
                for batch in self._iter_parquet_batches(parquet_file):
                    versions = batch.column(batch.schema.get_field_index("format_version"))
                    puzzle_ids = batch.column(batch.schema.get_field_index("puzzle_id"))
                    train_col = batch.column(batch.schema.get_field_index("train"))
                    test_col = batch.column(batch.schema.get_field_index("test"))
                    offline_augs_col = batch.column(batch.schema.get_field_index("offline_augs"))
                    for i in range(batch.num_rows):
                        version = versions[i].as_py()
                        if version != ARC_AUGMENTED_FORMAT_VERSION:
                            raise ValueError(
                                f"{path_name} row has ARC augmented format_version={version}; "
                                f"expected {ARC_AUGMENTED_FORMAT_VERSION}"
                            )
                        puzzle_id = puzzle_ids[i].as_py()
                        base_key = (path_idx, puzzle_id)
                        if base_key in base_by_key:
                            raise ValueError(
                                f"{path_name} contains duplicate v2 puzzle row for {puzzle_id!r}"
                            )
                        self._store_v5_base_row(
                            base_by_key,
                            self._noaug_test_data,
                            path_idx,
                            puzzle_id,
                            train_col[i].as_py(),
                            test_col[i].as_py(),
                        )
                        base = base_by_key[base_key]

                        offline_augs = offline_augs_col[i].as_py()
                        if not offline_augs:
                            raise ValueError(
                                f"{path_name} puzzle {puzzle_id!r} has no offline_augs"
                            )
                        offline_augs = [offline_aug or "" for offline_aug in offline_augs]
                        if offline_augs[0] != "":
                            raise ValueError(
                                f"{path_name} puzzle {puzzle_id!r} must list the no-augmentation "
                                "entry first in offline_augs"
                            )

                        if self.eval_mode:
                            train_count = 0
                            test_count = len(base["test"])
                        else:
                            train_count = len(base["train"])
                            test_count = len(base["test"]) if pair_type == "both" else 0
                        row_pair_count = train_count + test_count
                        if row_pair_count == 0:
                            continue

                        for offline_aug in offline_augs:
                            self._validate_v5_offline_aug(offline_aug)
                            path_candidate_pairs += row_pair_count

                            unique_str, unique_idx, _ = self._register_pair_ids(puzzle_id, offline_aug)
                            self._register_per_aug_ids(offline_aug)
                            if base_key not in base_key_to_idx:
                                base_key_to_idx[base_key] = len(base_keys)
                                base_keys.append(base_key)
                            row_unique_idx_buffer.append(unique_idx)
                            row_base_idx_buffer.append(base_key_to_idx[base_key])
                            row_train_count_buffer.append(train_count)
                            row_test_count_buffer.append(test_count)
                            used_puzzle_ids.add(puzzle_id)
                            used_unique_strs.add(unique_str)
                            max_row_count = max(max_row_count, train_count, test_count)
                            pair_count += row_pair_count
                    progress.update(batch.num_rows)

            self._path_pair_ranges.append((path_start_idx, pair_count))
            rows_loaded = path_rows if path_rows > 0 else 1
            print0(
                f"{path_name}: rows={path_rows:,} approx_pairs={path_candidate_pairs:,} "
                f"avg_pairs_per_row={path_candidate_pairs / rows_loaded:.1f}"
            )

        unique_strs, puzzle_ids, unique_puzzle_idx, unique_offline_aug_idx, offline_aug_strings = self._build_unique_tables()
        max_unique_idx = max(self.unique_str_to_int.values(), default=0)

        rows = {
            "row_unique_idx": np.frombuffer(
                row_unique_idx_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(max_unique_idx), copy=False),
            "row_base_idx": np.frombuffer(
                row_base_idx_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(len(base_keys) - 1), copy=False),
            "row_train_count": np.frombuffer(
                row_train_count_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(max_row_count), copy=False),
            "row_test_count": np.frombuffer(
                row_test_count_buffer,
                dtype=np.uint32,
            ).astype(_smallest_uint_dtype(max_row_count), copy=False),
        }
        base = self._build_v5_base_arrays(
            paths,
            base_keys,
            base_by_key,
            shape_dtype,
            shape_typecode,
        )
        meta = {
            "format_version": 5,
            "path_pair_ranges": self._path_pair_ranges,
            "puzzle_ids": puzzle_ids,
            "unique_strs": unique_strs,
            "unique_puzzle_idx": unique_puzzle_idx,
            "unique_offline_aug_idx": unique_offline_aug_idx,
            "offline_aug_strings": offline_aug_strings,
            "base_keys": [(int(path_idx), puzzle_id) for path_idx, puzzle_id in base_keys],
            "noaug_test_data": self._noaug_test_data,
            "per_aug_param_to_int": self._per_aug_param_to_int,
            "num_original_puzzles": len(used_puzzle_ids),
            "num_augmented_puzzles": len(used_unique_strs),
        }
        self.generated_cache_manager.save(cache_key, meta, rows, base)

    def _try_load_v5_cache(self, cache_key: str) -> bool:
        if not self.generated_cache_manager or not self.generated_cache_manager.exists(cache_key):
            return False
        data = self.generated_cache_manager.load(cache_key)
        meta = data["meta"]
        rows = data["rows"]
        base = data["base"]

        self.unique_str_to_int = {
            unique_str: idx
            for idx, unique_str in enumerate(meta["unique_strs"])
            if idx and unique_str
        }
        self._next_id = max(self.unique_str_to_int.values(), default=0) + 1
        self.puzzle_id_to_int = {
            puzzle_id: idx
            for idx, puzzle_id in enumerate(meta["puzzle_ids"])
            if puzzle_id
        }
        self._puzzle_id_next_id = max(self.puzzle_id_to_int.values(), default=-1) + 1
        self._path_pair_ranges = [(int(s), int(e)) for s, e in meta["path_pair_ranges"]]
        self._rebuild_idx_map_from_path_ranges()
        self._noaug_test_data = meta.get("noaug_test_data", {})
        self._per_aug_param_to_int = meta.get("per_aug_param_to_int", {})
        self._per_aug_next_id = {
            k: max(v.values(), default=0) + 1
            for k, v in self._per_aug_param_to_int.items()
        }
        self._num_original_puzzles_override = int(meta.get("num_original_puzzles", len(self.puzzle_id_to_int)))
        self._num_augmented_puzzles_override = int(meta.get("num_augmented_puzzles", len(self.unique_str_to_int)))

        self.pairs = GeneratedPairTable(
            rows=rows,
            base=base,
            input_flat=data["inputs"],
            output_flat=data["outputs"],
            unique_strs=meta["unique_strs"],
            puzzle_ids=meta["puzzle_ids"],
            offline_aug_strings=meta["offline_aug_strings"],
            unique_puzzle_idx=meta["unique_puzzle_idx"],
            unique_offline_aug_idx=meta["unique_offline_aug_idx"],
            per_aug_param_to_int=self._per_aug_param_to_int,
            eval_mode=self.eval_mode,
        )
        return True

    def _rebuild_idx_map_from_path_ranges(self):
        """Rebuild idx_map from path ranges and current multiplicities."""
        if not self._path_pair_ranges:
            return

        base_chunks = []
        for start, end in self._path_pair_ranges:
            if end > start:
                base_chunks.append(np.arange(start, end, dtype=np.int64))

        if self.eval_mode:
            self._idx_map = (
                np.concatenate(base_chunks)
                if base_chunks
                else np.empty(0, dtype=np.int64)
            )
            return

        idx_chunks = list(base_chunks)
        for path_idx, (start, end) in enumerate(self._path_pair_ranges):
            repeat_count = self.path_multiplicities[path_idx] - 1
            if repeat_count <= 0 or end <= start:
                continue
            path_indices = np.arange(start, end, dtype=np.int64)
            idx_chunks.append(np.tile(path_indices, repeat_count))
        self._idx_map = (
            np.concatenate(idx_chunks)
            if idx_chunks
            else np.empty(0, dtype=np.int64)
        )

    def _get_pair_arrays(self, pair_idx: int, copy: bool = False):
        return self.pairs.get_arrays(pair_idx, copy=copy)

    def _get_pair_metadata(self, pair_idx: int) -> dict:
        return self.pairs.get_metadata(pair_idx)

    @property
    def num_original_puzzles(self):
        if self._num_original_puzzles_override is not None:
            return self._num_original_puzzles_override
        return super().num_original_puzzles

    @property
    def num_augmented_puzzles(self):
        if self._num_augmented_puzzles_override is not None:
            return self._num_augmented_puzzles_override
        return super().num_augmented_puzzles

    @property
    def mean_pairs_per_puzzle(self):
        if self.num_augmented_puzzles == 0:
            return 0.0
        return len(self) / self.num_augmented_puzzles

    def __getitem__(self, idx):
        """
        Get a sample with online transforms applied.

        Returns dict with:
            - input: transformed input (2D grid or 1D sequence depending on transforms)
            - output: transformed output (or None in eval mode)
            - offline_aug: augmentation string from parquet
            - online_aug: augmentation string from runtime transforms
            - puzzle_id: original puzzle identifier
            - unique_str: "{puzzle_id}||{offline_aug}"
            - puzzle_embed_idx: integer index for embedding lookup
            - per_aug_embed_idxs: dict mapping aug type to embedding index (when enabled)
        """
        pair_idx = int(self._idx_map[idx])
        pair = self._get_pair_metadata(pair_idx)
        inp, out = self._get_pair_arrays(pair_idx)

        if self.online_transforms:
            rng = np.random.RandomState(idx) if self.deterministic_online_aug else np.random.RandomState()
            inp, out, online_aug = apply_chain_paired(inp, out, self.online_transforms, rng)
        else:
            online_aug = ""

        result = {
            "input": inp,
            "offline_aug": pair["offline_aug"],
            "online_aug": online_aug,
            "puzzle_id": pair["puzzle_id"],
            "puzzle_idx": self.puzzle_id_to_int[pair["puzzle_id"]],
            "unique_str": pair["unique_str"],
            "puzzle_embed_idx": self.unique_str_to_int[pair["unique_str"]],
        }
        if self.track_per_aug_embeddings:
            result["per_aug_embed_idxs"] = pair["per_aug_embed_idxs"]
        if out is not None:
            result["output"] = out
        return result

    def get_raw_pair(self, idx: int) -> dict:
        """
        Get raw pair data before any online transforms.

        Used by ARCContrastiveDataset to apply transforms multiple times with different RNG.

        Returns dict with:
            - input: raw input grid (numpy array)
            - output: raw output grid (numpy array) or None in eval mode
            - puzzle_id: original puzzle identifier
            - puzzle_idx: integer index for puzzle embedding
            - offline_aug: augmentation string from parquet
            - unique_str: "{puzzle_id}||{offline_aug}"
            - puzzle_embed_idx: integer index for embedding lookup
        """
        pair_idx = int(self._idx_map[idx])
        pair = self._get_pair_metadata(pair_idx)
        inp, out = self._get_pair_arrays(pair_idx, copy=True)
        result = {
            "input": inp,
            "output": out,
            "puzzle_id": pair["puzzle_id"],
            "puzzle_idx": self.puzzle_id_to_int[pair["puzzle_id"]],
            "offline_aug": pair["offline_aug"],
            "unique_str": pair["unique_str"],
            "puzzle_embed_idx": self.unique_str_to_int[pair["unique_str"]],
        }
        if self.track_per_aug_embeddings:
            result["per_aug_embed_idxs"] = pair["per_aug_embed_idxs"]
        return result

    def get_noaug_test_inputs(self) -> dict[str, list[np.ndarray]]:
        """Get original (non-augmented) test inputs for evaluator."""
        result = {}
        for puzzle_id, pairs in self._noaug_test_data.items():
            result[puzzle_id] = [np.array(p["input"], dtype=np.uint8) for p in pairs]
        return result

    def get_noaug_test_outputs(self) -> dict[str, list[list]]:
        """Get original test outputs for solution file generation.

        Note: Returns None entries if parquet lacks outputs (e.g., Kaggle test format).
        Only use on training-style datasets that include ground truth outputs.
        """
        result = {}
        for puzzle_id, pairs in self._noaug_test_data.items():
            result[puzzle_id] = [p["output"] for p in pairs]
        return result


def create_train_eval_datasets(
    train_paths: list[str | Path],
    eval_paths: list[str | Path],
    train_path_multiplicities: list[int] | None = None,
    train_transforms: list[Transform] | None = None,
    eval_transforms: list[Transform] | None = None,
    solution_output_path: str | Path | None = None,
    cache_dir: str | Path | bool | None = None,
    track_per_aug_embeddings: bool = False,
) -> tuple[ARCPerPairDataset, ARCPerPairDataset]:
    """
    Create train and eval datasets with appropriate transforms.

    Args:
        train_paths: Parquet files for training
        eval_paths: Parquet files for evaluation (must be subset of train_paths)
        train_path_multiplicities: Optional per-train-path multiplicity list.
            Eval dataset always uses multiplicity 1.
        train_transforms: Online transforms for training
        eval_transforms: Online transforms for evaluation (usually no random translation)
        solution_output_path: If provided, generate Kaggle-style solution JSON
        cache_dir: Directory for caching (None=default, False=disable)
        track_per_aug_embeddings: If True, build per-augmentation-type embedding indices

    Returns:
        (train_dataset, eval_dataset)
    """
    # Normalize to strings for set comparison (Path vs str equality)
    train_set = set(str(p) for p in train_paths)
    eval_set = set(str(p) for p in eval_paths)

    for ep in eval_set:
        if ep not in train_set:
            raise ValueError(
                f"Eval path '{ep}' not in train paths. "
                "Model won't have knowledge of puzzles from this file."
            )

    pair_types = []
    for tp in train_paths:
        if str(tp) in eval_set:
            pair_types.append("train")
        else:
            pair_types.append("both")

    train_dataset = ARCPerPairDataset(
        train_paths,
        pair_types=pair_types,
        path_multiplicities=train_path_multiplicities,
        eval_mode=False,
        online_transforms=train_transforms,
        cache_dir=cache_dir,
        track_per_aug_embeddings=track_per_aug_embeddings,
    )

    eval_dataset = ARCPerPairDataset(
        eval_paths,
        eval_mode=True,
        online_transforms=eval_transforms,
        unique_str_to_int=train_dataset.unique_str_to_int,
        puzzle_id_to_int=train_dataset.puzzle_id_to_int,
        cache_dir=cache_dir,
        track_per_aug_embeddings=track_per_aug_embeddings,
        per_aug_param_to_int=train_dataset._per_aug_param_to_int if track_per_aug_embeddings else None,
    )

    if solution_output_path is not None:
        solution = eval_dataset.get_noaug_test_outputs()
        with open(solution_output_path, "w") as f:
            json.dump(solution, f)

    return train_dataset, eval_dataset
