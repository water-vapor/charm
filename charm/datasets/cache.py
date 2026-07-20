"""Optional caching for puzzle datasets."""

import os
import json
import pickle
import hashlib
import fcntl
from contextlib import contextmanager
from typing import Any
from pathlib import Path
import numpy as np

from charm.utils.paths import DATA_DIR


class CacheManager:
    """
    Manages dataset caching to disk.

    Cache format:
        - {key}_meta.pkl: metadata (pair metadata, shapes, mappings)
        - {key}_inputs.npy: flattened input arrays
        - {key}_outputs.npy: flattened output arrays
    """

    def __init__(self, cache_dir: str | Path | None = None):
        self.cache_dir = str(cache_dir) if cache_dir else str(DATA_DIR / "cached")

    def compute_key(self, config: dict[str, Any]) -> str:
        """Compute cache key from configuration dict."""
        s = json.dumps(config, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _paths(self, cache_key: str) -> tuple:
        prefix = os.path.join(self.cache_dir, cache_key)
        return (
            prefix + "_meta.pkl",
            prefix + "_inputs.npy",
            prefix + "_outputs.npy",
        )

    def exists(self, cache_key: str) -> bool:
        """Check if cache exists for given key."""
        return all(os.path.exists(p) for p in self._paths(cache_key))

    def load(self, cache_key: str) -> dict[str, Any]:
        """
        Load cached data.

        Returns:
            dict with "meta", "inputs", "outputs" keys
        """
        meta_path, inputs_path, outputs_path = self._paths(cache_key)
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)
        return {
            "meta": meta,
            "inputs": np.load(inputs_path),
            "outputs": np.load(outputs_path),
        }

    def save(
        self,
        cache_key: str,
        meta: dict[str, Any],
        pairs: list[dict],
    ) -> None:
        """
        Save data to cache with file locking for DDP safety.

        Args:
            cache_key: Cache key
            meta: Metadata dict (must not contain input/output arrays)
            pairs: List of pair dicts with "input" and "output" keys
        """
        os.makedirs(self.cache_dir, exist_ok=True)
        lock_path = os.path.join(self.cache_dir, cache_key + ".lock")

        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                meta_path, inputs_path, outputs_path = self._paths(cache_key)

                with open(meta_path, "wb") as f:
                    pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)

                input_size = sum(p["input"].size for p in pairs)
                input_flat = np.empty(input_size, dtype=np.uint8)
                offset = 0
                for p in pairs:
                    arr = np.ravel(p["input"]).astype(np.uint8, copy=False)
                    next_offset = offset + arr.size
                    input_flat[offset:next_offset] = arr
                    offset = next_offset

                output_size = sum(
                    p["output"].size for p in pairs if p["output"] is not None
                )
                output_flat = np.empty(output_size, dtype=np.uint8)
                offset = 0
                for p in pairs:
                    if p["output"] is None:
                        continue
                    arr = np.ravel(p["output"]).astype(np.uint8, copy=False)
                    next_offset = offset + arr.size
                    output_flat[offset:next_offset] = arr
                    offset = next_offset

                np.save(inputs_path, input_flat)
                np.save(outputs_path, output_flat)
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)


class ARCGeneratedCacheManager:
    """Version-separated generated-grid cache for ARC per-pair datasets."""

    def __init__(self, cache_dir: str | Path | None = None):
        base_dir = Path(cache_dir) if cache_dir else DATA_DIR / "cached"
        self.cache_dir = base_dir / "v5"

    def compute_key(self, config: dict[str, Any]) -> str:
        s = json.dumps(config, sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:16]

    def _dir(self, cache_key: str) -> Path:
        return self.cache_dir / cache_key

    def _paths(self, cache_key: str) -> dict[str, Path]:
        cache_path = self._dir(cache_key)
        return {
            "dir": cache_path,
            "meta": cache_path / "meta.pkl",
            "rows": cache_path / "rows.npz",
            "base": cache_path / "base.npz",
            "inputs": cache_path / "inputs.npy",
            "outputs": cache_path / "outputs.npy",
        }

    def exists(self, cache_key: str) -> bool:
        paths = self._paths(cache_key)
        return all(
            paths[name].exists()
            for name in ("meta", "rows", "base", "inputs", "outputs")
        )

    @contextmanager
    def lock(self, cache_key: str):
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        lock_path = self.cache_dir / f"{cache_key}.lock"
        with open(lock_path, "w") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock_file, fcntl.LOCK_UN)

    def load(self, cache_key: str) -> dict[str, Any]:
        paths = self._paths(cache_key)
        with open(paths["meta"], "rb") as f:
            meta = pickle.load(f)
        rows_npz = np.load(paths["rows"], allow_pickle=False)
        rows = {name: rows_npz[name] for name in rows_npz.files}
        rows_npz.close()
        base_npz = np.load(paths["base"], allow_pickle=False)
        base = {name: base_npz[name] for name in base_npz.files}
        base_npz.close()
        return {
            "meta": meta,
            "rows": rows,
            "base": base,
            "inputs": np.load(paths["inputs"]),
            "outputs": np.load(paths["outputs"]),
        }

    def prepare_write(self, cache_key: str) -> dict[str, Path]:
        paths = self._paths(cache_key)
        paths["dir"].mkdir(parents=True, exist_ok=True)
        return paths

    def save(
        self,
        cache_key: str,
        meta: dict[str, Any],
        rows: dict[str, np.ndarray],
        base: dict[str, np.ndarray],
    ) -> None:
        paths = self._paths(cache_key)
        with open(paths["meta"], "wb") as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        np.savez(paths["rows"], **rows)
        np.savez(paths["base"], **base)
