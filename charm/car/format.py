"""Compact ARC-v2 serialization helpers used by CAR datasets."""

from __future__ import annotations

import hashlib

import numpy as np
import pyarrow as pa

from charm.preprocessing.arc import ARC_AUGMENTED_FORMAT_VERSION
from charm.transforms import grid_hash


ARC_AUGMENTED_METADATA_KEY = b"realm.arc_augmented_format_version"


def build_arc_v2_schema() -> pa.Schema:
    """Build the compact schema consumed by CHARM's ARC pair loader."""

    grid_type = pa.list_(pa.list_(pa.int8()))
    pair_type = pa.struct([("input", grid_type), ("output", grid_type)])
    return pa.schema(
        [
            ("format_version", pa.int16()),
            ("puzzle_id", pa.string()),
            ("puzzle_hash", pa.string()),
            ("train", pa.list_(pair_type)),
            ("test", pa.list_(pair_type)),
            ("offline_augs", pa.list_(pa.string())),
            ("aug_hashes", pa.list_(pa.string())),
        ]
    ).with_metadata(
        {ARC_AUGMENTED_METADATA_KEY: str(ARC_AUGMENTED_FORMAT_VERSION).encode("ascii")}
    )


def puzzle_hash(puzzle: dict) -> str:
    """Compute the order-independent content hash used by ARC-v2 files."""

    hashes = []
    for split in ("train", "test"):
        for pair in puzzle.get(split, []):
            input_hash = grid_hash(np.asarray(pair["input"]))
            output = pair.get("output")
            if output is None:
                hashes.append(input_hash)
            else:
                hashes.append(f"{input_hash}|{grid_hash(np.asarray(output))}")
    hashes.sort()
    return hashlib.sha256("|".join(hashes).encode()).hexdigest()
