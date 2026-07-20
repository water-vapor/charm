"""
Transform utilities: registry, chain operations, and hashing.
"""

import hashlib
import numpy as np

from .base import Transform


# Registry
TRANSFORM_REGISTRY: dict[str, type[Transform]] = {}


def register(name: str):
    """
    Decorator to register a transform class.

    Usage:
        @register("dih")
        class DihedralTransform(Transform):
            ...
    """
    def decorator(cls: type[Transform]) -> type[Transform]:
        TRANSFORM_REGISTRY[name] = cls
        cls.name = name
        return cls
    return decorator


# Chain operations

def apply_chain(
    grid: np.ndarray,
    transforms: list[Transform],
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, str]:
    """
    Apply chain of transforms to a single grid.

    Args:
        grid: Input grid
        transforms: List of Transform instances
        rng: Random state for reproducibility

    Returns:
        (result, aug_string) where aug_string is "name|params||name|params||..."
    """
    parts = []
    for t in transforms:
        params = t.sample_params(grid, rng)
        grid = t.apply(grid, params)
        parts.append(f"{t.name}|{params}")
    return grid, "||".join(parts)


def apply_chain_paired(
    inp: np.ndarray,
    out: np.ndarray | None,
    transforms: list[Transform],
    rng: np.random.RandomState | None = None,
) -> tuple[np.ndarray, np.ndarray | None, str]:
    """
    Apply same transform params to input and output grids.

    For ARC puzzles, input/output must receive identical augmentation.
    Parameters are sampled once and applied to both.

    For transforms that depend on grid shape (like canvas placement),
    we create a dummy grid with max(inp, out) shape to ensure both fit.

    Args:
        inp: Input grid
        out: Output grid (can be None for eval mode)
        transforms: List of Transform instances
        rng: Random state for reproducibility

    Returns:
        (inp_transformed, out_transformed, aug_string)
    """
    parts = []
    for t in transforms:
        # For sampling, use max shape between inp and out to ensure both fit
        if out is not None and inp.ndim == 2 and out.ndim == 2:
            max_h = max(inp.shape[0], out.shape[0])
            max_w = max(inp.shape[1], out.shape[1])
            sample_grid = np.zeros((max_h, max_w), dtype=inp.dtype)
        else:
            sample_grid = inp
        params = t.sample_params(sample_grid, rng)
        inp = t.apply(inp, params)
        if out is not None:
            out = t.apply(out, params)
        parts.append(f"{t.name}|{params}")
    return inp, out, "||".join(parts)


def apply_with_aug_str(data: np.ndarray, aug_str: str) -> np.ndarray:
    """
    Apply transforms using a pre-sampled aug_str.

    This is useful for applying the same augmentation to multiple grids
    (e.g., all pairs in a puzzle).

    Args:
        data: Input data (grid or sequence)
        aug_str: Augmentation string "name|params||name|params||..."

    Returns:
        Transformed data
    """
    if not aug_str:
        return data
    parts = aug_str.split("||")
    for part in parts:
        name, params = part.split("|", 1)
        transform_cls = TRANSFORM_REGISTRY[name]
        data = transform_cls().apply(data, params)
    return data


def inverse_chain(data: np.ndarray, aug_str: str) -> np.ndarray:
    """
    Parse aug_str and apply inverse transforms in reverse order.

    Args:
        data: Transformed data (grid or sequence)
        aug_str: Augmentation string "name|params||name|params||..."

    Returns:
        Original data (after inverting all transforms)
    """
    if not aug_str:
        return data
    parts = aug_str.split("||")
    for part in reversed(parts):
        name, params = part.split("|", 1)
        transform_cls = TRANSFORM_REGISTRY[name]
        data = transform_cls.inverse(data, params)
    return data


def parse_aug_str(aug_str: str) -> list[tuple[str, str]]:
    """
    Parse augmentation string into list of (name, params) tuples.

    Args:
        aug_str: "name|params||name|params||..."

    Returns:
        [(name1, params1), (name2, params2), ...]
    """
    if not aug_str:
        return []
    parts = aug_str.split("||")
    return [tuple(part.split("|", 1)) for part in parts]


def validate_transforms(transforms: list[Transform], problem_type: str) -> None:
    """
    Validate that all transforms support the given problem type.

    Args:
        transforms: List of Transform instances
        problem_type: Problem type ("arc" in this release)

    Raises:
        ValueError: If any transform doesn't support the problem type
    """
    for t in transforms:
        if problem_type not in t.supported_problems:
            raise ValueError(
                f"Transform '{t.name}' doesn't support problem type '{problem_type}'. "
                f"Supported: {t.supported_problems}"
            )


# Hashing utilities

def grid_hash(grid: np.ndarray) -> str:
    """
    Compute SHA256 hash of a grid.

    Args:
        grid: 2D numpy array

    Returns:
        Hex string hash
    """
    arr = np.asarray(grid, dtype=np.uint8)
    buffer = [x.to_bytes(1, byteorder='big') for x in arr.shape]
    buffer.append(arr.tobytes())
    return hashlib.sha256(b"".join(buffer)).hexdigest()


def pair_hash(grid_a: np.ndarray, grid_b: np.ndarray) -> str:
    """
    Compute combined hash of two grids.

    Args:
        grid_a: First grid
        grid_b: Second grid

    Returns:
        Combined hex string hash
    """
    h_a = grid_hash(grid_a)
    h_b = grid_hash(grid_b)
    return hashlib.sha256(f"{h_a}|{h_b}".encode()).hexdigest()
