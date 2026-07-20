"""
Augmentation transforms: dihedral, scaling, and color permutation.
"""

import numpy as np

from .base import Transform
from .utils import register


# Dihedral transform

_DIHEDRAL_INVERSE = [0, 3, 2, 1, 4, 5, 6, 7]


@register("dih")
class DihedralTransform(Transform):
    """
    8 dihedral transforms (4 rotations + 4 reflections).

    Param format: single digit 0-7
        0: identity
        1: rotate 90° CCW
        2: rotate 180°
        3: rotate 270° CCW
        4: horizontal flip
        5: vertical flip
        6: transpose (main diagonal)
        7: anti-diagonal reflection
    """
    supported_problems = ("arc",)

    def __init__(self, p: float = 1.0, fixed_id: int | None = None):
        """
        Args:
            p: Probability of applying non-identity transform
            fixed_id: If set, always use this transform ID (0-7)
        """
        self.p = p
        self.fixed_id = fixed_id

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        if self.fixed_id is not None:
            return str(self.fixed_id)
        if rng is not None and rng.random() > self.p:
            return "0"  # Identity
        tid = rng.randint(0, 8) if rng else 0
        return str(tid)

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        tid = int(param_str)
        if tid == 0:
            return grid
        elif tid == 1:
            return np.rot90(grid, k=1)
        elif tid == 2:
            return np.rot90(grid, k=2)
        elif tid == 3:
            return np.rot90(grid, k=3)
        elif tid == 4:
            return np.fliplr(grid)
        elif tid == 5:
            return np.flipud(grid)
        elif tid == 6:
            return grid.T
        elif tid == 7:
            return np.fliplr(np.rot90(grid, k=1))
        return grid

    @classmethod
    def inverse(cls, grid: np.ndarray, param_str: str) -> np.ndarray:
        tid = int(param_str)
        inv_tid = _DIHEDRAL_INVERSE[tid]
        if inv_tid == 0:
            return grid
        elif inv_tid == 1:
            return np.rot90(grid, k=1)
        elif inv_tid == 2:
            return np.rot90(grid, k=2)
        elif inv_tid == 3:
            return np.rot90(grid, k=3)
        elif inv_tid == 4:
            return np.fliplr(grid)
        elif inv_tid == 5:
            return np.flipud(grid)
        elif inv_tid == 6:
            return grid.T
        elif inv_tid == 7:
            return np.fliplr(np.rot90(grid, k=1))
        return grid


# Scale transform

@register("scale")
class ScaleTransform(Transform):
    """
    Scale grid by integer factor(s).

    Param format:
        "n"         - uniform n×n scaling, max inverse
        "h,w"       - non-uniform scaling (height, width), max inverse
        "n,vote"    - uniform scaling, majority vote inverse
        "h,w,vote"  - non-uniform scaling, majority vote inverse
    """
    supported_problems = ("arc",)

    def __init__(
        self,
        max_scale: int | tuple[int, int] = 1,
        fixed_scale: int | tuple[int, int] | None = None,
        max_output_size: int = 30,
        inverse_method: str = "max",
    ):
        """
        Args:
            max_scale: Maximum scale factor. Int for uniform, tuple (h, w) for non-uniform.
            fixed_scale: If set, always use this scale factor.
            max_output_size: Clamp scale to keep output within this size.
            inverse_method: "max" (default) or "vote" for majority vote downsampling.
        """
        self.max_scale = max_scale
        self.fixed_scale = fixed_scale
        self.max_output_size = max_output_size
        self.inverse_method = inverse_method

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        if self.fixed_scale is not None:
            return self._format_params(self.fixed_scale)

        if isinstance(self.max_scale, tuple):
            max_h, max_w = self.max_scale
        else:
            max_h = max_w = self.max_scale

        if max_h <= 1 and max_w <= 1:
            return self._format_params(1)

        h, w = grid.shape
        max_h_allowed = max(1, self.max_output_size // h)
        max_w_allowed = max(1, self.max_output_size // w)
        actual_max_h = min(max_h, max_h_allowed)
        actual_max_w = min(max_w, max_w_allowed)

        if actual_max_h <= 1 and actual_max_w <= 1:
            return self._format_params(1)

        h_scale = rng.randint(1, actual_max_h + 1) if rng and actual_max_h > 1 else 1
        w_scale = rng.randint(1, actual_max_w + 1) if rng and actual_max_w > 1 else 1

        if isinstance(self.max_scale, tuple):
            return self._format_params((h_scale, w_scale))
        else:
            # For uniform max_scale, use the same scale for both
            scale = min(h_scale, w_scale)
            return self._format_params(scale)

    def _format_params(self, scale: int | tuple[int, int]) -> str:
        if isinstance(scale, tuple):
            h, w = scale
            if self.inverse_method == "vote":
                return f"{h},{w},vote"
            return f"{h},{w}"
        else:
            if self.inverse_method == "vote":
                return f"{scale},vote"
            return str(scale)

    @staticmethod
    def _parse_params(param_str: str) -> tuple[int, int, str]:
        """Parse param string to (h_scale, w_scale, method)."""
        parts = param_str.split(",")
        if len(parts) == 1:
            n = int(parts[0])
            return n, n, "max"
        elif len(parts) == 2:
            if parts[1] in ("max", "vote"):
                n = int(parts[0])
                return n, n, parts[1]
            else:
                return int(parts[0]), int(parts[1]), "max"
        else:
            return int(parts[0]), int(parts[1]), parts[2]

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        h_scale, w_scale, _ = ScaleTransform._parse_params(param_str)
        if h_scale == 1 and w_scale == 1:
            return grid
        return np.repeat(np.repeat(grid, h_scale, axis=0), w_scale, axis=1)

    @classmethod
    def inverse(cls, grid: np.ndarray, param_str: str) -> np.ndarray:
        h_scale, w_scale, method = cls._parse_params(param_str)
        if h_scale == 1 and w_scale == 1:
            return grid
        if method == "vote":
            return cls._vote_downsample(grid, h_scale, w_scale)
        return cls._max_downsample(grid, h_scale, w_scale)

    @staticmethod
    def _max_downsample(grid: np.ndarray, h_scale: int, w_scale: int) -> np.ndarray:
        """Downsample by taking max within each block."""
        h, w = grid.shape
        new_h, new_w = h // h_scale, w // w_scale
        trimmed = grid[:new_h * h_scale, :new_w * w_scale]
        return trimmed.reshape(new_h, h_scale, new_w, w_scale).max(axis=(1, 3)).astype(grid.dtype)

    @staticmethod
    def _vote_downsample(grid: np.ndarray, h_scale: int, w_scale: int) -> np.ndarray:
        """Downsample by majority vote within each block."""
        h, w = grid.shape
        new_h, new_w = h // h_scale, w // w_scale
        result = np.zeros((new_h, new_w), dtype=grid.dtype)
        for i in range(new_h):
            for j in range(new_w):
                block = grid[i*h_scale:(i+1)*h_scale, j*w_scale:(j+1)*w_scale].flatten()
                counts = np.bincount(block, minlength=10)
                result[i, j] = np.argmax(counts)
        return result


# Color permutation transform

@register("colorperm")
class ColorPermTransform(Transform):
    """
    Permute colors 1-9 (0 stays fixed as background).

    Param format: 10-char string "0XXXXXXXXX" where X's are permuted digits 1-9
    Example: "0312456789" swaps colors 1,2,3 to 3,1,2
    """
    supported_problems = ("arc",)

    def __init__(
        self,
        p: float = 1.0,
        fixed_perm: str | None = None,
        perm_pool: list[str] | None = None,
    ):
        """
        Args:
            p: Probability of applying non-identity permutation
            fixed_perm: If set, always use this permutation string
            perm_pool: If set, sample from this pre-generated pool of permutations
        """
        self.p = p
        self.fixed_perm = fixed_perm
        self.perm_pool = perm_pool

    @staticmethod
    def generate_pool(limit: int, seed: int) -> list[str]:
        """Pre-generate a pool of unique color permutations.

        Args:
            limit: Maximum number of permutations in the pool
            seed: Random seed for reproducibility

        Returns:
            List of permutation strings (always includes identity)
        """
        rng = np.random.RandomState(seed)
        pool = set()
        pool.add("0123456789")  # always include identity
        while len(pool) < limit:
            perm = list(range(1, 10))
            rng.shuffle(perm)
            pool.add("0" + "".join(map(str, perm)))
        return list(pool)

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        if self.fixed_perm is not None:
            return self.fixed_perm

        if self.perm_pool is not None and rng is not None:
            return rng.choice(self.perm_pool)

        if rng is not None and rng.random() > self.p:
            return "0123456789"  # Identity

        perm = list(range(10))
        if rng:
            colors_1_9 = list(range(1, 10))
            rng.shuffle(colors_1_9)
            perm[1:] = colors_1_9

        return "".join(map(str, perm))

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        if param_str == "0123456789":
            return grid
        perm_arr = np.array([int(c) for c in param_str], dtype=grid.dtype)
        return perm_arr[grid]

    @classmethod
    def inverse(cls, grid: np.ndarray, param_str: str) -> np.ndarray:
        if param_str == "0123456789":
            return grid
        inv_perm = [0] * 10
        for i, c in enumerate(param_str):
            inv_perm[int(c)] = i
        inv_arr = np.array(inv_perm, dtype=grid.dtype)
        return inv_arr[grid]
