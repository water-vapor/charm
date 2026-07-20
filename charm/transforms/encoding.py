"""
Encoding transforms: canvas placement and sequence flattening.

These transforms handle conversion from variable-size grids to fixed-size
representations suitable for neural network input.
"""

import numpy as np

from .base import Transform
from .utils import register


# Default canvas configuration
CANVAS_SIZE = 30
SEQ_LEN = CANVAS_SIZE * CANVAS_SIZE  # 900

# Token constants for CanvasEOSTransform
PAD_TOKEN = 0
EOS_TOKEN = 1
COLOR_OFFSET = 2
VOCAB_SIZE = 12  # PAD + EOS + 10 colors


@register("canvas_eos")
class CanvasEOSTransform(Transform):
    """
    TRM-style canvas: place grid with L-shaped EOS boundary.

    Token scheme:
        PAD = 0 (background padding)
        EOS = 1 (end-of-sequence boundary marker)
        colors 0-9 -> tokens 2-11

    Param format: "size,row_offset,col_offset"
    Example: "30,5,3" = 30x30 canvas, grid placed at row 5, col 3
    """
    supported_problems = ("arc",)

    def __init__(self, canvas_size: int = 30, p_translate: float = 1.0):
        """
        Args:
            canvas_size: Size of square canvas
            p_translate: Probability of random translation (vs fixed at origin)
        """
        self.canvas_size = canvas_size
        self.p_translate = p_translate

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        h, w = grid.shape
        max_r = self.canvas_size - h - 1
        max_c = self.canvas_size - w - 1

        if (rng is not None and
            rng.random() < self.p_translate and
            max_r > 0 and max_c > 0):
            r = rng.randint(0, max_r + 1)
            c = rng.randint(0, max_c + 1)
        else:
            r, c = 0, 0

        return f"{self.canvas_size},{r},{c}"

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        size, r, c = map(int, param_str.split(","))
        h, w = grid.shape

        canvas = np.full((size, size), PAD_TOKEN, dtype=np.uint8)
        canvas[r:r + h, c:c + w] = grid + COLOR_OFFSET
        eos_row, eos_col = r + h, c + w
        if eos_col < size:
            canvas[r:eos_row, eos_col] = EOS_TOKEN
        if eos_row < size:
            canvas[eos_row, c:eos_col] = EOS_TOKEN

        return canvas

    @classmethod
    def inverse(cls, canvas: np.ndarray, param_str: str) -> np.ndarray:
        size, r, c = map(int, param_str.split(","))

        max_area = 0
        best_h, best_w = 0, 0
        max_w = size - c

        for h in range(1, size - r + 1):
            for w in range(1, max_w + 1):
                token = canvas[r + h - 1, c + w - 1]
                if token < COLOR_OFFSET or token > COLOR_OFFSET + 9:
                    max_w = w - 1
                    break

            area = h * max_w
            if area > max_area:
                max_area = area
                best_h, best_w = h, max_w

        grid = canvas[r:r + best_h, c:c + best_w] - COLOR_OFFSET
        return grid.astype(np.uint8)


@register("canvas_pad")
class CanvasPadTransform(Transform):
    """
    vis_arc-style canvas: place grid with IGNORE padding and PAD boundary.

    Token scheme:
        colors 0-9 -> tokens 0-9 (no offset)
        IGNORE = 10 (masked/invalid region)
        PAD = 11 (boundary marker)

    Param format: "size,row_offset,col_offset"
    """
    supported_problems = ("arc",)

    IGNORE = 10
    PAD = 11

    def __init__(self, canvas_size: int = 30, p_translate: float = 1.0):
        """
        Args:
            canvas_size: Size of square canvas
            p_translate: Probability of random translation
        """
        self.canvas_size = canvas_size
        self.p_translate = p_translate

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        h, w = grid.shape
        max_r = self.canvas_size - h - 1
        max_c = self.canvas_size - w - 1

        if (rng is not None and
            rng.random() < self.p_translate and
            max_r > 1 and max_c > 1):
            r = rng.randint(1, max_r + 1)
            c = rng.randint(1, max_c + 1)
        else:
            r, c = 1, 1

        return f"{self.canvas_size},{r},{c}"

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        size, r, c = map(int, param_str.split(","))
        h, w = grid.shape

        canvas = np.full((size, size), self.IGNORE, dtype=np.uint8)
        canvas[r:r + h, c:c + w] = grid
        canvas[r:r + h, c + w] = self.PAD
        canvas[r + h, c:c + w + 1] = self.PAD

        return canvas

    @classmethod
    def inverse(cls, canvas: np.ndarray, param_str: str) -> np.ndarray:
        size, r, c = map(int, param_str.split(","))

        w = 0
        while c + w < size and canvas[r, c + w] != cls.PAD:
            w += 1

        h = 0
        while r + h < size and canvas[r + h, c] != cls.PAD:
            h += 1

        return canvas[r:r + h, c:c + w].astype(np.uint8)


@register("flatten")
class FlattenTransform(Transform):
    """
    Flatten 2D grid to 1D sequence (row-major order).

    Param format: "height,width"
    Example: "30,30" = original 30x30 grid
    """
    supported_problems = ("arc",)

    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        return f"{grid.shape[0]},{grid.shape[1]}"

    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        return grid.flatten()

    @classmethod
    def inverse(cls, seq: np.ndarray, param_str: str) -> np.ndarray:
        h, w = map(int, param_str.split(","))
        return seq.reshape(h, w)
