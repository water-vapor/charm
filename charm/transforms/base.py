"""
Base class for grid-level transforms (per input-output pair augmentations).
"""

from abc import ABC, abstractmethod
import numpy as np


class Transform(ABC):
    """
    Base class for grid-level transforms.

    Each transform:
    - Has a unique name (used in aug string)
    - Has supported_problems indicating which problem types it works with
    - Can sample random parameters
    - Can apply with specific parameters
    - Has an inverse operation

    Aug string format: "name|params" where params is transform-specific.
    """

    name: str  # Set by @register decorator or subclass
    supported_problems: tuple[str, ...] = ("arc",)  # Override in subclass

    @abstractmethod
    def sample_params(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> str:
        """
        Sample random parameters for this transform.

        Args:
            grid: Input grid (used to determine valid param ranges)
            rng: Random state for reproducibility

        Returns:
            Parameter string (e.g., "4" for dih, "5,3" for translate)
        """
        pass

    @abstractmethod
    def apply(self, grid: np.ndarray, param_str: str) -> np.ndarray:
        """
        Apply transform with specific parameters.

        Args:
            grid: Input (2D grid or 1D sequence depending on transform)
            param_str: Parameters from sample_params or parsed from aug string

        Returns:
            Transformed data
        """
        pass

    @classmethod
    @abstractmethod
    def inverse(cls, data: np.ndarray, param_str: str) -> np.ndarray:
        """
        Inverse transform.

        Args:
            data: Transformed data
            param_str: Same parameters used in apply()

        Returns:
            Original data (approximately, for lossy transforms)
        """
        pass

    def __call__(
        self,
        grid: np.ndarray,
        rng: np.random.RandomState | None = None
    ) -> tuple[np.ndarray, str]:
        """
        Convenience: sample params and apply.

        Returns:
            (transformed_grid, param_string)
        """
        params = self.sample_params(grid, rng)
        return self.apply(grid, params), params
