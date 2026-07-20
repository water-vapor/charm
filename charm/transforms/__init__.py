"""
Composable transform system for ARC puzzle augmentation.

Usage:
    from charm.transforms import (
        DihedralTransform, ScaleTransform, ColorPermTransform,
        CanvasEOSTransform, CanvasPadTransform, FlattenTransform,
        apply_chain, apply_chain_paired, inverse_chain,
    )

    # Create transform pipeline
    transforms = [
        DihedralTransform(p=1.0),
        ColorPermTransform(p=1.0),
        ScaleTransform(max_scale=2),
        CanvasEOSTransform(canvas_size=30),
        FlattenTransform(),
    ]

    # Apply to input/output pair
    inp_aug, out_aug, aug_str = apply_chain_paired(inp, out, transforms, rng)
    # aug_str = "dih|3||colorperm|0213456789||scale|2||canvas_eos|30,5,3||flatten|30,30"

    # Inverse in evaluator
    original = inverse_chain(prediction, aug_str)
"""

from .base import Transform
from .utils import (
    TRANSFORM_REGISTRY,
    register,
    apply_chain,
    apply_chain_paired,
    apply_with_aug_str,
    inverse_chain,
    parse_aug_str,
    validate_transforms,
    grid_hash,
    pair_hash,
)
from .augmentation import DihedralTransform, ScaleTransform, ColorPermTransform
from .encoding import (
    CanvasEOSTransform, CanvasPadTransform, FlattenTransform,
    CANVAS_SIZE, SEQ_LEN, PAD_TOKEN, EOS_TOKEN, COLOR_OFFSET, VOCAB_SIZE,
)

__all__ = [
    # Base class
    "Transform",
    # Registry and utilities
    "TRANSFORM_REGISTRY",
    "register",
    "apply_chain",
    "apply_chain_paired",
    "apply_with_aug_str",
    "inverse_chain",
    "parse_aug_str",
    "validate_transforms",
    "grid_hash",
    "pair_hash",
    # Grid-level transforms
    "DihedralTransform",
    "ScaleTransform",
    "ColorPermTransform",
    "CanvasEOSTransform",
    "CanvasPadTransform",
    "FlattenTransform",
    # Canvas constants
    "CANVAS_SIZE",
    "SEQ_LEN",
    "PAD_TOKEN",
    "EOS_TOKEN",
    "COLOR_OFFSET",
    "VOCAB_SIZE",
]
