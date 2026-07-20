"""Centralized path configuration for the project."""

from pathlib import Path
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = Path(os.environ.get("CHARM_DATA_DIR", PROJECT_ROOT / "data"))
CHECKPOINT_DIR = Path(os.environ.get("CHARM_CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
