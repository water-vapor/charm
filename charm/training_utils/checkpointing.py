from typing import Optional, Any, List, Tuple
from dataclasses import dataclass
import os
import re
import random

import numpy as np
import torch
import torch.distributed as dist
from torch import nn


@dataclass
class LoadedCheckpoint:
    resolved_path: str
    state_dict: dict[str, Any]
    step: Optional[int]
    accum_step: Optional[int]
    optimizer_states: Optional[Any]
    python_rng_state: Any
    numpy_rng_state: Any
    rng_state: Any
    gpu_rng_state: Any
    real_epoch: Optional[int]
    last_eval_step: Optional[int]
    ema_state: Any


def _resolve_checkpoint_path(path: str) -> Optional[str]:
    """Resolve checkpoint path: if directory, prefer latest step_*.pt file."""
    if os.path.isfile(path):
        if not path.endswith(".pt"):
            raise ValueError(f"Unsupported checkpoint file format (expected .pt): '{path}'")
        return path

    if os.path.isdir(path):
        pattern = re.compile(r"step_(\d+)\.pt$")
        candidates: List[Tuple[int, str]] = []
        for file_name in os.listdir(path):
            match = pattern.match(file_name)
            if match:
                step = int(match.group(1))
                candidates.append((step, os.path.join(path, file_name)))

        if candidates:
            candidates.sort(key=lambda x: x[0])
            return candidates[-1][1]

    return None


def _load_checkpoint_payload(
    checkpoint_path: str,
    rank: int,
    device: str,
    log_prefix: str = "Loading checkpoint",
) -> LoadedCheckpoint:
    """
    Shared checkpoint payload loader.

    Resolves path, loads checkpoint dict, validates format, and extracts common fields
    used by both training resume and replay-eval.
    """
    resolved_path = _resolve_checkpoint_path(checkpoint_path)
    if resolved_path is None:
        raise FileNotFoundError(f"Could not resolve checkpoint path from '{checkpoint_path}'")

    if rank == 0:
        print(f"{log_prefix} {resolved_path}")

    # weights_only=False needed because we save Python/NumPy RNG states.
    checkpoint = torch.load(resolved_path, map_location=device, weights_only=False)

    if not (isinstance(checkpoint, dict) and "model_state_dict" in checkpoint):
        raise ValueError(
            f"Unsupported checkpoint payload format in '{resolved_path}'. "
            "Expected full checkpoint dict with 'model_state_dict'."
        )

    step = checkpoint.get("step")
    step = int(step) if step is not None else None
    accum_step = checkpoint.get("accum_step")
    accum_step = int(accum_step) if accum_step is not None else None
    real_epoch = checkpoint.get("real_epoch")
    real_epoch = int(real_epoch) if real_epoch is not None else None
    last_eval_step = checkpoint.get("last_eval_step")
    last_eval_step = int(last_eval_step) if last_eval_step is not None else None

    return LoadedCheckpoint(
        resolved_path=resolved_path,
        state_dict=checkpoint["model_state_dict"],
        step=step,
        accum_step=accum_step,
        optimizer_states=checkpoint.get("optimizer_states"),
        python_rng_state=checkpoint.get("python_rng_state"),
        numpy_rng_state=checkpoint.get("numpy_rng_state"),
        rng_state=checkpoint.get("rng_state"),
        gpu_rng_state=checkpoint.get("gpu_rng_state"),
        real_epoch=real_epoch,
        last_eval_step=last_eval_step,
        ema_state=checkpoint.get("ema_state"),
    )


def _resize_puzzle_embedding_if_needed(model: nn.Module, state_dict: dict[str, Any]) -> None:
    """Resize puzzle embedding if shape differs from checkpoint."""
    puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
    if puzzle_emb_name not in state_dict:
        return
    if not hasattr(model, "model") or not hasattr(model.model, "puzzle_emb"):
        return
    expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
    if puzzle_emb_name in state_dict:
        puzzle_emb = state_dict[puzzle_emb_name]
        if puzzle_emb.shape != expected_shape:
            print(f"Resetting puzzle embedding as shape is different. Found {puzzle_emb.shape}, Expected {expected_shape}")
            state_dict[puzzle_emb_name] = (
                torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
            )


def load_checkpoint(train_state: Any, config: Any, rank: int, device: str) -> dict[str, Any]:
    """
    Load checkpoint into train_state.

    Returns dict with:
        - real_epoch: epoch to resume from
        - last_eval_step: last evaluation step
        - ema_state: EMA state dict (if present)
        - resolved_checkpoint_path: resolved .pt path that was loaded
    """
    load_path = config.load_checkpoint
    if load_path is None:
        return {}

    if load_path == "latest":
        if config.checkpoint_path is None:
            raise ValueError("Cannot load latest checkpoint without a checkpoint_path configured.")
        load_path = config.checkpoint_path

    loaded = _load_checkpoint_payload(load_path, rank=rank, device=device)
    state_dict = loaded.state_dict
    optimizer_states = loaded.optimizer_states
    step = loaded.step
    accum_step = loaded.accum_step
    python_rng_state = loaded.python_rng_state
    numpy_rng_state = loaded.numpy_rng_state
    rng_state = loaded.rng_state
    gpu_rng_state = loaded.gpu_rng_state
    real_epoch = loaded.real_epoch
    last_eval_step = loaded.last_eval_step
    ema_state = loaded.ema_state

    assert step is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'step'"
    assert accum_step is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'accum_step'"
    assert real_epoch is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'real_epoch'"
    assert last_eval_step is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'last_eval_step'"
    assert optimizer_states is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'optimizer_states'"
    assert python_rng_state is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'python_rng_state'"
    assert numpy_rng_state is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'numpy_rng_state'"
    assert rng_state is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'rng_state'"
    assert gpu_rng_state is not None, f"Checkpoint '{loaded.resolved_path}' missing required resume field 'gpu_rng_state'"

    _resize_puzzle_embedding_if_needed(train_state.model, state_dict)
    train_state.model.load_state_dict(state_dict)

    if not config.load_optimizer_state:
        raise ValueError(
            "load_optimizer_state=False is not supported for strict resume loading."
        )
    if not isinstance(optimizer_states, (list, tuple)):
        raise ValueError(
            f"Checkpoint '{loaded.resolved_path}' has invalid optimizer_states type "
            f"{type(optimizer_states).__name__}; expected list/tuple."
        )
    if len(optimizer_states) != len(train_state.optimizers):
        raise ValueError(
            f"Optimizer count mismatch for '{loaded.resolved_path}' "
            f"({len(optimizer_states)} vs {len(train_state.optimizers)})."
        )
    for optimizer, optimizer_state in zip(train_state.optimizers, optimizer_states):
        optimizer.load_state_dict(optimizer_state)

    # Restore step counters
    train_state.step = int(step)
    train_state.accum_step = int(accum_step)

    # Reset carry since we do not serialize it
    train_state.carry = None

    # Restore RNG states.
    # For single-GPU: restore exact saved state for bit-identical reproducibility.
    # For multi-GPU: re-seed deterministically per rank (checkpoint only has rank 0's state).
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Fix for loading with different world size
    for optimizer in train_state.optimizers:
        for group in optimizer.param_groups:
            if "world_size" in group:
                group["world_size"] = world_size
        if "world_size" in optimizer.defaults:
            optimizer.defaults["world_size"] = world_size

    if world_size == 1:
        # Single-GPU: restore exact RNG state for bit-identical resume
        random.setstate(python_rng_state)
        np.random.set_state(numpy_rng_state)
        rng_tensor = torch.as_tensor(rng_state, device="cpu")
        if rng_tensor.dtype != torch.uint8:
            rng_tensor = rng_tensor.to(torch.uint8)
        torch.random.set_rng_state(rng_tensor)
        gpu_tensor = torch.as_tensor(gpu_rng_state, device="cpu")
        if gpu_tensor.dtype != torch.uint8:
            gpu_tensor = gpu_tensor.to(torch.uint8)
        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.set_rng_state(gpu_tensor)
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.set_rng_state(gpu_tensor)
    else:
        # Multi-GPU: deterministic re-seed per rank for diversity
        resume_seed = config.seed + rank + int(step) * 1000003  # large prime for mixing
        random.seed(resume_seed)
        np.random.seed(resume_seed % (2**32))  # numpy seed must be < 2^32
        torch.manual_seed(resume_seed)
        if device == "xpu" and torch.xpu.is_available():
            torch.xpu.manual_seed_all(resume_seed)
        elif device == "cuda" and torch.cuda.is_available():
            torch.cuda.manual_seed_all(resume_seed)

    return {
        "real_epoch": real_epoch,
        "last_eval_step": last_eval_step,
        "ema_state": ema_state,
        "resolved_checkpoint_path": loaded.resolved_path,
    }
