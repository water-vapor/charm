"""CAR model construction, logical parameter counts, and optimizers."""

from __future__ import annotations

import os
from typing import Optional

import torch
from torch import nn

from charm.car.dataset import CARDatasetMetadata
from charm.car.task_memory import CARTaskMemory
from charm.models.sparse_embedding import (
    CastedSparseEmbedding,
    CastedSparseEmbeddingSignSGD_Distributed,
)
from charm.training_utils.config_types import PretrainConfig, TrainState
from charm.training_utils.optimizers import build_dense_optimizer
from charm.training_utils.runtime import DEVICE
from charm.utils.model_loading import load_model_class


def logical_parameter_counts(model: nn.Module) -> dict[str, int]:
    """Include sparse embedding buffers in reported model sizes."""

    torch_parameters = sum(parameter.numel() for parameter in model.parameters())
    sparse_values = sum(
        module.weights.numel()
        for module in model.modules()
        if isinstance(module, CastedSparseEmbedding)
    )
    counts = {
        "num_params": torch_parameters + sparse_values,
        "num_torch_parameters": torch_parameters,
        "num_sparse_embedding_values": sparse_values,
    }
    memory_parameters = sum(
        module.task_memory_parameter_count()
        for module in model.modules()
        if isinstance(module, CARTaskMemory)
    )
    if memory_parameters:
        counts["car_task_memory_params"] = memory_parameters
    return counts


def create_car_model(
    config: PretrainConfig,
    metadata: CARDatasetMetadata,
    rank: int,
    world_size: int,
):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,
        batch_size=config.global_batch_size // world_size,
        vocab_size=metadata.vocab_size,
        seq_len=metadata.seq_len,
        puzzle_embed_vocab_size=metadata.puzzle_embed_vocab_size,
        car_num_tasks=metadata.car_num_tasks,
        car_num_rules=metadata.car_num_rules,
        causal=False,
    )
    model_cls = load_model_class(config.arch.name)
    loss_head_cls = load_model_class(config.arch.loss.name)
    loss_head_kwargs = config.arch.loss.__pydantic_extra__

    with torch.device(DEVICE):
        core_model: nn.Module = model_cls(model_cfg)
        print(core_model)
        memory = core_model.puzzle_emb
        if not isinstance(memory, CARTaskMemory):
            raise TypeError("arch=car must construct CARTaskMemory")
        sparse_embeddings = memory.sparse_embeddings
        model: nn.Module = loss_head_cls(
            core_model,
            **loss_head_kwargs,
        )
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)

    optimizers = [
        CastedSparseEmbeddingSignSGD_Distributed(
            sparse_embedding.buffers(),
            lr=0,
            weight_decay=config.puzzle_emb_weight_decay,
            world_size=world_size,
        )
        for sparse_embedding in sparse_embeddings
    ]
    optimizers.append(build_dense_optimizer(model.parameters(), config))
    optimizer_lrs = [config.puzzle_emb_lr] * len(sparse_embeddings) + [config.lr]
    return model, optimizers, optimizer_lrs


def init_car_train_state(
    config: PretrainConfig,
    metadata: CARDatasetMetadata,
    samples_per_effective_epoch: float,
    rank: int,
    world_size: int,
    total_steps_override: Optional[int] = None,
) -> TrainState:
    effective_batch_size = config.global_batch_size * config.grad_accum_steps
    if total_steps_override is not None:
        total_steps = int(total_steps_override)
    else:
        total_steps = int(
            config.epochs * samples_per_effective_epoch / effective_batch_size
        )
    model, optimizers, optimizer_lrs = create_car_model(
        config,
        metadata,
        rank=rank,
        world_size=world_size,
    )
    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )
