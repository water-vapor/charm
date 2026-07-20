import os
from typing import Optional

import torch
from torch import nn

from charm.utils.model_loading import load_model_class
from charm.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from charm.training_utils.config_types import PretrainConfig, DatasetMetadata, TrainState
from charm.training_utils.optimizers import build_dense_optimizer
from charm.training_utils.runtime import DEVICE


def create_model(config: PretrainConfig, train_metadata: DatasetMetadata, rank: int, world_size: int):
    model_cfg = dict(
        **config.arch.__pydantic_extra__,  # type: ignore
        batch_size=config.global_batch_size // world_size,
        vocab_size=train_metadata.vocab_size,
        seq_len=train_metadata.seq_len,
        puzzle_embed_vocab_size=train_metadata.puzzle_embed_vocab_size,
        causal=False,  # Non-autoregressive
    )
    use_smart_embed = config.arch.__pydantic_extra__.get("use_smart_embed", False)
    embed_source = config.arch.__pydantic_extra__.get("smart_embed_source", "slotperm")
    if use_smart_embed:
        model_cfg["num_puzzles"] = train_metadata.num_unique_puzzles
        if embed_source == "separated" and train_metadata.per_aug_vocab_sizes:
            model_cfg["per_aug_vocab_sizes"] = train_metadata.per_aug_vocab_sizes

    model_cls = load_model_class(config.arch.name)

    loss_head_cls = load_model_class(config.arch.loss.name)
    loss_head_kwargs = config.arch.loss.__pydantic_extra__

    with torch.device(DEVICE):
        model: nn.Module = model_cls(model_cfg)
        print(model)
        model = loss_head_cls(model, **loss_head_kwargs)  # type: ignore
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)  # type: ignore

    trm_embedding_mode = config.arch.__pydantic_extra__.get("trm_embedding_mode", "default")
    sparse_puzzle_embeddings = []
    if use_smart_embed and hasattr(model.model, "puzzle_emb"):
        puzzle_emb_module = model.model.puzzle_emb
        if getattr(puzzle_emb_module, "use_sparse_colorperm", False) and hasattr(puzzle_emb_module, "colorperm_embed"):
            sparse_puzzle_embeddings.append(puzzle_emb_module.colorperm_embed)
        if getattr(puzzle_emb_module, "use_sparse_instance_residual", False) and hasattr(puzzle_emb_module, "instance_residual_embed"):
            sparse_puzzle_embeddings.append(puzzle_emb_module.instance_residual_embed)

    if config.arch.puzzle_emb_ndim == 0 or trm_embedding_mode in ("untrainable", "bitvector"):
        optimizers = [
            build_dense_optimizer(model.parameters(), config)
        ]
        optimizer_lrs = [config.lr]
    elif sparse_puzzle_embeddings:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                sparse_embedding.buffers(),
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
            for sparse_embedding in sparse_puzzle_embeddings
        ]
        optimizers.append(build_dense_optimizer(model.parameters(), config))
        optimizer_lrs = [config.puzzle_emb_lr] * len(sparse_puzzle_embeddings) + [config.lr]
    elif use_smart_embed:
        optimizers = [
            build_dense_optimizer(model.parameters(), config)
        ]
        optimizer_lrs = [config.lr]
    elif config.freeze_weights:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            )
        ]
        optimizer_lrs = [config.puzzle_emb_lr]
    else:
        optimizers = [
            CastedSparseEmbeddingSignSGD_Distributed(
                model.model.puzzle_emb.buffers(),  # type: ignore
                lr=0,
                weight_decay=config.puzzle_emb_weight_decay,
                world_size=world_size,
            ),
            build_dense_optimizer(model.parameters(), config),
        ]
        optimizer_lrs = [config.puzzle_emb_lr, config.lr]

    return model, optimizers, optimizer_lrs


def init_train_state(
    config: PretrainConfig,
    train_metadata: DatasetMetadata,
    samples_per_effective_epoch: float,
    rank: int,
    world_size: int,
    total_steps_override: Optional[int] = None,
) -> TrainState:
    effective_batch_size = config.global_batch_size * config.grad_accum_steps
    if total_steps_override is not None:
        total_steps = int(total_steps_override)
    else:
        total_steps = int(config.epochs * samples_per_effective_epoch / effective_batch_size)

    model, optimizers, optimizer_lrs = create_model(
        config, train_metadata, rank=rank, world_size=world_size
    )

    return TrainState(
        step=0,
        total_steps=total_steps,
        model=model,
        optimizers=optimizers,
        optimizer_lrs=optimizer_lrs,
        carry=None,
    )
