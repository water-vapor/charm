"""Donor model construction, EMA-merged loading with table expansion, and arm setup."""

import os
from dataclasses import dataclass, field

import torch
from torch import nn
from torch.optim import AdamW

from charm.models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from charm.models.smart_task_embedding import SmartTaskEmbedding
from charm.utils.model_loading import load_model_class
from charm.transforms import SEQ_LEN, VOCAB_SIZE

# Table rows are appended for stream puzzles; these keys are zero-padded on load.
EXPAND_SUFFIXES = (
    "puzzle_emb.puzzle_embed.weight",              # semantic table (CoSE variants)
    "puzzle_emb.instance_residual_embed.weights",  # instance residual table (CoSE variants)
    "puzzle_emb.weights",                          # instance table (plain-table variants)
)


def inner_module(model: nn.Module):
    m = model._orig_mod if hasattr(model, "_orig_mod") else model
    return m.model.inner


def build_model(arch: dict, puzzle_vocab: int, num_puzzles: int, batch_size: int,
                device: str) -> nn.Module:
    arch = dict(arch)
    assert not arch.get("puzzle_embed_dual"), \
        "dual semantic tables store two banks; end-padding expansion would corrupt them"
    loss = dict(arch.pop("loss"))
    model_cls = load_model_class(arch.pop("name"))
    loss_cls = load_model_class(loss.pop("name"))
    model_cfg = dict(arch, batch_size=batch_size, vocab_size=VOCAB_SIZE, seq_len=SEQ_LEN,
                     puzzle_embed_vocab_size=puzzle_vocab, num_puzzles=num_puzzles, causal=False)
    with torch.device(device):
        model = model_cls(model_cfg)
        model = loss_cls(model, **loss)
        if "DISABLE_COMPILE" not in os.environ:
            model = torch.compile(model)
    return model


def load_expanded(model: nn.Module, ckpt_path: str, device: str) -> dict:
    """Load model_state_dict with EMA parameters merged in; zero-pad the expandable tables.

    EMA covers parameters only (the reported donor numbers were measured with EMA
    weights); sparse tables are buffers and load as saved.
    """
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = payload["model_state_dict"]
    for k, v in (payload.get("ema_state") or {}).items():
        assert k in sd
        sd[k] = v

    model_sd = model.state_dict()
    model_prefixed = next(iter(model_sd)).startswith("_orig_mod.")

    def rekey(k: str) -> str:
        bare = k[len("_orig_mod."):] if k.startswith("_orig_mod.") else k
        return f"_orig_mod.{bare}" if model_prefixed else bare

    sd = {rekey(k): v for k, v in sd.items()}
    expanded = {}
    for k, v in sd.items():
        target = model_sd[k].shape
        if v.shape != target:
            assert k.endswith(EXPAND_SUFFIXES) and v.shape[0] < target[0] and v.shape[1:] == target[1:]
            padded = torch.zeros(target, dtype=v.dtype, device=v.device)
            padded[: v.shape[0]] = v
            sd[k] = padded
            expanded[k] = (v.shape[0], target[0])
    model.load_state_dict(sd)
    return {"step": payload.get("step"), "expanded": expanded}


@dataclass
class Arm:
    embed_param: torch.Tensor | None            # semantic table; fresh row-local AdamW per stage
    sparse_opt: torch.optim.Optimizer | None    # instance rows; stateless row-local SignSGD
    shared_opt: torch.optim.Optimizer | None    # shared weights, when the arm trains them
    shared_params: list = field(default_factory=list)
    shared_lr: float = 0.0
    beta: tuple = (0.9, 0.95)

    def fresh_embed_opt(self):
        # Fresh per stage: no momentum carry-over between puzzles. weight_decay=0 so
        # rows outside the current batch (all donor rows) are exact no-ops.
        if self.embed_param is None:
            return None
        return AdamW([self.embed_param], lr=0, weight_decay=0.0, betas=self.beta)


def setup_arm(model: nn.Module, cfg, train_shared: bool, comp_only: bool = False,
              world_size: int = 1) -> Arm:
    """Freeze everything, then mark what the arm trains.

    Row parameters (semantic + instance) always train: they are puzzle-private, so
    updating them cannot interfere with any other puzzle. train_shared additionally
    unfreezes the shared weights (all of them, or only the CoSE composition params
    when comp_only — used by the CL-algorithm baselines).
    """
    inner = inner_module(model)
    puzzle_emb = inner.puzzle_emb
    smart = isinstance(puzzle_emb, SmartTaskEmbedding)
    for p in model.parameters():
        p.requires_grad_(False)

    embed_param = None
    if smart:
        puzzle_emb.puzzle_embed.weight.requires_grad_(True)
        embed_param = puzzle_emb.puzzle_embed.weight

    sparse_module = None
    if smart and puzzle_emb.use_sparse_instance_residual:
        sparse_module = puzzle_emb.instance_residual_embed
    elif not smart:
        sparse_module = puzzle_emb
    sparse_opt = None
    if sparse_module is not None:
        sparse_opt = CastedSparseEmbeddingSignSGD_Distributed(
            sparse_module.buffers(), lr=0, weight_decay=cfg.sparse_weight_decay,
            world_size=world_size)

    shared_params = []
    if train_shared:
        if comp_only:
            assert smart, "comp scope requires a CoSE variant"
            shared_params = [p for name, p in puzzle_emb.named_parameters()
                            if not name.startswith("puzzle_embed.")]
        else:
            shared_params = [p for p in model.parameters() if p is not embed_param]
        for p in shared_params:
            p.requires_grad_(True)
    shared_opt = None
    if shared_params:
        shared_opt = AdamW(shared_params, lr=0, weight_decay=cfg.ft_weight_decay,
                           betas=(cfg.beta1, cfg.beta2))

    return Arm(embed_param=embed_param, sparse_opt=sparse_opt, shared_opt=shared_opt,
               shared_params=shared_params, shared_lr=cfg.ft_lr, beta=(cfg.beta1, cfg.beta2))


def donor_snapshot(model: nn.Module, donor_vocab: int, donor_puzzles: int) -> dict:
    """CPU clone of the donor-owned table regions, for drift verification at sweeps."""
    puzzle_emb = inner_module(model).puzzle_emb
    snap = {}
    if isinstance(puzzle_emb, SmartTaskEmbedding):
        snap["semantic"] = puzzle_emb.puzzle_embed.weight[:donor_puzzles].detach().cpu().clone()
        if puzzle_emb.use_sparse_instance_residual:
            snap["instance"] = puzzle_emb.instance_residual_embed.weights[:donor_vocab].detach().cpu().clone()
    else:
        snap["instance"] = puzzle_emb.weights[:donor_vocab].detach().cpu().clone()
    return snap


def donor_drift(model: nn.Module, snap: dict) -> dict[str, float]:
    puzzle_emb = inner_module(model).puzzle_emb
    drift = {}
    for name, ref in snap.items():
        if name == "semantic":
            current = puzzle_emb.puzzle_embed.weight[: ref.shape[0]]
        elif isinstance(puzzle_emb, SmartTaskEmbedding):
            current = puzzle_emb.instance_residual_embed.weights[: ref.shape[0]]
        else:
            current = puzzle_emb.weights[: ref.shape[0]]
        drift[name] = float((current.detach().cpu() - ref).abs().max())
    return drift
