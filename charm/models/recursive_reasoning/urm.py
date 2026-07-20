from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from charm.models.common import trunc_normal_init_
from charm.models.layers import rms_norm, ConvSwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from charm.models.sparse_embedding import CastedSparseEmbedding


@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class URMConfig(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    puzzle_embed_vocab_size: int
    vocab_size: int
    num_layers: int
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    L_cycles: int
    H_cycles: int
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 0

    # Smart task embedding config
    use_smart_embed: bool = False
    smart_embed_strategy: str = "add"
    smart_embed_source: str = "slotperm"
    num_puzzles: int = 1000
    per_aug_vocab_sizes: dict = {}
    smart_embed_dim: int | None = None
    smart_embed_heads: int = 1
    smart_embed_rank: int | None = None
    smart_embed_moe_experts: int = 4
    seq_embed_strategy: str = "prepend"
    puzzle_embed_dual: bool = False
    puzzle_embed_dual_mode: str = "random"
    smart_embed_interaction_mode: str = "none"  # "none", "instance_residual", or "instance_residual_engram"
    smart_embed_interaction_rank: int | None = None  # low-rank size for interaction residual (None/full rank)
    smart_embed_interaction_gate: bool = False  # gate residual with semantic embedding (instance_residual only)


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(
            hidden_size=config.hidden_size,
            head_dim=config.hidden_size // config.num_heads,
            num_heads=config.num_heads,
            num_key_value_heads=config.num_heads,
            causal=False,
        )
        self.mlp = ConvSwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        attn_output = self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + attn_output, variance_epsilon=self.norm_eps)
        mlp_output = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + mlp_output, variance_epsilon=self.norm_eps)
        return hidden_states


class URM_Inner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)
        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(
            self.config.vocab_size,
            self.config.hidden_size,
            init_std=embed_init_std,
            cast_to=self.forward_dtype,
        )
        self.lm_head = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head = CastedLinear(self.config.hidden_size, 2, bias=True)
        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len

        if self.config.puzzle_emb_ndim > 0:
            if self.config.use_smart_embed:
                from ..smart_task_embedding import SmartTaskEmbedding
                self.puzzle_emb = SmartTaskEmbedding(
                    num_puzzles=self.config.num_puzzles,
                    hidden_size=self.config.hidden_size,
                    combine_strategy=self.config.smart_embed_strategy,
                    embed_source=self.config.smart_embed_source,
                    per_aug_vocab_sizes=self.config.per_aug_vocab_sizes,
                    smart_embed_dim=self.config.smart_embed_dim,
                    smart_embed_heads=self.config.smart_embed_heads,
                    smart_embed_rank=self.config.smart_embed_rank,
                    smart_embed_moe_experts=self.config.smart_embed_moe_experts,
                    batch_size=self.config.batch_size,
                    forward_dtype=self.forward_dtype,
                    puzzle_embed_dual=self.config.puzzle_embed_dual,
                    puzzle_embed_dual_mode=self.config.puzzle_embed_dual_mode,
                    interaction_mode=self.config.smart_embed_interaction_mode,
                    interaction_rank=self.config.smart_embed_interaction_rank,
                    interaction_use_gate=self.config.smart_embed_interaction_gate,
                    num_instance_embeddings=self.config.puzzle_embed_vocab_size,
                )
            else:
                self.puzzle_emb = CastedSparseEmbedding(
                    self.config.puzzle_embed_vocab_size,
                    self.config.puzzle_emb_ndim,
                    batch_size=self.config.batch_size,
                    init_std=0,
                    cast_to=self.forward_dtype,
                )

        self.rotary_emb = RotaryEmbedding(
            dim=self.config.hidden_size // self.config.num_heads,
            max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
            base=self.config.rope_theta,
        )

        self.layers = nn.ModuleList([URMBlock(self.config) for _ in range(self.config.num_layers)])

        self.init_hidden = nn.Buffer(
            trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1),
            persistent=True,
        )

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(
        self,
        input: torch.Tensor,
        puzzle_embed_idxs: torch.Tensor,
        puzzle_idxs: torch.Tensor = None,
        dih_idxs: torch.Tensor = None,
        colorperm_slots: torch.Tensor = None,
        per_aug_embed_idxs: dict = None,
    ):
        embedding = self.embed_tokens(input.to(torch.int32))

        if self.config.puzzle_emb_ndim > 0:
            batch_size = input.shape[0]
            device = input.device
            if self.config.use_smart_embed:
                if puzzle_idxs is None:
                    puzzle_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)

                if self.config.smart_embed_source in ("slotperm", "slotperm_full"):
                    if dih_idxs is None:
                        dih_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)
                    if colorperm_slots is None:
                        colorperm_slots = torch.arange(9, device=device).unsqueeze(0).expand(batch_size, -1)
                    puzzle_embedding = self.puzzle_emb(
                        puzzle_idxs.to(device),
                        dih_idxs.to(device),
                        colorperm_slots.to(device),
                        puzzle_embed_idxs.to(device),
                    )
                else:  # separated
                    per_aug_embed_idxs = per_aug_embed_idxs or {}
                    dih_idxs = per_aug_embed_idxs.get("dih", torch.zeros(batch_size, dtype=torch.long, device=device))
                    colorperm_idxs = per_aug_embed_idxs.get("colorperm", torch.zeros(batch_size, dtype=torch.long, device=device))
                    puzzle_embedding = self.puzzle_emb(
                        puzzle_idxs.to(device),
                        dih_idxs.to(device),
                        colorperm_idxs.to(device),
                        puzzle_embed_idxs.to(device),
                    )

                if self.config.smart_embed_strategy == "raw":
                    puzzle_embedding, dih_embedding, colorperm_embedding = puzzle_embedding
                    puzzle_embedding = torch.stack([puzzle_embedding, dih_embedding, colorperm_embedding], dim=1).to(self.forward_dtype)
                    puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 3))
                elif self.config.smart_embed_strategy == "raw_colorperm":
                    puzzle_emb, dih_emb, colorperm_emb = puzzle_embedding
                    puzzle_dih = torch.stack([puzzle_emb, dih_emb], dim=1).to(self.forward_dtype)
                    puzzle_embedding = torch.cat([puzzle_dih, colorperm_emb], dim=1).to(self.forward_dtype)
                    puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 11))
                else:
                    single_embedding = puzzle_embedding.to(self.forward_dtype)
                    puzzle_embedding = single_embedding.unsqueeze(1)
                    if self.puzzle_emb_len > 1:
                        puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 1))
            else:
                puzzle_embedding = self.puzzle_emb(puzzle_embed_idxs)
                single_embedding = puzzle_embedding.to(self.forward_dtype)
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)

            if self.config.seq_embed_strategy == "prepend":
                embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
            elif self.config.seq_embed_strategy == "add":
                embedding = torch.cat((torch.zeros((batch_size, self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype, device=device), embedding), dim=-2)
                embedding = embedding + single_embedding[:, None, :]
            else:
                raise ValueError(f"Invalid sequence embedding strategy: {self.config.seq_embed_strategy}")

        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> URMCarry:
        return URMCarry(
            current_hidden=torch.empty(
                batch_size,
                self.config.seq_len + self.puzzle_emb_len,
                self.config.hidden_size,
                dtype=self.forward_dtype,
            ),
        )

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        new_hidden = torch.where(
            reset_flag.view(-1, 1, 1),
            self.init_hidden,
            carry.current_hidden
        )
        return replace(carry, current_hidden=new_hidden)

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_embed_idxs"],
            puzzle_idxs=batch.get("puzzle_idxs"),
            dih_idxs=batch.get("dih_idxs"),
            colorperm_slots=batch.get("colorperm_slots"),
            per_aug_embed_idxs=batch.get("per_aug_embed_idxs"),
        )

        hidden_states = carry.current_hidden
        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range(self.config.H_cycles - 1):
                    for _ in range(self.config.L_cycles):
                        hidden_states = hidden_states + input_embeddings
                        for layer in self.layers:
                            hidden_states = layer(hidden_states=hidden_states, **seq_info)

        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1])


class URM(nn.Module):
    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = URMConfig(**config_dict)
        self.inner = URM_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        batch_size = batch["inputs"].shape[0]
        base = self.inner.empty_carry(batch_size)

        current_data = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                current_data[k] = torch.empty_like(v)
            elif isinstance(v, dict):
                current_data[k] = {dk: torch.empty_like(dv) for dk, dv in v.items()}

        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((batch_size,), dtype=torch.int32),
            halted=torch.ones((batch_size,), dtype=torch.bool),
            current_data=current_data,
        )

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
        compute_target_q=False
    ) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:

        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)

        new_current_data = {}
        for k, v in carry.current_data.items():
            if isinstance(v, dict):
                new_current_data[k] = {}
                for dk, dv in batch[k].items():
                    if dk in v:
                        new_current_data[k][dk] = torch.where(
                            carry.halted.view((-1,) + (1,) * (dv.ndim - 1)), dv, v[dk]
                        )
                    else:
                        new_current_data[k][dk] = dv
            else:
                new_current_data[k] = torch.where(
                    carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)),
                    batch[k],
                    v
                )

        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
        }

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and (self.config.loops > 1):
                halted = halted | (q_halt_logits > 0)

                halt_exploration_prob = 0.1
                min_halt_steps = (torch.rand_like(q_halt_logits) < halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return (
            URMCarry(
                current_hidden=new_carry.current_hidden,
                steps=new_steps,
                halted=halted,
                current_data=new_current_data,
            ),
            outputs,
        )
