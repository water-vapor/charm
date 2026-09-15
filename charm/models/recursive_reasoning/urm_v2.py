from typing import Tuple, Dict, Optional
from dataclasses import dataclass, replace
import math
import torch
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
from charm.models.common import trunc_normal_init_
from charm.models.layers import rms_norm, SwiGLU, ConvSwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
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
    mlp_type: str = "swiglu"  # "swiglu" or "convswiglu"
    pos_encodings: str = "rope"  # "rope" or "rope_2d"
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    loops: int
    early_stop_training: bool = True  # whether q head can trigger early halt during training
    input_injection: str = "recurrent"  # "layer", "recurrent", "loop", "sample"
    recurrent_steps: int   # total steps per forward call
    tbptt_steps: int       # steps with gradients at the end (recurrent-level granularity)
    tbptt_layer_steps: int | None = None  # when set, use layer-level granularity (ignores tbptt_steps)
    forward_dtype: str = "bfloat16"
    puzzle_emb_len: int = 0

    # Smart task embedding config
    use_smart_embed: bool = False
    smart_embed_strategy: str = "add"
    smart_embed_source: str = "slotperm"
    num_puzzles: int = 1000
    per_aug_vocab_sizes: dict = {}
    smart_embed_dim: int | None = None
    smart_embed_task_dim: int | None = None
    smart_embed_heads: int = 1
    smart_embed_rank: int | None = None
    smart_embed_moe_experts: int = 4
    seq_embed_strategy: str = "prepend"
    puzzle_embed_dual: bool = False
    puzzle_embed_dual_mode: str = "random"
    smart_embed_interaction_mode: str = "none"  # "none", "instance_residual", or "instance_residual_engram"
    smart_embed_interaction_rank: int | None = None  # low-rank size for interaction residual (None/full rank)
    smart_embed_interaction_gate: bool = False  # gate residual with semantic embedding (instance_residual only)
    trm_embedding_mode: str = "default"  # "default" or "lowrank"
    puzzle_emb_lowrank_dim: int = 32  # dimension for lowrank embedding mode


class RotaryEmbedding2D(nn.Module):
    """
    Axial 2D RoPE for flattened row-major grids, with optional unrotated prefix tokens.

    This matches the half-split `rotate_half` convention used by `apply_rotary_pos_emb`.
    """
    def __init__(
        self,
        dim: int,
        seq_len: int,
        base: float,
        no_rope: int = 0,
        device=None,
    ):
        super().__init__()

        if dim % 4 != 0:
            raise ValueError(f"2D RoPE requires head dim divisible by 4, got {dim}")

        side = int(math.isqrt(seq_len))
        if side * side != seq_len:
            raise ValueError(
                f"2D RoPE requires square token grid, got seq_len={seq_len}"
            )

        axis_dim = dim // 2
        inv_freq = 1.0 / (
            base ** (torch.arange(0, axis_dim, 2, dtype=torch.float32, device=device) / axis_dim)
        )

        pos = torch.arange(seq_len, dtype=torch.float32, device=device)
        row_pos = torch.div(pos, side, rounding_mode="floor")
        col_pos = torch.remainder(pos, side)
        row_freqs = torch.outer(row_pos, inv_freq)
        col_freqs = torch.outer(col_pos, inv_freq)

        # First half encodes row+col channels, second half mirrors it for rotate_half().
        freqs = torch.cat((row_freqs, col_freqs), dim=-1)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos()
        sin = emb.sin()

        if no_rope > 0:
            prefix_shape = (no_rope, dim)
            cos = torch.cat((torch.ones(prefix_shape, dtype=cos.dtype, device=cos.device), cos), dim=0)
            sin = torch.cat((torch.zeros(prefix_shape, dtype=sin.dtype, device=sin.device), sin), dim=0)

        self.cos_cached = nn.Buffer(cos, persistent=False)
        self.sin_cached = nn.Buffer(sin, persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


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
        mlp_cls = ConvSwiGLU if config.mlp_type == "convswiglu" else SwiGLU
        self.mlp = mlp_cls(
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
        self.q_head = CastedLinear(self.config.hidden_size, 1, bias=True)
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
                    smart_embed_task_dim=self.config.smart_embed_task_dim,
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
                if self.config.trm_embedding_mode == "default":
                    self.puzzle_emb = CastedSparseEmbedding(
                        self.config.puzzle_embed_vocab_size,
                        self.config.puzzle_emb_ndim,
                        batch_size=self.config.batch_size,
                        init_std=0,
                        cast_to=self.forward_dtype,
                    )
                elif self.config.trm_embedding_mode == "lowrank":
                    self.puzzle_emb = CastedSparseEmbedding(
                        self.config.puzzle_embed_vocab_size,
                        self.config.puzzle_emb_lowrank_dim,
                        batch_size=self.config.batch_size,
                        init_std=0,
                        cast_to=self.forward_dtype,
                    )
                    self.puzzle_emb_up_proj = CastedLinear(
                        self.config.puzzle_emb_lowrank_dim,
                        self.config.puzzle_emb_ndim,
                        bias=False,
                    )
                else:
                    raise ValueError(f"Invalid TRM embedding mode: {self.config.trm_embedding_mode}")

        head_dim = self.config.hidden_size // self.config.num_heads
        pos_mode = self.config.pos_encodings.lower()
        if pos_mode == "rope":
            self.rotary_emb = RotaryEmbedding(
                dim=head_dim,
                max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                base=self.config.rope_theta,
            )
        elif pos_mode == "rope_2d":
            self.rotary_emb = RotaryEmbedding2D(
                dim=head_dim,
                seq_len=self.config.seq_len,
                base=self.config.rope_theta,
                no_rope=self.puzzle_emb_len,
            )
        else:
            raise ValueError(
                f"Unsupported pos_encodings '{self.config.pos_encodings}'. "
                "Expected 'rope' or 'rope_2d'."
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
                if self.config.trm_embedding_mode == "lowrank":
                    puzzle_embedding = self.puzzle_emb_up_proj(puzzle_embedding)
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

    def _forward_recurrent_tbptt(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        injection = self.config.input_injection

        if injection == "loop":
            hidden_states = hidden_states + input_embeddings

        warmup_steps = self.config.recurrent_steps - self.config.tbptt_steps
        if warmup_steps > 0:
            with torch.no_grad():
                for _ in range(warmup_steps):
                    if injection == "recurrent":
                        hidden_states = hidden_states + input_embeddings
                    for layer in self.layers:
                        if injection == "layer":
                            hidden_states = hidden_states + input_embeddings
                        hidden_states = layer(hidden_states=hidden_states, **seq_info)

        for _ in range(warmup_steps, self.config.recurrent_steps):
            if injection == "recurrent":
                hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                if injection == "layer":
                    hidden_states = hidden_states + input_embeddings
                hidden_states = layer(hidden_states=hidden_states, **seq_info)

        return hidden_states, {}

    def _forward_layer_tbptt(
        self,
        hidden_states: torch.Tensor,
        input_embeddings: torch.Tensor,
        seq_info: dict,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        injection = self.config.input_injection
        num_layers = self.config.num_layers
        total_blocks = self.config.recurrent_steps * num_layers
        assert num_layers <= self.config.tbptt_layer_steps <= total_blocks, \
            f"tbptt_layer_steps ({self.config.tbptt_layer_steps}) must be in [{num_layers}, {total_blocks}]"
        warmup_blocks = total_blocks - self.config.tbptt_layer_steps

        if warmup_blocks > 0:
            with torch.no_grad():
                for block_idx in range(warmup_blocks):
                    layer_idx = block_idx % num_layers
                    if injection == "loop" and block_idx == 0:
                        hidden_states = hidden_states + input_embeddings
                    elif injection == "recurrent" and layer_idx == 0:
                        hidden_states = hidden_states + input_embeddings
                    elif injection == "layer":
                        hidden_states = hidden_states + input_embeddings
                    hidden_states = self.layers[layer_idx](hidden_states=hidden_states, **seq_info)

        for block_idx in range(warmup_blocks, total_blocks):
            layer_idx = block_idx % num_layers
            if injection == "loop" and block_idx == 0:
                hidden_states = hidden_states + input_embeddings
            elif injection == "recurrent" and layer_idx == 0:
                hidden_states = hidden_states + input_embeddings
            elif injection == "layer":
                hidden_states = hidden_states + input_embeddings
            hidden_states = self.layers[layer_idx](hidden_states=hidden_states, **seq_info)

        return hidden_states, {}

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[URMCarry, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
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

        if self.config.tbptt_layer_steps is not None:
            hidden_states, aux_outputs = self._forward_layer_tbptt(
                hidden_states,
                input_embeddings,
                seq_info,
            )
        else:
            hidden_states, aux_outputs = self._forward_recurrent_tbptt(
                hidden_states,
                input_embeddings,
                seq_info,
            )

        new_carry = replace(carry, current_hidden=hidden_states.detach())
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len:]
        q_halt_logits = self.q_head(hidden_states[:, 0]).to(torch.float32).squeeze(-1)
        return new_carry, output, q_halt_logits, aux_outputs


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

        new_carry, logits, q_halt_logits, aux_outputs = self.inner(new_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
        }
        outputs.update(aux_outputs)

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = (new_steps >= self.config.loops)

            if self.training and self.config.early_stop_training and (self.config.loops > 1):
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
