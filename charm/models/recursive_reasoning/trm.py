from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
import math
import torch
import copy
import torch.nn.functional as F
from torch import nn
from pydantic import BaseModel
import random
from charm.models.common import trunc_normal_init_
from charm.models.layers import rms_norm, LinearSwish, SwiGLU, Attention, RotaryEmbedding, CosSin, CastedEmbedding, CastedLinear
from charm.models.sparse_embedding import CastedSparseEmbedding

IGNORE_LABEL_ID = -100

@dataclass
class TinyRecursiveReasoningModel_ACTV1InnerCarry:
    z_H: torch.Tensor
    z_L: torch.Tensor


@dataclass
class TinyRecursiveReasoningModel_ACTV1Carry:
    inner_carry: TinyRecursiveReasoningModel_ACTV1InnerCarry
    
    steps: torch.Tensor
    halted: torch.Tensor
    
    current_data: Dict[str, torch.Tensor]


class TinyRecursiveReasoningModel_ACTV1Config(BaseModel):
    batch_size: int
    seq_len: int
    puzzle_emb_ndim: int = 0
    puzzle_embed_vocab_size: int
    vocab_size: int

    H_cycles: int
    L_cycles: int

    H_layers: int # ignored
    L_layers: int

    # Transformer config
    hidden_size: int
    expansion: float
    num_heads: int
    pos_encodings: str

    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    
    # Halting Q-learning config
    halt_max_steps: int
    halt_exploration_prob: float

    forward_dtype: str = "bfloat16"

    # Alexia: added
    mlp_t: bool = False # use mlp on L instead of transformer
    puzzle_emb_len: int = 16 # if non-zero, its specified to this value
    no_ACT_continue: bool =  True # No continue ACT loss, only use the sigmoid of the halt which makes much more sense

    # Smart task embedding config
    use_smart_embed: bool = False
    smart_embed_strategy: str = "add"  # "add", "film", "film_concat", "film_pc", "mlp", "raw", "raw_colorperm", "slotperm_pc", "slotperm_moe", "hyper_aug"
    smart_embed_source: str = "slotperm"  # "slotperm", "slotperm_full" or "separated"
    num_puzzles: int = 1000  # number of unique puzzle IDs
    per_aug_vocab_sizes: dict = {}  # for separated mode: {"dih": 9, "colorperm": 1001}
    smart_embed_dim: int | None = None  # per-head embedding dim (defaults to hidden_size)
    smart_embed_heads: int = 1  # number of parallel heads
    smart_embed_rank: int | None = None  # low-rank size for film_pc/slotperm_pc/hyper_aug (defaults to full rank)
    smart_embed_moe_experts: int = 4  # experts for slotperm_moe
    puzzle_embed_dual: bool = False  # each puzzle has 2 embedding entries
    puzzle_embed_dual_mode: str = "random"  # "random" (select one with 50% prob) or "average" (mean of both)
    smart_embed_interaction_mode: str = "none"  # "none", "instance_residual", or "instance_residual_engram"
    smart_embed_interaction_rank: int | None = None  # low-rank size for interaction residual (None/full rank)
    smart_embed_interaction_gate: bool = False  # gate residual with semantic embedding

    # Sequence embedding config
    seq_embed_strategy: str = "prepend"  # "prepend", "add"

    # TRM embedding mode
    trm_embedding_mode: str = "default"  # "default", "untrainable", "bitvector", "lowrank"
    puzzle_emb_lowrank_dim: int = 8  # dimension for lowrank embedding mode


class UntrainablePuzzleEmbedding(nn.Module):
    """Fixed random embedding - same interface as CastedSparseEmbedding but not trainable."""
    def __init__(self, num_embeddings: int, embedding_dim: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.weights = nn.Buffer(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=1.0),
            persistent=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.weights[inputs].to(self.cast_to)


class BitvectorPuzzleEmbedding(nn.Module):
    """32-bit integer decomposed into binary, summing embeddings for set bits."""
    def __init__(self, hidden_size: int, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.bit_embeddings = nn.Buffer(
            trunc_normal_init_(torch.empty((32, hidden_size)), std=1.0),
            persistent=True
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        bit_mask = (1 << torch.arange(32, device=inputs.device)).unsqueeze(0)  # (1, 32)
        bits = (inputs.unsqueeze(-1) & bit_mask) != 0  # (B, 32) bool
        result = torch.matmul(bits.float(), self.bit_embeddings)  # (B, hidden_size)
        return result.to(self.cast_to)


class TinyRecursiveReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()

        self.config = config
        if self.config.mlp_t:
            self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size) if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len
            self.mlp_t = SwiGLU(
                hidden_size=self.config.seq_len + self.puzzle_emb_len, # L
                expansion=config.expansion,
            )
        else:
            self.self_attn = Attention(
                hidden_size=config.hidden_size,
                head_dim=config.hidden_size // config.num_heads,
                num_heads=config.num_heads,
                num_key_value_heads=config.num_heads,
                causal=False
            )
        self.mlp = SwiGLU(
            hidden_size=config.hidden_size,
            expansion=config.expansion,
        )
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # B, L, D = hidden_states.shape
        # Post Norm
        if self.config.mlp_t:
            hidden_states = hidden_states.transpose(1,2)
            out = self.mlp_t(hidden_states)
            hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
            hidden_states = hidden_states.transpose(1,2)
        else:
            # Self Attention
            hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states), variance_epsilon=self.norm_eps)
        # Fully Connected
        out = self.mlp(hidden_states)
        hidden_states = rms_norm(hidden_states + out, variance_epsilon=self.norm_eps)
        return hidden_states

class TinyRecursiveReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[TinyRecursiveReasoningModel_ACTV1Block]):
        super().__init__()
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states


class TinyRecursiveReasoningModel_ACTV1_Inner(nn.Module):
    def __init__(self, config: TinyRecursiveReasoningModel_ACTV1Config) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, self.config.forward_dtype)

        # I/O

        self.embed_scale = math.sqrt(self.config.hidden_size)
        embed_init_std = 1.0 / self.embed_scale

        self.embed_tokens = CastedEmbedding(self.config.vocab_size, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        self.lm_head      = CastedLinear(self.config.hidden_size, self.config.vocab_size, bias=False)
        self.q_head       = CastedLinear(self.config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(self.config.puzzle_emb_ndim // -self.config.hidden_size)  if self.config.puzzle_emb_len == 0 else self.config.puzzle_emb_len  # ceil div
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
                # Keep puzzle_emb_len configurable - will pad in _input_embeddings
            else:
                if self.config.trm_embedding_mode == "default":
                    # Zero init puzzle embeddings
                    self.puzzle_emb = CastedSparseEmbedding(self.config.puzzle_embed_vocab_size, self.config.puzzle_emb_ndim,
                                                            batch_size=self.config.batch_size, init_std=0, cast_to=self.forward_dtype)
                elif self.config.trm_embedding_mode == "untrainable":
                    self.puzzle_emb = UntrainablePuzzleEmbedding(
                        self.config.puzzle_embed_vocab_size,
                        self.config.puzzle_emb_ndim,
                        cast_to=self.forward_dtype
                    )
                elif self.config.trm_embedding_mode == "bitvector":
                    self.puzzle_emb = BitvectorPuzzleEmbedding(
                        self.config.hidden_size,
                        cast_to=self.forward_dtype
                    )
                elif self.config.trm_embedding_mode == "lowrank":
                    self.puzzle_emb = CastedSparseEmbedding(
                        self.config.puzzle_embed_vocab_size,
                        self.config.puzzle_emb_lowrank_dim,
                        batch_size=self.config.batch_size,
                        init_std=0,
                        cast_to=self.forward_dtype
                    )
                    self.puzzle_emb_up_proj = CastedLinear(
                        self.config.puzzle_emb_lowrank_dim,
                        self.config.puzzle_emb_ndim,
                        bias=False
                    )
                else:
                    raise ValueError(f"Invalid TRM embedding mode: {self.config.trm_embedding_mode}")

        # LM Blocks
        if self.config.pos_encodings == "rope":
            self.rotary_emb = RotaryEmbedding(dim=self.config.hidden_size // self.config.num_heads,
                                              max_position_embeddings=self.config.seq_len + self.puzzle_emb_len,
                                              base=self.config.rope_theta)
        elif self.config.pos_encodings == "learned":
            self.embed_pos = CastedEmbedding(self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, init_std=embed_init_std, cast_to=self.forward_dtype)
        else:
            pass

        # Reasoning Layers
        self.L_level = TinyRecursiveReasoningModel_ACTV1ReasoningModule(layers=[TinyRecursiveReasoningModel_ACTV1Block(self.config) for _i in range(self.config.L_layers)])

        # Initial states
        self.H_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        self.L_init = nn.Buffer(trunc_normal_init_(torch.empty(self.config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        # Q head special init
        # Init Q to (almost) zero for faster learning during bootstrapping
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)  # type: ignore

    def _input_embeddings(self, input: torch.Tensor, puzzle_embed_idxs: torch.Tensor,
                          puzzle_idxs: torch.Tensor = None, dih_idxs: torch.Tensor = None,
                          colorperm_slots: torch.Tensor = None, per_aug_embed_idxs: dict = None):
        # Token embedding
        embedding = self.embed_tokens(input.to(torch.int32))

        # Puzzle embeddings
        if self.config.puzzle_emb_ndim > 0:
            batch_size = input.shape[0]
            device = input.device
            if self.config.use_smart_embed:
                # puzzle_idxs from collate (for both slotperm and separated)
                if puzzle_idxs is None:
                    puzzle_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)

                if self.config.smart_embed_source in ("slotperm", "slotperm_full"):
                    # Use pre-parsed tensors from collate
                    if dih_idxs is None:
                        dih_idxs = torch.zeros(batch_size, dtype=torch.long, device=device)
                    if colorperm_slots is None:
                        # Identity slots: [0,1,2,3,4,5,6,7,8]
                        colorperm_slots = torch.arange(9, device=device).unsqueeze(0).expand(batch_size, -1)
                    puzzle_embedding = self.puzzle_emb(
                        puzzle_idxs.to(device),
                        dih_idxs.to(device),
                        colorperm_slots.to(device),
                        puzzle_embed_idxs.to(device),
                    )
                else:  # separated
                    # Use pre-computed indices from dataloader
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
                    single_embedding = None
                    # in this case, puzzle_embedding is a tuple of (puzzle_emb, dih_emb, colorperm_emb)
                    puzzle_embedding, dih_embedding, colorperm_embedding = puzzle_embedding
                    # we stack the 3 vectors of (B, D) to (B, 3, D)
                    puzzle_embedding = torch.stack([puzzle_embedding, dih_embedding, colorperm_embedding], dim=1).to(self.forward_dtype)
                    assert self.puzzle_emb_len >= 3, "puzzle_emb_len must be >= 3 for raw embedding strategy"
                    # still pad to puzzle_emb_len
                    puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 3))
                elif self.config.smart_embed_strategy == "raw_colorperm":
                    single_embedding = None
                    # puzzle_embedding is a tuple of (puzzle_emb, dih_emb, colorperm_emb)
                    # where colorperm_emb is (B, 9, D) - 9 separate full-size color embeddings
                    puzzle_emb, dih_emb, colorperm_emb = puzzle_embedding
                    # Stack puzzle and dih to (B, 2, D)
                    puzzle_dih = torch.stack([puzzle_emb, dih_emb], dim=1).to(self.forward_dtype)
                    # Concatenate with colorperm to get (B, 11, D)
                    puzzle_embedding = torch.cat([puzzle_dih, colorperm_emb], dim=1).to(self.forward_dtype)
                    assert self.puzzle_emb_len >= 11, "puzzle_emb_len must be >= 11 for raw_colorperm strategy"
                    # Pad to puzzle_emb_len
                    puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 11))
                else:
                    single_embedding = puzzle_embedding.to(self.forward_dtype)
                    # Smart embed outputs (B, D), pad to (B, puzzle_emb_len, D)
                    puzzle_embedding = single_embedding.unsqueeze(1)  # (B, 1, D)
                    if self.puzzle_emb_len > 1:
                        # Pad with zeros to match sparse method behavior
                        puzzle_embedding = F.pad(puzzle_embedding, (0, 0, 0, self.puzzle_emb_len - 1))
            else:
                puzzle_embedding = self.puzzle_emb(puzzle_embed_idxs)
                if hasattr(self, 'puzzle_emb_up_proj'):
                    puzzle_embedding = self.puzzle_emb_up_proj(puzzle_embedding)
                single_embedding = puzzle_embedding.to(self.forward_dtype)
                pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
                if pad_count > 0:
                    puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
                puzzle_embedding = puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size)

            if self.config.seq_embed_strategy == "prepend":
                embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
            elif self.config.seq_embed_strategy == "add":
                # prepend zeros to match expected sequence length
                embedding = torch.cat((torch.zeros((batch_size, self.puzzle_emb_len, self.config.hidden_size), dtype=self.forward_dtype, device=device), embedding), dim=-2)
                # add task embedding to each token
                embedding = embedding + single_embedding[:, None, :]
            else:
                raise ValueError(f"Invalid sequence embedding strategy: {self.config.seq_embed_strategy}")

        # Position embeddings
        if self.config.pos_encodings == "learned":
            # scale by 1/sqrt(2) to maintain forward variance
            embedding = 0.707106781 * (embedding + self.embed_pos.embedding_weight.to(self.forward_dtype))

        # Scale
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
            z_L=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype),
        )
        
    def reset_carry(self, reset_flag: torch.Tensor, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry):
        return TinyRecursiveReasoningModel_ACTV1InnerCarry(
            z_H=torch.where(reset_flag.view(-1, 1, 1), self.H_init, carry.z_H),
            z_L=torch.where(reset_flag.view(-1, 1, 1), self.L_init, carry.z_L),
        )

    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1InnerCarry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1InnerCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )

        # Input encoding
        input_embeddings = self._input_embeddings(
            batch["inputs"],
            batch["puzzle_embed_idxs"],
            puzzle_idxs=batch.get("puzzle_idxs"),
            dih_idxs=batch.get("dih_idxs"),
            colorperm_slots=batch.get("colorperm_slots"),
            per_aug_embed_idxs=batch.get("per_aug_embed_idxs"),
        )

        # Forward iterations
        it = 0
        z_H, z_L = carry.z_H, carry.z_L
        # H_cycles-1 without grad
        with torch.no_grad():
            for _H_step in range(self.config.H_cycles-1):
                for _L_step in range(self.config.L_cycles):
                    z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
                z_H = self.L_level(z_H, z_L, **seq_info)
        # 1 with grad
        for _L_step in range(self.config.L_cycles):
            z_L = self.L_level(z_L, z_H + input_embeddings, **seq_info)
        z_H = self.L_level(z_H, z_L, **seq_info)

        # LM Outputs
        new_carry = TinyRecursiveReasoningModel_ACTV1InnerCarry(z_H=z_H.detach(), z_L=z_L.detach())  # New carry no grad
        output = self.lm_head(z_H)[:, self.puzzle_emb_len:]
        q_logits = self.q_head(z_H[:, 0]).to(torch.float32) # Q-head; uses the first puzzle_emb position
        # Return z_H for optional diagnostics.
        return new_carry, output, (q_logits[..., 0], q_logits[..., 1]), z_H


class TinyRecursiveReasoningModel_ACTV1(nn.Module):
    """ACT wrapper."""

    def __init__(self, config_dict: dict):
        super().__init__()
        self.config = TinyRecursiveReasoningModel_ACTV1Config(**config_dict)
        self.inner = TinyRecursiveReasoningModel_ACTV1_Inner(self.config)

    @property
    def puzzle_emb(self):
        return self.inner.puzzle_emb

    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        batch_size = batch["inputs"].shape[0]

        # Only cache tensor and dict items; skip string lists (they cause torch.compile recompilation)
        current_data = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                current_data[k] = torch.empty_like(v)
            elif isinstance(v, dict):
                current_data[k] = {dk: torch.empty_like(dv) for dk, dv in v.items()}
            # Skip list items (puzzle_ids, unique_strs) - not needed for model forward

        return TinyRecursiveReasoningModel_ACTV1Carry(
            inner_carry=self.inner.empty_carry(batch_size),

            steps=torch.zeros((batch_size, ), dtype=torch.int32),
            halted=torch.ones((batch_size, ), dtype=torch.bool),

            current_data=current_data
        )
        
    def forward(self, carry: TinyRecursiveReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[TinyRecursiveReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:

        # Update data, carry (removing halted sequences)
        new_inner_carry = self.inner.reset_carry(carry.halted, carry.inner_carry)

        new_steps = torch.where(carry.halted, 0, carry.steps)

        # Handle tensor and dict items (list items are skipped in initial_carry)
        new_current_data = {}
        for k, v in carry.current_data.items():
            if isinstance(v, dict):
                # For dict items (per_aug_embed_idxs), handle each sub-tensor
                new_current_data[k] = {}
                for dk, dv in batch[k].items():
                    if dk in v:
                        new_current_data[k][dk] = torch.where(
                            carry.halted.view((-1, ) + (1, ) * (dv.ndim - 1)), dv, v[dk]
                        )
                    else:
                        new_current_data[k][dk] = dv
            else:
                # Tensor items
                new_current_data[k] = torch.where(carry.halted.view((-1, ) + (1, ) * (batch[k].ndim - 1)), batch[k], v)

        # Forward inner model
        new_inner_carry, logits, (q_halt_logits, q_continue_logits), z_H = self.inner(new_inner_carry, new_current_data)

        outputs = {
            "logits": logits,
            "q_halt_logits": q_halt_logits,
            "q_continue_logits": q_continue_logits,
            "z_H": z_H,
        }

        with torch.no_grad():
            # Step
            new_steps = new_steps + 1
            is_last_step = new_steps >= self.config.halt_max_steps
            
            halted = is_last_step

            # if training, and ACT is enabled
            if self.training and (self.config.halt_max_steps > 1):

                # Halt signal
                # NOTE: During evaluation, always use max steps, this is to guarantee the same halting steps inside a batch for batching purposes
                
                if self.config.no_ACT_continue:
                    halted = halted | (q_halt_logits > 0)
                else:
                    halted = halted | (q_halt_logits > q_continue_logits)

                # Exploration
                min_halt_steps = (torch.rand_like(q_halt_logits) < self.config.halt_exploration_prob) * torch.randint_like(new_steps, low=2, high=self.config.halt_max_steps + 1)
                halted = halted & (new_steps >= min_halt_steps)

                if not self.config.no_ACT_continue:
                    # Compute target Q
                    # NOTE: No replay buffer and target networks for computing target Q-value.
                    # As batch_size is large, there're many parallel envs.
                    # Similar concept as PQN https://arxiv.org/abs/2407.04811
                    _, _, (next_q_halt_logits, next_q_continue_logits), _ = self.inner(new_inner_carry, new_current_data)
                    outputs["target_q_continue"] = torch.sigmoid(torch.where(is_last_step, next_q_halt_logits, torch.maximum(next_q_halt_logits, next_q_continue_logits)))

        return TinyRecursiveReasoningModel_ACTV1Carry(new_inner_carry, new_steps, halted, new_current_data), outputs
