import math

import torch
from torch import nn

from charm.models.sparse_embedding import CastedSparseEmbedding

SPARSE_COLORPERM_THRESHOLD = 1


class PCFiLMHead(nn.Module):
    def __init__(self, dim: int, rank: int) -> None:
        super().__init__()
        self.pc_p = nn.Linear(dim, rank, bias=False)
        self.pc_a = nn.Linear(dim * 2, rank, bias=False)
        self.pc_a2 = nn.Linear(dim * 2, dim)
        self.pc_gamma = nn.Linear(rank, dim)
        self.pc_beta = nn.Linear(rank, dim)

    def forward(self, puzzle_emb: torch.Tensor, aug_emb: torch.Tensor) -> torch.Tensor:
        gate = self.pc_p(puzzle_emb) * self.pc_a(aug_emb)
        gamma = self.pc_gamma(gate)
        beta = self.pc_beta(gate)
        return (1 + gamma) * puzzle_emb + beta + self.pc_a2(aug_emb)


class SmartTaskEmbedding(nn.Module):
    def __init__(
        self,
        num_puzzles: int,
        hidden_size: int,
        combine_strategy: str = "add",
        embed_source: str = "slotperm",
        per_aug_vocab_sizes: dict[str, int] | None = None,
        smart_embed_dim: int | None = None,
        smart_embed_heads: int = 1,
        smart_embed_rank: int | None = None,
        smart_embed_moe_experts: int = 4,
        batch_size: int | None = None,
        forward_dtype: torch.dtype = torch.bfloat16,
        puzzle_embed_dual: bool = False,
        puzzle_embed_dual_mode: str = "random",
        interaction_mode: str = "none",
        interaction_rank: int | None = None,
        interaction_use_gate: bool = False,
        num_instance_embeddings: int | None = None,
        smart_embed_task_dim: int | None = None,
    ):
        """
        Args:
            num_puzzles: number of unique puzzle IDs
            hidden_size: embedding dimension (model hidden size)
            combine_strategy: "add", "film", "film_concat", "film_pc", "mlp", "raw", "raw_colorperm",
                "slotperm_pc", "slotperm_moe", or "hyper_aug"
            embed_source: "slotperm" (parse strings, slot colorperm) or "separated" (dataloader indices)
            per_aug_vocab_sizes: vocab sizes for each aug type (required for separated mode)
            smart_embed_dim: per-head embedding dimension (defaults to hidden_size)
            smart_embed_task_dim: task-ID table width only (defaults to heads * per-head
                dimension); projected to the composition width before combining with augmentations
            smart_embed_heads: number of parallel heads (defaults to 1)
            smart_embed_rank: low-rank size for film_pc (defaults to full rank)
            batch_size: batch size (required for sparse colorperm embedding)
            forward_dtype: dtype for sparse embedding cast
            puzzle_embed_dual: if True, each puzzle has 2 embedding entries
            puzzle_embed_dual_mode: "random" (select one with 50% prob) or "average" (mean of both)
            interaction_mode: "none", "instance_residual", or "instance_residual_engram"
            interaction_rank: residual rank (<=0 or None means full hidden_size rank)
            interaction_use_gate: apply sigmoid gate from semantic embedding to residual path
                (instance_residual_engram has its own query-key gate)
            num_instance_embeddings: vocab size for instance residual lookup (puzzle_embed_idx space)
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.combine_strategy = combine_strategy
        self.embed_source = embed_source
        self.smart_embed_heads = smart_embed_heads
        self.head_dim = smart_embed_dim if smart_embed_dim is not None else hidden_size
        self.total_dim = self.head_dim * self.smart_embed_heads
        self.task_dim = self.total_dim if smart_embed_task_dim is None else smart_embed_task_dim
        if self.task_dim < 1:
            raise ValueError("smart_embed_task_dim must be >= 1")
        self.smart_embed_rank = smart_embed_rank
        self.smart_embed_moe_experts = smart_embed_moe_experts
        self.interaction_mode = interaction_mode
        self.interaction_rank = interaction_rank if interaction_rank is not None and interaction_rank > 0 else hidden_size
        self.interaction_use_gate = interaction_use_gate

        if self.smart_embed_heads < 1:
            raise ValueError("smart_embed_heads must be >= 1")
        if self.interaction_mode not in ("none", "instance_residual", "instance_residual_engram"):
            raise ValueError(f"Unknown interaction_mode: {self.interaction_mode}")

        if self.total_dim == hidden_size:
            self.final_proj = nn.Identity()
        else:
            self.final_proj = nn.Linear(self.total_dim, hidden_size)

        # Dual embedding config
        self.puzzle_embed_dual = puzzle_embed_dual
        self.puzzle_embed_dual_mode = puzzle_embed_dual_mode
        self._num_puzzles_base = num_puzzles

        # Puzzle ID embedding (double size if dual mode)
        actual_num_puzzles = num_puzzles * 2 if puzzle_embed_dual else num_puzzles
        self.puzzle_embed = nn.Embedding(actual_num_puzzles, self.task_dim)
        nn.init.zeros_(self.puzzle_embed.weight)
        # Identity preserves the original parameters and initialization when unset.
        self.puzzle_embed_up_proj = (
            nn.Identity() if self.task_dim == self.total_dim
            else nn.Linear(self.task_dim, self.total_dim, bias=False)
        )

        self.use_sparse_colorperm = False  # default, may be overridden in separated branch
        self.use_sparse_instance_residual = False

        if embed_source in ("slotperm", "slotperm_full"):
            # Dihedral embedding: 8 values (D4 group elements 0-7)
            self.dih_embed = nn.Embedding(8, self.total_dim)
            nn.init.zeros_(self.dih_embed.weight)

            # Color permutation embedding: slot-based
            self.slot_dim = self.head_dim // 10
            self.base_dim = self.head_dim - 9 * self.slot_dim
            self.colorperm_base = nn.Parameter(torch.zeros(self.smart_embed_heads, self.base_dim))
            self.colorperm_slots = nn.Parameter(torch.zeros(self.smart_embed_heads, 9, self.slot_dim))
            if embed_source == "slotperm_full":
                self.dih_slot_dim = self.head_dim // 9
                self.dih_base_dim = self.head_dim - 8 * self.dih_slot_dim
                self.dih_base = nn.Parameter(torch.zeros(self.smart_embed_heads, self.dih_base_dim))
                self.dih_slots = nn.Parameter(torch.zeros(self.smart_embed_heads, 8, self.dih_slot_dim))
        elif embed_source == "separated":
            per_aug_vocab_sizes = per_aug_vocab_sizes or {}
            dih_vocab = per_aug_vocab_sizes.get("dih", 9)
            self.dih_embed = nn.Embedding(dih_vocab, self.total_dim)
            nn.init.zeros_(self.dih_embed.weight)

            colorperm_vocab = per_aug_vocab_sizes.get("colorperm", 1)
            self.use_sparse_colorperm = colorperm_vocab >= SPARSE_COLORPERM_THRESHOLD
            if self.use_sparse_colorperm:
                assert batch_size is not None, "batch_size required for sparse colorperm embedding"
                self.colorperm_embed = CastedSparseEmbedding(
                    colorperm_vocab, self.total_dim,
                    batch_size=batch_size, init_std=0, cast_to=forward_dtype
                )
            else:
                self.colorperm_embed = nn.Embedding(colorperm_vocab, self.total_dim)
                nn.init.zeros_(self.colorperm_embed.weight)
        else:
            raise ValueError(f"Unknown embed_source: {embed_source}")

        if combine_strategy == "raw_colorperm":
            self.colorperm_full_slots = nn.Parameter(torch.zeros(9, self.total_dim))

        if combine_strategy == "film":
            if self.smart_embed_heads == 1:
                self.film_gamma = nn.Linear(self.head_dim, self.head_dim)
                self.film_beta = nn.Linear(self.head_dim, self.head_dim)
            else:
                self.film_gamma = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.smart_embed_heads)])
                self.film_beta = nn.ModuleList([nn.Linear(self.head_dim, self.head_dim) for _ in range(self.smart_embed_heads)])
        elif combine_strategy == "film_concat":
            if self.smart_embed_heads == 1:
                self.film_gamma = nn.Linear(self.head_dim * 2, self.head_dim)
                self.film_beta = nn.Linear(self.head_dim * 2, self.head_dim)
            else:
                self.film_gamma = nn.ModuleList([nn.Linear(self.head_dim * 2, self.head_dim) for _ in range(self.smart_embed_heads)])
                self.film_beta = nn.ModuleList([nn.Linear(self.head_dim * 2, self.head_dim) for _ in range(self.smart_embed_heads)])
        elif combine_strategy == "film_pc":
            pc_rank = self.smart_embed_rank if self.smart_embed_rank is not None and self.smart_embed_rank > 0 else self.head_dim
            pc_rank = min(pc_rank, self.head_dim)
            if self.smart_embed_heads == 1:
                self.pc_film = PCFiLMHead(self.head_dim, pc_rank)
            else:
                self.pc_film = nn.ModuleList([PCFiLMHead(self.head_dim, pc_rank) for _ in range(self.smart_embed_heads)])
        elif combine_strategy == "slotperm_pc":
            if embed_source not in ("slotperm", "slotperm_full"):
                raise ValueError("slotperm_pc requires embed_source 'slotperm' or 'slotperm_full'")
            pc_rank = self.smart_embed_rank if self.smart_embed_rank is not None and self.smart_embed_rank > 0 else self.head_dim
            pc_rank = min(pc_rank, self.head_dim)
            if self.smart_embed_heads == 1:
                self.slot_pc_gate = nn.Linear(self.head_dim, pc_rank, bias=False)
            else:
                self.slot_pc_gate = nn.ModuleList([nn.Linear(self.head_dim, pc_rank, bias=False) for _ in range(self.smart_embed_heads)])
            self.slot_pc_base_proj = nn.Parameter(torch.zeros(self.smart_embed_heads, pc_rank, self.base_dim))
            self.slot_pc_slot_proj = nn.Parameter(torch.zeros(self.smart_embed_heads, 9, pc_rank, self.slot_dim))
        elif combine_strategy == "slotperm_moe":
            if embed_source not in ("slotperm", "slotperm_full"):
                raise ValueError("slotperm_moe requires embed_source 'slotperm' or 'slotperm_full'")
            experts = max(1, int(self.smart_embed_moe_experts))
            self.smart_embed_moe_experts = experts
            self.moe_gate = nn.Linear(self.total_dim, experts)
            self.moe_base = nn.Parameter(torch.zeros(experts, self.smart_embed_heads, self.base_dim))
            self.moe_slots = nn.Parameter(torch.zeros(experts, self.smart_embed_heads, 9, self.slot_dim))
        elif combine_strategy == "hyper_aug":
            hyper_rank = self.smart_embed_rank if self.smart_embed_rank is not None and self.smart_embed_rank > 0 else self.head_dim
            hyper_rank = min(hyper_rank, self.head_dim)
            if self.smart_embed_heads == 1:
                self.hyper_gate = nn.Linear(self.head_dim, hyper_rank, bias=False)
            else:
                self.hyper_gate = nn.ModuleList([nn.Linear(self.head_dim, hyper_rank, bias=False) for _ in range(self.smart_embed_heads)])
            self.hyper_u = nn.Parameter(torch.zeros(self.smart_embed_heads, hyper_rank, self.head_dim))
            self.hyper_v = nn.Parameter(torch.zeros(self.smart_embed_heads, self.head_dim, hyper_rank))
        elif combine_strategy == "mlp":
            if self.smart_embed_heads == 1:
                self.combine_mlp = nn.Sequential(
                    nn.Linear(self.head_dim * 3, self.head_dim * 2),
                    nn.SiLU(),
                    nn.Linear(self.head_dim * 2, self.head_dim),
                )
            else:
                self.combine_mlp = nn.ModuleList([
                    nn.Sequential(
                        nn.Linear(self.head_dim * 3, self.head_dim * 2),
                        nn.SiLU(),
                        nn.Linear(self.head_dim * 2, self.head_dim),
                    )
                    for _ in range(self.smart_embed_heads)
                ])

        if self.interaction_mode == "instance_residual" and self.interaction_use_gate:
            self.interaction_gate = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.interaction_gate.weight)
            nn.init.constant_(self.interaction_gate.bias, 2.0)

        # Runtime telemetry for W&B logging (updated each forward call).
        self.last_gate_mean = nn.Buffer(torch.tensor(float("nan"), dtype=torch.float32), persistent=False)
        self.last_gate_active_ratio = nn.Buffer(torch.tensor(float("nan"), dtype=torch.float32), persistent=False)

        if self.interaction_mode in ("instance_residual", "instance_residual_engram"):
            if num_instance_embeddings is None or num_instance_embeddings <= 0:
                raise ValueError(
                    "num_instance_embeddings must be set for interaction_mode "
                    "in {'instance_residual', 'instance_residual_engram'}"
                )
            residual_rank = min(self.interaction_rank, hidden_size)
            self.instance_residual_rank = residual_rank
            assert batch_size is not None, "batch_size required for sparse instance residual embedding"
            self.instance_residual_embed = CastedSparseEmbedding(
                num_instance_embeddings,
                residual_rank,
                batch_size=batch_size,
                init_std=0,
                cast_to=forward_dtype,
            )
            self.use_sparse_instance_residual = True
            if self.interaction_mode == "instance_residual" and residual_rank != hidden_size:
                self.instance_residual_up_proj = nn.Linear(residual_rank, hidden_size, bias=False)
            if self.interaction_mode == "instance_residual_engram":
                self.instance_engram_key_proj = nn.Linear(residual_rank, hidden_size, bias=False)
                self.instance_engram_value_proj = nn.Linear(residual_rank, hidden_size, bias=False)
                self.instance_engram_query_norm = nn.RMSNorm(hidden_size)
                self.instance_engram_key_norm = nn.RMSNorm(hidden_size)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 3:
            return x
        return x.view(x.shape[0], self.smart_embed_heads, self.head_dim)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            merged = x
        elif self.smart_embed_heads == 1:
            merged = x[:, 0, :]
        else:
            merged = x.reshape(x.shape[0], self.total_dim)
        return self.final_proj(merged)

    def _merge_colorperm_heads(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            batch_size, slots, heads, dim = x.shape
            merged = x.reshape(batch_size, slots, heads * dim)
        else:
            batch_size, slots, _ = x.shape
            merged = x
        merged = merged.reshape(batch_size * slots, -1)
        merged = self.final_proj(merged)
        return merged.reshape(batch_size, slots, self.hidden_size)

    def _apply_head_modules(self, modules: nn.Module | nn.ModuleList, x: torch.Tensor) -> torch.Tensor:
        if self.smart_embed_heads == 1:
            return modules(x.squeeze(1)).unsqueeze(1)
        return torch.stack([modules[i](x[:, i]) for i in range(self.smart_embed_heads)], dim=1)

    def _get_colorperm_embedding(self, slot_indices: torch.Tensor) -> torch.Tensor:
        batch_size = slot_indices.shape[0]
        slot_indices = slot_indices.long()

        if self.slot_dim == 0:
            base = self.colorperm_base.unsqueeze(0).expand(batch_size, -1, -1)
            return base

        slots = self.colorperm_slots.unsqueeze(0).expand(batch_size, -1, -1, -1)
        idx = slot_indices.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.smart_embed_heads, 9, self.slot_dim)
        permuted_slots = torch.gather(slots, 2, idx)
        permuted_slots = permuted_slots.flatten(start_dim=2)

        base_expanded = self.colorperm_base.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([base_expanded, permuted_slots], dim=-1)

    def _get_colorperm_embedding_pc(self, slot_indices: torch.Tensor, puzzle_emb: torch.Tensor) -> torch.Tensor:
        batch_size = slot_indices.shape[0]
        slot_indices = slot_indices.long()

        puzzle_heads = self._split_heads(puzzle_emb)
        gate = self._apply_head_modules(self.slot_pc_gate, puzzle_heads)

        base_delta = torch.einsum("bhr,hrd->bhd", gate, self.slot_pc_base_proj)
        base = self.colorperm_base.unsqueeze(0).expand(batch_size, -1, -1) + base_delta

        if self.slot_dim == 0:
            return base

        slots = self.colorperm_slots.unsqueeze(0).expand(batch_size, -1, -1, -1)
        slot_delta = torch.einsum("bhr,hjrd->bhjd", gate, self.slot_pc_slot_proj)
        slots = slots + slot_delta

        idx = slot_indices.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.smart_embed_heads, 9, self.slot_dim)
        permuted_slots = torch.gather(slots, 2, idx)
        permuted_slots = permuted_slots.flatten(start_dim=2)

        return torch.cat([base, permuted_slots], dim=-1)

    def _get_colorperm_embedding_moe(self, slot_indices: torch.Tensor, puzzle_emb: torch.Tensor) -> torch.Tensor:
        batch_size = slot_indices.shape[0]
        slot_indices = slot_indices.long()

        alpha = torch.softmax(self.moe_gate(puzzle_emb), dim=-1)
        base = torch.einsum("bk,khd->bhd", alpha, self.moe_base)

        if self.slot_dim == 0:
            return base

        slots = torch.einsum("bk,khjd->bhjd", alpha, self.moe_slots)
        idx = slot_indices.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.smart_embed_heads, 9, self.slot_dim)
        permuted_slots = torch.gather(slots, 2, idx)
        permuted_slots = permuted_slots.flatten(start_dim=2)

        return torch.cat([base, permuted_slots], dim=-1)

    def _get_colorperm_full_embedding(self, slot_indices: torch.Tensor) -> torch.Tensor:
        return self.colorperm_full_slots[slot_indices.long()]

    def _get_dih_embedding(self, dih_indices: torch.Tensor) -> torch.Tensor:
        D4_PERM_TABLE = [
            [0, 1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 0, 1],
            [4, 5, 6, 7, 0, 1, 2, 3],
            [6, 7, 0, 1, 2, 3, 4, 5],
            [0, 7, 6, 5, 4, 3, 2, 1],
            [4, 3, 2, 1, 0, 7, 6, 5],
            [6, 5, 4, 3, 2, 1, 0, 7],
            [2, 1, 0, 7, 6, 5, 4, 3],
        ]

        batch_size = dih_indices.shape[0]
        device = dih_indices.device

        perm_table = torch.tensor(D4_PERM_TABLE, device=device)
        slot_indices = perm_table[dih_indices.long()]

        if self.dih_slot_dim == 0:
            base = self.dih_base.unsqueeze(0).expand(batch_size, -1, -1)
            return base

        slots = self.dih_slots.unsqueeze(0).expand(batch_size, -1, -1, -1)
        idx = slot_indices.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.smart_embed_heads, 8, self.dih_slot_dim)
        permuted_slots = torch.gather(slots, 2, idx)
        permuted_slots = permuted_slots.flatten(start_dim=2)

        base_expanded = self.dih_base.unsqueeze(0).expand(batch_size, -1, -1)
        return torch.cat([base_expanded, permuted_slots], dim=-1)

    def _combine(
        self,
        puzzle_emb: torch.Tensor,
        dih_emb: torch.Tensor,
        colorperm_emb: torch.Tensor,
    ) -> torch.Tensor:
        puzzle_emb = self._split_heads(puzzle_emb)
        dih_emb = self._split_heads(dih_emb)

        if self.combine_strategy == "raw":
            colorperm_emb = self._split_heads(colorperm_emb)
            return (
                self._merge_heads(puzzle_emb),
                self._merge_heads(dih_emb),
                self._merge_heads(colorperm_emb),
            )

        if self.combine_strategy == "raw_colorperm":
            return (
                self._merge_heads(puzzle_emb),
                self._merge_heads(dih_emb),
                self._merge_colorperm_heads(colorperm_emb),
            )

        colorperm_emb = self._split_heads(colorperm_emb)

        if self.combine_strategy in ("add", "slotperm_pc", "slotperm_moe"):
            combined = puzzle_emb + dih_emb + colorperm_emb
            return self._merge_heads(combined)

        if self.combine_strategy == "film":
            aug_emb = dih_emb + colorperm_emb
            gamma = self._apply_head_modules(self.film_gamma, aug_emb)
            beta = self._apply_head_modules(self.film_beta, aug_emb)
            combined = (1 + gamma) * puzzle_emb + beta
            return self._merge_heads(combined)

        if self.combine_strategy == "film_concat":
            aug_emb = torch.cat([dih_emb, colorperm_emb], dim=-1)
            gamma = self._apply_head_modules(self.film_gamma, aug_emb)
            beta = self._apply_head_modules(self.film_beta, aug_emb)
            combined = (1 + gamma) * puzzle_emb + beta
            return self._merge_heads(combined)

        if self.combine_strategy == "film_pc":
            aug_emb = torch.cat([dih_emb, colorperm_emb], dim=-1)
            if self.smart_embed_heads == 1:
                combined = self.pc_film(puzzle_emb[:, 0], aug_emb[:, 0]).unsqueeze(1)
            else:
                combined = torch.stack(
                    [self.pc_film[i](puzzle_emb[:, i], aug_emb[:, i]) for i in range(self.smart_embed_heads)],
                    dim=1,
                )
            return self._merge_heads(combined)

        if self.combine_strategy == "hyper_aug":
            aug_emb = dih_emb + colorperm_emb
            gate = self._apply_head_modules(self.hyper_gate, puzzle_emb)
            proj = torch.einsum("bhd,hdr->bhr", aug_emb, self.hyper_v)
            proj = proj * gate
            aug_hyper = torch.einsum("bhr,hrd->bhd", proj, self.hyper_u)
            combined = puzzle_emb + aug_emb + aug_hyper
            return self._merge_heads(combined)

        if self.combine_strategy == "mlp":
            combined = torch.cat([puzzle_emb, dih_emb, colorperm_emb], dim=-1)
            combined = self._apply_head_modules(self.combine_mlp, combined)
            return self._merge_heads(combined)

        raise ValueError(f"Unknown combine_strategy: {self.combine_strategy}")

    def _apply_interaction_residual(
        self,
        combined: torch.Tensor | tuple[torch.Tensor, ...],
        puzzle_emb: torch.Tensor,
        dih_emb: torch.Tensor,
        colorperm_emb: torch.Tensor,
        puzzle_idxs: torch.Tensor,
        dih_idxs: torch.Tensor,
        colorperm_input: torch.Tensor,
        puzzle_embed_idxs: torch.Tensor | None,
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.interaction_mode == "none":
            with torch.no_grad():
                self.last_gate_mean.fill_(float("nan"))
                self.last_gate_active_ratio.fill_(float("nan"))
            return combined
        if not torch.is_tensor(combined):
            with torch.no_grad():
                self.last_gate_mean.fill_(float("nan"))
                self.last_gate_active_ratio.fill_(float("nan"))
            raise ValueError("interaction_mode requires non-raw smart_embed_strategy")

        if self.interaction_mode in ("instance_residual", "instance_residual_engram"):
            if puzzle_embed_idxs is None:
                instance_idxs = puzzle_idxs.long()
            else:
                instance_idxs = puzzle_embed_idxs.long()
            memory = self.instance_residual_embed(instance_idxs)
            if self.interaction_mode == "instance_residual":
                residual = memory
                if hasattr(self, "instance_residual_up_proj"):
                    residual = residual.to(self.instance_residual_up_proj.weight.dtype)
                    residual = self.instance_residual_up_proj(residual)
            else:
                memory = memory.to(self.instance_engram_key_proj.weight.dtype)
                key = self.instance_engram_key_proj(memory)
                value = self.instance_engram_value_proj(memory)
                query = combined.to(key.dtype)
                normed_query = self.instance_engram_query_norm(query)
                normed_key = self.instance_engram_key_norm(key)
                alpha = torch.sigmoid((normed_query * normed_key).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size))
                residual = value * alpha
                with torch.no_grad():
                    gate_f = alpha.detach().to(torch.float32)
                    self.last_gate_mean.copy_(gate_f.mean())
                    self.last_gate_active_ratio.copy_((gate_f > 0.5).to(torch.float32).mean())
        else:
            raise ValueError(f"Unknown interaction_mode: {self.interaction_mode}")

        residual = residual.to(combined.dtype)
        if self.interaction_mode == "instance_residual_engram":
            return combined + residual
        if self.interaction_use_gate:
            gate = torch.sigmoid(self.interaction_gate(combined))
            with torch.no_grad():
                gate_f = gate.detach().to(torch.float32)
                self.last_gate_mean.copy_(gate_f.mean())
                self.last_gate_active_ratio.copy_((gate_f > 0.5).to(torch.float32).mean())
            residual = residual * gate
        else:
            with torch.no_grad():
                self.last_gate_mean.fill_(1.0)
                self.last_gate_active_ratio.fill_(1.0)
        return combined + residual

    def _get_puzzle_embedding(self, puzzle_idxs: torch.Tensor) -> torch.Tensor:
        device = puzzle_idxs.device
        idx = puzzle_idxs.long().to(device)

        if self.puzzle_embed_dual:
            if self.puzzle_embed_dual_mode == "random":
                if self.training:
                    offset = (torch.rand(idx.shape, device=device) < 0.5).long() * self._num_puzzles_base
                    idx = idx + offset
                # eval: use entry 0 (no offset)
                return self.puzzle_embed(idx)
            elif self.puzzle_embed_dual_mode == "average":
                emb_0 = self.puzzle_embed(idx)
                emb_1 = self.puzzle_embed(idx + self._num_puzzles_base)
                return (emb_0 + emb_1) / 2
        return self.puzzle_embed(idx)

    def forward(
        self,
        puzzle_idxs: torch.Tensor,
        dih_idxs: torch.Tensor,
        colorperm_input: torch.Tensor,
        puzzle_embed_idxs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = puzzle_idxs.device
        puzzle_emb = self.puzzle_embed_up_proj(self._get_puzzle_embedding(puzzle_idxs))
        if self.embed_source == "slotperm_full":
            dih_emb = self._get_dih_embedding(dih_idxs.to(device))
        else:
            dih_emb = self.dih_embed(dih_idxs.long().to(device))

        if self.combine_strategy == "slotperm_pc":
            colorperm_emb = self._get_colorperm_embedding_pc(colorperm_input.to(device), puzzle_emb)
        elif self.combine_strategy == "slotperm_moe":
            colorperm_emb = self._get_colorperm_embedding_moe(colorperm_input.to(device), puzzle_emb)
        elif self.combine_strategy == "raw_colorperm":
            colorperm_emb = self._get_colorperm_full_embedding(colorperm_input.to(device))
        elif self.embed_source in ("slotperm", "slotperm_full"):
            colorperm_emb = self._get_colorperm_embedding(colorperm_input.to(device))
        else:
            colorperm_emb = self.colorperm_embed(colorperm_input.long().to(device))

        combined = self._combine(puzzle_emb, dih_emb, colorperm_emb)
        return self._apply_interaction_residual(
            combined=combined,
            puzzle_emb=puzzle_emb,
            dih_emb=dih_emb,
            colorperm_emb=colorperm_emb,
            puzzle_idxs=puzzle_idxs.to(device),
            dih_idxs=dih_idxs.to(device),
            colorperm_input=colorperm_input.to(device),
            puzzle_embed_idxs=None if puzzle_embed_idxs is None else puzzle_embed_idxs.to(device),
        )
