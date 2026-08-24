"""Task-memory parameterizations for CAR experiments."""

from __future__ import annotations

import torch
from torch import nn

from charm.models.sparse_embedding import CastedSparseEmbedding


MODES = {
    "full_table",
    "lowrank_table",
    "two_factor_composition",
    "two_factor_cose",
    "three_factor_composition",
    "three_factor_cose",
}


class CARTaskMemory(nn.Module):
    """Map CAR task factors and a D4 element to one conditioning vector."""

    def __init__(
        self,
        *,
        mode: str,
        num_tasks: int,
        num_rules: int,
        hidden_size: int,
        rank: int,
        batch_size: int,
        forward_dtype: torch.dtype,
    ) -> None:
        super().__init__()
        if mode not in MODES:
            raise ValueError(f"unknown CAR task-memory mode: {mode}")

        self.mode = mode
        self.num_tasks = num_tasks
        self.num_rules = num_rules
        self.hidden_size = hidden_size
        self.rank = rank
        self.forward_dtype = forward_dtype
        self.three_factor = mode in {
            "three_factor_composition",
            "three_factor_cose",
        }
        self.composes = mode in {
            "two_factor_composition",
            "two_factor_cose",
            "three_factor_composition",
            "three_factor_cose",
        }
        self.has_table = mode in {
            "full_table",
            "lowrank_table",
            "two_factor_cose",
            "three_factor_cose",
        }
        self.full_table = mode == "full_table"

        if self.composes:
            if self.three_factor:
                self.rule_embed = nn.Embedding(num_rules, hidden_size)
                self.horizon_embed = nn.Embedding(4, hidden_size)
                nn.init.zeros_(self.rule_embed.weight)
                nn.init.zeros_(self.horizon_embed.weight)
            else:
                self.task_embed = nn.Embedding(num_tasks, hidden_size)
                nn.init.zeros_(self.task_embed.weight)

            self.dihedral_embed = nn.Embedding(8, hidden_size)
            self.film_scale = nn.Linear(hidden_size, hidden_size)
            self.film_shift = nn.Linear(hidden_size, hidden_size)
            nn.init.zeros_(self.dihedral_embed.weight)

        if self.has_table:
            table_dim = hidden_size if self.full_table else rank
            self.instance_embed = CastedSparseEmbedding(
                1 + num_tasks * 8,
                table_dim,
                batch_size=batch_size,
                init_std=0,
                cast_to=forward_dtype,
            )
            if not self.full_table:
                self.instance_projection = nn.Linear(
                    rank,
                    hidden_size,
                    bias=False,
                )

    @property
    def sparse_embeddings(self) -> tuple[CastedSparseEmbedding, ...]:
        return (self.instance_embed,) if self.has_table else ()

    def task_memory_parameter_count(self) -> int:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        sparse_values = sum(
            embedding.weights.numel() for embedding in self.sparse_embeddings
        )
        return parameters + sparse_values

    def _composition(
        self,
        task_indices: torch.Tensor,
        rule_indices: torch.Tensor,
        horizon_indices: torch.Tensor,
        dihedral_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self.three_factor:
            semantic = self.rule_embed(rule_indices.long()) + self.horizon_embed(
                horizon_indices.long()
            )
        else:
            semantic = self.task_embed(task_indices.long())
        dihedral = self.dihedral_embed(dihedral_indices.long())
        return (1 + self.film_scale(dihedral)) * semantic + self.film_shift(dihedral)

    def _table(
        self,
        task_indices: torch.Tensor,
        dihedral_indices: torch.Tensor,
    ) -> torch.Tensor:
        # Row zero is reserved for combinations not observed during training.
        rows = torch.where(
            task_indices >= 0,
            1 + task_indices * 8 + dihedral_indices.long(),
            torch.zeros_like(task_indices, dtype=torch.long),
        )
        memory = self.instance_embed(rows)
        if self.full_table:
            return memory
        return self.instance_projection(
            memory.to(self.instance_projection.weight.dtype)
        )

    def forward(
        self,
        *,
        task_indices: torch.Tensor,
        rule_indices: torch.Tensor,
        horizon_indices: torch.Tensor,
        dihedral_indices: torch.Tensor,
    ) -> torch.Tensor:
        composition = (
            self._composition(
                task_indices,
                rule_indices,
                horizon_indices,
                dihedral_indices,
            )
            if self.composes
            else None
        )
        table = (
            self._table(task_indices.long(), dihedral_indices)
            if self.has_table
            else None
        )
        if composition is None:
            result = table
        elif table is None:
            result = composition
        else:
            result = composition + table.to(composition.dtype)
        if result is None:
            raise RuntimeError("CAR task memory produced no conditioning vector")
        return result.to(self.forward_dtype)
