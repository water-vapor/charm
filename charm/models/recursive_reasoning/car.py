"""URM-v2 with CAR's structured task-memory parameterizations."""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from charm.car.task_memory import CARTaskMemory
from charm.models.recursive_reasoning.urm_v2 import (
    URM,
    URMCarry,
    URMConfig,
    URM_Inner,
)


class CARURMConfig(URMConfig):
    car_task_memory_mode: str = "full_table"
    car_task_memory_rank: int = 16
    car_num_tasks: int = 0
    car_num_rules: int = 0


class CARURMInner(URM_Inner):
    def __init__(self, config: CARURMConfig) -> None:
        if config.use_smart_embed:
            raise ValueError("CAR task memory and SmartTaskEmbedding are exclusive")
        if config.puzzle_emb_ndim <= 0:
            raise ValueError("CAR task memory requires puzzle_emb_ndim > 0")
        super().__init__(config)
        self.config = config
        self.puzzle_emb = CARTaskMemory(
            mode=config.car_task_memory_mode,
            num_tasks=config.car_num_tasks,
            num_rules=config.car_num_rules,
            hidden_size=config.hidden_size,
            rank=config.car_task_memory_rank,
            batch_size=config.batch_size,
            forward_dtype=self.forward_dtype,
        )

    def _car_input_embeddings(
        self,
        input_tensor: torch.Tensor,
        task_indices: torch.Tensor,
        rule_indices: torch.Tensor,
        horizon_indices: torch.Tensor,
        dihedral_indices: torch.Tensor,
    ) -> torch.Tensor:
        embedding = self.embed_tokens(input_tensor.to(torch.int32))
        device = input_tensor.device
        single_embedding = self.puzzle_emb(
            task_indices=task_indices.to(device),
            rule_indices=rule_indices.to(device),
            horizon_indices=horizon_indices.to(device),
            dihedral_indices=dihedral_indices.to(device),
        )
        puzzle_embedding = single_embedding.unsqueeze(1)
        if self.puzzle_emb_len > 1:
            puzzle_embedding = F.pad(
                puzzle_embedding,
                (0, 0, 0, self.puzzle_emb_len - 1),
            )

        if self.config.seq_embed_strategy == "prepend":
            embedding = torch.cat((puzzle_embedding, embedding), dim=-2)
        elif self.config.seq_embed_strategy == "add":
            prefix = torch.zeros(
                (
                    input_tensor.shape[0],
                    self.puzzle_emb_len,
                    self.config.hidden_size,
                ),
                dtype=self.forward_dtype,
                device=device,
            )
            embedding = torch.cat((prefix, embedding), dim=-2)
            embedding = embedding + single_embedding[:, None, :]
        else:
            raise ValueError(
                f"invalid sequence embedding strategy: {self.config.seq_embed_strategy}"
            )
        return self.embed_scale * embedding

    def forward(
        self,
        carry: URMCarry,
        batch: Dict[str, torch.Tensor],
    ) -> Tuple[URMCarry, torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        required = (
            "car_task_idxs",
            "car_rule_idxs",
            "car_horizon_idxs",
            "dih_idxs",
        )
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"CAR model batch is missing: {', '.join(missing)}")

        seq_info = dict(cos_sin=self.rotary_emb())
        input_embeddings = self._car_input_embeddings(
            batch["inputs"],
            batch["car_task_idxs"],
            batch["car_rule_idxs"],
            batch["car_horizon_idxs"],
            batch["dih_idxs"],
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
        output = self.lm_head(hidden_states)[:, self.puzzle_emb_len :]
        q_halt_logits = self.q_head(hidden_states[:, 0]).to(torch.float32).squeeze(-1)
        return new_carry, output, q_halt_logits, aux_outputs


class CARURM(URM):
    def __init__(self, config_dict: dict):
        nn.Module.__init__(self)
        self.config = CARURMConfig(**config_dict)
        self.inner = CARURMInner(self.config)
