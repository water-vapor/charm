from __future__ import annotations

from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch

from charm.car.build_dataset import (
    CARDatasetConfig,
    FILES,
    _build_rows,
    _generate_selection,
    _rollout,
    _sample_selections,
    _valid,
    build_car_dataset,
)
from charm.car.dataset import (
    CARPairDataset,
    ExhaustiveDihedralDataset,
    car_collate_fn,
)
from charm.car.evaluator import CAREvaluator, build_car_evaluator
from charm.car.format import build_arc_v2_schema
from charm.car.metadata import build_car_id_maps, load_car_id_maps
from charm.car.rules import FAMILIES, HORIZONS, GenerationsRule, step
from charm.car.task_memory import CARTaskMemory, MODES
from charm.transforms import (
    CanvasEOSTransform,
    DihedralTransform,
    FlattenTransform,
    inverse_chain,
)


def _small_config(output_dir: Path, workers: int = 1) -> CARDatasetConfig:
    return CARDatasetConfig(
        output_dir=output_dir,
        rules_per_family=4,
        train_pairs=2,
        eval_pairs=1,
        height=10,
        width=10,
        seed=13,
        workers=workers,
    )


def test_generation_split_and_worker_determinism(tmp_path):
    serial = tmp_path / "serial"
    parallel = tmp_path / "parallel"
    build_car_dataset(_small_config(serial, workers=1))
    build_car_dataset(_small_config(parallel, workers=2))

    expected_rows = {
        "all_horizons_train": 48,
        "all_horizons_eval": 48,
        "single_horizon_train": 12,
        "single_horizon_eval": 12,
    }
    for name, filename in FILES.items():
        assert (serial / filename).read_bytes() == (parallel / filename).read_bytes()
        table = pq.read_table(serial / filename)
        assert table.num_rows == expected_rows[name]
        assert table.schema == build_arc_v2_schema()

    tables = {
        name: pq.read_table(serial / filename).to_pylist()
        for name, filename in FILES.items()
    }
    all_horizons_train = {row["puzzle_id"]: row for row in tables["all_horizons_train"]}
    all_horizons_eval = {row["puzzle_id"]: row for row in tables["all_horizons_eval"]}
    single_horizon_train = {
        row["puzzle_id"]: row for row in tables["single_horizon_train"]
    }
    single_horizon_eval = {
        row["puzzle_id"]: row for row in tables["single_horizon_eval"]
    }
    assert set(single_horizon_train) == set(single_horizon_eval)
    assert all(
        row == all_horizons_train[puzzle_id]
        for puzzle_id, row in single_horizon_train.items()
    )
    assert all(
        row == all_horizons_eval[puzzle_id]
        for puzzle_id, row in single_horizon_eval.items()
    )
    selected_counts = Counter(
        (
            puzzle_id.rsplit("_", 2)[0],
            int(puzzle_id.rsplit("_t", 1)[1]),
        )
        for puzzle_id in single_horizon_train
    )
    assert selected_counts == Counter(
        {(family, horizon): 1 for family in FAMILIES for horizon in HORIZONS}
    )


def test_generated_outputs_are_exact_rollouts(tmp_path):
    config = _small_config(tmp_path)
    for selection in _sample_selections(config):
        rows = _build_rows(_generate_selection((selection, config)))[
            "all_horizons_train"
        ]
        for row in rows:
            horizon = int(row["puzzle_id"].rsplit("_t", 1)[1])
            pair = row["train"][0]
            state = np.asarray(pair["input"], dtype=np.int8)
            for _ in range(horizon):
                state = step(state, selection.rule)
            np.testing.assert_array_equal(state, pair["output"])


def test_nontrivial_filter_rejects_empty_and_constant_dynamics():
    rule = GenerationsRule(
        birth=1 << 0,
        survival=1 << 8,
        states=3,
    )
    empty = np.zeros((8, 8), dtype=np.int8)
    full = np.ones((8, 8), dtype=np.int8)
    assert not _valid(empty, _rollout(empty, rule))
    assert not _valid(full, _rollout(full, rule))


def test_single_horizon_manifest_is_balanced():
    manifest = Path(__file__).parents[1] / "charm" / "car" / "single_horizon_tasks.txt"
    task_ids = manifest.read_text().splitlines()
    assert len(task_ids) == len(set(task_ids)) == 1500
    counts = Counter(
        (task_id.rsplit("_", 2)[0], int(task_id.rsplit("_t", 1)[1]))
        for task_id in task_ids
    )
    assert counts == Counter(
        {(family, horizon): 125 for family in FAMILIES for horizon in HORIZONS}
    )


def test_metadata_and_exhaustive_d4(tmp_path):
    data_dir = tmp_path / "data"
    build_car_dataset(_small_config(data_dir))
    train_path = data_dir / FILES["single_horizon_train"]
    eval_path = data_dir / FILES["all_horizons_eval"]
    task_ids, rule_ids = load_car_id_maps([train_path])
    train = CARPairDataset(
        parquet_paths=[train_path],
        pair_types=["train"],
        eval_mode=False,
        car_task_id_to_int=task_ids,
        car_rule_id_to_int=rule_ids,
        cache_dir=tmp_path / "cache",
    )
    evaluation = CARPairDataset(
        parquet_paths=[eval_path],
        eval_mode=True,
        online_transforms=[
            DihedralTransform(p=1.0),
            CanvasEOSTransform(canvas_size=30, p_translate=0.0),
            FlattenTransform(),
        ],
        unique_str_to_int=train.unique_str_to_int,
        puzzle_id_to_int=train.puzzle_id_to_int,
        car_task_id_to_int=task_ids,
        car_rule_id_to_int=rule_ids,
        cache_dir=tmp_path / "cache",
    )
    views = ExhaustiveDihedralDataset(evaluation)
    assert len(views) == 8 * len(evaluation)

    known = next(
        views[index] for index in range(len(views)) if views[index]["car_task_idx"] >= 0
    )
    unseen = next(
        views[index] for index in range(len(views)) if views[index]["car_task_idx"] < 0
    )
    _, batch, batch_size = car_collate_fn([known, unseen])
    assert batch_size == 2
    assert batch["car_task_idxs"].tolist() == [
        known["car_task_idx"],
        -1,
    ]
    assert unseen["car_rule_idx"] >= 0


def _memory(mode: str, num_tasks: int = 12, num_rules: int = 3):
    memory = CARTaskMemory(
        mode=mode,
        num_tasks=num_tasks,
        num_rules=num_rules,
        hidden_size=8,
        rank=2,
        batch_size=2,
        forward_dtype=torch.float32,
    )
    memory.eval()
    return memory


@pytest.mark.parametrize("mode", sorted(MODES))
def test_all_task_memories_return_one_vector(mode):
    output = _memory(mode)(
        task_indices=torch.tensor([1, 7]),
        rule_indices=torch.tensor([0, 1]),
        horizon_indices=torch.tensor([1, 3]),
        dihedral_indices=torch.tensor([2, 6]),
    )
    assert output.shape == (2, 8)


@pytest.mark.parametrize("mode", ("full_table", "lowrank_table"))
def test_unseen_task_uses_zero_table_row(mode):
    memory = _memory(mode, num_tasks=2, num_rules=2)
    memory.instance_embed.weights.fill_(1)
    memory.instance_embed.weights[0].zero_()
    output = memory(
        task_indices=torch.tensor([-1], dtype=torch.int32),
        rule_indices=torch.tensor([1]),
        horizon_indices=torch.tensor([3]),
        dihedral_indices=torch.tensor([6]),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("mode", "tasks", "rules", "expected"),
    [
        ("full_table", 1500, 1500, 3_072_256),
        ("lowrank_table", 1500, 1500, 196_112),
        ("two_factor_composition", 1500, 1500, 517_632),
        ("two_factor_cose", 1500, 1500, 713_744),
        ("three_factor_composition", 1500, 1500, 518_656),
        ("three_factor_cose", 1500, 1500, 714_768),
        ("full_table", 6000, 1500, 12_288_256),
        ("lowrank_table", 6000, 1500, 772_112),
        ("two_factor_composition", 6000, 1500, 1_669_632),
        ("two_factor_cose", 6000, 1500, 2_441_744),
        ("three_factor_composition", 6000, 1500, 518_656),
        ("three_factor_cose", 6000, 1500, 1_290_768),
    ],
)
def test_task_memory_parameter_counts(mode, tasks, rules, expected):
    memory = CARTaskMemory(
        mode=mode,
        num_tasks=tasks,
        num_rules=rules,
        hidden_size=256,
        rank=16,
        batch_size=8,
        forward_dtype=torch.float32,
    )
    assert memory.task_memory_parameter_count() == expected


class _EvalDataset:
    eval_mode = True
    online_transforms = [DihedralTransform(p=1.0)]

    def __init__(self):
        self.grid = np.arange(9, dtype=np.uint8).reshape(3, 3)

    def __len__(self):
        return 1

    def get_raw_pair(self, _):
        return {
            "input": self.grid,
            "output": None,
            "offline_aug": "",
            "puzzle_id": "lifelike_00000001_t4",
            "puzzle_idx": 0,
            "unique_str": "lifelike_00000001_t4||",
            "puzzle_embed_idx": 1,
            "car_task_idx": 0,
            "car_rule_idx": 0,
            "car_horizon_idx": 2,
        }


def test_exhaustive_d4_vote_and_horizon_metric():
    base = _EvalDataset()
    dataset = ExhaustiveDihedralDataset(base)
    items = [dataset[index] for index in range(8)]
    for item in items:
        np.testing.assert_array_equal(
            inverse_chain(item["input"], item["online_aug"]),
            base.grid,
        )
    _, batch, _ = car_collate_fn(items)
    evaluator = CAREvaluator(
        ground_truth_inputs={items[0]["puzzle_id"]: [base.grid]},
        ground_truth_outputs={items[0]["puzzle_id"]: [base.grid.tolist()]},
        pass_ks=(1, 2),
    )
    evaluator.clear_eval(event_id=1)
    evaluator.update(
        {
            "puzzle_ids": batch["puzzle_ids"],
            "offline_aug": batch["offline_aug"],
            "online_aug": batch["online_aug"],
            "inputs": batch["inputs"],
        },
        batch["inputs"],
        torch.zeros(8),
    )
    assert evaluator.result() == {
        "pass@1": 1.0,
        "pass@2": 1.0,
        "by_horizon/H=4/pass@1": 1.0,
        "by_horizon/H=4/pass@2": 1.0,
    }
    assert evaluator._pred_grids == {}


def test_evaluator_labels_seen_and_unseen_combinations():
    seen = "lifelike_00000001_t1"
    unseen = "lifelike_00000001_t2"
    grid = np.ones((2, 2), dtype=np.uint8)
    eval_dataset = SimpleNamespace(
        get_noaug_test_inputs=lambda: {seen: [grid], unseen: [grid]},
        get_noaug_test_outputs=lambda: {
            seen: [grid.tolist()],
            unseen: [grid.tolist()],
        },
    )
    evaluator = build_car_evaluator(
        SimpleNamespace(
            eval_recent_window_size=None,
            eval_exp_decay=False,
            eval_decay_half_life=10.0,
        ),
        SimpleNamespace(car_task_id_to_int={seen: 0}),
        eval_dataset,
    )
    assert evaluator.puzzle_groups == {
        "seen": (seen,),
        "unseen": (unseen,),
    }


def test_metadata_indices_ignore_duplicates():
    task_a = "lifelike_00000001_t1"
    task_b = "lifelike_00000001_t2"
    tasks, rules = build_car_id_maps([task_a, task_a, task_b])
    assert tasks == {task_a: 0, task_b: 1}
    assert rules == {"lifelike_00000001": 0}
