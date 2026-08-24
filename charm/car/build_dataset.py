"""Build the controlled single-horizon and all-horizons CAR datasets."""

from __future__ import annotations

import argparse
import hashlib
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from importlib.resources import files
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from tqdm import tqdm

from charm.car.format import build_arc_v2_schema, puzzle_hash
from charm.car.rules import FAMILIES, HORIZONS, Rule, sample_rule, step
from charm.preprocessing.arc import ARC_AUGMENTED_FORMAT_VERSION


GENERATED_FILES = {
    "all_horizons_train": "car_all_horizons_train.parquet",
    "all_horizons_eval": "car_all_horizons_eval.parquet",
    "single_horizon_train": "car_single_horizon_train.parquet",
}
FILES = {
    **GENERATED_FILES,
    "single_horizon_eval": "car_single_horizon_eval.parquet",
}


@dataclass(frozen=True)
class CARDatasetConfig:
    output_dir: Path
    rules_per_family: int = 500
    train_pairs: int = 500
    eval_pairs: int = 4
    height: int = 16
    width: int = 16
    seed: int = 0
    workers: int = 1


@dataclass(frozen=True)
class HorizonSelection:
    family: str
    rule: Rule
    selected_horizon: int


@dataclass(frozen=True)
class PairPool:
    inputs: np.ndarray
    outputs: dict[int, np.ndarray]


@dataclass(frozen=True)
class GeneratedRuleData:
    selection: HorizonSelection
    train: PairPool
    evaluation: PairPool


def _seed(master: int, *parts: object) -> int:
    text = "|".join((str(master), *(str(part) for part in parts)))
    return int.from_bytes(
        hashlib.sha256(text.encode()).digest()[:4],
        "little",
    )


def _rollout(initial: np.ndarray, rule: Rule) -> dict[int, np.ndarray]:
    state = initial
    outputs = {}
    for tick in range(1, max(HORIZONS) + 1):
        state = step(state, rule)
        if tick in HORIZONS:
            outputs[tick] = state.copy()
    return outputs


def _valid(initial: np.ndarray, outputs: dict[int, np.ndarray]) -> bool:
    return bool(initial.any()) and all(
        outputs[horizon].any()
        and (outputs[horizon] == 0).any()
        and not np.array_equal(initial, outputs[horizon])
        for horizon in HORIZONS
    )


def _viable(family: str, rule: Rule, config: CARDatasetConfig) -> bool:
    rng = np.random.RandomState(_seed(config.seed, "viability", family, rule.code))
    valid = 0
    for _ in range(64):
        density = rng.uniform(0.1, 0.5)
        initial = (rng.random_sample((config.height, config.width)) < density).astype(
            np.int8
        )
        valid += _valid(initial, _rollout(initial, rule))
    return valid >= 8


def _selection_rank(seed: int, family: str, rule: Rule) -> bytes:
    # Keep the original namespace so noncanonical builds remain byte-for-byte
    # compatible with earlier versions of the generator.
    text = f"car-balanced-assignment|{seed}|{family}_{rule.code:08x}"
    return hashlib.sha256(text.encode()).digest()


def _canonical_selection() -> dict[str, int]:
    resource = files("charm.car").joinpath("single_horizon_tasks.txt")
    result = {}
    for line in resource.read_text(encoding="utf-8").splitlines():
        puzzle_id = line.strip()
        if not puzzle_id or puzzle_id.startswith("#"):
            continue
        rule_id, horizon = puzzle_id.rsplit("_t", 1)
        result[rule_id] = int(horizon)
    return result


def _sample_selections(config: CARDatasetConfig) -> list[HorizonSelection]:
    if config.rules_per_family % len(HORIZONS):
        raise ValueError(
            f"rules_per_family must be divisible by {len(HORIZONS)} for a "
            "balanced horizon selection"
        )

    sampled: dict[str, list[Rule]] = {}
    for family in FAMILIES:
        rng = np.random.RandomState(_seed(config.seed, "rules", family))
        used_codes: set[int] = set()
        rules = []
        while len(rules) < config.rules_per_family:
            rule = sample_rule(family, rng, used_codes)
            used_codes.add(rule.code)
            if _viable(family, rule, config):
                rules.append(rule)
        sampled[family] = rules

    rule_ids = {
        f"{family}_{rule.code:08x}"
        for family, rules in sampled.items()
        for rule in rules
    }
    canonical = _canonical_selection()
    use_canonical = rule_ids == set(canonical)

    selections = []
    for family, rules in sampled.items():
        if use_canonical:
            selected_horizons = {
                rule.code: canonical[f"{family}_{rule.code:08x}"] for rule in rules
            }
        else:
            ranked_rules = sorted(
                rules,
                key=lambda rule: (
                    _selection_rank(config.seed, family, rule),
                    rule.code,
                ),
            )
            selected_horizons = {
                rule.code: HORIZONS[index % len(HORIZONS)]
                for index, rule in enumerate(ranked_rules)
            }
        selections.extend(
            HorizonSelection(
                family=family,
                rule=rule,
                selected_horizon=selected_horizons[rule.code],
            )
            for rule in rules
        )
    return sorted(
        selections,
        key=lambda item: (item.family, item.rule.code),
    )


def _sample_pairs(
    selection: HorizonSelection,
    config: CARDatasetConfig,
    pool: str,
    count: int,
    used_inputs: set[bytes],
) -> PairPool:
    rng = np.random.RandomState(
        _seed(
            config.seed,
            pool,
            selection.family,
            selection.rule.code,
        )
    )
    inputs = []
    outputs_by_horizon = {horizon: [] for horizon in HORIZONS}
    while len(inputs) < count:
        density = rng.uniform(0.1, 0.5)
        initial = (rng.random_sample((config.height, config.width)) < density).astype(
            np.int8
        )
        key = initial.tobytes()
        if key in used_inputs:
            continue
        rollout = _rollout(initial, selection.rule)
        if not _valid(initial, rollout):
            continue
        used_inputs.add(key)
        inputs.append(initial)
        for horizon, output in rollout.items():
            outputs_by_horizon[horizon].append(output.astype(np.int8))
    return PairPool(
        inputs=np.stack(inputs),
        outputs={
            horizon: np.stack(horizon_outputs)
            for horizon, horizon_outputs in outputs_by_horizon.items()
        },
    )


def _puzzle_id(selection: HorizonSelection, horizon: int) -> str:
    return f"{selection.family}_{selection.rule.code:08x}_t{horizon}"


def _row(
    selection: HorizonSelection,
    horizon: int,
    train_pairs: list[dict],
    test_pairs: list[dict],
) -> dict:
    puzzle = {"train": train_pairs, "test": test_pairs}
    digest = puzzle_hash(puzzle)
    return {
        "format_version": ARC_AUGMENTED_FORMAT_VERSION,
        "puzzle_id": _puzzle_id(selection, horizon),
        "puzzle_hash": digest,
        "train": train_pairs,
        "test": test_pairs,
        "offline_augs": [""],
        "aug_hashes": [digest],
    }


def _pairs(pool: PairPool, horizon: int) -> list[dict]:
    return [
        {"input": initial.tolist(), "output": output.tolist()}
        for initial, output in zip(pool.inputs, pool.outputs[horizon])
    ]


def _generate_selection(
    request: tuple[HorizonSelection, CARDatasetConfig],
) -> GeneratedRuleData:
    selection, config = request
    used_inputs: set[bytes] = set()
    train = _sample_pairs(
        selection,
        config,
        "train",
        config.train_pairs,
        used_inputs,
    )
    evaluation = _sample_pairs(
        selection,
        config,
        "eval",
        config.eval_pairs,
        used_inputs,
    )
    return GeneratedRuleData(
        selection=selection,
        train=train,
        evaluation=evaluation,
    )


def _build_rows(generated: GeneratedRuleData) -> dict[str, list[dict]]:
    selection = generated.selection
    rows = {name: [] for name in GENERATED_FILES}
    for horizon in HORIZONS:
        train_row = _row(
            selection,
            horizon,
            _pairs(generated.train, horizon),
            [],
        )
        eval_row = _row(
            selection,
            horizon,
            [],
            _pairs(generated.evaluation, horizon),
        )
        rows["all_horizons_train"].append(train_row)
        rows["all_horizons_eval"].append(eval_row)
        if horizon == selection.selected_horizon:
            rows["single_horizon_train"].append(train_row)
    return rows


def _write_single_horizon_eval(output_dir: Path) -> int:
    """Filter the all-horizons evaluation set to the selected combinations."""

    selected_ids = pq.read_table(
        output_dir / FILES["single_horizon_train"],
        columns=["puzzle_id"],
    )["puzzle_id"]
    all_horizons_eval = pq.read_table(output_dir / FILES["all_horizons_eval"])
    single_horizon_eval = all_horizons_eval.filter(
        pc.is_in(all_horizons_eval["puzzle_id"], value_set=selected_ids)
    )
    if single_horizon_eval.num_rows != len(selected_ids):
        raise ValueError(
            f"expected {len(selected_ids)} single-horizon evaluation rows, "
            f"found {single_horizon_eval.num_rows}"
        )
    pq.write_table(
        single_horizon_eval,
        output_dir / FILES["single_horizon_eval"],
        compression="zstd",
    )
    return single_horizon_eval.num_rows


def _generated_selections(
    selections: list[HorizonSelection],
    config: CARDatasetConfig,
) -> Iterable[GeneratedRuleData]:
    requests = ((selection, config) for selection in selections)
    if config.workers == 1:
        yield from map(_generate_selection, requests)
        return

    with ProcessPoolExecutor(
        max_workers=config.workers,
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        buffered = config.workers * 2
        pending = {
            index: executor.submit(
                _generate_selection,
                (selection, config),
            )
            for index, selection in enumerate(selections[:buffered])
        }
        remaining = iter(selections[buffered:])
        submit_index = len(pending)
        result_index = 0
        while pending:
            yield pending.pop(result_index).result()
            selection = next(remaining, None)
            if selection is not None:
                pending[submit_index] = executor.submit(
                    _generate_selection,
                    (selection, config),
                )
                submit_index += 1
            result_index += 1


def build_car_dataset(config: CARDatasetConfig) -> None:
    """Write all-horizons data and its exact single-horizon subsets."""

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    selections = _sample_selections(config)
    schema = build_arc_v2_schema()
    writers = {
        name: pq.ParquetWriter(
            output_dir / filename,
            schema,
            compression="zstd",
        )
        for name, filename in GENERATED_FILES.items()
    }
    counts = {name: 0 for name in FILES}

    try:
        for generated in tqdm(
            _generated_selections(selections, config),
            total=len(selections),
            desc="CAR rules",
        ):
            for name, rows in _build_rows(generated).items():
                writers[name].write_table(pa.Table.from_pylist(rows, schema=schema))
                counts[name] += len(rows)
    finally:
        for writer in writers.values():
            writer.close()

    counts["single_horizon_eval"] = _write_single_horizon_eval(output_dir)
    print(" ".join(f"{name}={count}" for name, count in counts.items()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--rules-per-family", type=int, default=500)
    parser.add_argument("--train-pairs", type=int, default=500)
    parser.add_argument("--eval-pairs", type=int, default=4)
    parser.add_argument("--height", type=int, default=16)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    build_car_dataset(CARDatasetConfig(**vars(args)))


if __name__ == "__main__":
    main()
