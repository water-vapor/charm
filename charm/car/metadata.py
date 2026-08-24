"""Task-factor indices encoded by CAR puzzle identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Iterable

import pyarrow.parquet as pq

from charm.car.rules import HORIZONS


_HORIZON_INDEX = {horizon: index for index, horizon in enumerate(HORIZONS)}
_PUZZLE_ID = re.compile(
    r"^(?P<family>generations|lifelike|ltl)_"
    r"(?P<rule_code>[0-9a-f]{8})_t(?P<horizon>1|2|4|8)$"
)


@dataclass(frozen=True)
class CARFactors:
    family: str
    rule_code: int
    rule_id: str
    horizon: int
    horizon_idx: int


def parse_car_puzzle_id(puzzle_id: str) -> CARFactors | None:
    match = _PUZZLE_ID.fullmatch(puzzle_id)
    if match is None:
        return None
    family = match.group("family")
    rule_code = int(match.group("rule_code"), 16)
    horizon = int(match.group("horizon"))
    return CARFactors(
        family=family,
        rule_code=rule_code,
        rule_id=f"{family}_{rule_code:08x}",
        horizon=horizon,
        horizon_idx=_HORIZON_INDEX[horizon],
    )


def build_car_id_maps(
    puzzle_ids: Iterable[str],
) -> tuple[dict[str, int], dict[str, int]]:
    parsed = {}
    for puzzle_id in puzzle_ids:
        factors = parse_car_puzzle_id(puzzle_id)
        if factors is None:
            raise ValueError("CAR task memory requires CAR puzzle identifiers")
        parsed[puzzle_id] = factors

    rules = sorted(
        {
            (factors.family, factors.rule_code, factors.rule_id)
            for factors in parsed.values()
        }
    )
    tasks = sorted(
        (
            factors.family,
            factors.rule_code,
            factors.horizon_idx,
            puzzle_id,
        )
        for puzzle_id, factors in parsed.items()
    )
    return (
        {task[-1]: index for index, task in enumerate(tasks)},
        {rule[-1]: index for index, rule in enumerate(rules)},
    )


def load_car_id_maps(
    parquet_paths: Iterable[str | Path],
) -> tuple[dict[str, int], dict[str, int]]:
    puzzle_ids = set()
    for path in parquet_paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=["puzzle_id"]):
            puzzle_ids.update(batch.column(0).to_pylist())
    return build_car_id_maps(puzzle_ids)
