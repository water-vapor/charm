"""The three totalistic cellular-automaton families used by CAR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


FAMILIES = ("generations", "lifelike", "ltl")
HORIZONS = (1, 2, 4, 8)


@dataclass(frozen=True)
class LifelikeRule:
    birth: int
    survival: int

    @property
    def code(self) -> int:
        return (self.birth << 9) | self.survival


@dataclass(frozen=True)
class GenerationsRule:
    birth: int
    survival: int
    states: int

    @property
    def code(self) -> int:
        states = (3, 4, 5, 6, 8).index(self.states)
        return (states << 18) | (self.birth << 9) | self.survival


@dataclass(frozen=True)
class LargerThanLifeRule:
    radius: int
    birth_low: int
    birth_high: int
    survival_low: int
    survival_high: int

    @property
    def code(self) -> int:
        radius = (2, 3).index(self.radius)
        return (
            (radius << 24)
            | (self.birth_low << 18)
            | (self.birth_high << 12)
            | (self.survival_low << 6)
            | self.survival_high
        )


Rule = LifelikeRule | GenerationsRule | LargerThanLifeRule


def _moore_count(
    state: np.ndarray,
    radius: int,
    alive_state: int = 1,
) -> np.ndarray:
    alive = state == alive_state
    count = np.zeros_like(state, dtype=np.int16)
    for row_shift in range(-radius, radius + 1):
        for col_shift in range(-radius, radius + 1):
            if row_shift or col_shift:
                count += np.roll(
                    np.roll(alive, row_shift, axis=0),
                    col_shift,
                    axis=1,
                )
    return count


def _mask_contains(mask: int, counts: np.ndarray) -> np.ndarray:
    return ((mask >> counts) & 1).astype(bool)


def step(state: np.ndarray, rule: Rule) -> np.ndarray:
    """Advance one tick on a toroidal grid."""

    if isinstance(rule, LifelikeRule):
        neighbors = _moore_count(state, radius=1)
        born = (state == 0) & _mask_contains(rule.birth, neighbors)
        survives = (state == 1) & _mask_contains(rule.survival, neighbors)
        return (born | survives).astype(np.int8)

    if isinstance(rule, GenerationsRule):
        neighbors = _moore_count(state, radius=1)
        born = (state == 0) & _mask_contains(rule.birth, neighbors)
        survives = (state == 1) & _mask_contains(rule.survival, neighbors)
        result = np.zeros_like(state)
        result[born | survives] = 1
        result[(state == 1) & ~survives] = 2
        for dying_state in range(2, rule.states - 1):
            result[state == dying_state] = dying_state + 1
        return result.astype(np.int8)

    neighbors = _moore_count(state, radius=rule.radius)
    born = (state == 0) & (neighbors >= rule.birth_low) & (neighbors <= rule.birth_high)
    survives = (
        (state == 1)
        & (neighbors >= rule.survival_low)
        & (neighbors <= rule.survival_high)
    )
    return (born | survives).astype(np.int8)


def sample_rule(
    family: str,
    rng: np.random.RandomState,
    used_codes: set[int],
) -> Rule:
    """Sample a rule not already used within its family."""

    while True:
        if family == "lifelike":
            rule: Rule = LifelikeRule(
                birth=int(rng.randint(1, 1 << 9)),
                survival=int(rng.randint(1 << 9)),
            )
        elif family == "generations":
            rule = GenerationsRule(
                birth=int(rng.randint(1, 1 << 9)),
                survival=int(rng.randint(1 << 9)),
                states=int((3, 4, 5, 6, 8)[rng.randint(5)]),
            )
        elif family == "ltl":
            radius = int((2, 3)[rng.randint(2)])
            neighborhood = (2 * radius + 1) ** 2 - 1
            birth_low = int(rng.randint(1, neighborhood + 1))
            survival_low = int(rng.randint(neighborhood + 1))
            rule = LargerThanLifeRule(
                radius=radius,
                birth_low=birth_low,
                birth_high=int(rng.randint(birth_low, neighborhood + 1)),
                survival_low=survival_low,
                survival_high=int(rng.randint(survival_low, neighborhood + 1)),
            )
        else:
            raise ValueError(f"unknown CAR family: {family}")

        if rule.code not in used_codes:
            return rule
