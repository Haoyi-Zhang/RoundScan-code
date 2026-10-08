"""Exact finite enumeration for short-horizon instances."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Iterator, Sequence

from .arithmetic import output_difference
from .model import Box, Instance


@dataclass(frozen=True)
class OracleResult:
    safe: bool
    earliest_output: int | None
    witness: tuple[int, ...] | None
    ranges: tuple[tuple[tuple[int, int], ...], ...]
    prefix_evaluations: int


def points(boxes: Sequence[Box]) -> Iterator[tuple[int, ...]]:
    return product(*(range(box.lo, box.hi + 1) for box in boxes))


def exact_oracle(instance: Instance) -> OracleResult:
    ranges: list[tuple[tuple[int, int], ...]] = []
    first_output: int | None = None
    first_witness: tuple[int, ...] | None = None
    count = 0
    for t in range(instance.horizon):
        lo_word, hi_word = instance.word_bounds
        span = hi_word - lo_word
        mins = [span] * instance.dimension
        maxs = [-span] * instance.dimension
        for point in points(instance.prefix_boxes(t)):
            count += 1
            diff = output_difference(instance, point, t)
            for i, value in enumerate(diff):
                mins[i] = min(mins[i], value)
                maxs[i] = max(maxs[i], value)
            if first_output is None and any(abs(diff[i]) > instance.tolerances[t][i] for i in range(instance.dimension)):
                first_output = t
                first_witness = tuple(point)
                # Enumeration is already lexicographic, but continue to compute ranges.
        ranges.append(tuple(zip(mins, maxs)))
    return OracleResult(
        safe=first_output is None,
        earliest_output=first_output,
        witness=first_witness,
        ranges=tuple(ranges),
        prefix_evaluations=count,
    )
