"""Deterministic pre-clipping observations for finite execution tests.

The scientific semantics store only clipped fixed-point words.  Reduction tests
also need to establish that clipping was *inactive* in the bounded executable
families.  This module replays the same producer arithmetic while recording the
rounded value immediately before every clip at four semantically distinct
sites: sequential-state updates, summary-matrix entries, summary biases, and
summary application to a chunk carry.

These observations are finite regression evidence.  They do not replace the
hand proof that the full reductions avoid clipping.
"""
from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from typing import Protocol, Sequence

from .arithmetic import Matrix, Pair, Vector, round_div, word_bounds
from .model import Instance


@dataclass(frozen=True)
class QuantizationEvent:
    category: str
    site: str
    numerator: int
    rounded: int
    stored: int

    @property
    def clipped(self) -> bool:
        return self.rounded != self.stored


class EventSink(Protocol):
    def record(
        self,
        category: str,
        site: str,
        numerator: int,
        rounded: int,
        stored: int,
    ) -> None: ...


@dataclass
class EventList:
    events: list[QuantizationEvent]

    @classmethod
    def empty(cls) -> "EventList":
        return cls([])

    def record(
        self,
        category: str,
        site: str,
        numerator: int,
        rounded: int,
        stored: int,
    ) -> None:
        self.events.append(QuantizationEvent(category, site, numerator, rounded, stored))


@dataclass
class QuantizationAudit:
    """Allocation-light aggregate used for exhaustive bounded reductions."""

    counts: Counter[str]
    clipped_counts: Counter[str]

    @classmethod
    def empty(cls) -> "QuantizationAudit":
        return cls(Counter(), Counter())

    def record(
        self,
        category: str,
        site: str,
        numerator: int,
        rounded: int,
        stored: int,
    ) -> None:
        del site, numerator
        self.counts[category] += 1
        if rounded != stored:
            self.clipped_counts[category] += 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())


def _observe_quantize(
    numerator: int,
    *,
    width: int,
    scale: int,
    mode: str,
    category: str,
    site: str,
    events: EventSink,
) -> int:
    rounded = round_div(numerator, scale, mode)
    lo, hi = word_bounds(width)
    stored = min(hi, max(lo, rounded))
    events.record(category, site, numerator, rounded, stored)
    return stored


def _apply_pair(
    pair: Pair,
    state: Vector,
    *,
    width: int,
    scale: int,
    mode: str,
    category: str,
    site: str,
    events: EventSink,
) -> Vector:
    matrix, bias = pair
    d = len(state)
    detailed = not isinstance(events, QuantizationAudit)
    return tuple(
        _observe_quantize(
            sum(matrix[i][k] * state[k] for k in range(d)) + scale * bias[i],
            width=width,
            scale=scale,
            mode=mode,
            category=category,
            site=f"{site}[{i}]" if detailed else "",
            events=events,
        )
        for i in range(d)
    )


def _compose_pair(
    left: Pair,
    right: Pair,
    *,
    width: int,
    scale: int,
    mode: str,
    site: str,
    events: EventSink,
) -> Pair:
    a, b = left
    c, e = right
    d = len(b)
    detailed = not isinstance(events, QuantizationAudit)
    matrix = tuple(
        tuple(
            _observe_quantize(
                sum(c[i][k] * a[k][j] for k in range(d)),
                width=width,
                scale=scale,
                mode=mode,
                category="summary.matrix",
                site=f"{site}.matrix[{i},{j}]" if detailed else "",
                events=events,
            )
            for j in range(d)
        )
        for i in range(d)
    )
    bias = tuple(
        _observe_quantize(
            sum(c[i][k] * b[k] for k in range(d)) + scale * e[i],
            width=width,
            scale=scale,
            mode=mode,
            category="summary.bias",
            site=f"{site}.bias[{i}]" if detailed else "",
            events=events,
        )
        for i in range(d)
    )
    return matrix, bias


def _compose_many(
    pairs: Sequence[Pair],
    *,
    width: int,
    scale: int,
    mode: str,
    association: str,
    site: str,
    events: EventSink,
) -> Pair:
    if not pairs:
        raise ValueError("cannot compose an empty sequence")
    if len(pairs) == 1:
        return pairs[0]
    if association == "left":
        acc = pairs[0]
        detailed = not isinstance(events, QuantizationAudit)
        for index, pair in enumerate(pairs[1:], 1):
            acc = _compose_pair(
                acc,
                pair,
                width=width,
                scale=scale,
                mode=mode,
                site=f"{site}.left{index}" if detailed else "",
                events=events,
            )
        return acc
    if association == "right":
        acc = pairs[-1]
        detailed = not isinstance(events, QuantizationAudit)
        for index, pair in enumerate(reversed(pairs[:-1]), 1):
            acc = _compose_pair(
                pair,
                acc,
                width=width,
                scale=scale,
                mode=mode,
                site=f"{site}.right{index}" if detailed else "",
                events=events,
            )
        return acc
    if association == "balanced":
        middle = len(pairs) // 2
        detailed = not isinstance(events, QuantizationAudit)
        first = _compose_many(
            pairs[:middle],
            width=width,
            scale=scale,
            mode=mode,
            association=association,
            site=f"{site}.L" if detailed else "",
            events=events,
        )
        second = _compose_many(
            pairs[middle:],
            width=width,
            scale=scale,
            mode=mode,
            association=association,
            site=f"{site}.R" if detailed else "",
            events=events,
        )
        return _compose_pair(
            first,
            second,
            width=width,
            scale=scale,
            mode=mode,
            site=f"{site}.root" if detailed else "",
            events=events,
        )
    raise ValueError(f"unknown association {association!r}")


def _audit_execution(
    instance: Instance,
    inputs: Sequence[Sequence[int]],
    events: EventSink,
) -> tuple[tuple[Vector, ...], tuple[Vector, ...]]:
    if len(inputs) != instance.horizon:
        raise ValueError("input sequence has wrong horizon")
    normalized: list[Vector] = []
    detailed = not isinstance(events, QuantizationAudit)
    for t, row in enumerate(inputs):
        if len(row) != instance.dimension:
            raise ValueError("input vector has wrong dimension")
        values = tuple(row)
        if any(type(value) is not int for value in values):
            raise TypeError("inputs must be integers")
        if any(not box.lo <= value <= box.hi for value, box in zip(values, instance.boxes[t])):
            raise ValueError("input lies outside declared box")
        normalized.append(values)

    reference: list[Vector] = []
    state = instance.h0
    for t, (matrix, bias) in enumerate(zip(instance.matrices, normalized)):
        state = _apply_pair(
            (matrix, bias),
            state,
            width=instance.width,
            scale=instance.scale,
            mode=instance.rounding,
            category="sequential.state",
            site=f"step{t}" if detailed else "",
            events=events,
        )
        reference.append(state)

    lowered: list[Vector] = []
    carry = instance.h0
    for start in range(0, instance.horizon, instance.chunk_length):
        stop = min(instance.horizon, start + instance.chunk_length)
        pairs: list[Pair] = []
        last = carry
        for t in range(start, stop):
            pairs.append((instance.matrices[t], normalized[t]))
            summary = _compose_many(
                pairs,
                width=instance.width,
                scale=instance.scale,
                mode=instance.rounding,
                association=instance.association,
                site=f"chunk{start}.prefix{t}" if detailed else "",
                events=events,
            )
            last = _apply_pair(
                summary,
                carry,
                width=instance.width,
                scale=instance.scale,
                mode=instance.rounding,
                category="carry.apply",
                site=f"chunk{start}.prefix{t}" if detailed else "",
                events=events,
            )
            lowered.append(last)
        carry = last

    return tuple(reference), tuple(lowered)


def audit_execution(
    instance: Instance,
    inputs: Sequence[Sequence[int]],
) -> tuple[tuple[Vector, ...], tuple[Vector, ...], tuple[QuantizationEvent, ...]]:
    """Replay both executions and return their traces plus every pre-clip event."""
    sink = EventList.empty()
    reference, lowered = _audit_execution(instance, inputs, sink)
    return reference, lowered, tuple(sink.events)


def audit_execution_summary(
    instance: Instance,
    inputs: Sequence[Sequence[int]],
) -> tuple[tuple[Vector, ...], tuple[Vector, ...], QuantizationAudit]:
    """Replay both executions with aggregate counts and no per-event storage."""
    audit = QuantizationAudit.empty()
    reference, lowered = _audit_execution(instance, inputs, audit)
    return reference, lowered, audit
