"""Independent-run interval baseline for reference and lowered executions."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .arithmetic import Matrix, round_div, word_bounds
from .model import Box, Instance


@dataclass(frozen=True)
class I:
    lo: int
    hi: int

    def add(self, other: "I") -> "I":
        return I(self.lo + other.lo, self.hi + other.hi)

    def scale(self, c: int) -> "I":
        a, b = c * self.lo, c * self.hi
        return I(min(a, b), max(a, b))


def q_interval(x: I, instance: Instance) -> I:
    lo = round_div(x.lo, instance.scale, instance.rounding)
    hi = round_div(x.hi, instance.scale, instance.rounding)
    wlo, whi = word_bounds(instance.width)
    return I(max(wlo, min(whi, lo)), max(wlo, min(whi, hi)))


def sum_i(values: Sequence[I]) -> I:
    out = I(0, 0)
    for value in values:
        out = out.add(value)
    return out


def apply_i(matrix: Matrix, bias: Sequence[I], state: Sequence[I], instance: Instance) -> tuple[I, ...]:
    d = instance.dimension
    return tuple(
        q_interval(sum_i([state[k].scale(matrix[i][k]) for k in range(d)] + [bias[i].scale(instance.scale)]), instance)
        for i in range(d)
    )


IPair = tuple[Matrix, tuple[I, ...]]


def compose_i(left: IPair, right: IPair, instance: Instance) -> IPair:
    a, b = left
    c, e = right
    d = instance.dimension
    p = tuple(
        tuple(q_interval(I(sum(c[i][k] * a[k][j] for k in range(d)), sum(c[i][k] * a[k][j] for k in range(d))), instance).lo for j in range(d))
        for i in range(d)
    )
    z = tuple(
        q_interval(sum_i([b[k].scale(c[i][k]) for k in range(d)] + [e[i].scale(instance.scale)]), instance)
        for i in range(d)
    )
    return p, z


def compose_many_i(pairs: Sequence[IPair], instance: Instance) -> IPair:
    if len(pairs) == 1:
        return pairs[0]
    if instance.association == "left":
        out = pairs[0]
        for p in pairs[1:]:
            out = compose_i(out, p, instance)
        return out
    if instance.association == "right":
        out = pairs[-1]
        for p in reversed(pairs[:-1]):
            out = compose_i(p, out, instance)
        return out
    mid = len(pairs) // 2
    return compose_i(compose_many_i(pairs[:mid], instance), compose_many_i(pairs[mid:], instance), instance)


def independent_bounds(instance: Instance, output_index: int, flat_boxes: Sequence[Box]) -> tuple[I, ...]:
    expected = (output_index + 1) * instance.dimension
    if len(flat_boxes) != expected:
        raise ValueError("wrong prefix dimension")
    inputs: list[tuple[I, ...]] = []
    pos = 0
    for _ in range(output_index + 1):
        row = tuple(I(flat_boxes[pos + i].lo, flat_boxes[pos + i].hi) for i in range(instance.dimension))
        pos += instance.dimension
        inputs.append(row)

    ref: tuple[I, ...] = tuple(I(x, x) for x in instance.h0)
    refs: list[tuple[I, ...]] = []
    for t in range(output_index + 1):
        ref = apply_i(instance.matrices[t], inputs[t], ref, instance)
        refs.append(ref)

    carry: tuple[I, ...] = tuple(I(x, x) for x in instance.h0)
    lows: list[tuple[I, ...]] = []
    for start in range(0, output_index + 1, instance.chunk_length):
        stop = min(output_index + 1, start + instance.chunk_length)
        pairs: list[IPair] = []
        last = carry
        for t in range(start, stop):
            pairs.append((instance.matrices[t], inputs[t]))
            summary = compose_many_i(pairs, instance)
            last = apply_i(summary[0], summary[1], carry, instance)
            lows.append(last)
        carry = last

    return tuple(I(l.lo - r.hi, l.hi - r.lo) for l, r in zip(lows[output_index], refs[output_index]))
