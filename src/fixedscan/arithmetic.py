"""Concrete fixed-point arithmetic and declared chunk-prefix semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .model import Instance

Vector = tuple[int, ...]
Matrix = tuple[tuple[int, ...], ...]
Pair = tuple[Matrix, Vector]


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def word_bounds(width: int) -> tuple[int, int]:
    return -(2 ** (width - 1)), 2 ** (width - 1) - 1


def clip(value: int, width: int) -> int:
    lo, hi = word_bounds(width)
    return min(hi, max(lo, value))


def round_div(numerator: int, scale: int, mode: str) -> int:
    """Divide by a positive scale using floor or round-to-nearest-even.

    Python's // is Euclidean floor division for a positive divisor, so the
    quotient/remainder definition is valid for negative numerators as well.
    """
    q, r = divmod(numerator, scale)
    if mode == "floor":
        return q
    if mode != "rne":
        raise ValueError(f"unknown rounding mode {mode!r}")
    twice = 2 * r
    if twice > scale or (twice == scale and q % 2 != 0):
        q += 1
    return q


def quantize(numerator: int, width: int, scale: int, mode: str) -> int:
    return clip(round_div(numerator, scale, mode), width)


def matvec_numerator(matrix: Matrix, state: Vector, bias: Vector, scale: int) -> list[int]:
    d = len(state)
    return [sum(matrix[i][k] * state[k] for k in range(d)) + scale * bias[i] for i in range(d)]


def apply_pair(pair: Pair, state: Vector, *, width: int, scale: int, mode: str) -> Vector:
    matrix, bias = pair
    return tuple(
        quantize(n, width, scale, mode)
        for n in matvec_numerator(matrix, state, bias, scale)
    )


def compose_pair(left: Pair, right: Pair, *, width: int, scale: int, mode: str) -> Pair:
    """Compose stored affine pairs; right acts after left.

    The matrix product and transformed bias are each rounded and clipped once,
    exactly as specified by the lowering semantics.
    """
    a, b = left
    c, e = right
    d = len(b)
    p = tuple(
        tuple(
            quantize(sum(c[i][k] * a[k][j] for k in range(d)), width, scale, mode)
            for j in range(d)
        )
        for i in range(d)
    )
    z = tuple(
        quantize(sum(c[i][k] * b[k] for k in range(d)) + scale * e[i], width, scale, mode)
        for i in range(d)
    )
    return p, z


def compose_many(pairs: Sequence[Pair], *, width: int, scale: int, mode: str, association: str) -> Pair:
    if not pairs:
        raise ValueError("cannot compose an empty sequence")
    if len(pairs) == 1:
        return pairs[0]
    if association == "left":
        acc = pairs[0]
        for pair in pairs[1:]:
            acc = compose_pair(acc, pair, width=width, scale=scale, mode=mode)
        return acc
    if association == "right":
        acc = pairs[-1]
        for pair in reversed(pairs[:-1]):
            acc = compose_pair(pair, acc, width=width, scale=scale, mode=mode)
        return acc
    if association == "balanced":
        mid = len(pairs) // 2
        left = compose_many(pairs[:mid], width=width, scale=scale, mode=mode, association=association)
        right = compose_many(pairs[mid:], width=width, scale=scale, mode=mode, association=association)
        return compose_pair(left, right, width=width, scale=scale, mode=mode)
    raise ValueError(f"unknown association {association!r}")


def _normalize_inputs(instance: Instance, inputs: Sequence[Sequence[int]]) -> tuple[Vector, ...]:
    if len(inputs) != instance.horizon:
        raise ValueError("input sequence has wrong horizon")
    out: list[Vector] = []
    for t, row in enumerate(inputs):
        if len(row) != instance.dimension:
            raise ValueError("input vector has wrong dimension")
        v = tuple(_require_int(x, f"inputs[{t}][{i}]") for i, x in enumerate(row))
        for x, box in zip(v, instance.boxes[t]):
            if not box.lo <= x <= box.hi:
                raise ValueError("input lies outside declared box")
        out.append(v)
    return tuple(out)


def sequential_trace(instance: Instance, inputs: Sequence[Sequence[int]]) -> tuple[Vector, ...]:
    b = _normalize_inputs(instance, inputs)
    state = instance.h0
    trace: list[Vector] = []
    for matrix, bias in zip(instance.matrices, b):
        state = apply_pair((matrix, bias), state, width=instance.width, scale=instance.scale, mode=instance.rounding)
        trace.append(state)
    return tuple(trace)


def lowered_trace(instance: Instance, inputs: Sequence[Sequence[int]]) -> tuple[Vector, ...]:
    b = _normalize_inputs(instance, inputs)
    trace: list[Vector] = []
    carry = instance.h0
    for start in range(0, instance.horizon, instance.chunk_length):
        stop = min(instance.horizon, start + instance.chunk_length)
        pairs: list[Pair] = []
        last = carry
        for t in range(start, stop):
            pairs.append((instance.matrices[t], b[t]))
            summary = compose_many(
                pairs,
                width=instance.width,
                scale=instance.scale,
                mode=instance.rounding,
                association=instance.association,
            )
            last = apply_pair(summary, carry, width=instance.width, scale=instance.scale, mode=instance.rounding)
            trace.append(last)
        carry = last
    return tuple(trace)


def prefix_inputs(instance: Instance, flattened: Sequence[int], output_index: int, fill: int = 0) -> tuple[Vector, ...]:
    """Convert a flattened prefix into a full input sequence.

    Future values are filled with a legal point (box lower bound by default if
    `fill` is outside the box). Causality makes the future immaterial to the
    requested output.
    """
    _require_int(output_index, "output_index")
    _require_int(fill, "fill")
    if not 0 <= output_index < instance.horizon:
        raise IndexError(output_index)
    needed = (output_index + 1) * instance.dimension
    if len(flattened) != needed:
        raise ValueError(f"expected {needed} flattened values")
    rows: list[Vector] = []
    pos = 0
    for t in range(instance.horizon):
        row: list[int] = []
        for box in instance.boxes[t]:
            if t <= output_index:
                x = _require_int(flattened[pos], f"flattened[{pos}]"); pos += 1
                if not box.lo <= x <= box.hi:
                    raise ValueError("prefix lies outside declared box")
            else:
                x = fill if box.lo <= fill <= box.hi else box.lo
            row.append(x)
        rows.append(tuple(row))
    return tuple(rows)


def output_difference(instance: Instance, flattened_prefix: Sequence[int], output_index: int) -> Vector:
    inputs = prefix_inputs(instance, flattened_prefix, output_index)
    ref = sequential_trace(instance, inputs)[output_index]
    low = lowered_trace(instance, inputs)[output_index]
    return tuple(low[i] - ref[i] for i in range(instance.dimension))


def violates(instance: Instance, flattened_prefix: Sequence[int], output_index: int) -> bool:
    diff = output_difference(instance, flattened_prefix, output_index)
    return any(abs(diff[i]) > instance.tolerances[output_index][i] for i in range(instance.dimension))
