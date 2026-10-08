"""Concrete semantics for the checker path.

This module intentionally duplicates the small arithmetic kernel used by the
producer.  It does not import :mod:`fixedscan.arithmetic`; cross-path tests
compare the two implementations on the complete development and frozen suites.
"""
from __future__ import annotations
from typing import Sequence
from .model import Instance

Vector = tuple[int, ...]
Matrix = tuple[tuple[int, ...], ...]
Pair = tuple[Matrix, Vector]


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


def word_bounds(width: int) -> tuple[int, int]:
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def clip(value: int, width: int) -> int:
    lo, hi = word_bounds(width)
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def round_div(numerator: int, scale: int, mode: str) -> int:
    quotient = numerator // scale
    remainder = numerator - quotient * scale
    if mode == "floor":
        return quotient
    if mode != "rne":
        raise ValueError("unsupported rounding mode")
    doubled = remainder + remainder
    if doubled > scale or (doubled == scale and quotient % 2):
        return quotient + 1
    return quotient


def quantize(numerator: int, width: int, scale: int, mode: str) -> int:
    return clip(round_div(numerator, scale, mode), width)


def apply_pair(pair: Pair, state: Vector, *, width: int, scale: int, mode: str) -> Vector:
    matrix, bias = pair
    result = []
    for row, offset in zip(matrix, bias):
        numerator = scale * offset
        for coefficient, value in zip(row, state):
            numerator += coefficient * value
        result.append(quantize(numerator, width, scale, mode))
    return tuple(result)


def compose_pair(left: Pair, right: Pair, *, width: int, scale: int, mode: str) -> Pair:
    a, b = left
    c, e = right
    d = len(b)
    product = []
    for i in range(d):
        row = []
        for j in range(d):
            numerator = 0
            for k in range(d):
                numerator += c[i][k] * a[k][j]
            row.append(quantize(numerator, width, scale, mode))
        product.append(tuple(row))
    offset = []
    for i in range(d):
        numerator = scale * e[i]
        for k in range(d):
            numerator += c[i][k] * b[k]
        offset.append(quantize(numerator, width, scale, mode))
    return tuple(product), tuple(offset)


def compose_many(pairs: Sequence[Pair], *, width: int, scale: int, mode: str, association: str) -> Pair:
    if not pairs:
        raise ValueError("empty composition")
    if len(pairs) == 1:
        return pairs[0]
    if association == "left":
        current = pairs[0]
        for pair in pairs[1:]:
            current = compose_pair(current, pair, width=width, scale=scale, mode=mode)
        return current
    if association == "right":
        current = pairs[-1]
        for pair in reversed(pairs[:-1]):
            current = compose_pair(pair, current, width=width, scale=scale, mode=mode)
        return current
    if association == "balanced":
        middle = len(pairs) // 2
        first = compose_many(pairs[:middle], width=width, scale=scale, mode=mode, association=association)
        second = compose_many(pairs[middle:], width=width, scale=scale, mode=mode, association=association)
        return compose_pair(first, second, width=width, scale=scale, mode=mode)
    raise ValueError("unsupported association")


def _complete_inputs(instance: Instance, prefix: Sequence[int], output: int) -> tuple[Vector, ...]:
    _require_int(output, "output")
    if not 0 <= output < instance.horizon:
        raise IndexError(output)
    need = (output + 1) * instance.dimension
    if len(prefix) != need:
        raise ValueError("wrong prefix length")
    rows = []
    cursor = 0
    for t, step_boxes in enumerate(instance.boxes):
        row = []
        for box in step_boxes:
            if t <= output:
                value = _require_int(prefix[cursor], f"prefix[{cursor}]"); cursor += 1
            else:
                value = box.lo
            if value < box.lo or value > box.hi:
                raise ValueError("prefix outside box")
            row.append(value)
        rows.append(tuple(row))
    return tuple(rows)


def sequential_trace(instance: Instance, inputs: Sequence[Sequence[int]]) -> tuple[Vector, ...]:
    if len(inputs) != instance.horizon:
        raise ValueError("input sequence has wrong horizon")
    state = instance.h0
    trace = []
    for t in range(instance.horizon):
        if len(inputs[t]) != instance.dimension:
            raise ValueError("input vector has wrong dimension")
        row = tuple(_require_int(x, f"inputs[{t}][{i}]") for i, x in enumerate(inputs[t]))
        state = apply_pair((instance.matrices[t], row), state,
                           width=instance.width, scale=instance.scale, mode=instance.rounding)
        trace.append(state)
    return tuple(trace)


def lowered_trace(instance: Instance, inputs: Sequence[Sequence[int]]) -> tuple[Vector, ...]:
    if len(inputs) != instance.horizon:
        raise ValueError("input sequence has wrong horizon")
    carry = instance.h0
    trace = []
    for start in range(0, instance.horizon, instance.chunk_length):
        stop = min(instance.horizon, start + instance.chunk_length)
        leaves: list[Pair] = []
        last = carry
        for t in range(start, stop):
            if len(inputs[t]) != instance.dimension:
                raise ValueError("input vector has wrong dimension")
            row = tuple(_require_int(x, f"inputs[{t}][{i}]") for i, x in enumerate(inputs[t]))
            leaves.append((instance.matrices[t], row))
            summary = compose_many(leaves, width=instance.width, scale=instance.scale,
                                   mode=instance.rounding, association=instance.association)
            last = apply_pair(summary, carry, width=instance.width,
                              scale=instance.scale, mode=instance.rounding)
            trace.append(last)
        carry = last
    return tuple(trace)


def output_difference(instance: Instance, prefix: Sequence[int], output: int) -> Vector:
    inputs = _complete_inputs(instance, prefix, output)
    reference = sequential_trace(instance, inputs)[output]
    lowered = lowered_trace(instance, inputs)[output]
    return tuple(a - b for a, b in zip(lowered, reference))
