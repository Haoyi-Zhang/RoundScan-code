"""Exact finite-state baseline for bounded verification."""
from __future__ import annotations

from itertools import product
from math import prod
from time import process_time, monotonic
from typing import Sequence

from .checker_arithmetic import output_difference
from .exact_box import ExactBoxLimit, exact_box_range
from .model import Box, Instance


def exact_baseline(instance: Instance, timeout_seconds: float = 12.0) -> dict:
    start_cpu = process_time(); start_wall = monotonic(); deadline = start_wall + timeout_seconds
    queries = 0
    for t in range(instance.horizon):
        if monotonic() > deadline:
            return _finish("unknown", None, None, queries, start_cpu, start_wall, "wall-clock limit")
        boxes = instance.prefix_boxes(t)
        try:
            ranges = exact_box_range(instance, t, boxes); queries += 1
        except (ExactBoxLimit, MemoryError) as exc:
            return _finish("unknown", None, None, queries, start_cpu, start_wall, str(exc))
        if monotonic() > deadline:
            return _finish("unknown", None, None, queries, start_cpu, start_wall, "wall-clock limit")
        if any(lo < -tol or hi > tol for (lo, hi), tol in zip(ranges, instance.tolerances[t])):
            try:
                witness = _lex_witness(instance, t, boxes, deadline)
            except (ExactBoxLimit, MemoryError) as exc:
                return _finish("unknown", None, None, queries, start_cpu, start_wall, str(exc))
            if monotonic() > deadline:
                return _finish("unknown", None, None, queries, start_cpu, start_wall, "wall-clock limit")
            if witness is None:
                return _finish("unknown", None, None, queries, start_cpu, start_wall, "witness search limit")
            return _finish("violated", t, witness, queries, start_cpu, start_wall, None)
    if monotonic() > deadline:
        return _finish("unknown", None, None, queries, start_cpu, start_wall, "wall-clock limit")
    return _finish("safe", None, None, queries, start_cpu, start_wall, None)


def _finish(status: str, earliest: int | None, witness: tuple[int, ...] | None, queries: int, start_cpu: float, start_wall: float, reason: str | None) -> dict:
    return {
        "status": status,
        "earliest_output": earliest,
        "witness": list(witness) if witness is not None else None,
        "queries": queries,
        "cpu_seconds": process_time() - start_cpu,
        "wall_seconds": monotonic() - start_wall,
        "stop_reason": reason,
    }


def _lex_witness(instance: Instance, output: int, boxes: Sequence[Box], deadline: float) -> tuple[int, ...] | None:
    size = prod(b.size() for b in boxes)
    if size <= 200_000:
        for point in product(*(range(b.lo, b.hi + 1) for b in boxes)):
            if monotonic() > deadline:
                return None
            diff = output_difference(instance, point, output)
            if any(abs(diff[i]) > instance.tolerances[output][i] for i in range(instance.dimension)):
                return tuple(point)
        return None
    # Exact range-guided left-first subdivision for large scalar boxes.
    current = list(boxes)
    for coordinate, box in enumerate(tuple(current)):
        while current[coordinate].lo < current[coordinate].hi:
            if monotonic() > deadline:
                return None
            b = current[coordinate]
            cut = (b.lo + b.hi) // 2
            left = list(current); left[coordinate] = Box(b.lo, cut)
            ranges = exact_box_range(instance, output, tuple(left))
            if any(lo < -tol or hi > tol for (lo, hi), tol in zip(ranges, instance.tolerances[output])):
                current = left
            else:
                current[coordinate] = Box(cut + 1, b.hi)
    point = tuple(b.lo for b in current)
    diff = output_difference(instance, point, output)
    return point if any(abs(diff[i]) > instance.tolerances[output][i] for i in range(instance.dimension)) else None
