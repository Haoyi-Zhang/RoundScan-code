"""Ordered cover certificate producer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import process_time, monotonic
from typing import Callable, Sequence
import json

from .arithmetic import output_difference
from .interval_analysis import independent_bounds
from .leaf_admission import exact_leaf_admissible
from .model import Box, Instance
from .symbolic import paired_bounds


class ResourceStop(RuntimeError):
    pass


@dataclass
class Budget:
    max_nodes: int
    deadline: float
    nodes: int = 0

    def check_time(self) -> None:
        if monotonic() > self.deadline:
            raise ResourceStop("wall-clock limit")

    def admit(self) -> None:
        if self.nodes >= self.max_nodes:
            raise ResourceStop("node limit")
        self.check_time()
        self.nodes += 1


def _bounds(instance: Instance, output: int, boxes: Sequence[Box], method: str) -> tuple[tuple[int, int], ...]:
    if method == "interval":
        return tuple((x.lo, x.hi) for x in independent_bounds(instance, output, boxes))
    if method in {"fresh", "syntactic", "residue"}:
        return tuple((x.lo, x.hi) for x in paired_bounds(instance, output, boxes, method))
    raise ValueError(method)


def _safe(bounds: Sequence[tuple[int, int]], tolerance: Sequence[int]) -> bool:
    return all(lo >= -tol and hi <= tol for (lo, hi), tol in zip(bounds, tolerance))


def _box_json(boxes: Sequence[Box]) -> list[list[int]]:
    return [[b.lo, b.hi] for b in boxes]


def _first_split(boxes: Sequence[Box]) -> int | None:
    for i, box in enumerate(boxes):
        if box.lo < box.hi:
            return i
    return None


def _search(instance: Instance, output: int, boxes: tuple[Box, ...], method: str, budget: Budget) -> tuple[dict, str, tuple[int, ...] | None]:
    budget.admit()
    bounds = _bounds(instance, output, boxes, method)
    budget.check_time()
    # Abstract safety is not enough to close a certificate leaf: the shipped
    # checker must also have an exact replay path for this box.  Otherwise keep
    # refining canonically; if the resource budget expires, the producer
    # returns ``unknown`` rather than a certificate the checker must reject.
    if _safe(bounds, instance.tolerances[output]) and exact_leaf_admissible(instance, output, boxes):
        return ({"kind": "safe", "box": _box_json(boxes), "bounds": [list(x) for x in bounds], "proof": "analysis"}, "safe", None)

    split = _first_split(boxes)
    if split is None:
        point = tuple(b.lo for b in boxes)
        diff = output_difference(instance, point, output)
        budget.check_time()
        if any(abs(diff[i]) > instance.tolerances[output][i] for i in range(instance.dimension)):
            return ({"kind": "witness", "box": _box_json(boxes), "point": list(point), "difference": list(diff)}, "witness", point)
        exact = tuple((x, x) for x in diff)
        return ({"kind": "safe", "box": _box_json(boxes), "bounds": [list(x) for x in exact], "proof": "exact-singleton"}, "safe", None)

    current = boxes[split]
    cut = (current.lo + current.hi) // 2
    left_boxes = list(boxes); left_boxes[split] = Box(current.lo, cut)
    left, left_status, witness = _search(instance, output, tuple(left_boxes), method, budget)
    node: dict = {
        "kind": "split", "box": _box_json(boxes), "coordinate": split, "cut": cut, "left": left
    }
    if left_status == "witness":
        return node, "witness", witness
    right_boxes = list(boxes); right_boxes[split] = Box(cut + 1, current.hi)
    right, right_status, witness = _search(instance, output, tuple(right_boxes), method, budget)
    node["right"] = right
    if right_status == "witness":
        return node, "witness", witness
    return node, "safe", None


def produce(instance: Instance, method: str, *, max_nodes: int = 20000, timeout_seconds: float = 8.0) -> dict:
    start_cpu = process_time(); start_wall = monotonic()
    budget = Budget(max_nodes=max_nodes, deadline=start_wall + timeout_seconds)
    roots: list[dict] = []
    status = "safe"
    witness: tuple[int, ...] | None = None
    earliest: int | None = None
    stop_reason: str | None = None
    try:
        for output in range(instance.horizon):
            tree, root_status, root_witness = _search(instance, output, instance.prefix_boxes(output), method, budget)
            budget.check_time()
            roots.append({"output": output, "tree": tree})
            if root_status == "witness":
                status = "violated"; witness = root_witness; earliest = output
                break
    except (ResourceStop, MemoryError) as exc:
        status = "unknown"
        stop_reason = str(exc)
    except RecursionError:
        # A finite canonical cover can still exceed the interpreter's stack.
        # An incomplete search is a resource stop, never a safety conclusion.
        status = "unknown"
        stop_reason = "recursion limit"
    return {
        "schema": "fixedscan-ordered-cover-1",
        "instance": instance.to_dict(),
        "method": method,
        "status": status,
        "roots": roots,
        "earliest_output": earliest,
        "witness": list(witness) if witness is not None else None,
        "diagnostics": {
            "nodes": budget.nodes,
            "cpu_seconds": process_time() - start_cpu,
            "wall_seconds": monotonic() - start_wall,
            "max_nodes": max_nodes,
            "timeout_seconds": timeout_seconds,
            "stop_reason": stop_reason,
        },
    }


def save_certificate(certificate: dict, path: str | Path) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(certificate, f, sort_keys=True, separators=(",", ":"))
        f.write("\n")
