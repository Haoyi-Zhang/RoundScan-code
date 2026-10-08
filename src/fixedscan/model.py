"""Data model for bounded fixed-point affine scan instances.

The JSON representation is intentionally explicit and contains no executable
code.  Public JSON loading is strict: semantic integers must be JSON integers
(not booleans, floats, or numeric strings), schemas are closed, duplicate keys
and non-finite constants are rejected, and a byte limit is applied before
parsing.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

DEFAULT_MAX_JSON_BYTES = 16 * 1024 * 1024


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-standard JSON constant: {value}")


def _strict_int(value: object, label: str) -> int:
    # bool is a subclass of int, so an isinstance check is insufficient.
    if type(value) is not int:
        raise TypeError(f"{label} must be a JSON integer")
    return value


def _strict_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a string")
    return value


def _strict_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    return value


def _strict_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise TypeError(f"{label} has a non-string key")
    return value


def _check_keys(
    obj: dict[str, object],
    *,
    allowed: set[str],
    required: set[str],
    label: str,
) -> None:
    unknown = set(obj) - allowed
    missing = required - set(obj)
    if unknown:
        raise ValueError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise ValueError(f"{label} is missing field(s): {', '.join(sorted(missing))}")


def _int_vector(value: object, label: str) -> tuple[int, ...]:
    return tuple(
        _strict_int(item, f"{label}[{index}]")
        for index, item in enumerate(_strict_list(value, label))
    )


@dataclass(frozen=True)
class Box:
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if type(self.lo) is not int or type(self.hi) is not int:
            raise TypeError("box endpoints must be integers")
        if self.lo > self.hi:
            raise ValueError(f"empty box [{self.lo}, {self.hi}]")

    def values(self) -> range:
        return range(self.lo, self.hi + 1)

    def size(self) -> int:
        return self.hi - self.lo + 1

    def to_list(self) -> list[int]:
        return [self.lo, self.hi]

    @staticmethod
    def from_obj(obj: object, label: str = "box") -> "Box":
        values = _strict_list(obj, label)
        if len(values) != 2:
            raise ValueError(f"{label} must contain exactly two endpoints")
        return Box(
            _strict_int(values[0], f"{label}[0]"),
            _strict_int(values[1], f"{label}[1]"),
        )


@dataclass(frozen=True)
class Instance:
    name: str
    family: str
    width: int
    scale: int
    rounding: str
    dimension: int
    horizon: int
    chunk_length: int
    association: str
    h0: tuple[int, ...]
    matrices: tuple[tuple[tuple[int, ...], ...], ...]
    boxes: tuple[tuple[Box, ...], ...]
    tolerances: tuple[tuple[int, ...], ...]
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        for label, value in (
            ("name", self.name),
            ("family", self.family),
            ("rounding", self.rounding),
            ("association", self.association),
        ):
            if type(value) is not str:
                raise TypeError(f"{label} must be a string")
        for label, value in (
            ("width", self.width),
            ("scale", self.scale),
            ("dimension", self.dimension),
            ("horizon", self.horizon),
            ("chunk_length", self.chunk_length),
        ):
            if type(value) is not int:
                raise TypeError(f"{label} must be an integer")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be an object")
        if self.width < 2:
            raise ValueError("width must be at least two")
        if self.scale <= 0 or self.scale & (self.scale - 1):
            raise ValueError("scale must be a positive power of two")
        if self.scale >= 2 ** (self.width - 1):
            raise ValueError("scale does not fit positive signed word range")
        if self.rounding not in {"floor", "rne"}:
            raise ValueError("rounding must be floor or rne")
        if self.dimension <= 0 or self.horizon <= 0:
            raise ValueError("dimension and horizon must be positive")
        if self.chunk_length <= 0:
            raise ValueError("chunk length must be positive")
        if self.association not in {"left", "right", "balanced"}:
            raise ValueError("unsupported association")
        if len(self.h0) != self.dimension:
            raise ValueError("bad initial-state dimension")
        if len(self.matrices) != self.horizon or len(self.boxes) != self.horizon:
            raise ValueError("bad horizon")
        if len(self.tolerances) != self.horizon:
            raise ValueError("one tolerance vector is required per output")
        lo, hi = self.word_bounds
        for x in self.h0:
            if type(x) is not int:
                raise TypeError("initial-state entries must be integers")
            if not lo <= x <= hi:
                raise ValueError("initial state is outside the word range")
        for matrix in self.matrices:
            if len(matrix) != self.dimension:
                raise ValueError("bad matrix row count")
            for row in matrix:
                if len(row) != self.dimension:
                    raise ValueError("bad matrix column count")
                for x in row:
                    if type(x) is not int:
                        raise TypeError("matrix entries must be integers")
                    if not lo <= x <= hi:
                        raise ValueError("coefficient outside the word range")
        for step_boxes in self.boxes:
            if len(step_boxes) != self.dimension:
                raise ValueError("bad input-box dimension")
            for box in step_boxes:
                if not isinstance(box, Box):
                    raise TypeError("input boxes must be Box values")
                if box.lo < lo or box.hi > hi:
                    raise ValueError("input box outside the word range")
        for tol in self.tolerances:
            if len(tol) != self.dimension:
                raise ValueError("bad tolerance vector")
            for x in tol:
                if type(x) is not int:
                    raise TypeError("tolerance entries must be integers")
                if x < 0:
                    raise ValueError("bad tolerance vector")

    @property
    def word_bounds(self) -> tuple[int, int]:
        return -(2 ** (self.width - 1)), 2 ** (self.width - 1) - 1

    def prefix_boxes(self, output_index: int) -> tuple[Box, ...]:
        """Return flattened time-major boxes for output index 0..horizon-1."""
        if type(output_index) is not int or not 0 <= output_index < self.horizon:
            raise IndexError(output_index)
        return tuple(box for step in self.boxes[: output_index + 1] for box in step)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "width": self.width,
            "scale": self.scale,
            "rounding": self.rounding,
            "dimension": self.dimension,
            "horizon": self.horizon,
            "chunk_length": self.chunk_length,
            "association": self.association,
            "h0": list(self.h0),
            "matrices": [[list(row) for row in matrix] for matrix in self.matrices],
            "boxes": [[[box.lo, box.hi] for box in step] for step in self.boxes],
            "tolerances": [list(v) for v in self.tolerances],
            "metadata": self.metadata,
        }

    @staticmethod
    def from_dict(value: object) -> "Instance":
        obj = _strict_object(value, "instance")
        allowed = {
            "name", "family", "width", "scale", "rounding", "dimension",
            "horizon", "chunk_length", "association", "h0", "matrices",
            "boxes", "tolerances", "metadata",
        }
        required = allowed - {"family", "association", "metadata"}
        _check_keys(obj, allowed=allowed, required=required, label="instance")

        matrices_raw = _strict_list(obj["matrices"], "instance.matrices")
        matrices = tuple(
            tuple(
                _int_vector(row, f"instance.matrices[{t}][{i}]")
                for i, row in enumerate(_strict_list(matrix, f"instance.matrices[{t}]"))
            )
            for t, matrix in enumerate(matrices_raw)
        )
        boxes_raw = _strict_list(obj["boxes"], "instance.boxes")
        boxes = tuple(
            tuple(
                Box.from_obj(box, f"instance.boxes[{t}][{i}]")
                for i, box in enumerate(_strict_list(step, f"instance.boxes[{t}]"))
            )
            for t, step in enumerate(boxes_raw)
        )
        tolerances = tuple(
            _int_vector(row, f"instance.tolerances[{t}]")
            for t, row in enumerate(_strict_list(obj["tolerances"], "instance.tolerances"))
        )
        metadata = _strict_object(obj.get("metadata", {}), "instance.metadata")

        return Instance(
            name=_strict_str(obj["name"], "instance.name"),
            family=_strict_str(obj.get("family", "unspecified"), "instance.family"),
            width=_strict_int(obj["width"], "instance.width"),
            scale=_strict_int(obj["scale"], "instance.scale"),
            rounding=_strict_str(obj["rounding"], "instance.rounding"),
            dimension=_strict_int(obj["dimension"], "instance.dimension"),
            horizon=_strict_int(obj["horizon"], "instance.horizon"),
            chunk_length=_strict_int(obj["chunk_length"], "instance.chunk_length"),
            association=_strict_str(obj.get("association", "balanced"), "instance.association"),
            h0=_int_vector(obj["h0"], "instance.h0"),
            matrices=matrices,
            boxes=boxes,
            tolerances=tolerances,
            metadata=dict(metadata),
        )

    @staticmethod
    def load(path: str | Path, *, max_bytes: int = DEFAULT_MAX_JSON_BYTES) -> "Instance":
        if type(max_bytes) is not int or max_bytes < 1:
            raise ValueError("max_bytes must be a positive integer")
        p = Path(path)
        if p.stat().st_size > max_bytes:
            raise ValueError("instance exceeds input-size limit")
        with p.open("r", encoding="utf-8") as f:
            value = json.load(f, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
        return Instance.from_dict(value)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(self.to_dict(), f, indent=2, sort_keys=True, allow_nan=False)
            f.write("\n")
