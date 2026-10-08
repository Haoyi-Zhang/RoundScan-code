"""Independent exact checker for ordered cover certificates.

The checker intentionally treats certificate JSON as an adversarial input.  In
particular, semantic integers must be JSON integers rather than values that
Python's ``int`` constructor would silently coerce (booleans, integral floats,
or numeric strings).  The schema is closed, duplicate keys and non-finite JSON
constants are rejected, and explicit node/depth limits bound recursive input
processing.  These parser checks are separate from the mathematical
certificate contract enforced below.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from time import process_time
from typing import Any, Sequence
import json

from .checker_arithmetic import output_difference
from .exact_box import ExactBoxLimit, exact_box_range
from .model import Box, Instance


class CertificateError(ValueError):
    pass


# Engineering admission limits for the public checker.  The mathematical
# statements are parametric; these caps bound parser allocation and exact
# replay in the shipped implementation.
MAX_WIDTH = 32
MAX_SCALE = 256
MAX_DIMENSION = 32
MAX_HORIZON = 128
MAX_CHUNK_LENGTH = 64
DEFAULT_MAX_BYTES = 16 * 1024 * 1024


@dataclass
class CheckState:
    checked_nodes: int = 0
    max_nodes: int = 500_000
    max_depth: int = 512

    def enter(self, depth: int) -> None:
        if depth > self.max_depth:
            raise CertificateError("certificate tree exceeds depth limit")
        self.checked_nodes += 1
        if self.checked_nodes > self.max_nodes:
            raise CertificateError("certificate tree exceeds node limit")


def _strict_int(value: object, label: str) -> int:
    # ``bool`` is a subclass of ``int`` in Python, hence the exact type test.
    if type(value) is not int:
        raise CertificateError(f"{label} must be a JSON integer")
    return value


def _strict_str(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise CertificateError(f"{label} must be a string")
    return value


def _strict_list(value: object, label: str) -> list[object]:
    if not isinstance(value, list):
        raise CertificateError(f"{label} must be a list")
    return value


def _expect_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise CertificateError(f"{label} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise CertificateError(f"{label} has a non-string key")
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
        raise CertificateError(f"{label} has unknown field(s): {', '.join(sorted(unknown))}")
    if missing:
        raise CertificateError(f"{label} is missing field(s): {', '.join(sorted(missing))}")


def _parse_int_vector(value: object, label: str) -> tuple[int, ...]:
    return tuple(_strict_int(item, f"{label}[{i}]") for i, item in enumerate(_strict_list(value, label)))


def _parse_one_box(value: object, label: str) -> Box:
    items = _strict_list(value, label)
    if len(items) != 2:
        raise CertificateError(f"{label} must contain exactly two endpoints")
    try:
        return Box(_strict_int(items[0], f"{label}[0]"), _strict_int(items[1], f"{label}[1]"))
    except ValueError as exc:
        raise CertificateError(str(exc)) from exc


def _parse_box(obj: object) -> tuple[Box, ...]:
    return tuple(_parse_one_box(x, f"box[{i}]") for i, x in enumerate(_strict_list(obj, "box")))


def _parse_instance(value: object) -> Instance:
    obj = _expect_object(value, "instance")
    allowed = {
        "name", "family", "width", "scale", "rounding", "dimension", "horizon",
        "chunk_length", "association", "h0", "matrices", "boxes", "tolerances",
        "metadata",
    }
    required = allowed - {"family", "association", "metadata"}
    _check_keys(obj, allowed=allowed, required=required, label="instance")

    name = _strict_str(obj["name"], "instance.name")
    family = _strict_str(obj.get("family", "unspecified"), "instance.family")
    width = _strict_int(obj["width"], "instance.width")
    scale = _strict_int(obj["scale"], "instance.scale")
    rounding = _strict_str(obj["rounding"], "instance.rounding")
    dimension = _strict_int(obj["dimension"], "instance.dimension")
    horizon = _strict_int(obj["horizon"], "instance.horizon")
    chunk_length = _strict_int(obj["chunk_length"], "instance.chunk_length")
    association = _strict_str(obj.get("association", "balanced"), "instance.association")
    if width > MAX_WIDTH:
        raise CertificateError(f"instance.width exceeds checker limit {MAX_WIDTH}")
    if scale > MAX_SCALE:
        raise CertificateError(f"instance.scale exceeds checker limit {MAX_SCALE}")
    if dimension > MAX_DIMENSION:
        raise CertificateError(f"instance.dimension exceeds checker limit {MAX_DIMENSION}")
    if horizon > MAX_HORIZON:
        raise CertificateError(f"instance.horizon exceeds checker limit {MAX_HORIZON}")
    if chunk_length > MAX_CHUNK_LENGTH:
        raise CertificateError(f"instance.chunk_length exceeds checker limit {MAX_CHUNK_LENGTH}")
    h0 = _parse_int_vector(obj["h0"], "instance.h0")

    matrices_raw = _strict_list(obj["matrices"], "instance.matrices")
    matrices: list[tuple[tuple[int, ...], ...]] = []
    for t, matrix_value in enumerate(matrices_raw):
        matrix_raw = _strict_list(matrix_value, f"instance.matrices[{t}]")
        matrices.append(tuple(
            _parse_int_vector(row, f"instance.matrices[{t}][{i}]")
            for i, row in enumerate(matrix_raw)
        ))

    boxes_raw = _strict_list(obj["boxes"], "instance.boxes")
    boxes: list[tuple[Box, ...]] = []
    for t, step_value in enumerate(boxes_raw):
        step_raw = _strict_list(step_value, f"instance.boxes[{t}]")
        boxes.append(tuple(
            _parse_one_box(box, f"instance.boxes[{t}][{i}]")
            for i, box in enumerate(step_raw)
        ))

    tolerances_raw = _strict_list(obj["tolerances"], "instance.tolerances")
    tolerances = tuple(
        _parse_int_vector(row, f"instance.tolerances[{t}]")
        for t, row in enumerate(tolerances_raw)
    )
    metadata = obj.get("metadata", {})
    if not isinstance(metadata, dict):
        raise CertificateError("instance.metadata must be an object")

    try:
        return Instance(
            name=name,
            family=family,
            width=width,
            scale=scale,
            rounding=rounding,
            dimension=dimension,
            horizon=horizon,
            chunk_length=chunk_length,
            association=association,
            h0=h0,
            matrices=tuple(matrices),
            boxes=tuple(boxes),
            tolerances=tolerances,
            metadata=dict(metadata),
        )
    except (TypeError, ValueError) as exc:
        raise CertificateError(f"invalid embedded instance: {exc}") from exc


def _same_boxes(a: Sequence[Box], b: Sequence[Box]) -> bool:
    return len(a) == len(b) and all(x == y for x, y in zip(a, b))


def _first_split(boxes: Sequence[Box]) -> int | None:
    for i, box in enumerate(boxes):
        if box.lo < box.hi:
            return i
    return None


def _check_node(
    instance: Instance,
    output: int,
    node: object,
    expected_box: tuple[Box, ...],
    state: CheckState,
    depth: int = 1,
) -> tuple[str, tuple[int, ...] | None]:
    state.enter(depth)
    node = _expect_object(node, "node")
    if _parse_box(node.get("box")) != expected_box:
        raise CertificateError("node box does not match traversal box")
    kind = _strict_str(node.get("kind"), "node.kind")
    if kind == "safe":
        _check_keys(
            node,
            allowed={"kind", "box", "bounds", "proof"},
            required={"kind", "box", "bounds"},
            label="safe node",
        )
        bounds_obj = node.get("bounds")
        if not isinstance(bounds_obj, list) or len(bounds_obj) != instance.dimension:
            raise CertificateError("bad safe-leaf bounds")
        bounds: list[tuple[int, int]] = []
        for i, x in enumerate(bounds_obj):
            if not isinstance(x, list) or len(x) != 2:
                raise CertificateError("bad bound pair")
            lo = _strict_int(x[0], f"safe bounds[{i}][0]")
            hi = _strict_int(x[1], f"safe bounds[{i}][1]")
            if lo > hi:
                raise CertificateError("empty supplied bound")
            bounds.append((lo, hi))
        exact = exact_box_range(instance, output, expected_box)
        for i, ((elo, ehi), (lo, hi), tol) in enumerate(zip(exact, bounds, instance.tolerances[output])):
            if not (lo <= elo and ehi <= hi):
                raise CertificateError(f"supplied bound does not contain exact range at coordinate {i}")
            if lo < -tol or hi > tol:
                raise CertificateError(f"safe bound exceeds tolerance at coordinate {i}")
        return "safe", None
    if kind == "witness":
        _check_keys(
            node,
            allowed={"kind", "box", "point", "difference", "proof"},
            required={"kind", "box", "point", "difference"},
            label="witness node",
        )
        if any(box.lo != box.hi for box in expected_box):
            raise CertificateError("witness leaf is not singleton")
        point = _parse_int_vector(node.get("point"), "witness point")
        expected_point = tuple(box.lo for box in expected_box)
        if point != expected_point:
            raise CertificateError("witness point does not equal singleton box")
        diff = output_difference(instance, point, output)
        supplied = _parse_int_vector(node.get("difference"), "witness difference")
        if supplied != diff:
            raise CertificateError("witness difference does not replay")
        if not any(abs(diff[i]) > instance.tolerances[output][i] for i in range(instance.dimension)):
            raise CertificateError("witness does not violate the contract")
        return "witness", point
    if kind == "split":
        _check_keys(
            node,
            allowed={"kind", "box", "coordinate", "cut", "left", "right"},
            required={"kind", "box", "coordinate", "cut", "left"},
            label="split node",
        )
        coordinate = _strict_int(node.get("coordinate"), "split coordinate")
        canonical = _first_split(expected_box)
        if canonical is None or coordinate != canonical:
            raise CertificateError("split is not at the first nonsingleton coordinate")
        box = expected_box[coordinate]
        cut = _strict_int(node.get("cut"), "split cut")
        if not box.lo <= cut < box.hi:
            raise CertificateError("cut is not interior")
        left_box = list(expected_box); left_box[coordinate] = Box(box.lo, cut)
        left_status, witness = _check_node(instance, output, node.get("left"), tuple(left_box), state, depth + 1)
        if left_status == "witness":
            if "right" in node:
                raise CertificateError("right child must be absent after a left witness")
            return left_status, witness
        if "right" not in node:
            raise CertificateError("safe left child requires a right child")
        right_box = list(expected_box); right_box[coordinate] = Box(cut + 1, box.hi)
        return _check_node(instance, output, node["right"], tuple(right_box), state, depth + 1)
    raise CertificateError("unknown node kind")


def check(
    certificate: dict,
    expected_instance: Instance | None = None,
    *,
    max_nodes: int = 500_000,
    max_depth: int = 512,
) -> dict:
    start = process_time()
    state = CheckState(max_nodes=max_nodes, max_depth=max_depth)
    try:
        if type(max_nodes) is not int or max_nodes < 1:
            raise CertificateError("max_nodes must be a positive integer")
        if type(max_depth) is not int or max_depth < 1:
            raise CertificateError("max_depth must be a positive integer")
        certificate = _expect_object(certificate, "certificate")
        _check_keys(
            certificate,
            allowed={"schema", "instance", "method", "status", "roots", "earliest_output", "witness", "diagnostics"},
            required={"schema", "instance", "status", "roots", "earliest_output", "witness"},
            label="certificate",
        )
        if certificate.get("schema") != "fixedscan-ordered-cover-1":
            raise CertificateError("unsupported schema")
        if "method" in certificate:
            _strict_str(certificate["method"], "certificate.method")
        if "diagnostics" in certificate and not isinstance(certificate["diagnostics"], dict):
            raise CertificateError("certificate.diagnostics must be an object")
        instance = _parse_instance(certificate.get("instance"))
        if expected_instance is not None and instance.to_dict() != expected_instance.to_dict():
            raise CertificateError("embedded instance differs from supplied instance")
        status = _strict_str(certificate.get("status"), "certificate.status")
        roots = _strict_list(certificate.get("roots"), "certificate.roots")
        if status == "unknown":
            return {"accepted": False, "status": "unknown", "checked_nodes": 0, "cpu_seconds": process_time() - start, "error": "unknown producer outcome is not a proof"}
        if status == "safe":
            expected_root_count = instance.horizon
        elif status == "violated":
            earliest_output = _strict_int(certificate.get("earliest_output"), "certificate.earliest_output")
            if not 0 <= earliest_output < instance.horizon:
                raise CertificateError("certificate.earliest_output is outside the instance horizon")
            expected_root_count = earliest_output + 1
        else:
            raise CertificateError("unsupported claimed status")
        if len(roots) != expected_root_count:
            raise CertificateError("wrong number of roots")
        found_witness: tuple[int, ...] | None = None
        for t, root in enumerate(roots):
            root = _expect_object(root, f"root[{t}]")
            _check_keys(root, allowed={"output", "tree"}, required={"output", "tree"}, label=f"root[{t}]")
            if _strict_int(root.get("output"), f"root[{t}].output") != t:
                raise CertificateError("roots are not consecutive outputs")
            root_status, witness = _check_node(instance, t, root.get("tree"), instance.prefix_boxes(t), state)
            if t < len(roots) - 1 and root_status != "safe":
                raise CertificateError("an earlier root is not safe")
            if t == len(roots) - 1:
                if status == "safe" and root_status != "safe":
                    raise CertificateError("safe certificate ends in witness")
                if status == "violated" and root_status != "witness":
                    raise CertificateError("violation certificate lacks witness")
                found_witness = witness
        if status == "safe":
            if certificate.get("earliest_output") is not None or certificate.get("witness") is not None:
                raise CertificateError("safe certificate has violation metadata")
        elif status == "violated":
            if _strict_int(certificate.get("earliest_output"), "certificate.earliest_output") != len(roots) - 1:
                raise CertificateError("bad earliest-output metadata")
            if _parse_int_vector(certificate.get("witness"), "certificate.witness") != found_witness:
                raise CertificateError("top-level witness mismatch")
        return {"accepted": True, "status": status, "checked_nodes": state.checked_nodes, "cpu_seconds": process_time() - start, "error": None}
    except (CertificateError, ExactBoxLimit, IndexError, KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        return {"accepted": False, "status": "rejected", "checked_nodes": state.checked_nodes, "cpu_seconds": process_time() - start, "error": str(exc)}


def _strict_pairs(pairs: list[tuple[str, object]]) -> dict:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise CertificateError(f"duplicate JSON key: {key}")
        out[key] = value
    return out


def _reject_constant(value: str) -> None:
    raise CertificateError(f"non-standard JSON constant: {value}")


def load_instance_file(
    path: str | Path,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> Instance:
    """Load an independent instance through the same strict admitted schema."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise CertificateError("max_bytes must be a positive integer")
    p = Path(path)
    if p.stat().st_size > max_bytes:
        raise CertificateError("instance exceeds input-size limit")
    with p.open("r", encoding="utf-8") as f:
        value = json.load(f, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
    return _parse_instance(value)


def load_and_check(
    path: str | Path,
    expected_instance: Instance | None = None,
    *,
    max_bytes: int = DEFAULT_MAX_BYTES,
    max_nodes: int = 500_000,
    max_depth: int = 512,
) -> dict:
    p = Path(path)
    start = process_time()
    try:
        if type(max_bytes) is not int or max_bytes < 1:
            raise CertificateError("max_bytes must be a positive integer")
        if p.stat().st_size > max_bytes:
            raise CertificateError("certificate exceeds input-size limit")
        with p.open("r", encoding="utf-8") as f:
            obj = json.load(f, object_pairs_hook=_strict_pairs, parse_constant=_reject_constant)
        return check(obj, expected_instance, max_nodes=max_nodes, max_depth=max_depth)
    except (CertificateError, json.JSONDecodeError, OSError, ValueError, RecursionError) as exc:
        return {"accepted": False, "status": "rejected", "checked_nodes": 0, "cpu_seconds": process_time() - start, "error": str(exc)}
