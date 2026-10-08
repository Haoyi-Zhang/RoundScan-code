"""Executable witnesses for the rounding-only complexity constructions.

The routines are finite construction tests, not a mechanized complexity proof.
They instantiate the two reductions used in the accompanying hand proof.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Sequence
from .model import Box, Instance


def _require_int(value: object, label: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an integer")
    return value


@dataclass(frozen=True)
class Gate:
    kind: str                 # "and" or "not"
    left: int
    right: int | None = None

    def __post_init__(self) -> None:
        if type(self.kind) is not str or self.kind not in {"and", "not"}:
            raise ValueError("gate kind must be 'and' or 'not'")
        _require_int(self.left, "gate.left")
        if self.kind == "and":
            if self.right is None:
                raise ValueError("AND needs two arguments")
            _require_int(self.right, "gate.right")
        elif self.right is not None:
            raise ValueError("NOT gate must not have a right argument")


def _identity(d: int, q: int) -> list[list[int]]:
    return [[q if i == j else 0 for j in range(d)] for i in range(d)]


def _freeze(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    return tuple(
        tuple(_require_int(x, f"matrix[{i}][{j}]") for j, x in enumerate(row))
        for i, row in enumerate(matrix)
    )


def compile_circuit(num_inputs: int, gates: Sequence[Gate], output_register: int, mode: str) -> Instance:
    """Compile an AND/NOT circuit to the fixed-width all-output reduction.

    Registers 0..num_inputs-1 hold variables, successive registers hold gates,
    and the final register is the annihilator scratch.  Every ordinary update is
    followed by an identity update so two-step summary lowering is exact until
    the final annihilating chunk.
    """
    _require_int(num_inputs, "num_inputs")
    _require_int(output_register, "output_register")
    if num_inputs < 1:
        raise ValueError("num_inputs must be positive")
    if mode not in {"floor", "rne"}:
        raise ValueError(mode)
    for gate_index, gate in enumerate(gates):
        if not isinstance(gate, Gate):
            raise TypeError("gates must contain Gate values")
        available = num_inputs + gate_index
        if not 0 <= gate.left < available:
            raise ValueError("gate.left references an unavailable register")
        if gate.right is not None and not 0 <= gate.right < available:
            raise ValueError("gate.right references an unavailable register")
    if not 0 <= output_register < num_inputs + len(gates):
        raise ValueError("output_register references an unavailable register")
    q, width = 2, 3
    d = num_inputs + len(gates) + 1
    scratch = d - 1
    matrices: list[tuple[tuple[int, ...], ...]] = []
    boxes: list[tuple[Box, ...]] = []

    # Load all Boolean variables through the affine biases; all state is zero.
    load = [[0 for _ in range(d)] for _ in range(d)]
    matrices.append(_freeze(load))
    boxes.append(tuple(Box(0, 1) if i < num_inputs else Box(0, 0) for i in range(d)))
    matrices.append(_freeze(_identity(d, q)))
    boxes.append(tuple(Box(0, 0) for _ in range(d)))

    for gate_index, gate in enumerate(gates):
        target = num_inputs + gate_index
        m = _identity(d, q)
        m[target] = [0 for _ in range(d)]
        bias = [Box(0, 0) for _ in range(d)]
        if gate.kind == "and":
            if gate.right is None:
                raise ValueError("AND needs two arguments")
            if gate.left == gate.right:
                m[target][gate.left] = q
            else:
                m[target][gate.left] = 1
                m[target][gate.right] = 1
        elif gate.kind == "not":
            m[target][gate.left] = -q
            bias[target] = Box(1, 1)
        else:
            raise ValueError(f"unknown gate {gate.kind}")
        matrices.append(_freeze(m)); boxes.append(tuple(bias))
        matrices.append(_freeze(_identity(d, q))); boxes.append(tuple(Box(0, 0) for _ in range(d)))

    # Annihilator first step: preserve x and set scratch = round(x/2).
    first = _identity(d, q)
    first[scratch] = [0 for _ in range(d)]
    first[scratch][output_register] = 1
    matrices.append(_freeze(first)); boxes.append(tuple(Box(0, 0) for _ in range(d)))
    # Second step: x <- x - 2 scratch.  Coefficient -4 fits signed 3-bit.
    second = _identity(d, q)
    second[output_register] = [0 for _ in range(d)]
    second[output_register][output_register] = q
    second[output_register][scratch] = -2 * q
    matrices.append(_freeze(second)); boxes.append(tuple(Box(0, 0) for _ in range(d)))

    horizon = len(matrices)
    return Instance(
        name=f"circuit-{mode}", family="complexity-reduction", width=width, scale=q,
        rounding=mode, dimension=d, horizon=horizon, chunk_length=2,
        association="balanced", h0=tuple(0 for _ in range(d)),
        matrices=tuple(matrices), boxes=tuple(boxes),
        tolerances=tuple(tuple(0 for _ in range(d)) for _ in range(horizon)),
        metadata={"num_inputs": num_inputs, "gates": len(gates), "output_register": output_register},
    )


def circuit_inputs(instance: Instance, assignment: Sequence[int]) -> tuple[tuple[int, ...], ...]:
    num_inputs = _require_int(instance.metadata.get("num_inputs"), "instance.metadata.num_inputs")
    if len(assignment) != num_inputs:
        raise ValueError("wrong assignment length")
    values = tuple(_require_int(x, f"assignment[{i}]") for i, x in enumerate(assignment))
    if any(x not in {0, 1} for x in values):
        raise ValueError("circuit assignment must be Boolean")
    rows = []
    for t, step in enumerate(instance.boxes):
        if t == 0:
            row = tuple(values[i] if i < len(values) else 0 for i in range(instance.dimension))
        else:
            row = tuple(box.lo for box in step)
        rows.append(row)
    return tuple(rows)


def evaluate_circuit(assignment: Sequence[int], gates: Sequence[Gate]) -> tuple[int, ...]:
    values = [_require_int(x, f"assignment[{i}]") for i, x in enumerate(assignment)]
    if any(x not in {0, 1} for x in values):
        raise ValueError("circuit assignment must be Boolean")
    for gate in gates:
        if gate.kind == "and":
            assert gate.right is not None
            values.append(values[gate.left] & values[gate.right])
        elif gate.kind == "not":
            values.append(1 - values[gate.left])
        else:
            raise ValueError(gate.kind)
    return tuple(values)


# ----- Five-coordinate packed CNF construction -----
X, E, C, P, F = range(5)


def _operation_matrix(row_updates: dict[int, dict[int, int]], q: int = 2) -> tuple[tuple[int, ...], ...]:
    m = _identity(5, q)
    for row, coefficients in row_updates.items():
        m[row] = [0] * 5
        for col, value in coefficients.items():
            m[row][col] = value
    return _freeze(m)


def compile_packed_cnf(num_variables: int, clauses: Sequence[Sequence[int]]) -> Instance:
    """Compile CNF to the five-coordinate, final-output construction.

    Literals use 1-based variable indices; negative values denote negation.
    """
    _require_int(num_variables, "num_variables")
    if num_variables < 1:
        raise ValueError("need at least one variable")
    normalized_clauses: list[tuple[int, ...]] = []
    for clause_index, clause in enumerate(clauses):
        normalized: list[int] = []
        for literal_index, literal in enumerate(clause):
            literal = _require_int(literal, f"clauses[{clause_index}][{literal_index}]")
            if literal == 0 or abs(literal) > num_variables:
                raise ValueError("literal outside variable range")
            normalized.append(literal)
        normalized_clauses.append(tuple(normalized))
    width = max(3, num_variables + 1)
    q = 2
    matrices: list[tuple[tuple[int, ...], ...]] = []
    boxes: list[tuple[Box, ...]] = []

    def append_step(matrix, bias: Sequence[Box] | None = None, pad_identity: bool = True) -> None:
        matrices.append(matrix)
        boxes.append(tuple(bias) if bias is not None else tuple(Box(0, 0) for _ in range(5)))
        if pad_identity:
            matrices.append(_freeze(_identity(5, q)))
            boxes.append(tuple(Box(0, 0) for _ in range(5)))

    # Load packed assignment X and initialize P=1.
    load = [[0] * 5 for _ in range(5)]
    load_bias = [Box(0, (1 << num_variables) - 1), Box(0, 0), Box(0, 0), Box(1, 1), Box(0, 0)]
    append_step(_freeze(load), load_bias)

    def set_zero(register: int) -> None:
        append_step(_operation_matrix({register: {}}))

    def copy_x_to_e() -> None:
        append_step(_operation_matrix({E: {X: q}}))

    def halve_e() -> None:
        append_step(_operation_matrix({E: {E: 1}}))

    def annihilate(register: int) -> None:
        # This is a full two-step chunk and therefore must not receive padding.
        first = _operation_matrix({F: {register: 1}})
        second = _operation_matrix({register: {register: q, F: -2 * q}})
        matrices.append(first); boxes.append(tuple(Box(0, 0) for _ in range(5)))
        matrices.append(second); boxes.append(tuple(Box(0, 0) for _ in range(5)))

    def negate(register: int) -> None:
        b = [Box(0, 0) for _ in range(5)]; b[register] = Box(1, 1)
        append_step(_operation_matrix({register: {register: -q}}), b)

    def and_into(target: int, left: int, right: int) -> None:
        coeffs: dict[int, int]
        if left == right:
            coeffs = {left: q}
        else:
            coeffs = {left: 1, right: 1}
        append_step(_operation_matrix({target: coeffs}))

    def or_c_e() -> None:
        # C or E = not((not C) and (not E)).
        negate(C); negate(E); and_into(C, C, E); negate(C)

    for clause in normalized_clauses:
        if not clause:
            set_zero(C)
            and_into(P, P, C)
            continue
        set_zero(C)
        for literal in clause:
            variable = abs(literal)
            copy_x_to_e()
            for _ in range(variable - 1):
                halve_e()
            annihilate(E)
            if literal < 0:
                negate(E)
            or_c_e()
        and_into(P, P, C)
    annihilate(P)

    horizon = len(matrices)
    span = (1 << width) - 1
    tolerances = [tuple(span for _ in range(5)) for _ in range(horizon)]
    last = list(tolerances[-1]); last[P] = 0; tolerances[-1] = tuple(last)
    return Instance(
        name="packed-cnf", family="constant-state-reduction", width=width, scale=q,
        rounding="floor", dimension=5, horizon=horizon, chunk_length=2,
        association="balanced", h0=(0, 0, 0, 0, 0), matrices=tuple(matrices),
        boxes=tuple(boxes), tolerances=tuple(tolerances),
        metadata={"num_variables": num_variables, "clauses": [list(x) for x in normalized_clauses]},
    )


def packed_inputs(instance: Instance, code: int) -> tuple[tuple[int, ...], ...]:
    code = _require_int(code, "code")
    num_variables = _require_int(instance.metadata.get("num_variables"), "instance.metadata.num_variables")
    max_code = (1 << num_variables) - 1
    if not 0 <= code <= max_code:
        raise ValueError("assignment code outside range")
    rows = []
    for t, step in enumerate(instance.boxes):
        if t == 0:
            rows.append((code, 0, 0, 1, 0))
        else:
            rows.append(tuple(box.lo for box in step))
    return tuple(rows)


def cnf_value(code: int, num_variables: int, clauses: Sequence[Sequence[int]]) -> int:
    code = _require_int(code, "code")
    num_variables = _require_int(num_variables, "num_variables")
    if num_variables < 1 or not 0 <= code < (1 << num_variables):
        raise ValueError("assignment code outside range")
    normalized_clauses: list[tuple[int, ...]] = []
    for clause_index, clause in enumerate(clauses):
        normalized: list[int] = []
        for literal_index, literal in enumerate(clause):
            literal = _require_int(literal, f"clauses[{clause_index}][{literal_index}]")
            if literal == 0 or abs(literal) > num_variables:
                raise ValueError("literal outside variable range")
            normalized.append(literal)
        normalized_clauses.append(tuple(normalized))

    def bit(variable: int) -> int:
        return (code >> (variable - 1)) & 1
    for clause in normalized_clauses:
        if not any((bit(abs(lit)) if lit > 0 else 1 - bit(abs(lit))) for lit in clause):
            return 0
    return 1
