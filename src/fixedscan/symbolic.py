"""Paired symbolic enclosure with explicit rounding-residual sharing.

This module implements the producer path. It uses two descriptions of a value:
(1) an exact integer affine expression over input/quotient/clipped-value atoms,
and (2) a rational affine expression over input and residual atoms. Atom ranges
are treated as a Cartesian overapproximation. Three policies differ only in the
key used for residual atoms: fresh, exact syntactic numerator, or the sound
rounding-translation class modulo q (floor) / 2q (nearest-even).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from math import gcd
from typing import Iterable, Mapping, Sequence

from .arithmetic import Matrix, round_div, word_bounds
from .model import Box, Instance


def floor_fraction(x: Fraction) -> int:
    return x.numerator // x.denominator


def ceil_fraction(x: Fraction) -> int:
    return -((-x.numerator) // x.denominator)


@dataclass(frozen=True)
class Interval:
    lo: int
    hi: int

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"empty interval [{self.lo}, {self.hi}]")

    def add(self, other: "Interval") -> "Interval":
        return Interval(self.lo + other.lo, self.hi + other.hi)

    def scale(self, coefficient: int) -> "Interval":
        a = coefficient * self.lo
        b = coefficient * self.hi
        return Interval(min(a, b), max(a, b))

    def intersect(self, other: "Interval") -> "Interval":
        lo, hi = max(self.lo, other.lo), min(self.hi, other.hi)
        if lo > hi:
            raise AssertionError(f"inconsistent sound enclosures {self} and {other}")
        return Interval(lo, hi)


@dataclass(frozen=True)
class Affine:
    const: Fraction = Fraction(0)
    coeffs: tuple[tuple[str, Fraction], ...] = ()

    @staticmethod
    def constant(value: int | Fraction) -> "Affine":
        return Affine(Fraction(value), ())

    @staticmethod
    def atom(name: str, coefficient: int | Fraction = 1) -> "Affine":
        c = Fraction(coefficient)
        return Affine(Fraction(0), ((name, c),)) if c else Affine.constant(0)

    @staticmethod
    def make(const: int | Fraction, mapping: Mapping[str, int | Fraction]) -> "Affine":
        items = tuple(sorted((k, Fraction(v)) for k, v in mapping.items() if Fraction(v) != 0))
        return Affine(Fraction(const), items)

    def mapping(self) -> dict[str, Fraction]:
        return dict(self.coeffs)

    def __add__(self, other: "Affine") -> "Affine":
        d = self.mapping()
        for k, v in other.coeffs:
            d[k] = d.get(k, Fraction(0)) + v
            if d[k] == 0:
                del d[k]
        return Affine.make(self.const + other.const, d)

    def __sub__(self, other: "Affine") -> "Affine":
        return self + other.scale(Fraction(-1))

    def scale(self, coefficient: int | Fraction) -> "Affine":
        c = Fraction(coefficient)
        if c == 0:
            return Affine.constant(0)
        return Affine(self.const * c, tuple((k, v * c) for k, v in self.coeffs))

    def key(self) -> tuple[tuple[int, int], tuple[tuple[str, int, int], ...]]:
        return (
            (self.const.numerator, self.const.denominator),
            tuple((k, v.numerator, v.denominator) for k, v in self.coeffs),
        )

    def integer_mod_key(self, modulus: int) -> tuple[int, tuple[tuple[str, int], ...]]:
        if self.const.denominator != 1 or any(v.denominator != 1 for _, v in self.coeffs):
            raise ValueError("modular key requires an integer affine expression")
        return (
            int(self.const) % modulus,
            tuple((k, int(v) % modulus) for k, v in self.coeffs if int(v) % modulus),
        )

    def bounds(self, atom_bounds: Mapping[str, Interval]) -> tuple[Fraction, Fraction]:
        lo = hi = self.const
        for name, coefficient in self.coeffs:
            bound = atom_bounds[name]
            a = coefficient * bound.lo
            b = coefficient * bound.hi
            lo += min(a, b)
            hi += max(a, b)
        return lo, hi


@dataclass(frozen=True)
class SymVal:
    z: Affine
    e: Affine
    interval: Interval

    @staticmethod
    def constant(value: int) -> "SymVal":
        a = Affine.constant(value)
        return SymVal(a, a, Interval(value, value))

    @staticmethod
    def atom(name: str, bound: Interval) -> "SymVal":
        a = Affine.atom(name)
        return SymVal(a, a, bound)

    def scale(self, coefficient: int) -> "SymVal":
        return SymVal(self.z.scale(coefficient), self.e.scale(coefficient), self.interval.scale(coefficient))

    def add(self, other: "SymVal") -> "SymVal":
        return SymVal(self.z + other.z, self.e + other.e, self.interval.add(other.interval))


@dataclass
class Context:
    width: int
    scale: int
    mode: str
    policy: str
    atom_bounds: dict[str, Interval] = field(default_factory=dict)
    quotient_cache: dict[object, str] = field(default_factory=dict)
    residual_cache: dict[object, str] = field(default_factory=dict)
    clip_cache: dict[object, str] = field(default_factory=dict)
    counter: int = 0

    def new_atom(self, prefix: str, bound: Interval) -> str:
        name = f"{prefix}{self.counter}"
        self.counter += 1
        self.atom_bounds[name] = bound
        return name

    def input_atom(self, name: str, bound: Interval) -> SymVal:
        self.atom_bounds[name] = bound
        return SymVal.atom(name, bound)

    @property
    def modulus(self) -> int:
        return self.scale if self.mode == "floor" else 2 * self.scale

    def affine_integer_interval(self, expr: Affine) -> Interval:
        lo, hi = expr.bounds(self.atom_bounds)
        if lo.denominator != 1 or hi.denominator != 1:
            raise AssertionError("exact integer expression has fractional interval endpoint")
        return Interval(int(lo), int(hi))

    def residual_bound(self, expr: Affine) -> Interval:
        residues = possible_residues(expr, self.atom_bounds, self.modulus)
        vals = [self.scale * round_div(r, self.scale, self.mode) - r for r in residues]
        return Interval(min(vals), max(vals))

    def quantize(self, numerator: SymVal) -> SymVal:
        z_range = self.affine_integer_interval(numerator.z)
        q_range = Interval(
            round_div(z_range.lo, self.scale, self.mode),
            round_div(z_range.hi, self.scale, self.mode),
        )

        if z_range.lo == z_range.hi:
            z_out = Affine.constant(round_div(z_range.lo, self.scale, self.mode))
        else:
            qkey = numerator.z.key()
            name = self.quotient_cache.get(qkey)
            if name is None:
                name = self.new_atom("j", q_range)
                self.quotient_cache[qkey] = name
            z_out = Affine.atom(name)

        if self.policy == "fresh":
            rkey: object = ("fresh", self.counter)
        elif self.policy == "syntactic":
            rkey = ("syntactic", numerator.z.key())
        elif self.policy == "residue":
            rkey = ("residue", numerator.z.integer_mod_key(self.modulus))
        else:
            raise ValueError(f"unknown residual policy {self.policy!r}")
        rname = self.residual_cache.get(rkey)
        if rname is None:
            rname = self.new_atom("r", self.residual_bound(numerator.z))
            self.residual_cache[rkey] = rname
        e_out = numerator.e.scale(Fraction(1, self.scale)) + Affine.atom(rname, Fraction(1, self.scale))
        elo, ehi = e_out.bounds(self.atom_bounds)
        e_range = Interval(ceil_fraction(elo), floor_fraction(ehi))
        interval = q_range.intersect(e_range)

        lo_word, hi_word = word_bounds(self.width)
        if interval.hi < lo_word:
            return SymVal.constant(lo_word)
        if interval.lo > hi_word:
            return SymVal.constant(hi_word)
        clipped = Interval(max(lo_word, interval.lo), min(hi_word, interval.hi))
        if clipped == interval:
            return SymVal(z_out, e_out, interval)
        if clipped.lo == clipped.hi:
            return SymVal.constant(clipped.lo)
        ckey = ("clip", z_out.key())
        cname = self.clip_cache.get(ckey)
        if cname is None:
            cname = self.new_atom("c", clipped)
            self.clip_cache[ckey] = cname
        a = Affine.atom(cname)
        return SymVal(a, a, clipped)


def possible_residues(expr: Affine, atom_bounds: Mapping[str, Interval], modulus: int) -> set[int]:
    """Sound residue overapproximation for an integer affine expression."""
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    if expr.const.denominator != 1 or any(v.denominator != 1 for _, v in expr.coeffs):
        raise ValueError("residue analysis requires integer coefficients")
    residues = {int(expr.const) % modulus}
    for name, coefficient_fraction in expr.coeffs:
        coefficient = int(coefficient_fraction)
        bound = atom_bounds[name]
        period = modulus // gcd(abs(coefficient), modulus) if coefficient else 1
        count = min(bound.hi - bound.lo + 1, period)
        atom_residues = {(coefficient * (bound.lo + i)) % modulus for i in range(count)}
        residues = {(a + b) % modulus for a in residues for b in atom_residues}
        if len(residues) == modulus:
            break
    return residues


def sum_sym(values: Iterable[SymVal]) -> SymVal:
    acc = SymVal.constant(0)
    for value in values:
        acc = acc.add(value)
    return acc


SymPair = tuple[Matrix, tuple[SymVal, ...]]


def compose_sym(left: SymPair, right: SymPair, context: Context) -> SymPair:
    a, b = left
    c, e = right
    d = len(b)
    # Matrix summaries are input-independent and evaluated concretely. Reuse the
    # context quantizer on constants so arithmetic stays exactly aligned.
    p_rows: list[tuple[int, ...]] = []
    for i in range(d):
        row: list[int] = []
        for j in range(d):
            val = context.quantize(SymVal.constant(sum(c[i][k] * a[k][j] for k in range(d))))
            if val.interval.lo != val.interval.hi:
                raise AssertionError("constant matrix entry did not remain constant")
            row.append(val.interval.lo)
        p_rows.append(tuple(row))
    z = tuple(
        context.quantize(
            sum_sym([b[k].scale(c[i][k]) for k in range(d)] + [e[i].scale(context.scale)])
        )
        for i in range(d)
    )
    return tuple(p_rows), z


def compose_many_sym(pairs: Sequence[SymPair], context: Context, association: str) -> SymPair:
    if not pairs:
        raise ValueError("empty composition")
    if len(pairs) == 1:
        return pairs[0]
    if association == "left":
        acc = pairs[0]
        for pair in pairs[1:]:
            acc = compose_sym(acc, pair, context)
        return acc
    if association == "right":
        acc = pairs[-1]
        for pair in reversed(pairs[:-1]):
            acc = compose_sym(pair, acc, context)
        return acc
    if association == "balanced":
        mid = len(pairs) // 2
        return compose_sym(
            compose_many_sym(pairs[:mid], context, association),
            compose_many_sym(pairs[mid:], context, association),
            context,
        )
    raise ValueError(association)


def apply_sym(matrix: Matrix, bias: Sequence[SymVal], state: Sequence[SymVal], context: Context) -> tuple[SymVal, ...]:
    d = len(state)
    return tuple(
        context.quantize(
            sum_sym([state[k].scale(matrix[i][k]) for k in range(d)] + [bias[i].scale(context.scale)])
        )
        for i in range(d)
    )


def paired_bounds(instance: Instance, output_index: int, flat_boxes: Sequence[Box], policy: str) -> tuple[Interval, ...]:
    expected = (output_index + 1) * instance.dimension
    if len(flat_boxes) != expected:
        raise ValueError(f"expected {expected} prefix boxes")
    context = Context(instance.width, instance.scale, instance.rounding, policy)
    inputs: list[tuple[SymVal, ...]] = []
    pos = 0
    for t in range(output_index + 1):
        row: list[SymVal] = []
        for i in range(instance.dimension):
            box = flat_boxes[pos]; pos += 1
            row.append(context.input_atom(f"u{t}_{i}", Interval(box.lo, box.hi)))
        inputs.append(tuple(row))

    ref: tuple[SymVal, ...] = tuple(SymVal.constant(x) for x in instance.h0)
    ref_trace: list[tuple[SymVal, ...]] = []
    for t in range(output_index + 1):
        ref = apply_sym(instance.matrices[t], inputs[t], ref, context)
        ref_trace.append(ref)

    lowered_trace: list[tuple[SymVal, ...]] = []
    carry: tuple[SymVal, ...] = tuple(SymVal.constant(x) for x in instance.h0)
    for start in range(0, output_index + 1, instance.chunk_length):
        stop = min(output_index + 1, start + instance.chunk_length)
        pairs: list[SymPair] = []
        last = carry
        for t in range(start, stop):
            pairs.append((instance.matrices[t], inputs[t]))
            summary = compose_many_sym(pairs, context, instance.association)
            last = apply_sym(summary[0], summary[1], carry, context)
            lowered_trace.append(last)
        carry = last

    ref_out = ref_trace[output_index]
    low_out = lowered_trace[output_index]
    results: list[Interval] = []
    for lo, re in zip(low_out, ref_out):
        ez = lo.z - re.z
        ee = lo.e - re.e
        zlo, zhi = ez.bounds(context.atom_bounds)
        elo, ehi = ee.bounds(context.atom_bounds)
        z_range = Interval(floor_fraction(zlo), ceil_fraction(zhi))
        e_range = Interval(floor_fraction(elo), ceil_fraction(ehi))
        direct = Interval(lo.interval.lo - re.interval.hi, lo.interval.hi - re.interval.lo)
        results.append(z_range.intersect(e_range).intersect(direct))
    return tuple(results)
