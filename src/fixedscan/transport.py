"""Executable checks for rounding transport and carry-error composition.

The residual translation period (``q`` for floor, ``2q`` for nearest-even) and
the error modulus ``omega_q(D)`` are different objects.  The former justifies
residual sharing; the latter bounds how an incoming integer state difference
crosses rounding thresholds.  Clipping is handled only through monotonic
nonexpansiveness and the final word-span cap--it is not itself periodic.
"""
from __future__ import annotations

from typing import Sequence

from .checker_arithmetic import clip, round_div


def residual(n: int, scale: int, mode: str) -> int:
    return scale * round_div(n, scale, mode) - n


def translation_modulus(scale: int, mode: str) -> int:
    if scale <= 0:
        raise ValueError("scale must be positive")
    if mode == "floor":
        return scale
    if mode == "rne":
        return 2 * scale
    raise ValueError(mode)


def verify_translation(scale: int, mode: str, radius: int = 256) -> bool:
    modulus = translation_modulus(scale, mode)
    return all(
        residual(n + modulus, scale, mode) == residual(n, scale, mode)
        for n in range(-radius, radius + 1)
    )


def nearest_even_scale_counterexample(scale: int) -> tuple[int, int]:
    """Return n and n+scale whose RNE residuals differ (scale must be even)."""
    if scale % 2:
        raise ValueError("binary fixed-point scale is even except q=1")
    n = scale // 2
    assert residual(n, scale, "rne") != residual(n + scale, scale, "rne")
    return n, n + scale


def rounding_modulus(distance: int, scale: int, mode: str) -> int:
    """Closed form of omega_q(D) for integer floor or nearest-even rounding."""
    if type(distance) is not int or distance < 0:
        raise ValueError("distance must be a nonnegative integer")
    if type(scale) is not int or scale <= 0:
        raise ValueError("scale must be a positive integer")
    if mode == "floor":
        return (distance + scale - 1) // scale
    if mode != "rne":
        raise ValueError(mode)
    if scale == 1:
        return distance
    if scale % 2:
        raise ValueError("nearest-even closed form requires q=1 or even q")
    quotient, remainder = divmod(distance, scale)
    return quotient + int(remainder > 0) + int(remainder == 0 and quotient % 2 == 1)


def exhaustive_rounding_modulus(distance: int, scale: int, mode: str) -> int:
    """Compute omega_q(D) over one complete rounding-threshold period."""
    period = translation_modulus(scale, mode)
    return max(
        abs(round_div(n + distance, scale, mode) - round_div(n, scale, mode))
        for n in range(period)
    )


def carry_step_bound(
    matrix: Sequence[Sequence[int]],
    incoming_error: Sequence[int],
    scale: int,
    mode: str,
) -> tuple[int, ...]:
    """Apply the theorem's component map T_A to a nonnegative error vector."""
    if any(type(value) is not int or value < 0 for value in incoming_error):
        raise ValueError("incoming_error must contain nonnegative integers")
    dimension = len(incoming_error)
    if len(matrix) != dimension or any(len(row) != dimension for row in matrix):
        raise ValueError("matrix shape does not match incoming_error")
    return tuple(
        rounding_modulus(
            sum(abs(matrix[i][k]) * incoming_error[k] for k in range(dimension)),
            scale,
            mode,
        )
        for i in range(dimension)
    )


def iterate_carry_bound(
    matrices: Sequence[Sequence[Sequence[int]]],
    incoming_error: Sequence[int],
    scale: int,
    mode: str,
) -> tuple[int, ...]:
    bound = tuple(incoming_error)
    for matrix in matrices:
        bound = carry_step_bound(matrix, bound, scale, mode)
    return bound


def saturated_residual(n: int, width: int, scale: int, mode: str) -> int:
    return scale * clip(round_div(n, scale, mode), width) - n


def saturation_period_counterexample(width: int, scale: int, mode: str) -> tuple[int, int]:
    """Find a witness that clipping destroys residual translation periodicity."""
    modulus = translation_modulus(scale, mode)
    radius = (2 ** (width - 1) + 4) * scale
    for n in range(-radius, radius + 1):
        if saturated_residual(n + modulus, width, scale, mode) != saturated_residual(n, width, scale, mode):
            return n, n + modulus
    raise AssertionError("failed to find saturation-period counterexample")


def contraction_counterexample(scale: int, mode: str) -> tuple[int, int, int, int]:
    """Find a one-code threshold jump missed by floor(|a|D/q), with a=1,D=1."""
    if scale < 2:
        raise ValueError("counterexample needs scale at least two")
    naive = 1 // scale
    for state in range(-4 * scale, 4 * scale):
        actual = abs(round_div(state + 1, scale, mode) - round_div(state, scale, mode))
        if actual > naive:
            return state, state + 1, naive, actual
    raise AssertionError("failed to find contraction counterexample")
