"""Certificate-leaf admission shared by producer and exact checker.

This module contains only the syntactic/size predicate for the current
certificate language.  It deliberately performs no arithmetic reasoning.  A
producer may use it to avoid emitting an abstractly safe leaf that the exact
checker cannot replay; the checker remains responsible for recomputing the
exact range of every admitted leaf.
"""
from __future__ import annotations

from math import prod
from typing import Sequence

from .model import Box, Instance

DEFAULT_ENUMERATION_LIMIT = 100_000


def box_size(boxes: Sequence[Box]) -> int:
    return prod(box.size() for box in boxes)


def exact_leaf_admissible(
    instance: Instance,
    output_index: int,
    boxes: Sequence[Box],
    *,
    enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
) -> bool:
    """Return whether the shipped exact checker has a replay path for a leaf.

    Every scalar leaf is admitted because the checker has a finite-state scalar
    dynamic program.  A multidimensional leaf is admitted only when complete
    enumeration is within ``enumeration_limit``.  This predicate is about the
    current executable certificate language, not mathematical safety.
    """
    if type(enumeration_limit) is not int or enumeration_limit < 1:
        raise ValueError("enumeration_limit must be a positive integer")
    expected = (output_index + 1) * instance.dimension
    if len(boxes) != expected:
        raise ValueError("wrong prefix box dimension")
    return instance.dimension == 1 or box_size(boxes) <= enumeration_limit
