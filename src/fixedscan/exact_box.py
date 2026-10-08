"""Independent exact box evaluator used by the certificate checker.

Small multidimensional boxes are enumerated. Large scalar prefixes are handled
by a finite-state dynamic program over the reference state, chunk carry, and
current chunk input tuple. This module intentionally does not import the producer's
abstract interpretation.
"""
from __future__ import annotations

from itertools import product
from typing import Sequence

from .checker_arithmetic import Pair, apply_pair, compose_many, output_difference
from .leaf_admission import DEFAULT_ENUMERATION_LIMIT, box_size, exact_leaf_admissible
from .model import Box, Instance


class ExactBoxLimit(RuntimeError):
    pass


# Each layer owns its summaries. At the cap, compute without admitting more.
_LAYER_SUMMARY_CACHE_LIMIT = 4096


def exact_box_range(
    instance: Instance,
    output_index: int,
    boxes: Sequence[Box],
    enumeration_limit: int = DEFAULT_ENUMERATION_LIMIT,
) -> tuple[tuple[int, int], ...]:
    if len(boxes) != (output_index + 1) * instance.dimension:
        raise ValueError("wrong prefix box dimension")
    if instance.dimension == 1 and tuple(boxes) == instance.prefix_boxes(output_index):
        return (scalar_full_prefix_ranges(instance)[output_index],)
    size = box_size(boxes)
    if instance.dimension == 1 and (size > 1000 or output_index > 6):
        return (_scalar_dp_range(instance, output_index, boxes),)
    if size <= enumeration_limit:
        # Both traces store words, so their difference lies within this span.
        # A fixed decimal sentinel is incorrect for the parametric model API.
        lo_word, hi_word = instance.word_bounds
        span = hi_word - lo_word
        mins = [span] * instance.dimension
        maxs = [-span] * instance.dimension
        for point in product(*(range(b.lo, b.hi + 1) for b in boxes)):
            diff = output_difference(instance, point, output_index)
            for i, value in enumerate(diff):
                mins[i] = min(mins[i], value)
                maxs[i] = max(maxs[i], value)
        return tuple((mins[i], maxs[i]) for i in range(instance.dimension))
    if instance.dimension == 1:
        return (_scalar_dp_range(instance, output_index, boxes),)
    if exact_leaf_admissible(
        instance,
        output_index,
        boxes,
        enumeration_limit=enumeration_limit,
    ):
        raise AssertionError("admitted multidimensional leaf bypassed enumeration")
    raise ExactBoxLimit("large non-scalar exact box is outside checker admission limits")


def _scalar_dp_range(instance: Instance, output_index: int, boxes: Sequence[Box]) -> tuple[int, int]:
    """Exact scalar range for an arbitrary prefix box.

    The state stores the concrete inputs accumulated in the current chunk.  We
    deliberately recompute the declared prefix summary with ``compose_many``
    rather than incrementally updating one summary: incremental updating would
    silently impose left association and is wrong for balanced/right trees.
    Chunk length is a frozen small parameter in the admitted scalar fragment.
    """
    # State: (reference_state, incoming_chunk_carry, current_chunk_inputs)
    states: set[tuple[int, int, tuple[int, ...]]] = {(instance.h0[0], instance.h0[0], ())}
    final_diffs: set[int] = set()
    for t in range(output_index + 1):
        a = instance.matrices[t][0][0]
        chunk_start = (t // instance.chunk_length) * instance.chunk_length
        next_states: set[tuple[int, int, tuple[int, ...]]] = set()
        summaries: dict[tuple[int, ...], Pair] = {}
        box = boxes[t]
        end_chunk = ((t + 1) % instance.chunk_length == 0) or t == instance.horizon - 1
        for ref, carry, partial_inputs in states:
            for b in range(box.lo, box.hi + 1):
                ref2 = apply_pair((((a,),), (b,)), (ref,), width=instance.width, scale=instance.scale, mode=instance.rounding)[0]
                chunk_inputs = partial_inputs + (b,)
                summary = summaries.get(chunk_inputs)
                if summary is None:
                    pairs = [
                        (instance.matrices[chunk_start + j], (value,))
                        for j, value in enumerate(chunk_inputs)
                    ]
                    summary = compose_many(
                        pairs,
                        width=instance.width,
                        scale=instance.scale,
                        mode=instance.rounding,
                        association=instance.association,
                    )
                    if len(summaries) < _LAYER_SUMMARY_CACHE_LIMIT:
                        summaries[chunk_inputs] = summary
                low = apply_pair(summary, (carry,), width=instance.width, scale=instance.scale, mode=instance.rounding)[0]
                if t == output_index:
                    final_diffs.add(low - ref2)
                if end_chunk:
                    next_states.add((ref2, low, ()))
                else:
                    next_states.add((ref2, carry, chunk_inputs))
        states = next_states
    if not final_diffs:
        raise AssertionError("scalar DP produced no final states")
    return min(final_diffs), max(final_diffs)

_FULL_SCALAR_CACHE: dict[str, tuple[tuple[int, int], ...]] = {}


def _instance_key(instance: Instance) -> str:
    # Metadata is deliberately excluded; only arithmetic fields determine ranges.
    import json
    obj = instance.to_dict()
    obj.pop("metadata", None)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def scalar_full_prefix_ranges(instance: Instance) -> tuple[tuple[int, int], ...]:
    if instance.dimension != 1:
        raise ValueError("scalar cache requested for non-scalar instance")
    key = _instance_key(instance)
    cached = _FULL_SCALAR_CACHE.get(key)
    if cached is not None:
        return cached
    states: set[tuple[int, int, tuple[int, ...]]] = {(instance.h0[0], instance.h0[0], ())}
    ranges: list[tuple[int, int]] = []
    for t in range(instance.horizon):
        a = instance.matrices[t][0][0]
        chunk_start = (t // instance.chunk_length) * instance.chunk_length
        next_states: set[tuple[int, int, tuple[int, ...]]] = set()
        summaries: dict[tuple[int, ...], Pair] = {}
        diffs: set[int] = set()
        box = instance.boxes[t][0]
        end_chunk = ((t + 1) % instance.chunk_length == 0) or t == instance.horizon - 1
        for ref, carry, partial_inputs in states:
            for b in range(box.lo, box.hi + 1):
                ref2 = apply_pair((((a,),), (b,)), (ref,), width=instance.width, scale=instance.scale, mode=instance.rounding)[0]
                chunk_inputs = partial_inputs + (b,)
                summary = summaries.get(chunk_inputs)
                if summary is None:
                    pairs = [
                        (instance.matrices[chunk_start + j], (value,))
                        for j, value in enumerate(chunk_inputs)
                    ]
                    summary = compose_many(
                        pairs,
                        width=instance.width,
                        scale=instance.scale,
                        mode=instance.rounding,
                        association=instance.association,
                    )
                    if len(summaries) < _LAYER_SUMMARY_CACHE_LIMIT:
                        summaries[chunk_inputs] = summary
                low = apply_pair(summary, (carry,), width=instance.width, scale=instance.scale, mode=instance.rounding)[0]
                diffs.add(low - ref2)
                if end_chunk:
                    next_states.add((ref2, low, ()))
                else:
                    next_states.add((ref2, carry, chunk_inputs))
        ranges.append((min(diffs), max(diffs)))
        states = next_states
    out = tuple(ranges)
    _FULL_SCALAR_CACHE[key] = out
    return out
