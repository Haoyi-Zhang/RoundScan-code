"""Portable scalar-DP regressions with an independent recursive definition.

No producer arithmetic, retained result input, benchmark or filesystem writes.
Four methods are separate from the frozen assertion-unit ledger.
"""
from dataclasses import replace
from fractions import Fraction
from itertools import product
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fixedscan import baseline, exact_box
from fixedscan.model import Box, Instance


def owned_cases():
    patterns = ((2, 2, 2, 2, 2, 2), (7, -8, 3, -3, 7, -8),
                (-1, 3, -2, 5, -7, 1))
    for mode, association, chunk, initial, coefficients in product(
            ('floor', 'rne'), ('left', 'right', 'balanced'), (2, 3, 4),
            (-8, 0, 7), patterns):
        yield Instance('owned-scalar', 'finite-regression', 4, 4, mode, 1,
                       6, chunk, association, (initial,),
                       tuple(((a,),) for a in coefficients),
                       tuple((Box(-1, 0) if t % 2 == 0 else Box(0, 1),)
                             for t in range(6)), ((0,),) * 6, {})


def quantized(instance, numerator):
    value = Fraction(numerator, instance.scale)
    lower = value.numerator // value.denominator
    rounded = lower if instance.rounding == 'floor' else min(
        (lower, lower + 1), key=lambda n: (abs(value - n), n % 2))
    lo, hi = instance.word_bounds
    return min(hi, max(lo, rounded))


def joined(instance, earlier, later):
    a, b = earlier
    c, d = later
    return (quantized(instance, c * a),
            quantized(instance, c * b + instance.scale * d))


def tree(instance, leaves):
    if len(leaves) == 1:
        return leaves[0]
    split = (len(leaves) - 1 if instance.association == 'left' else
             1 if instance.association == 'right' else len(leaves) // 2)
    return joined(instance, tree(instance, leaves[:split]),
                  tree(instance, leaves[split:]))


def difference(instance, point):
    reference = carry = instance.h0[0]
    leaves = []
    for t, value in enumerate(point):
        a = instance.matrices[t][0][0]
        reference = quantized(instance, a * reference + instance.scale * value)
        leaves.append((a, value))
        c, d = tree(instance, leaves)
        lowered = quantized(instance, c * carry + instance.scale * d)
        if (t + 1) % instance.chunk_length == 0:
            carry = lowered
            leaves = []
    return lowered - reference


def defined_range(instance, boxes):
    values = [difference(instance, point)
              for point in product(*(b.values() for b in boxes))]
    return min(values), max(values)


class ScalarSummaryTests(unittest.TestCase):
    def test_full_ranges_all_associations_rounding_and_clipping(self):
        for instance in owned_cases():
            expected = tuple(defined_range(instance, instance.prefix_boxes(t))
                             for t in range(instance.horizon))
            self.assertEqual(exact_box.scalar_full_prefix_ranges(instance), expected)

    def test_arbitrary_subboxes_keep_complete_tuple_and_carry(self):
        for instance in list(owned_cases())[::9]:
            for t in range(instance.horizon):
                boxes = tuple(Box(b.lo, b.lo) if j % 2 else b
                              for j, b in enumerate(instance.prefix_boxes(t)))
                self.assertEqual(exact_box._scalar_dp_range(instance, t, boxes),
                                 defined_range(instance, boxes))

    def test_cap_changes_only_reuse_and_cache_is_layer_local(self):
        instance = replace(next(owned_cases()), name='owned-cap', horizon=8,
                           matrices=(((2,),),) * 8,
                           boxes=((Box(-1, 1),),) * 8,
                           tolerances=((0,),) * 8)
        expected = tuple(defined_range(instance, instance.prefix_boxes(t))
                         for t in range(8))
        saved = dict(exact_box._FULL_SCALAR_CACHE)
        try:
            for cap in (0, 1, 2, 4096):
                exact_box._FULL_SCALAR_CACHE.clear()
                with patch.object(exact_box, '_LAYER_SUMMARY_CACHE_LIMIT', cap):
                    self.assertEqual(exact_box.scalar_full_prefix_ranges(instance), expected)
                    self.assertEqual(exact_box._scalar_dp_range(
                        instance, 7, instance.prefix_boxes(7)), expected[7])
        finally:
            exact_box._FULL_SCALAR_CACHE.clear()
            exact_box._FULL_SCALAR_CACHE.update(saved)

    def test_first_witness_deadline_and_multidimensional_admission(self):
        for instance in list(owned_cases())[::27]:
            first = next(((t, point) for t in range(instance.horizon)
                          for point in product(*(b.values() for b in instance.prefix_boxes(t)))
                          if abs(difference(instance, point)) > 0), None)
            with patch.object(baseline, 'monotonic', return_value=0.):
                result = baseline.exact_baseline(instance, timeout_seconds=1.)
            self.assertEqual((result['status'], result['earliest_output'], result['witness']),
                             ('safe', None, None) if first is None else
                             ('violated', first[0], list(first[1])))
        with patch.object(baseline, 'monotonic', side_effect=[0., 0., 2., 2.]):
            self.assertEqual(baseline.exact_baseline(next(owned_cases()), 1.)['status'], 'unknown')
        vector = Instance('owned-vector', 'finite-regression', 10, 2, 'floor',
                          2, 1, 1, 'left', (0, 0), (((0, 0), (0, 0)),),
                          ((Box(0, 316), Box(0, 316)),), ((316, 316),), {})
        with self.assertRaises(exact_box.ExactBoxLimit):
            exact_box.exact_box_range(vector, 0, vector.prefix_boxes(0))


if __name__ == '__main__':
    unittest.main(verbosity=2)
