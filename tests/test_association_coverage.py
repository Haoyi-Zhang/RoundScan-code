"""Small portable controls separate from the inherited assertion-unit ledger."""
import copy
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'tests'))
from association_definition import literal_trace, shape, stored
from association_coverage import cohort
from fixedscan.checker import check
from fixedscan.model import Instance
from fixedscan.producer import produce


class AssociationCoverageTests(unittest.TestCase):
    def test_literal_negative_ties_floor_and_clipping(self):
        for n, expected in ((-7, -4), (-5, -2), (-3, -2), (-1, 0),
                            (1, 0), (3, 2), (5, 2), (7, 4)):
            self.assertEqual(stored(n, 5, 2, 'rne'), expected)
            self.assertEqual(stored(n, 5, 2, 'floor'), n // 2)
        self.assertEqual(stored(-1000, 4, 4, 'rne'), -8)
        self.assertEqual(stored(1000, 4, 4, 'floor'), 7)

    def test_tree_shapes_do_not_overstate_length_three(self):
        self.assertEqual(shape(3, 'right'), shape(3, 'balanced'))
        self.assertEqual(len({shape(4, a) for a in ('left', 'right', 'balanced')}), 3)

    def test_declared_product_is_complete_and_unfiltered(self):
        specs = cohort()
        self.assertEqual(len(specs), 144)
        self.assertEqual(len({s['metadata']['base'] for s in specs}), 24)
        for spec in specs:
            self.assertEqual((spec['dimension'], spec['horizon']), (2, 4))
            self.assertIn(spec['width'], (4, 5))
            self.assertIn(spec['chunk_length'], (3, 4))
            self.assertEqual({hi-lo+1 for row in spec['boxes'] for lo, hi in row}, {2})
            self.assertNotEqual(spec['matrices'][0], spec['matrices'][1])
            self.assertTrue(all(m[0][1] and m[1][0] for m in spec['matrices']))
            self.assertNotIn('oracle_safe', spec['metadata'])
            self.assertIn(spec['tolerances'][0][0], (0, 2))
            self.assertEqual(spec['tolerances'], [spec['tolerances'][0]] * 4)

    def test_hand_calculated_common_carry_example(self):
        spec = copy.deepcopy(cohort()[0])
        spec.update(dimension=1, horizon=3, chunk_length=3, width=4, scale=2,
                    rounding='floor', h0=[-5], matrices=[[[1]]] * 3,
                    boxes=[[[-2, -2]]] * 3, tolerances=[[0]] * 3)
        for association, expected in (('left', -4), ('right', -3), ('balanced', -3)):
            spec['association'] = association
            ref, low = literal_trace(spec, (-2, -2, -2))
            self.assertEqual(ref, ((-5,),) * 3)
            self.assertEqual(low, ((-5,), (-3,), (expected,)))

    def test_unknown_budget_remains_unaccepted(self):
        instance = Instance.from_dict(cohort()[0])
        for method in ('fresh', 'syntactic', 'residue'):
            cert = produce(instance, method, max_nodes=1, timeout_seconds=20.)
            self.assertEqual(cert['status'], 'unknown')
            self.assertEqual(cert['diagnostics']['stop_reason'], 'node limit')
            checked = check(cert, instance)
            self.assertFalse(checked['accepted'])
            self.assertEqual(checked['status'], 'unknown')


if __name__ == '__main__':
    unittest.main(verbosity=2)
