"""Narrow finite regressions, separate from the retained historical test ledger."""
from dataclasses import replace
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fixedscan.arithmetic import output_difference
from fixedscan.baseline import exact_baseline
from fixedscan.checker import check
from fixedscan.exact_box import ExactBoxLimit, exact_box_range
from fixedscan.model import Box, Instance
from fixedscan.oracle import exact_oracle
from fixedscan.producer import produce


def singleton(width=3, dimension=1, sign=1):
    value = sign * ((1 << (width - 2)) - 1)
    matrix = tuple(tuple(int(i == j) for j in range(dimension)) for i in range(dimension))
    return Instance('boundary', 'finite-regression', width, 2, 'floor', dimension,
                    2, 2, 'left', (value,)*dimension, (matrix,)*2,
                    (tuple(Box(0, 0) for _ in range(dimension)),)*2,
                    ((1 << width,)*dimension,)*2, {})


class ExactBoundaryTests(unittest.TestCase):
    def test_wide_singletons_have_exact_extrema(self):
        # The mathematical model API is parametric. Public certificate admission
        # remains capped at width 32; these are not accepted wide-word proofs.
        for width in (32, 128):
            for mode in ('floor', 'rne'):
                for sign in (-1, 1):
                    instance = replace(singleton(width, 2, sign), rounding=mode)
                    expected = tuple((v, v) for v in output_difference(instance, (0,)*4, 1))
                    self.assertEqual(exact_box_range(instance, 1, instance.prefix_boxes(1)), expected)
                    self.assertEqual(exact_oracle(instance).ranges[1], expected)

    def test_wide_word_certificate_limit_is_unchanged(self):
        certificate = produce(singleton(128, 2), 'interval', timeout_seconds=1.)
        result = check(certificate)
        self.assertFalse(result['accepted'])
        self.assertIn('width exceeds checker limit', result['error'])

    def test_baseline_expiry_after_exact_query_is_unknown(self):
        instance = replace(singleton(), horizon=1, matrices=(((0,),),),
                           boxes=((Box(0, 0),),), tolerances=((0,),))
        with patch('fixedscan.baseline.monotonic', side_effect=[0., 0., 2., 2.]):
            result = exact_baseline(instance, timeout_seconds=1.)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['stop_reason'], 'wall-clock limit')

    def test_baseline_expiry_after_witness_search_is_unknown(self):
        instance = singleton()
        with patch('fixedscan.baseline.exact_box_range', return_value=((1, 1),)), \
             patch('fixedscan.baseline._lex_witness', return_value=(0,)), \
             patch('fixedscan.baseline.monotonic', side_effect=[0., 0., 0., 2., 2.]):
            result = exact_baseline(replace(instance, tolerances=((0,),)*2), 1.)
        self.assertEqual(result['status'], 'unknown')

    def test_baseline_inadmissible_leaf_is_unknown(self):
        with patch('fixedscan.baseline.exact_box_range', side_effect=ExactBoxLimit('owned admission boundary')):
            result = exact_baseline(singleton())
        self.assertEqual(result['status'], 'unknown')
        self.assertIn('admission boundary', result['stop_reason'])

    def test_producer_expiry_after_analysis_is_unknown(self):
        with patch('fixedscan.producer.monotonic', side_effect=[0., 0., 2., 2.]):
            result = produce(singleton(), 'interval', timeout_seconds=1.)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['diagnostics']['stop_reason'], 'wall-clock limit')
        self.assertFalse(check(result)['accepted'])

    def test_producer_stack_exhaustion_is_unknown(self):
        dimension = 8
        instance = Instance('stack-boundary', 'finite-regression', 32, 2, 'floor',
                            dimension, 1, 1, 'left', (0,)*dimension,
                            (tuple((0,)*dimension for _ in range(dimension)),),
                            (tuple(Box(-(1 << 31), (1 << 31)-1)
                                   for _ in range(dimension)),),
                            ((0,)*dimension,), {})
        original_limit = sys.getrecursionlimit()
        try:
            sys.setrecursionlimit(200)
            result = produce(instance, 'interval', max_nodes=300, timeout_seconds=60.)
        finally:
            sys.setrecursionlimit(original_limit)
        self.assertEqual(result['status'], 'unknown')
        self.assertEqual(result['diagnostics']['stop_reason'], 'recursion limit')
        self.assertFalse(check(result)['accepted'])

    def test_model_save_uses_canonical_lf(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'case.json'
            singleton().save(path)
            self.assertNotIn(b'\r', path.read_bytes())
            self.assertEqual(Instance.load(path).to_dict(), singleton().to_dict())

    def test_campaign_output_preserves_shipped_records(self):
        paths = [ROOT / 'results' / 'frozen-metrics.csv',
                 ROOT / 'results' / 'raw' / 'frozen' / 'case-000-interval.json']
        before = [path.read_bytes() for path in paths]
        env = dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1')
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / 'campaign'
            result = subprocess.run(
                [sys.executable, '-B', str(ROOT / 'scripts' / 'run_campaign.py'),
                 '--split', 'frozen', '--count', '1', '--out', str(output)],
                env=env, capture_output=True, text=True, timeout=30.)
            self.assertEqual(result.returncode, 0, result.stderr)
            raw = sorted((output / 'raw' / 'frozen').glob('*.json'))
            self.assertEqual(len(raw), 5)
            self.assertEqual(len(list((output / 'certificates' / 'frozen').glob('*.json'))), 4)
            for path in raw:
                row = json.loads(path.read_text(encoding='utf-8'))
                self.assertIn('worker_stdout', row['task'])
                self.assertIn('worker_stderr', row['task'])
        self.assertEqual([path.read_bytes() for path in paths], before)

    def test_campaign_rejects_historical_output_path(self):
        result = subprocess.run(
            [sys.executable, '-B', str(ROOT / 'scripts' / 'run_campaign.py'),
             '--out', str(ROOT / 'results')],
            capture_output=True, text=True, timeout=10.,
            env=dict(os.environ, PYTHONUTF8='1', PYTHONDONTWRITEBYTECODE='1'))
        self.assertEqual(result.returncode, 2)
        self.assertIn('shipped results tree', result.stderr)


if __name__ == '__main__':
    unittest.main(verbosity=2)
