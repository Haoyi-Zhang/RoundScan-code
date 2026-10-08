#!/usr/bin/env python3
"""Unfiltered bounded semantic/topology study; no latency comparison.

Writes only to a new absent output outside inherited cases/results. Each
owned offline case is isolated and finite; no generation, retry or cleanup.
"""
import argparse
from collections import Counter
import hashlib
from itertools import product
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))
from association_definition import literal_oracle
from fixedscan.arithmetic import output_difference as producer_difference
from fixedscan.checker import load_and_check
from fixedscan.checker_arithmetic import output_difference as checker_difference
from fixedscan.exact_box import exact_box_range
from fixedscan.model import Instance
from fixedscan.oracle import exact_oracle
from fixedscan.producer import produce

METHODS = ('fresh', 'syntactic', 'residue')
PROFILES = {
    'alternating': (2, ((1, 1, -1, 2), (2, -1, 1, 1),
                       (1, -1, 1, 2), (2, 1, -1, 1))),
    'cross_gain': (4, ((5, 1, -1, 3), (3, -1, 2, 5),
                      (4, 2, 1, 3), (5, -2, -1, 4))),
    'endpoints': (4, ((7, -2, 1, 5), (5, 2, -2, 7),
                     (7, 1, -1, 5), (5, -1, 2, 7))),
}


def cohort():
    declarations = []
    for number, (width, chunk, profile, tolerance) in enumerate(product(
            (4, 5), (3, 4), PROFILES, (0, 2))):
        lo, hi = -(1 << (width-1)), (1 << (width-1))-1
        q, pattern = PROFILES[profile]
        matrices = [[list(row[:2]), list(row[2:])] for row in pattern]
        boxes = []
        for t in range(4):
            row = []
            for i in range(2):
                lower = ((t+i) % 2 - 1 if profile == 'alternating' else
                         (t+i) % 3 - 1 if profile == 'cross_gain' else
                         hi-1 if (t+i) % 2 == 0 else lo)
                row.append([lower, lower+1])
            boxes.append(row)
        for association, rounding in product(('left', 'right', 'balanced'),
                                              ('floor', 'rne')):
            spec = {
                'name': f'association-{number:02d}-{association}-{rounding}',
                'family': profile, 'width': width, 'scale': q,
                'rounding': rounding, 'dimension': 2, 'horizon': 4,
                'chunk_length': chunk, 'association': association,
                'h0': [-2, 3] if profile == 'alternating' else
                      [2, -3] if profile == 'cross_gain' else [hi, lo],
                'matrices': matrices, 'boxes': boxes,
                'tolerances': [[tolerance, tolerance] for _ in range(4)],
                'metadata': {'cohort': 'bounded-association', 'base': number,
                             'tolerance_rule': 'declared-zero-or-two'},
            }
            Instance.from_dict(spec)  # Structure only; no oracle label.
            declarations.append(spec)
    if len(declarations) != 144 or len({s['name'] for s in declarations}) != 144:
        raise RuntimeError('declaration cardinality failure')
    return declarations


def exclusive_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write('\n')


def scientific_conclusion(value):
    return (value['status'], value['earliest_output'], value['witness'])


def worker(case, output):
    spec = json.loads(case.read_text(encoding='utf-8'))
    if spec not in cohort():
        raise ValueError('worker input is not an exact declared cohort member')
    instance = Instance.from_dict(spec)
    output.mkdir(exist_ok=False)
    literal = literal_oracle(spec)
    exclusive_json(output / 'literal-prefixes.json', literal)
    for record in literal['records']:
        point, t = record['point'], record['output']
        if (list(producer_difference(instance, point, t)) != record['difference']
                or list(checker_difference(instance, point, t)) != record['difference']):
            raise AssertionError(('pointwise concrete disagreement', record))
    shipped = exact_oracle(instance)
    oracle = {'status': 'safe' if shipped.safe else 'violated',
              'earliest_output': shipped.earliest_output,
              'witness': list(shipped.witness) if shipped.witness is not None else None,
              'ranges': [[list(pair) for pair in row] for row in shipped.ranges],
              'prefix_evaluations': shipped.prefix_evaluations}
    exclusive_json(output / 'shipped-oracle.json', oracle)
    if (scientific_conclusion(oracle) != scientific_conclusion(literal)
            or oracle['ranges'] != literal['ranges']
            or oracle['prefix_evaluations'] != 340):
        raise AssertionError('complete oracle/range/ordered-witness disagreement')
    exact_ranges = [[list(pair) for pair in exact_box_range(
        instance, t, instance.prefix_boxes(t))] for t in range(4)]
    exclusive_json(output / 'checker-ranges.json', exact_ranges)
    if exact_ranges != literal['ranges']:
        raise AssertionError('independent checker exact-range disagreement')
    records = []
    for method in METHODS:
        certificate = produce(instance, method, max_nodes=5000, timeout_seconds=20.0)
        path = output / f'{method}-certificate.json'
        exclusive_json(path, certificate)  # Includes unknown partial objects.
        checked = load_and_check(path, instance)
        exclusive_json(output / f'{method}-checker.json', checked)
        valid = (not checked['accepted'] and checked['status'] == 'unknown'
                 if certificate['status'] == 'unknown' else
                 checked['accepted'] and
                 scientific_conclusion(certificate) == scientific_conclusion(literal))
        row = {'method': method, 'status': certificate['status'],
               'nodes': certificate['diagnostics']['nodes'],
               'certificate_bytes': path.stat().st_size,
               'checked_nodes': checked['checked_nodes'],
               'checker_accepted': checked['accepted'], 'consistent': bool(valid),
               'stop_reason': certificate['diagnostics']['stop_reason']}
        records.append(row)
        # Save every method before failing; unknown is never promoted to proof.
    report = {'case': spec['name'], 'base': spec['metadata']['base'],
              'association': spec['association'], 'rounding': spec['rounding'],
              'oracle': {k: v for k, v in oracle.items() if k != 'ranges'},
              'methods': records, 'consistent': all(r['consistent'] for r in records)}
    exclusive_json(output / 'case-result.json', report)
    print(json.dumps(report, sort_keys=True), flush=True)
    return 0 if report['consistent'] else 1


def run(output):
    output = output.resolve()
    for protected in (ROOT / 'cases', ROOT / 'results'):
        if output == protected.resolve() or protected.resolve() in output.parents:
            raise ValueError('output must not enter inherited cases/results')
    specs = cohort()
    output.mkdir(parents=True, exist_ok=False)
    cases = output / 'cases'
    cases.mkdir()
    # Entire declaration and all inputs committed before any worker/oracle.
    declaration = {'instances': 144, 'base_contracts': 24, 'full_assignments': 36864,
                   'nonempty_prefix_assignments': 48960, 'max_full_per_instance': 256,
                   'methods': list(METHODS), 'max_nodes': 5000,
                   'cooperative_timeout_seconds': 20.0, 'worker_hard_seconds': 120,
                   'outcome_filter': None, 'timing_comparison': False,
                   'instances_sha256': hashlib.sha256(json.dumps(
                       specs, sort_keys=True, separators=(',', ':')).encode()).hexdigest()}
    exclusive_json(output / 'declaration.json', declaration)
    for spec in specs:
        exclusive_json(cases / (spec['name'] + '.json'), spec)
    results, failures = [], []
    for spec in specs:
        target = output / spec['name']
        command = [sys.executable, '-B', str(Path(__file__).resolve()),
                   '--worker', str(cases / (spec['name'] + '.json')),
                   '--out', str(target)]
        try:
            completed = subprocess.run(command, capture_output=True, text=True,
                                       encoding='utf-8', timeout=120, check=False)
            captured = {'exit_code': completed.returncode,
                        'stdout': completed.stdout, 'stderr': completed.stderr}
        except subprocess.TimeoutExpired as exc:
            captured = {'exit_code': None, 'hard_timeout': True,
                        'stdout': (exc.stdout or b'').decode('utf-8', errors='replace')
                                  if isinstance(exc.stdout, bytes) else exc.stdout or '',
                        'stderr': (exc.stderr or b'').decode('utf-8', errors='replace')
                                  if isinstance(exc.stderr, bytes) else exc.stderr or ''}
        exclusive_json(output / (spec['name'] + '-worker.json'), captured)
        if captured['exit_code'] != 0:
            failures.append({'case': spec['name'], 'exit_code': captured['exit_code']})
        result = target / 'case-result.json'
        if result.is_file():
            results.append(json.loads(result.read_text(encoding='utf-8')))
        print(json.dumps({'completed_workers': len(results), 'planned_instances': 144,
                          'case': spec['name'], 'exit_code': captured['exit_code']}), flush=True)
    by_method = {}
    for method in METHODS:
        rows = [r for case in results for r in case['methods'] if r['method'] == method]
        by_method[method] = {'denominator': 144,
                             'statuses': dict(Counter(r['status'] for r in rows)),
                             'missing': 144-len(rows),
                             'nodes': sum(r['nodes'] for r in rows),
                             'accepted_certificates': sum(r['checker_accepted'] for r in rows)}
    comparisons = {}
    for method in ('syntactic', 'residue'):
        comparison = Counter()
        for case in results:
            rs = {r['method']: r for r in case['methods']}
            fresh, other = rs['fresh'], rs[method]
            if not (fresh['consistent'] and other['consistent']) or 'unknown' in (
                    fresh['status'], other['status']):
                comparison['not_comparable'] += 1
            else:
                comparison['fewer' if other['nodes'] < fresh['nodes'] else
                           'more' if other['nodes'] > fresh['nodes'] else 'equal'] += 1
        comparison['missing'] = 144-len(results)
        comparisons[method + '_vs_fresh_same_contract'] = dict(comparison)
    # Output-signature sensitivity is not a sharing-method comparison.
    sensitivity = Counter()
    for base, mode in product(range(24), ('floor', 'rne')):
        signatures = []
        for association in ('left', 'right', 'balanced'):
            path = output / f'association-{base:02d}-{association}-{mode}' / 'literal-prefixes.json'
            if path.is_file():
                data = json.loads(path.read_text(encoding='utf-8'))
                signatures.append(tuple(tuple(r['difference']) for r in data['records']))
        sensitivity['missing' if len(signatures) != 3 else
                    'changed' if len(set(signatures)) > 1 else 'equal'] += 1
    report = {'planned_instances': 144, 'completed_instances': len(results),
              'all_consistent': not failures and len(results) == 144,
              'failures': failures, 'methods': by_method,
              'same_contract_node_comparisons': comparisons,
              'cross_association_semantic_sensitivity': dict(sensitivity),
              'timing_claim': None, 'historical_evidence_modified': False}
    exclusive_json(output / 'summary.json', report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report['all_consistent'] else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--worker', type=Path)
    args = parser.parse_args()
    raise SystemExit(worker(args.worker, args.out) if args.worker else run(args.out))
