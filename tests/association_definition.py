"""Literal integer specification for the separate finite association study.

Test-local oracle, not a production or historical implementation copy.
No fixedscan imports, random generation, clocks, files or external execution.
"""
from itertools import product


def stored(n, width, scale, mode):
    lower = n // scale
    if mode == 'floor':
        answer = lower
    elif mode == 'rne':
        # Compare exact integer distances, then choose the even code on a tie.
        answer = min((lower, lower + 1),
                     key=lambda code: (abs(n - scale * code), code % 2))
    else:
        raise ValueError('unsupported literal rounding')
    return max(-(1 << (width - 1)), min((1 << (width - 1)) - 1, answer))


def shape(length, association, start=0):
    if length < 1:
        raise ValueError('empty tree')
    if length == 1:
        return start
    if association not in ('left', 'right', 'balanced'):
        raise ValueError('unsupported literal association')
    cut = {'left': length - 1, 'right': 1,
           'balanced': length // 2}[association]
    return (shape(cut, association, start),
            shape(length - cut, association, start + cut))


def literal_trace(spec, prefix):
    """Both traces through exactly the given causal time-major prefix."""
    d, q = spec['dimension'], spec['scale']
    if not prefix or len(prefix) % d or len(prefix) > d * spec['horizon']:
        raise ValueError('bad literal prefix length')
    quant = lambda n: stored(n, spec['width'], q, spec['rounding'])
    inputs = [prefix[j:j+d] for j in range(0, len(prefix), d)]
    reference = list(spec['h0'])
    carry = list(reference)
    sequential, lowered = [], []

    def apply(matrix, offset, state):
        return [quant(q * offset[i] + sum(x * y for x, y in zip(row, state)))
                for i, row in enumerate(matrix)]

    def summary(node, chunk_start):
        if isinstance(node, int):
            step = chunk_start + node
            return spec['matrices'][step], inputs[step]
        earlier, earlier_bias = summary(node[0], chunk_start)
        later, later_bias = summary(node[1], chunk_start)
        columns = list(zip(*earlier))
        matrix = [[quant(sum(x * y for x, y in zip(row, col)))
                   for col in columns] for row in later]
        bias = [quant(q * later_bias[i] +
                      sum(x * y for x, y in zip(row, earlier_bias)))
                for i, row in enumerate(later)]
        return matrix, bias

    for t, bias in enumerate(inputs):
        for i, value in enumerate(bias):
            lo, hi = spec['boxes'][t][i]
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError('literal input outside declared box')
        reference = apply(spec['matrices'][t], bias, reference)
        sequential.append(tuple(reference))
        start = (t // spec['chunk_length']) * spec['chunk_length']
        matrix, offset = summary(shape(t - start + 1, spec['association']), start)
        output = apply(matrix, offset, carry)
        lowered.append(tuple(output))
        if (t + 1) % spec['chunk_length'] == 0:
            carry = output
    return tuple(sequential), tuple(lowered)


def prefix_points(spec, output):
    boxes = [box for row in spec['boxes'][:output+1] for box in row]
    return product(*(range(lo, hi+1) for lo, hi in boxes))


def literal_oracle(spec):
    """Keep every prefix, all ranges and signed lexicographic first witness."""
    rows, records, first = [], [], None
    for t in range(spec['horizon']):
        differences = []
        for point in prefix_points(spec, t):
            sequential, lowered = literal_trace(spec, point)
            difference = tuple(a-b for a, b in zip(lowered[-1], sequential[-1]))
            differences.append(difference)
            records.append({'output': t, 'point': list(point),
                            'difference': list(difference)})
            if first is None and any(abs(x) > tol for x, tol in
                                     zip(difference, spec['tolerances'][t])):
                first = (t, list(point))
        rows.append([[min(col), max(col)] for col in zip(*differences)])
    return {'status': 'safe' if first is None else 'violated',
            'earliest_output': None if first is None else first[0],
            'witness': None if first is None else first[1],
            'ranges': rows, 'prefix_evaluations': len(records), 'records': records}
