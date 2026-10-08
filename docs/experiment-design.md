# Experiment design

## Frozen semantic suite

The frozen suite is a deterministic, generated semantic test collection, not a workload benchmark. Its 72 cases were generated with seed 90173 before campaign outcomes were inspected. It contains 18 instances in each of four structural families: scalar lower-gain (stored under the inherited `stable` label), scalar above-one (`unstable`), diagonal, and coupled. Five of the 18 lower-gain cases contain at least one unit-gain step, so `stable` is not a strict coefficient bound or a dynamical-stability claim. Scalar horizons are four and two-coordinate horizons are three. Scales, widths, rounding modes, saturation exposure, and tolerance construction vary deterministically. Every frozen case has chunk length two. Left, balanced, and right labels occur equally often, but a two-element chunk has only one binary tree and therefore does not exercise nontrivial association differences. The separate eight-step scalar regression supplies that coverage with 48 association/mode output comparisons. Every varying input coordinate has three values, enabling an exhaustive prefix oracle. Tolerances are derived from oracle maxima either at the exact safe maximum or one code below the first positive maximum, yielding 33 safe and 39 violated cases without dropping unfavorable instances.

The compared methods are independent interval covers; paired covers with fresh residuals, syntactic residual sharing, and modular residual sharing; and an exact finite-state dynamic program. Each cover producer receives the same 5,000-node and 20-second internal limits on the frozen suite. The checker is outside the production budget.

## Long-horizon stress suite

The stress suite contains 16 scalar cases at horizons 8, 16, 32, and 64, crossing two coefficient regimes and both rounding modes. All contracts are safe. Cover methods receive 20,000 nodes and five internal seconds; exact DP receives 12 seconds. `unknown` is retained in every denominator. These cases probe resource-bound behavior only; they do not establish model or deployment scalability.

## Negative controls and mutations

The transport regressions keep three notions separate. Residual translation uses period `q` for floor and `2q` for nearest-even; `q` is an explicit nearest-even negative control. The closed form for the distinct rounding-error modulus `omega_q(D)` is compared with exhaustive threshold-period maxima. Carry-aware one-step and multistep recurrences are checked on small deterministic domains. Further negative controls show that clipping destroys residual periodicity and that a real-gain floor can miss a one-code rounding-threshold jump. Certificate mutations alter semantic bindings, omit required roots or children, corrupt ordered splitting, alter witnesses or first times, forge safe ranges, and exploit JSON Boolean/integer equality. Duplicate JSON keys are rejected.

## Measurement interpretation

Claim-bearing values are statuses, accepted/rejected checks, deterministic node counts for completed searches, and exact oracle comparisons. CPU times and peak RSS describe this Python implementation and host only. They are not hardware speedups, asymptotic bounds, or bit-identical reproduction targets. `scripts/analyze_results.py` derives both metrics CSVs, JSON summaries, `stress-by-horizon.csv`, and the manuscript table fragment from the same raw task records; it also checks every recorded certificate byte count against the stored file.
