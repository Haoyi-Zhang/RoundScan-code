# Bounded non-inert association study

This declaration is written before any oracle or producer evaluation. It is
a separate finite semantic/cover-topology cohort, not an extension of the
frozen or stress campaign and not a timing study. No case is selected, relabeled
or discarded on an observed outcome. All tolerances are chosen below, without
oracle maxima. No production arithmetic or checker rule is changed.

The complete product is widths `{4,5}` x chunk lengths `{3,4}` x profiles
`{alternating,cross_gain,endpoints}` x uniform code tolerances `{0,2}`. These
24 base declarations are crossed with association `{left,right,balanced}` and
rounding `{floor,rne}`: exactly 144 instances. Every instance has dimension two,
horizon four, and exactly two adjacent legal bias codes in **each** time and
coordinate. Every matrix profile varies with time and has nonzero coupling.

| Profile | Scale | Initial state | Matrices in time order | Bias lower codes |
|---|---:|---|---|---|
| alternating | 2 | `(-2,3)` | `[[1,1],[-1,2]]; [[2,-1],[1,1]]; [[1,-1],[1,2]]; [[2,1],[-1,1]]` | `(t+i)%2-1` |
| cross_gain | 4 | `(2,-3)` | `[[5,1],[-1,3]]; [[3,-1],[2,5]]; [[4,2],[1,3]]; [[5,-2],[-1,4]]` | `(t+i)%3-1` |
| endpoints | 4 | `(H,L)` | `[[7,-2],[1,5]]; [[5,2],[-2,7]]; [[7,1],[-1,5]]; [[5,-1],[2,7]]` | `H-1` if `(t+i)%2==0`, otherwise `L` |

Here `t,i` start at zero; each box is `[lower,lower+1]`; `L,H` are the signed
word endpoints. The tolerance is the chosen fixed value at every output and
coordinate. Width changes do not enlarge the bias cardinality. Each instance
has 256 full assignments and `4+16+64+256=340` nonempty time-prefix assignments.
The cohort therefore has 36,864 full assignments (the final-prefix subset) and
48,960 nonempty prefix assignments. Counting one empty initial prefix per
instance would give 49,104, but the runner does not evaluate empty prefixes.
Counts are distinct cohort assignments, not the number of evaluations across
the independent, producer and checker paths. No independent full-trace sweep
is added on top of final-prefix enumeration.

Association changes the lowering and hence the contract. Only `fresh`,
`syntactic` and `residue` on the **same** complete instance are compared for
node counts. Cross-association output differences are semantic sensitivity,
not a method win/loss. Balanced and right have the same length-three shape;
all three full length-four shapes differ. Chunk three also exercises carry
propagation into a final one-step chunk.

`tests/association_definition.py` is a test-local literal evaluator: exact
integer dot products, mathematical rounding by integer nearest-distance and
even-tie selection, clipping at every prescribed site, and explicit recursive
trees. It imports no production module. `scripts/association_coverage.py`
requires complete pointwise agreement with both concrete paths, complete
range/ordered-witness agreement with the shipped oracle, and exact checker
range agreement. It checks every emitted conclusive certificate through the
unchanged strict file loader. All unfavorable cases, unknowns, faults, node
counts and raw producer/checker diagnostics are retained. A mismatch fails the
study, not its denominator.

The three producers receive equal frozen-suite limits: 5,000 nodes and 20
cooperative seconds, with the existing after-analysis deadline checks intact.
Each isolated case worker additionally has a 120-second hard boundary covering
oracle, producers and checker; a hard failure remains a failed/unknown record
in the planned 144-instance denominator. No worker is retried. Existing CPU/
wall diagnostic fields are retained verbatim for provenance, not aggregated,
benchmarked or used as runtime evidence. There is no RSS/native/provider run.

Run from the artifact into a **new absent** directory:

```sh
python -B tests/test_association_coverage.py -v
python -B scripts/association_coverage.py --out ../association-study
```

The new output must be outside inherited `cases/` and `results/`; it is created
exclusively. The runner never imports generation or calls `write_collections`.
Scientific CI has a separate bounded step and preserves its output even on
failure; the old scalar, 188,847-unit, ten-method and campaign gates remain.
No finite outcome mechanizes the hand proofs or establishes workload utility.
