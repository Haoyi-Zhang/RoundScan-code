# Compositional Rounding Certificates

This repository is the standalone artifact for *Compositional Rounding Certificates for Chunked Affine Scans*. It specifies a bounded integer fixed-point recurrence, a declared chunk-prefix lowering, proof-producing ordered input covers, and an exact checker. The implementation has no third-party Python dependency.

## What is checked

An instance embeds signed word width, power-of-two scale, floor or round-to-nearest-even rounding, saturation, initial state, time-varying affine matrices, Cartesian input boxes, chunk length, association rule, and one tolerance vector per output. The reference executes every recurrence update. The lowering quantizes affine transfer summaries and evaluates every causal prefix from the same incoming chunk carry.

A conclusive ordered-cover certificate contains one root per proven-safe output and, when the contract fails, a singleton witness at the first failing output. The checker recomputes exact leaf ranges. It also enforces canonical first-nonsingleton splitting and left-before-right traversal, so an accepted violation certificate establishes the lexicographically first violating prefix at the earliest failing output. An `unknown` producer result is never accepted as a proof.

The producer closes an abstractly safe leaf only when the current exact checker admits that leaf: every scalar leaf has an exact finite-state DP path, while a multidimensional leaf must contain at most 100,000 points for complete enumeration. Otherwise the producer subdivides canonically and returns `unknown` if its node or time budget expires or its interpreter stack is exhausted. This shared admission predicate contains no arithmetic fact; the checker still independently recomputes the exact range, and the unsplit 100,489-point root remains rejected even when supplied with truthful-looking bounds.

## Repository map

- `src/fixedscan/`: concrete semantics, analyses, producer, checker, exact baseline, leaf admission, transport recurrences, clipping instrumentation, block projection, and executable reduction constructions.
- `cases/`: deterministic development, frozen, and long-horizon stress instances.
- `results/raw/`: one authoritative raw record per instance and method.
- `results/certificates/`: all conclusive ordered-cover certificates.
- `results/*-metrics.csv`, `results/*-summary.json`, and `results/stress-by-horizon.csv`: derived campaign tables.
- `results/paper-results.tex`: manuscript rows generated from the same raw records and stored certificate files.
- `tests/run_all.py`: arithmetic cross-checks, exact-oracle comparisons, certificate and mutation checks, reduction checks, leaf-admission checks, transport checks, and result reconciliation.
- `proofs/`: hand proofs and claim-level statement index. These are not a mechanized formalization.
- `docs/validation-contract.md`: requirement-to-test map for the repaired acceptance boundary.
- `claim_evidence_ledger.csv`: paper claim to theorem, implementation, and result mapping.
- `reference-audit.csv`: one provenance row for every manuscript reference.
- `external_resources.csv`: scholarly/software resources and acquisition notes.

## Reproduce the checked evidence

Python 3.11 or newer is sufficient.

```sh
./reproduce.sh
```

The default path regenerates all deterministic case files in a temporary directory and compares them byte-for-byte, rebuilds every result table from the raw records, verifies recorded certificate byte counts against the actual files, audits the 30-entry scholarly-reference inventory, independently checks every stored certificate, reruns all finite oracles and mutation tests, and writes `results/reproduction.json`. It deliberately does not overwrite recorded wall/CPU timings.

To rerun the complete one-worker campaign into a new directory before rechecking the retained evidence:

```sh
./reproduce.sh --rerun-campaign --campaign-out ../campaign-output
```

The campaign output must not already exist and must not be inside the shipped `results/` tree. The original raw records, tables, certificates, and historical timing checks remain intact; fresh raw records, certificates, summaries, and worker output are written separately. Resource-bound stress outcomes can vary by host. The Unix RSS measurement is reported as unavailable on Windows, rather than fabricated.

Each frozen task has a 20-second internal limit and 5,000-node cover limit. Each stress cover task has a 5-second internal limit and 20,000-node limit; the exact dynamic-program baseline gets 12 seconds. The producer and baseline recheck the deadline after analysis and before reporting a conclusion. These are cooperative deadlines, not preemption within one arithmetic call; the campaign runner additionally terminates workers after the internal limit plus five seconds. `unknown` is an intended resource-bound outcome. Timings are machine dependent, while statuses, certificate acceptance, node counts on completed deterministic searches, and test assertions are the claim-bearing quantities.

`python3 -B tests/test_exact_boundaries.py` runs ten additional regression methods for wide-word exact extrema, unchanged public width admission, cooperative deadlines, exact-leaf rejection, interpreter stack exhaustion, canonical LF serialization, and separate campaign output. They are separate from the historical 188,847-unit ledger. `.github/workflows/scientific-checks.yml` now runs the full fresh campaign and then rechecks retained evidence and these regressions on Ubuntu 24.04 with Python 3.12, one CPU, a 2 GiB address-space limit, and a 540-second whole-command limit; it uploads raw logs and result files even on failure. A workflow definition alone is not evidence of successful execution.

### Verified subsequent native run

The successful full run `37480596336` at head `32924e0042115499378dfad9648f6d0d619fc4a6`, artifact `11420554117`, produced a complete fresh 440-task grid: 72 frozen plus 16 stress inputs, each under five methods (360 + 80 tasks). `results/current-replay/` retains exactly 780 ordinary-data/log files copied byte-for-byte from that archive: 440 raw records, 334 certificates, two summary JSONs, two metrics CSVs, and two scientific gate/log files. Native CPU/wall/RSS, certificate serialization lengths and timed-unknown progress stay here, not in historical tables. The original ZIP and earlier failed-upload/replay history are not altered; the older proof-check-only replay is not credited with this campaign.

All 334 conclusive certificates were accepted: 178 safe conclusions and 156 violation conclusions. The other 106 checker fields are null (88 exact-DP baselines and 18 unknown covers), **not** 106 counterexamples. All 440 scientific raw fields and all complete certificate bodies agree with historical and prior private records, apart from timing fields and host-dependent partial progress in timed unknowns. Frozen counts and completed-search topology agree; the original Linux/Python 3.13.5 measurements and earlier Windows checks remain separate. The same run replayed 188,847 heterogeneous assertion units and passed ten separate boundary methods; neither count is an independent statistical sample size or blanket proof validation.

Full native certificate-embedded inputs are available for 85 of 88 instances. `stress-005`, `stress-009` and `stress-013` have no conclusive cover, and the archive has no standalone copies of their full matrices/boxes. Their five-method raw rows and available input metadata agree; the driver records successful case regeneration. Independent full-input-byte equality for those three cannot be claimed from this archive. This evidence-scope limit is not a scientific failure. No archive code was executed during verification or retention, and no inputs were reconstructed to fill the gap. The original `results/` remains the historical source for generated manuscript tables.

Check one certificate directly:

```sh
python3 scripts/check_certificate.py \
  results/certificates/frozen/case-001-fresh.json \
  --instance cases/frozen/case-001.json
```

## Recorded evidence

The frozen suite contains 72 instances and 33,804 exact prefix evaluations. All frozen cases use chunk length two. The left, balanced, and right labels occur 24 times each, but a two-element chunk has only one binary composition tree; those labels are not nontrivial tree-shape evidence. Association-sensitive coverage instead comes from the separate eight-step scalar regression with 48 exact DP-versus-enumeration output comparisons. The inherited `stable` family is a lower-gain label, not a strict contraction condition: five of its 18 cases contain at least one unit-gain step.

All five methods classify the same 33 safe and 39 violating frozen instances. The four cover producers yield 288 certificates, all accepted by the checker. Independent interval covers use 24,774 nodes; each paired policy uses 3,260. The three paired residual-sharing policies therefore have identical final cover topology on this suite.

The raw-record-derived frozen totals are:

| Method | Nodes | Certificate bytes | Producer CPU (s) | Checker CPU (s) |
|---|---:|---:|---:|---:|
| Independent intervals | 24,774 | 2,583,453 | 1.557 | 0.598 |
| Fresh residuals | 3,260 | 384,903 | 4.182 | 0.374 |
| Syntactic sharing | 3,260 | 385,230 | 4.099 | 0.381 |
| Modular sharing | 3,260 | 385,082 | 4.100 | 0.385 |
| Exact finite-state DP | - | - | 0.353 | - |

The 16 stress instances have horizons 8, 16, 32, and 64. Exact finite-state dynamic programming resolves 16/16; paired covers resolve 12/16; independent interval covers resolve 10/16. Unresolved cases remain `unknown` rather than being counted as safe. `results/stress-by-horizon.csv` is regenerated from the raw records; for example, the horizon-eight fresh row contains 2,083 nodes.

The test suite records 188,847 assertion units. It includes 45,072 producer/checker concrete-prefix cross-checks, 45,072 exact-oracle prefix evaluations, 334 accepted stored certificates, 57 rejected mutation/parser/admission inputs, twelve leaf-admission closure/rejection/budget checks, 48 dynamic-program/exhaustive association checks, 4,914 block-factorization prefix checks and four lifted witnesses, 258 exact rounding-modulus comparisons, 73,572 carry-bound checks, 832 fixed-width circuit assignment/mode executions, 867 packed-CNF formulas over 6,474 assignments, 11,320,776 recorded pre-clip values across 7,306 reduction executions, and 30 audited scholarly-reference records. The reduction traces contain no clipping; a separate negative control activates clipping at sequential-state, summary-matrix, summary-bias, and carry-application sites.

## Trusted boundary and limitations

### Separate non-inert association coverage

`docs/association-coverage.md` declares an unfiltered 144-instance product:
widths four/five, two coordinates, horizon four, chunks three/four, three
time-varying coupled profiles, two fixed tolerances, three associations and two
rounding modes. Tolerances are chosen before oracle evaluation, not derived
from oracle maxima. The test-local integer definition imports no production
module. The runner checks all 48,960 nonempty prefixes against both concrete
paths, the shipped oracle and exact checker ranges; all three cover methods
receive the same 5,000-node/20-second cooperative limits. Association changes
the contract, so sharing policies are compared only on the same instance.

One bounded local coverage execution retained all 144 instances: 24 safe and
120 violated, no unknowns or missing cases, and 432 accepted certificates.
Fresh/syntactic/residue totals were 2,126/2,126/2,110 nodes. Residue used fewer
nodes than fresh in eight contracts and equal nodes in 136; syntactic tied in
all 144. Association changed pointwise outputs in 40 of 48 base/rounding
groups; those groups are not independent samples (the tolerance factor
duplicates execution semantics). This is finite coverage/topology evidence,
not a timing, general dominance or workload-prevalence claim. The retained
frozen null ablation, stress unknowns and all historical measurements above
remain unchanged.

Run `python3 -B tests/test_association_coverage.py -v` (five separate methods),
then `python3 -B scripts/association_coverage.py --out ../association-study`
into a new absent directory. No inherited cases/results are overwritten;
`write_collections` is never called. A separate scientific CI job requires
these controls and this bounded cohort, preserving raw outputs even on
failure. Its definition is not a successful hosted-run receipt. The local raw
cohort evidence is a separate review handoff, not inserted into the historical
result tree or its 188,847-unit ledger.

The current scalar exact evaluator reuses immutable summaries for identical
complete chunk-input tuples within a single layer, capped at 4,096 entries.
Every reference/carry transition is still evaluated, and each summary is rebuilt
under the declared association rather than incrementally imposing a left tree.
At the cap, further tuples compute without being cached. This bound is on
entries, not a measured peak-memory or end-to-end speedup claim. The existing
full-instance cache and every admission/deadline rule remain unchanged.
Run the separate portable four-method definition-oracle regression with
`python3 -B tests/test_scalar_summary_cache.py -v`; scientific CI requires it
and retains its log without changing the old 188,847-unit or ten-method gates.
All historical/native measurements above remain evidence for their original
pre-summary-reuse sources; no new timing has been performed.

The checker accepts a closed JSON schema: semantic integers must be JSON integers (not booleans, floats, or strings), duplicate keys and nonstandard numeric constants are rejected, and explicit byte/node/depth plus model-size limits are enforced. It uses a separately written concrete-integer interpreter plus complete multidimensional enumeration or exact scalar DP. It does not use the producer's numbered-atom abstraction to discharge a leaf. The producer and checker share the structural leaf-admission predicate solely to avoid producing leaves the checker cannot replay.

The checker remains code produced and reviewed within one development process, not an independently authored or machine-verified checker. General theorem arguments in `proofs/arguments.tex` are hand proofs. Finite tests validate implemented cases but do not mechanize those general arguments. The generated suites are discriminating semantic and certificate tests, not trained-model, accelerator, deployment, or workload-performance evidence. Input boxes are Cartesian; one scale and rounding mode apply per instance; other large multidimensional leaves can exceed the exact checker's admitted limit and force subdivision or `unknown`.

## License

Original artifact code, generated cases, results, and documentation are available under the MIT License in `LICENSE`. No ACM class or bibliography-style file is included in this standalone repository.
