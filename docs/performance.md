# Performance Baselines

`tracebook` does not publish fixed throughput or latency claims until they are measured on the target machine. Use the benchmark runner to create local, reproducible reports.

## Benchmark Command

```bash
tracebook-benchmark \
  --scenario all \
  --seed 1337 \
  --warmup-seconds 0.05 \
  --output benchmark_results/local.json
```

## Scenarios

- `smoke`: short CI-friendly FIFO run.
- `fifo_baseline`: FIFO matching baseline.
- `pro_rata_baseline`: pro-rata matching baseline.
- `cancellation_mix`: FIFO run with cancel and replace events.
- `all`: runs every scenario above.

## Report Shape

Each JSON report includes:

- machine metadata and dependency versions
- scenario config, seed, and warmup duration
- measured order throughput
- mean, p50, p95, p99, and max matching latency
- generation latency reported separately from matching latency
- cancel/replace event latency reported separately from new-order matching latency
- memory usage and monitoring overhead

Use these files as local baselines. If README performance numbers are added later, they should cite the command, seed, machine, Python version, and report file used to produce them.

## Local Baseline Sample

Measured on May 9, 2026 with:

- Python: 3.11.5
- Platform: macOS-15.4.1-arm64-arm-64bit
- Command: `tracebook-benchmark --scenario all --duration 1 --throughput 100 --seed 2026 --warmup-seconds 0.05 --output /private/tmp/tracebook-benchmark-doc-baseline-current.json`

These are local smoke baselines only. They are useful for regression checks on this machine, not portable performance claims.

| Scenario | Orders | Throughput ops/s | Mean ms | P50 ms | P95 ms | P99 ms | Generation mean ms | Event mean ms | Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 100 | 99.93 | 2.922 | 0.060 | 0.219 | 3.186 | 0.855 | 0.000 | 134.28 |
| fifo_baseline | 110 | 109.98 | 0.139 | 0.112 | 0.363 | 0.512 | 1.181 | 0.000 | 134.31 |
| pro_rata_baseline | 100 | 99.93 | 0.124 | 0.095 | 0.398 | 0.630 | 0.868 | 0.000 | 134.19 |
| cancellation_mix | 112 | 111.53 | 0.130 | 0.087 | 0.369 | 0.590 | 1.752 | 0.036 | 100.03 |

## Scenario Baselines (all scenarios)

A separate local sample covering every scenario, including the newer ones
(`deep_book`, `high_cancellation`, `pro_rata_cancellation`, `multi_symbol`).
Measured on July 2, 2026 with:

- Python: 3.10.5
- Platform: macOS-15.4.1-arm64-arm-64bit
- Command: `tracebook-benchmark --scenario all --duration 1 --throughput 100 --seed 2026 --warmup-seconds 0.05`

Local sample only, on a different machine than the table above; latency is
wall-clock and thread-scheduling dependent, so tail figures vary run to run
(the `multi_symbol` P99 below is one such spike across its three books).

| Scenario | Orders | Throughput ops/s | Mean ms | P50 ms | P95 ms | P99 ms | Generation mean ms | Event mean ms | Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 100 | 99.9 | 0.045 | 0.030 | 0.128 | 0.240 | 0.512 | 0.000 | 32.8 |
| fifo_baseline | 110 | 110.0 | 0.038 | 0.028 | 0.105 | 0.145 | 0.436 | 0.000 | 27.1 |
| pro_rata_baseline | 100 | 99.9 | 0.052 | 0.032 | 0.143 | 0.307 | 0.586 | 0.000 | 27.4 |
| cancellation_mix | 102 | 101.9 | 0.046 | 0.030 | 0.143 | 0.159 | 0.511 | 0.026 | 26.1 |
| deep_book | 100 | 99.9 | 0.062 | 0.036 | 0.163 | 0.335 | 0.846 | 0.000 | 21.9 |
| high_cancellation | 113 | 112.9 | 0.038 | 0.028 | 0.085 | 0.199 | 0.405 | 0.031 | 22.2 |
| pro_rata_cancellation | 103 | 102.9 | 0.043 | 0.029 | 0.102 | 0.180 | 0.455 | 0.033 | 22.6 |
| multi_symbol | 108 | 108.0 | 0.127 | 0.034 | 0.219 | 2.336 | 1.219 | 0.000 | 23.2 |

## Operation Microbenchmarks

The simulation benchmark above is wall-clock and thread-scheduling dependent. To
judge data-structure changes in isolation, a deterministic single-threaded
microbenchmark measures individual operations:

```bash
python tests/benchmarks/microbench.py --n 5000
python tests/benchmarks/microbench.py --n 20000
```

The `ns/op` figures below were measured locally and are illustrative, not
portable claims:

- Python: 3.10.5
- Platform: macOS-15.4.1-arm64-arm-64bit
- Harness: `tests/benchmarks/microbench.py` (deterministic, single-threaded,
  1,000-op warmup), at `--n 5000` and `--n 20000`

What is portable is the **scaling**: keying each price level's orders by an
insertion-ordered dict makes in-level removal O(1), so `cancel_deep` is now flat
in the number of orders on a level instead of growing with it. Moving the level
off the Numba `jitclass` (whose typed containers were driven from the pure-Python
matching loop, paying boundary cost on every access) additionally sped up the
matching path substantially.

| Scenario | ns/op original (n=5k) | ns/op now (n=5k) | ns/op original (n=20k) | ns/op now (n=20k) |
| --- | ---: | ---: | ---: | ---: |
| `cancel_deep` | 2,974 | 1,290 | 6,243 | 1,285 |
| `match` | 3,523,378 | 17,779 | 14,112,117 | 22,443 |
| `add_wide` | 687,178 | ~65,000 | 3,484,626 | 1,492,484 |
| `add_deep` | 40,497 | 10,206 | 42,659 | 10,066 |

Five changes produced these numbers:

1. Keying each level's orders by an insertion-ordered dict (O(1) removal).
2. Replacing the price-level index's O(n) Python linear scan with `bisect`.
3. Iterating the FIFO match loop by taking the O(1) level head instead of
   copying the whole level per aggressive order.
4. Making `Order` and `Trade` plain `__slots__` classes instead of Numba
   `jitclass` types. Profiling the per-order path showed the jitclass was ~half
   the cost: `inspect`-based argument binding on every construction plus boxing
   on every field access, all paid because the matching loop is pure Python.
5. A trusted fast path for orders built by the book's own factory: the factory
   already validates side/type/price/quantity/owner and uses the book symbol
   with a fresh id, so the redundant book-level re-validation, symbol
   re-normalization, factory-id reconciliation, and a resting-state index
   lookup are skipped. This cut `add_deep` ~26% and `match` ~19%.

`cancel_deep` and `match` are flat in the number of orders on a level (O(1) per
operation); `add_wide` stays super-linear in the number of distinct price levels
(`list` memmove) and is high-variance at n=5k because it rebuilds the whole book
each call. Numba is no longer used on the order-book path at all.

### Why `bisect` and not a sorted-container dependency

`add_wide` is still super-linear at large level counts: `bisect` gives an
O(log n) search but `list` insert/remove is an O(n) memmove. A true O(log n)
structure (`sortedcontainers.SortedList`) would fix that -- but only helps once a
single book holds thousands of distinct price levels, which is unrealistic. An
isolated comparison of the index insert (random ticks) shows the crossover:

| Distinct levels | `bisect` + list ns/op | `SortedList` ns/op |
| ---: | ---: | ---: |
| 2,000 | 1,455 | 2,094 |
| 5,000 | 2,177 | 2,428 |
| 20,000 | 5,583 | 2,709 |
| 50,000 | 11,856 | 2,766 |

(Isolated index-insert comparison of random ticks, same machine and Python as
above, 1,000-op warmup.) For realistic level counts (hundreds to a few thousand)
`bisect` + list is faster than `SortedList` and adds no dependency; the sorted
container only wins for pathologically deep books. So no dependency was taken.
