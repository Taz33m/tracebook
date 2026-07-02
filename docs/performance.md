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
| `cancel_deep` | 2,974 | 1,283 | 6,243 | 1,265 |
| `match` | 3,523,378 | 21,968 | 14,112,117 | 26,284 |
| `add_wide` | 687,178 | 62,035 | 3,484,626 | 1,510,401 |
| `add_deep` | 40,497 | 13,710 | 42,659 | 13,554 |

Four changes produced these numbers:

1. Keying each level's orders by an insertion-ordered dict (O(1) removal).
2. Replacing the price-level index's O(n) Python linear scan with `bisect`.
3. Iterating the FIFO match loop by taking the O(1) level head instead of
   copying the whole level per aggressive order.
4. Making `Order` and `Trade` plain `__slots__` classes instead of Numba
   `jitclass` types. Profiling the per-order path showed the jitclass was ~half
   the cost: `inspect`-based argument binding on every construction plus boxing
   on every field access, all paid because the matching loop is pure Python.
   This roughly tripled `add_deep` and `match` throughput.

`cancel_deep` and `match` are flat in the number of orders on a level (O(1) per
operation); `add_wide` stays super-linear in the number of distinct price levels
(`list` memmove). Numba is no longer used on the order-book path at all.

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
