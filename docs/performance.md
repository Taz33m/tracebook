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

Measured on May 7, 2026 with:

- Python: 3.13.0
- Platform: macOS-15.4.1-arm64-arm-64bit-Mach-O
- Command: `tracebook-benchmark --scenario all --duration 1 --throughput 100 --seed 2026 --warmup-seconds 0.05 --output /private/tmp/tracebook-benchmark-doc-baseline.json`

These are local smoke baselines only. They are useful for regression checks on this machine, not portable performance claims.

| Scenario | Orders | Throughput ops/s | Mean ms | P50 ms | P95 ms | P99 ms | Generation mean ms | Event mean ms | Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke | 200 | 199.95 | 1.390 | 0.032 | 0.115 | 0.205 | 5.307 | 0.000 | 168.98 |
| fifo_baseline | 200 | 199.91 | 0.063 | 0.048 | 0.156 | 0.262 | 4.804 | 0.000 | 130.95 |
| pro_rata_baseline | 200 | 194.37 | 0.195 | 0.052 | 0.249 | 1.539 | 3.109 | 0.000 | 124.52 |
| cancellation_mix | 103 | 102.96 | 0.041 | 0.026 | 0.097 | 0.115 | 2.006 | 0.022 | 124.75 |
