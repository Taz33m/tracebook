# Tracebook Project Plan

## Product Position

`tracebook` is the conformance-testing and reproducible failure-analysis toolkit
for matching engines. It runs canonical order-event traces against its
inspectable Python reference engine and external engines, localizes semantic
drift, reduces failures, and keeps correctness and performance artifacts
reproducible. It is distributed as `tracebook-sim` and imported as `tracebook`.

The project optimizes for deterministic behavior and auditability before raw
speed. Its optional public capture is research tooling, not a production feed
handler or trading infrastructure.

## Maintained Surfaces

```text
src/tracebook/
  core/             matching, lifecycle, snapshots, deterministic replay
  conformance/      adapters, campaigns, semantic diffing, minimization, standard suite
  events/           normalized event replay and offline venue adapters
  corpus/           safe local capture, manifests, golden state, import benchmarks
  simulation/       synthetic order flow and paced workload execution
  benchmarks/       reproducible local scenario reports
  profiling/        metrics, magic-trace, and selected-function fallback tracing
  visualization/    Dash dashboard and dependency-free live web frontend
integrations/
  python_matching_engine/  pinned external adapter and compatibility trace
  orderbook_rs/             native Rust adapter, faulty example, regression proof
```

## Next Milestones

1. Get the `orderbook-rs` adapter reviewed or acknowledged by an upstream
   maintainer, and record any protocol or translation feedback publicly.
2. Help one third-party engine author connect their own Rust, C++, Java, or
   Python engine and adopt a saved regression trace in CI.
3. Use those external adapters to find and publish one real semantic discrepancy
   that is not intentionally injected.
4. Revisit candidate benchmarking and additional profiles only after adapter
   authors reveal where protocol v1 is awkward or underspecified.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Conformance protocol changes require cross-process tests, artifact schema
  tests, a version decision, and migration notes.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
