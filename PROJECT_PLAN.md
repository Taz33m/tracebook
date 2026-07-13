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
  conformance/      adapters, semantic diffing, trace minimization, standard suite
  events/           normalized event replay and offline venue adapters
  corpus/           safe local capture, manifests, golden state, import benchmarks
  simulation/       synthetic order flow and paced workload execution
  benchmarks/       reproducible local scenario reports
  profiling/        metrics, magic-trace, and selected-function fallback tracing
  visualization/    Dash dashboard and dependency-free live web frontend
integrations/
  python_matching_engine/  pinned external adapter and compatibility trace
```

## Next Milestones

1. Validate the protocol with one independently implemented Rust, C++, or Java
   adapter and publish it as an integration example.
2. Add grammar-aware, property-based trace generation for lifecycle and
   self-trade-prevention state machines.
3. Add a benchmark mode for candidate adapters that separates protocol overhead
   from engine execution and never presents it as production latency.
4. Stabilize the protocol, event, and report schemas after external-adapter
   feedback, then publish a 1.0 policy.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Conformance protocol changes require cross-process tests, artifact schema
  tests, a version decision, and migration notes.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
