# Tracebook Project Plan

## Product Position

`tracebook` is an inspectable Python market-microstructure workbench for matching
semantics, normalized order-event replay, verified local data corpora,
synthetic workload experiments, and honest local profiling. It is distributed
as `tracebook-sim` and imported as `tracebook`.

The project optimizes for deterministic behavior and auditability before raw
speed. Its optional public capture is research tooling, not a production feed
handler or trading infrastructure.

## Maintained Surfaces

```text
src/tracebook/
  core/             matching, lifecycle, snapshots, deterministic replay
  events/           normalized event replay and offline venue adapters
  corpus/           safe local capture, manifests, golden state, import benchmarks
  simulation/       synthetic order flow and paced workload execution
  benchmarks/       reproducible local scenario reports
  profiling/        metrics, magic-trace, and selected-function fallback tracing
  visualization/    Dash dashboard and dependency-free live web frontend
```

## Next Milestones

1. Complete the 0.3.0 corpus contract with stable manifest, golden-state, and
   benchmark/comparison schemas.
2. Follow the Coinbase Exchange L3 adapter with one documented equities order
   event format, both normalizing into `MarketEvent`.
3. Separate paced workload reports from an explicit unpaced engine-capacity
   benchmark so the two measurements cannot be confused.
4. Stabilize the public API and event/report schemas, then publish a 1.0 policy.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
