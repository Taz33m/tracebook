# Tracebook Project Plan

## Product Position

`tracebook` is an inspectable Python market-microstructure workbench for matching
semantics, normalized order-event replay, synthetic workload experiments, and
honest local profiling. It is distributed as `tracebook-sim` and imported as
`tracebook`.

The project optimizes for deterministic behavior and auditability before raw
speed. It is not exchange connectivity or production trading infrastructure.

## Maintained Surfaces

```text
src/tracebook/
  core/             matching, lifecycle, snapshots, deterministic replay
  events/           CSV/JSON/JSONL normalized event loading and replay
  simulation/       synthetic order flow and paced workload execution
  benchmarks/       reproducible local scenario reports
  profiling/        metrics, magic-trace, and selected-function fallback tracing
  visualization/    Dash dashboard and dependency-free live web frontend
```

## Next Milestones

1. Publish `tracebook-sim` 0.2.0 through PyPI Trusted Publishing and verify the
   clean-environment quickstart on macOS and Linux.
2. Add documented adapters for one public crypto L3 feed and one equities order
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
