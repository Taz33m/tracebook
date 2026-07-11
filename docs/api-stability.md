# API Stability

`0.x` remains an alpha series, but public behavior is now divided deliberately.

## Compatibility Intent

- Top-level imports listed in `tracebook.__all__` are public.
- Installed `tracebook-*` console commands are public.
- Benchmark, simulation, event-log, replay-summary, corpus-manifest,
  golden-state, and corpus-comparison JSON structures are public artifacts and
  require schema tests plus changelog notes when changed.
- Objects returned by order submission, lookup, recent-trade, and callback APIs
  are detached from internal mutable state.
- Normalized replay `order_id` values are source identifiers; replacement keeps
  them addressable even when the engine allocates a new internal id.
- `EventLog` version 2 and normalized replay-summary version 2 add
  priority-preserving `reduce` lifecycle events. Version 1 event logs remain
readable; summary consumers should branch on `schema_version`.
- Corpus schema version 1 binds canonical source, events, and golden state by
  hash. A format change must create a new schema version; changing a fixture's
  corpus ID requires explicit review.

Private methods, internal matching data structures, dashboard layout internals,
and synthetic generator implementation details may change during the alpha.

## Deprecation Policy

Before 1.0, avoidable breaking changes receive a changelog entry and at least one
minor release of migration guidance when practical. Security and correctness
fixes may change previously unsafe behavior immediately.
