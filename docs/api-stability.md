# API Stability

`0.x` remains an alpha series, but public behavior is now divided deliberately.

## Compatibility Intent

- Top-level imports listed in `tracebook.__all__` are public.
- Installed `tracebook-*` console commands are public.
- Benchmark, simulation, event-log, replay-summary, corpus, conformance,
  minimization, suite-report, and campaign JSON structures are public artifacts
  and require schema tests plus changelog notes when changed.
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
- Conformance protocol version 1 and its observation/state wire structures are
  public cross-language contracts. The standard suite manifest and all three
  conformance artifact types use schema version 1. Changing field meaning,
  state ordering, decimal normalization, or hashing requires an explicit
  protocol/schema version decision. `suite_hash` binds case configuration and
  fixture identity; an intentional suite edit must update it.
- Campaign artifact schema version 1 is public. Campaign generator version 1,
  the built-in versioned profile definitions, seed derivation, and trace hashes
  are reproducibility contracts. An intentional generation change requires a
  new generator or profile version rather than silently changing existing
  campaign output.
- `OrderBook.get_resting_orders()` returns detached orders in matching-priority
  order. Same-price list order is observable public behavior.

Private methods, internal matching data structures, dashboard layout internals,
and non-campaign synthetic generator implementation details may change during
the alpha.
Adapters under `integrations/` are maintained source examples, not installed
package APIs. Their upstream revision, native compatibility trace, and expected
profile are pinned, but private APIs used to inspect an external engine can
change only with an integration test and documentation update.

## Deprecation Policy

Before 1.0, avoidable breaking changes receive a changelog entry and at least one
minor release of migration guidance when practical. Security and correctness
fixes may change previously unsafe behavior immediately.
