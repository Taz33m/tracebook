# API Stability

`0.x` remains an alpha series, but public behavior is now divided deliberately.

## Compatibility Intent

- Top-level imports listed in `tracebook.__all__` are public.
- Installed `tracebook-*` console commands are public.
- Benchmark, simulation, event-log, replay-summary, corpus, conformance,
  minimization, suite-report, campaign, failure-corpus, and reproduction JSON
  structures are public artifacts and require schema tests plus changelog notes
  when changed. JUnit is a documented projection of those JSON artifacts; it
  does not imply pull-request annotation unless CI configures a reporter to
  consume it.
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
  public cross-language contracts. Bundled suite v1 and suite v2 both use
  manifest schema version 1; suite v2 is the current default, while suite v1 is
  retained for explicit historical reproduction. Conformance artifact types
  also use schema version 1. Changing field meaning, state ordering, decimal
  normalization, or hashing requires an explicit protocol/schema version
  decision. `suite_hash` binds case configuration and fixture identity; an
  intentional suite edit must update it.
- Protocol-v1 engine metadata may add the optional `revision` and `snapshot_id`
  identity fields. Ordinary adapters may omit them; a command using the three
  task-pinned `--candidate-*` flags requires an exact match before any event is
  sent. Evidence plan schema v1 and evidence manifest schema v1 are public
  artifacts for the canonical captured two-run qualification workflow.
- Campaign artifact schema version 1 is public. Campaign generator version 2,
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

Operating-system portability is not an artifact compatibility promise. The
atomic campaign and qualification evidence path is release-tested on Ubuntu and
requires descriptor-relative filesystem operations; Windows publication is
currently unsupported and fails closed rather than weakening atomicity.

## Deprecation Policy

Before 1.0, avoidable breaking changes receive a changelog entry and at least one
minor release of migration guidance when practical. Security and correctness
fixes may change previously unsafe behavior immediately.
