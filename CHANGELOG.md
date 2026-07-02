# Changelog

All notable changes to `tracebook` will be documented here.

The project follows a lightweight alpha changelog until formal semantic-versioned releases begin.

## Unreleased

- Made `replace_order` atomic: an invalid replacement is rejected before the original is cancelled, and a replacement that fails to submit after cancellation restores the original resting order, so a replace never destroys liquidity.
- Bounded the order-id replay guard so long-running books no longer leak memory (most-recent-N window, configurable via `_seen_id_cap`).
- Keyed price levels by integer ticks (configurable `tick_size`, default `0.01`); prices now snap to a canonical grid, removing the float-identity hazard of dict-keying on raw prices.
- Consolidated execution pricing to a single rule (matches fill at the resting/maker price) and removed the unused JIT `match_orders_fifo` helper and the divergent timestamp-based `calculate_match_price`.
- Added property-based matching invariants (quantity conservation, no crossed book, level consistency) and tick-grid tests; added `hypothesis` as a dev dependency.
- Moved package code under the `tracebook` namespace.
- Added structured order submission results and richer order book lifecycle APIs.
- Added event-based simulation with new, cancel, and replace events.
- Added reproducible benchmark runner and local performance baseline docs.
- Added dashboard demo simulation support.
- Added GitHub Actions CI, packaging metadata, and open-source project templates.
- Reworked the README and docs into an artifact-first professional repo front door.
- Added release guidance, architecture docs, command docs, and richer package metadata.
- Added a source distribution manifest for docs, examples, tests, and project metadata.
- Hardened symbol, enum, numeric, duplicate-order-id, dashboard host, and magic-trace artifact validation.
- Made simulation lifecycle replacement IDs thread-safe and documented analyzer output as not yet collected.
- Refreshed local benchmark sample numbers from the current benchmark runner.
- Removed the unused mypy dev dependency/config until typed-package support has a real passing baseline.
- Added compile and dependency consistency checks to CI and clarified dependency grouping.
