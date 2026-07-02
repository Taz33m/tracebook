# Changelog

All notable changes to `tracebook` will be documented here.

The project follows a lightweight alpha changelog until formal semantic-versioned releases begin.

## Unreleased

- Replaced the price-level index's O(n) Python linear-scan insert and `list.remove` with `bisect` (per-side key), removing the Python-loop cost from level insert/removal. Measured against `sortedcontainers.SortedList`, `bisect` + list is faster for realistic level counts (the crossover is ~5k distinct levels), so no sorted-container dependency was added; see `docs/performance.md`.
- Made in-level order removal O(1) by keying each price level's orders with an insertion-ordered dict (preserving FIFO order), replacing the previous O(n) linear scan. Moving `PriceLevel` off the Numba `jitclass` also removed per-access boundary cost from the pure-Python matching loop, substantially speeding up matching. Added a deterministic operation microbenchmark (`tests/benchmarks/microbench.py`) and before/after numbers in `docs/performance.md`.
- Added self-trade prevention: orders carry an optional `owner` id, and `OrderBook(self_trade_policy=...)` supports `CANCEL_RESTING` and `CANCEL_INCOMING` (default `NONE`). Both policies keep the book uncrossed, FOK fillability mirrors each policy (own liquidity excluded, and `CANCEL_INCOMING` stops at the first same-owner order so a FOK never partially executes), and the policy plus owners are captured in the replay log. Exported `SelfTradePolicy` and `NO_OWNER`.
- Made replay fail fast on a malformed or incompatible log: a hard-rejected submission (via a new `OrderResult.accepted` flag) or a cancel that finds no order now raises instead of diverging silently; soft outcomes such as an unfillable FOK still replay faithfully.
- Computed rolling throughput with an O(1) running window total instead of re-summing the deque on every order.
- Made `infer_price_decimals` use the tick's exact decimal representation so fine or scientific-notation tick sizes no longer collapse canonical prices to whole numbers.
- Added deterministic record/replay: `OrderBook.start_recording()`/`stop_recording()` capture a serializable `EventLog`, and `replay()` reconstructs the identical trade sequence and final book state (also across a JSON round-trip). Exported `EventLog` and `replay` from the package root.
- Reported throughput as a rolling one-second window rate instead of a lifetime cumulative average, so the dashboard shows an instantaneous rate.
- Removed the synchronous psutil sweep from the order-processing hot path; alerts and summaries now read the background-sampled resource snapshot.
- Attributed a replacement that crosses the book to matching latency rather than lifecycle-event latency, so replace-heavy scenarios no longer hide matching cost.
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
