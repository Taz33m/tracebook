# Changelog

All notable changes to `tracebook` will be documented here.

The project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased

- Added a dependency-free live order-book web frontend (`tracebook-web`): a static HTML/CSS/JS page served by a stdlib HTTP server, backed by a background simulation, showing a live depth ladder, top-of-book quote, engine metrics, and a trade tape (polls `/api/state`). No Dash or JS build toolchain; loopback-only by default (`--allow-remote` to bind elsewhere). Ships the static assets in the wheel/sdist.

## 0.1.1 - 2026-07-02

Patch release from an adversarial QA pass: eight confirmed bugs fixed, each with a regression test.

- Required an empty book for `start_recording()` (QA found silent replay divergence): the event log captures only operations from the recording point on, so recording a book with pre-existing resting liquidity would replay to a different result (or crash on a later cancel of a pre-existing order). `start_recording()` now raises if any orders are resting.
- Stopped firing user callbacks while a lock is held (a deadlock vector found in QA): `replace_order` now submits the replacement with the book lock released, and `PerformanceMonitor.record_order_processing` fires alert callbacks after releasing the monitor lock. `replace_order` is now documented as cancel-then-new (its callbacks are lock-free like every other submission).
- Bounded the matching engine's trade history (found leaking in QA): `trades` is now a capped deque (retaining the recent tail for `get_recent_trades`) instead of an unbounded list, so a long-running book no longer grows memory per fill; `total_trades` remains the cumulative count.
- Hardened `MagicTraceSession.stop()` (found in QA): it now checks `poll()` before signalling, tolerates a self-exited child (magic-trace runs with a duration cap), always reaps the process, and marks the session inactive in a `finally` — previously a self-exited child raised `ProcessLookupError`, wedging the session and leaving a zombie.
- Fixed two tick-snapping matching bugs found in QA: an incoming limit whose price was within half a tick of a resting order could rest beside it and lock the book (the match decision used the raw price while resting prices were snapped), and a positive sub-half-tick price could snap to `0.0` and rest/execute for free. Incoming prices are now snapped onto the tick grid before validation and matching, and a price that snaps to a non-positive tick is rejected. The property-based invariants now fuzz off-grid prices.

## 0.1.0 - 2026-07-02

First tagged alpha release. Consolidates the full alpha development history.

- Enabled `check_untyped_defs` in the mypy baseline (still `mypy src/tracebook` clean) and fixed the untyped-body issues it surfaced (nullable thread handles and an optional trace-file guard).
- Published a local benchmark baseline covering every scenario, including the newer ones, in `docs/performance.md`.
- Added benchmark scenarios: `deep_book` (deep resting liquidity), `high_cancellation` (heavier cancel/replace mix), `pro_rata_cancellation` (pro-rata with lifecycle events), and `multi_symbol` (independent books per symbol). `BenchmarkScenario` gained a `symbols` field threaded through to the simulation.
- Added artifact-level schema tests that lock the structure of the public JSON outputs (benchmark report, simulation results, and the replay `EventLog`), round-tripped through `json` the way the CLIs write them, so a refactor cannot silently change the output contract.
- Extended the mypy type-check baseline to the whole `src/tracebook` package and widened the CI gate to `mypy src/tracebook`. Fixed a real type-contract gap (the `OrderGenerator` ABC now declares the `order_factory`/`price_model` attributes its subclasses set and the stream assigns), made `SimulationConfig.symbols` a non-optional field with a default factory, and annotated the profiling/dashboard/simulation collections and implicit-`Optional` parameters.
- Established a mypy type-check baseline on the `core` package and enforce it in CI (initially `mypy src/tracebook/core`, clean). Added the annotations to get there, guarded `infer_price_decimals` against non-int `Decimal` exponents, and restored `mypy` as a dev dependency.
- Added a trusted fast path for orders built by the book's own factory: since the factory already validates side/type/price/quantity/owner and uses the book symbol with a fresh id, the `add_*`/`submit_*` methods skip the redundant book-level re-validation, symbol re-normalization, factory-id reconciliation, and a resting-state index lookup. Cut `add_deep` ~26% and `match` ~19%. The public `add_order`/`submit_order` (which accept externally built orders) still fully validate.
- Dropped `numba` as a dependency and removed the standalone `algorithms/{fifo,pro_rata}.py` analysis helpers, which were never wired into the live matching path. With `Order`/`Trade`/`PriceLevel` all plain Python, nothing on the order-book path used Numba; the package now imports and runs with Numba absent (verified with the import blocked).
- Made `Order` and `Trade` plain `__slots__` classes instead of Numba `jitclass` types. Profiling the per-order path showed the jitclass was about half the cost (`inspect`-based argument binding on every construction plus field-access boxing, all paid because the matching loop is pure Python); this roughly tripled `add_deep` and `match` throughput. Numba is no longer used on the order-book path.
- Made the FIFO match loop copy-free: it now iterates by repeatedly taking the O(1) price-level head instead of copying the entire level per aggressive order, so matching against a deep level is flat (O(1) per match) rather than O(level size). Verified by the `match` microbenchmark (now flat in level size) and a deep-sweep FIFO-order test.
- Replaced the price-level index's O(n) Python linear-scan insert and `list.remove` with `bisect` (per-side key), removing the Python-loop cost from level insert/removal. A local comparison against `sortedcontainers.SortedList` (command, machine, and Python cited in `docs/performance.md`) found `bisect` + list faster for realistic level counts, so no sorted-container dependency was added.
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
