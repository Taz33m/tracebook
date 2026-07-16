# Changelog

All notable changes to `tracebook` will be documented here.

The project follows [Keep a Changelog](https://keepachangelog.com/) conventions.

## Unreleased

- Consolidated native Rust adapter framing, validation, canonical state hashing,
  and process serving into one shared crate. Migrated package metadata,
  dependencies, scripts, and package-data declarations to `pyproject.toml`,
  removing duplicate Rust protocol copies and the legacy `setup.py` metadata
  source.
- Added qualification contract version 1 and `tracebook-conformance qualify`.
  One atomic evidence bundle now combines only the immutable fixed cases inside
  a declared profile, a deterministic generated campaign, complete semantic
  coverage, JSON, JUnit, and any minimized failure.
- Reworked the project plan around external-adoption evidence and equal-budget
  discovery experiments. Added a primary-source research roadmap covering
  Flash, RESTler, TCP-Fuzz, NEZHA, AFLNet, FEST, PMA, and modern LOB simulators,
  while explicitly deferring feedback-guided generation until it beats the
  current generator on held-out defects.
- Added a commit-pinned `gocronx/matcher` Rust adapter. Its `fifo-limit-v1`
  qualification passes 3/3 fixed cases, 25/25 generated traces, and 10/10
  semantic capabilities; JSON/JUnit evidence is retained in scheduled CI while
  two upstream contract questions remain explicitly pending.
- Published the frozen qualification-friction and held-out discovery study.
  Semantic-transition-guided suffix mutation improved one injected defect but
  regressed on the historical defect and a second injected defect, so no guided
  generator mode, profile, CLI, or artifact-contract change was shipped.

- Added immutable `tracebook-conformance-v2` as the default bundled suite while
  retaining v1 behind `sample --suite-version v1`. The new four-event case
  pins FIFO `CANCEL_RESTING` as cancel-on-encounter and exposes
  `orderbook-rs`'s valid cancel-all-at-touched-level policy as a documented
  state difference.
- Documented the reference engine's binary64 half-even tick behavior and why
  the Rust adapter's explicit cancel-and-new replacement path is load-bearing.
- Added `fifo-partial-fill-v1`, a hash-stable generated profile whose
  four-event continuation probe verifies that a partially filled FIFO maker
  retains priority over later same-price makers.
- Added a provenance-locked adapter for the historical `orderbook-rs` issue
  #88 revision. A 200-event campaign reproduces Flash's real maker-priority
  discrepancy at event 173, minimizes it to four events, and passes the same
  committed regression on current `orderbook-rs` 0.12.0.
- Documented the Flash-to-Tracebook discovery and minimization handoff as a
  bounded case study, with exact campaign, failure, JSON, and JUnit identities.
- Consumed Flash's merged schema-v1 canonical divergence at sequence 15738,
  converted its 15,739-message workload prefix, and reduced the actual upstream
  IDs, prices, quantities, and IOC instruction to a one-minimal four-event
  regression in 193 runs. The affected engine reproduces queue-priority drift;
  current `orderbook-rs` passes the same trace.
- Updated the maintained native adapter to `orderbook-rs` 0.12.0 and
  `pricelevel` 0.9.1 after upstream end-to-end validation. Added a native
  regression proving snapshots and matching expose the same queue-consumption
  order after an in-place quantity increase; the versioned campaign profiles
  remain unchanged.

## 0.4.1 - 2026-07-14

- Updated the maintained native adapter to the upstream-reviewed
  `orderbook-rs` 0.11.0 pin and configured deterministic, symbol-scoped native
  trade-ID namespaces. Protocol v1 continues to compare portable source-order
  fill semantics rather than candidate-private trade IDs.
- Recorded the upstream maintainer's confirmation of Tracebook's reduce,
  replace, queue-order, and maker/taker mappings. Clarified that generated
  profiles exclude in-place quantity increases; the review also prompted
  upstream priority documentation, property tests, and a snapshot-restore fix.

## 0.4.0 - 2026-07-13

- Began 0.4.0 development with deterministic, versioned differential campaigns.
  Stateful generated traces are driven only by the reference engine, stop at
  the first candidate divergence, and atomically emit the original failure,
  exact semantic report, and automatically minimized reproducer.
- Added `fifo-limit-v1` and `fifo-full-v1` campaign profiles, a specified
  cross-Python SplitMix64 generator, a stable campaign identity, and the
  `tracebook-conformance campaign` command for local and CI use.
- Locked candidate identity across campaign and minimization subprocesses,
  reserved the exact output path before candidate work, rejected changed output
  reservations without overwriting them, failed closed where descriptor-safe
  commits are unavailable, and fixed campaign IDs and trace hashes across the
  Python CI matrix.
- Added a native Rust adapter for exactly pinned `orderbook-rs` 0.10.4. It
  agrees on seven of eight standard cases, passes a 1,000-event generated FIFO
  instruction campaign, documents the intentional pro-rata boundary, and runs
  an injected Rust-side trade drift as a negative CI control. A scoped weekly
  Dependabot feed proposes reviewed Cargo pin and lockfile updates.
- Added generator version 2's structured FIFO queue-priority probe. The public
  seed-42 campaign detects an intentionally injected Rust maker-priority bug at
  original event 173 and reduces the failing prefix to a one-minimal five-event
  reproducer.
- Added deterministic failure-corpus bundles, stable failure classes,
  `tracebook-conformance reproduce`, the option-friendly `--candidate-cmd`
  form, and exact expected-versus-observed reproduction reports.
- Added reference-derived semantic capability coverage plus JSON and JUnit
  outputs across run, suite, minimization, campaign, and reproduction paths.
- Included both correct and intentionally faulty native Rust adapter source in
  the sdist, documented the `fifo-limit-v1` capability boundary, and updated the
  copy-paste CI workflow to install `tracebook-sim==0.4.0` from public PyPI.

## 0.3.0 - 2026-07-13

- Added the first maintained third-party engine integration: a commit-pinned
  PythonMatchingEngine adapter, a 13-event native FIFO/LIMIT compatibility
  trace, and a scheduled workflow that locks both the passing trace and the
  honest `2/8` standard-suite contract profile.
- Added a real-engine README demo, a concise conformance architecture diagram,
  a copy-paste GitHub Actions gate, and dedicated 0.3.0 release notes explaining
  the move from Python simulator to matching-engine correctness lab.
- Repositioned Tracebook as a matching-engine conformance and reproducible
  failure-analysis toolkit. Added a typed pluggable `EngineAdapter` API and an
  incremental, language-neutral NDJSON protocol for testing external Rust,
  C++, Java, Python, or other engines against the Tracebook reference path.
- Added `tracebook-conformance run` and `suite` with per-event applied/rejected
  outcomes, stable rejection codes, source-ID trades, canonical queue-state
  hashes, on-demand full snapshots, exact first-divergence paths, versioned JSON
  artifacts, subprocess timeouts, and final-snapshot verification.
- Added deterministic delta-debugging through `tracebook-conformance minimize`,
  which truncates after the first failure, emits a smaller JSONL reproducer,
  and distinguishes a proven one-minimal result from run-budget exhaustion.
- Added a hash-locked, redistributable eight-case conformance suite covering
  FIFO lifecycle and queue priority, IOC/FOK/market instructions, both
  self-trade policies, pro-rata allocation, multiple symbols, tick-grid edge
  cases, and deep-book cancellation pressure. Added property-based first-fault
  tests and wheel-installed suite smoke coverage.
- Added `OrderBook.get_resting_orders()` for a detached, matching-priority view
  of every resting order, including same-price queue order.
- Added `tracebook-corpus` for synchronized local Coinbase capture, offline
  corpus preparation, pre-write field allowlisting and order-ID
  pseudonymization, hash-locked manifests, exact golden replay verification,
  machine-attributed import benchmarks, and environment-aware report
  comparison. Live capture is an optional dependency and records that Coinbase
  market-data redistribution rights are not granted by sanitization.
- Added a distributable synthetic BTC-USD corpus with canonical snapshot,
  feed, normalized events, golden final state, and a stable corpus ID. System
  smoke and wheel CI now reproduce it exactly.
- Added the dependency-free `tracebook-coinbase` offline adapter for Coinbase
  Exchange REST L3 snapshots plus recorded `full` or compact `level3` feeds. It
  validates dynamic compact schemas and per-product sequences, streams
  normalized events, preserves observed exchange trades separately, ignores
  future protocol message types without skipping sequence validation, and
  rejects auction/crossed books rather than manufacturing queue state.
- Added priority-preserving `reduce_order()` and normalized `reduce` events for
  partial maker fills and same-price size decreases. Core event logs and replay
  summaries move to schema version 2; version 1 event logs remain readable.

## 0.2.0 - 2026-07-10

Trust, installability, and real-data workflow release.

- Renamed the Python distribution to `tracebook-sim` because the `tracebook`
  project name on PyPI belongs to an unrelated package. The import namespace and
  existing `tracebook-*` console commands remain unchanged.
- Added `tracebook-replay` and the `tracebook.events` API for normalized CSV,
  JSON, and JSONL order-event replay across multiple symbols. The strict mode
  fails with event context; lenient mode records rejections and continues.
- Kept imported source order ids stable across replacement and later cancellation,
  added configurable self-trade policy, and added optional trade output carrying
  both source and engine ids.
- Made replacement an atomic cancel-and-new transaction with callbacks delivered
  after unlocking, and preserved the original owner so replacement cannot bypass
  either self-trade-prevention policy.
- Detached accepted orders, lookup results, trades, and callback payloads from
  internal mutable state so callers cannot corrupt price levels or trade history.
- Tightened existing-order validation: ids, sides, order types, owners, timestamps,
  prices, and quantities are type-checked without lossy integer coercion, and
  malformed submissions return structured rejections.
- Added `clear` to deterministic event logs so recording and replay retain the
  identical final book after a reset. Recorded submissions now retain an
  externally supplied partial remaining quantity as well, and recording now
  requires a pristine book with no earlier non-resting activity.
- Made pro-rata FOK preflight mirror level-wide `CANCEL_INCOMING` execution.
- Made fallback profiling collect configured matching functions through a real
  Python profiling hook and emit completed calls and raw events.
- Made live frontend state atomic across depth, quote, trades, and statistics;
  corrected chronological trade-direction coloring.
- Made the optional dashboard command keep `--help` and `--version` usable from
  a core install and report the exact dashboard-extra install command otherwise.
- Separated achieved new-order rate from total event rate, made simulation metric
  collection and batch ingestion configuration effective, and hardened cleanup
  for failed or interrupted simulation and web sessions. Registered simulation
  completion callbacks now receive the result artifact instead of being inert.
- Added PEP 561 `py.typed` metadata, Python 3.12/3.13 CI, a 75% coverage gate,
  wheel-installed CLI smoke tests, package metadata checks, and a PyPI Trusted
  Publishing release workflow.
- Added a dependency-free live order-book web frontend (`tracebook-web`): a
  static HTML/CSS/JS page served by a stdlib HTTP server, backed by a background
  simulation, showing a live depth ladder, top-of-book quote, engine metrics,
  and a trade tape. It is loopback-only by default and ships in wheel/sdist.

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
