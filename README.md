<p align="center">
  <img src="docs/logo.png" alt="tracebook logo" width="200"/>
</p>

<h1 align="center">tracebook</h1>

<p align="center">
  <strong>Conformance testing and reproducible failure analysis for matching engines.</strong>
</p>

<p align="center">
  <a href="https://github.com/Taz33m/tracebook/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/Taz33m/tracebook/actions/workflows/ci.yml/badge.svg"/></a>
  <a href="https://github.com/Taz33m/tracebook/actions/workflows/orderbook-rs.yml"><img alt="orderbook-rs integration" src="https://github.com/Taz33m/tracebook/actions/workflows/orderbook-rs.yml/badge.svg"/></a>
  <a href="LICENSE"><img alt="License: MIT" src="https://img.shields.io/badge/license-MIT-green"/></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10--3.13-blue"/>
  <img alt="matching" src="https://img.shields.io/badge/matching-FIFO%20%2B%20pro--rata-7fc7a6"/>
  <img alt="tests" src="https://img.shields.io/badge/tests-292%20passing-brightgreen"/>
  <img alt="claims" src="https://img.shields.io/badge/claims-bounded-important"/>
</p>

> **TL;DR:** Give Tracebook a normalized event trace and an adapter for your Rust, C++, Java, Python, or other matching engine. It runs the trace against an inspectable reference engine, identifies the first difference in outcomes, trades, resting orders, or queue priority, and reduces failures to a small reproducible trace. It also retains deterministic replay, verified L3 data workflows, and explicitly bounded local benchmarks.

## The Five-Event Rust Failure

The release demo starts with the correct `orderbook-rs` adapter, then runs the
same engine with one intentionally injected queue-priority defect. Build both
native binaries:

```bash
python3 -m pip install -e .
cargo build --release --locked --manifest-path integrations/orderbook_rs/Cargo.toml
```

Generate 200 lifecycle events per trace. Seed 42 places a structured FIFO probe
at events 169-173; candidate behavior never influences generation:

```bash
tracebook-conformance campaign \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 1000 \
  --events-per-trace 200 \
  --candidate-cmd ./integrations/orderbook_rs/target/release/faulty-orderbook-adapter \
  --corpus-dir .tracebook/corpus \
  --stop-after-first \
  --junit-output .tracebook/campaign.xml
```

The command exits `1`, as a CI conformance gate should, and reports:

```text
Divergence detected at original event 173
Failure class: queue-priority drift
Original trace: 173 events
Reduced reproducer: 5 events
Campaign seed: 42
Campaign hash: sha256:3630ea1789e27b6e416e1e241ca4ef8d5a28f0f2bf660fae3c0a1c8ff39ba7c6
Failure id: failure-bc8b19d3e0e3441a98db
```

Replay the five-event result and require the exact stored maker-ID mismatch:

```bash
tracebook-conformance reproduce \
  .tracebook/corpus/failure-bc8b19d3e0e3441a98db/reduced.jsonl \
  --candidate-cmd ./integrations/orderbook_rs/target/release/faulty-orderbook-adapter
```

Then use that same trace as a regression gate for the correct engine:

```bash
tracebook-conformance run \
  .tracebook/corpus/failure-bc8b19d3e0e3441a98db/reduced.jsonl \
  --candidate-cmd ./integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The first command verifies that a known failure remains exactly reproducible;
the second exits `0` only when the candidate follows FIFO replacement priority.
The corpus bundle contains `campaign.json`, `failure.json`, `original.jsonl`,
`reduced.jsonl`, detailed minimization evidence, and semantic coverage. JSON and
JUnit are first-class outputs, so the same evidence is readable by people, CI
test reporters, and downstream tooling.

## Differential Campaigns

`fifo-limit-v1` is the portable public profile: FIFO limit orders, partial
fills, cancellation, reduction, replacement, clear, duplicate active IDs,
inactive lifecycle requests, multiple symbols, and the structured FIFO
priority probe. `fifo-full-v1` adds market, IOC, and FOK instructions. Generator
version 2 uses specified SplitMix64 trace seeds and deterministic probe
placement. Profile name, generator version, campaign seed, requested trace
count, and events per trace form the campaign hash.

Every report states which declared capabilities had reference-observed evidence
in candidate-compared events. This is semantic workload coverage, not Python
line coverage and not a claim about unsupported candidate features. See the
[capability profile and artifact contract](docs/conformance.md).

## Real-Engine Demos

### Native Rust: orderbook-rs

The strongest maintained integration wraps the independently implemented,
MIT-licensed [`orderbook-rs`](https://github.com/joaquinbejar/OrderBook-rs)
0.10.4. The candidate is a native Rust process: no Tracebook Python matching
code runs on its side of the protocol.

```bash
python3 -m pip install -e .
cd integrations/orderbook_rs
cargo build --release --locked
cd ../..

tracebook-conformance run \
  integrations/orderbook_rs/fifo-compatible.jsonl \
  --output /tmp/orderbook-rs-report.json \
  --candidate integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The 13-event trace exits `0` with this proof:

```json
{
  "candidate_engine": {
    "language": "Rust",
    "name": "orderbook-rs FIFO adapter",
    "version": "0.10.4"
  },
  "compared_events": 13,
  "conformant": true,
  "final_state_hash": "21a9606e7c77c3b239259f5032245c6330ddcd1d3f7fa25394612d9818becee3"
}
```

Against the hash-locked standard suite it agrees on `7/8` cases: FIFO
lifecycle, all order instructions, both STP modes, multiple symbols, tick-grid
behavior, and deep cancellation. The one expected difference is
`pro-rata-allocation`, because upstream implements FIFO. It also passes a
deterministic 1,000-event `fifo-full-v1` campaign with ID
`sha256:95c3dac9d27b770a5cccebe9ff16b6e71af443001d633b640983f02f3e04b3c9`.

The integration also ships the separate, clearly named
`faulty-orderbook-adapter` binary used by the event-173 demonstration. Its
source is included in the public sdist beside the correct adapter, and CI proves
both the five-event failure and the corrected regression case.

See the [native adapter, architecture, boundaries, and copyable CI
gate](integrations/orderbook_rs/README.md).

### Python: PythonMatchingEngine

The repository includes a pinned integration with the MIT-licensed
[PythonMatchingEngine](https://github.com/Surbeivol/PythonMatchingEngine). This
is a real second implementation, not a renamed Tracebook adapter.

```bash
git clone https://github.com/Surbeivol/PythonMatchingEngine.git /tmp/PythonMatchingEngine
git -C /tmp/PythonMatchingEngine checkout f94150294a85d7b415ca4518590b5a661d6f9958
python -m pip install -e . "pandas>=2.3.3" "PyYAML>=6.0.2"

export PYTHON_MATCHING_ENGINE_PATH=/tmp/PythonMatchingEngine
tracebook-conformance run \
  integrations/python_matching_engine/fifo-compatible.jsonl \
  --output /tmp/python-matching-engine-report.json \
  --timeout 20 \
  --candidate python integrations/python_matching_engine/adapter.py
```

Relevant report fields:

```json
{
  "candidate_engine": {
    "language": "Python",
    "name": "PythonMatchingEngine FIFO/LIMIT",
    "version": "f94150294a85"
  },
  "compared_events": 13,
  "conformant": true,
  "divergence": null
}
```

The same pinned engine also passes the first generated campaign gate:

```bash
tracebook-conformance campaign \
  --output-dir /tmp/python-matching-engine-campaign \
  --profile fifo-limit-v1 \
  --seed 20260713 \
  --traces 10 \
  --events-per-trace 50 \
  --timeout 20 \
  --candidate python integrations/python_matching_engine/adapter.py
```

That run checks 500 generated events and records campaign ID
`sha256:53e31761dbcc5b5858506c7f11b81b1ad9cae281d46fb8212c1d62a89d058a2d`.

Those 13 events cover FIFO fills, decimal partial fills, reduction, cancellation,
replacement priority, rejection, clear, and multiple symbols. Against the full
eight-case suite, the same unmodified engine agrees on two native FIFO cases and
reports six expected contract differences for instructions, STP, pro-rata,
market orders, and tick policy. Tracebook records those differences instead of
calling a narrower feature set a failure or silently emulating it.

See the [Python integration guide](integrations/python_matching_engine/README.md)
and the [generic copy-paste CI workflow](docs/ci.md). The
[0.4.0 release notes](docs/releases/0.4.0.md) explain the failure-corpus release;
the [0.3.0 notes](docs/releases/0.3.0.md) explain the original project-category
change.

## Video Walkthrough

<p align="center">
  <a href="https://youtu.be/RXOcB2k7qTQ">
    <img src="https://img.youtube.com/vi/RXOcB2k7qTQ/maxresdefault.jpg" alt="Trace The Match video walkthrough" width="820"/>
  </a>
</p>

Watch **Trace The Match** on YouTube: https://youtu.be/RXOcB2k7qTQ

## Best Way To Review

1. Run the bundled conformance suite through the example external adapter.
2. Inspect a first-divergence artifact and a minimized failing trace.
3. Run the unit tests and system smoke.
4. Verify the checked Coinbase corpus and its deterministic golden state.
5. Generate a benchmark JSON report with warmup excluded.
6. Read the claims and limitations before treating any number as production latency.

```bash
python -m pip install -e ".[dev,dashboard]"
python -m pytest
python test_system.py
tracebook-conformance sample /tmp/tracebook-conformance-v1
tracebook-conformance suite /tmp/tracebook-conformance-v1 --candidate python examples/conformance_adapter.py
tracebook-conformance campaign --output-dir /tmp/tracebook-campaign --traces 3 --events-per-trace 25 --candidate python examples/conformance_adapter.py
tracebook-replay examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json
tracebook-coinbase examples/data/coinbase_btcusd_l3_snapshot.json examples/data/coinbase_btcusd_full.jsonl --tick-size 0.01 --output /tmp/tracebook-coinbase.json
tracebook-corpus sample /tmp/tracebook-sample-corpus
tracebook-corpus verify /tmp/tracebook-sample-corpus
tracebook-sim --duration 1 --throughput 50 --algorithm FIFO --seed 1337 --cancel-ratio 0.05 --replace-ratio 0.02 --warmup-seconds 0.01
tracebook-benchmark --scenario smoke --seed 1337 --warmup-seconds 0.01 --output benchmark_results/smoke.json
```

## Why This Matters

Matching-engine bugs often hide behind a long event prefix: a reduction that should retain priority, a replacement that should lose it, a partial fill followed by self-trade prevention, or a cancellation addressed through an old internal ID. A final depth snapshot can look plausible while the wrong order sits at the front of a queue.

`tracebook` makes those semantics executable. It compares a candidate after every event, uses source IDs across implementations, hashes the complete queue state, requests a full snapshot only when needed, and preserves the exact disagreement as a versioned artifact. Its minimizer removes irrelevant prefixes and events, then states whether the result is one-minimal or stopped at its run budget.

The reference engine is intentionally small and readable. It is an oracle and learning surface, not a production exchange. See [`docs/positioning.md`](docs/positioning.md) for the product boundary.

## Current Local Benchmark Snapshot

The sample below is a local paced-workload baseline, not a portable performance
or capacity claim. It was measured for 0.2.0 on July 10, 2026 with Python 3.10.5
on macOS 15.4.1 using:

```bash
tracebook-benchmark --scenario all --duration 1 --throughput 100 --seed 2026 --warmup-seconds 0.05 --output /tmp/tracebook-v020-baseline.json
```

| Scenario | New | New/s | Events/s | Mean ms | P95 ms | P99 ms | Generation ms | Event ms | Memory MB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `smoke` | 100 | 99.91 | 99.91 | 0.045 | 0.095 | 0.119 | 0.282 | 0.000 | 34.38 |
| `fifo_baseline` | 110 | 109.91 | 109.91 | 0.040 | 0.081 | 0.093 | 0.260 | 0.000 | 35.62 |
| `pro_rata_baseline` | 100 | 99.92 | 99.92 | 0.041 | 0.092 | 0.105 | 0.268 | 0.000 | 35.67 |
| `cancellation_mix` | 100 | 99.91 | 109.90 | 0.041 | 0.089 | 0.112 | 0.287 | 0.017 | 36.00 |
| `deep_book` | 100 | 99.96 | 99.96 | 0.051 | 0.120 | 0.200 | 0.381 | 0.000 | 36.23 |
| `high_cancellation` | 100 | 99.88 | 147.82 | 0.040 | 0.084 | 0.147 | 0.259 | 0.016 | 36.45 |
| `pro_rata_cancellation` | 100 | 99.97 | 115.97 | 0.052 | 0.125 | 0.233 | 0.431 | 0.021 | 36.69 |
| `multi_symbol` | 108 | 107.92 | 107.92 | 0.043 | 0.080 | 0.118 | 0.135 | 0.000 | 36.92 |

See [`docs/performance.md`](docs/performance.md) before adding or changing benchmark claims.

## Current Artifact Proof

All checks below were run during the latest production repo pass in this checkout.

| Proof surface | Verified result |
| --- | --- |
| Unit tests | `292` pytest tests passing with `80.86%` statement coverage and a `75%` gate |
| System smoke | `python test_system.py` passes all 6 checks |
| Format and lint | Black and Flake8 cover package, tests, examples, and smoke tooling with `0` issues |
| Type check | `python -m mypy src/tracebook` reports `0` issues |
| Compile and dependency checks | `python -m compileall -q src tests examples install_deps.py` and `python -m pip check` pass |
| Package build | sdist and wheel build successfully |
| Simulation CLI | deterministic FIFO smoke run completes |
| Benchmark CLI | smoke scenario writes JSON report |
| Dashboard loop | dashboard factory/import works, help completes, and local demo server returns `HTTP 200` on loopback |
| Remote CI | GitHub Actions covers Python 3.10 through 3.13 |

## What It Implements

| Component | What it does | Why it matters |
| --- | --- | --- |
| Differential campaigns | Generates versioned, stateful traces from a specified PRNG and minimizes the first drift | Finds lifecycle interactions that fixed cases miss while preserving exact, replayable evidence |
| External-engine conformance | Drives any stdio NDJSON adapter event by event against the reference engine | Tests Rust, C++, Java, Python, or other engines without embedding them in Tracebook |
| Semantic diffing | Compares outcomes, rejection codes, ordered trades, resting orders, and queue priority | Reports the exact first event and state path where behavior diverges |
| Failing-trace minimization | Uses deterministic delta debugging to remove irrelevant events and reports whether the result is one-minimal or budget-limited | Turns long failures into reviewable regression fixtures without overstating reduction completeness |
| Failure reproduction | Requires a saved reduced trace to produce the exact stored failure class and semantic diff | Makes known defects deterministic CI assets instead of informal bug reports |
| Semantic coverage | Reports reference-observed evidence for every capability declared by a campaign profile | Shows what a generated workload actually exercised without conflating it with source coverage |
| Standard conformance suite | Ships eight SHA-256-locked synthetic cases across FIFO, pro-rata, IOC/FOK, STP, tick, lifecycle, depth, and multi-symbol semantics | Gives engine authors a stable shared correctness corpus |
| FIFO matching | Matches resting orders by price-time priority | Provides the standard exchange-style baseline |
| Pro-rata matching | Allocates fills by resting size at a price level | Supports futures-style allocation experiments |
| Decimal quantities | Handles float quantities for crypto-style sizing | Avoids legacy integer-only simulator behavior |
| Order types | Supports limit, market, IOC, and FOK semantics | Covers common execution workflows |
| Lifecycle APIs | Cancels, priority-preserving quantity reductions, replaces, active-order lookup, and structured `OrderResult` submissions | Makes simulations and imported books inspectable |
| Self-trade prevention | Owner-tagged orders with `CANCEL_RESTING`/`CANCEL_INCOMING` policies | Stops a participant from matching its own resting liquidity |
| Historical event replay | Loads normalized CSV, JSON, and JSONL order events across symbols while preserving source ids through replacement | Connects real feed adapters to the validated matching path |
| Coinbase Exchange L3 adapter | Streams REST L3 snapshots plus recorded `full`/compact `level3` messages with sequence validation | Provides one concrete, auditable path from venue data to normalized events |
| Verified market-data corpora | Captures or prepares sanitized local sessions with SHA-256 manifests, canonical events, deterministic golden state, and comparable import benchmarks | Makes adapter correctness and performance independently reproducible |
| Detached public state | Returns copies from submission, lookup, trade, and callback APIs | Prevents callers from mutating live engine indexes |
| Event simulation | Interleaves `NEW`, `CANCEL`, and `REPLACE` events with deterministic seeds | Exercises more than one-way order ingestion |
| Synthetic streams | Generates random, trend, mean-reverting, momentum, passive, market-making, aggressive, and mixed flows | Enables repeatable workload variation |
| Performance monitor | Tracks throughput, latency, resources, generation time, event latency, and overhead | Separates signal from instrumentation cost |
| Benchmark runner | Runs fixed scenarios with warmup and machine metadata | Makes performance regression checks reproducible |
| Dashboard demo | Starts a Dash dashboard with optional background simulation | Gives a live review path without external services |
| Web frontend | Dependency-free static live order-book UI over a stdlib server | A clean live view with no Dash/JS-build dependencies |
| CI and packaging | Tests, lint, smoke runs, benchmark smoke, dashboard smoke, and wheel/sdist build | Keeps the repo usable for contributors |

## Architecture

```mermaid
flowchart LR
    G["Versioned campaign generator"] --> R["Conformance runner"]
    T["Normalized event trace"] --> R
    R --> O["Tracebook reference"]
    R --> A["stdio adapter"]
    A <--> E["External engine"]
    O --> D["Semantic diff"]
    A --> D
    D --> P["Versioned report"]
    D --> M["Deterministic minimizer"]
    M --> B["Failure bundle"]
    B --> X["Exact reproduction"]
    P --> J["JSON and JUnit"]
```

Core paths:

- `src/tracebook/core/order.py`: order, trade, side/type enums, and order factory.
- `src/tracebook/core/orderbook.py`: public book API, validation, lifecycle operations, callbacks, snapshots.
- `src/tracebook/core/matching_engine.py`: FIFO and pro-rata matching coordination.
- `src/tracebook/core/price_level.py`: price-level storage and depth snapshots.
- `src/tracebook/conformance/`: adapters, protocol, semantic diffing, minimization, and standard suite.
- `src/tracebook/events/`: normalized event replay and Coinbase Exchange L3 adaptation.
- `src/tracebook/corpus/`: safe local capture, corpus manifests, golden verification, and corpus benchmarks.
- `src/tracebook/simulation/`: synthetic order streams and event-based simulation engine.
- `src/tracebook/benchmarks/runner.py`: reproducible benchmark scenarios and JSON reports.
- `src/tracebook/profiling/`: performance monitor and magic-trace/fallback profiling.
- `src/tracebook/visualization/dashboard.py`: Dash dashboard and demo simulation entry point.

See [`docs/architecture.md`](docs/architecture.md) for the deeper component map.

## Quick Start

Install the published distribution (the import remains `tracebook`):

```bash
python -m pip install tracebook-sim
```

Contributor install:

```bash
git clone https://github.com/Taz33m/tracebook.git
cd tracebook
python -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev,dashboard]"
```

Run the bundled conformance suite through the example process adapter:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v1
tracebook-conformance suite \
  /tmp/tracebook-conformance-v1 \
  --output /tmp/conformance-suite.json \
  --candidate python examples/conformance_adapter.py
```

See [`docs/conformance.md`](docs/conformance.md) to adapt an external engine and
to read the campaign profiles, versioned protocol, hashing rules, report
schemas, and minimizer guarantees.

Run a minimal match:

```python
from tracebook import OrderBook, OrderSide

book = OrderBook("BTCUSD", matching_algorithm="fifo")

book.add_limit_order(OrderSide.BUY, price=50_000.0, quantity=1.0)
trades = book.add_limit_order(OrderSide.SELL, price=49_999.0, quantity=0.5)

for trade in trades:
    print(trade.quantity, trade.price)
```

Use structured result APIs when the caller needs lifecycle detail:

```python
from tracebook import OrderBook, OrderSide

book = OrderBook("BTCUSD")

result = book.submit_limit_order(OrderSide.BUY, price=49_950.0, quantity=0.25)

print(result.order.order_id)
print(result.rested)
print(result.cancelled)
print(result.rejected_reason)
```

## Order Lifecycle Example

```python
from tracebook import OrderBook, OrderSide

book = OrderBook("ETHUSD")

resting = book.submit_limit_order(OrderSide.BUY, price=3_000.0, quantity=2.0)
order_id = resting.order.order_id

print(book.get_active_order_ids())
print(book.get_order(order_id).remaining_quantity)

replacement = book.replace_order(order_id, price=3_001.0, quantity=1.5)
print(replacement.rested, replacement.rejected_reason)

cancelled = book.cancel_order(replacement.order.order_id)
print(cancelled)
```

## Simulation

Run a deterministic FIFO simulation with cancel and replace events:

```bash
tracebook-sim \
  --duration 5 \
  --throughput 500 \
  --algorithm FIFO \
  --seed 1337 \
  --cancel-ratio 0.05 \
  --replace-ratio 0.02 \
  --warmup-seconds 0.05 \
  --output benchmark_results/simulation.json
```

Run the pro-rata path:

```bash
tracebook-sim --duration 5 --throughput 500 --algorithm PRO_RATA --seed 1337
```

Enable magic-trace integration or fallback tracing:

```bash
tracebook-sim --duration 5 --throughput 500 --algorithm FIFO --magic-trace
```

## Self-Trade Prevention

Tag orders with an `owner` id and choose a policy so a participant never trades
against its own resting liquidity.

```python
from tracebook import OrderBook, OrderSide, SelfTradePolicy

book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)

book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)   # own resting bid
book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=2)   # someone else
trades = book.add_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)

# Owner 1's sell skips its own bid and fills against owner 2 instead.
print(book.get_statistics()["self_trades_prevented"])  # 1
```

Policies (`SelfTradePolicy`):

| Policy | Behavior |
| --- | --- |
| `NONE` | Default; self-trades are allowed |
| `CANCEL_RESTING` | Cancel the same-owner resting order; the incoming order continues |
| `CANCEL_INCOMING` | Cancel the incoming order's remainder on contact with a same-owner order |

Orders without an owner (the default `NO_OWNER`) are anonymous and never
prevented. Both policies keep the book uncrossed, a FOK is not reported fillable
by its own liquidity, and the chosen policy is captured in the replay log.

## Record And Replay

Record every accepted submission, successful cancellation, and full clear to a serializable event log, then
replay it against a fresh book to reconstruct the identical sequence of trades
and the identical final book state. This makes bug reproduction, regression
fixtures, and deterministic experiments trivial.

```python
from tracebook import OrderBook, OrderSide, EventLog, replay

book = OrderBook("BTCUSD")
log = book.start_recording()

book.add_limit_order(OrderSide.BUY, 50_000.0, 1.0)
book.add_limit_order(OrderSide.SELL, 49_999.0, 0.5)
book.stop_recording()

# Persist and restore across processes.
restored = EventLog.from_json(log.to_json())
rebuilt = replay(restored)

assert rebuilt.get_best_bid() == book.get_best_bid()
```

Matching does not depend on wall-clock time (execution price is the resting
price and FIFO priority is insertion order), so a fixed event log always
reproduces the same trades and resting book. Per-trade wall-clock timestamps are
metadata and are excluded from the determinism guarantee.

Recording must begin on a pristine book: the log captures only operations from
the `start_recording()` call onward, so `start_recording()` raises after any
prior activity. Call `clear()` before recording a previously used book.

## Historical Order-Event Replay

Replay normalized order-level data through the same validated books used by the
Python API. Input can be CSV, JSON, or JSONL and can contain multiple symbols.

```bash
tracebook-replay examples/data/sample_events.jsonl \
  --algorithm fifo \
  --self-trade-policy CANCEL_INCOMING \
  --include-trades \
  --output replay-summary.json
```

Strict mode fails at the first rejected event with its file-order index. Add
`--lenient` to collect rejections and continue. Source order ids remain
addressable after cancel-and-new replacement; optional trade output includes
both source and engine ids. See
[`docs/event-replay.md`](docs/event-replay.md) for the schema and adapter guidance.

### Coinbase Exchange L3

Normalize and replay the included Coinbase-style REST L3 snapshot and recorded
`full` feed without adding a network or authentication dependency:

```bash
tracebook-coinbase \
  examples/data/coinbase_btcusd_l3_snapshot.json \
  examples/data/coinbase_btcusd_full.jsonl \
  --tick-size 0.01 \
  --events-output /tmp/coinbase-events.jsonl \
  --include-trades \
  --output /tmp/coinbase-replay.json
```

The adapter enforces per-product sequence continuity, parses the compact
channel's announced schema, preserves maker reductions without resetting FIFO
priority, and keeps observed exchange trades separate from simulated trades.
See [`docs/coinbase-l3.md`](docs/coinbase-l3.md) for synchronization and
limitations.

### Verified Coinbase Corpora

The checked synthetic corpus binds sanitized source input, normalized events,
and complete final depth to one stable SHA-256 identity:

```bash
tracebook-corpus sample /tmp/tracebook-sample-corpus
tracebook-corpus verify /tmp/tracebook-sample-corpus

tracebook-corpus benchmark \
  /tmp/tracebook-sample-corpus \
  --iterations 10 \
  --warmups 2 \
  --output benchmark_results/corpus.json
```

Optional live capture subscribes before taking one REST L3 snapshot, sanitizes
before disk, and never accepts credentials. Coinbase's market-data terms may
restrict redistribution, so live manifests say `redistribution=not_granted`
and the root `corpora/` directory is ignored by Git. Install
`tracebook-sim[capture]` and review [`docs/corpora.md`](docs/corpora.md) before
capturing.

## Reproducible Paced Workloads

```bash
tracebook-benchmark \
  --scenario all \
  --seed 1337 \
  --warmup-seconds 0.05 \
  --output benchmark_results/local.json
```

Scenarios:

| Scenario | Purpose |
| --- | --- |
| `smoke` | Short CI-friendly FIFO run |
| `fifo_baseline` | FIFO matching baseline |
| `pro_rata_baseline` | Pro-rata matching baseline |
| `cancellation_mix` | FIFO run with cancel and replace events |
| `deep_book` | Higher-throughput FIFO run that builds deep resting liquidity |
| `high_cancellation` | FIFO run with a heavier cancel/replace mix |
| `pro_rata_cancellation` | Pro-rata run with cancel and replace events |
| `multi_symbol` | FIFO run across multiple symbols (independent books) |
| `all` | Runs every scenario above |

These scenarios drive a configured input rate and measure latency under that
load; they are not maximum-capacity claims. Benchmark JSON includes machine
metadata, dependency versions, scenario config, seed, warmup, achieved new-order
and event rates, matching latency percentiles, generation latency, lifecycle
event latency, memory, and monitoring overhead.

See [`docs/performance.md`](docs/performance.md) for local baseline guidance and sample measured results.

## Dashboard

```bash
tracebook-dashboard --port 8050 --demo-simulation --demo-throughput 200 --seed 1337
```

Open `http://localhost:8050` to inspect live throughput, latency, resource usage, trade volume, and book depth.
The dashboard binds to loopback by default; non-loopback hosts require `--allow-remote`
because the demo dashboard has no authentication.

Dashboard dependencies are optional:

```bash
python -m pip install "tracebook-sim[dashboard]"
```

## Live Web Frontend

A dependency-free live order-book frontend: a static HTML/CSS/JS page served by a
stdlib HTTP server, backed by a background simulation. No Dash, no build step, no
extras — it ships with the core package.

```bash
tracebook-web --port 8080 --throughput 500 --seed 1337
```

Open `http://localhost:8080` for a live depth ladder, top-of-book quote,
throughput and latency metrics, and a trade tape. The page polls `/api/state`
(a JSON snapshot) a couple of times a second. Like the dashboard it is
unauthenticated, so it binds to loopback by default and a non-loopback host
requires `--allow-remote`.

## Command Surface

| Command | Purpose |
| --- | --- |
| `tracebook-conformance sample suite/` | Copy the hash-locked synthetic conformance suite |
| `tracebook-conformance campaign --corpus-dir corpus/ --candidate-cmd ./adapter` | Generate stateful traces, report semantic coverage, and minimize the first drift |
| `tracebook-conformance suite suite/ --candidate ./adapter` | Test an external engine across every standard case |
| `tracebook-conformance run events.jsonl --candidate ./adapter` | Stop at the first semantic divergence in one trace |
| `tracebook-conformance minimize events.jsonl --events-output minimal.jsonl --candidate ./adapter` | Reduce a failure and report minimality or budget exhaustion |
| `tracebook-conformance reproduce corpus/failure-id/reduced.jsonl --candidate-cmd ./adapter` | Require the exact saved failure to replay deterministically |
| `tracebook-sim --duration 5 --throughput 500 --algorithm FIFO` | Run a FIFO simulation |
| `tracebook-sim --algorithm PRO_RATA --seed 1337` | Run the pro-rata path deterministically |
| `tracebook-sim --cancel-ratio 0.05 --replace-ratio 0.02` | Interleave lifecycle events |
| `tracebook-sim --output results.json` | Export simulation results |
| `tracebook-benchmark --scenario smoke` | Run the benchmark smoke scenario |
| `tracebook-benchmark --scenario all --output benchmark_results/local.json` | Produce a full local benchmark report |
| `tracebook-replay events.jsonl --output replay.json` | Replay normalized historical order events |
| `tracebook-coinbase snapshot.json full.jsonl --tick-size 0.01` | Normalize and replay Coinbase Exchange L3 data |
| `tracebook-corpus verify corpus/` | Verify hashes, events, and deterministic golden state |
| `tracebook-corpus benchmark corpus/ --output report.json` | Measure corpus import and replay phases |
| `tracebook-dashboard --demo-simulation` | Launch the Dash dashboard with live demo data |
| `tracebook-web --port 8080` | Serve the dependency-free live order-book frontend |
| `python -m pytest` | Run unit tests |
| `python test_system.py` | Run integration smoke checks |
| `python -m build --sdist --wheel --outdir dist` | Build package artifacts |

See [`docs/commands.md`](docs/commands.md) for CLI options and review workflows.

## Python API

```python
from tracebook import OrderBook, OrderBookManager, OrderSide

manager = OrderBookManager()
book = manager.create_order_book("BTCUSD", matching_algorithm="fifo")

book.submit_limit_order(OrderSide.BUY, 50_000.0, 1.0)
book.submit_limit_order(OrderSide.BUY, 49_950.0, 0.25)

result = book.submit_ioc_order(OrderSide.SELL, 49_900.0, 0.5)

print(len(result.trades))
print(book.get_best_bid())
print(book.get_best_ask())
print(book.get_order_book_depth(levels=3))
print(book.get_statistics())
```

Public top-level exports:

| Export | Purpose |
| --- | --- |
| `OrderBook` | Single-symbol order book |
| `OrderBookManager` | Multi-symbol book registry |
| `Order` | Explicit external order representation |
| `OrderFactory` | Explicit order construction |
| `OrderResult` | Structured submission result |
| `OrderSide` | `BUY` and `SELL` enum |
| `OrderType` | `MARKET`, `LIMIT`, `IOC`, `FOK` enum |
| `Trade` | Executed trade record |
| `SelfTradePolicy` | `NONE`, `CANCEL_RESTING`, `CANCEL_INCOMING` self-trade policy |
| `EventLog` | Serializable record of book operations for replay |
| `replay` | Reconstruct a book from a recorded `EventLog` |
| `MarketEvent` | Validated normalized historical order event |
| `MarketReplayResult` | Reconstructed books, source-id mapping, trades, and rejections |
| `ReplayTrade` | Trade record annotated with source and engine order ids |
| `load_market_events` | Load CSV, JSON, or JSONL event files |
| `replay_market_events` | Replay normalized events into per-symbol books |
| `ConformanceConfig` | Matching and numeric-normalization contract for a comparison |
| `CampaignProfile` | Versioned generated semantic surface and conformance config |
| `CampaignResult` | Multi-trace result with stable identity and optional first failure |
| `SemanticCoverage` | Reference-observed evidence for a campaign profile's declared capabilities |
| `ReproductionResult` | Expected-versus-observed result for a saved failure replay |
| `EngineAdapter` | Typed interface for pluggable in-process candidate engines |
| `ReferenceEngineAdapter` | Incremental adapter over Tracebook's reference semantics |
| `ExternalProcessAdapterFactory` | Fresh stdio candidate process for each run or minimization trial |
| `run_conformance` | Produce the first-divergence or conformant report for one trace |
| `run_campaign` | Generate and compare traces through the first minimized divergence |
| `minimize_failing_trace` | Delta-debug a divergent trace under a run budget |
| `run_reproduction` | Require a reduced trace to reproduce its stored semantic divergence |
| `write_campaign_artifacts` | Atomically persist a campaign report and optional failure bundle |
| `write_campaign_corpus` | Store a campaign under its deterministic campaign or failure ID |

## Outputs

| Output | Description |
| --- | --- |
| Simulation JSON | Raw simulation config, summary metrics, performance data, order book stats, stream stats, algorithm analysis |
| Benchmark JSON | Scenario summaries plus raw simulation results, machine metadata, dependency versions, warmup and seed |
| Event replay JSON | Config, applied/rejected counts, active source-id mapping, final depth, per-book statistics, and optional trades |
| Corpus manifest and golden JSON | Source rights, sanitization/capture metadata, file hashes, canonical event digest, sequence range, and complete final depth |
| Corpus benchmark/comparison JSON | Raw timing samples, machine/dependency metadata, corpus identity, phase summaries, and explicit environment differences |
| Conformance report JSON | Trace/config identity, engine metadata, compared event count, final state hash, and exact first divergence |
| Campaign JSON + failure corpus | Stable generator/profile identity, semantic coverage, per-trace hashes, original first-divergence evidence, and reduced JSONL reproducer |
| Minimization JSON + JSONL | Reduction statistics, target failure category, minimized trace hash, and executable reproducer |
| Reproduction JSON | Exact expected and observed failure classes, divergence paths, values, and full conformance report |
| Conformance JUnit | CI test-case projection for run, suite, minimization, campaign, and reproduction commands |
| Conformance suite JSON | Per-case fixture hashes, tags, and complete candidate reports |
| Dashboard charts | Throughput, latency, resources, trade volume, and depth |
| Performance docs | Local baseline samples and reporting rules |

Generated benchmark outputs and trace artifacts are ignored by git.

## Repository Layout

```text
src/tracebook/              package source
  core/                     orders, price levels, matching engine, order book API
  conformance/              adapters, campaigns, protocol, semantic diffing, minimizer, suite
  events/                   normalized file loading and historical event replay
  corpus/                   capture, manifests, bundled fixture, verification, benchmarks
  simulation/               synthetic order streams and event simulation
  benchmarks/               reproducible benchmark runner
  profiling/                performance monitor and tracing tools
  visualization/            Dash dashboard + static web frontend (web/)
tests/                      pytest correctness and integration coverage
integrations/               pinned optional adapters for independently built engines
docs/                       architecture, CI, release, command, and performance notes
examples/                   runnable scripts, CI templates, and source feed fixtures
.github/workflows/          CI across supported Python versions
setup.py                    package metadata, extras, console scripts
pyproject.toml              build-system and tool configuration
```

## Claims And Non-Claims

Claims:

- Runs external matching engines through a versioned, language-neutral stdio protocol and compares each event's observable semantics.
- Generates versioned stateful campaigns independently of candidate behavior and records stable identities and trace hashes.
- Localizes the first difference in outcome, rejection code, trade, resting state, or queue priority and can reduce the failing trace.
- Ships a synthetic, SHA-256-locked standard conformance suite with independently configurable matching policies.
- Implements FIFO and pro-rata matching paths for supported order types.
- Supports decimal order quantities.
- Validates symbols, sides, order types, prices, and quantities before matching.
- Supports atomic replacement, cancellation, detached active-order lookup, and coherent state snapshots.
- Replays normalized CSV, JSON, and JSONL order events across independent symbol books with stable source-id lifecycle mapping.
- Verifies hash-locked corpus inputs by reproducing canonical events and golden final book state exactly.
- Runs deterministic synthetic simulations with new, cancel, and replace events.
- Reports benchmark output with warmup, seed, machine metadata, generation timing, matching latency, event latency, memory, and monitoring overhead.
- Provides a dashboard demo path without requiring external market connectivity.

Non-claims:

- Not a production exchange matching engine.
- Not a trading venue, broker, production feed handler, or market-data vendor.
- Not a grant of rights to redistribute captured exchange market data.
- Not investment advice.
- Not a guarantee of live low-latency performance.
- Not a full fixed-point implementation yet; prices snap to an integer tick grid but quantities remain float64.
- Not a complete market microstructure research platform.
- Not exchange certification or proof of thread safety, durability, risk controls, networking behavior, or adapter honesty.
- Not proof that a listed benchmark number will reproduce on another machine.

## Limitations

- Alpha software; APIs may still evolve before a stable v1 release.
- Current storage uses plain Python dicts and lists (orders per level are an insertion-ordered dict; price levels are a bisect-indexed list), not a final low-latency memory layout.
- Prices snap to a configurable integer tick grid (`OrderBook(symbol, tick_size=...)`, default `0.01`); quantities remain float64 and full fixed-point accounting is a later performance phase.
- The normalized replay contract is venue-neutral; exchange sequence checks and feed-specific semantics belong in adapters.
- Protocol version 1 compares quantities after explicit decimal normalization; engines requiring different fixed-point rules must configure and document that boundary.
- Secure campaign bundle publication requires descriptor-relative directory operations; campaign generation and comparison remain available without them.
- Live Coinbase corpora are local artifacts. Pseudonymization removes unnecessary identifiers but does not alter Coinbase's market-data terms.
- Dashboard is a local demo and monitoring surface, not a secured production service.
- Magic-trace is optional and platform-dependent; fallback profiling is available when magic-trace is not installed.
- Benchmark results are local artifacts and should always cite the command, seed, machine, Python version, and dependency versions.

## Roadmap

- Validate the protocol with an independently implemented Rust, C++, or Java adapter.
- Add state-machine-aware property generation for lifecycle and self-trade-prevention traces.
- Separate adapter/protocol overhead from candidate engine timing in a dedicated benchmark mode.
- Stabilize protocol and artifact schemas after external-engine feedback.

## Open Source Project Health

- Contribution guide: [`CONTRIBUTING.md`](CONTRIBUTING.md)
- Security policy: [`SECURITY.md`](SECURITY.md)
- Code of conduct: [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md)
- Support guide: [`SUPPORT.md`](SUPPORT.md)
- Changelog: [`CHANGELOG.md`](CHANGELOG.md)
- Project plan: [`PROJECT_PLAN.md`](PROJECT_PLAN.md)

Pull requests should include tests and should not add benchmark claims without a reproducible command and machine context.

## License

MIT License. See [`LICENSE`](LICENSE).
