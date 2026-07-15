# Quick Start

For the native Rust `orderbook-rs` comparison, see
[`integrations/orderbook_rs/README.md`](integrations/orderbook_rs/README.md).
The narrower PythonMatchingEngine comparison remains in
[`integrations/python_matching_engine/README.md`](integrations/python_matching_engine/README.md).
For a pull-request gate, copy
[`examples/github-actions/conformance.yml`](examples/github-actions/conformance.yml).

## Setup

Use Python 3.10 through 3.13.

Install the published distribution:

```bash
python -m pip install tracebook-sim
```

For a contributor checkout:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,dashboard]"
```

For the full contributor environment, use:

```bash
make setup
```

## Basic Order Book Usage

```python
from tracebook import OrderBook, OrderSide

order_book = OrderBook("BTCUSD")

order_book.add_limit_order(OrderSide.BUY, 50000.0, 1.0)
trades = order_book.add_limit_order(OrderSide.SELL, 49999.0, 0.5)

print(f"Executed {len(trades)} trades")
print(order_book.get_statistics())
```

## Test A Matching Engine

The public package can materialize the standard synthetic suite in any clean
directory:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v2
```

From a contributor checkout, run that suite through the included example
process adapter:

```bash
tracebook-conformance suite \
  /tmp/tracebook-conformance-v2 \
  --output /tmp/conformance-suite.json \
  --candidate python examples/conformance_adapter.py
```

Replace the final command with an adapter around your engine. The adapter may
be written in any language and communicates over versioned NDJSON. For one
failing trace, use `tracebook-conformance minimize` to emit a smaller JSONL
reproducer. See `docs/conformance.md` for the full contract.

Use `submit_*` APIs when you need a structured result:

```python
result = order_book.submit_limit_order(OrderSide.BUY, 49950.0, 0.25)
print(result.order.order_id, result.rested, result.cancelled)
```

## Order Type Semantics

- Limit orders may rest when partially or fully unfilled.
- Market orders execute against available opposite-side liquidity and never rest.
- IOC orders execute immediately up to available liquidity and cancel the remainder.
- FOK orders execute only if the full quantity is immediately available.

## Run A Benchmark-Style Simulation

```bash
python -m tracebook.simulation.simulation_engine \
  --duration 10 \
  --throughput 500 \
  --algorithm FIFO \
  --seed 1337 \
  --cancel-ratio 0.05 \
  --replace-ratio 0.02
```

Use `--algorithm PRO_RATA` to compare the pro-rata path.

## Start The Dashboard

```bash
tracebook-dashboard --port 8050 --demo-simulation
```

Then open `http://localhost:8050`.

## Run Reproducible Benchmarks

```bash
tracebook-benchmark \
  --scenario smoke \
  --seed 1337 \
  --warmup-seconds 0.05 \
  --output benchmark_results/smoke.json
```

## Replay Order Events

```bash
tracebook-replay examples/data/sample_events.jsonl --include-trades --output replay-summary.json
```

The normalized schema accepts CSV, JSON, and JSONL and preserves source order ids
through replacement. See `docs/event-replay.md` for fields, self-trade policy,
and strict/lenient behavior.

## Replay Coinbase Exchange L3 Data

```bash
tracebook-coinbase \
  examples/data/coinbase_btcusd_l3_snapshot.json \
  examples/data/coinbase_btcusd_full.jsonl \
  --tick-size 0.01 \
  --events-output /tmp/coinbase-events.jsonl \
  --output /tmp/coinbase-replay.json
```

This offline adapter validates the snapshot/feed sequence and converts Coinbase
order IDs and lifecycle messages into normalized events. See
`docs/coinbase-l3.md` before using a captured dataset.

## Verify A Reproducible Corpus

```bash
tracebook-corpus sample /tmp/tracebook-sample-corpus
tracebook-corpus verify /tmp/tracebook-sample-corpus

tracebook-corpus benchmark \
  /tmp/tracebook-sample-corpus \
  --iterations 5 \
  --warmups 1
```

Install `tracebook-sim[capture]` only when live public capture is needed. Live
corpora remain local and are marked as not licensed for redistribution by
default. Read `docs/corpora.md` before accessing Coinbase market data.
