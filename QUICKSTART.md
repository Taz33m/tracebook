# Quick Start

## Setup

Use Python 3.10 or 3.11.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

For CI-style development installs, use:

```bash
pip install -e ".[dev,dashboard]"
```

## Basic Order Book Usage

```python
from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook

order_book = OrderBook("BTCUSD")

order_book.add_limit_order(OrderSide.BUY, 50000.0, 1.0)
trades = order_book.add_limit_order(OrderSide.SELL, 49999.0, 0.5)

print(f"Executed {len(trades)} trades")
print(order_book.get_statistics())
```

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
