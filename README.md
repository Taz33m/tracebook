<p align="center">
  <img src="docs/logo.png" alt="tracebook logo" width="200"/>
</p>

<h1 align="center">tracebook</h1>
<p align="center"><em>Latency-focused order book simulation with profiling hooks</em></p>

## Overview

`tracebook` is a Python order book simulator for experimenting with matching semantics, synthetic order flow, and profiling. It currently includes FIFO and pro-rata matching, decimal order quantities, IOC/FOK/market-order handling, synthetic order streams, a performance monitor, magic-trace integration with fallback tracing, and a Dash-based dashboard.

The project is still alpha software. The core matching behavior is covered by pytest tests, while performance claims are published only as reproducible local benchmark results.

## Features

- FIFO and pro-rata matching algorithms
- Decimal quantities for crypto-style order sizes
- Limit, market, IOC, and FOK order semantics
- Synthetic order generation with random, trend, mean-reverting, momentum, passive, market-making, aggressive, and mixed flows
- Event-based simulations with new, cancel, and replace events
- Structured order submission results for richer demos and benchmarks
- Performance metrics collection with optional magic-trace/fallback profiling
- Reproducible benchmark runner with warmup and JSON output
- Interactive dashboard entry point with a self-contained demo mode

## Installation

Python 3.10 or 3.11 is recommended; those are the versions exercised in CI.

```bash
git clone https://github.com/Taz33m/tracebook.git
cd tracebook

python -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -e .
```

For a smaller core install, use `pip install -e .`; dashboard and analysis dependencies are available through extras such as `pip install -e ".[dashboard]"` and `pip install -e ".[dev,dashboard]"`.

## Quick Start

```python
from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook

orderbook = OrderBook("BTCUSD", matching_algorithm="fifo")

orderbook.add_limit_order(OrderSide.BUY, price=50000.0, quantity=1.0)
trades = orderbook.add_limit_order(OrderSide.SELL, price=49999.0, quantity=0.5)

for trade in trades:
    print(f"{trade.quantity}@{trade.price}")
```

For a structured result object:

```python
result = orderbook.submit_limit_order(OrderSide.BUY, price=49950.0, quantity=0.25)
print(result.order.order_id, result.rested, result.cancelled, result.rejected_reason)
```

## Run A Simulation

```bash
python -m tracebook.simulation.simulation_engine \
  --duration 5 \
  --throughput 500 \
  --algorithm FIFO \
  --seed 1337 \
  --cancel-ratio 0.05 \
  --replace-ratio 0.02
```

Enable magic-trace/fallback tracing for a run:

```bash
python -m tracebook.simulation.simulation_engine \
  --duration 5 \
  --throughput 500 \
  --algorithm FIFO \
  --magic-trace
```

## Dashboard

```bash
tracebook-dashboard --port 8050 --demo-simulation
```

Visit `http://localhost:8050`.

## Reproducible Benchmarks

```bash
tracebook-benchmark \
  --scenario smoke \
  --seed 1337 \
  --warmup-seconds 0.05 \
  --output benchmark_results/smoke.json
```

Available scenarios are `smoke`, `fifo_baseline`, `pro_rata_baseline`, `cancellation_mix`, and `all`. See `docs/performance.md` for the benchmark report format and how to publish measured local baselines.

## Testing

```bash
pytest
python test_system.py
```

The old placeholder benchmark filenames are no longer advertised; benchmark-style runs use `tracebook-benchmark` or the simulation module above.

## Project Layout

```text
src/tracebook/
  core/           # Orders, price levels, order book, matching engine
  algorithms/     # FIFO and pro-rata analysis helpers
  profiling/      # Performance monitor, tracing, visualization
  simulation/     # Synthetic order streams and simulation runner
  visualization/  # Dash dashboard
tests/            # Behavioral pytest coverage
examples/         # Interactive demo script
```

## License

MIT License - see [LICENSE](LICENSE).
