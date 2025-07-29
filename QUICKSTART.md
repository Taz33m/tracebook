# Quick Start Guide

## High-Performance Order Book Simulator

This is a high-performance order book simulator built in Python with Numba JIT compilation for GPU-style parallelism on CPU.

### Quick Setup

1. **Install Dependencies**:
   ```bash
   python3 install_deps.py
   ```

2. **Test the System**:
   ```bash
   python3 test_system.py
   ```

3. **Run Full Demo**:
   ```bash
   python3 examples/full_simulation_demo.py
   ```

### Key Features

- **High Throughput**: Processes thousands of orders per second
- **JIT-Compiled Matching**: FIFO and Pro-Rata algorithms optimized with Numba
- **Magic-Trace Integration**: Nanosecond-precision profiling with Jane Street's magic-trace
- **Real-Time Dashboard**: Interactive performance visualization
- **Comprehensive Benchmarking**: Detailed performance analysis and optimization recommendations

### Quick Examples

#### Basic Order Book Usage
```python
from src.core.orderbook import OrderBook
from src.core.order import OrderFactory, OrderSide, OrderType

# Create order book
order_book = OrderBook("BTCUSD")

# Create orders
factory = OrderFactory()
buy_order = factory.create_order("BTCUSD", OrderSide.BUY, OrderType.LIMIT, 50000.0, 1.0)
sell_order = factory.create_order("BTCUSD", OrderSide.SELL, OrderType.LIMIT, 49999.0, 0.5)

# Add orders and get trades
trades = order_book.add_order(buy_order)
trades = order_book.add_order(sell_order)

print(f"Executed {len(trades)} trades")
```

#### Run Benchmark Simulation
```python
from src.simulation.simulation_engine import run_benchmark_simulation

results = run_benchmark_simulation(
    duration=60.0,      # 60 seconds
    throughput=1000.0,  # 1000 orders/sec
    algorithm="FIFO"    # or "PRO_RATA"
)
```

#### Start Real-Time Dashboard
```python
from src.visualization.dashboard import create_dashboard

dashboard = create_dashboard(port=8050)
dashboard.run()  # Visit http://localhost:8050
```

### Performance Targets

- **Latency**: Sub-millisecond order processing
- **Throughput**: 10,000+ orders/second
- **Memory**: Efficient memory usage with minimal allocations
- **Profiling Overhead**: <5% performance impact

### Architecture

```
src/
├── core/           # Core order book components
├── algorithms/     # Matching algorithms (FIFO, Pro-Rata)
├── profiling/      # Performance monitoring and magic-trace
├── simulation/     # Order generation and simulation engine
├── visualization/  # Real-time dashboards
└── utils/          # Utility functions
```

### Next Steps

1. **Explore Examples**: Check `examples/full_simulation_demo.py` for comprehensive demos
2. **Run Benchmarks**: Use the Makefile commands for standardized benchmarking
3. **Customize Algorithms**: Implement your own matching algorithms
4. **Add Magic-Trace**: Install Jane Street's magic-trace for advanced profiling

For detailed documentation, see `README.md` and the `docs/` directory.
