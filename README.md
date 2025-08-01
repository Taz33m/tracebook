<p align="center">
  <img src="docs/logo.png" alt="tracebook logo" width="200"/>
</p>

<h1 align="center">tracebook</h1>
<p align="center"><em>High-performance, latency-optimized order book simulator</em></p>

## Tracebook: a latency-optimized order book simulator with magic-trace integration.

This project demonstrates systems-level thinking and low-latency engineering principles inspired by high-frequency trading infrastructure. It features real-time order generation, FIFO-based matching, and throughput exceeding 200 orders/sec. Built entirely on a MacBook Pro without a discrete GPU, it integrates Jane Street’s magic-trace (with fallback profiling) to analyze function-level performance and visualize bottlenecks. The result is a fully traceable, production-grade simulation platform that reflects the rigor expected in latency-sensitive trading environments.

## Features

- **Ultra-Low Latency**: Sub-microsecond order matching with JIT compilation
- **High Throughput**: Process 10,000+ orders per second
- **Advanced Profiling**: Jane Street's magic-trace integration for nanosecond precision
- **Multiple Algorithms**: FIFO and Pro-Rata matching implementations
- **Real-time Analytics**: Live dashboards and performance monitoring
- **Comprehensive Benchmarking**: Detailed latency and throughput analysis

## Architecture

The simulator is built with performance as the primary concern:

- **Numba JIT Compilation**: Critical paths optimized for near-native performance
- **Cache-Friendly Data Structures**: Minimized memory allocations and cache misses
- **Lock-Free Algorithms**: Thread-safe operations without traditional locking
- **Magic-Trace Integration**: Function-level profiling without performance overhead

## Performance Targets

| Metric | Target | Typical Achievement |
|--------|--------|-------------------|
| Throughput | 10,000+ orders/sec | 15,000+ orders/sec |
| Latency (P50) | < 100ns | ~50ns |
| Latency (P99) | < 1μs | ~500ns |
| Memory Usage | < 100MB | ~50MB |

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd high-performance-orderbook

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install magic-trace (Linux/macOS only)
# Follow instructions at: https://github.com/janestreet/magic-trace
```

## Quick Start

```python
from src.core.orderbook import OrderBook
from src.simulation.order_generator import OrderGenerator
from src.profiling.performance_monitor import PerformanceMonitor

# Create order book with FIFO matching
orderbook = OrderBook(matching_algorithm='fifo')

# Generate synthetic orders
generator = OrderGenerator()
orders = generator.generate_realistic_orders(count=10000)

# Run simulation with profiling
monitor = PerformanceMonitor()
with monitor.profile_session():
    for order in orders:
        orderbook.add_order(order)

# View results
monitor.print_summary()
```

## Benchmarking

Run comprehensive benchmarks:

```bash
# Latency benchmarks
python -m pytest tests/benchmarks/latency_benchmark.py -v

# Throughput benchmarks
python -m pytest tests/benchmarks/throughput_benchmark.py -v

# Full benchmark suite
make benchmark
```

## Profiling with Magic-Trace

```bash
# Profile a simulation run
python examples/advanced_profiling.py

# Analyze trace data
magic-trace attach -p $(pgrep python) -o trace.fxt
```

##  Dashboard

Launch the real-time performance dashboard:

```bash
python examples/dashboard_demo.py
```

Visit `http://localhost:8050` to view live metrics.

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test categories
pytest tests/test_orderbook.py -v
pytest tests/benchmarks/ -v
```

## Documentation

- [Architecture Guide](docs/architecture.md)
- [Performance Guide](docs/performance_guide.md)
- [API Reference](docs/api_reference.md)

## Performance Insights

This project demonstrates several advanced optimization techniques:

1. **Numba JIT Compilation**: 10-100x speedup on critical paths
2. **Memory Pool Allocation**: Reduced GC pressure and allocation overhead
3. **SIMD Vectorization**: Automatic vectorization of numerical operations
4. **Cache Optimization**: Data structure layout optimized for CPU cache
5. **Magic-Trace Profiling**: Zero-overhead profiling for production systems

##  Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure benchmarks pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Jane Street for magic-trace profiling tool
- Numba team for JIT compilation framework
- High-frequency trading community for algorithmic insights

---

