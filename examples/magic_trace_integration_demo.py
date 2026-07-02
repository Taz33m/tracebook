#!/usr/bin/env python3
"""
Magic-Trace Integration Demo

Demonstrates the enhanced magic-trace profiling capabilities including:
- High-resolution nanosecond-level tracing
- Function-level performance analysis
- Interactive visualization
- Comprehensive performance insights
"""

import os
import time
import json
from pathlib import Path

from tracebook.core.order import OrderFactory, OrderSide
from tracebook.core.orderbook import OrderBook
from tracebook.profiling.magic_trace_wrapper import MagicTraceProfiler, MagicTraceConfig
from tracebook.profiling.trace_analyzer import (
    NumpyJSONEncoder,
    get_tracer,
    profile_function,
)
from tracebook.profiling.trace_visualizer import TraceVisualizer
from tracebook.simulation.simulation_engine import SimulationEngine, SimulationConfig


def demo_basic_profiling():
    """Demonstrate basic high-resolution profiling."""
    print("🔍 DEMO: Basic High-Resolution Profiling")
    print("=" * 60)

    tracer = get_tracer()
    tracer.start_session("basic_demo")

    @profile_function("fibonacci")
    def fibonacci(n):
        if n <= 1:
            return n
        return fibonacci(n - 1) + fibonacci(n - 2)

    @profile_function("matrix_multiply")
    def matrix_multiply():
        import numpy as np

        a = np.random.rand(100, 100)
        b = np.random.rand(100, 100)
        return np.dot(a, b)

    @profile_function("string_operations")
    def string_operations():
        result = ""
        for i in range(1000):
            result += f"item_{i}_"
        return result

    print("Executing traced functions...")

    # Execute functions with different performance characteristics
    fib_result = fibonacci(10)
    matrix_multiply()
    string_operations()

    # Stop profiling and analyze
    analysis = tracer.stop_session("basic_demo")

    print(f"✓ Fibonacci(10) = {fib_result}")
    print("✓ Matrix multiplication completed")
    print("✓ String operations completed")

    # Display results
    summary = analysis["summary"]
    print("\n📊 PROFILING RESULTS:")
    print(f"  Total function calls: {summary['total_function_calls']}")
    print(f"  Unique functions: {summary['unique_functions']}")
    print(f"  Total traced time: {summary['total_traced_time_ms']:.2f}ms")
    print(f"  Trace overhead: {summary['trace_overhead_percentage']:.2f}%")

    # Show top functions
    function_analysis = analysis["function_analysis"]
    sorted_functions = sorted(
        function_analysis.items(), key=lambda x: x[1]["total_time_ms"], reverse=True
    )

    print("\n🏆 TOP FUNCTIONS BY TOTAL TIME:")
    for i, (func_name, stats) in enumerate(sorted_functions[:5]):
        print(
            f"  {i+1}. {func_name}: {stats['total_time_ms']:.2f}ms "
            f"({stats['call_count']} calls)"
        )

    # Export trace data
    os.makedirs("demo_traces", exist_ok=True)
    export_file = "demo_traces/basic_demo_trace.json"

    with open(export_file, "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2, cls=NumpyJSONEncoder)

    print(f"\n💾 Trace data exported to: {export_file}")
    return export_file


def demo_order_book_profiling():
    """Demonstrate order book profiling with magic-trace integration."""
    print("\n🏦 DEMO: Order Book Profiling")
    print("=" * 60)

    # Configure magic-trace profiler
    config = MagicTraceConfig()
    config.profile_functions = ["add_order", "_execute_fifo_match", "_execute_matches_at_level"]

    profiler = MagicTraceProfiler(config)

    with profiler.profile_session("orderbook_demo"):
        print("📈 Processing high-frequency order book operations...")

        # Create order book
        order_book = OrderBook("DEMOUSD", "FIFO")
        factory = OrderFactory()

        # Add profiling to key methods
        original_add_order = order_book.add_order

        @profile_function("OrderBook.add_order")
        def traced_add_order(*args, **kwargs):
            return original_add_order(*args, **kwargs)

        order_book.add_order = traced_add_order

        # Simulate high-frequency trading
        start_time = time.time()
        total_trades = 0

        for i in range(500):  # 500 orders
            # Alternate between buy and sell orders
            if i % 2 == 0:
                order = factory.create_limit_order("DEMOUSD", OrderSide.BUY, 100.0 + (i % 10), 100)
            else:
                order = factory.create_limit_order("DEMOUSD", OrderSide.SELL, 100.0 + (i % 10), 100)

            trades = order_book.add_order(order)
            total_trades += len(trades) if trades else 0

        processing_time = time.time() - start_time
        throughput = 500 / processing_time

        print(f"✓ Processed 500 orders in {processing_time:.3f}s")
        print(f"✓ Throughput: {throughput:.1f} orders/sec")
        print(f"✓ Total trades executed: {total_trades}")

        # Get final statistics
        stats = order_book.get_statistics()
        print(f"✓ Order book stats: {stats}")

    print("✓ Order book profiling completed")


def demo_simulation_profiling():
    """Demonstrate full simulation profiling."""
    print("\n🎯 DEMO: Full Simulation Profiling")
    print("=" * 60)

    # Configure simulation
    config = SimulationConfig()
    config.duration_seconds = 3.0
    config.target_throughput = 200.0
    config.symbols = ["DEMO1", "DEMO2"]
    config.matching_algorithm = "FIFO"
    config.enable_profiling = True
    config.enable_magic_trace = True

    print("🚀 Running simulation:")
    print(f"  Duration: {config.duration_seconds}s")
    print(f"  Target throughput: {config.target_throughput} orders/sec")
    print(f"  Symbols: {config.symbols}")
    print(f"  Algorithm: {config.matching_algorithm}")

    # Run simulation
    engine = SimulationEngine(config)
    results = engine.run_simulation()

    # Display key metrics
    summary = results["summary_metrics"]
    print("\n📊 SIMULATION RESULTS:")
    print(f"  Orders processed: {summary['total_orders_processed']:,}")
    print(f"  Trades executed: {summary['total_trades_executed']:,}")
    print(f"  Actual throughput: {summary['actual_throughput']:.1f} orders/sec")
    print(f"  Trade ratio: {summary['trade_ratio']:.1%}")

    # Performance analysis
    if "performance_data" in results:
        perf_data = results["performance_data"]
        if "performance_metrics" in perf_data:
            metrics = perf_data["performance_metrics"]
            if "order_processing_latency_ms" in metrics:
                latency = metrics["order_processing_latency_ms"]
                print(f"  Latency - Mean: {latency.get('mean', 0):.3f}ms")
                print(f"  Latency - P95: {latency.get('p95', 0):.3f}ms")
                print(f"  Latency - P99: {latency.get('p99', 0):.3f}ms")

    return results


def demo_trace_visualization():
    """Demonstrate trace visualization capabilities."""
    print("\n📊 DEMO: Trace Visualization")
    print("=" * 60)

    # Find available trace files
    trace_files = []
    for trace_dir in ["demo_traces", "traces"]:
        if Path(trace_dir).exists():
            trace_files.extend(Path(trace_dir).glob("*.json"))

    if not trace_files:
        print("❌ No trace files found for visualization")
        return

    # Use the first available trace file
    trace_file = trace_files[0]
    print(f"📁 Visualizing trace file: {trace_file}")

    # Create visualizer
    visualizer = TraceVisualizer()
    if not visualizer.load_trace_data(str(trace_file)):
        print("❌ Failed to load trace data")
        return

    # Generate insights report
    insights = visualizer.create_insights_report()
    print("\n" + insights)

    # Generate HTML report
    output_dir = "demo_visualizations"
    os.makedirs(output_dir, exist_ok=True)

    html_file = Path(output_dir) / "trace_analysis_report.html"
    if visualizer.generate_html_report(str(html_file)):
        print(f"\n📈 Interactive HTML report generated: {html_file}")
        print("   Open this file in a web browser to view interactive charts")

    return str(html_file)


def main():
    """Run complete magic-trace integration demo."""
    print("🎭 MAGIC-TRACE INTEGRATION DEMO")
    print("=" * 80)
    print("Demonstrating enhanced profiling capabilities for the")
    print("high-performance order book simulator")
    print("=" * 80)

    demos = [
        ("Basic High-Resolution Profiling", demo_basic_profiling),
        ("Order Book Profiling", demo_order_book_profiling),
        ("Full Simulation Profiling", demo_simulation_profiling),
        ("Trace Visualization", demo_trace_visualization),
    ]

    results = {}

    for demo_name, demo_func in demos:
        try:
            print(f"\n🎬 Starting: {demo_name}")
            result = demo_func()
            results[demo_name] = result
            print(f"✅ Completed: {demo_name}")
        except Exception as e:
            print(f"❌ Failed: {demo_name} - {e}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print("🎉 DEMO SUMMARY")
    print("=" * 80)

    print("✅ Enhanced magic-trace integration completed successfully!")
    print("\nKey Features Demonstrated:")
    print("  🔍 Nanosecond-level function tracing")
    print("  📊 Comprehensive performance analysis")
    print("  🎯 Order book and simulation profiling")
    print("  📈 Interactive visualization and reporting")
    print("  🚀 Fallback profiling for non-Linux environments")

    print("\nGenerated Outputs:")
    for demo_name, result in results.items():
        if result:
            print(f"  📁 {demo_name}: {result}")

    print("\n🏁 Magic-trace integration demo completed!")
    print("The system now provides production-ready profiling capabilities")
    print("with detailed performance insights and visualization tools.")


if __name__ == "__main__":
    main()
