#!/usr/bin/env python3
"""
Advanced profiling test for magic-trace integration.

Tests the enhanced profiling capabilities including high-resolution
tracing, function-level analysis, and performance insights.
"""

import sys
import os
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from profiling.magic_trace_wrapper import MagicTraceProfiler, MagicTraceConfig
from profiling.trace_analyzer import get_tracer, profile_function, NumpyJSONEncoder
from core.orderbook import OrderBook, OrderBookManager
from core.order import OrderFactory, OrderSide, OrderType
from simulation.order_generator import OrderGenerator
from simulation.simulation_engine import SimulationEngine, SimulationConfig


def test_basic_profiling():
    """Test basic profiling functionality."""
    print("=" * 60)
    print("Testing Basic Profiling")
    print("=" * 60)
    
    tracer = get_tracer()
    
    # Start profiling session
    success = tracer.start_session("basic_test")
    if not success:
        print("❌ Failed to start profiling session")
        return False
    
    print("✓ Started profiling session")
    
    # Create some traced functions
    @profile_function("test_function_1")
    def test_function_1():
        time.sleep(0.001)  # 1ms
        return "result1"
    
    @profile_function("test_function_2")
    def test_function_2():
        time.sleep(0.002)  # 2ms
        test_function_1()
        return "result2"
    
    @profile_function("test_function_3")
    def test_function_3():
        for i in range(5):
            test_function_1()
        test_function_2()
        return "result3"
    
    # Execute traced functions
    print("Executing traced functions...")
    test_function_3()
    test_function_2()
    test_function_1()
    
    # Stop profiling and get analysis
    analysis = tracer.stop_session("basic_test")
    if not analysis:
        print("❌ Failed to get analysis")
        return False
    
    print("✓ Profiling session completed")
    
    # Display results
    summary = analysis.get('summary', {})
    print(f"✓ Total function calls: {summary.get('total_function_calls', 0)}")
    print(f"✓ Unique functions: {summary.get('unique_functions', 0)}")
    print(f"✓ Total traced time: {summary.get('total_traced_time_ms', 0):.2f}ms")
    print(f"✓ Trace overhead: {summary.get('trace_overhead_percentage', 0):.2f}%")
    
    # Show function analysis
    function_analysis = analysis.get('function_analysis', {})
    print("\nFunction Performance:")
    for func_name, stats in function_analysis.items():
        print(f"  {func_name}:")
        print(f"    Calls: {stats['call_count']}")
        print(f"    Avg: {stats['mean_duration_ms']:.3f}ms")
        print(f"    Total: {stats['total_time_ms']:.3f}ms")
    
    # Show insights
    insights = analysis.get('performance_insights', [])
    if insights:
        print("\nPerformance Insights:")
        for insight in insights:
            print(f"  • {insight}")
    
    return True


def test_order_book_profiling():
    """Test profiling of order book operations."""
    print("\n" + "=" * 60)
    print("Testing Order Book Profiling")
    print("=" * 60)
    
    # Configure magic-trace profiler
    config = MagicTraceConfig()
    config.profile_functions = [
        "add_order",
        "match_orders_fifo",
        "execute_matches_at_level",
        "get_statistics"
    ]
    
    profiler = MagicTraceProfiler(config)
    
    # Start profiling session
    with profiler.profile_session("orderbook_test") as session:
        print("✓ Started order book profiling session")
        
        # Create order book and factory
        order_book = OrderBook("BTCUSD", "FIFO")
        factory = OrderFactory()
        
        # Add profiling decorators to key methods
        original_add_order = order_book.add_order
        
        @profile_function("OrderBook.add_order")
        def traced_add_order(*args, **kwargs):
            return original_add_order(*args, **kwargs)
        
        order_book.add_order = traced_add_order
        
        # Generate and process orders
        print("Processing orders...")
        start_time = time.time()
        
        for i in range(100):
            # Create buy order
            buy_order = factory.create_limit_order(
                "BTCUSD", OrderSide.BUY, 50000.0 + i, 100
            )
            order_book.add_order(buy_order)
            
            # Create sell order
            sell_order = factory.create_limit_order(
                "BTCUSD", OrderSide.SELL, 50001.0 - i, 100
            )
            trades = order_book.add_order(sell_order)
            
            if i % 20 == 0:
                stats = order_book.get_statistics()
        
        processing_time = time.time() - start_time
        print(f"✓ Processed 200 orders in {processing_time:.3f}s")
        
        # Get final statistics
        final_stats = order_book.get_statistics()
        print(f"✓ Final stats: {final_stats.get('orders_added', 0)} orders, "
              f"{final_stats.get('trades_executed', 0)} trades")
    
    print("✓ Order book profiling completed")
    return True


def test_simulation_profiling():
    """Test profiling of full simulation."""
    print("\n" + "=" * 60)
    print("Testing Simulation Profiling")
    print("=" * 60)
    
    # Configure simulation
    config = SimulationConfig()
    config.duration_seconds = 2.0
    config.target_orders_per_second = 100.0
    config.symbols = ["TESTUSD"]
    config.matching_algorithm = "FIFO"
    
    # Configure profiling
    trace_config = MagicTraceConfig()
    trace_config.profile_functions.extend([
        "generate_order",
        "process_orders_batch",
        "run_simulation"
    ])
    
    profiler = MagicTraceProfiler(trace_config)
    
    try:
        # Start profiling session
        session = profiler.create_session("simulation_test")
        session.start()
        print("✓ Started simulation profiling session")
        
        # Import and run simulation
        from simulation.simulation_engine import run_benchmark_simulation
        
        print("Running benchmark simulation...")
        results = run_benchmark_simulation(
            duration=config.duration_seconds,
            throughput=config.target_orders_per_second,
            algorithm=config.matching_algorithm
        )
        
        print(f"✓ Simulation completed:")
        print(f"  Orders processed: {results.get('orders_processed', 0)}")
        print(f"  Trades executed: {results.get('trades_executed', 0)}")
        print(f"  Throughput: {results.get('throughput', 0):.1f} orders/sec")
        
        # Stop profiling
        session.stop()
        print("✓ Simulation profiling completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Simulation profiling failed: {e}")
        return False


def test_trace_export():
    """Test trace data export functionality."""
    print("\n" + "=" * 60)
    print("Testing Trace Export")
    print("=" * 60)
    
    tracer = get_tracer()
    
    # Start session and generate some data
    tracer.start_session("export_test")
    
    @profile_function("export_test_function")
    def export_test_function():
        time.sleep(0.001)
        return "test"
    
    # Generate multiple calls
    for i in range(10):
        export_test_function()
    
    # Stop and export
    analysis = tracer.stop_session("export_test")
    
    # Export to file
    export_file = "traces/export_test_trace.json"
    os.makedirs("traces", exist_ok=True)
    
    try:
        with open(export_file, 'w') as f:
            json.dump(analysis, f, indent=2, cls=NumpyJSONEncoder)
        
        print(f"✓ Trace data exported to: {export_file}")
        
        # Verify file exists and has content
        if Path(export_file).exists():
            file_size = Path(export_file).stat().st_size
            print(f"✓ Export file size: {file_size} bytes")
            return True
        else:
            print("❌ Export file not found")
            return False
            
    except Exception as e:
        print(f"❌ Export failed: {e}")
        return False


def main():
    """Run all profiling tests."""
    print("Advanced Profiling Test Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Profiling", test_basic_profiling),
        ("Order Book Profiling", test_order_book_profiling),
        ("Simulation Profiling", test_simulation_profiling),
        ("Trace Export", test_trace_export),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"✓ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All profiling tests passed!")
        return 0
    else:
        print("❌ Some profiling tests failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
