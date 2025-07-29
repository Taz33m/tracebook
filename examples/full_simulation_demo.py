#!/usr/bin/env python3
"""
Comprehensive demonstration of the high-performance order book simulator.

This example showcases:
- Order generation with multiple patterns
- High-throughput order processing
- Magic-trace profiling integration
- Real-time performance monitoring
- Interactive dashboard visualization
- Comprehensive benchmarking and analysis
"""

import sys
import os
import time
import threading
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from simulation.simulation_engine import SimulationEngine, SimulationConfig, run_benchmark_simulation
from simulation.order_generator import OrderPattern, create_order_stream
from visualization.dashboard import create_dashboard
from profiling.performance_monitor import PerformanceMonitor
from core.orderbook import OrderBookManager
from algorithms.fifo import FIFOOptimizer
from algorithms.pro_rata import ProRataOptimizer


def demo_basic_simulation():
    """Demonstrate basic simulation functionality."""
    print("\n" + "="*60)
    print("BASIC SIMULATION DEMO")
    print("="*60)
    
    # Run a quick benchmark
    results = run_benchmark_simulation(
        duration=10.0,
        throughput=500.0,
        algorithm="FIFO"
    )
    
    return results


def demo_advanced_simulation():
    """Demonstrate advanced simulation with custom configuration."""
    print("\n" + "="*60)
    print("ADVANCED SIMULATION DEMO")
    print("="*60)
    
    # Custom configuration
    config = SimulationConfig(
        duration_seconds=30.0,
        target_throughput=2000.0,
        order_pattern=OrderPattern.MIXED,
        matching_algorithm="PRO_RATA",
        enable_profiling=True,
        enable_magic_trace=True,
        batch_processing=True,
        batch_size=50,
        symbols=["BTCUSD", "ETHUSD"]
    )
    
    # Create and run simulation
    engine = SimulationEngine(config)
    
    # Register callbacks for real-time monitoring
    trade_count = 0
    order_count = 0
    
    def on_trade(trades):
        nonlocal trade_count
        trade_count += len(trades)
        if trade_count % 100 == 0:
            print(f"Trades executed: {trade_count}")
    
    def on_order(order):
        nonlocal order_count
        order_count += 1
        if order_count % 1000 == 0:
            print(f"Orders processed: {order_count}")
    
    engine.register_trade_callback(on_trade)
    engine.register_order_callback(on_order)
    
    # Run simulation
    results = engine.run_simulation()
    engine.print_results(results)
    
    # Export results
    filename = engine.export_results(results)
    print(f"Results saved to: {filename}")
    
    return results


def demo_algorithm_comparison():
    """Compare FIFO vs Pro-Rata algorithms."""
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON DEMO")
    print("="*60)
    
    algorithms = ["FIFO", "PRO_RATA"]
    results = {}
    
    for algorithm in algorithms:
        print(f"\nTesting {algorithm} algorithm...")
        
        result = run_benchmark_simulation(
            duration=15.0,
            throughput=1500.0,
            algorithm=algorithm
        )
        
        results[algorithm] = result
        
        # Brief pause between tests
        time.sleep(2)
    
    # Compare results
    print("\n" + "="*60)
    print("ALGORITHM COMPARISON RESULTS")
    print("="*60)
    
    for algorithm, result in results.items():
        summary = result['summary_metrics']
        perf_data = result['performance_data']
        
        print(f"\n{algorithm} Algorithm:")
        print(f"  Throughput: {summary['actual_throughput']:.1f} orders/sec")
        print(f"  Trade Ratio: {summary['trade_ratio']:.1%}")
        print(f"  Total Volume: {summary['total_volume']:,.2f}")
        
        # Get latency if available
        perf_metrics = perf_data.get('performance_metrics', {})
        if 'order_processing_latency_ms' in perf_metrics:
            latency = perf_metrics['order_processing_latency_ms']
            print(f"  Avg Latency: {latency.get('mean', 0):.3f}ms")
            print(f"  P99 Latency: {latency.get('p99', 0):.3f}ms")
    
    return results


def demo_order_patterns():
    """Demonstrate different order generation patterns."""
    print("\n" + "="*60)
    print("ORDER PATTERN DEMO")
    print("="*60)
    
    patterns = [
        OrderPattern.RANDOM,
        OrderPattern.MARKET_MAKING,
        OrderPattern.AGGRESSIVE,
        OrderPattern.MIXED
    ]
    
    for pattern in patterns:
        print(f"\nTesting {pattern.name} pattern...")
        
        # Create order stream
        stream = create_order_stream(
            pattern=pattern,
            throughput=1000.0,
            duration=5.0,
            symbol="TESTUSD",
            initial_price=100.0
        )
        
        # Generate some orders
        with stream:
            time.sleep(2.0)  # Let it generate orders
            
            orders = stream.get_orders()
            stats = stream.get_stream_stats()
            
            print(f"  Generated {len(orders)} orders")
            print(f"  Actual throughput: {stats['runtime_stats']['actual_throughput']:.1f} orders/sec")
            
            # Analyze order characteristics
            if orders:
                market_orders = sum(1 for o in orders if o.order_type == 0)  # MARKET
                buy_orders = sum(1 for o in orders if o.side == 0)  # BUY
                
                print(f"  Market orders: {market_orders/len(orders):.1%}")
                print(f"  Buy orders: {buy_orders/len(orders):.1%}")
                
                prices = [o.price for o in orders if o.price > 0]
                if prices:
                    print(f"  Price range: ${min(prices):.2f} - ${max(prices):.2f}")


def demo_performance_monitoring():
    """Demonstrate performance monitoring capabilities."""
    print("\n" + "="*60)
    print("PERFORMANCE MONITORING DEMO")
    print("="*60)
    
    # Create performance monitor
    with PerformanceMonitor(enable_magic_trace=True) as monitor:
        
        # Set up alerts
        def alert_callback(alert_type, alert_data):
            print(f"ALERT: {alert_type} - {alert_data['metric']}: {alert_data['value']:.2f}")
        
        monitor.register_alert_callback(alert_callback)
        monitor.set_alert_threshold('latency_p99_ms', 5.0)  # 5ms threshold
        
        # Run a simulation while monitoring
        print("Running simulation with performance monitoring...")
        
        config = SimulationConfig(
            duration_seconds=20.0,
            target_throughput=1000.0,
            matching_algorithm="FIFO",
            enable_profiling=True
        )
        
        engine = SimulationEngine(config)
        results = engine.run_simulation()
        
        # Print monitoring summary
        monitor.print_summary()
        
        # Export metrics
        metrics_file = monitor.export_metrics()
        print(f"Performance metrics exported to: {metrics_file}")
    
    return results


def demo_dashboard(duration: int = 60):
    """Demonstrate real-time dashboard (runs in background)."""
    print("\n" + "="*60)
    print("DASHBOARD DEMO")
    print("="*60)
    
    print(f"Starting dashboard demo for {duration} seconds...")
    print("Dashboard will be available at: http://localhost:8050")
    
    # Create dashboard
    dashboard = create_dashboard(port=8050, update_interval=1000)
    
    # Create simulation to generate data
    config = SimulationConfig(
        duration_seconds=duration,
        target_throughput=800.0,
        matching_algorithm="FIFO",
        enable_profiling=True
    )
    
    engine = SimulationEngine(config)
    dashboard.set_order_book_manager(engine.order_book_manager)
    
    # Run dashboard and simulation in parallel
    def run_simulation():
        time.sleep(2)  # Let dashboard start first
        print("Starting simulation...")
        results = engine.run_simulation()
        engine.print_results(results)
    
    # Start simulation in background
    sim_thread = threading.Thread(target=run_simulation, daemon=True)
    sim_thread.start()
    
    # Run dashboard (this will block)
    try:
        dashboard.run(debug=False, host='127.0.0.1')
    except KeyboardInterrupt:
        print("\nDashboard stopped by user")
    
    return None


def demo_optimization_recommendations():
    """Demonstrate algorithm optimization recommendations."""
    print("\n" + "="*60)
    print("OPTIMIZATION RECOMMENDATIONS DEMO")
    print("="*60)
    
    # Run simulation to get data
    results = run_benchmark_simulation(
        duration=15.0,
        throughput=1200.0,
        algorithm="FIFO"
    )
    
    # Get statistics for analysis
    stats = results['order_book_statistics']
    
    if stats:
        symbol = list(stats.keys())[0]
        symbol_stats = stats[symbol]
        
        print(f"\nAnalyzing performance for {symbol}...")
        
        # FIFO optimization
        fifo_optimizer = FIFOOptimizer()
        fifo_recommendations = fifo_optimizer.optimize_configuration(symbol_stats)
        
        print("\nFIFO Optimization Recommendations:")
        for rec in fifo_recommendations:
            print(f"  - {rec}")
        
        # Pro-Rata optimization
        pro_rata_optimizer = ProRataOptimizer()
        pro_rata_recommendations = pro_rata_optimizer.optimize_configuration(symbol_stats)
        
        print("\nPro-Rata Optimization Recommendations:")
        for rec in pro_rata_recommendations:
            print(f"  - {rec}")
    
    return results


def main():
    """Run all demonstrations."""
    print("High-Performance Order Book Simulator - Full Demo")
    print("=" * 80)
    
    demos = [
        ("Basic Simulation", demo_basic_simulation),
        ("Advanced Simulation", demo_advanced_simulation),
        ("Algorithm Comparison", demo_algorithm_comparison),
        ("Order Patterns", demo_order_patterns),
        ("Performance Monitoring", demo_performance_monitoring),
        ("Optimization Recommendations", demo_optimization_recommendations),
    ]
    
    # Ask user which demos to run
    print("\nAvailable demonstrations:")
    for i, (name, _) in enumerate(demos, 1):
        print(f"{i}. {name}")
    print(f"{len(demos) + 1}. Dashboard Demo (interactive)")
    print(f"{len(demos) + 2}. Run All (except dashboard)")
    
    try:
        choice = input(f"\nSelect demo (1-{len(demos) + 2}): ").strip()
        
        if choice == str(len(demos) + 1):
            # Dashboard demo
            duration = input("Dashboard duration in seconds (default 60): ").strip()
            duration = int(duration) if duration.isdigit() else 60
            demo_dashboard(duration)
            
        elif choice == str(len(demos) + 2):
            # Run all demos
            for name, demo_func in demos:
                print(f"\n{'='*20} {name} {'='*20}")
                try:
                    demo_func()
                    print(f"✓ {name} completed successfully")
                except Exception as e:
                    print(f"✗ {name} failed: {e}")
                
                # Brief pause between demos
                time.sleep(2)
        
        elif choice.isdigit() and 1 <= int(choice) <= len(demos):
            # Run specific demo
            idx = int(choice) - 1
            name, demo_func = demos[idx]
            
            print(f"\nRunning {name}...")
            demo_func()
            print(f"✓ {name} completed successfully")
        
        else:
            print("Invalid choice. Running basic simulation...")
            demo_basic_simulation()
    
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
    
    print("\nDemo completed. Thank you for trying the order book simulator!")


if __name__ == "__main__":
    main()
