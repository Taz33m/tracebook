#!/usr/bin/env python3
"""
Quick system test to verify the order book simulator works correctly.
"""

import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_basic_functionality():
    """Test basic order book functionality."""
    print("Testing basic functionality...")
    
    try:
        # Test core imports
        from core.order import Order, OrderFactory, OrderSide, OrderType
        from core.orderbook import OrderBook
        print("✓ Core imports successful")
        
        # Create order factory and orders
        factory = OrderFactory()
        
        # Create some test orders
        buy_order = factory.create_order("BTCUSD", OrderSide.BUY, OrderType.LIMIT, 50000.0, 1.0)
        sell_order = factory.create_order("BTCUSD", OrderSide.SELL, OrderType.LIMIT, 49999.0, 0.5)
        
        print(f"✓ Created buy order: {buy_order.quantity}@{buy_order.price}")
        print(f"✓ Created sell order: {sell_order.quantity}@{sell_order.price}")
        
        # Create order book
        order_book = OrderBook("BTCUSD")
        print(f"✓ Created order book for {order_book.symbol}")
        
        # Add orders and check for trades
        trades1 = order_book.add_order(buy_order)
        print(f"✓ Added buy order, trades: {len(trades1)}")
        
        trades2 = order_book.add_order(sell_order)
        print(f"✓ Added sell order, trades: {len(trades2)}")
        
        if trades2:
            trade = trades2[0]
            print(f"✓ Trade executed: {trade.quantity}@{trade.price}")
        
        # Get statistics
        stats = order_book.get_statistics()
        print(f"✓ Order book stats: {stats['orders_added']} orders, {stats['trades_executed']} trades")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring."""
    print("\nTesting performance monitoring...")
    
    try:
        from profiling.performance_monitor import PerformanceMonitor
        
        monitor = PerformanceMonitor(enable_magic_trace=False)  # Disable magic-trace for testing
        monitor.start_monitoring()
        
        # Record some fake metrics
        monitor.record_order_processing(1000000, 1)  # 1ms processing time
        monitor.record_trade_execution(1, 50000.0)
        
        time.sleep(1)  # Let monitoring collect data
        
        summary = monitor.get_performance_summary()
        print(f"✓ Performance monitoring working, collected {len(summary)} metric categories")
        
        monitor.stop_monitoring()
        print("✓ Performance monitoring test passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_order_generation():
    """Test order generation."""
    print("\nTesting order generation...")
    
    try:
        from simulation.order_generator import create_order_stream, OrderPattern
        
        # Create a short order stream
        stream = create_order_stream(
            pattern=OrderPattern.RANDOM,
            throughput=100.0,  # 100 orders/sec
            duration=2.0,      # 2 seconds
            symbol="TESTUSD",
            initial_price=100.0
        )
        
        stream.start_stream()
        time.sleep(1)  # Let it generate some orders
        
        orders = stream.get_orders()
        stream.stop_stream()
        
        print(f"✓ Generated {len(orders)} orders")
        
        if orders:
            print(f"✓ Sample order: {orders[0].symbol} {orders[0].side} {orders[0].quantity}@{orders[0].price}")
        
        return True
        
    except Exception as e:
        print(f"✗ Order generation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simulation_engine():
    """Test simulation engine."""
    print("\nTesting simulation engine...")
    
    try:
        from simulation.simulation_engine import run_benchmark_simulation
        
        # Run a very short simulation
        results = run_benchmark_simulation(
            duration=3.0,      # 3 seconds
            throughput=200.0,  # 200 orders/sec
            algorithm="FIFO"
        )
        
        summary = results['summary_metrics']
        print(f"✓ Simulation completed:")
        print(f"  - Orders processed: {summary['total_orders_processed']}")
        print(f"  - Trades executed: {summary['total_trades_executed']}")
        print(f"  - Throughput: {summary['actual_throughput']:.1f} orders/sec")
        
        return True
        
    except Exception as e:
        print(f"✗ Simulation engine test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("High-Performance Order Book Simulator - System Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Performance Monitoring", test_performance_monitoring),
        ("Order Generation", test_order_generation),
        ("Simulation Engine", test_simulation_engine),
    ]
    
    passed = 0
    total = len(tests)
    
    for name, test_func in tests:
        print(f"\n{'='*20} {name} {'='*20}")
        if test_func():
            passed += 1
        print()
    
    print("=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
