"""
High-performance simulation engine for order book testing.

Coordinates order generation, order book processing, and performance monitoring
to provide comprehensive simulation capabilities.
"""

import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
import numpy as np

from core.orderbook import OrderBook, OrderBookManager
from core.order import Order, Trade, OrderSide, OrderType
from algorithms.fifo import FIFOAnalyzer
from algorithms.pro_rata import ProRataAnalyzer
from profiling.performance_monitor import PerformanceMonitor
from profiling.magic_trace_wrapper import MagicTraceProfiler
from simulation.order_generator import SyntheticOrderStream, MarketParameters, OrderGenerationConfig, OrderPattern


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    duration_seconds: float = 60.0
    target_throughput: float = 1000.0
    order_pattern: OrderPattern = OrderPattern.MIXED
    matching_algorithm: str = "FIFO"  # "FIFO" or "PRO_RATA"
    enable_profiling: bool = True
    enable_magic_trace: bool = True
    batch_processing: bool = True
    batch_size: int = 100
    symbols: List[str] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ["BTCUSD"]


class SimulationEngine:
    """
    High-performance simulation engine.
    
    Orchestrates order generation, order book processing, and performance
    monitoring to provide comprehensive testing and benchmarking capabilities.
    """
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        
        # Core components
        self.order_book_manager = OrderBookManager()
        self.performance_monitor = PerformanceMonitor(enable_magic_trace=config.enable_magic_trace)
        
        # Order streams per symbol
        self.order_streams = {}
        self._setup_order_streams()
        
        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.start_time = 0
        self.end_time = 0
        
        # Statistics
        self.total_orders_processed = 0
        self.total_trades_executed = 0
        self.total_volume = 0.0
        self.processing_times = []
        
        # Event callbacks
        self.trade_callbacks = []
        self.order_callbacks = []
        self.simulation_callbacks = []
    
    def _setup_order_streams(self):
        """Setup order streams for each symbol."""
        for symbol in self.config.symbols:
            market_params = MarketParameters(
                symbol=symbol,
                order_arrival_rate=self.config.target_throughput / len(self.config.symbols)
            )
            
            stream_config = OrderGenerationConfig(
                pattern=self.config.order_pattern,
                duration_seconds=self.config.duration_seconds,
                target_throughput=self.config.target_throughput / len(self.config.symbols),
                batch_size=self.config.batch_size
            )
            
            stream = SyntheticOrderStream(market_params, stream_config)
            self.order_streams[symbol] = stream
            
            # Setup order book for symbol
            order_book = OrderBook(
                symbol=symbol,
                matching_algorithm=self.config.matching_algorithm
            )
            
            # Register callbacks
            order_book.register_trade_callback(self._on_trade)
            order_book.register_order_callback(self._on_order_processed)
            
            self.order_book_manager.add_order_book(symbol, order_book)
    
    def register_trade_callback(self, callback: Callable):
        """Register callback for trade events."""
        self.trade_callbacks.append(callback)
    
    def register_order_callback(self, callback: Callable):
        """Register callback for order events."""
        self.order_callbacks.append(callback)
    
    def register_simulation_callback(self, callback: Callable):
        """Register callback for simulation events."""
        self.simulation_callbacks.append(callback)
    
    def run_simulation(self) -> Dict[str, Any]:
        """Run the complete simulation."""
        print(f"Starting simulation - Duration: {self.config.duration_seconds}s, "
              f"Target: {self.config.target_throughput} orders/sec")
        
        # Start monitoring
        self.performance_monitor.start_monitoring()
        
        # Start profiling session if enabled
        profiling_session = None
        if self.config.enable_magic_trace:
            profiling_session = self.performance_monitor.profile_session("full_simulation")
            profiling_session.__enter__()
        
        try:
            # Start order streams
            for stream in self.order_streams.values():
                stream.start_stream()
            
            # Run simulation
            self.start_time = time.time_ns()
            self.is_running = True
            
            # Main simulation loop
            self._simulation_loop()
            
            self.end_time = time.time_ns()
            self.is_running = False
            
        finally:
            # Stop order streams
            for stream in self.order_streams.values():
                stream.stop_stream()
            
            # Stop profiling
            if profiling_session:
                profiling_session.__exit__(None, None, None)
            
            # Stop monitoring
            self.performance_monitor.stop_monitoring()
        
        # Generate results
        results = self._generate_results()
        
        print(f"Simulation completed - Processed {self.total_orders_processed:,} orders, "
              f"Executed {self.total_trades_executed:,} trades")
        
        return results
    
    def _simulation_loop(self):
        """Main simulation processing loop."""
        end_time = time.time() + self.config.duration_seconds
        
        while time.time() < end_time and self.is_running:
            loop_start = time.time_ns()
            
            # Process orders from all streams
            for symbol, stream in self.order_streams.items():
                orders = stream.get_orders(self.config.batch_size)
                
                if orders:
                    order_book = self.order_book_manager.get_order_book(symbol)
                    
                    if self.config.batch_processing:
                        # Batch processing
                        processing_start = time.time_ns()
                        trades = order_book.process_orders_batch(orders)
                        processing_time = time.time_ns() - processing_start
                        
                        self.performance_monitor.record_order_processing(
                            processing_time, len(orders)
                        )
                        
                        if trades:
                            self.performance_monitor.record_trade_execution(
                                len(trades), sum(trade.quantity * trade.price for trade in trades)
                            )
                    else:
                        # Individual processing
                        for order in orders:
                            processing_start = time.time_ns()
                            trades = order_book.add_order(order)
                            processing_time = time.time_ns() - processing_start
                            
                            self.performance_monitor.record_order_processing(processing_time, 1)
                            
                            if trades:
                                self.performance_monitor.record_trade_execution(
                                    len(trades), sum(trade.quantity * trade.price for trade in trades)
                                )
            
            # Brief sleep to prevent CPU spinning
            loop_time = (time.time_ns() - loop_start) / 1_000_000  # ms
            if loop_time < 1.0:  # If loop took less than 1ms
                time.sleep(0.001)  # Sleep for 1ms
    
    def _on_trade(self, trades: List):
        """Handle trade events."""
        self.total_trades_executed += len(trades)
        self.total_volume += sum(trade.quantity * trade.price for trade in trades)
        
        for callback in self.trade_callbacks:
            try:
                callback(trades)
            except Exception as e:
                print(f"Error in trade callback: {e}")
    
    def _on_order_processed(self, order: Order, trades: List[Trade]):
        """Handle order processed events."""
        self.total_orders_processed += 1
        self.total_trades_executed += len(trades)
        
        for callback in self.order_callbacks:
            try:
                callback(order, trades)
            except Exception as e:
                print(f"Error in order callback: {e}")
    
    def _generate_results(self) -> Dict[str, Any]:
        """Generate comprehensive simulation results."""
        duration_seconds = (self.end_time - self.start_time) / 1_000_000_000
        
        # Performance summary
        performance_summary = self.performance_monitor.get_performance_summary()
        
        # Order book statistics
        order_book_stats = {}
        for symbol in self.config.symbols:
            order_book = self.order_book_manager.get_order_book(symbol)
            order_book_stats[symbol] = order_book.get_statistics()
        
        # Stream statistics
        stream_stats = {}
        for symbol, stream in self.order_streams.items():
            stream_stats[symbol] = stream.get_stream_stats()
        
        # Algorithm analysis
        algorithm_analysis = {}
        for symbol in self.config.symbols:
            order_book = self.order_book_manager.get_order_book(symbol)
            
            if self.config.matching_algorithm == "FIFO":
                analyzer = FIFOAnalyzer()
                algorithm_analysis[symbol] = analyzer.analyze_performance(
                    order_book.get_statistics(),
                    performance_summary
                )
            elif self.config.matching_algorithm == "PRO_RATA":
                analyzer = ProRataAnalyzer()
                algorithm_analysis[symbol] = analyzer.analyze_performance(
                    order_book.get_statistics(),
                    performance_summary
                )
        
        return {
            'simulation_config': {
                'duration_seconds': self.config.duration_seconds,
                'target_throughput': self.config.target_throughput,
                'actual_duration': duration_seconds,
                'matching_algorithm': self.config.matching_algorithm,
                'symbols': self.config.symbols,
            },
            'summary_metrics': {
                'total_orders_processed': self.total_orders_processed,
                'total_trades_executed': self.total_trades_executed,
                'total_volume': self.total_volume,
                'actual_throughput': self.total_orders_processed / max(duration_seconds, 0.001),
                'trade_ratio': self.total_trades_executed / max(self.total_orders_processed, 1),
                'average_trade_size': self.total_volume / max(self.total_trades_executed, 1),
            },
            'performance_data': performance_summary,
            'order_book_statistics': order_book_stats,
            'stream_statistics': stream_stats,
            'algorithm_analysis': algorithm_analysis,
            'timestamp': time.time_ns(),
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted simulation results."""
        print("\n" + "="*80)
        print("SIMULATION RESULTS")
        print("="*80)
        
        # Summary
        config = results['simulation_config']
        summary = results['summary_metrics']
        
        print(f"Duration: {config['actual_duration']:.2f}s (target: {config['duration_seconds']}s)")
        print(f"Algorithm: {config['matching_algorithm']}")
        print(f"Symbols: {', '.join(config['symbols'])}")
        print()
        
        print(f"Orders Processed: {summary['total_orders_processed']:,}")
        print(f"Trades Executed: {summary['total_trades_executed']:,}")
        print(f"Total Volume: {summary['total_volume']:,.2f}")
        print(f"Actual Throughput: {summary['actual_throughput']:.1f} orders/sec")
        print(f"Trade Ratio: {summary['trade_ratio']:.1%}")
        print(f"Avg Trade Size: {summary['average_trade_size']:.4f}")
        print()
        
        # Performance metrics
        perf_data = results['performance_data']
        if 'performance_metrics' in perf_data:
            perf_metrics = perf_data['performance_metrics']
            if 'order_processing_latency_ms' in perf_metrics:
                latency = perf_metrics['order_processing_latency_ms']
                print(f"Latency - Mean: {latency.get('mean', 0):.3f}ms, "
                      f"P95: {latency.get('p95', 0):.3f}ms, "
                      f"P99: {latency.get('p99', 0):.3f}ms")
        
        print("="*80)
    
    def export_results(self, results: Dict[str, Any], filename: str = None) -> str:
        """Export results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"simulation_results_{timestamp}.json"
        
        import json
        try:
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"Results exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Failed to export results: {e}")
            return ""


def run_benchmark_simulation(duration: float = 60.0, 
                           throughput: float = 1000.0,
                           algorithm: str = "FIFO") -> Dict[str, Any]:
    """
    Run a standard benchmark simulation.
    
    Args:
        duration: Simulation duration in seconds
        throughput: Target throughput in orders/sec
        algorithm: Matching algorithm ("FIFO" or "PRO_RATA")
    
    Returns:
        Simulation results dictionary
    """
    config = SimulationConfig(
        duration_seconds=duration,
        target_throughput=throughput,
        matching_algorithm=algorithm,
        enable_profiling=True,
        enable_magic_trace=True,
        batch_processing=True
    )
    
    engine = SimulationEngine(config)
    results = engine.run_simulation()
    engine.print_results(results)
    
    return results
