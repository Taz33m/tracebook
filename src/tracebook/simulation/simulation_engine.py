"""
High-performance simulation engine for order book testing.

Coordinates order generation, order book processing, and performance monitoring
to provide comprehensive simulation capabilities.
"""

import argparse
import json
import math
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np

from .. import __version__
from ..core.orderbook import OrderBook, OrderBookManager
from ..core.order import Order, Trade, OrderSide, normalize_symbol
from ..profiling.performance_monitor import PerformanceMonitor
from .order_generator import (
    SyntheticOrderStream,
    MarketParameters,
    OrderGenerationConfig,
    OrderPattern,
    SimulationEvent,
    SimulationEventType,
)


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
    symbols: List[str] = field(default_factory=lambda: ["BTCUSD"])
    seed: Optional[int] = None
    cancel_ratio: float = 0.0
    replace_ratio: float = 0.0
    warmup_seconds: float = 0.0
    output_path: Optional[str] = None

    def __post_init__(self):
        if not self.symbols:
            raise ValueError("symbols must contain at least one symbol")
        self.symbols = [normalize_symbol(symbol) for symbol in self.symbols]

        self.order_pattern = OrderPattern(self.order_pattern)
        self.matching_algorithm = self.matching_algorithm.upper()
        if self.matching_algorithm not in ("FIFO", "PRO_RATA"):
            raise ValueError(f"Unsupported matching algorithm: {self.matching_algorithm}")

        if self.duration_seconds < 0:
            raise ValueError("duration_seconds must be non-negative")
        if self.target_throughput <= 0:
            raise ValueError("target_throughput must be positive")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.warmup_seconds < 0:
            raise ValueError("warmup_seconds must be non-negative")
        if not 0 <= self.cancel_ratio <= 1:
            raise ValueError("cancel_ratio must be between 0 and 1")
        if not 0 <= self.replace_ratio <= 1:
            raise ValueError("replace_ratio must be between 0 and 1")
        if self.cancel_ratio + self.replace_ratio > 1:
            raise ValueError("cancel_ratio + replace_ratio must be <= 1")


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
        self.random_state = np.random.default_rng(config.seed)

        # Order streams per symbol
        self.order_streams: Dict[str, Any] = {}
        self._setup_order_streams()

        # Simulation state
        self.is_running = False
        self.simulation_thread = None
        self.start_time = 0
        self.end_time = 0

        # Statistics
        self.total_orders_processed = 0
        self.total_events_processed = 0
        self.total_cancel_events = 0
        self.total_replace_events = 0
        self.total_trades_executed = 0
        self.total_volume = 0.0
        self.processing_times: List[float] = []

        # Event callbacks
        self.trade_callbacks: List[Callable] = []
        self.order_callbacks: List[Callable] = []
        self.simulation_callbacks: List[Callable] = []

    def _setup_order_streams(self):
        """Setup order streams for each symbol."""
        for index, symbol in enumerate(self.config.symbols):
            stream_seed = None if self.config.seed is None else self.config.seed + index
            stream_throughput = self.config.target_throughput / len(self.config.symbols)
            stream_batch_size = min(
                self.config.batch_size,
                max(1, math.ceil(stream_throughput * 0.1)),
            )
            market_params = MarketParameters(
                symbol=symbol,
                order_arrival_rate=stream_throughput,
                cancel_ratio=self.config.cancel_ratio,
            )

            stream_config = OrderGenerationConfig(
                pattern=self.config.order_pattern,
                duration_seconds=self.config.duration_seconds,
                target_throughput=stream_throughput,
                batch_size=stream_batch_size,
                seed=stream_seed,
            )

            stream = SyntheticOrderStream(market_params, stream_config, self.performance_monitor)
            self.order_streams[symbol] = stream

            # Setup order book for symbol
            order_book = OrderBook(symbol=symbol, matching_algorithm=self.config.matching_algorithm)
            # Share ID allocation between generated new orders and book-created replacements.
            # Otherwise a replacement created while processing a prefetched batch can collide
            # with a generated order that has not reached the book yet.
            order_book.order_factory = stream.order_factory

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
        print(
            f"Starting simulation - Duration: {self.config.duration_seconds}s, "
            f"Target: {self.config.target_throughput} orders/sec"
        )

        if self.config.warmup_seconds > 0:
            self._warm_up()

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

        if self.config.output_path:
            self.export_results(results, self.config.output_path)

        print(
            f"Simulation completed - Processed {self.total_orders_processed:,} orders, "
            f"Executed {self.total_trades_executed:,} trades"
        )

        return results

    def _simulation_loop(self):
        """Main simulation processing loop."""
        end_time = time.time() + self.config.duration_seconds

        while time.time() < end_time and self.is_running:
            loop_start = time.time_ns()

            # Process events from all streams
            for symbol, stream in self.order_streams.items():
                events = stream.get_events(self.config.batch_size)

                if events:
                    order_book = self._get_order_book(symbol)
                    self._process_events(symbol, order_book, events)

            # Brief sleep to prevent CPU spinning
            loop_time = (time.time_ns() - loop_start) / 1_000_000  # ms
            if loop_time < 1.0:  # If loop took less than 1ms
                time.sleep(0.001)  # Sleep for 1ms

    def _process_events(self, symbol: str, order_book: OrderBook, events: List[SimulationEvent]):
        """Process generated events and interleave lifecycle events."""
        for event in events:
            self._process_event(order_book, event)

            lifecycle_event = self._maybe_create_lifecycle_event(symbol, order_book)
            if lifecycle_event is not None:
                self._process_event(order_book, lifecycle_event)

    def _process_event(self, order_book: OrderBook, event: SimulationEvent):
        """Process one simulation event and record matching-only timing."""
        processing_start = time.time_ns()
        trades = []

        if event.event_type == SimulationEventType.NEW and event.order is not None:
            trades = order_book.add_order(event.order)
            processing_time = time.time_ns() - processing_start
            self.performance_monitor.record_order_processing(processing_time, 1)

        elif event.event_type == SimulationEventType.CANCEL and event.order_id is not None:
            order_book.cancel_order(event.order_id)
            self.total_cancel_events += 1
            processing_time = time.time_ns() - processing_start
            self.performance_monitor.metrics_collector.record_metric(
                name="order_event_latency_ms",
                value=processing_time / 1_000_000,
                unit="milliseconds",
                category="performance",
                metadata={"event_type": "cancel"},
            )

        elif event.event_type == SimulationEventType.REPLACE and event.order_id is not None:
            result = order_book.replace_order(event.order_id, event.price, event.quantity)
            trades = result.trades
            self.total_replace_events += 1
            processing_time = time.time_ns() - processing_start
            if trades:
                # A replacement that crosses the book is matching work; count it
                # as matching latency so replace-heavy scenarios don't hide the
                # matching cost under lifecycle-event latency.
                self.performance_monitor.record_order_processing(processing_time, 1)
            else:
                self.performance_monitor.metrics_collector.record_metric(
                    name="order_event_latency_ms",
                    value=processing_time / 1_000_000,
                    unit="milliseconds",
                    category="performance",
                    metadata={"event_type": "replace"},
                )
        else:
            return

        self.total_events_processed += 1

        if trades:
            self.performance_monitor.record_trade_execution(
                len(trades), sum(trade.quantity * trade.price for trade in trades)
            )

    def _maybe_create_lifecycle_event(
        self,
        symbol: str,
        order_book: OrderBook,
    ) -> Optional[SimulationEvent]:
        """Create a cancel or replace event for an active resting order."""
        active_order_ids = order_book.get_active_order_ids()
        if not active_order_ids:
            return None

        draw = self.random_state.random()
        if draw < self.config.cancel_ratio:
            order_id = int(self.random_state.choice(active_order_ids))
            return SimulationEvent(SimulationEventType.CANCEL, symbol, order_id=order_id)

        if draw < self.config.cancel_ratio + self.config.replace_ratio:
            order_id = int(self.random_state.choice(active_order_ids))
            order = order_book.get_order(order_id)
            if order is None:
                return None

            tick = 0.01
            price_shift = float(self.random_state.choice([-1, 1])) * tick
            new_price = max(tick, order.price + price_shift)
            quantity_multiplier = float(self.random_state.uniform(0.75, 1.25))
            new_quantity = max(0.001, order.remaining_quantity * quantity_multiplier)
            return SimulationEvent(
                SimulationEventType.REPLACE,
                symbol,
                order_id=order_id,
                price=new_price,
                quantity=new_quantity,
            )

        return None

    def _warm_up(self):
        """Exercise the matching paths (warming interpreter caches) before measuring."""
        deadline = time.time() + self.config.warmup_seconds
        while time.time() < deadline:
            for algorithm in ("FIFO", "PRO_RATA"):
                book = OrderBook("WARMUP", matching_algorithm=algorithm)
                book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
                book.add_limit_order(OrderSide.SELL, 100.0, 0.5)
                book.add_market_order(OrderSide.BUY, 0.1)
            if self.config.warmup_seconds < 0.05:
                break

    def _get_order_book(self, symbol: str) -> OrderBook:
        """Return a configured order book or fail with a clear simulation error."""
        order_book = self.order_book_manager.get_order_book(symbol)
        if order_book is None:
            raise RuntimeError(f"No order book configured for symbol {symbol!r}")
        return order_book

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
            order_book = self._get_order_book(symbol)
            order_book_stats[symbol] = order_book.get_statistics()

        # Stream statistics
        stream_stats = {}
        for symbol, stream in self.order_streams.items():
            stream_stats[symbol] = stream.get_stream_stats()

        # Algorithm analyzers are available as standalone helpers, but the live
        # matching path does not yet emit the per-match records they require.
        algorithm_analysis = {
            symbol: {
                "algorithm": self.config.matching_algorithm,
                "status": "not_collected",
                "reason": "Per-match analyzer instrumentation is not wired into the simulator.",
                "matches_observed": order_book_stats[symbol].get("total_matches", 0),
            }
            for symbol in self.config.symbols
        }

        return {
            "simulation_config": {
                "duration_seconds": self.config.duration_seconds,
                "target_throughput": self.config.target_throughput,
                "actual_duration": duration_seconds,
                "matching_algorithm": self.config.matching_algorithm,
                "symbols": self.config.symbols,
                "seed": self.config.seed,
                "cancel_ratio": self.config.cancel_ratio,
                "replace_ratio": self.config.replace_ratio,
                "warmup_seconds": self.config.warmup_seconds,
            },
            "summary_metrics": {
                "total_orders_processed": self.total_orders_processed,
                "total_events_processed": self.total_events_processed,
                "total_cancel_events": self.total_cancel_events,
                "total_replace_events": self.total_replace_events,
                "total_trades_executed": self.total_trades_executed,
                "total_volume": self.total_volume,
                "actual_throughput": self.total_orders_processed / max(duration_seconds, 0.001),
                "trade_ratio": self.total_trades_executed / max(self.total_orders_processed, 1),
                "trades_per_order": self.total_trades_executed
                / max(self.total_orders_processed, 1),
                "average_trade_size": self.total_volume / max(self.total_trades_executed, 1),
            },
            "performance_data": performance_summary,
            "order_book_statistics": order_book_stats,
            "stream_statistics": stream_stats,
            "algorithm_analysis": algorithm_analysis,
            "timestamp": time.time_ns(),
        }

    def print_results(self, results: Dict[str, Any]):
        """Print formatted simulation results."""
        print("\n" + "=" * 80)
        print("SIMULATION RESULTS")
        print("=" * 80)

        # Summary
        config = results["simulation_config"]
        summary = results["summary_metrics"]

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
        perf_data = results["performance_data"]
        if "performance_metrics" in perf_data:
            perf_metrics = perf_data["performance_metrics"]
            if "order_processing_latency_ms" in perf_metrics:
                latency = perf_metrics["order_processing_latency_ms"]
                print(
                    f"Latency - Mean: {latency.get('mean', 0):.3f}ms, "
                    f"P95: {latency.get('p95', 0):.3f}ms, "
                    f"P99: {latency.get('p99', 0):.3f}ms"
                )

        print("=" * 80)

    def export_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> str:
        """Export results to file."""
        if filename is None:
            timestamp = int(time.time())
            filename = f"simulation_results_{timestamp}.json"

        try:
            output_path = Path(filename)
            if output_path.parent != Path("."):
                output_path.parent.mkdir(parents=True, exist_ok=True)

            with output_path.open("w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, default=str)

            print(f"Results exported to: {output_path}")
            return str(output_path)

        except Exception as e:
            print(f"Failed to export results: {e}")
            return ""


def run_benchmark_simulation(
    duration: float = 60.0,
    throughput: float = 1000.0,
    algorithm: str = "FIFO",
    enable_magic_trace: bool = False,
    seed: Optional[int] = None,
    cancel_ratio: float = 0.0,
    replace_ratio: float = 0.0,
    warmup_seconds: float = 0.0,
    output_path: Optional[str] = None,
    symbols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run a standard benchmark simulation.

    Args:
        duration: Simulation duration in seconds
        throughput: Target throughput in orders/sec
        algorithm: Matching algorithm ("FIFO" or "PRO_RATA")
        symbols: Optional list of symbols; defaults to a single book

    Returns:
        Simulation results dictionary
    """
    config = SimulationConfig(
        duration_seconds=duration,
        target_throughput=throughput,
        matching_algorithm=algorithm,
        enable_profiling=True,
        enable_magic_trace=enable_magic_trace,
        batch_processing=True,
        seed=seed,
        cancel_ratio=cancel_ratio,
        replace_ratio=replace_ratio,
        warmup_seconds=warmup_seconds,
        output_path=output_path,
        # Fall back to the config's single-book default when none are given.
        symbols=list(symbols) if symbols else ["BTCUSD"],
    )

    engine = SimulationEngine(config)
    results = engine.run_simulation()
    engine.print_results(results)

    return results


def main(argv: Optional[List[str]] = None) -> int:
    """Run a benchmark simulation from the command line."""
    parser = argparse.ArgumentParser(description="Run a tracebook benchmark simulation.")
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    parser.add_argument(
        "--duration", type=float, default=5.0, help="Simulation duration in seconds."
    )
    parser.add_argument("--throughput", type=float, default=500.0, help="Target orders per second.")
    parser.add_argument(
        "--algorithm",
        choices=["FIFO", "PRO_RATA", "fifo", "pro_rata"],
        default="FIFO",
        help="Matching algorithm to use.",
    )
    parser.add_argument(
        "--magic-trace",
        action="store_true",
        help="Enable magic-trace integration or fallback tracing for the run.",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for deterministic order flow.")
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    parser.add_argument(
        "--cancel-ratio",
        type=float,
        default=0.0,
        help="Probability of cancel events after new orders.",
    )
    parser.add_argument(
        "--replace-ratio",
        type=float,
        default=0.0,
        help="Probability of replace events after new orders.",
    )
    parser.add_argument(
        "--warmup-seconds",
        type=float,
        default=0.0,
        help="JIT warmup duration excluded from results.",
    )
    args = parser.parse_args(argv)

    run_benchmark_simulation(
        duration=args.duration,
        throughput=args.throughput,
        algorithm=args.algorithm.upper(),
        enable_magic_trace=args.magic_trace,
        seed=args.seed,
        cancel_ratio=args.cancel_ratio,
        replace_ratio=args.replace_ratio,
        warmup_seconds=args.warmup_seconds,
        output_path=args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
