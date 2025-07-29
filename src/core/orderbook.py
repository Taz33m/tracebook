"""
Main OrderBook implementation - the primary interface for the high-performance order book.

This module provides the main OrderBook class that coordinates all components
and provides a clean API for order management and market data access.
"""

from typing import List, Optional, Dict, Any
import time
import threading
from collections import defaultdict

from .order import Order, Trade, OrderFactory, OrderSide
from .matching_engine import MatchingEngine
from .price_level import MarketDataSnapshot


class OrderBook:
    """
    High-performance order book with JIT-compiled matching engine.
    
    Features:
    - Multiple matching algorithms (FIFO, Pro-Rata)
    - Thread-safe operations
    - Real-time market data snapshots
    - Comprehensive performance metrics
    - Event-driven architecture for callbacks
    """
    
    def __init__(self, symbol: str, matching_algorithm: str = 'fifo'):
        """
        Initialize order book.
        
        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
            matching_algorithm: 'fifo' or 'pro_rata'
        """
        self.symbol = symbol
        self.matching_algorithm = matching_algorithm
        
        # Core components
        self.matching_engine = MatchingEngine(symbol, matching_algorithm)
        self.order_factory = OrderFactory()
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Event callbacks
        self._trade_callbacks = []
        self._order_callbacks = []
        self._market_data_callbacks = []
        
        # Performance tracking
        self._start_time = time.time_ns()
        self._last_snapshot_time = 0
        self._snapshot_interval_ns = 1_000_000  # 1ms default
        
        # Statistics
        self.stats = {
            'orders_added': 0,
            'orders_cancelled': 0,
            'trades_executed': 0,
            'total_volume': 0,
            'avg_processing_time_ns': 0,
            'max_processing_time_ns': 0,
            'min_processing_time_ns': float('inf'),
        }
    
    def add_limit_order(self, side: OrderSide, price: float, quantity: int) -> List[Trade]:
        """
        Add a limit order to the book.
        
        Args:
            side: OrderSide.BUY or OrderSide.SELL
            price: Limit price
            quantity: Order quantity
            
        Returns:
            List[Trade]: Executed trades
        """
        order = self.order_factory.create_limit_order(self.symbol, side, price, quantity)
        return self.add_order(order)
    
    def add_market_order(self, side: OrderSide, quantity: int) -> List[Trade]:
        """
        Add a market order to the book.
        
        Args:
            side: OrderSide.BUY or OrderSide.SELL
            quantity: Order quantity
            
        Returns:
            List[Trade]: Executed trades
        """
        order = self.order_factory.create_market_order(self.symbol, side, quantity)
        return self.add_order(order)
    
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book.
        
        Args:
            order: Order to add
            
        Returns:
            List[Trade]: Executed trades
        """
        start_time = time.time_ns()
        
        with self._lock:
            # Execute order through matching engine
            trades = self.matching_engine.add_order(order)
            
            # Update statistics
            self.stats['orders_added'] += 1
            self.stats['trades_executed'] += len(trades)
            
            # Calculate volume
            total_volume = sum(trade.price * trade.quantity for trade in trades)
            self.stats['total_volume'] += total_volume
            
            # Update processing time stats
            processing_time = time.time_ns() - start_time
            self._update_processing_time_stats(processing_time)
            
            # Trigger callbacks
            self._trigger_order_callbacks(order, trades)
            if trades:
                self._trigger_trade_callbacks(trades)
            
            # Check if market data snapshot is needed
            self._check_market_data_snapshot()
        
        return trades
    
    def process_orders_batch(self, orders: List[Order]) -> List[Trade]:
        """
        Process a batch of orders efficiently.
        
        Args:
            orders: List of orders to process
            
        Returns:
            List[Trade]: All executed trades from the batch
        """
        all_trades = []
        
        for order in orders:
            trades = self.add_order(order)
            all_trades.extend(trades)
        
        return all_trades
    
    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by ID.
        
        Args:
            order_id: ID of order to cancel
            
        Returns:
            bool: True if order was found and cancelled
        """
        with self._lock:
            success = self.matching_engine.cancel_order(order_id)
            if success:
                self.stats['orders_cancelled'] += 1
            return success
    
    def get_best_bid(self) -> Optional[float]:
        """Get best bid price."""
        with self._lock:
            return self.matching_engine.buy_side.get_best_price()
    
    def get_best_ask(self) -> Optional[float]:
        """Get best ask price."""
        with self._lock:
            return self.matching_engine.sell_side.get_best_price()
    
    def get_spread(self) -> Optional[float]:
        """Get bid-ask spread."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return ask - bid
        return None
    
    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        bid = self.get_best_bid()
        ask = self.get_best_ask()
        if bid is not None and ask is not None:
            return (bid + ask) / 2.0
        return None
    
    def get_market_data_snapshot(self) -> MarketDataSnapshot:
        """Get current market data snapshot."""
        with self._lock:
            return self.matching_engine.get_market_data_snapshot()
    
    def get_order_book_depth(self, levels: int = 5) -> Dict[str, Any]:
        """Get order book depth."""
        with self._lock:
            return self.matching_engine.get_order_book_depth(levels)
    
    def get_recent_trades(self, count: int = 10) -> List[Trade]:
        """Get recent trades."""
        with self._lock:
            return self.matching_engine.trades[-count:] if self.matching_engine.trades else []
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._lock:
            engine_stats = self.matching_engine.get_statistics()
            
            # Combine with order book stats
            combined_stats = {
                **self.stats,
                **engine_stats,
                'uptime_seconds': (time.time_ns() - self._start_time) / 1_000_000_000,
                'orders_per_second': self._calculate_orders_per_second(),
                'trades_per_second': self._calculate_trades_per_second(),
            }
            
            return combined_stats
    
    def register_trade_callback(self, callback):
        """Register callback for trade events."""
        self._trade_callbacks.append(callback)
    
    def register_order_callback(self, callback):
        """Register callback for order events."""
        self._order_callbacks.append(callback)
    
    def register_market_data_callback(self, callback):
        """Register callback for market data events."""
        self._market_data_callbacks.append(callback)
    
    def set_snapshot_interval(self, interval_ms: float):
        """Set market data snapshot interval in milliseconds."""
        self._snapshot_interval_ns = int(interval_ms * 1_000_000)
    
    def clear(self):
        """Clear all orders and reset statistics."""
        with self._lock:
            self.matching_engine.clear()
            self.stats = {
                'orders_added': 0,
                'orders_cancelled': 0,
                'trades_executed': 0,
                'total_volume': 0,
                'avg_processing_time_ns': 0,
                'max_processing_time_ns': 0,
                'min_processing_time_ns': float('inf'),
            }
            self._start_time = time.time_ns()
    
    def _update_processing_time_stats(self, processing_time_ns: int):
        """Update processing time statistics."""
        self.stats['max_processing_time_ns'] = max(
            self.stats['max_processing_time_ns'], processing_time_ns
        )
        self.stats['min_processing_time_ns'] = min(
            self.stats['min_processing_time_ns'], processing_time_ns
        )
        
        # Update running average
        total_orders = self.stats['orders_added']
        if total_orders > 1:
            current_avg = self.stats['avg_processing_time_ns']
            self.stats['avg_processing_time_ns'] = (
                (current_avg * (total_orders - 1) + processing_time_ns) / total_orders
            )
        else:
            self.stats['avg_processing_time_ns'] = processing_time_ns
    
    def _calculate_orders_per_second(self) -> float:
        """Calculate orders per second."""
        uptime_seconds = (time.time_ns() - self._start_time) / 1_000_000_000
        if uptime_seconds > 0:
            return self.stats['orders_added'] / uptime_seconds
        return 0.0
    
    def _calculate_trades_per_second(self) -> float:
        """Calculate trades per second."""
        uptime_seconds = (time.time_ns() - self._start_time) / 1_000_000_000
        if uptime_seconds > 0:
            return self.stats['trades_executed'] / uptime_seconds
        return 0.0
    
    def _trigger_trade_callbacks(self, trades: List[Trade]):
        """Trigger trade event callbacks."""
        for callback in self._trade_callbacks:
            try:
                callback(trades)
            except Exception as e:
                # Log error but don't let callback failures affect order processing
                print(f"Trade callback error: {e}")
    
    def _trigger_order_callbacks(self, order: Order, trades: List[Trade]):
        """Trigger order event callbacks."""
        for callback in self._order_callbacks:
            try:
                callback(order, trades)
            except Exception as e:
                print(f"Order callback error: {e}")
    
    def _trigger_market_data_callbacks(self, snapshot: MarketDataSnapshot):
        """Trigger market data callbacks."""
        for callback in self._market_data_callbacks:
            try:
                callback(snapshot)
            except Exception as e:
                print(f"Market data callback error: {e}")
    
    def _check_market_data_snapshot(self):
        """Check if market data snapshot should be generated."""
        current_time = time.time_ns()
        if current_time - self._last_snapshot_time >= self._snapshot_interval_ns:
            snapshot = self.get_market_data_snapshot()
            self._trigger_market_data_callbacks(snapshot)
            self._last_snapshot_time = current_time
    
    def __str__(self) -> str:
        """String representation of order book."""
        snapshot = self.get_market_data_snapshot()
        
        lines = [f"OrderBook({self.symbol}) - {self.matching_algorithm.upper()}"]
        lines.append(f"Best Bid: {snapshot.best_bid}, Best Ask: {snapshot.best_ask}")
        lines.append(f"Spread: {snapshot.spread}, Mid: {snapshot.mid_price}")
        
        # Show top 3 levels
        lines.append("\nAsks:")
        for price, qty, count in snapshot.ask_levels[:3]:
            lines.append(f"  {price:8.2f} | {qty:6d} ({count:2d})")
        
        lines.append("Bids:")
        for price, qty, count in snapshot.bid_levels[:3]:
            lines.append(f"  {price:8.2f} | {qty:6d} ({count:2d})")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """Detailed representation."""
        return f"OrderBook(symbol='{self.symbol}', algorithm='{self.matching_algorithm}')"


class OrderBookManager:
    """
    Manages multiple order books for different symbols.
    
    Provides a centralized interface for multi-symbol trading
    with shared performance monitoring and event handling.
    """
    
    def __init__(self):
        self.order_books: Dict[str, OrderBook] = {}
        self._global_stats = defaultdict(int)
        self._lock = threading.RLock()
    
    def create_order_book(self, symbol: str, matching_algorithm: str = 'fifo') -> OrderBook:
        """Create a new order book for a symbol."""
        with self._lock:
            if symbol in self.order_books:
                raise ValueError(f"Order book for {symbol} already exists")
            
            order_book = OrderBook(symbol, matching_algorithm)
            self.order_books[symbol] = order_book
            return order_book
    
    def add_order_book(self, symbol: str, order_book: OrderBook):
        """Add an existing order book for a symbol."""
        with self._lock:
            if symbol in self.order_books:
                raise ValueError(f"Order book for {symbol} already exists")
            self.order_books[symbol] = order_book
    
    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        return self.order_books.get(symbol)
    
    def get_all_symbols(self) -> List[str]:
        """Get all available symbols."""
        return list(self.order_books.keys())
    
    def get_global_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics across all order books."""
        with self._lock:
            total_stats = defaultdict(int)
            
            for order_book in self.order_books.values():
                stats = order_book.get_statistics()
                for key, value in stats.items():
                    if isinstance(value, (int, float)):
                        total_stats[key] += value
            
            return dict(total_stats)
    
    def clear_all(self):
        """Clear all order books."""
        with self._lock:
            for order_book in self.order_books.values():
                order_book.clear()
    
    def remove_order_book(self, symbol: str) -> bool:
        """Remove an order book."""
        with self._lock:
            if symbol in self.order_books:
                del self.order_books[symbol]
                return True
            return False
