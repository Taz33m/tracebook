"""
Main OrderBook implementation - the primary interface for the high-performance order book.

This module provides the main OrderBook class that coordinates all components
and provides a clean API for order management and market data access.
"""

from typing import Any, Callable, Dict, List, Optional, Set
from dataclasses import dataclass
import math
import time
import threading
from collections import defaultdict, deque

from .order import (
    NO_OWNER,
    Order,
    OrderFactory,
    OrderSide,
    OrderType,
    SelfTradePolicy,
    Trade,
    normalize_symbol,
)
from .matching_engine import MatchingEngine
from .price_level import MarketDataSnapshot
from .replay import EventLog


@dataclass
class OrderResult:
    """Structured outcome for richer order-submission APIs.

    `accepted` is True when the order passed validation and was processed by the
    matching engine (even if it did not fill, e.g. an unfillable FOK). It is
    False only for a hard rejection where the order never entered the book.
    """

    order: Optional[Order]
    trades: List[Trade]
    rested: bool
    cancelled: bool
    rejected_reason: Optional[str] = None
    accepted: bool = True


class OrderBook:
    """
    Order book with a FIFO / pro-rata matching engine.

    Features:
    - Multiple matching algorithms (FIFO, Pro-Rata)
    - Thread-safe operations
    - Real-time market data snapshots
    - Comprehensive performance metrics
    - Event-driven architecture for callbacks
    """

    def __init__(
        self,
        symbol: str,
        matching_algorithm: str = "fifo",
        tick_size: float = 0.01,
        self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
    ):
        """
        Initialize order book.

        Args:
            symbol: Trading symbol (e.g., 'AAPL', 'BTCUSD')
            matching_algorithm: 'fifo' or 'pro_rata'
            tick_size: Minimum price increment; prices are snapped onto this grid
            self_trade_policy: How to handle an order that would match its own
                owner's resting order (see SelfTradePolicy); requires orders to
                carry an owner id via the `owner` argument
        """
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("tick_size must be a positive, finite number")
        self.symbol = normalize_symbol(symbol)
        self.matching_algorithm = matching_algorithm
        self.tick_size = tick_size
        self.self_trade_policy = SelfTradePolicy(self_trade_policy)

        # Core components
        self.matching_engine = MatchingEngine(
            self.symbol, matching_algorithm, tick_size, self.self_trade_policy
        )
        self.order_factory = OrderFactory()

        # Thread safety
        self._lock = threading.RLock()

        # Event callbacks
        self._trade_callbacks: List[Callable] = []
        self._order_callbacks: List[Callable] = []
        self._market_data_callbacks: List[Callable] = []

        # Performance tracking
        self._start_time = time.time_ns()
        self._last_snapshot_time = 0
        self._snapshot_interval_ns = 1_000_000  # 1ms default

        # Statistics
        self.stats: Dict[str, float] = {
            "orders_added": 0,
            "orders_cancelled": 0,
            "trades_executed": 0,
            "total_volume": 0,
            "avg_processing_time_ns": 0,
            "max_processing_time_ns": 0,
            "min_processing_time_ns": float("inf"),
        }
        # Replay/duplicate guard. Bounded so long-running books don't leak: only
        # the most recent `_seen_id_cap` processed ids are remembered. Ids evicted
        # beyond that window may be reused (a degenerate case for real workloads).
        self._seen_id_cap = 1_000_000
        self._seen_order_ids: Set[int] = set()
        self._seen_order_id_queue: deque = deque()

        # Optional event recorder for deterministic replay (see start_recording).
        self._recorder: Optional[EventLog] = None

    # The book exposes two submission surfaces over one core (`_process_order`):
    #   * submit_* -> OrderResult (canonical): never raises; input and validation
    #     errors surface as `rejected_reason`.
    #   * add_*    -> List[Trade] (terse convenience): raises ValueError on invalid
    #     input, and returns the executed trades otherwise. Note that an
    #     unfillable FOK is a normal empty result here, not an error.

    def add_limit_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> List[Trade]:
        """Add a limit order to the book and return executed trades."""
        order = self.order_factory.create_limit_order(self.symbol, side, price, quantity, owner)
        return self._process_order(order, validate=False).trades

    def submit_limit_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> OrderResult:
        """Submit a limit order and return a structured result."""
        return self._submit_new_order(
            self.order_factory.create_limit_order, side, price, quantity, owner
        )

    def add_market_order(
        self, side: OrderSide, quantity: float, owner: int = NO_OWNER
    ) -> List[Trade]:
        """Add a market order to the book and return executed trades."""
        order = self.order_factory.create_market_order(self.symbol, side, quantity, owner)
        return self._process_order(order, validate=False).trades

    def submit_market_order(
        self, side: OrderSide, quantity: float, owner: int = NO_OWNER
    ) -> OrderResult:
        """Submit a market order and return a structured result."""
        return self._submit_new_order(self.order_factory.create_market_order, side, quantity, owner)

    def add_ioc_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> List[Trade]:
        """
        Add an Immediate-or-Cancel order to the book.

        Any unfilled quantity is cancelled rather than resting.
        """
        order = self.order_factory.create_ioc_order(self.symbol, side, price, quantity, owner)
        return self._process_order(order, validate=False).trades

    def submit_ioc_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> OrderResult:
        """Submit an Immediate-or-Cancel order and return a structured result."""
        return self._submit_new_order(
            self.order_factory.create_ioc_order, side, price, quantity, owner
        )

    def add_fok_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> List[Trade]:
        """
        Add a Fill-or-Kill order to the book.

        The order executes only when the full quantity is immediately available.
        """
        order = self.order_factory.create_fok_order(self.symbol, side, price, quantity, owner)
        return self._process_order(order, validate=False).trades

    def submit_fok_order(
        self, side: OrderSide, price: float, quantity: float, owner: int = NO_OWNER
    ) -> OrderResult:
        """Submit a Fill-or-Kill order and return a structured result."""
        return self._submit_new_order(
            self.order_factory.create_fok_order, side, price, quantity, owner
        )

    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book, returning executed trades.

        Raises ValueError if the order fails validation.
        """
        return self._process_order(order).trades

    def submit_order(self, order: Order) -> OrderResult:
        """Submit an existing order and return a structured result (never raises)."""
        try:
            return self._process_order(order)
        except ValueError as exc:
            return OrderResult(order, [], False, False, str(exc), accepted=False)

    def _submit_new_order(self, create, *args) -> OrderResult:
        """Build an order via the factory and submit it.

        Construction errors (invalid side/price/quantity) are captured as a
        structured rejection instead of raising, matching submit_* semantics.
        """
        try:
            order = create(self.symbol, *args)
        except ValueError as exc:
            return OrderResult(None, [], False, False, str(exc), accepted=False)
        # The factory already validated the order, so use the trusted fast path.
        return self._process_order(order, validate=False)

    def _process_order(self, order: Order, validate: bool = True) -> OrderResult:
        """Process and summarize an incoming order.

        `validate=False` is the trusted fast path for orders built by this book's
        own factory (which already validates side/type/price/quantity/owner and
        uses the book symbol with a fresh id), so the redundant book-level
        validation and factory-id reconciliation are skipped.
        """
        start_time = time.time_ns()
        snapshot = None

        with self._lock:
            order_id = int(order.order_id)
            if validate:
                self._validate_order(order)
                # Reconcile a caller-supplied id; factory ids are already ahead.
                self.order_factory.advance_past(order_id)
            self._mark_seen(order_id)

            # Record the order as submitted, before matching mutates its price
            # (tick snapping) or remaining quantity.
            if self._recorder is not None:
                self._recorder.record_submit(order)

            # Execute order through matching engine
            trades = self.matching_engine.add_order(order)

            # Update statistics
            self.stats["orders_added"] += 1
            self.stats["trades_executed"] += len(trades)

            # Calculate volume
            total_volume = sum(trade.price * trade.quantity for trade in trades)
            self.stats["total_volume"] += total_volume

            # Update processing time stats
            processing_time = time.time_ns() - start_time
            self._update_processing_time_stats(processing_time)

            snapshot = self._get_market_data_snapshot_if_due()

            # The engine rests leftover quantity only for can_rest() orders, so
            # this matches book membership without a second index lookup.
            rested = order.can_rest() and order.remaining_quantity > 1e-12
            cancelled = order.remaining_quantity > 1e-12 and not rested
            rejected_reason = None
            if order.is_fok_order() and cancelled and not trades:
                rejected_reason = "FOK order could not be fully filled"

        self._trigger_order_callbacks(order, trades)
        if trades:
            self._trigger_trade_callbacks(trades)
        if snapshot is not None:
            self._trigger_market_data_callbacks(snapshot)

        return OrderResult(order, trades, rested, cancelled, rejected_reason)

    def _mark_seen(self, order_id: int) -> None:
        """Record a processed order id, evicting the oldest past the cap."""
        if order_id in self._seen_order_ids:
            return
        self._seen_order_ids.add(order_id)
        self._seen_order_id_queue.append(order_id)
        if len(self._seen_order_id_queue) > self._seen_id_cap:
            evicted = self._seen_order_id_queue.popleft()
            self._seen_order_ids.discard(evicted)

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
                self.stats["orders_cancelled"] += 1
                if self._recorder is not None:
                    self._recorder.record_cancel(order_id)
            return success

    def start_recording(self) -> EventLog:
        """Begin recording mutating operations into a fresh event log.

        Returns the log, which can later be serialized and replayed with
        `tracebook.core.replay.replay` to reconstruct identical trades and book
        state. Recording an already-recording book restarts the log.
        """
        with self._lock:
            self._recorder = EventLog(
                self.symbol,
                self.matching_algorithm,
                self.tick_size,
                int(self.self_trade_policy),
            )
            return self._recorder

    def stop_recording(self) -> Optional[EventLog]:
        """Stop recording and return the accumulated event log (or None)."""
        with self._lock:
            log = self._recorder
            self._recorder = None
            return log

    def replace_order(
        self,
        order_id: int,
        price: Optional[float] = None,
        quantity: Optional[float] = None,
    ) -> OrderResult:
        """
        Replace a resting limit order by cancelling it and submitting a new order.

        The step is atomic: an invalid replacement is rejected before the original
        is cancelled, and if the replacement fails to submit after cancellation the
        original resting order is restored. A replace therefore never destroys
        resting liquidity. The replacement receives a new order id and timestamp.
        """
        with self._lock:
            existing_order = self.get_order(order_id)
            if existing_order is None:
                return OrderResult(None, [], False, False, "Order not found", accepted=False)

            replacement_price = existing_order.price if price is None else price
            replacement_quantity = (
                existing_order.remaining_quantity if quantity is None else quantity
            )

            try:
                replacement_order = self.order_factory.create_limit_order(
                    self.symbol,
                    OrderSide(existing_order.side),
                    replacement_price,
                    replacement_quantity,
                )
            except ValueError as exc:
                # Invalid replacement: original order is left untouched.
                return OrderResult(None, [], False, False, str(exc), accepted=False)

            if not self.cancel_order(order_id):
                return OrderResult(
                    None, [], False, False, "Order could not be cancelled", accepted=False
                )

            result = self.submit_order(replacement_order)
            if result.rejected_reason is not None:
                # Replacement failed after cancellation: restore the original.
                self._restore_resting_order(existing_order)
                self.stats["orders_cancelled"] -= 1
            return result

    def _restore_resting_order(self, order: Order) -> None:
        """Re-rest a previously cancelled order after a failed replacement."""
        if int(order.side) == int(OrderSide.BUY):
            self.matching_engine.buy_side.add_order(order)
        else:
            self.matching_engine.sell_side.add_order(order)

    def get_order(self, order_id: int) -> Optional[Order]:
        """Return a resting order by id, if it is currently active."""
        with self._lock:
            order = self.matching_engine.buy_side.get_order(order_id)
            if order is not None:
                return order
            return self.matching_engine.sell_side.get_order(order_id)

    def get_active_order_ids(self, side: Optional[OrderSide] = None) -> List[int]:
        """Return active resting order ids, optionally filtered by side."""
        with self._lock:
            if side is None:
                return sorted(
                    list(self.matching_engine.buy_side.orders.keys())
                    + list(self.matching_engine.sell_side.orders.keys())
                )

            side_value = int(side)
            if side_value == int(OrderSide.BUY):
                return sorted(self.matching_engine.buy_side.orders.keys())
            if side_value == int(OrderSide.SELL):
                return sorted(self.matching_engine.sell_side.orders.keys())

            raise ValueError(f"Unsupported order side: {side}")

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
        if levels < 0:
            raise ValueError("levels must be non-negative")
        with self._lock:
            return self.matching_engine.get_order_book_depth(levels)

    def get_recent_trades(self, count: int = 10) -> List[Trade]:
        """Get recent trades."""
        if count <= 0:
            return []
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
                "uptime_seconds": (time.time_ns() - self._start_time) / 1_000_000_000,
                "orders_per_second": self._calculate_orders_per_second(),
                "trades_per_second": self._calculate_trades_per_second(),
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
        if interval_ms <= 0:
            raise ValueError("snapshot interval must be positive")
        self._snapshot_interval_ns = int(interval_ms * 1_000_000)

    def _validate_order(self, order: Order):
        """Validate an incoming order before matching."""
        order_symbol = normalize_symbol(order.symbol)
        if order_symbol != self.symbol:
            raise ValueError(
                f"Order symbol {order.symbol!r} does not match book symbol {self.symbol!r}"
            )
        order.symbol = order_symbol

        order_id = int(order.order_id)
        if order_id <= 0:
            raise ValueError("Order id must be positive")

        if self.get_order(order.order_id) is not None:
            raise ValueError(f"Order id {order.order_id} is already active")

        if order_id in self._seen_order_ids:
            raise ValueError(f"Order id {order.order_id} has already been processed")

        if int(order.side) not in (int(OrderSide.BUY), int(OrderSide.SELL)):
            raise ValueError(f"Unsupported order side: {order.side}")

        supported_types = (
            int(OrderType.MARKET),
            int(OrderType.LIMIT),
            int(OrderType.IOC),
            int(OrderType.FOK),
        )
        if int(order.order_type) not in supported_types:
            raise ValueError(f"Unsupported order type: {order.order_type}")

        if (
            not math.isfinite(order.quantity)
            or not math.isfinite(order.remaining_quantity)
            or order.quantity <= 0
            or order.remaining_quantity <= 1e-12
            or order.remaining_quantity > order.quantity + 1e-12
        ):
            raise ValueError("Order quantity must be positive")

        if not math.isfinite(order.price):
            raise ValueError("Order price must be finite")

        if (
            int(order.order_type) in (int(OrderType.LIMIT), int(OrderType.IOC), int(OrderType.FOK))
            and order.price <= 0
        ):
            raise ValueError("Limit-style orders must have a positive price")

    def clear(self):
        """Clear all orders and reset statistics."""
        with self._lock:
            self.matching_engine.clear()
            self.stats = {
                "orders_added": 0,
                "orders_cancelled": 0,
                "trades_executed": 0,
                "total_volume": 0,
                "avg_processing_time_ns": 0,
                "max_processing_time_ns": 0,
                "min_processing_time_ns": float("inf"),
            }
            self._start_time = time.time_ns()
            self._seen_order_ids.clear()
            self._seen_order_id_queue.clear()

    def _update_processing_time_stats(self, processing_time_ns: int):
        """Update processing time statistics."""
        self.stats["max_processing_time_ns"] = max(
            self.stats["max_processing_time_ns"], processing_time_ns
        )
        self.stats["min_processing_time_ns"] = min(
            self.stats["min_processing_time_ns"], processing_time_ns
        )

        # Update running average
        total_orders = self.stats["orders_added"]
        if total_orders > 1:
            current_avg = self.stats["avg_processing_time_ns"]
            self.stats["avg_processing_time_ns"] = (
                current_avg * (total_orders - 1) + processing_time_ns
            ) / total_orders
        else:
            self.stats["avg_processing_time_ns"] = processing_time_ns

    def _calculate_orders_per_second(self) -> float:
        """Calculate orders per second."""
        uptime_seconds = (time.time_ns() - self._start_time) / 1_000_000_000
        if uptime_seconds > 0:
            return self.stats["orders_added"] / uptime_seconds
        return 0.0

    def _calculate_trades_per_second(self) -> float:
        """Calculate trades per second."""
        uptime_seconds = (time.time_ns() - self._start_time) / 1_000_000_000
        if uptime_seconds > 0:
            return self.stats["trades_executed"] / uptime_seconds
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
        snapshot = self._get_market_data_snapshot_if_due()
        if snapshot is not None:
            self._trigger_market_data_callbacks(snapshot)

    def _get_market_data_snapshot_if_due(self) -> Optional[MarketDataSnapshot]:
        """Return a market data snapshot if the snapshot interval elapsed."""
        current_time = time.time_ns()
        if current_time - self._last_snapshot_time >= self._snapshot_interval_ns:
            snapshot = self.matching_engine.get_market_data_snapshot()
            self._last_snapshot_time = current_time
            return snapshot
        return None

    def __str__(self) -> str:
        """String representation of order book."""
        snapshot = self.get_market_data_snapshot()

        lines = [f"OrderBook({self.symbol}) - {self.matching_algorithm.upper()}"]
        lines.append(f"Best Bid: {snapshot.best_bid}, Best Ask: {snapshot.best_ask}")
        lines.append(f"Spread: {snapshot.spread}, Mid: {snapshot.mid_price}")

        # Show top 3 levels
        lines.append("\nAsks:")
        for price, qty, count in snapshot.ask_levels[:3]:
            lines.append(f"  {price:8.2f} | {qty:10.6g} ({count:2d})")

        lines.append("Bids:")
        for price, qty, count in snapshot.bid_levels[:3]:
            lines.append(f"  {price:8.2f} | {qty:10.6g} ({count:2d})")

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

    def create_order_book(
        self,
        symbol: str,
        matching_algorithm: str = "fifo",
        tick_size: float = 0.01,
        self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
    ) -> OrderBook:
        """Create a new order book for a symbol."""
        symbol = normalize_symbol(symbol)
        with self._lock:
            if symbol in self.order_books:
                raise ValueError(f"Order book for {symbol} already exists")

            order_book = OrderBook(symbol, matching_algorithm, tick_size, self_trade_policy)
            self.order_books[symbol] = order_book
            return order_book

    def add_order_book(self, symbol: str, order_book: OrderBook):
        """Add an existing order book for a symbol."""
        symbol = normalize_symbol(symbol)
        with self._lock:
            if symbol in self.order_books:
                raise ValueError(f"Order book for {symbol} already exists")
            if order_book.symbol != symbol:
                raise ValueError(
                    f"Order book symbol {order_book.symbol!r} does not match "
                    f"registry key {symbol!r}"
                )
            self.order_books[symbol] = order_book

    def get_order_book(self, symbol: str) -> Optional[OrderBook]:
        """Get order book for a symbol."""
        symbol = normalize_symbol(symbol)
        with self._lock:
            return self.order_books.get(symbol)

    def get_all_symbols(self) -> List[str]:
        """Get all available symbols."""
        with self._lock:
            return list(self.order_books.keys())

    def get_all_order_books(self) -> Dict[str, OrderBook]:
        """Get a shallow copy of all managed order books keyed by symbol."""
        with self._lock:
            return dict(self.order_books)

    def get_global_statistics(self) -> Dict[str, Any]:
        """Get aggregated statistics across all order books."""
        with self._lock:
            summed_keys = {
                "orders_added",
                "orders_cancelled",
                "trades_executed",
                "total_volume",
                "total_orders_processed",
                "total_matches",
                "total_trades",
                "buy_side_orders",
                "sell_side_orders",
                "buy_side_quantity",
                "sell_side_quantity",
            }
            aggregate: Dict[str, float] = {key: 0 for key in summed_keys}
            max_processing_time = 0
            min_processing_time = float("inf")
            weighted_processing_time = 0.0
            processing_weight = 0
            max_uptime = 0.0
            last_trade_time = 0

            for order_book in self.order_books.values():
                stats = order_book.get_statistics()

                for key in summed_keys:
                    aggregate[key] += stats.get(key, 0)

                max_processing_time = max(
                    max_processing_time, stats.get("max_processing_time_ns", 0)
                )
                min_processing_time = min(
                    min_processing_time, stats.get("min_processing_time_ns", float("inf"))
                )
                orders_added = stats.get("orders_added", 0)
                weighted_processing_time += stats.get("avg_processing_time_ns", 0) * orders_added
                processing_weight += orders_added
                max_uptime = max(max_uptime, stats.get("uptime_seconds", 0.0))
                last_trade_time = max(last_trade_time, stats.get("last_trade_time", 0))

            aggregate["avg_processing_time_ns"] = (
                weighted_processing_time / processing_weight if processing_weight else 0
            )
            aggregate["max_processing_time_ns"] = max_processing_time
            aggregate["min_processing_time_ns"] = (
                0 if min_processing_time == float("inf") else min_processing_time
            )
            aggregate["uptime_seconds"] = max_uptime
            aggregate["orders_per_second"] = (
                aggregate["orders_added"] / max_uptime if max_uptime > 0 else 0
            )
            aggregate["trades_per_second"] = (
                aggregate["trades_executed"] / max_uptime if max_uptime > 0 else 0
            )
            aggregate["last_trade_time"] = last_trade_time

            return aggregate

    def clear_all(self):
        """Clear all order books."""
        with self._lock:
            for order_book in self.order_books.values():
                order_book.clear()

    def remove_order_book(self, symbol: str) -> bool:
        """Remove an order book."""
        symbol = normalize_symbol(symbol)
        with self._lock:
            if symbol in self.order_books:
                del self.order_books[symbol]
                return True
            return False
