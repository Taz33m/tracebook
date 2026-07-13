"""
Main OrderBook implementation - the primary interface for the matching simulator.

This module provides the main OrderBook class that coordinates all components
and provides a clean API for order management and market data access.
"""

from typing import Any, Callable, Deque, Dict, List, Optional, Set
from dataclasses import dataclass
from numbers import Integral, Real
import math
import time
import threading
from collections import deque

from .order import (
    NO_OWNER,
    Order,
    OrderFactory,
    OrderSide,
    OrderType,
    SelfTradePolicy,
    Trade,
    copy_order,
    copy_trade,
    normalize_symbol,
)
from .matching_engine import MatchingEngine
from .price_level import MarketDataSnapshot, infer_price_decimals
from .replay import EventLog


def _validate_nonnegative_count(value: int, name: str) -> int:
    """Validate a public depth/history count without accepting bools or floats."""
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be non-negative integer")
    return int(value)


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

    symbol: str
    matching_algorithm: str
    tick_size: float
    _price_decimals: int
    self_trade_policy: SelfTradePolicy
    matching_engine: MatchingEngine
    order_factory: OrderFactory
    _lock: Any
    _trade_callbacks: List[Callable]
    _order_callbacks: List[Callable]
    _market_data_callbacks: List[Callable]
    _start_time: int
    _last_snapshot_time: int
    _snapshot_interval_ns: int
    stats: Dict[str, float]
    _seen_id_cap: int
    _seen_order_ids: Set[int]
    _seen_order_id_queue: Deque[int]
    _recorder: Optional[EventLog]

    def __init__(
        self,
        symbol: str,
        matching_algorithm: str = "fifo",
        tick_size: float = 0.01,
        self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
    ) -> None:
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
        if isinstance(tick_size, bool) or not isinstance(tick_size, Real):
            raise ValueError("tick_size must be a positive, finite number")
        tick_size = float(tick_size)
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("tick_size must be a positive, finite number")
        if isinstance(self_trade_policy, bool):
            raise ValueError("self_trade_policy must be a SelfTradePolicy value")
        if not isinstance(matching_algorithm, str):
            raise ValueError("matching_algorithm must be 'fifo' or 'pro_rata'")
        matching_algorithm = matching_algorithm.strip().lower()
        if matching_algorithm not in {"fifo", "pro_rata"}:
            raise ValueError("matching_algorithm must be 'fifo' or 'pro_rata'")
        self.symbol = normalize_symbol(symbol)
        self.matching_algorithm = matching_algorithm
        self.tick_size = tick_size
        self._price_decimals = infer_price_decimals(tick_size)
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
        self._seen_order_id_queue = deque()

        # Optional event recorder for deterministic replay (see start_recording).
        self._recorder: Optional[EventLog] = None

    # The book exposes two submission surfaces over one core (`_process_order`):
    #   * submit_* -> OrderResult (canonical): input and validation errors surface
    #     as `rejected_reason`.
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
        """Submit an existing order and return a structured result.

        The supplied object is normalized into a detached engine-owned copy. This
        prevents later caller mutation from corrupting live price-level indexes.
        Input and validation failures are returned as hard rejections.
        """
        try:
            return self._process_order(order)
        except (TypeError, ValueError, OverflowError) as exc:
            rejected_order = order if isinstance(order, Order) else None
            return OrderResult(rejected_order, [], False, False, str(exc), accepted=False)

    def _submit_new_order(self, create, *args) -> OrderResult:
        """Build an order via the factory and submit it.

        Construction errors (invalid side/price/quantity) are captured as a
        structured rejection instead of raising, matching submit_* semantics.
        """
        try:
            order = create(self.symbol, *args)
        except (TypeError, ValueError, OverflowError) as exc:
            return OrderResult(None, [], False, False, str(exc), accepted=False)
        # The factory already validated the order, so use the trusted fast path.
        # Price snapping still runs and may reject a sub-tick price, so capture
        # that as a structured rejection rather than raising.
        try:
            return self._process_order(order, validate=False)
        except (TypeError, ValueError, OverflowError) as exc:
            return OrderResult(order, [], False, False, str(exc), accepted=False)

    def _process_order(self, order: Order, validate: bool = True) -> OrderResult:
        """Process and summarize an incoming order.

        `validate=False` is the trusted fast path for orders built by this book's
        own factory (which already validates side/type/price/quantity/owner and
        uses the book symbol with a fresh id), so the redundant book-level
        validation and factory-id reconciliation are skipped.
        """
        with self._lock:
            result, snapshot, _ = self._process_order_locked(order, validate=validate)

        self._dispatch_order_events(result, snapshot)
        return result

    def _process_order_locked(
        self,
        order: Order,
        validate: bool = True,
        record: bool = True,
    ) -> tuple[OrderResult, Optional[MarketDataSnapshot], Order]:
        """Process an order while the caller holds ``self._lock``.

        The split lets replacement remain one atomic cancel-and-new transaction
        while still delivering user callbacks after the lock is released.
        """
        start_time = time.time_ns()
        working_order = self._normalize_external_order(order) if validate else order
        order_id = working_order.order_id

        # Matching and resting prices must use the same canonical tick.
        self._snap_order_price(working_order)
        if validate:
            self.order_factory.advance_past(order_id)
        self._mark_seen(order_id)

        # Capture the submitted state before the matching engine mutates remaining
        # quantity. Recording is committed only after matching succeeds.
        submitted_order = copy_order(working_order)
        trades = self.matching_engine.add_order(working_order)
        if record and self._recorder is not None:
            self._recorder.record_submit(submitted_order)

        self.stats["orders_added"] += 1
        self.stats["trades_executed"] += len(trades)
        self.stats["total_volume"] += sum(trade.price * trade.quantity for trade in trades)
        self._update_processing_time_stats(time.time_ns() - start_time)

        snapshot = self._get_market_data_snapshot_if_due()
        rested = working_order.can_rest() and working_order.remaining_quantity > 1e-12
        cancelled = working_order.remaining_quantity > 1e-12 and not rested
        rejected_reason = None
        if working_order.is_fok_order() and cancelled and not trades:
            rejected_reason = "FOK order could not be fully filled"

        result = OrderResult(
            copy_order(working_order),
            [copy_trade(trade) for trade in trades],
            rested,
            cancelled,
            rejected_reason,
        )
        return result, snapshot, submitted_order

    def _dispatch_order_events(
        self, result: OrderResult, snapshot: Optional[MarketDataSnapshot]
    ) -> None:
        """Deliver detached callback payloads after book mutation is complete."""
        if result.order is not None:
            self._trigger_order_callbacks(result.order, result.trades)
        if result.trades:
            self._trigger_trade_callbacks(result.trades)
        if snapshot is not None:
            self._trigger_market_data_callbacks(snapshot)

    def _snap_order_price(self, order: Order) -> None:
        """Snap a price-bearing order onto the book's tick grid before matching.

        Raises ValueError if the price snaps to a non-positive tick, so a
        sub-half-tick price cannot rest or execute at 0.0.
        """
        if int(order.order_type) == int(OrderType.MARKET):
            return
        if not math.isfinite(order.price):
            return  # non-finite prices are rejected by _validate_order
        tick = round(order.price / self.tick_size)
        snapped = round(tick * self.tick_size, self._price_decimals)
        if snapped <= 0:
            raise ValueError("Order price snaps to a non-positive tick")
        order.price = snapped

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
            return self._cancel_order_locked(order_id)

    def _cancel_order_locked(self, order_id: int, record: bool = True) -> bool:
        """Cancel an order while the caller holds ``self._lock``."""
        if isinstance(order_id, bool) or not isinstance(order_id, Integral) or order_id <= 0:
            raise ValueError(f"Order id must be a positive integer: {order_id!r}")
        order_id = int(order_id)
        success = self.matching_engine.cancel_order(order_id)
        if success:
            self.stats["orders_cancelled"] += 1
            if record and self._recorder is not None:
                self._recorder.record_cancel(order_id)
        return success

    def reduce_order(self, order_id: int, quantity: float) -> bool:
        """Reduce a resting order without changing its queue priority.

        ``quantity`` is the amount removed from the order, not its new absolute
        size. A full reduction removes the order. Invalid or excessive
        reductions raise ``ValueError`` and leave the book unchanged; a missing
        order returns ``False`` like :meth:`cancel_order`.
        """
        if isinstance(order_id, bool) or not isinstance(order_id, Integral) or order_id <= 0:
            raise ValueError(f"Order id must be a positive integer: {order_id!r}")
        quantity = self.order_factory._validate_quantity(quantity)

        with self._lock:
            order_id = int(order_id)
            order = self._get_order_unlocked(order_id)
            if order is None:
                return False
            if quantity > order.remaining_quantity + 1e-12:
                raise ValueError(
                    "Reduction quantity cannot exceed remaining quantity: "
                    f"{quantity} > {order.remaining_quantity}"
                )

            reduction = min(quantity, order.remaining_quantity)
            side_manager = (
                self.matching_engine.buy_side
                if int(order.side) == int(OrderSide.BUY)
                else self.matching_engine.sell_side
            )
            if not side_manager.update_order_quantity(order_id, reduction):
                return False
            if self._recorder is not None:
                self._recorder.record_reduce(order_id, reduction)
            return True

    def start_recording(self) -> EventLog:
        """Begin recording mutating operations into a fresh event log.

        Returns the log, which can later be serialized and replayed with
        `tracebook.core.replay.replay` to reconstruct identical trades and book
        state. Recording an already-recording book restarts the log.

        The book must be pristine: the log records only operations from this
        point on and cannot capture pre-existing trades, statistics, or resting
        liquidity. Call ``clear()`` before recording a previously used book.
        """
        with self._lock:
            if (
                self.stats["orders_added"]
                or self.matching_engine.trades
                or self.matching_engine.buy_side.orders
                or self.matching_engine.sell_side.orders
            ):
                raise ValueError(
                    "start_recording requires an empty book; a pristine book has no "
                    "pre-existing activity (call clear() before recording)"
                )
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
        timestamp: Optional[int] = None,
    ) -> OrderResult:
        """
        Replace a resting limit order by cancelling it and submitting a new order.

        Cancel-and-new is one locked transaction. Invalid input leaves the original
        untouched, the replacement receives a new id and timestamp, and callbacks
        are dispatched only after the transaction releases the book lock.
        """
        snapshot = None
        with self._lock:
            if isinstance(order_id, bool) or not isinstance(order_id, Integral) or order_id <= 0:
                return OrderResult(
                    None,
                    [],
                    False,
                    False,
                    f"Order id must be a positive integer: {order_id!r}",
                    accepted=False,
                )
            order_id = int(order_id)
            existing_order = self._get_order_unlocked(order_id)
            if existing_order is None:
                return OrderResult(None, [], False, False, "Order not found", accepted=False)

            replacement_price = existing_order.price if price is None else price
            replacement_quantity = (
                existing_order.remaining_quantity if quantity is None else quantity
            )
            if timestamp is not None and (
                isinstance(timestamp, bool) or not isinstance(timestamp, Integral) or timestamp < 0
            ):
                return OrderResult(
                    None,
                    [],
                    False,
                    False,
                    f"Order timestamp must be a non-negative integer: {timestamp!r}",
                    accepted=False,
                )

            try:
                replacement_order = self.order_factory.create_limit_order(
                    self.symbol,
                    OrderSide(existing_order.side),
                    replacement_price,
                    replacement_quantity,
                    existing_order.owner,
                )
                if timestamp is not None:
                    replacement_order.timestamp = int(timestamp)
                    replacement_order.priority = int(timestamp)
            except (TypeError, ValueError, OverflowError) as exc:
                # Invalid replacement: original order is left untouched.
                return OrderResult(None, [], False, False, str(exc), accepted=False)

            if not self._cancel_order_locked(order_id, record=False):
                return OrderResult(
                    None, [], False, False, "Order could not be cancelled", accepted=False
                )

            try:
                result, snapshot, submitted_replacement = self._process_order_locked(
                    replacement_order, validate=False, record=False
                )
            except Exception:
                self._restore_resting_order(existing_order)
                self.stats["orders_cancelled"] -= 1
                raise

            if self._recorder is not None:
                self._recorder.record_cancel(order_id)
                self._recorder.record_submit(submitted_replacement)

        self._dispatch_order_events(result, snapshot)
        return result

    def _restore_resting_order(self, order: Order) -> None:
        """Re-rest a previously cancelled order after a failed replacement."""
        if int(order.side) == int(OrderSide.BUY):
            self.matching_engine.buy_side.add_order(order)
        else:
            self.matching_engine.sell_side.add_order(order)

    def get_order(self, order_id: int) -> Optional[Order]:
        """Return a detached copy of a resting order, if it is active."""
        if isinstance(order_id, bool) or not isinstance(order_id, Integral) or order_id <= 0:
            raise ValueError(f"Order id must be a positive integer: {order_id!r}")
        with self._lock:
            order = self._get_order_unlocked(int(order_id))
            return copy_order(order) if order is not None else None

    def _get_order_unlocked(self, order_id: int) -> Optional[Order]:
        """Return the live internal order while the caller holds the book lock."""
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

            side_value = int(self.order_factory._validate_side(side))
            if side_value == int(OrderSide.BUY):
                return sorted(self.matching_engine.buy_side.orders.keys())
            if side_value == int(OrderSide.SELL):
                return sorted(self.matching_engine.sell_side.orders.keys())

            raise ValueError(f"Unsupported order side: {side}")

    def get_resting_orders(self, side: Optional[OrderSide] = None) -> List[Order]:
        """Return detached resting orders in matching-priority order.

        Bids are ordered from highest price to lowest and asks from lowest to
        highest. Orders at the same price retain their queue order. When
        ``side`` is omitted, all bids are returned before all asks; callers that
        need an explicit side boundary should request each side separately.
        """
        with self._lock:
            if side is None:
                managers = [
                    self.matching_engine.buy_side,
                    self.matching_engine.sell_side,
                ]
            else:
                side_value = int(self.order_factory._validate_side(side))
                managers = [
                    (
                        self.matching_engine.buy_side
                        if side_value == int(OrderSide.BUY)
                        else self.matching_engine.sell_side
                    )
                ]

            resting: List[Order] = []
            for manager in managers:
                for tick in manager.sorted_ticks:
                    level = manager.price_levels[tick]
                    for order_id in level.orders:
                        order = manager.orders.get(order_id)
                        if order is not None:
                            resting.append(copy_order(order))
            return resting

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
        with self._lock:
            bid = self.matching_engine.buy_side.get_best_price()
            ask = self.matching_engine.sell_side.get_best_price()
            if bid is not None and ask is not None:
                return ask - bid
            return None

    def get_mid_price(self) -> Optional[float]:
        """Get mid price."""
        with self._lock:
            bid = self.matching_engine.buy_side.get_best_price()
            ask = self.matching_engine.sell_side.get_best_price()
            if bid is not None and ask is not None:
                return (bid + ask) / 2.0
            return None

    def get_market_data_snapshot(self) -> MarketDataSnapshot:
        """Get current market data snapshot."""
        with self._lock:
            return self.matching_engine.get_market_data_snapshot()

    def get_order_book_depth(self, levels: int = 5) -> Dict[str, Any]:
        """Get order book depth."""
        levels = _validate_nonnegative_count(levels, "levels")
        with self._lock:
            return self.matching_engine.get_order_book_depth(levels)

    def get_recent_trades(self, count: int = 10) -> List[Trade]:
        """Get detached copies of recent trades."""
        if isinstance(count, bool) or not isinstance(count, Integral):
            raise ValueError("count must be an integer")
        count = int(count)
        if count <= 0:
            return []
        with self._lock:
            # self.matching_engine.trades is a bounded deque; materialize to slice.
            return [copy_trade(trade) for trade in list(self.matching_engine.trades)[-count:]]

    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        with self._lock:
            return self._get_statistics_unlocked()

    def _get_statistics_unlocked(self) -> Dict[str, Any]:
        """Return statistics while the caller holds the book lock."""
        book_stats = dict(self.stats)
        if book_stats["min_processing_time_ns"] == float("inf"):
            book_stats["min_processing_time_ns"] = 0
        return {
            **book_stats,
            **self.matching_engine.get_statistics(),
            "uptime_seconds": (time.time_ns() - self._start_time) / 1_000_000_000,
            "orders_per_second": self._calculate_orders_per_second(),
            "trades_per_second": self._calculate_trades_per_second(),
        }

    def get_state_snapshot(self, levels: int = 5, trade_count: int = 10) -> Dict[str, Any]:
        """Return one coherent, detached view of depth, trades, and statistics.

        All values are captured under a single lock acquisition so live UIs do not
        combine depth from one book state with top-of-book values from another.
        """
        levels = _validate_nonnegative_count(levels, "levels")
        trade_count = _validate_nonnegative_count(trade_count, "trade_count")

        with self._lock:
            market = self.matching_engine.get_market_data_snapshot()
            trades = (
                [copy_trade(trade) for trade in list(self.matching_engine.trades)[-trade_count:]]
                if trade_count
                else []
            )
            return {
                "symbol": self.symbol,
                "tick_size": self.tick_size,
                "timestamp": market.timestamp,
                "best_bid": market.best_bid,
                "best_ask": market.best_ask,
                "mid": market.mid_price,
                "spread": market.spread,
                "bids": list(market.bid_levels[:levels]),
                "asks": list(market.ask_levels[:levels]),
                "trades": trades,
                "statistics": self._get_statistics_unlocked(),
            }

    def register_trade_callback(self, callback):
        """Register callback for trade events."""
        if not callable(callback):
            raise ValueError("trade callback must be callable")
        with self._lock:
            self._trade_callbacks.append(callback)

    def register_order_callback(self, callback):
        """Register callback for order events."""
        if not callable(callback):
            raise ValueError("order callback must be callable")
        with self._lock:
            self._order_callbacks.append(callback)

    def register_market_data_callback(self, callback):
        """Register callback for market data events."""
        if not callable(callback):
            raise ValueError("market data callback must be callable")
        with self._lock:
            self._market_data_callbacks.append(callback)

    def set_snapshot_interval(self, interval_ms: float):
        """Set market data snapshot interval in milliseconds."""
        if isinstance(interval_ms, bool) or not isinstance(interval_ms, Real):
            raise ValueError("snapshot interval must be positive and finite")
        interval_ms = float(interval_ms)
        if not math.isfinite(interval_ms) or interval_ms <= 0:
            raise ValueError("snapshot interval must be positive and finite")
        with self._lock:
            self._snapshot_interval_ns = int(interval_ms * 1_000_000)

    def _normalize_external_order(self, order: Order) -> Order:
        """Validate an external order and return an engine-owned normalized copy."""
        if not isinstance(order, Order):
            raise ValueError(f"Expected an Order instance: {order!r}")

        order_symbol = normalize_symbol(order.symbol)
        if order_symbol != self.symbol:
            raise ValueError(
                f"Order symbol {order.symbol!r} does not match book symbol {self.symbol!r}"
            )

        if isinstance(order.order_id, bool) or not isinstance(order.order_id, Integral):
            raise ValueError(f"Order id must be a positive integer: {order.order_id!r}")
        order_id = int(order.order_id)
        if order_id <= 0 or order_id >= 2**63:
            raise ValueError("Order id must be a positive int64")

        if self._get_order_unlocked(order_id) is not None:
            raise ValueError(f"Order id {order.order_id} is already active")
        if order_id in self._seen_order_ids:
            raise ValueError(f"Order id {order.order_id} has already been processed")

        side = self.order_factory._validate_side(order.side)
        order_type = self.order_factory._validate_order_type(order.order_type)
        quantity = self.order_factory._validate_quantity(order.quantity)
        owner = self.order_factory._validate_owner(order.owner)

        if isinstance(order.remaining_quantity, bool) or not isinstance(
            order.remaining_quantity, Real
        ):
            raise ValueError(
                f"Order remaining quantity must be numeric: {order.remaining_quantity!r}"
            )
        remaining_quantity = float(order.remaining_quantity)
        if (
            not math.isfinite(remaining_quantity)
            or remaining_quantity <= 1e-12
            or remaining_quantity > quantity + 1e-12
        ):
            raise ValueError("Order quantity must be positive")

        if isinstance(order.price, bool) or not isinstance(order.price, Real):
            raise ValueError(f"Order price must be numeric: {order.price!r}")
        price = float(order.price)
        if not math.isfinite(price):
            raise ValueError("Order price must be finite")
        if order_type == OrderType.MARKET:
            if price != 0.0:
                raise ValueError("Market orders must use price 0.0")
        elif price <= 0:
            raise ValueError("Limit-style orders must have a positive price")

        if isinstance(order.timestamp, bool) or not isinstance(order.timestamp, Integral):
            raise ValueError(f"Order timestamp must be a non-negative integer: {order.timestamp!r}")
        if order.timestamp < 0:
            raise ValueError("Order timestamp must be a non-negative integer")

        normalized = Order(
            order_id=order_id,
            symbol=order_symbol,
            side=int(side),
            order_type=int(order_type),
            price=price,
            quantity=quantity,
            timestamp=int(order.timestamp),
            owner=owner,
        )
        normalized.remaining_quantity = remaining_quantity
        return normalized

    def clear(self):
        """Clear all orders and reset statistics."""
        with self._lock:
            if self._recorder is not None:
                self._recorder.record_clear()
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
            self._last_snapshot_time = 0
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
        with self._lock:
            callbacks = list(self._trade_callbacks)
        for callback in callbacks:
            try:
                callback([copy_trade(trade) for trade in trades])
            except Exception as e:
                # Log error but don't let callback failures affect order processing
                print(f"Trade callback error: {e}")

    def _trigger_order_callbacks(self, order: Order, trades: List[Trade]):
        """Trigger order event callbacks."""
        with self._lock:
            callbacks = list(self._order_callbacks)
        for callback in callbacks:
            try:
                callback(copy_order(order), [copy_trade(trade) for trade in trades])
            except Exception as e:
                print(f"Order callback error: {e}")

    def _trigger_market_data_callbacks(self, snapshot: MarketDataSnapshot):
        """Trigger market data callbacks."""
        with self._lock:
            callbacks = list(self._market_data_callbacks)
        for callback in callbacks:
            try:
                callback(snapshot.copy())
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

    order_books: Dict[str, OrderBook]
    _lock: Any

    def __init__(self) -> None:
        self.order_books: Dict[str, OrderBook] = {}
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
            order_books = list(self.order_books.values())

        # Do not hold the manager lock while acquiring individual book locks.
        # This avoids lock-order inversion for callers that already hold a book.
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

        for order_book in order_books:
            stats = order_book.get_statistics()

            for key in summed_keys:
                aggregate[key] += stats.get(key, 0)

            max_processing_time = max(max_processing_time, stats.get("max_processing_time_ns", 0))
            orders_added = stats.get("orders_added", 0)
            if orders_added:
                min_processing_time = min(
                    min_processing_time,
                    stats.get("min_processing_time_ns", float("inf")),
                )
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
            order_books = list(self.order_books.values())
        for order_book in order_books:
            order_book.clear()

    def remove_order_book(self, symbol: str) -> bool:
        """Remove an order book."""
        symbol = normalize_symbol(symbol)
        with self._lock:
            if symbol in self.order_books:
                del self.order_books[symbol]
                return True
            return False
