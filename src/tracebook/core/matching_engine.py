"""
Order matching engine.

Implements the core FIFO and pro-rata matching logic in plain Python over
dict-backed price levels. (Earlier versions used Numba; profiling showed the
JIT wrapper was a net cost when driven from the Python matching loop -- see
docs/performance.md.)
"""

from collections import deque
from typing import Deque, List, Optional, Tuple
import time
from .order import (
    NO_OWNER,
    Order,
    SelfTradePolicy,
    Trade,
    calculate_match_quantity,
    orders_can_match,
)
from .price_level import PriceLevelManager, MarketDataSnapshot

EPSILON = 1e-12


class MatchingEngine:
    """
    Matching engine supporting multiple algorithms.

    Features:
    - FIFO and pro-rata matching
    - Integer-tick price levels with O(1) in-level insert/removal
    - Optional self-trade prevention
    - Nanosecond-precision timing
    """

    def __init__(
        self,
        symbol: str,
        matching_algorithm: str = "fifo",
        tick_size: float = 0.01,
        self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
    ):
        self.symbol = symbol
        self.matching_algorithm = matching_algorithm.lower()
        self.tick_size = tick_size
        self.self_trade_policy = SelfTradePolicy(self_trade_policy)

        # Price level managers for each side
        self.buy_side = PriceLevelManager(is_buy_side=True, tick_size=tick_size)
        self.sell_side = PriceLevelManager(is_buy_side=False, tick_size=tick_size)

        # Trade history: bounded so a long-running book doesn't leak memory.
        # get_recent_trades reads the tail; total_matches tracks the true total.
        self._trade_history_cap = 10000
        self.trades: Deque[Trade] = deque(maxlen=self._trade_history_cap)
        self.trade_count = 0

        # Performance metrics
        self.total_orders_processed = 0
        self.total_matches = 0
        self.self_trades_prevented = 0
        self.last_trade_time = 0

        # Validate matching algorithm
        if self.matching_algorithm not in ["fifo", "pro_rata"]:
            raise ValueError(f"Unsupported matching algorithm: {matching_algorithm}")

    def _is_self_trade(self, incoming_order: Order, resting_order: Order) -> bool:
        """Return True if the two orders share a non-anonymous owner."""
        if self.self_trade_policy == SelfTradePolicy.NONE:
            return False
        return incoming_order.owner != NO_OWNER and incoming_order.owner == resting_order.owner

    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book and execute any possible matches.

        Args:
            order: Order to add

        Returns:
            List[Trade]: List of executed trades
        """
        self.total_orders_processed += 1
        trades: List[Trade] = []

        # FOK orders must be completely executable before touching resting liquidity.
        if order.is_fok_order() and not self._can_fully_fill(order):
            return trades

        if order.is_buy():
            trades = self._match_buy_order(order)
        else:
            trades = self._match_sell_order(order)

        # Add remaining quantity to book if not fully filled
        if order.remaining_quantity > EPSILON and order.can_rest():
            if order.is_buy():
                self.buy_side.add_order(order)
            else:
                self.sell_side.add_order(order)

        # Update metrics
        self.total_matches += len(trades)
        if trades:
            self.last_trade_time = trades[-1].timestamp

        return trades

    def cancel_order(self, order_id: int) -> bool:
        """
        Cancel an order by ID.

        Args:
            order_id: ID of order to cancel

        Returns:
            bool: True if order was found and cancelled
        """
        # Try buy side first
        if self.buy_side.remove_order(order_id):
            return True

        # Try sell side
        return self.sell_side.remove_order(order_id)

    def _match_buy_order(self, buy_order: Order) -> List[Trade]:
        """Match a buy order against sell side."""
        trades = []

        while buy_order.remaining_quantity > EPSILON:
            best_sell_level = self.sell_side.get_best_price_level()
            if best_sell_level is None:
                break

            # Check if prices can match
            if not buy_order.can_match_price(best_sell_level.price):
                break

            # Execute matches at this price level
            level_trades, progressed = self._execute_matches_at_level(
                buy_order, best_sell_level, self.sell_side
            )
            trades.extend(level_trades)

            # No trade and nothing cancelled here means no further progress is
            # possible (guards against looping when a level only holds own orders).
            if not progressed:
                break

        return trades

    def _match_sell_order(self, sell_order: Order) -> List[Trade]:
        """Match a sell order against buy side."""
        trades = []

        while sell_order.remaining_quantity > EPSILON:
            best_buy_level = self.buy_side.get_best_price_level()
            if best_buy_level is None:
                break

            # Check if prices can match
            if not sell_order.can_match_price(best_buy_level.price):
                break

            # Execute matches at this price level
            level_trades, progressed = self._execute_matches_at_level(
                sell_order, best_buy_level, self.buy_side
            )
            trades.extend(level_trades)

            if not progressed:
                break

        return trades

    def _execute_matches_at_level(
        self, incoming_order: Order, price_level, side_manager
    ) -> Tuple[List[Trade], bool]:
        """Execute matches at a price level. Returns (trades, progressed).

        `progressed` is True if any trade executed or any resting order was
        cancelled by self-trade prevention, so a level that only holds the
        incoming owner's orders does not stall the outer matching loop.
        """
        if self.matching_algorithm == "pro_rata":
            return self._execute_pro_rata_matches_at_level(
                incoming_order, price_level, side_manager
            )

        trades = []
        progressed = False

        # Iterate by repeatedly taking the FIFO head (O(1)) rather than copying
        # the whole level per aggressive order. A matched head is removed by
        # update_order_quantity when fully filled, advancing the head; a partial
        # fill of the incoming order exits the loop with the head still resting.
        while incoming_order.remaining_quantity > EPSILON:
            order_id = price_level.get_first_order_id()
            if order_id == -1:
                break

            resting_order = side_manager.get_order(order_id)
            if resting_order is None or resting_order.remaining_quantity <= EPSILON:
                # Stale/spent head (unreachable under the level/orders invariant):
                # evict it, cleaning the index if the level empties, then keep
                # matching the depth behind it rather than stalling at this price.
                side_manager.discard_from_level(order_id, price_level)
                progressed = True
                continue

            if self._is_self_trade(incoming_order, resting_order):
                if self.self_trade_policy == SelfTradePolicy.CANCEL_RESTING:
                    side_manager.remove_order(order_id)
                    self.self_trades_prevented += 1
                    progressed = True
                    continue
                # CANCEL_INCOMING: drop the aggressor's remainder and stop here
                # so it never rests into a self-cross.
                self.self_trades_prevented += 1
                incoming_order.remaining_quantity = 0.0
                progressed = True
                break

            trade = self._execute_fifo_match(incoming_order, resting_order)
            if trade is None:
                break  # no progress possible (guards against a stuck head)

            trades.append(trade)
            fill_qty = trade.quantity
            incoming_order.fill(fill_qty)
            side_manager.update_order_quantity(order_id, fill_qty)
            progressed = True

        return trades, progressed

    def _execute_pro_rata_matches_at_level(
        self, incoming_order: Order, price_level, side_manager
    ) -> Tuple[List[Trade], bool]:
        """Allocate the incoming order proportionally across resting orders at a level.

        Returns (trades, progressed). Same-owner resting orders are excluded from
        the allocation under self-trade prevention (and cancelled under
        CANCEL_RESTING); CANCEL_INCOMING drops the aggressor's remainder once it
        has contacted any same-owner order at the level.
        """
        trades: List[Trade] = []
        progressed = False
        resting_orders = []
        self_trade_seen = False

        for order_id in list(price_level.orders):
            resting_order = side_manager.get_order(order_id)
            if resting_order is None or resting_order.remaining_quantity <= EPSILON:
                continue

            if not orders_can_match(
                incoming_order if incoming_order.is_buy() else resting_order,
                resting_order if resting_order.is_sell() else incoming_order,
            ):
                continue

            if self._is_self_trade(incoming_order, resting_order):
                self_trade_seen = True
                if self.self_trade_policy == SelfTradePolicy.CANCEL_RESTING:
                    side_manager.remove_order(order_id)
                    self.self_trades_prevented += 1
                    progressed = True
                # Either way the own order is excluded from the allocation.
                continue

            resting_orders.append(resting_order)

        available_quantity = sum(order.remaining_quantity for order in resting_orders)
        if available_quantity <= EPSILON:
            if self_trade_seen and self.self_trade_policy == SelfTradePolicy.CANCEL_INCOMING:
                self.self_trades_prevented += 1
                incoming_order.remaining_quantity = 0.0
                progressed = True
            return trades, progressed

        total_fill = min(incoming_order.remaining_quantity, available_quantity)
        remaining_fill = total_fill

        for index, resting_order in enumerate(resting_orders):
            if incoming_order.remaining_quantity <= EPSILON or remaining_fill <= EPSILON:
                break

            if index == len(resting_orders) - 1:
                fill_qty = min(resting_order.remaining_quantity, remaining_fill)
            else:
                allocation = total_fill * (resting_order.remaining_quantity / available_quantity)
                fill_qty = min(resting_order.remaining_quantity, allocation, remaining_fill)

            if fill_qty <= EPSILON:
                continue

            if incoming_order.is_buy():
                buy_order_id = incoming_order.order_id
                sell_order_id = resting_order.order_id
            else:
                buy_order_id = resting_order.order_id
                sell_order_id = incoming_order.order_id

            # Matches always execute at the resting (maker) order's price. When a
            # market order is involved the resting side is always the limit order,
            # so this yields the same value the old price-time rule produced --
            # the book now has a single, consistent execution-price rule.
            execution_price = resting_order.price

            trade = Trade(
                buy_order_id=buy_order_id,
                sell_order_id=sell_order_id,
                price=execution_price,
                quantity=fill_qty,
                timestamp=time.time_ns(),
            )
            self.trades.append(trade)
            trades.append(trade)

            incoming_order.fill(fill_qty)
            side_manager.update_order_quantity(resting_order.order_id, fill_qty)
            remaining_fill -= fill_qty
            progressed = True

        # Having contacted a same-owner order, CANCEL_INCOMING drops the remainder
        # after allocating to everyone else at this level.
        if self_trade_seen and self.self_trade_policy == SelfTradePolicy.CANCEL_INCOMING:
            self.self_trades_prevented += 1
            incoming_order.remaining_quantity = 0.0
            progressed = True

        return trades, progressed

    def _execute_fifo_match(self, incoming_order: Order, resting_order: Order) -> Optional[Trade]:
        """Execute a FIFO match between two orders."""
        if not orders_can_match(
            incoming_order if incoming_order.is_buy() else resting_order,
            resting_order if resting_order.is_sell() else incoming_order,
        ):
            return None

        buy_order = incoming_order if incoming_order.is_buy() else resting_order
        sell_order = resting_order if resting_order.is_sell() else incoming_order

        # Executions use the resting order's price. This keeps matching
        # deterministic even when caller-supplied timestamps are unusual.
        execution_price = resting_order.price
        execution_quantity = calculate_match_quantity(buy_order, sell_order)

        if incoming_order.is_buy():
            buy_order_id = incoming_order.order_id
            sell_order_id = resting_order.order_id
        else:
            buy_order_id = resting_order.order_id
            sell_order_id = incoming_order.order_id

        if execution_quantity > EPSILON:
            trade = Trade(
                buy_order_id=buy_order_id,
                sell_order_id=sell_order_id,
                price=execution_price,
                quantity=execution_quantity,
                timestamp=time.time_ns(),
            )
            self.trades.append(trade)
            return trade

        return None

    def _can_fully_fill(self, order: Order) -> bool:
        """Return True if the visible opposite book can fill the full order.

        The accounting mirrors how each self-trade policy actually matches so a
        FOK is never misreported as fillable:
        - NONE: all resting quantity counts.
        - CANCEL_RESTING: the order's own resting quantity is skipped (it would
          be cancelled), and matching continues past it.
        - CANCEL_INCOMING: FIFO matching halts at the first same-owner order;
          pro-rata can use all non-self liquidity at that price level before the
          incoming remainder is cancelled.
        """
        remaining_needed = order.remaining_quantity
        side_manager = self.sell_side if order.is_buy() else self.buy_side
        policy = self.self_trade_policy
        owner_aware = policy != SelfTradePolicy.NONE and order.owner != NO_OWNER

        for tick in list(side_manager.sorted_ticks):
            level = side_manager.price_levels[tick]
            if not order.can_match_price(level.price):
                break

            if not owner_aware:
                remaining_needed -= level.total_quantity
                if remaining_needed <= EPSILON:
                    return True
                continue

            # Pro-rata considers every resting order at a level together. Under
            # CANCEL_INCOMING it allocates against all non-self liquidity at that
            # level, then cancels only an unfilled remainder if self liquidity was
            # present. FOK preflight must mirror that level-wide behavior rather
            # than inheriting FIFO insertion-order semantics.
            if self.matching_algorithm == "pro_rata" and policy == SelfTradePolicy.CANCEL_INCOMING:
                self_trade_seen = False
                for resting_id in level.orders:
                    resting = side_manager.get_order(resting_id)
                    if resting is None:
                        continue
                    if resting.owner == order.owner:
                        self_trade_seen = True
                    else:
                        remaining_needed -= resting.remaining_quantity
                if remaining_needed <= EPSILON:
                    return True
                if self_trade_seen:
                    return False
                continue

            for resting_id in level.orders:
                resting = side_manager.get_order(resting_id)
                if resting is None:
                    continue
                if resting.owner == order.owner:
                    if policy == SelfTradePolicy.CANCEL_INCOMING:
                        # Matching would stop here; nothing beyond is reachable.
                        return False
                    # CANCEL_RESTING: own order is skipped, keep scanning.
                    continue
                remaining_needed -= resting.remaining_quantity
                if remaining_needed <= EPSILON:
                    return True

        return False

    def can_fully_fill(self, order: Order) -> bool:
        """Return True if an incoming order could fully fill immediately."""
        return self._can_fully_fill(order)

    def get_market_data_snapshot(self) -> MarketDataSnapshot:
        """Get current market data snapshot."""
        snapshot = MarketDataSnapshot(self.symbol, time.time_ns())

        # Get bid levels (buy side)
        snapshot.set_bid_levels(self.buy_side.get_price_levels_snapshot())

        # Get ask levels (sell side)
        snapshot.set_ask_levels(self.sell_side.get_price_levels_snapshot())

        # Calculate derived metrics
        snapshot.calculate_derived_metrics()

        return snapshot

    def get_order_book_depth(self, levels: int = 5) -> dict:
        """Get order book depth up to specified levels."""
        if levels < 0:
            raise ValueError("levels must be non-negative")

        bid_levels = self.buy_side.get_price_levels_snapshot()[:levels]
        ask_levels = self.sell_side.get_price_levels_snapshot()[:levels]

        return {
            "symbol": self.symbol,
            "timestamp": time.time_ns(),
            "bids": bid_levels,
            "asks": ask_levels,
        }

    def get_statistics(self) -> dict:
        """Get matching engine statistics."""
        return {
            "symbol": self.symbol,
            "matching_algorithm": self.matching_algorithm,
            "self_trade_policy": self.self_trade_policy.name,
            "total_orders_processed": self.total_orders_processed,
            "total_matches": self.total_matches,
            "total_trades": self.total_matches,  # cumulative; self.trades is a bounded tail
            "self_trades_prevented": self.self_trades_prevented,
            "buy_side_orders": self.buy_side.get_total_orders(),
            "sell_side_orders": self.sell_side.get_total_orders(),
            "buy_side_quantity": self.buy_side.get_total_quantity(),
            "sell_side_quantity": self.sell_side.get_total_quantity(),
            "last_trade_time": self.last_trade_time,
        }

    def clear(self):
        """Clear all orders and trades."""
        self.buy_side.clear()
        self.sell_side.clear()
        self.trades.clear()
        self.trade_count = 0
        self.total_orders_processed = 0
        self.total_matches = 0
        self.self_trades_prevented = 0
        self.last_trade_time = 0
