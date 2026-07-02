"""
Price level management for high-performance order book.

This module implements cache-friendly data structures for managing
orders at each price level with minimal memory allocations.
"""

import bisect
import math
from decimal import Decimal

from typing import List, Optional
from .order import Order


def infer_price_decimals(tick_size: float) -> int:
    """Return the number of decimal places implied by a tick size (0.01 -> 2).

    Uses the tick's exact decimal representation so fine grids (e.g. 1e-6 or
    smaller) are not truncated to whole numbers, which would collapse distinct
    price levels onto the same canonical price.
    """
    exponent = Decimal(str(tick_size)).normalize().as_tuple().exponent
    return -exponent if exponent < 0 else 0


class PriceLevel:
    """
    Represents all orders at a specific price level.

    Order ids are held in an insertion-ordered dict used as an ordered set, so
    removal and FIFO-head lookup are O(1) while price-time (insertion) order is
    preserved. `orders` iterates order ids in arrival order, exactly like the
    previous list did, so consumers are unaffected.
    """

    def __init__(self, price):
        self.price = price
        self.total_quantity = 0.0
        self.order_count = 0
        self.orders = {}  # order_id -> True, insertion-ordered (FIFO)

    def add_order(self, order_id, quantity):
        """Add an order to this price level."""
        self.orders[order_id] = True
        self.total_quantity += quantity
        self.order_count += 1

    def remove_order(self, order_id, quantity):
        """Remove an order from this price level in O(1)."""
        if order_id in self.orders:
            del self.orders[order_id]
            self.total_quantity -= quantity
            self.order_count -= 1
            return True

        return False

    def update_quantity(self, quantity_change):
        """Update total quantity (for partial fills)."""
        self.total_quantity += quantity_change

    def is_empty(self):
        """Check if price level has no orders."""
        return self.order_count == 0

    def get_first_order_id(self):
        """Get the first order ID (FIFO) in O(1)."""
        if self.order_count > 0:
            return next(iter(self.orders))
        return -1


class PriceLevelManager:
    """
    Manages price levels for one side of the order book.

    Price levels are keyed by integer ticks rather than raw floats. Two prices
    that round to the same tick (e.g. 100.0 and 100.00000000001) therefore share
    a single level, which removes the float-identity hazard of dict-keying on
    prices. Resting orders are snapped onto the canonical grid price for their
    tick so every consumer (execution price, snapshots) sees one value per level.
    """

    def __init__(self, is_buy_side: bool, tick_size: float = 0.01):
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ValueError("tick_size must be a positive, finite number")
        self.is_buy_side = is_buy_side
        self.tick_size = tick_size
        self._price_decimals = infer_price_decimals(tick_size)
        self.price_levels = {}  # tick (int) -> PriceLevel
        self.sorted_ticks = []  # ticks in book order (buy: desc, sell: asc)
        self.orders = {}  # order_id -> Order (shared storage)
        # Bisect key that keeps sorted_ticks best-first for this side: buy is
        # descending (negate), sell is ascending (identity). Used for O(log n)
        # index lookups instead of a linear scan.
        self._tick_key = (lambda t: -t) if is_buy_side else (lambda t: t)

    def price_to_tick(self, price: float) -> int:
        """Map a price onto the integer tick grid (round to nearest tick)."""
        return int(round(price / self.tick_size))

    def tick_to_price(self, tick: int) -> float:
        """Map a tick back to its canonical grid price, free of FP dust."""
        return round(tick * self.tick_size, self._price_decimals)

    def add_order(self, order: Order):
        """Add an order to the appropriate price level."""
        tick = self.price_to_tick(order.price)
        # Snap the resting order onto the canonical grid for its tick.
        order.price = self.tick_to_price(tick)

        if tick not in self.price_levels:
            self.price_levels[tick] = PriceLevel(order.price)
            self._insert_tick_sorted(tick)

        self.price_levels[tick].add_order(order.order_id, order.remaining_quantity)
        self.orders[order.order_id] = order

    def discard_from_level(self, order_id: int, price_level: "PriceLevel"):
        """Evict a stale order id from a known level, cleaning up an emptied level.

        Recovery helper for a level entry whose order is missing or spent (which
        should not happen under the level/orders invariant). Safe if the order is
        already gone from either structure.
        """
        order = self.orders.pop(order_id, None)
        if order is not None:
            price_level.remove_order(order_id, order.remaining_quantity)
        else:
            # The order object is gone, so its quantity is unknown. Remove the id
            # and recompute the level total from the orders that remain, so the
            # cached total is not left overstated.
            price_level.remove_order(order_id, 0.0)
            price_level.total_quantity = sum(
                self.orders[oid].remaining_quantity
                for oid in price_level.orders
                if oid in self.orders
            )
        if price_level.is_empty():
            tick = self.price_to_tick(price_level.price)
            if tick in self.price_levels:
                del self.price_levels[tick]
                self._remove_tick_sorted(tick)

    def remove_order(self, order_id: int):
        """Remove an order completely."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        tick = self.price_to_tick(order.price)

        if tick in self.price_levels:
            price_level = self.price_levels[tick]
            removed = price_level.remove_order(order_id, order.remaining_quantity)
            if not removed:
                return False

            # Remove empty price level
            if price_level.is_empty():
                del self.price_levels[tick]
                self._remove_tick_sorted(tick)

        del self.orders[order_id]
        return True

    def update_order_quantity(self, order_id: int, quantity_filled: float):
        """Update order quantity after partial fill."""
        if order_id not in self.orders:
            return False

        order = self.orders[order_id]
        tick = self.price_to_tick(order.price)

        # Update order
        order.remaining_quantity -= quantity_filled

        # Update price level
        if tick in self.price_levels:
            self.price_levels[tick].update_quantity(-quantity_filled)

        # Remove if fully filled
        if order.remaining_quantity <= 1e-12:
            order.remaining_quantity = 0.0
            self.remove_order(order_id)

        return True

    def get_best_price(self) -> Optional[float]:
        """Get the best price (highest for buy, lowest for sell)."""
        if not self.sorted_ticks:
            return None
        # sorted_ticks[0] is the best tick for either side by construction.
        return self.tick_to_price(self.sorted_ticks[0])

    def get_best_price_level(self) -> Optional[PriceLevel]:
        """Get the price level with the best price."""
        if not self.sorted_ticks:
            return None
        return self.price_levels[self.sorted_ticks[0]]

    def get_orders_at_price(self, price: float) -> List[int]:
        """Get all order IDs at a specific price."""
        tick = self.price_to_tick(price)
        if tick in self.price_levels:
            return list(self.price_levels[tick].orders)
        return []

    def get_order(self, order_id: int) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_total_quantity(self) -> float:
        """Get total quantity across all price levels."""
        return sum(level.total_quantity for level in self.price_levels.values())

    def get_total_orders(self) -> int:
        """Get total number of orders."""
        return len(self.orders)

    def get_price_levels_snapshot(self) -> List[tuple]:
        """Get snapshot of all price levels (price, quantity, count)."""
        result = []
        for tick in self.sorted_ticks:
            level = self.price_levels[tick]
            result.append((level.price, level.total_quantity, level.order_count))
        return result

    def _insert_tick_sorted(self, tick: int):
        """Insert a tick into the sorted list maintaining best-first order."""
        bisect.insort(self.sorted_ticks, tick, key=self._tick_key)

    def _remove_tick_sorted(self, tick: int):
        """Remove a tick from the sorted list via binary search (best-first)."""
        index = bisect.bisect_left(self.sorted_ticks, self._tick_key(tick), key=self._tick_key)
        if index < len(self.sorted_ticks) and self.sorted_ticks[index] == tick:
            self.sorted_ticks.pop(index)

    def clear(self):
        """Clear all orders and price levels."""
        self.price_levels.clear()
        self.sorted_ticks.clear()
        self.orders.clear()


class MarketDataSnapshot:
    """
    Immutable snapshot of market data for a specific point in time.

    Used for analytics and visualization without affecting
    the performance of the main order book.
    """

    def __init__(self, symbol: str, timestamp: int):
        self.symbol = symbol
        self.timestamp = timestamp
        self.bid_levels = []  # [(price, quantity, count), ...]
        self.ask_levels = []  # [(price, quantity, count), ...]
        self.best_bid = None
        self.best_ask = None
        self.spread = None
        self.mid_price = None

    def set_bid_levels(self, levels: List[tuple]):
        """Set bid price levels."""
        self.bid_levels = levels
        if levels:
            self.best_bid = levels[0][0]

    def set_ask_levels(self, levels: List[tuple]):
        """Set ask price levels."""
        self.ask_levels = levels
        if levels:
            self.best_ask = levels[0][0]

    def calculate_derived_metrics(self):
        """Calculate spread and mid-price."""
        if self.best_bid is not None and self.best_ask is not None:
            self.spread = self.best_ask - self.best_bid
            self.mid_price = (self.best_bid + self.best_ask) / 2.0

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp,
            "bid_levels": self.bid_levels,
            "ask_levels": self.ask_levels,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "spread": self.spread,
            "mid_price": self.mid_price,
        }
