"""
Price level management for high-performance order book.

This module implements cache-friendly data structures for managing
orders at each price level with minimal memory allocations.
"""

import numpy as np
from numba import types, typed
from numba.experimental import jitclass
from typing import List, Optional
from .order import Order, Trade


# Price level specification for Numba JIT
price_level_spec = [
    ('price', types.float64),
    ('total_quantity', types.int64),
    ('order_count', types.int32),
    ('orders', types.ListType(types.int64)),  # Store order IDs for memory efficiency
]


@jitclass(price_level_spec)
class PriceLevel:
    """
    Represents all orders at a specific price level.
    
    Optimized for:
    - Fast insertion/removal of orders
    - Efficient quantity tracking
    - Minimal memory allocations
    - Cache-friendly access patterns
    """
    
    def __init__(self, price):
        self.price = price
        self.total_quantity = 0
        self.order_count = 0
        self.orders = typed.List.empty_list(types.int64)
    
    def add_order(self, order_id, quantity):
        """Add an order to this price level."""
        self.orders.append(order_id)
        self.total_quantity += quantity
        self.order_count += 1
    
    def remove_order(self, order_id, quantity):
        """Remove an order from this price level."""
        # Find and remove the order ID
        for i in range(len(self.orders)):
            if self.orders[i] == order_id:
                self.orders.pop(i)
                break
        
        self.total_quantity -= quantity
        self.order_count -= 1
    
    def update_quantity(self, quantity_change):
        """Update total quantity (for partial fills)."""
        self.total_quantity += quantity_change
    
    def is_empty(self):
        """Check if price level has no orders."""
        return self.order_count == 0
    
    def get_first_order_id(self):
        """Get the first order ID (FIFO)."""
        if self.order_count > 0:
            return self.orders[0]
        return -1


class PriceLevelManager:
    """
    Manages price levels for one side of the order book.
    
    Uses sorted arrays for efficient price-based operations
    and maintains separate order storage for memory efficiency.
    """
    
    def __init__(self, is_buy_side: bool):
        self.is_buy_side = is_buy_side
        self.price_levels = {}  # price -> PriceLevel
        self.sorted_prices = []  # Maintained in sorted order
        self.orders = {}  # order_id -> Order (shared storage)
    
    def add_order(self, order: Order):
        """Add an order to the appropriate price level."""
        price = order.price
        
        # Create price level if it doesn't exist
        if price not in self.price_levels:
            self.price_levels[price] = PriceLevel(price)
            self._insert_price_sorted(price)
        
        # Add order to price level and storage
        self.price_levels[price].add_order(order.order_id, order.remaining_quantity)
        self.orders[order.order_id] = order
    
    def remove_order(self, order_id: int):
        """Remove an order completely."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        price = order.price
        
        if price in self.price_levels:
            price_level = self.price_levels[price]
            price_level.remove_order(order_id, order.remaining_quantity)
            
            # Remove empty price level
            if price_level.is_empty():
                del self.price_levels[price]
                self.sorted_prices.remove(price)
        
        del self.orders[order_id]
        return True
    
    def update_order_quantity(self, order_id: int, quantity_filled: int):
        """Update order quantity after partial fill."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        price = order.price
        
        # Update order
        order.remaining_quantity -= quantity_filled
        
        # Update price level
        if price in self.price_levels:
            self.price_levels[price].update_quantity(-quantity_filled)
        
        # Remove if fully filled
        if order.remaining_quantity == 0:
            self.remove_order(order_id)
        
        return True
    
    def get_best_price(self) -> Optional[float]:
        """Get the best price (highest for buy, lowest for sell)."""
        if not self.sorted_prices:
            return None
        
        if self.is_buy_side:
            return self.sorted_prices[-1]  # Highest price
        else:
            return self.sorted_prices[0]   # Lowest price
    
    def get_best_price_level(self) -> Optional[PriceLevel]:
        """Get the price level with the best price."""
        best_price = self.get_best_price()
        if best_price is not None:
            return self.price_levels[best_price]
        return None
    
    def get_orders_at_price(self, price: float) -> List[int]:
        """Get all order IDs at a specific price."""
        if price in self.price_levels:
            return list(self.price_levels[price].orders)
        return []
    
    def get_order(self, order_id: int) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_total_quantity(self) -> int:
        """Get total quantity across all price levels."""
        return sum(level.total_quantity for level in self.price_levels.values())
    
    def get_total_orders(self) -> int:
        """Get total number of orders."""
        return len(self.orders)
    
    def get_price_levels_snapshot(self) -> List[tuple]:
        """Get snapshot of all price levels (price, quantity, count)."""
        result = []
        for price in self.sorted_prices:
            level = self.price_levels[price]
            result.append((price, level.total_quantity, level.order_count))
        return result
    
    def _insert_price_sorted(self, price: float):
        """Insert price into sorted list maintaining order."""
        if self.is_buy_side:
            # Buy side: highest price first
            for i, existing_price in enumerate(self.sorted_prices):
                if price > existing_price:
                    self.sorted_prices.insert(i, price)
                    return
            self.sorted_prices.append(price)
        else:
            # Sell side: lowest price first
            for i, existing_price in enumerate(self.sorted_prices):
                if price < existing_price:
                    self.sorted_prices.insert(i, price)
                    return
            self.sorted_prices.append(price)
    
    def clear(self):
        """Clear all orders and price levels."""
        self.price_levels.clear()
        self.sorted_prices.clear()
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
            'symbol': self.symbol,
            'timestamp': self.timestamp,
            'bid_levels': self.bid_levels,
            'ask_levels': self.ask_levels,
            'best_bid': self.best_bid,
            'best_ask': self.best_ask,
            'spread': self.spread,
            'mid_price': self.mid_price,
        }
