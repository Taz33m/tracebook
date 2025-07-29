"""
High-performance order data structures optimized for Numba JIT compilation.
"""

import numpy as np
from numba import types
from numba.experimental import jitclass
from enum import IntEnum
from typing import NamedTuple
import time


class OrderSide(IntEnum):
    """Order side enumeration for efficient comparison."""
    BUY = 1
    SELL = -1


class OrderType(IntEnum):
    """Order type enumeration."""
    MARKET = 1
    LIMIT = 2
    IOC = 3  # Immediate or Cancel
    FOK = 4  # Fill or Kill


# Numba-compatible order specification
order_spec = [
    ('order_id', types.int64),
    ('symbol', types.unicode_type),
    ('side', types.int8),
    ('order_type', types.int8),
    ('price', types.float64),
    ('quantity', types.int64),
    ('remaining_quantity', types.int64),
    ('timestamp', types.int64),
    ('priority', types.int64),
]


@jitclass(order_spec)
class Order:
    """
    High-performance order structure optimized for JIT compilation.
    
    All fields are primitive types to ensure maximum performance
    and compatibility with Numba's JIT compiler.
    """
    
    def __init__(self, order_id, symbol, side, order_type, price, quantity, timestamp):
        self.order_id = order_id
        self.symbol = symbol
        self.side = side
        self.order_type = order_type
        self.price = price
        self.quantity = quantity
        self.remaining_quantity = quantity
        self.timestamp = timestamp
        self.priority = timestamp  # FIFO priority based on timestamp
    
    def is_buy(self):
        """Check if order is a buy order."""
        return self.side == OrderSide.BUY
    
    def is_sell(self):
        """Check if order is a sell order."""
        return self.side == OrderSide.SELL
    
    def is_market_order(self):
        """Check if order is a market order."""
        return self.order_type == OrderType.MARKET
    
    def is_limit_order(self):
        """Check if order is a limit order."""
        return self.order_type == OrderType.LIMIT
    
    def is_filled(self):
        """Check if order is completely filled."""
        return self.remaining_quantity == 0
    
    def can_match_price(self, other_price):
        """
        Check if this order can match with another price.
        
        Args:
            other_price: Price to check against
            
        Returns:
            bool: True if prices can match
        """
        if self.is_market_order():
            return True
        
        if self.is_buy():
            return self.price >= other_price
        else:
            return self.price <= other_price
    
    def fill(self, quantity):
        """
        Fill part of the order.
        
        Args:
            quantity: Quantity to fill
            
        Returns:
            int: Actual quantity filled
        """
        fill_qty = min(quantity, self.remaining_quantity)
        self.remaining_quantity -= fill_qty
        return fill_qty


# Trade execution result
trade_spec = [
    ('buy_order_id', types.int64),
    ('sell_order_id', types.int64),
    ('price', types.float64),
    ('quantity', types.int64),
    ('timestamp', types.int64),
]


@jitclass(trade_spec)
class Trade:
    """Trade execution result."""
    
    def __init__(self, buy_order_id, sell_order_id, price, quantity, timestamp):
        self.buy_order_id = buy_order_id
        self.sell_order_id = sell_order_id
        self.price = price
        self.quantity = quantity
        self.timestamp = timestamp


class OrderFactory:
    """Factory for creating orders with automatic ID generation."""
    
    def __init__(self):
        self._next_id = 1
    
    def create_order(self, symbol: str, side: OrderSide, order_type: OrderType, price: float, quantity: int) -> Order:
        """Create an order of the specified type."""
        if order_type == OrderType.LIMIT:
            return self.create_limit_order(symbol, side, price, quantity)
        elif order_type == OrderType.MARKET:
            return self.create_market_order(symbol, side, quantity)
        elif order_type == OrderType.IOC:
            return self.create_ioc_order(symbol, side, price, quantity)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")
    
    def create_limit_order(self, symbol: str, side: OrderSide, price: float, quantity: int) -> Order:
        """Create a limit order."""
        order_id = self._next_id
        self._next_id += 1
        timestamp = time.time_ns()
        
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.LIMIT),
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )
    
    def create_market_order(self, symbol: str, side: OrderSide, quantity: int) -> Order:
        """Create a market order."""
        order_id = self._next_id
        self._next_id += 1
        timestamp = time.time_ns()
        
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.MARKET),
            price=0.0,  # Market orders don't have a price limit
            quantity=quantity,
            timestamp=timestamp
        )
    
    def create_ioc_order(self, symbol: str, side: OrderSide, price: float, quantity: int) -> Order:
        """Create an Immediate or Cancel order."""
        order_id = self._next_id
        self._next_id += 1
        timestamp = time.time_ns()
        
        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.IOC),
            price=price,
            quantity=quantity,
            timestamp=timestamp
        )


# Utility functions for order manipulation
def get_current_timestamp_ns():
    """Get current timestamp in nanoseconds."""
    return time.time_ns()


def orders_can_match(buy_order: Order, sell_order: Order) -> bool:
    """
    Check if two orders can match.
    
    Args:
        buy_order: Buy order
        sell_order: Sell order
        
    Returns:
        bool: True if orders can match
    """
    if buy_order.symbol != sell_order.symbol:
        return False
    
    if buy_order.is_filled() or sell_order.is_filled():
        return False
    
    # Market orders can always match
    if buy_order.is_market_order() or sell_order.is_market_order():
        return True
    
    # Limit orders match if buy price >= sell price
    return buy_order.price >= sell_order.price


def calculate_match_price(buy_order: Order, sell_order: Order) -> float:
    """
    Calculate the execution price for matching orders.
    
    Uses price-time priority: the order that arrived first gets their price.
    
    Args:
        buy_order: Buy order
        sell_order: Sell order
        
    Returns:
        float: Execution price
    """
    # Market orders execute at the limit order's price
    if buy_order.is_market_order():
        return sell_order.price
    if sell_order.is_market_order():
        return buy_order.price
    
    # For limit orders, use the price of the order that arrived first
    if buy_order.timestamp <= sell_order.timestamp:
        return buy_order.price
    else:
        return sell_order.price


def calculate_match_quantity(buy_order: Order, sell_order: Order) -> int:
    """
    Calculate the execution quantity for matching orders.
    
    Args:
        buy_order: Buy order
        sell_order: Sell order
        
    Returns:
        int: Execution quantity
    """
    return min(buy_order.remaining_quantity, sell_order.remaining_quantity)
