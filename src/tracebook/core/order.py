"""
High-performance order data structures optimized for Numba JIT compilation.
"""

from numba import types
from numba.experimental import jitclass
from enum import IntEnum
from numbers import Real
import math
import threading
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


def normalize_symbol(symbol: str) -> str:
    """Validate and normalize a trading symbol."""
    if not isinstance(symbol, str):
        raise ValueError(f"Order symbol must be a non-empty string: {symbol!r}")

    normalized_symbol = symbol.strip()
    if not normalized_symbol:
        raise ValueError("Order symbol must be a non-empty string")

    return normalized_symbol


# Numba-compatible order specification
order_spec = [
    ("order_id", types.int64),
    ("symbol", types.unicode_type),
    ("side", types.int8),
    ("order_type", types.int8),
    ("price", types.float64),
    ("quantity", types.float64),
    ("remaining_quantity", types.float64),
    ("timestamp", types.int64),
    ("priority", types.int64),
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
        return self.remaining_quantity <= 1e-12

    def is_ioc_order(self):
        """Check if order is an Immediate-or-Cancel order."""
        return self.order_type == OrderType.IOC

    def is_fok_order(self):
        """Check if order is a Fill-or-Kill order."""
        return self.order_type == OrderType.FOK

    def can_rest(self):
        """Return True if any unfilled quantity may remain on the book."""
        return self.order_type == OrderType.LIMIT

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
            float: Actual quantity filled
        """
        fill_qty = min(quantity, self.remaining_quantity)
        self.remaining_quantity -= fill_qty
        return fill_qty


# Trade execution result
trade_spec = [
    ("buy_order_id", types.int64),
    ("sell_order_id", types.int64),
    ("price", types.float64),
    ("quantity", types.float64),
    ("timestamp", types.int64),
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
        self._lock = threading.Lock()

    def _allocate_order_id(self) -> int:
        """Allocate a unique order id."""
        with self._lock:
            order_id = self._next_id
            self._next_id += 1
            return order_id

    def advance_past(self, order_id: int):
        """Advance the allocator beyond a caller-supplied order id."""
        with self._lock:
            if order_id >= self._next_id:
                self._next_id = order_id + 1

    def _validate_side(self, side: OrderSide):
        """Validate and normalize an order side."""
        if isinstance(side, bool):
            raise ValueError(f"Unsupported order side: {side}")
        try:
            side_value = int(side)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported order side: {side}") from exc

        if side_value not in (int(OrderSide.BUY), int(OrderSide.SELL)):
            raise ValueError(f"Unsupported order side: {side}")
        return OrderSide(side_value)

    def _validate_order_type(self, order_type: OrderType):
        """Validate and normalize an order type."""
        if isinstance(order_type, bool):
            raise ValueError(f"Unsupported order type: {order_type}")
        try:
            order_type_value = int(order_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Unsupported order type: {order_type}") from exc

        valid_types = (
            int(OrderType.MARKET),
            int(OrderType.LIMIT),
            int(OrderType.IOC),
            int(OrderType.FOK),
        )
        if order_type_value not in valid_types:
            raise ValueError(f"Unsupported order type: {order_type}")
        return OrderType(order_type_value)

    def _validate_quantity(self, quantity: float):
        """Validate that quantity is positive."""
        if isinstance(quantity, bool) or not isinstance(quantity, Real):
            raise ValueError(f"Order quantity must be positive: {quantity!r}")
        quantity = float(quantity)
        if not math.isfinite(quantity) or quantity <= 0:
            raise ValueError(f"Order quantity must be positive: {quantity}")
        return quantity

    def _validate_price(self, price: float):
        """Validate that limit-style order price is positive."""
        if isinstance(price, bool) or not isinstance(price, Real):
            raise ValueError(f"Order price must be positive: {price!r}")
        price = float(price)
        if not math.isfinite(price) or price <= 0:
            raise ValueError(f"Order price must be positive: {price}")
        return price

    def create_order(
        self, symbol: str, side: OrderSide, order_type: OrderType, price: float, quantity: float
    ) -> Order:
        """Create an order of the specified type."""
        order_type = self._validate_order_type(order_type)
        if order_type == OrderType.LIMIT:
            return self.create_limit_order(symbol, side, price, quantity)
        elif order_type == OrderType.MARKET:
            return self.create_market_order(symbol, side, quantity)
        elif order_type == OrderType.IOC:
            return self.create_ioc_order(symbol, side, price, quantity)
        elif order_type == OrderType.FOK:
            return self.create_fok_order(symbol, side, price, quantity)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def create_limit_order(
        self, symbol: str, side: OrderSide, price: float, quantity: float
    ) -> Order:
        """Create a limit order."""
        symbol = normalize_symbol(symbol)
        side = self._validate_side(side)
        price = self._validate_price(price)
        quantity = self._validate_quantity(quantity)
        order_id = self._allocate_order_id()
        timestamp = time.time_ns()

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.LIMIT),
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )

    def create_market_order(self, symbol: str, side: OrderSide, quantity: float) -> Order:
        """Create a market order."""
        symbol = normalize_symbol(symbol)
        side = self._validate_side(side)
        quantity = self._validate_quantity(quantity)
        order_id = self._allocate_order_id()
        timestamp = time.time_ns()

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.MARKET),
            price=0.0,  # Market orders don't have a price limit
            quantity=quantity,
            timestamp=timestamp,
        )

    def create_ioc_order(
        self, symbol: str, side: OrderSide, price: float, quantity: float
    ) -> Order:
        """Create an Immediate or Cancel order."""
        symbol = normalize_symbol(symbol)
        side = self._validate_side(side)
        price = self._validate_price(price)
        quantity = self._validate_quantity(quantity)
        order_id = self._allocate_order_id()
        timestamp = time.time_ns()

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.IOC),
            price=price,
            quantity=quantity,
            timestamp=timestamp,
        )

    def create_fok_order(
        self, symbol: str, side: OrderSide, price: float, quantity: float
    ) -> Order:
        """Create a Fill or Kill order."""
        symbol = normalize_symbol(symbol)
        side = self._validate_side(side)
        price = self._validate_price(price)
        quantity = self._validate_quantity(quantity)
        order_id = self._allocate_order_id()
        timestamp = time.time_ns()

        return Order(
            order_id=order_id,
            symbol=symbol,
            side=int(side),
            order_type=int(OrderType.FOK),
            price=price,
            quantity=quantity,
            timestamp=timestamp,
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


def calculate_match_quantity(buy_order: Order, sell_order: Order) -> float:
    """
    Calculate the execution quantity for matching orders.

    Args:
        buy_order: Buy order
        sell_order: Sell order

    Returns:
        float: Execution quantity
    """
    return min(buy_order.remaining_quantity, sell_order.remaining_quantity)
