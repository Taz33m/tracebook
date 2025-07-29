"""
High-performance matching engine with JIT-compiled algorithms.

This module implements the core order matching logic optimized for
maximum throughput and minimum latency using Numba JIT compilation.
"""

import numpy as np
from numba import jit, types
from typing import List, Tuple, Optional
import time
from .order import Order, Trade, OrderSide, orders_can_match, calculate_match_price, calculate_match_quantity
from .price_level import PriceLevelManager, MarketDataSnapshot


@jit(nopython=True, cache=True)
def match_orders_fifo(buy_price: float, buy_quantity: int, buy_timestamp: int,
                      sell_price: float, sell_quantity: int, sell_timestamp: int) -> Tuple[float, int]:
    """
    JIT-compiled FIFO matching logic.
    
    Args:
        buy_price, buy_quantity, buy_timestamp: Buy order details
        sell_price, sell_quantity, sell_timestamp: Sell order details
        
    Returns:
        Tuple[float, int]: (execution_price, execution_quantity)
    """
    # Check if orders can match
    if buy_price < sell_price:
        return 0.0, 0
    
    # Calculate execution price (price-time priority)
    if buy_timestamp <= sell_timestamp:
        execution_price = buy_price
    else:
        execution_price = sell_price
    
    # Calculate execution quantity
    execution_quantity = min(buy_quantity, sell_quantity)
    
    return execution_price, execution_quantity


@jit(nopython=True, cache=True)
def match_orders_pro_rata(buy_price: float, buy_quantity: int,
                          sell_price: float, sell_quantity: int,
                          total_sell_quantity: int) -> Tuple[float, int]:
    """
    JIT-compiled Pro-Rata matching logic.
    
    Args:
        buy_price, buy_quantity: Buy order details
        sell_price, sell_quantity: Sell order details
        total_sell_quantity: Total quantity available at this price level
        
    Returns:
        Tuple[float, int]: (execution_price, execution_quantity)
    """
    # Check if orders can match
    if buy_price < sell_price:
        return 0.0, 0
    
    # Pro-rata allocation based on order size
    if total_sell_quantity > 0:
        allocation_ratio = sell_quantity / total_sell_quantity
        max_fill = min(buy_quantity, sell_quantity)
        execution_quantity = int(max_fill * allocation_ratio)
    else:
        execution_quantity = 0
    
    execution_price = sell_price  # Use sell price for pro-rata
    
    return execution_price, execution_quantity


class MatchingEngine:
    """
    High-performance matching engine supporting multiple algorithms.
    
    Features:
    - JIT-compiled matching algorithms
    - Configurable matching logic (FIFO, Pro-Rata)
    - Minimal memory allocations
    - Nanosecond-precision timing
    """
    
    def __init__(self, symbol: str, matching_algorithm: str = 'fifo'):
        self.symbol = symbol
        self.matching_algorithm = matching_algorithm.lower()
        
        # Price level managers for each side
        self.buy_side = PriceLevelManager(is_buy_side=True)
        self.sell_side = PriceLevelManager(is_buy_side=False)
        
        # Trade history
        self.trades = []
        self.trade_count = 0
        
        # Performance metrics
        self.total_orders_processed = 0
        self.total_matches = 0
        self.last_trade_time = 0
        
        # Validate matching algorithm
        if self.matching_algorithm not in ['fifo', 'pro_rata']:
            raise ValueError(f"Unsupported matching algorithm: {matching_algorithm}")
    
    def add_order(self, order: Order) -> List[Trade]:
        """
        Add an order to the book and execute any possible matches.
        
        Args:
            order: Order to add
            
        Returns:
            List[Trade]: List of executed trades
        """
        self.total_orders_processed += 1
        trades = []
        
        if order.is_buy():
            trades = self._match_buy_order(order)
        else:
            trades = self._match_sell_order(order)
        
        # Add remaining quantity to book if not fully filled
        if order.remaining_quantity > 0:
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
        
        while buy_order.remaining_quantity > 0:
            best_sell_level = self.sell_side.get_best_price_level()
            if best_sell_level is None:
                break
            
            # Check if prices can match
            if not buy_order.can_match_price(best_sell_level.price):
                break
            
            # Execute matches at this price level
            level_trades = self._execute_matches_at_level(
                buy_order, best_sell_level, self.sell_side
            )
            trades.extend(level_trades)
            
            if not level_trades:  # No more matches possible
                break
        
        return trades
    
    def _match_sell_order(self, sell_order: Order) -> List[Trade]:
        """Match a sell order against buy side."""
        trades = []
        
        while sell_order.remaining_quantity > 0:
            best_buy_level = self.buy_side.get_best_price_level()
            if best_buy_level is None:
                break
            
            # Check if prices can match
            if not sell_order.can_match_price(best_buy_level.price):
                break
            
            # Execute matches at this price level
            level_trades = self._execute_matches_at_level(
                sell_order, best_buy_level, self.buy_side
            )
            trades.extend(level_trades)
            
            if not level_trades:  # No more matches possible
                break
        
        return trades
    
    def _execute_matches_at_level(self, incoming_order: Order, price_level, side_manager) -> List[Trade]:
        """Execute matches at a specific price level."""
        trades = []
        order_ids = list(price_level.orders)  # Copy to avoid modification during iteration
        
        for order_id in order_ids:
            if incoming_order.remaining_quantity == 0:
                break
            
            resting_order = side_manager.get_order(order_id)
            if resting_order is None or resting_order.remaining_quantity == 0:
                continue
            
            # Execute match based on algorithm
            if self.matching_algorithm == 'fifo':
                trade = self._execute_fifo_match(incoming_order, resting_order)
            else:  # pro_rata
                trade = self._execute_pro_rata_match(
                    incoming_order, resting_order, price_level.total_quantity
                )
            
            if trade is not None:
                trades.append(trade)
                
                # Update order quantities
                fill_qty = trade.quantity
                incoming_order.fill(fill_qty)
                resting_order.fill(fill_qty)
                
                # Update price level
                side_manager.update_order_quantity(order_id, fill_qty)
        
        return trades
    
    def _execute_fifo_match(self, incoming_order: Order, resting_order: Order) -> Optional[Trade]:
        """Execute a FIFO match between two orders."""
        if not orders_can_match(
            incoming_order if incoming_order.is_buy() else resting_order,
            resting_order if resting_order.is_sell() else incoming_order
        ):
            return None
        
        # Use JIT-compiled matching logic
        if incoming_order.is_buy():
            execution_price, execution_quantity = match_orders_fifo(
                incoming_order.price, incoming_order.remaining_quantity, incoming_order.timestamp,
                resting_order.price, resting_order.remaining_quantity, resting_order.timestamp
            )
            buy_order_id = incoming_order.order_id
            sell_order_id = resting_order.order_id
        else:
            execution_price, execution_quantity = match_orders_fifo(
                resting_order.price, resting_order.remaining_quantity, resting_order.timestamp,
                incoming_order.price, incoming_order.remaining_quantity, incoming_order.timestamp
            )
            buy_order_id = resting_order.order_id
            sell_order_id = incoming_order.order_id
        
        if execution_quantity > 0:
            trade = Trade(
                buy_order_id=buy_order_id,
                sell_order_id=sell_order_id,
                price=execution_price,
                quantity=execution_quantity,
                timestamp=time.time_ns()
            )
            self.trades.append(trade)
            return trade
        
        return None
    
    def _execute_pro_rata_match(self, incoming_order: Order, resting_order: Order, 
                               total_quantity: int) -> Optional[Trade]:
        """Execute a Pro-Rata match between two orders."""
        if not orders_can_match(
            incoming_order if incoming_order.is_buy() else resting_order,
            resting_order if resting_order.is_sell() else incoming_order
        ):
            return None
        
        # Use JIT-compiled matching logic
        if incoming_order.is_buy():
            execution_price, execution_quantity = match_orders_pro_rata(
                incoming_order.price, incoming_order.remaining_quantity,
                resting_order.price, resting_order.remaining_quantity,
                total_quantity
            )
            buy_order_id = incoming_order.order_id
            sell_order_id = resting_order.order_id
        else:
            execution_price, execution_quantity = match_orders_pro_rata(
                resting_order.price, resting_order.remaining_quantity,
                incoming_order.price, incoming_order.remaining_quantity,
                total_quantity
            )
            buy_order_id = resting_order.order_id
            sell_order_id = incoming_order.order_id
        
        if execution_quantity > 0:
            trade = Trade(
                buy_order_id=buy_order_id,
                sell_order_id=sell_order_id,
                price=execution_price,
                quantity=execution_quantity,
                timestamp=time.time_ns()
            )
            self.trades.append(trade)
            return trade
        
        return None
    
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
        bid_levels = self.buy_side.get_price_levels_snapshot()[:levels]
        ask_levels = self.sell_side.get_price_levels_snapshot()[:levels]
        
        return {
            'symbol': self.symbol,
            'timestamp': time.time_ns(),
            'bids': bid_levels,
            'asks': ask_levels,
        }
    
    def get_statistics(self) -> dict:
        """Get matching engine statistics."""
        return {
            'symbol': self.symbol,
            'matching_algorithm': self.matching_algorithm,
            'total_orders_processed': self.total_orders_processed,
            'total_matches': self.total_matches,
            'total_trades': len(self.trades),
            'buy_side_orders': self.buy_side.get_total_orders(),
            'sell_side_orders': self.sell_side.get_total_orders(),
            'buy_side_quantity': self.buy_side.get_total_quantity(),
            'sell_side_quantity': self.sell_side.get_total_quantity(),
            'last_trade_time': self.last_trade_time,
        }
    
    def clear(self):
        """Clear all orders and trades."""
        self.buy_side.clear()
        self.sell_side.clear()
        self.trades.clear()
        self.trade_count = 0
        self.total_orders_processed = 0
        self.total_matches = 0
        self.last_trade_time = 0
