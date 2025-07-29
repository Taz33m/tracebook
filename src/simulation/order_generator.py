"""
High-performance synthetic order generator for order book simulation.

This module provides various order generation strategies to simulate
realistic market conditions with configurable parameters for testing
and benchmarking the order book system.
"""

import numpy as np
import time
import threading
from typing import List, Dict, Any, Optional, Callable, Iterator
from dataclasses import dataclass
from enum import IntEnum
import random
from abc import ABC, abstractmethod

from core.order import Order, OrderSide, OrderType, OrderFactory
from profiling.performance_monitor import get_performance_monitor


class OrderPattern(IntEnum):
    """Order generation patterns."""
    RANDOM = 0
    TREND_FOLLOWING = 1
    MEAN_REVERTING = 2
    MOMENTUM = 3
    MARKET_MAKING = 4
    AGGRESSIVE = 5
    PASSIVE = 6
    MIXED = 7


@dataclass
class MarketParameters:
    """Market simulation parameters."""
    symbol: str = "BTCUSD"
    initial_price: float = 50000.0
    price_volatility: float = 0.02  # 2% volatility
    tick_size: float = 0.01
    min_quantity: float = 0.001
    max_quantity: float = 10.0
    spread_bps: int = 5  # 5 basis points spread
    
    # Order flow parameters
    order_arrival_rate: float = 1000.0  # orders per second
    market_order_ratio: float = 0.3  # 30% market orders
    cancel_ratio: float = 0.4  # 40% of limit orders get cancelled
    
    # Price movement parameters
    trend_strength: float = 0.0  # -1 to 1, 0 = no trend
    mean_reversion_speed: float = 0.1
    volatility_clustering: float = 0.8
    
    # Market making parameters
    mm_spread_multiplier: float = 1.5
    mm_quantity_multiplier: float = 2.0
    mm_order_ratio: float = 0.2  # 20% market making orders


@dataclass
class OrderGenerationConfig:
    """Configuration for order generation."""
    pattern: OrderPattern = OrderPattern.MIXED
    duration_seconds: float = 60.0
    target_throughput: float = 1000.0  # orders per second
    batch_size: int = 100
    enable_cancellations: bool = True
    enable_modifications: bool = True
    randomize_timing: bool = True
    seed: Optional[int] = None


class PriceModel:
    """
    Sophisticated price model for realistic market simulation.
    
    Implements geometric Brownian motion with jumps, volatility clustering,
    and mean reversion components.
    """
    
    def __init__(self, initial_price: float, volatility: float, 
                 trend: float = 0.0, mean_reversion_speed: float = 0.1):
        self.initial_price = initial_price
        self.current_price = initial_price
        self.volatility = volatility
        self.trend = trend
        self.mean_reversion_speed = mean_reversion_speed
        
        # State variables
        self.price_history = [initial_price]
        self.volatility_history = [volatility]
        self.last_update_time = time.time_ns()
        
        # GARCH-like volatility clustering
        self.volatility_alpha = 0.1
        self.volatility_beta = 0.85
        self.long_term_volatility = volatility
        
        # Jump parameters
        self.jump_intensity = 0.01  # 1% chance per update
        self.jump_size_std = 0.02  # 2% jump size
        
    def update_price(self, dt_seconds: float = 0.001) -> float:
        """Update price using sophisticated model."""
        current_time = time.time_ns()
        actual_dt = (current_time - self.last_update_time) / 1_000_000_000
        dt = min(actual_dt, dt_seconds)  # Cap dt for stability
        
        if dt <= 0:
            return self.current_price
        
        # Mean reversion component
        mean_reversion = -self.mean_reversion_speed * (
            self.current_price - self.initial_price
        ) * dt
        
        # Trend component
        trend_component = self.trend * self.current_price * dt
        
        # Update volatility (GARCH-like)
        if len(self.price_history) > 1:
            last_return = (self.price_history[-1] - self.price_history[-2]) / self.price_history[-2]
            self.volatility = np.sqrt(
                self.volatility_alpha * last_return**2 +
                self.volatility_beta * self.volatility**2 +
                (1 - self.volatility_alpha - self.volatility_beta) * self.long_term_volatility**2
            )
        
        # Brownian motion component
        random_component = (
            self.volatility * self.current_price * 
            np.random.normal(0, np.sqrt(dt))
        )
        
        # Jump component
        jump_component = 0.0
        if np.random.random() < self.jump_intensity * dt:
            jump_size = np.random.normal(0, self.jump_size_std)
            jump_component = self.current_price * jump_size
        
        # Update price
        price_change = mean_reversion + trend_component + random_component + jump_component
        self.current_price = max(0.01, self.current_price + price_change)
        
        # Update history
        self.price_history.append(self.current_price)
        self.volatility_history.append(self.volatility)
        
        # Keep history bounded
        if len(self.price_history) > 10000:
            self.price_history = self.price_history[-5000:]
            self.volatility_history = self.volatility_history[-5000:]
        
        self.last_update_time = current_time
        return self.current_price
    
    def get_bid_ask_prices(self, spread_bps: int) -> tuple[float, float]:
        """Get bid and ask prices based on current price and spread."""
        spread = self.current_price * (spread_bps / 10000.0)
        bid_price = self.current_price - spread / 2
        ask_price = self.current_price + spread / 2
        return bid_price, ask_price
    
    def get_price_levels(self, num_levels: int = 5, spread_multiplier: float = 1.0) -> Dict[str, List[float]]:
        """Get multiple price levels for order book depth."""
        bid_price, ask_price = self.get_bid_ask_prices(int(5 * spread_multiplier))
        
        tick_size = self.current_price * 0.0001  # 1 basis point
        
        bid_levels = [bid_price - i * tick_size for i in range(num_levels)]
        ask_levels = [ask_price + i * tick_size for i in range(num_levels)]
        
        return {
            'bids': bid_levels,
            'asks': ask_levels
        }


class OrderGenerator(ABC):
    """Abstract base class for order generators."""
    
    @abstractmethod
    def generate_orders(self, count: int) -> List[Order]:
        """Generate a batch of orders."""
        pass
    
    @abstractmethod
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        pass


class RandomOrderGenerator(OrderGenerator):
    """Generates random orders with configurable parameters."""
    
    def __init__(self, market_params: MarketParameters, config: OrderGenerationConfig):
        self.market_params = market_params
        self.config = config
        self.price_model = PriceModel(
            market_params.initial_price,
            market_params.price_volatility,
            market_params.trend_strength,
            market_params.mean_reversion_speed
        )
        self.order_factory = OrderFactory()
        
        # Statistics
        self.orders_generated = 0
        self.orders_by_side = {OrderSide.BUY: 0, OrderSide.SELL: 0}
        self.orders_by_type = {OrderType.MARKET: 0, OrderType.LIMIT: 0}
        
        # Random state
        if config.seed is not None:
            np.random.seed(config.seed)
            random.seed(config.seed)
    
    def generate_orders(self, count: int) -> List[Order]:
        """Generate random orders."""
        orders = []
        
        for _ in range(count):
            # Update price model
            self.price_model.update_price()
            
            # Determine order type
            is_market_order = np.random.random() < self.market_params.market_order_ratio
            order_type = OrderType.MARKET if is_market_order else OrderType.LIMIT
            
            # Determine side (slightly biased based on trend)
            side_bias = 0.5 + self.market_params.trend_strength * 0.1
            side = OrderSide.BUY if np.random.random() < side_bias else OrderSide.SELL
            
            # Determine quantity
            quantity = np.random.uniform(
                self.market_params.min_quantity,
                self.market_params.max_quantity
            )
            
            # Determine price
            if order_type == OrderType.MARKET:
                price = 0.0  # Market orders don't have a price
            else:
                # Limit order price
                bid_price, ask_price = self.price_model.get_bid_ask_prices(
                    self.market_params.spread_bps
                )
                
                if side == OrderSide.BUY:
                    # Buy limit order: price around or below current bid
                    price_range = bid_price * 0.05  # 5% range
                    price = np.random.uniform(bid_price - price_range, bid_price + price_range * 0.2)
                else:
                    # Sell limit order: price around or above current ask
                    price_range = ask_price * 0.05  # 5% range
                    price = np.random.uniform(ask_price - price_range * 0.2, ask_price + price_range)
                
                # Round to tick size
                price = round(price / self.market_params.tick_size) * self.market_params.tick_size
            
            # Create order
            order = self.order_factory.create_order(
                symbol=self.market_params.symbol,
                side=side,
                order_type=order_type,
                price=price,
                quantity=quantity
            )
            
            orders.append(order)
            
            # Update statistics
            self.orders_generated += 1
            self.orders_by_side[side] += 1
            self.orders_by_type[order_type] += 1
        
        return orders
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'orders_generated': self.orders_generated,
            'orders_by_side': dict(self.orders_by_side),
            'orders_by_type': dict(self.orders_by_type),
            'current_price': self.price_model.current_price,
            'current_volatility': self.price_model.volatility,
            'price_history_length': len(self.price_model.price_history),
        }


class MarketMakingOrderGenerator(OrderGenerator):
    """Generates market making orders with tight spreads and regular updates."""
    
    def __init__(self, market_params: MarketParameters, config: OrderGenerationConfig):
        self.market_params = market_params
        self.config = config
        self.price_model = PriceModel(
            market_params.initial_price,
            market_params.price_volatility
        )
        self.order_factory = OrderFactory()
        
        # Market making specific parameters
        self.spread_multiplier = market_params.mm_spread_multiplier
        self.quantity_multiplier = market_params.mm_quantity_multiplier
        self.num_levels = 5  # Number of price levels to quote
        
        # Statistics
        self.orders_generated = 0
        self.bid_orders = 0
        self.ask_orders = 0
        
    def generate_orders(self, count: int) -> List[Order]:
        """Generate market making orders."""
        orders = []
        
        # Update price
        self.price_model.update_price()
        
        # Get price levels
        price_levels = self.price_model.get_price_levels(
            self.num_levels, self.spread_multiplier
        )
        
        orders_per_side = count // 2
        
        # Generate bid orders
        for i in range(min(orders_per_side, len(price_levels['bids']))):
            price = price_levels['bids'][i]
            quantity = np.random.uniform(
                self.market_params.min_quantity * self.quantity_multiplier,
                self.market_params.max_quantity * self.quantity_multiplier
            )
            
            order = self.order_factory.create_order(
                symbol=self.market_params.symbol,
                side=OrderSide.BUY,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity
            )
            
            orders.append(order)
            self.bid_orders += 1
        
        # Generate ask orders
        for i in range(min(count - len(orders), len(price_levels['asks']))):
            price = price_levels['asks'][i]
            quantity = np.random.uniform(
                self.market_params.min_quantity * self.quantity_multiplier,
                self.market_params.max_quantity * self.quantity_multiplier
            )
            
            order = self.order_factory.create_order(
                symbol=self.market_params.symbol,
                side=OrderSide.SELL,
                order_type=OrderType.LIMIT,
                price=price,
                quantity=quantity
            )
            
            orders.append(order)
            self.ask_orders += 1
        
        self.orders_generated += len(orders)
        return orders
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'orders_generated': self.orders_generated,
            'bid_orders': self.bid_orders,
            'ask_orders': self.ask_orders,
            'current_price': self.price_model.current_price,
            'spread_multiplier': self.spread_multiplier,
            'num_levels': self.num_levels,
        }


class AggressiveOrderGenerator(OrderGenerator):
    """Generates aggressive orders that cross the spread."""
    
    def __init__(self, market_params: MarketParameters, config: OrderGenerationConfig):
        self.market_params = market_params
        self.config = config
        self.price_model = PriceModel(
            market_params.initial_price,
            market_params.price_volatility,
            market_params.trend_strength
        )
        self.order_factory = OrderFactory()
        
        # Aggressive order parameters
        self.market_order_ratio = 0.7  # 70% market orders
        self.aggressive_limit_ratio = 0.8  # 80% of limit orders are aggressive
        
        # Statistics
        self.orders_generated = 0
        self.market_orders = 0
        self.aggressive_limit_orders = 0
        
    def generate_orders(self, count: int) -> List[Order]:
        """Generate aggressive orders."""
        orders = []
        
        for _ in range(count):
            # Update price
            self.price_model.update_price()
            
            # Determine order type
            is_market_order = np.random.random() < self.market_order_ratio
            
            # Determine side (biased by trend)
            side_bias = 0.5 + self.market_params.trend_strength * 0.2
            side = OrderSide.BUY if np.random.random() < side_bias else OrderSide.SELL
            
            # Determine quantity (larger for aggressive orders)
            quantity = np.random.uniform(
                self.market_params.min_quantity * 2,
                self.market_params.max_quantity * 1.5
            )
            
            if is_market_order:
                # Market order
                order = self.order_factory.create_order(
                    symbol=self.market_params.symbol,
                    side=side,
                    order_type=OrderType.MARKET,
                    price=0.0,
                    quantity=quantity
                )
                self.market_orders += 1
                
            else:
                # Aggressive limit order
                bid_price, ask_price = self.price_model.get_bid_ask_prices(
                    self.market_params.spread_bps
                )
                
                if side == OrderSide.BUY:
                    # Aggressive buy: price at or above ask
                    if np.random.random() < self.aggressive_limit_ratio:
                        price = ask_price + np.random.uniform(0, ask_price * 0.01)
                        self.aggressive_limit_orders += 1
                    else:
                        price = bid_price
                else:
                    # Aggressive sell: price at or below bid
                    if np.random.random() < self.aggressive_limit_ratio:
                        price = bid_price - np.random.uniform(0, bid_price * 0.01)
                        self.aggressive_limit_orders += 1
                    else:
                        price = ask_price
                
                # Round to tick size
                price = round(price / self.market_params.tick_size) * self.market_params.tick_size
                
                order = self.order_factory.create_order(
                    symbol=self.market_params.symbol,
                    side=side,
                    order_type=OrderType.LIMIT,
                    price=price,
                    quantity=quantity
                )
            
            orders.append(order)
            self.orders_generated += 1
        
        return orders
    
    def get_generator_stats(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            'orders_generated': self.orders_generated,
            'market_orders': self.market_orders,
            'aggressive_limit_orders': self.aggressive_limit_orders,
            'market_order_ratio': self.market_order_ratio,
            'aggressive_limit_ratio': self.aggressive_limit_ratio,
            'current_price': self.price_model.current_price,
        }


class SyntheticOrderStream:
    """
    High-performance synthetic order stream generator.
    
    Coordinates multiple order generators to create realistic market conditions
    with configurable throughput and patterns.
    """
    
    def __init__(self, market_params: MarketParameters, config: OrderGenerationConfig):
        self.market_params = market_params
        self.config = config
        
        # Initialize generators based on pattern
        self.generators = self._create_generators()
        
        # Stream control
        self.is_running = False
        self.stream_thread = None
        self.orders_queue = []
        self.queue_lock = threading.Lock()
        
        # Performance monitoring
        self.performance_monitor = get_performance_monitor()
        
        # Statistics
        self.total_orders_generated = 0
        self.generation_start_time = 0
        self.generation_times = []
        
        # Callbacks
        self.order_callbacks = []
        
    def _create_generators(self) -> List[OrderGenerator]:
        """Create order generators based on configuration."""
        generators = []
        
        if self.config.pattern == OrderPattern.RANDOM:
            generators.append(RandomOrderGenerator(self.market_params, self.config))
            
        elif self.config.pattern == OrderPattern.MARKET_MAKING:
            generators.append(MarketMakingOrderGenerator(self.market_params, self.config))
            
        elif self.config.pattern == OrderPattern.AGGRESSIVE:
            generators.append(AggressiveOrderGenerator(self.market_params, self.config))
            
        elif self.config.pattern == OrderPattern.MIXED:
            # Mixed pattern: combine multiple generators
            generators.extend([
                RandomOrderGenerator(self.market_params, self.config),
                MarketMakingOrderGenerator(self.market_params, self.config),
                AggressiveOrderGenerator(self.market_params, self.config),
            ])
            
        else:
            # Default to random
            generators.append(RandomOrderGenerator(self.market_params, self.config))
        
        return generators
    
    def register_order_callback(self, callback: Callable[[List[Order]], None]):
        """Register callback for generated orders."""
        self.order_callbacks.append(callback)
    
    def start_stream(self):
        """Start the order generation stream."""
        if self.is_running:
            return
        
        print(f"Starting order stream - Target: {self.config.target_throughput:.0f} orders/sec")
        
        self.is_running = True
        self.generation_start_time = time.time_ns()
        self.stream_thread = threading.Thread(target=self._generation_loop, daemon=True)
        self.stream_thread.start()
    
    def stop_stream(self):
        """Stop the order generation stream."""
        if not self.is_running:
            return
        
        print("Stopping order stream...")
        self.is_running = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        print("Order stream stopped")
    
    def get_orders(self, max_count: int = None) -> List[Order]:
        """Get generated orders from the queue."""
        with self.queue_lock:
            if max_count is None:
                orders = list(self.orders_queue)
                self.orders_queue.clear()
            else:
                orders = self.orders_queue[:max_count]
                self.orders_queue = self.orders_queue[max_count:]
            
            return orders
    
    def get_stream_stats(self) -> Dict[str, Any]:
        """Get comprehensive stream statistics."""
        current_time = time.time_ns()
        uptime_seconds = (current_time - self.generation_start_time) / 1_000_000_000
        
        # Aggregate generator stats
        generator_stats = {}
        for i, generator in enumerate(self.generators):
            generator_stats[f'generator_{i}'] = generator.get_generator_stats()
        
        # Calculate throughput
        actual_throughput = self.total_orders_generated / max(uptime_seconds, 0.001)
        
        # Generation time statistics
        generation_stats = {}
        if self.generation_times:
            generation_array = np.array(self.generation_times)
            generation_stats = {
                'mean_generation_time_ms': float(np.mean(generation_array)) / 1_000_000,
                'max_generation_time_ms': float(np.max(generation_array)) / 1_000_000,
                'p95_generation_time_ms': float(np.percentile(generation_array, 95)) / 1_000_000,
            }
        
        return {
            'config': {
                'pattern': self.config.pattern.name,
                'target_throughput': self.config.target_throughput,
                'batch_size': self.config.batch_size,
                'duration_seconds': self.config.duration_seconds,
            },
            'runtime_stats': {
                'uptime_seconds': uptime_seconds,
                'is_running': self.is_running,
                'total_orders_generated': self.total_orders_generated,
                'actual_throughput': actual_throughput,
                'queue_size': len(self.orders_queue),
            },
            'generation_performance': generation_stats,
            'generator_stats': generator_stats,
            'market_params': {
                'symbol': self.market_params.symbol,
                'current_price': self.generators[0].price_model.current_price if self.generators else 0,
                'volatility': self.market_params.price_volatility,
                'spread_bps': self.market_params.spread_bps,
            }
        }
    
    def _generation_loop(self):
        """Main order generation loop."""
        target_interval = 1.0 / self.config.target_throughput * self.config.batch_size
        last_generation_time = time.time()
        
        while self.is_running:
            generation_start = time.time_ns()
            
            try:
                # Generate orders from all generators
                new_orders = []
                
                if len(self.generators) == 1:
                    # Single generator
                    new_orders = self.generators[0].generate_orders(self.config.batch_size)
                else:
                    # Multiple generators: distribute batch size
                    orders_per_generator = self.config.batch_size // len(self.generators)
                    remainder = self.config.batch_size % len(self.generators)
                    
                    for i, generator in enumerate(self.generators):
                        count = orders_per_generator + (1 if i < remainder else 0)
                        if count > 0:
                            new_orders.extend(generator.generate_orders(count))
                
                # Add to queue
                with self.queue_lock:
                    self.orders_queue.extend(new_orders)
                
                # Update statistics
                self.total_orders_generated += len(new_orders)
                
                generation_time = time.time_ns() - generation_start
                self.generation_times.append(generation_time)
                
                # Keep generation times bounded
                if len(self.generation_times) > 1000:
                    self.generation_times = self.generation_times[-500:]
                
                # Record performance metrics
                self.performance_monitor.record_order_processing(
                    generation_time, len(new_orders)
                )
                
                # Trigger callbacks
                for callback in self.order_callbacks:
                    try:
                        callback(new_orders)
                    except Exception as e:
                        print(f"Error in order callback: {e}")
                
                # Rate limiting
                if self.config.randomize_timing:
                    # Add some jitter to avoid perfect timing
                    jitter = np.random.uniform(0.8, 1.2)
                    sleep_time = target_interval * jitter
                else:
                    sleep_time = target_interval
                
                current_time = time.time()
                elapsed = current_time - last_generation_time
                
                if elapsed < sleep_time:
                    time.sleep(sleep_time - elapsed)
                
                last_generation_time = time.time()
                
            except Exception as e:
                print(f"Error in order generation loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def __enter__(self):
        """Context manager entry."""
        self.start_stream()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_stream()


def create_order_stream(pattern: OrderPattern = OrderPattern.MIXED,
                       throughput: float = 1000.0,
                       duration: float = 60.0,
                       symbol: str = "BTCUSD",
                       initial_price: float = 50000.0) -> SyntheticOrderStream:
    """
    Convenience function to create a configured order stream.
    
    Args:
        pattern: Order generation pattern
        throughput: Target orders per second
        duration: Stream duration in seconds
        symbol: Trading symbol
        initial_price: Initial price for the symbol
    
    Returns:
        Configured SyntheticOrderStream instance
    """
    market_params = MarketParameters(
        symbol=symbol,
        initial_price=initial_price,
        order_arrival_rate=throughput
    )
    
    config = OrderGenerationConfig(
        pattern=pattern,
        duration_seconds=duration,
        target_throughput=throughput,
        batch_size=min(100, max(1, int(throughput / 10)))  # Adaptive batch size
    )
    
    return SyntheticOrderStream(market_params, config)
