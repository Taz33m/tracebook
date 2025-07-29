"""
FIFO (First In, First Out) matching algorithm implementation.

This module provides optimized FIFO matching logic with detailed
analytics and performance monitoring capabilities.
"""

import numpy as np
from numba import jit, types
from typing import List, Tuple, Dict, Any
from core.order import Order, Trade
import time


@jit(nopython=True, cache=True)
def calculate_fifo_priority(timestamp: int, price: float) -> float:
    """
    Calculate FIFO priority score.
    
    Args:
        timestamp: Order timestamp in nanoseconds
        price: Order price
        
    Returns:
        float: Priority score (lower is higher priority)
    """
    # Primary sort by timestamp (earlier = higher priority)
    # Secondary sort by price for tie-breaking
    return float(timestamp) + (price * 1e-12)  # Price as tie-breaker


@jit(nopython=True, cache=True)
def fifo_match_single(buy_price: float, buy_qty: int, buy_timestamp: int,
                      sell_price: float, sell_qty: int, sell_timestamp: int) -> Tuple[float, int, bool]:
    """
    Execute a single FIFO match between two orders.
    
    Args:
        buy_price, buy_qty, buy_timestamp: Buy order details
        sell_price, sell_qty, sell_timestamp: Sell order details
        
    Returns:
        Tuple[float, int, bool]: (execution_price, execution_quantity, can_match)
    """
    # Check if orders can match
    if buy_price < sell_price:
        return 0.0, 0, False
    
    # Calculate execution quantity
    execution_quantity = min(buy_qty, sell_qty)
    
    # Price-time priority: earlier order gets their price
    if buy_timestamp <= sell_timestamp:
        execution_price = buy_price
    else:
        execution_price = sell_price
    
    return execution_price, execution_quantity, True


@jit(nopython=True, cache=True)
def fifo_batch_match(buy_prices: np.ndarray, buy_quantities: np.ndarray, buy_timestamps: np.ndarray,
                     sell_prices: np.ndarray, sell_quantities: np.ndarray, sell_timestamps: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Execute batch FIFO matching for multiple orders.
    
    Args:
        buy_prices, buy_quantities, buy_timestamps: Buy order arrays
        sell_prices, sell_quantities, sell_timestamps: Sell order arrays
        
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: (execution_prices, execution_quantities, match_flags)
    """
    n_matches = min(len(buy_prices), len(sell_prices))
    execution_prices = np.zeros(n_matches, dtype=np.float64)
    execution_quantities = np.zeros(n_matches, dtype=np.int64)
    match_flags = np.zeros(n_matches, dtype=np.bool_)
    
    for i in range(n_matches):
        price, qty, can_match = fifo_match_single(
            buy_prices[i], buy_quantities[i], buy_timestamps[i],
            sell_prices[i], sell_quantities[i], sell_timestamps[i]
        )
        execution_prices[i] = price
        execution_quantities[i] = qty
        match_flags[i] = can_match
    
    return execution_prices, execution_quantities, match_flags


class FIFOAnalyzer:
    """
    Analyzer for FIFO matching algorithm performance and behavior.
    
    Provides detailed insights into FIFO matching patterns,
    queue dynamics, and fairness metrics.
    """
    
    def __init__(self):
        self.match_history = []
        self.queue_times = []  # Time orders spend in queue
        self.price_improvements = []  # Price improvement from FIFO
        self.fairness_metrics = {
            'total_matches': 0,
            'price_time_priority_violations': 0,
            'average_queue_time_ns': 0,
            'queue_time_variance': 0,
        }
    
    def record_match(self, buy_order: Order, sell_order: Order, execution_price: float, 
                    execution_quantity: int, match_timestamp: int):
        """Record a FIFO match for analysis."""
        # Calculate queue times
        buy_queue_time = match_timestamp - buy_order.timestamp
        sell_queue_time = match_timestamp - sell_order.timestamp
        
        self.queue_times.extend([buy_queue_time, sell_queue_time])
        
        # Calculate price improvement
        if buy_order.timestamp <= sell_order.timestamp:
            # Buy order had priority, got their price
            expected_price = buy_order.price
            price_improvement = abs(execution_price - expected_price)
        else:
            # Sell order had priority, got their price
            expected_price = sell_order.price
            price_improvement = abs(execution_price - expected_price)
        
        self.price_improvements.append(price_improvement)
        
        # Record match details
        match_record = {
            'timestamp': match_timestamp,
            'buy_order_id': buy_order.order_id,
            'sell_order_id': sell_order.order_id,
            'execution_price': execution_price,
            'execution_quantity': execution_quantity,
            'buy_queue_time': buy_queue_time,
            'sell_queue_time': sell_queue_time,
            'price_improvement': price_improvement,
            'priority_order': 'buy' if buy_order.timestamp <= sell_order.timestamp else 'sell'
        }
        
        self.match_history.append(match_record)
        self.fairness_metrics['total_matches'] += 1
        
        # Update running averages
        self._update_fairness_metrics()
    
    def get_queue_time_statistics(self) -> Dict[str, float]:
        """Get queue time statistics."""
        if not self.queue_times:
            return {}
        
        queue_times_array = np.array(self.queue_times)
        
        return {
            'mean_queue_time_ns': float(np.mean(queue_times_array)),
            'median_queue_time_ns': float(np.median(queue_times_array)),
            'std_queue_time_ns': float(np.std(queue_times_array)),
            'min_queue_time_ns': float(np.min(queue_times_array)),
            'max_queue_time_ns': float(np.max(queue_times_array)),
            'p95_queue_time_ns': float(np.percentile(queue_times_array, 95)),
            'p99_queue_time_ns': float(np.percentile(queue_times_array, 99)),
        }
    
    def get_price_improvement_statistics(self) -> Dict[str, float]:
        """Get price improvement statistics."""
        if not self.price_improvements:
            return {}
        
        improvements_array = np.array(self.price_improvements)
        
        return {
            'mean_price_improvement': float(np.mean(improvements_array)),
            'median_price_improvement': float(np.median(improvements_array)),
            'total_price_improvement': float(np.sum(improvements_array)),
            'max_price_improvement': float(np.max(improvements_array)),
        }
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report."""
        queue_stats = self.get_queue_time_statistics()
        price_stats = self.get_price_improvement_statistics()
        
        # Calculate fairness score (0-100, higher is more fair)
        fairness_score = self._calculate_fairness_score()
        
        return {
            'algorithm': 'FIFO',
            'fairness_score': fairness_score,
            'total_matches': self.fairness_metrics['total_matches'],
            'priority_violations': self.fairness_metrics['price_time_priority_violations'],
            'queue_time_stats': queue_stats,
            'price_improvement_stats': price_stats,
            'match_count': len(self.match_history),
        }
    
    def get_recent_matches(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent match records."""
        return self.match_history[-count:] if self.match_history else []
    
    def analyze_queue_dynamics(self) -> Dict[str, Any]:
        """Analyze queue dynamics and patterns."""
        if len(self.match_history) < 2:
            return {}
        
        # Analyze match frequency over time
        timestamps = [match['timestamp'] for match in self.match_history]
        time_diffs = np.diff(timestamps)
        
        # Analyze order priority patterns
        priority_patterns = [match['priority_order'] for match in self.match_history]
        buy_priority_count = priority_patterns.count('buy')
        sell_priority_count = priority_patterns.count('sell')
        
        return {
            'match_frequency_stats': {
                'mean_interval_ns': float(np.mean(time_diffs)),
                'std_interval_ns': float(np.std(time_diffs)),
                'min_interval_ns': float(np.min(time_diffs)),
                'max_interval_ns': float(np.max(time_diffs)),
            },
            'priority_distribution': {
                'buy_priority_matches': buy_priority_count,
                'sell_priority_matches': sell_priority_count,
                'buy_priority_ratio': buy_priority_count / len(self.match_history),
            },
            'queue_efficiency': self._calculate_queue_efficiency(),
        }
    
    def _update_fairness_metrics(self):
        """Update running fairness metrics."""
        if self.queue_times:
            self.fairness_metrics['average_queue_time_ns'] = np.mean(self.queue_times)
            self.fairness_metrics['queue_time_variance'] = np.var(self.queue_times)
    
    def _calculate_fairness_score(self) -> float:
        """Calculate overall fairness score (0-100)."""
        if not self.match_history:
            return 100.0
        
        # Base score starts at 100
        score = 100.0
        
        # Penalize priority violations
        violation_rate = (self.fairness_metrics['price_time_priority_violations'] / 
                         max(1, self.fairness_metrics['total_matches']))
        score -= violation_rate * 50  # Up to 50 point penalty
        
        # Penalize high queue time variance (unfairness indicator)
        if self.queue_times:
            queue_cv = np.std(self.queue_times) / max(1, np.mean(self.queue_times))
            score -= min(queue_cv * 10, 25)  # Up to 25 point penalty
        
        return max(0.0, score)
    
    def _calculate_queue_efficiency(self) -> float:
        """Calculate queue processing efficiency."""
        if not self.queue_times:
            return 1.0
        
        # Efficiency based on queue time consistency
        mean_time = np.mean(self.queue_times)
        std_time = np.std(self.queue_times)
        
        if mean_time == 0:
            return 1.0
        
        # Lower coefficient of variation = higher efficiency
        cv = std_time / mean_time
        efficiency = 1.0 / (1.0 + cv)
        
        return efficiency
    
    def clear(self):
        """Clear all analysis data."""
        self.match_history.clear()
        self.queue_times.clear()
        self.price_improvements.clear()
        self.fairness_metrics = {
            'total_matches': 0,
            'price_time_priority_violations': 0,
            'average_queue_time_ns': 0,
            'queue_time_variance': 0,
        }
    
    def analyze_performance(self, order_book_stats: Dict[str, Any], 
                          performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze FIFO algorithm performance with order book and system metrics."""
        fairness_report = self.get_fairness_report()
        queue_stats = self.get_queue_time_statistics()
        price_stats = self.get_price_improvement_statistics()
        
        return {
            'algorithm': 'FIFO',
            'fairness_analysis': fairness_report,
            'queue_dynamics': {
                'queue_time_stats': queue_stats,
                'price_improvement_stats': price_stats,
            },
            'performance_summary': {
                'total_matches': self.fairness_metrics['total_matches'],
                'average_queue_time_ms': queue_stats.get('mean_queue_time_ns', 0) / 1_000_000,
                'fairness_score': fairness_report.get('fairness_score', 0),
                'priority_violations': self.fairness_metrics['price_time_priority_violations'],
            },
            'recommendations': self._generate_recommendations(order_book_stats, performance_metrics)
        }
    
    def _generate_recommendations(self, order_book_stats: Dict[str, Any], 
                                performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze queue times
        queue_stats = self.get_queue_time_statistics()
        if queue_stats.get('mean_queue_time_ns', 0) > 1_000_000:  # > 1ms
            recommendations.append("Consider optimizing order matching speed - high queue times detected")
        
        # Analyze fairness
        if self.fairness_metrics['price_time_priority_violations'] > 0:
            recommendations.append("Price-time priority violations detected - review matching logic")
        
        # Analyze throughput
        if performance_metrics.get('throughput', 0) < 1000:
            recommendations.append("Low throughput detected - consider batch processing optimization")
        
        return recommendations


class FIFOOptimizer:
    """
    Optimizer for FIFO matching performance.
    
    Provides recommendations and optimizations for FIFO
    algorithm configuration and performance tuning.
    """
    
    def __init__(self):
        self.performance_history = []
        self.optimization_suggestions = []
    
    def analyze_performance(self, analyzer: FIFOAnalyzer) -> Dict[str, Any]:
        """Analyze FIFO performance and provide optimization suggestions."""
        fairness_report = analyzer.get_fairness_report()
        queue_dynamics = analyzer.analyze_queue_dynamics()
        
        suggestions = []
        
        # Analyze queue time performance
        queue_stats = fairness_report.get('queue_time_stats', {})
        if queue_stats:
            mean_queue_time = queue_stats.get('mean_queue_time_ns', 0)
            p99_queue_time = queue_stats.get('p99_queue_time_ns', 0)
            
            if mean_queue_time > 1_000_000:  # > 1ms
                suggestions.append({
                    'type': 'performance',
                    'severity': 'high',
                    'message': 'High average queue time detected',
                    'recommendation': 'Consider optimizing order processing pipeline'
                })
            
            if p99_queue_time > 10_000_000:  # > 10ms
                suggestions.append({
                    'type': 'latency',
                    'severity': 'critical',
                    'message': 'P99 queue time exceeds 10ms',
                    'recommendation': 'Investigate latency spikes and optimize hot paths'
                })
        
        # Analyze fairness
        fairness_score = fairness_report.get('fairness_score', 100)
        if fairness_score < 80:
            suggestions.append({
                'type': 'fairness',
                'severity': 'medium',
                'message': f'Fairness score below threshold: {fairness_score:.1f}',
                'recommendation': 'Review priority violation causes'
            })
        
        # Analyze queue efficiency
        if queue_dynamics:
            efficiency = queue_dynamics.get('queue_efficiency', 1.0)
            if efficiency < 0.8:
                suggestions.append({
                    'type': 'efficiency',
                    'severity': 'medium',
                    'message': f'Queue efficiency below optimal: {efficiency:.2f}',
                    'recommendation': 'Consider batch processing or queue optimization'
                })
        
        self.optimization_suggestions = suggestions
        
        return {
            'performance_score': self._calculate_performance_score(fairness_report, queue_dynamics),
            'suggestions': suggestions,
            'metrics_summary': {
                'fairness_score': fairness_score,
                'queue_efficiency': queue_dynamics.get('queue_efficiency', 1.0),
                'total_matches': fairness_report.get('total_matches', 0),
            }
        }
    
    def _calculate_performance_score(self, fairness_report: Dict, queue_dynamics: Dict) -> float:
        """Calculate overall performance score."""
        fairness_score = fairness_report.get('fairness_score', 100)
        efficiency = queue_dynamics.get('queue_efficiency', 1.0) * 100
        
        # Weighted average
        performance_score = (fairness_score * 0.6 + efficiency * 0.4)
        
        return performance_score
