"""
Pro-Rata matching algorithm implementation.

This module provides optimized Pro-Rata matching logic with
proportional allocation and detailed analytics.
"""

import numpy as np
from numba import jit, types
from typing import List, Tuple, Dict, Any
from core.order import Order, Trade
import time


@jit(nopython=True, cache=True)
def calculate_pro_rata_allocation(order_quantity: int, total_quantity: int, 
                                 available_quantity: int, min_allocation: int = 1) -> int:
    """
    Calculate pro-rata allocation for an order.
    
    Args:
        order_quantity: Size of the order requesting allocation
        total_quantity: Total quantity available at this price level
        available_quantity: Quantity available for allocation
        min_allocation: Minimum allocation size
        
    Returns:
        int: Allocated quantity
    """
    if total_quantity <= 0 or available_quantity <= 0:
        return 0
    
    # Calculate proportional allocation
    allocation_ratio = order_quantity / total_quantity
    raw_allocation = int(available_quantity * allocation_ratio)
    
    # Ensure minimum allocation if possible
    if raw_allocation < min_allocation and available_quantity >= min_allocation:
        return min_allocation
    
    return max(0, raw_allocation)


@jit(nopython=True, cache=True)
def pro_rata_batch_allocation(order_quantities: np.ndarray, available_quantity: int,
                             min_allocation: int = 1) -> np.ndarray:
    """
    Batch pro-rata allocation for multiple orders.
    
    Args:
        order_quantities: Array of order quantities
        available_quantity: Total quantity to allocate
        min_allocation: Minimum allocation per order
        
    Returns:
        np.ndarray: Allocated quantities for each order
    """
    n_orders = len(order_quantities)
    allocations = np.zeros(n_orders, dtype=np.int64)
    
    if available_quantity <= 0 or n_orders == 0:
        return allocations
    
    total_quantity = np.sum(order_quantities)
    if total_quantity <= 0:
        return allocations
    
    remaining_quantity = available_quantity
    
    # First pass: calculate proportional allocations
    for i in range(n_orders):
        if remaining_quantity <= 0:
            break
        
        allocation = calculate_pro_rata_allocation(
            order_quantities[i], total_quantity, available_quantity, min_allocation
        )
        
        # Don't exceed remaining quantity
        allocation = min(allocation, remaining_quantity)
        allocations[i] = allocation
        remaining_quantity -= allocation
    
    # Second pass: distribute any remaining quantity
    # Give priority to larger orders for remaining quantity
    if remaining_quantity > 0:
        # Sort indices by order size (descending)
        sorted_indices = np.argsort(-order_quantities)
        
        for idx in sorted_indices:
            if remaining_quantity <= 0:
                break
            
            # Give one additional unit to larger orders
            if order_quantities[idx] > 0:
                allocations[idx] += 1
                remaining_quantity -= 1
    
    return allocations


@jit(nopython=True, cache=True)
def pro_rata_match_single(buy_price: float, buy_qty: int, 
                         sell_price: float, sell_qty: int,
                         total_sell_qty: int) -> Tuple[float, int, bool]:
    """
    Execute a single pro-rata match.
    
    Args:
        buy_price, buy_qty: Buy order details
        sell_price, sell_qty: Sell order details
        total_sell_qty: Total sell quantity at this price level
        
    Returns:
        Tuple[float, int, bool]: (execution_price, execution_quantity, can_match)
    """
    # Check if orders can match
    if buy_price < sell_price:
        return 0.0, 0, False
    
    # Calculate pro-rata allocation
    if total_sell_qty > 0:
        allocation_ratio = sell_qty / total_sell_qty
        max_fill = min(buy_qty, sell_qty)
        execution_quantity = int(max_fill * allocation_ratio)
    else:
        execution_quantity = 0
    
    # Use sell price for pro-rata (market convention)
    execution_price = sell_price
    
    return execution_price, execution_quantity, execution_quantity > 0


class ProRataAnalyzer:
    """
    Analyzer for Pro-Rata matching algorithm performance and fairness.
    
    Provides detailed insights into allocation patterns, fairness metrics,
    and market impact analysis.
    """
    
    def __init__(self):
        self.allocation_history = []
        self.fairness_metrics = {
            'total_allocations': 0,
            'allocation_variance': 0.0,
            'small_order_penalty': 0.0,
            'large_order_advantage': 0.0,
        }
        self.size_bias_analysis = {
            'small_orders': [],  # < 100 shares
            'medium_orders': [], # 100-1000 shares
            'large_orders': [],  # > 1000 shares
        }
    
    def record_allocation(self, orders: List[Order], allocations: List[int], 
                         available_quantity: int, timestamp: int):
        """Record a pro-rata allocation for analysis."""
        if len(orders) != len(allocations):
            return
        
        total_requested = sum(order.remaining_quantity for order in orders)
        total_allocated = sum(allocations)
        
        allocation_record = {
            'timestamp': timestamp,
            'available_quantity': available_quantity,
            'total_requested': total_requested,
            'total_allocated': total_allocated,
            'fill_rate': total_allocated / max(1, available_quantity),
            'demand_ratio': total_requested / max(1, available_quantity),
            'order_details': []
        }
        
        # Analyze individual order allocations
        for order, allocation in zip(orders, allocations):
            order_size = order.remaining_quantity
            fill_rate = allocation / max(1, order_size)
            
            order_detail = {
                'order_id': order.order_id,
                'order_size': order_size,
                'allocation': allocation,
                'fill_rate': fill_rate,
                'size_category': self._categorize_order_size(order_size)
            }
            
            allocation_record['order_details'].append(order_detail)
            
            # Update size bias analysis
            self._update_size_bias_analysis(order_size, fill_rate)
        
        self.allocation_history.append(allocation_record)
        self.fairness_metrics['total_allocations'] += 1
        
        # Update fairness metrics
        self._update_fairness_metrics(allocation_record)
    
    def get_allocation_statistics(self) -> Dict[str, Any]:
        """Get comprehensive allocation statistics."""
        if not self.allocation_history:
            return {}
        
        # Calculate aggregate statistics
        fill_rates = [record['fill_rate'] for record in self.allocation_history]
        demand_ratios = [record['demand_ratio'] for record in self.allocation_history]
        
        # Individual order statistics
        all_order_details = []
        for record in self.allocation_history:
            all_order_details.extend(record['order_details'])
        
        if not all_order_details:
            return {}
        
        individual_fill_rates = [detail['fill_rate'] for detail in all_order_details]
        order_sizes = [detail['order_size'] for detail in all_order_details]
        
        return {
            'aggregate_stats': {
                'mean_fill_rate': float(np.mean(fill_rates)),
                'std_fill_rate': float(np.std(fill_rates)),
                'mean_demand_ratio': float(np.mean(demand_ratios)),
                'total_allocations': len(self.allocation_history),
            },
            'individual_order_stats': {
                'mean_individual_fill_rate': float(np.mean(individual_fill_rates)),
                'std_individual_fill_rate': float(np.std(individual_fill_rates)),
                'fill_rate_variance': float(np.var(individual_fill_rates)),
                'total_orders_analyzed': len(all_order_details),
            },
            'order_size_correlation': float(np.corrcoef(order_sizes, individual_fill_rates)[0, 1])
            if len(order_sizes) > 1 else 0.0
        }
    
    def get_size_bias_analysis(self) -> Dict[str, Any]:
        """Analyze bias towards different order sizes."""
        analysis = {}
        
        for category, fill_rates in self.size_bias_analysis.items():
            if fill_rates:
                analysis[category] = {
                    'count': len(fill_rates),
                    'mean_fill_rate': float(np.mean(fill_rates)),
                    'std_fill_rate': float(np.std(fill_rates)),
                    'median_fill_rate': float(np.median(fill_rates)),
                }
        
        # Calculate bias metrics
        if all(category in analysis for category in ['small_orders', 'large_orders']):
            small_mean = analysis['small_orders']['mean_fill_rate']
            large_mean = analysis['large_orders']['mean_fill_rate']
            
            analysis['size_bias_metrics'] = {
                'large_order_advantage': large_mean - small_mean,
                'bias_ratio': large_mean / max(small_mean, 1e-10),
                'fairness_score': 1.0 - abs(large_mean - small_mean),
            }
        
        return analysis
    
    def get_fairness_report(self) -> Dict[str, Any]:
        """Generate comprehensive fairness report for Pro-Rata."""
        allocation_stats = self.get_allocation_statistics()
        size_bias = self.get_size_bias_analysis()
        
        # Calculate overall fairness score
        fairness_score = self._calculate_fairness_score(allocation_stats, size_bias)
        
        return {
            'algorithm': 'Pro-Rata',
            'fairness_score': fairness_score,
            'allocation_statistics': allocation_stats,
            'size_bias_analysis': size_bias,
            'fairness_metrics': self.fairness_metrics,
            'recommendations': self._generate_recommendations(allocation_stats, size_bias)
        }
    
    def analyze_market_impact(self) -> Dict[str, Any]:
        """Analyze market impact of pro-rata allocation."""
        if not self.allocation_history:
            return {}
        
        # Analyze allocation efficiency over time
        timestamps = [record['timestamp'] for record in self.allocation_history]
        fill_rates = [record['fill_rate'] for record in self.allocation_history]
        demand_ratios = [record['demand_ratio'] for record in self.allocation_history]
        
        # Calculate market pressure indicators
        high_demand_periods = [ratio > 2.0 for ratio in demand_ratios]
        high_demand_count = sum(high_demand_periods)
        
        # Analyze allocation consistency
        fill_rate_consistency = 1.0 - (np.std(fill_rates) / max(np.mean(fill_rates), 1e-10))
        
        return {
            'market_pressure': {
                'high_demand_periods': high_demand_count,
                'high_demand_ratio': high_demand_count / len(self.allocation_history),
                'average_demand_ratio': float(np.mean(demand_ratios)),
                'max_demand_ratio': float(np.max(demand_ratios)),
            },
            'allocation_consistency': {
                'consistency_score': float(fill_rate_consistency),
                'fill_rate_volatility': float(np.std(fill_rates)),
                'predictability_index': self._calculate_predictability_index(),
            },
            'efficiency_metrics': {
                'average_utilization': float(np.mean(fill_rates)),
                'utilization_variance': float(np.var(fill_rates)),
                'allocation_efficiency': self._calculate_allocation_efficiency(),
            }
        }
    
    def _categorize_order_size(self, size: int) -> str:
        """Categorize order by size."""
        if size < 100:
            return 'small_orders'
        elif size <= 1000:
            return 'medium_orders'
        else:
            return 'large_orders'
    
    def _update_size_bias_analysis(self, order_size: int, fill_rate: float):
        """Update size bias analysis with new data point."""
        category = self._categorize_order_size(order_size)
        self.size_bias_analysis[category].append(fill_rate)
    
    def _update_fairness_metrics(self, allocation_record: Dict):
        """Update running fairness metrics."""
        order_details = allocation_record['order_details']
        if not order_details:
            return
        
        fill_rates = [detail['fill_rate'] for detail in order_details]
        self.fairness_metrics['allocation_variance'] = float(np.var(fill_rates))
        
        # Update size bias metrics
        small_orders = [detail for detail in order_details 
                       if detail['size_category'] == 'small_orders']
        large_orders = [detail for detail in order_details 
                       if detail['size_category'] == 'large_orders']
        
        if small_orders and large_orders:
            small_avg = np.mean([detail['fill_rate'] for detail in small_orders])
            large_avg = np.mean([detail['fill_rate'] for detail in large_orders])
            
            self.fairness_metrics['small_order_penalty'] = max(0, small_avg - large_avg)
            self.fairness_metrics['large_order_advantage'] = max(0, large_avg - small_avg)
    
    def _calculate_fairness_score(self, allocation_stats: Dict, size_bias: Dict) -> float:
        """Calculate overall fairness score (0-100)."""
        score = 100.0
        
        # Penalize high allocation variance
        if allocation_stats and 'individual_order_stats' in allocation_stats:
            variance = allocation_stats['individual_order_stats'].get('fill_rate_variance', 0)
            score -= min(variance * 100, 30)  # Up to 30 point penalty
        
        # Penalize size bias
        if 'size_bias_metrics' in size_bias:
            bias_ratio = size_bias['size_bias_metrics'].get('bias_ratio', 1.0)
            bias_penalty = abs(bias_ratio - 1.0) * 20  # Up to 20 point penalty
            score -= min(bias_penalty, 20)
        
        # Penalize large order advantage
        large_advantage = self.fairness_metrics.get('large_order_advantage', 0)
        score -= min(large_advantage * 50, 25)  # Up to 25 point penalty
        
        return max(0.0, score)
    
    def _calculate_predictability_index(self) -> float:
        """Calculate how predictable allocations are."""
        if len(self.allocation_history) < 2:
            return 1.0
        
        fill_rates = [record['fill_rate'] for record in self.allocation_history]
        
        # Calculate autocorrelation as predictability measure
        if len(fill_rates) > 1:
            correlation = np.corrcoef(fill_rates[:-1], fill_rates[1:])[0, 1]
            return abs(correlation) if not np.isnan(correlation) else 0.0
        
        return 0.0
    
    def _calculate_allocation_efficiency(self) -> float:
        """Calculate overall allocation efficiency."""
        if not self.allocation_history:
            return 1.0
        
        total_available = sum(record['available_quantity'] for record in self.allocation_history)
        total_allocated = sum(record['total_allocated'] for record in self.allocation_history)
        
        if total_available > 0:
            return total_allocated / total_available
        
        return 1.0
    
    def _generate_recommendations(self, allocation_stats: Dict, size_bias: Dict) -> List[str]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # Check allocation variance
        if allocation_stats and 'individual_order_stats' in allocation_stats:
            variance = allocation_stats['individual_order_stats'].get('fill_rate_variance', 0)
            if variance > 0.1:  # High variance threshold
                recommendations.append(
                    "High allocation variance detected. Consider implementing minimum allocation rules."
                )
        
        # Check size bias
        if 'size_bias_metrics' in size_bias:
            bias_ratio = size_bias['size_bias_metrics'].get('bias_ratio', 1.0)
            if bias_ratio > 1.2:  # 20% bias threshold
                recommendations.append(
                    "Large order bias detected. Consider size-weighted fairness adjustments."
                )
            elif bias_ratio < 0.8:
                recommendations.append(
                    "Small order bias detected. Review minimum allocation policies."
                )
        
        # Check overall fairness
        fairness_score = self._calculate_fairness_score(allocation_stats, size_bias)
        if fairness_score < 70:
            recommendations.append(
                "Low fairness score. Consider hybrid allocation mechanisms."
            )
        
        return recommendations
    
    def clear(self):
        """Clear all analysis data."""
        self.allocation_history.clear()
        self.fairness_metrics = {
            'total_allocations': 0,
            'allocation_variance': 0.0,
            'small_order_penalty': 0.0,
            'large_order_advantage': 0.0,
        }
        for category in self.size_bias_analysis:
            self.size_bias_analysis[category].clear()
    
    def analyze_performance(self, order_book_stats: Dict[str, Any], 
                          performance_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze Pro-Rata algorithm performance with order book and system metrics."""
        fairness_report = self.get_fairness_report()
        allocation_stats = self.get_allocation_statistics()
        size_bias = self.get_size_bias_analysis()
        
        return {
            'algorithm': 'PRO_RATA',
            'fairness_analysis': fairness_report,
            'allocation_dynamics': {
                'allocation_stats': allocation_stats,
                'size_bias_analysis': size_bias,
            },
            'performance_summary': {
                'total_allocations': self.fairness_metrics['total_allocations'],
                'allocation_variance': self.fairness_metrics['allocation_variance'],
                'fairness_score': fairness_report.get('fairness_score', 0),
                'size_bias_score': size_bias.get('size_bias_metrics', {}).get('bias_score', 0),
            },
            'recommendations': self._generate_recommendations(order_book_stats, performance_metrics)
        }
    
    def _generate_recommendations(self, order_book_stats: Dict[str, Any], 
                                performance_metrics: Dict[str, Any]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        # Analyze allocation variance
        if self.fairness_metrics['allocation_variance'] > 0.5:
            recommendations.append("High allocation variance detected - consider minimum allocation thresholds")
        
        # Analyze size bias
        if self.fairness_metrics['small_order_penalty'] > 0.2:
            recommendations.append("Small orders experiencing significant penalty - review allocation algorithm")
        
        # Analyze throughput
        if performance_metrics.get('throughput', 0) < 500:
            recommendations.append("Low throughput for Pro-Rata - consider optimizing allocation calculations")
        
        return recommendations


class ProRataOptimizer:
    """
    Optimizer for Pro-Rata matching performance and fairness.
    
    Provides configuration recommendations and parameter tuning
    for optimal pro-rata allocation behavior.
    """
    
    def __init__(self):
        self.optimization_history = []
        self.parameter_recommendations = {}
    
    def optimize_parameters(self, analyzer: ProRataAnalyzer) -> Dict[str, Any]:
        """Optimize pro-rata parameters based on historical performance."""
        fairness_report = analyzer.get_fairness_report()
        market_impact = analyzer.analyze_market_impact()
        
        recommendations = {
            'min_allocation_size': self._recommend_min_allocation(fairness_report),
            'size_weighting_factor': self._recommend_size_weighting(fairness_report),
            'fairness_threshold': self._recommend_fairness_threshold(fairness_report),
            'allocation_rounding': self._recommend_rounding_strategy(fairness_report),
        }
        
        optimization_score = self._calculate_optimization_score(fairness_report, market_impact)
        
        return {
            'optimization_score': optimization_score,
            'parameter_recommendations': recommendations,
            'performance_impact': self._estimate_performance_impact(recommendations),
            'implementation_priority': self._prioritize_recommendations(recommendations),
        }
    
    def _recommend_min_allocation(self, fairness_report: Dict) -> int:
        """Recommend minimum allocation size."""
        size_bias = fairness_report.get('size_bias_analysis', {})
        
        if 'small_orders' in size_bias:
            small_order_stats = size_bias['small_orders']
            mean_fill_rate = small_order_stats.get('mean_fill_rate', 0)
            
            if mean_fill_rate < 0.1:  # Very low fill rate for small orders
                return 5  # Increase minimum allocation
            elif mean_fill_rate < 0.3:
                return 3
            else:
                return 1  # Standard minimum
        
        return 1
    
    def _recommend_size_weighting(self, fairness_report: Dict) -> float:
        """Recommend size weighting factor."""
        size_bias = fairness_report.get('size_bias_analysis', {})
        
        if 'size_bias_metrics' in size_bias:
            bias_ratio = size_bias['size_bias_metrics'].get('bias_ratio', 1.0)
            
            if bias_ratio > 1.5:  # Strong large order bias
                return 0.8  # Reduce size weighting
            elif bias_ratio < 0.7:  # Strong small order bias
                return 1.2  # Increase size weighting
        
        return 1.0  # Standard weighting
    
    def _recommend_fairness_threshold(self, fairness_report: Dict) -> float:
        """Recommend fairness threshold for allocation adjustments."""
        fairness_score = fairness_report.get('fairness_score', 100)
        
        if fairness_score < 60:
            return 0.9  # High threshold for fairness adjustments
        elif fairness_score < 80:
            return 0.8  # Medium threshold
        else:
            return 0.7  # Standard threshold
    
    def _recommend_rounding_strategy(self, fairness_report: Dict) -> str:
        """Recommend allocation rounding strategy."""
        allocation_stats = fairness_report.get('allocation_statistics', {})
        
        if 'individual_order_stats' in allocation_stats:
            variance = allocation_stats['individual_order_stats'].get('fill_rate_variance', 0)
            
            if variance > 0.2:
                return 'probabilistic'  # Use probabilistic rounding for fairness
            else:
                return 'standard'  # Standard floor rounding
        
        return 'standard'
    
    def _calculate_optimization_score(self, fairness_report: Dict, market_impact: Dict) -> float:
        """Calculate overall optimization potential score."""
        fairness_score = fairness_report.get('fairness_score', 100)
        
        efficiency_metrics = market_impact.get('efficiency_metrics', {})
        allocation_efficiency = efficiency_metrics.get('allocation_efficiency', 1.0)
        
        # Weighted score combining fairness and efficiency
        optimization_score = (fairness_score * 0.6 + allocation_efficiency * 100 * 0.4)
        
        return optimization_score
    
    def _estimate_performance_impact(self, recommendations: Dict) -> Dict[str, str]:
        """Estimate performance impact of recommendations."""
        impact = {}
        
        min_allocation = recommendations.get('min_allocation_size', 1)
        if min_allocation > 1:
            impact['latency'] = 'minimal_increase'
            impact['fairness'] = 'significant_improvement'
        
        size_weighting = recommendations.get('size_weighting_factor', 1.0)
        if abs(size_weighting - 1.0) > 0.1:
            impact['computation'] = 'slight_increase'
            impact['fairness'] = 'moderate_improvement'
        
        rounding_strategy = recommendations.get('allocation_rounding', 'standard')
        if rounding_strategy == 'probabilistic':
            impact['computation'] = 'moderate_increase'
            impact['fairness'] = 'significant_improvement'
        
        return impact
    
    def _prioritize_recommendations(self, recommendations: Dict) -> List[str]:
        """Prioritize implementation of recommendations."""
        priorities = []
        
        min_allocation = recommendations.get('min_allocation_size', 1)
        if min_allocation > 1:
            priorities.append('min_allocation_size')
        
        size_weighting = recommendations.get('size_weighting_factor', 1.0)
        if abs(size_weighting - 1.0) > 0.2:
            priorities.append('size_weighting_factor')
        
        fairness_threshold = recommendations.get('fairness_threshold', 0.7)
        if fairness_threshold > 0.8:
            priorities.append('fairness_threshold')
        
        rounding_strategy = recommendations.get('allocation_rounding', 'standard')
        if rounding_strategy != 'standard':
            priorities.append('allocation_rounding')
        
        return priorities
