"""
Performance monitoring system for the high-performance order book.

This module provides comprehensive performance tracking, metrics collection,
and real-time monitoring capabilities with minimal overhead.
"""

import time
import threading
import numpy as np
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from dataclasses import dataclass
import json
import psutil
import os

from .magic_trace_wrapper import MagicTraceProfiler, profile_function


@dataclass
class PerformanceMetric:
    """Individual performance metric data point."""
    name: str
    value: float
    timestamp: int
    unit: str
    category: str
    metadata: Dict[str, Any] = None


class MetricsCollector:
    """
    High-performance metrics collector with circular buffers.
    
    Designed for minimal overhead collection of performance metrics
    with configurable retention and aggregation policies.
    """
    
    def __init__(self, max_samples: int = 10000):
        self.max_samples = max_samples
        self.metrics = defaultdict(lambda: deque(maxlen=max_samples))
        self.aggregated_metrics = {}
        self._lock = threading.RLock()
        
        # Performance tracking
        self.collection_overhead_ns = deque(maxlen=1000)
        self.last_aggregation_time = 0
        self.aggregation_interval_ns = 1_000_000_000  # 1 second
    
    def record_metric(self, name: str, value: float, unit: str = "", 
                     category: str = "general", metadata: Dict = None):
        """Record a performance metric with minimal overhead."""
        start_time = time.time_ns()
        
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=start_time,
            unit=unit,
            category=category,
            metadata=metadata or {}
        )
        
        with self._lock:
            self.metrics[name].append(metric)
            
            # Periodic aggregation
            if start_time - self.last_aggregation_time > self.aggregation_interval_ns:
                self._update_aggregated_metrics()
                self.last_aggregation_time = start_time
        
        # Track collection overhead
        overhead = time.time_ns() - start_time
        self.collection_overhead_ns.append(overhead)
    
    def get_metric_statistics(self, name: str, window_seconds: int = 60) -> Dict[str, float]:
        """Get statistics for a specific metric over a time window."""
        with self._lock:
            if name not in self.metrics:
                return {}
            
            # Filter by time window
            cutoff_time = time.time_ns() - (window_seconds * 1_000_000_000)
            recent_metrics = [m for m in self.metrics[name] if m.timestamp >= cutoff_time]
            
            if not recent_metrics:
                return {}
            
            values = [m.value for m in recent_metrics]
            values_array = np.array(values)
            
            return {
                'count': len(values),
                'mean': float(np.mean(values_array)),
                'median': float(np.median(values_array)),
                'std': float(np.std(values_array)),
                'min': float(np.min(values_array)),
                'max': float(np.max(values_array)),
                'p95': float(np.percentile(values_array, 95)),
                'p99': float(np.percentile(values_array, 99)),
                'sum': float(np.sum(values_array)),
                'rate_per_second': len(values) / window_seconds if window_seconds > 0 else 0,
            }
    
    def get_all_metrics_summary(self, window_seconds: int = 60) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        summary = {}
        with self._lock:
            for metric_name in self.metrics.keys():
                summary[metric_name] = self.get_metric_statistics(metric_name, window_seconds)
        return summary
    
    def get_collection_overhead_stats(self) -> Dict[str, float]:
        """Get statistics about metrics collection overhead."""
        if not self.collection_overhead_ns:
            return {}
        
        overhead_array = np.array(list(self.collection_overhead_ns))
        
        return {
            'mean_overhead_ns': float(np.mean(overhead_array)),
            'max_overhead_ns': float(np.max(overhead_array)),
            'p95_overhead_ns': float(np.percentile(overhead_array, 95)),
            'total_samples': len(overhead_array),
        }
    
    def _update_aggregated_metrics(self):
        """Update aggregated metrics for faster access."""
        for metric_name, metric_deque in self.metrics.items():
            if not metric_deque:
                continue
            
            recent_values = [m.value for m in list(metric_deque)[-100:]]  # Last 100 samples
            if recent_values:
                self.aggregated_metrics[metric_name] = {
                    'current': recent_values[-1],
                    'recent_mean': float(np.mean(recent_values)),
                    'recent_max': float(np.max(recent_values)),
                    'sample_count': len(metric_deque),
                    'last_updated': time.time_ns(),
                }
    
    def clear_metrics(self, metric_name: str = None):
        """Clear metrics data."""
        with self._lock:
            if metric_name:
                if metric_name in self.metrics:
                    self.metrics[metric_name].clear()
                if metric_name in self.aggregated_metrics:
                    del self.aggregated_metrics[metric_name]
            else:
                self.metrics.clear()
                self.aggregated_metrics.clear()


class SystemResourceMonitor:
    """
    Monitor system resources (CPU, memory, etc.) with minimal overhead.
    """
    
    def __init__(self, sample_interval_seconds: float = 1.0):
        self.sample_interval = sample_interval_seconds
        self.is_monitoring = False
        self.monitor_thread = None
        self.metrics_collector = MetricsCollector()
        
        # Process handle for efficient monitoring
        self.process = psutil.Process()
        
    def start_monitoring(self):
        """Start system resource monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system resource monitoring."""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
    
    def get_current_resources(self) -> Dict[str, float]:
        """Get current resource usage."""
        try:
            # CPU usage
            cpu_percent = self.process.cpu_percent()
            
            # Memory usage
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            
            # System-wide metrics
            system_cpu = psutil.cpu_percent()
            system_memory = psutil.virtual_memory().percent
            
            return {
                'process_cpu_percent': cpu_percent,
                'process_memory_mb': memory_mb,
                'system_cpu_percent': system_cpu,
                'system_memory_percent': system_memory,
                'process_threads': self.process.num_threads(),
                'process_fds': self.process.num_fds() if hasattr(self.process, 'num_fds') else 0,
            }
        except Exception as e:
            print(f"Error getting resource metrics: {e}")
            return {}
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.is_monitoring:
            try:
                resources = self.get_current_resources()
                timestamp = time.time_ns()
                
                for metric_name, value in resources.items():
                    self.metrics_collector.record_metric(
                        name=metric_name,
                        value=value,
                        unit="percent" if "percent" in metric_name else "count",
                        category="system_resources"
                    )
                
                time.sleep(self.sample_interval)
                
            except Exception as e:
                print(f"Error in resource monitoring loop: {e}")
                time.sleep(self.sample_interval)


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system for the order book.
    
    Integrates magic-trace profiling, metrics collection, and system monitoring
    to provide complete visibility into system performance.
    """
    
    def __init__(self, enable_magic_trace: bool = True):
        self.enable_magic_trace = enable_magic_trace
        
        # Core components
        self.metrics_collector = MetricsCollector()
        self.system_monitor = SystemResourceMonitor()
        self.magic_trace_profiler = MagicTraceProfiler() if enable_magic_trace else None
        
        # Performance tracking
        self.start_time = time.time_ns()
        self.session_metrics = {
            'orders_processed': 0,
            'trades_executed': 0,
            'total_volume': 0.0,
            'peak_throughput_ops_per_sec': 0.0,
            'peak_memory_mb': 0.0,
        }
        
        # Alert thresholds
        self.alert_thresholds = {
            'latency_p99_ms': 10.0,
            'memory_usage_mb': 1000.0,
            'cpu_usage_percent': 80.0,
            'throughput_ops_per_sec': 1000.0,
        }
        
        # Alert callbacks
        self.alert_callbacks = []
        
        # Lock for thread safety
        self._lock = threading.RLock()
    
    def start_monitoring(self):
        """Start all monitoring components."""
        print("Starting performance monitoring...")
        
        # Start system resource monitoring
        self.system_monitor.start_monitoring()
        
        # Record start time
        self.start_time = time.time_ns()
        
        print("Performance monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring components."""
        print("Stopping performance monitoring...")
        
        # Stop system monitoring
        self.system_monitor.stop_monitoring()
        
        # Stop any active magic-trace sessions
        if self.magic_trace_profiler:
            self.magic_trace_profiler.stop_all_sessions()
        
        print("Performance monitoring stopped")
    
    @profile_function("record_order_processing")
    def record_order_processing(self, processing_time_ns: int, order_count: int = 1):
        """Record order processing performance."""
        with self._lock:
            self.session_metrics['orders_processed'] += order_count
            
            # Record latency
            latency_ms = processing_time_ns / 1_000_000
            self.metrics_collector.record_metric(
                name="order_processing_latency_ms",
                value=latency_ms,
                unit="milliseconds",
                category="performance"
            )
            
            # Calculate and record throughput
            current_time = time.time_ns()
            uptime_seconds = (current_time - self.start_time) / 1_000_000_000
            
            if uptime_seconds > 0:
                throughput = self.session_metrics['orders_processed'] / uptime_seconds
                self.metrics_collector.record_metric(
                    name="throughput_ops_per_sec",
                    value=throughput,
                    unit="ops/sec",
                    category="performance"
                )
                
                # Update peak throughput
                self.session_metrics['peak_throughput_ops_per_sec'] = max(
                    self.session_metrics['peak_throughput_ops_per_sec'], throughput
                )
            
            # Check for alerts
            self._check_performance_alerts(latency_ms, throughput if uptime_seconds > 0 else 0)
    
    @profile_function("record_trade_execution")
    def record_trade_execution(self, trade_count: int, total_volume: float):
        """Record trade execution metrics."""
        with self._lock:
            self.session_metrics['trades_executed'] += trade_count
            self.session_metrics['total_volume'] += total_volume
            
            self.metrics_collector.record_metric(
                name="trades_per_batch",
                value=trade_count,
                unit="count",
                category="trading"
            )
            
            self.metrics_collector.record_metric(
                name="trade_volume",
                value=total_volume,
                unit="currency",
                category="trading"
            )
    
    def profile_session(self, session_name: str = None):
        """Get a magic-trace profiling session context manager."""
        if self.magic_trace_profiler:
            return self.magic_trace_profiler.profile_session(session_name)
        else:
            # Return a no-op context manager
            from contextlib import nullcontext
            return nullcontext()
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            current_time = time.time_ns()
            uptime_seconds = (current_time - self.start_time) / 1_000_000_000
            
            # Get metrics statistics
            metrics_summary = self.metrics_collector.get_all_metrics_summary(window_seconds=60)
            
            # Get system resources
            system_resources = self.system_monitor.get_current_resources()
            
            # Get magic-trace performance data
            magic_trace_summary = {}
            if self.magic_trace_profiler:
                magic_trace_summary = self.magic_trace_profiler.get_performance_summary()
            
            # Calculate derived metrics
            avg_throughput = (self.session_metrics['orders_processed'] / uptime_seconds 
                            if uptime_seconds > 0 else 0)
            
            summary = {
                'session_info': {
                    'uptime_seconds': uptime_seconds,
                    'start_time': self.start_time,
                    'current_time': current_time,
                },
                'session_metrics': dict(self.session_metrics),
                'derived_metrics': {
                    'average_throughput_ops_per_sec': avg_throughput,
                    'orders_per_trade': (self.session_metrics['orders_processed'] / 
                                       max(1, self.session_metrics['trades_executed'])),
                    'average_trade_size': (self.session_metrics['total_volume'] / 
                                         max(1, self.session_metrics['trades_executed'])),
                },
                'performance_metrics': metrics_summary,
                'system_resources': system_resources,
                'magic_trace_data': magic_trace_summary,
                'collection_overhead': self.metrics_collector.get_collection_overhead_stats(),
                'alert_thresholds': self.alert_thresholds,
            }
            
            return summary
    
    def print_summary(self):
        """Print a formatted performance summary."""
        summary = self.get_performance_summary()
        
        print("\n" + "="*60)
        print("PERFORMANCE SUMMARY")
        print("="*60)
        
        # Session info
        session_info = summary['session_info']
        print(f"Uptime: {session_info['uptime_seconds']:.2f} seconds")
        
        # Key metrics
        session_metrics = summary['session_metrics']
        derived_metrics = summary['derived_metrics']
        
        print(f"Orders Processed: {session_metrics['orders_processed']:,}")
        print(f"Trades Executed: {session_metrics['trades_executed']:,}")
        print(f"Total Volume: {session_metrics['total_volume']:,.2f}")
        print(f"Average Throughput: {derived_metrics['average_throughput_ops_per_sec']:.1f} ops/sec")
        print(f"Peak Throughput: {session_metrics['peak_throughput_ops_per_sec']:.1f} ops/sec")
        
        # Performance metrics
        perf_metrics = summary['performance_metrics']
        if 'order_processing_latency_ms' in perf_metrics:
            latency_stats = perf_metrics['order_processing_latency_ms']
            print(f"Latency - Mean: {latency_stats.get('mean', 0):.3f}ms, "
                  f"P95: {latency_stats.get('p95', 0):.3f}ms, "
                  f"P99: {latency_stats.get('p99', 0):.3f}ms")
        
        # System resources
        resources = summary['system_resources']
        if resources:
            print(f"CPU Usage: {resources.get('process_cpu_percent', 0):.1f}%")
            print(f"Memory Usage: {resources.get('process_memory_mb', 0):.1f} MB")
        
        # Collection overhead
        overhead = summary['collection_overhead']
        if overhead:
            print(f"Monitoring Overhead: {overhead.get('mean_overhead_ns', 0)/1000:.1f} μs/sample")
        
        print("="*60)
    
    def register_alert_callback(self, callback: Callable[[str, Dict], None]):
        """Register a callback for performance alerts."""
        self.alert_callbacks.append(callback)
    
    def set_alert_threshold(self, metric: str, threshold: float):
        """Set alert threshold for a metric."""
        self.alert_thresholds[metric] = threshold
    
    def export_metrics(self, filename: str = None) -> str:
        """Export metrics to JSON file."""
        if filename is None:
            filename = f"performance_metrics_{int(time.time())}.json"
        
        summary = self.get_performance_summary()
        
        try:
            with open(filename, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"Metrics exported to: {filename}")
            return filename
            
        except Exception as e:
            print(f"Failed to export metrics: {e}")
            return ""
    
    def _check_performance_alerts(self, latency_ms: float, throughput_ops_per_sec: float):
        """Check for performance alerts and trigger callbacks."""
        alerts = []
        
        # Check latency
        if latency_ms > self.alert_thresholds.get('latency_p99_ms', float('inf')):
            alerts.append({
                'type': 'high_latency',
                'metric': 'order_processing_latency_ms',
                'value': latency_ms,
                'threshold': self.alert_thresholds['latency_p99_ms'],
                'timestamp': time.time_ns(),
            })
        
        # Check throughput
        if throughput_ops_per_sec < self.alert_thresholds.get('throughput_ops_per_sec', 0):
            alerts.append({
                'type': 'low_throughput',
                'metric': 'throughput_ops_per_sec',
                'value': throughput_ops_per_sec,
                'threshold': self.alert_thresholds['throughput_ops_per_sec'],
                'timestamp': time.time_ns(),
            })
        
        # Check system resources
        resources = self.system_monitor.get_current_resources()
        
        memory_mb = resources.get('process_memory_mb', 0)
        if memory_mb > self.alert_thresholds.get('memory_usage_mb', float('inf')):
            alerts.append({
                'type': 'high_memory_usage',
                'metric': 'process_memory_mb',
                'value': memory_mb,
                'threshold': self.alert_thresholds['memory_usage_mb'],
                'timestamp': time.time_ns(),
            })
        
        cpu_percent = resources.get('process_cpu_percent', 0)
        if cpu_percent > self.alert_thresholds.get('cpu_usage_percent', float('inf')):
            alerts.append({
                'type': 'high_cpu_usage',
                'metric': 'process_cpu_percent',
                'value': cpu_percent,
                'threshold': self.alert_thresholds['cpu_usage_percent'],
                'timestamp': time.time_ns(),
            })
        
        # Trigger alert callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert['type'], alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()


# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get the global performance monitor instance."""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor
