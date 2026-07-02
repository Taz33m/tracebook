"""Tests that performance metrics report honestly.

Throughput is a rolling-window rate rather than a lifetime average, and the
order-processing hot path never makes a synchronous psutil call.
"""

import pytest

from tracebook.profiling.performance_monitor import PerformanceMonitor


def test_throughput_is_windowed_not_lifetime_average():
    monitor = PerformanceMonitor(enable_magic_trace=False)

    for _ in range(3):
        monitor.record_order_processing(1_000_000, 1)  # 1 ms each, sub-window apart

    stats = monitor.metrics_collector.get_metric_statistics(
        "throughput_ops_per_sec", window_seconds=60
    )
    # 3 orders inside the 1 s window -> 3 ops/sec. The old lifetime-average
    # (orders / tiny uptime) would report an enormous number instead.
    assert stats["current"] == pytest.approx(3.0)


def test_alerts_do_not_call_psutil_on_the_hot_path(monkeypatch):
    monitor = PerformanceMonitor(enable_magic_trace=False)

    def _boom():
        raise AssertionError("psutil was called synchronously on the hot path")

    monkeypatch.setattr(monitor.system_monitor, "get_current_resources", _boom)

    # Reads the last sampled snapshot (empty here), so this must not raise.
    monitor.record_order_processing(1_000_000, 1)


def test_throughput_running_total_stays_consistent_with_window():
    monitor = PerformanceMonitor(enable_magic_trace=False)

    for _ in range(20):
        monitor.record_order_processing(500_000, 2)

    # The O(1) running total equals a full re-sum of the window contents.
    assert monitor._recent_orders_total == sum(c for _, c in monitor._recent_orders)


def test_latest_resources_snapshot_is_returned_without_psutil():
    monitor = PerformanceMonitor(enable_magic_trace=False)
    monitor.system_monitor.latest_resources = {"process_memory_mb": 42.0}

    assert monitor.system_monitor.get_latest_resources() == {"process_memory_mb": 42.0}
