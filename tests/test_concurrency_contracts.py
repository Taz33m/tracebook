"""Concurrency contract tests: user callbacks never fire while a lock is held.

Holding a lock across a user callback is a deadlock vector (a callback that waits
on another thread which needs the same lock hangs forever). These pin that both
the order-book replace path and the performance-monitor alert path deliver
callbacks with their lock released.
"""

import threading

from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook
from tracebook.profiling.performance_monitor import PerformanceMonitor


def _lock_free_from_other_thread(lock) -> bool:
    """True if a separate thread can acquire `lock` right now (i.e. it is free)."""
    out = []

    def probe():
        got = lock.acquire(blocking=False)
        out.append(got)
        if got:
            lock.release()

    thread = threading.Thread(target=probe)
    thread.start()
    thread.join()
    return out[0]


def test_replace_order_callback_fires_without_the_book_lock_held():
    book = OrderBook("BTCUSD")
    observed = {}

    def on_order(order, trades):
        observed["lock_free"] = _lock_free_from_other_thread(book._lock)

    book.register_order_callback(on_order)
    resting = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)

    observed.clear()
    book.replace_order(resting.order.order_id, price=99.0)

    assert observed.get("lock_free") is True


def test_alert_callback_fires_without_the_monitor_lock_held():
    monitor = PerformanceMonitor(enable_magic_trace=False)
    # Trip a memory alert using the sampled snapshot (no psutil on this path).
    monitor.system_monitor.latest_resources = {"process_memory_mb": 99999.0}
    monitor.set_alert_threshold("memory_usage_mb", 1.0)
    observed = {}

    def on_alert(kind, alert):
        observed["lock_free"] = _lock_free_from_other_thread(monitor._lock)

    monitor.register_alert_callback(on_alert)
    monitor.record_order_processing(1_000_000, 1)

    assert observed.get("lock_free") is True
