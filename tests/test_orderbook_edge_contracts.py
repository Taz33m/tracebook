import threading

import pytest

from tracebook.core.order import OrderFactory, OrderSide
from tracebook.core.orderbook import OrderBook, OrderBookManager


def test_submit_order_rejects_invalid_existing_order_without_side_effects():
    order_book = OrderBook("BTCUSD")
    callbacks = []
    order_book.register_order_callback(lambda order, trades: callbacks.append((order, trades)))

    wrong_symbol = OrderFactory().create_limit_order("ETHUSD", OrderSide.BUY, 100.0, 1.0)
    result = order_book.submit_order(wrong_symbol)

    assert result.order is wrong_symbol
    assert result.trades == []
    assert result.rested is False
    assert result.cancelled is False
    assert "does not match book symbol" in result.rejected_reason
    assert callbacks == []
    assert order_book.get_active_order_ids() == []

    stats = order_book.get_statistics()
    assert stats["orders_added"] == 0
    assert stats["trades_executed"] == 0
    assert stats["total_orders_processed"] == 0


def test_submit_fok_rejection_is_reported_and_does_not_consume_resting_liquidity():
    order_book = OrderBook("BTCUSD")
    ask = order_book.submit_limit_order(OrderSide.SELL, 101.0, 2.0)

    rejected = order_book.submit_fok_order(OrderSide.BUY, 101.0, 3.0)

    assert rejected.trades == []
    assert rejected.rested is False
    assert rejected.cancelled is True
    assert rejected.rejected_reason == "FOK order could not be fully filled"
    assert order_book.get_order(ask.order.order_id) is not None
    assert order_book.get_statistics()["sell_side_quantity"] == pytest.approx(2.0)


def test_replace_order_that_crosses_spread_executes_and_removes_original_order():
    order_book = OrderBook("BTCUSD")
    bid = order_book.submit_limit_order(OrderSide.BUY, 99.0, 2.0)
    ask = order_book.submit_limit_order(OrderSide.SELL, 101.0, 1.5)

    replacement = order_book.replace_order(bid.order.order_id, price=102.0, quantity=1.5)

    assert replacement.rejected_reason is None
    assert replacement.trades
    assert replacement.rested is False
    assert replacement.cancelled is False
    assert replacement.order.order_id != bid.order.order_id
    assert order_book.get_order(bid.order.order_id) is None
    assert order_book.get_order(replacement.order.order_id) is None
    assert order_book.get_order(ask.order.order_id) is None
    assert [trade.price for trade in replacement.trades] == pytest.approx([101.0])
    assert [trade.quantity for trade in replacement.trades] == pytest.approx([1.5])

    stats = order_book.get_statistics()
    assert stats["orders_cancelled"] == 1
    assert stats["trades_executed"] == 1
    assert stats["buy_side_quantity"] == pytest.approx(0.0)
    assert stats["sell_side_quantity"] == pytest.approx(0.0)


def test_manager_global_statistics_do_not_sum_rates_or_averages():
    class FakeBook:
        def __init__(self, stats):
            self.stats = stats

        def get_statistics(self):
            return dict(self.stats)

    manager = OrderBookManager()
    manager.order_books = {
        "A": FakeBook(
            {
                "orders_added": 2,
                "orders_cancelled": 1,
                "trades_executed": 1,
                "total_volume": 100.0,
                "total_orders_processed": 2,
                "total_matches": 1,
                "total_trades": 1,
                "buy_side_orders": 1,
                "sell_side_orders": 0,
                "buy_side_quantity": 2.0,
                "sell_side_quantity": 0.0,
                "avg_processing_time_ns": 10.0,
                "max_processing_time_ns": 30,
                "min_processing_time_ns": 5,
                "uptime_seconds": 4.0,
                "orders_per_second": 999.0,
                "trades_per_second": 999.0,
                "last_trade_time": 12,
            }
        ),
        "B": FakeBook(
            {
                "orders_added": 6,
                "orders_cancelled": 0,
                "trades_executed": 3,
                "total_volume": 300.0,
                "total_orders_processed": 6,
                "total_matches": 3,
                "total_trades": 3,
                "buy_side_orders": 0,
                "sell_side_orders": 2,
                "buy_side_quantity": 0.0,
                "sell_side_quantity": 4.0,
                "avg_processing_time_ns": 20.0,
                "max_processing_time_ns": 40,
                "min_processing_time_ns": 8,
                "uptime_seconds": 5.0,
                "orders_per_second": 999.0,
                "trades_per_second": 999.0,
                "last_trade_time": 20,
            }
        ),
        "EMPTY": FakeBook(
            {
                "orders_added": 0,
                "min_processing_time_ns": 0,
                "uptime_seconds": 1.0,
            }
        ),
    }

    stats = manager.get_global_statistics()

    assert stats["orders_added"] == 8
    assert stats["trades_executed"] == 4
    assert stats["avg_processing_time_ns"] == pytest.approx(17.5)
    assert stats["max_processing_time_ns"] == 40
    assert stats["min_processing_time_ns"] == 5
    assert stats["uptime_seconds"] == pytest.approx(5.0)
    assert stats["orders_per_second"] == pytest.approx(8 / 5)
    assert stats["trades_per_second"] == pytest.approx(4 / 5)
    assert stats["last_trade_time"] == 20


def test_manager_rejects_symbol_mismatch_for_existing_book():
    manager = OrderBookManager()
    order_book = OrderBook("BTCUSD")

    with pytest.raises(ValueError, match="does not match registry key"):
        manager.add_order_book("ETHUSD", order_book)

    assert manager.get_all_symbols() == []
    assert manager.get_order_book("ETHUSD") is None


def test_order_callbacks_run_after_book_lock_is_released():
    order_book = OrderBook("BTCUSD")
    callback_observed_blocking = []

    def callback(order, trades):
        thread = threading.Thread(target=order_book.get_best_bid)
        thread.start()
        thread.join(timeout=0.2)
        callback_observed_blocking.append(thread.is_alive())
        thread.join(timeout=1.0)

    order_book.register_order_callback(callback)

    order_book.add_limit_order(OrderSide.BUY, 100.0, 1.0)

    assert callback_observed_blocking == [False]
