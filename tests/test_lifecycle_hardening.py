"""Tests for order lifecycle hardening: atomic replace and bounded id memory."""

from collections import deque

import pytest

from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook, OrderResult


def test_trade_history_is_bounded_while_total_stays_cumulative():
    book = OrderBook("BTCUSD")
    # Shrink the retained-trade window for the test.
    book.matching_engine.trades = deque(maxlen=5)

    for _ in range(20):
        book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
        book.add_limit_order(OrderSide.SELL, 100.0, 1.0)  # crosses -> one trade

    assert len(book.matching_engine.trades) == 5  # bounded tail, not 20
    assert book.get_statistics()["total_trades"] == 20  # cumulative count preserved
    assert len(book.get_recent_trades(3)) == 3  # tail slice still works on the deque


def test_replace_restores_original_when_replacement_submit_fails(monkeypatch):
    book = OrderBook("BTCUSD")
    resting = book.submit_limit_order(OrderSide.BUY, 100.0, 2.0)
    original_id = resting.order.order_id
    cancels_before = book.get_statistics()["orders_cancelled"]

    # Force the replacement submission to fail *after* the cancel step.
    def _fail_submit(order):
        return OrderResult(order, [], False, False, "forced failure")

    monkeypatch.setattr(book, "submit_order", _fail_submit)

    result = book.replace_order(original_id, price=101.0, quantity=1.5)

    assert result.rejected_reason == "forced failure"
    # Original resting order is restored, unchanged.
    restored = book.get_order(original_id)
    assert restored is not None
    assert restored.price == pytest.approx(100.0)
    assert restored.remaining_quantity == pytest.approx(2.0)
    assert book.get_active_order_ids(OrderSide.BUY) == [original_id]
    # The rolled-back cancel is not counted.
    assert book.get_statistics()["orders_cancelled"] == cancels_before


def test_replace_with_invalid_params_leaves_original_untouched():
    book = OrderBook("BTCUSD")
    resting = book.submit_limit_order(OrderSide.BUY, 100.0, 2.0)
    original_id = resting.order.order_id

    result = book.replace_order(original_id, price=-5.0)

    assert result.rejected_reason
    assert book.get_order(original_id) is not None
    assert book.get_active_order_ids(OrderSide.BUY) == [original_id]


def test_seen_order_id_memory_is_bounded():
    book = OrderBook("BTCUSD")
    book._seen_id_cap = 10  # shrink the window for the test

    for _ in range(50):
        book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
        # Keep the book from growing without bound; cancel each resting order.
        for order_id in list(book.get_active_order_ids(OrderSide.BUY)):
            book.cancel_order(order_id)

    # The replay guard never exceeds its cap.
    assert len(book._seen_order_ids) <= book._seen_id_cap
    assert len(book._seen_order_id_queue) <= book._seen_id_cap


def test_duplicate_active_order_id_is_still_rejected_within_window():
    book = OrderBook("BTCUSD")
    factory = book.order_factory
    order = factory.create_limit_order("BTCUSD", OrderSide.BUY, 100.0, 1.0)
    book.add_order(order)

    # Re-submitting the same id while it is active is rejected.
    with pytest.raises(ValueError, match="already active"):
        book.add_order(order)
