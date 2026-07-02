import pytest

from tracebook.core.order import OrderFactory, OrderSide, OrderType
from tracebook.core.orderbook import OrderBook, OrderBookManager
from tracebook.core.price_level import PriceLevel


def assert_close(actual, expected):
    assert actual == pytest.approx(expected)


def test_limit_orders_support_fractional_quantities_without_negative_resting_state():
    order_book = OrderBook("BTCUSD")

    assert order_book.add_limit_order(OrderSide.BUY, 50000.0, 1.0) == []
    trades = order_book.add_limit_order(OrderSide.SELL, 49999.0, 0.5)

    assert len(trades) == 1
    assert_close(trades[0].quantity, 0.5)
    assert_close(trades[0].price, 50000.0)

    stats = order_book.get_statistics()
    assert_close(stats["buy_side_quantity"], 0.5)
    assert_close(stats["sell_side_quantity"], 0.0)
    assert stats["buy_side_orders"] == 1
    assert stats["sell_side_orders"] == 0


def test_market_buy_executes_at_resting_ask_and_never_rests():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.SELL, 101.0, 5.0)

    trades = order_book.add_market_order(OrderSide.BUY, 2.0)

    assert len(trades) == 1
    assert_close(trades[0].price, 101.0)
    assert_close(trades[0].quantity, 2.0)

    stats = order_book.get_statistics()
    assert stats["buy_side_orders"] == 0
    assert_close(stats["buy_side_quantity"], 0.0)
    assert stats["sell_side_orders"] == 1
    assert_close(stats["sell_side_quantity"], 3.0)


def test_unfilled_market_order_is_cancelled_instead_of_resting():
    order_book = OrderBook("BTCUSD")

    trades = order_book.add_market_order(OrderSide.BUY, 2.0)

    assert trades == []
    stats = order_book.get_statistics()
    assert stats["buy_side_orders"] == 0
    assert_close(stats["buy_side_quantity"], 0.0)


def test_ioc_order_cancels_unfilled_remainder():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.SELL, 101.0, 2.0)

    trades = order_book.add_ioc_order(OrderSide.BUY, 101.0, 5.0)

    assert len(trades) == 1
    assert_close(trades[0].quantity, 2.0)

    stats = order_book.get_statistics()
    assert stats["buy_side_orders"] == 0
    assert stats["sell_side_orders"] == 0


def test_fok_order_does_not_partially_fill_or_mutate_book():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.SELL, 101.0, 2.0)

    rejected_trades = order_book.add_fok_order(OrderSide.BUY, 101.0, 5.0)
    assert rejected_trades == []
    assert_close(order_book.get_statistics()["sell_side_quantity"], 2.0)

    filled_trades = order_book.add_fok_order(OrderSide.BUY, 101.0, 2.0)
    assert len(filled_trades) == 1
    assert_close(filled_trades[0].quantity, 2.0)
    assert_close(order_book.get_statistics()["sell_side_quantity"], 0.0)


def test_pro_rata_allocates_decimal_quantity_across_price_level():
    order_book = OrderBook("BTCUSD", matching_algorithm="pro_rata")
    order_book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.SELL, 100.0, 3.0)

    trades = order_book.add_limit_order(OrderSide.BUY, 100.0, 4.0)

    assert len(trades) == 2
    assert [trade.quantity for trade in trades] == pytest.approx([1.0, 3.0])
    assert_close(order_book.get_statistics()["sell_side_quantity"], 0.0)


def test_order_book_manager_exposes_copy_for_dashboard_reads():
    manager = OrderBookManager()
    order_book = manager.create_order_book("BTCUSD")

    all_books = manager.get_all_order_books()
    all_books.clear()

    assert manager.get_order_book("BTCUSD") is order_book


def test_cancel_order_removes_resting_liquidity():
    order_book = OrderBook("BTCUSD")
    result = order_book.submit_limit_order(OrderSide.BUY, 100.0, 2.0)

    assert result.rested is True
    assert order_book.cancel_order(result.order.order_id) is True
    assert order_book.cancel_order(result.order.order_id) is False
    assert order_book.get_active_order_ids() == []
    assert_close(order_book.get_statistics()["buy_side_quantity"], 0.0)


def test_replace_order_cancels_old_order_and_submits_new_order():
    order_book = OrderBook("BTCUSD")
    result = order_book.submit_limit_order(OrderSide.BUY, 100.0, 2.0)

    replacement = order_book.replace_order(result.order.order_id, price=101.0, quantity=1.25)

    assert replacement.rejected_reason is None
    assert replacement.order.order_id != result.order.order_id
    assert order_book.get_order(result.order.order_id) is None
    assert order_book.get_order(replacement.order.order_id) is not None
    assert order_book.get_best_bid() == pytest.approx(101.0)
    assert_close(order_book.get_statistics()["buy_side_quantity"], 1.25)


def test_limit_buy_sweeps_multiple_ask_levels():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.SELL, 101.0, 2.0)
    order_book.add_limit_order(OrderSide.SELL, 102.0, 3.0)

    trades = order_book.add_limit_order(OrderSide.BUY, 101.0, 4.0)

    assert [trade.price for trade in trades] == pytest.approx([100.0, 101.0])
    assert [trade.quantity for trade in trades] == pytest.approx([1.0, 2.0])
    assert_close(order_book.get_statistics()["sell_side_quantity"], 3.0)
    assert_close(order_book.get_statistics()["buy_side_quantity"], 1.0)


def test_limit_sell_matches_highest_bid_first_without_resting_crossed_order():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.BUY, 105.0, 1.0)

    trades = order_book.add_limit_order(OrderSide.SELL, 104.0, 1.0)

    assert len(trades) == 1
    assert trades[0].price == pytest.approx(105.0)
    assert trades[0].quantity == pytest.approx(1.0)
    assert order_book.get_best_bid() == pytest.approx(100.0)
    assert order_book.get_best_ask() is None


def test_market_sell_executes_at_highest_resting_bid():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.BUY, 105.0, 1.0)

    trades = order_book.add_market_order(OrderSide.SELL, 1.0)

    assert len(trades) == 1
    assert trades[0].price == pytest.approx(105.0)
    assert order_book.get_best_bid() == pytest.approx(100.0)


def test_fok_sell_uses_executable_bid_depth_without_mutating_on_reject():
    order_book = OrderBook("BTCUSD")
    order_book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.BUY, 105.0, 1.0)

    rejected = order_book.submit_fok_order(OrderSide.SELL, 104.0, 2.0)
    filled = order_book.submit_fok_order(OrderSide.SELL, 104.0, 1.0)

    assert rejected.trades == []
    assert rejected.rejected_reason == "FOK order could not be fully filled"
    assert len(filled.trades) == 1
    assert filled.trades[0].price == pytest.approx(105.0)
    assert order_book.get_best_bid() == pytest.approx(100.0)


def test_pro_rata_sell_allocates_against_bid_level():
    order_book = OrderBook("BTCUSD", matching_algorithm="pro_rata")
    order_book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.BUY, 100.0, 3.0)

    trades = order_book.add_limit_order(OrderSide.SELL, 100.0, 2.0)

    assert [trade.quantity for trade in trades] == pytest.approx([0.5, 1.5])
    assert_close(order_book.get_statistics()["buy_side_quantity"], 2.0)


def test_fifo_priority_fills_resting_orders_in_arrival_order():
    order_book = OrderBook("BTCUSD")
    first = order_book.submit_limit_order(OrderSide.SELL, 100.0, 1.0)
    second = order_book.submit_limit_order(OrderSide.SELL, 100.0, 1.0)

    trades = order_book.add_limit_order(OrderSide.BUY, 100.0, 1.5)

    assert [trade.sell_order_id for trade in trades] == [
        first.order.order_id,
        second.order.order_id,
    ]
    assert [trade.quantity for trade in trades] == pytest.approx([1.0, 0.5])


def test_pro_rata_partial_fill_allocates_proportionally():
    order_book = OrderBook("BTCUSD", matching_algorithm="pro_rata")
    order_book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    order_book.add_limit_order(OrderSide.SELL, 100.0, 3.0)

    trades = order_book.add_limit_order(OrderSide.BUY, 100.0, 2.0)

    assert [trade.quantity for trade in trades] == pytest.approx([0.5, 1.5])
    assert_close(order_book.get_statistics()["sell_side_quantity"], 2.0)


def test_invalid_orders_are_rejected_or_raise_without_mutating_book():
    order_book = OrderBook("BTCUSD")
    rejected = order_book.submit_limit_order(OrderSide.BUY, -1.0, 1.0)

    assert rejected.rejected_reason
    assert order_book.get_active_order_ids() == []

    factory = OrderFactory()
    wrong_symbol = factory.create_limit_order("ETHUSD", OrderSide.BUY, 100.0, 1.0)
    with pytest.raises(ValueError, match="does not match"):
        order_book.add_order(wrong_symbol)

    with pytest.raises(ValueError):
        factory.create_order("BTCUSD", OrderSide.BUY, OrderType.LIMIT, 100.0, 0.0)


def test_invalid_numeric_state_is_rejected_without_resting():
    order_book = OrderBook("BTCUSD")
    factory = OrderFactory()

    with pytest.raises(ValueError):
        factory.create_limit_order("BTCUSD", OrderSide.BUY, float("nan"), 1.0)

    bad_price = factory.create_limit_order("BTCUSD", OrderSide.BUY, 100.0, 1.0)
    bad_price.price = float("nan")
    assert order_book.submit_order(bad_price).rejected_reason == "Order price must be finite"

    bad_quantity = factory.create_limit_order("BTCUSD", OrderSide.BUY, 100.0, 1.0)
    bad_quantity.remaining_quantity = 2.0
    assert (
        order_book.submit_order(bad_quantity).rejected_reason == "Order quantity must be positive"
    )

    assert order_book.get_active_order_ids() == []


def test_duplicate_active_order_id_is_rejected_without_phantom_liquidity():
    order_book = OrderBook("BTCUSD")
    result = order_book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)

    duplicate = order_book.submit_order(result.order)

    assert "already active" in duplicate.rejected_reason
    assert order_book.get_active_order_ids() == [result.order.order_id]
    assert order_book.cancel_order(result.order.order_id) is True
    assert order_book.get_active_order_ids() == []
    assert order_book.get_order_book_depth()["bids"] == []


def test_fifo_execution_uses_resting_price_even_with_mutated_aggressor_timestamp():
    order_book = OrderBook("BTCUSD")
    ask = order_book.submit_limit_order(OrderSide.SELL, 100.0, 1.0).order
    buy = order_book.order_factory.create_limit_order("BTCUSD", OrderSide.BUY, 101.0, 1.0)
    buy.timestamp = ask.timestamp - 1

    trades = order_book.add_order(buy)

    assert len(trades) == 1
    assert trades[0].price == pytest.approx(100.0)


def test_snapshot_stays_consistent_after_cancel_and_replace():
    order_book = OrderBook("BTCUSD")
    bid = order_book.submit_limit_order(OrderSide.BUY, 99.0, 2.0)
    ask = order_book.submit_limit_order(OrderSide.SELL, 101.0, 3.0)

    order_book.cancel_order(bid.order.order_id)
    order_book.replace_order(ask.order.order_id, price=102.0, quantity=1.5)
    snapshot = order_book.get_market_data_snapshot()

    assert snapshot.bid_levels == []
    assert len(snapshot.ask_levels) == 1
    assert snapshot.ask_levels[0][0] == pytest.approx(102.0)
    assert snapshot.ask_levels[0][1] == pytest.approx(1.5)
    assert snapshot.ask_levels[0][2] == 1
    assert snapshot.best_ask == pytest.approx(102.0)


def test_order_and_trade_are_slotted_plain_objects():
    from tracebook.core.order import OrderFactory, Trade

    order = OrderFactory().create_limit_order("BTCUSD", OrderSide.BUY, 100.0, 2.0, owner=5)

    # Plain __slots__ objects: no per-instance __dict__, attributes are stored directly.
    assert not hasattr(order, "__dict__")
    assert order.owner == 5
    assert order.remaining_quantity == pytest.approx(2.0)
    assert order.is_buy() and order.is_limit_order() and order.can_rest()
    assert order.fill(0.5) == pytest.approx(0.5)
    assert order.remaining_quantity == pytest.approx(1.5)
    assert order.can_match_price(99.0) and not order.can_match_price(101.0)

    market = OrderFactory().create_market_order("BTCUSD", OrderSide.SELL, 1.0)
    assert market.is_market_order() and not market.can_rest()
    assert market.can_match_price(50.0)  # market matches any price

    trade = Trade(1, 2, 100.0, 0.5, 123)
    assert not hasattr(trade, "__dict__")
    assert (trade.buy_order_id, trade.sell_order_id, trade.quantity) == (1, 2, pytest.approx(0.5))


def test_price_level_missing_remove_is_noop():
    level = PriceLevel(100.0)
    level.add_order(1, 2.0)

    assert level.remove_order(2, 1.0) is False
    assert level.order_count == 1
    assert level.total_quantity == pytest.approx(2.0)


def test_price_level_preserves_fifo_order_across_removals():
    level = PriceLevel(100.0)
    for order_id in (10, 20, 30, 40):
        level.add_order(order_id, 1.0)

    # Removing a middle order does not disturb the arrival order of the rest.
    assert level.remove_order(20, 1.0) is True
    assert list(level.orders) == [10, 30, 40]
    assert level.get_first_order_id() == 10

    # Removing the head advances the FIFO front.
    assert level.remove_order(10, 1.0) is True
    assert level.get_first_order_id() == 30
    assert level.order_count == 2
    assert level.total_quantity == pytest.approx(2.0)
