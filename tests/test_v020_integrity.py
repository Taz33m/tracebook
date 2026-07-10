"""Regression coverage for the 0.2.0 public-boundary integrity pass."""

import pytest

from tracebook import OrderBook, OrderSide, SelfTradePolicy, replay
from tracebook.core.order import Order


@pytest.mark.parametrize(
    "policy",
    [SelfTradePolicy.CANCEL_RESTING, SelfTradePolicy.CANCEL_INCOMING],
)
def test_replace_preserves_owner_and_self_trade_prevention(policy):
    book = OrderBook("BTCUSD", self_trade_policy=policy)
    original = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=42)

    replacement = book.replace_order(original.order.order_id, price=100.0)
    incoming = book.submit_limit_order(OrderSide.SELL, 100.0, 1.0, owner=42)

    assert replacement.order.owner == 42
    assert incoming.trades == []
    assert book.get_statistics()["self_trades_prevented"] == 1


def test_public_order_results_and_lookups_are_detached_from_live_state():
    book = OrderBook("BTCUSD")
    submitted = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    order_id = submitted.order.order_id

    submitted.order.price = 250.0
    looked_up = book.get_order(order_id)
    looked_up.price = 300.0
    looked_up.remaining_quantity = 99.0

    live = book.get_order(order_id)
    assert live.price == pytest.approx(100.0)
    assert live.remaining_quantity == pytest.approx(1.0)
    assert book.cancel_order(order_id) is True
    assert book.get_order_book_depth()["bids"] == []


def test_external_order_object_is_not_adopted_as_live_mutable_state():
    book = OrderBook("BTCUSD")
    external = Order(77, "BTCUSD", 1, 2, 100.0, 1.0, 1, -1)

    accepted = book.submit_order(external)
    external.price = 200.0
    external.remaining_quantity = 50.0

    live = book.get_order(accepted.order.order_id)
    assert live.price == pytest.approx(100.0)
    assert live.remaining_quantity == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("field", "value", "reason"),
    [
        ("side", 1.5, "Unsupported order side"),
        ("order_type", 2.5, "Unsupported order type"),
        ("price", "100", "price must be numeric"),
        ("owner", 1.5, "owner must be an integer"),
        ("timestamp", 1.5, "timestamp must be"),
    ],
)
def test_submit_order_returns_structured_rejection_for_malformed_fields(field, value, reason):
    book = OrderBook("BTCUSD")
    order = Order(90, "BTCUSD", 1, 2, 100.0, 1.0, 1, -1)
    setattr(order, field, value)

    result = book.submit_order(order)

    assert result.accepted is False
    assert reason in result.rejected_reason
    assert book.get_active_order_ids() == []


def test_submit_order_rejects_non_order_without_leaking_it_into_result():
    result = OrderBook("BTCUSD").submit_order({"not": "an order"})

    assert result.accepted is False
    assert result.order is None
    assert "Expected an Order" in result.rejected_reason


def test_order_book_normalizes_and_validates_matching_algorithm():
    assert OrderBook("BTCUSD", matching_algorithm=" FIFO ").matching_algorithm == "fifo"

    with pytest.raises(ValueError, match="matching_algorithm"):
        OrderBook("BTCUSD", matching_algorithm=1)
    with pytest.raises(ValueError, match="matching_algorithm"):
        OrderBook("BTCUSD", matching_algorithm="price_time")


def test_factory_reports_int64_order_id_exhaustion_as_structured_rejection():
    book = OrderBook("BTCUSD")
    book.order_factory.advance_past(2**63 - 1)

    result = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)

    assert result.accepted is False
    assert result.order is None
    assert "allocator exhausted" in result.rejected_reason


def test_clear_is_recorded_and_replayed_without_resurrecting_liquidity():
    book = OrderBook("BTCUSD")
    log = book.start_recording()
    book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    book.clear()
    book.submit_limit_order(OrderSide.SELL, 101.0, 1.0)
    book.stop_recording()

    rebuilt = replay(log)

    assert [event.op for event in log.events] == ["submit", "clear", "submit"]
    assert rebuilt.get_order_book_depth()["bids"] == []
    assert rebuilt.get_order_book_depth()["asks"] == [(101.0, 1.0, 1)]


def test_pro_rata_fok_cancel_incoming_uses_level_wide_non_self_liquidity():
    book = OrderBook(
        "BTCUSD",
        matching_algorithm="pro_rata",
        self_trade_policy=SelfTradePolicy.CANCEL_INCOMING,
    )
    book.submit_limit_order(OrderSide.SELL, 100.0, 1.0, owner=7)
    other = book.submit_limit_order(OrderSide.SELL, 100.0, 2.0, owner=8)

    fok = book.submit_fok_order(OrderSide.BUY, 100.0, 2.0, owner=7)

    assert fok.rejected_reason is None
    assert sum(trade.quantity for trade in fok.trades) == pytest.approx(2.0)
    assert book.get_order(other.order.order_id) is None


def test_callbacks_receive_detached_order_trade_and_market_data_objects():
    book = OrderBook("BTCUSD")
    observed_market_levels = []

    def mutate_order(order, trades):
        order.price = 999.0
        if trades:
            trades[0].price = 1.0

    def mutate_trades(trades):
        trades[0].price = 2.0

    def mutate_market(snapshot):
        snapshot.bid_levels.clear()

    def observe_market(snapshot):
        observed_market_levels.append(list(snapshot.bid_levels))

    book.register_order_callback(mutate_order)
    book.register_trade_callback(mutate_trades)
    book.register_market_data_callback(mutate_market)
    book.register_market_data_callback(observe_market)
    resting = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    result = book.submit_limit_order(OrderSide.SELL, 100.0, 1.0)

    assert resting.order.price == pytest.approx(100.0)
    assert result.trades[0].price == pytest.approx(100.0)
    assert book.get_recent_trades()[0].price == pytest.approx(100.0)
    assert observed_market_levels[0] == [(100.0, 1.0, 1)]


def test_state_snapshot_validates_bounds_and_returns_detached_trades():
    book = OrderBook("BTCUSD")
    book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    book.submit_limit_order(OrderSide.SELL, 100.0, 1.0)

    state = book.get_state_snapshot(levels=1, trade_count=1)
    state["trades"][0].price = 1.0

    assert state["best_bid"] is None
    assert state["best_ask"] is None
    assert book.get_recent_trades()[0].price == pytest.approx(100.0)
    with pytest.raises(ValueError, match="trade_count"):
        book.get_state_snapshot(trade_count=-1)


@pytest.mark.parametrize("value", [True, 1.5, "1"])
def test_public_history_and_depth_counts_reject_ambiguous_types(value):
    book = OrderBook("BTCUSD")

    with pytest.raises(ValueError, match="non-negative integer"):
        book.get_order_book_depth(value)
    with pytest.raises(ValueError, match="integer"):
        book.get_recent_trades(value)
    with pytest.raises(ValueError, match="non-negative integer"):
        book.get_state_snapshot(levels=value)


def test_negative_history_count_keeps_empty_result_compatibility():
    book = OrderBook("BTCUSD")

    assert book.get_recent_trades(-1) == []
    with pytest.raises(ValueError, match="non-negative integer"):
        book.get_order_book_depth(-1)
    with pytest.raises(ValueError, match="non-negative integer"):
        book.get_state_snapshot(levels=-1)


@pytest.mark.parametrize("value", [True, 1.5, "1", 0, -1])
def test_order_id_apis_require_positive_integers(value):
    book = OrderBook("BTCUSD")

    with pytest.raises(ValueError, match="positive integer"):
        book.cancel_order(value)
    with pytest.raises(ValueError, match="positive integer"):
        book.get_order(value)
    assert book.replace_order(value).accepted is False
