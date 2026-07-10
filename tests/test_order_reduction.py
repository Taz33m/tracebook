import pytest

from tracebook import EventLog, OrderBook, OrderSide, replay
from tracebook.events import MarketEvent, MarketReplayError, replay_market_events


def test_reduce_order_preserves_fifo_priority_and_level_totals():
    book = OrderBook("BTCUSD")
    first = book.submit_limit_order(OrderSide.BUY, 100, 2).order
    second = book.submit_limit_order(OrderSide.BUY, 100, 1).order

    assert book.reduce_order(first.order_id, 1) is True
    assert book.get_order(first.order_id).remaining_quantity == pytest.approx(1)
    assert book.get_order_book_depth()["bids"] == [(100.0, 2.0, 2)]

    trades = book.add_market_order(OrderSide.SELL, 1.5)

    assert [(trade.buy_order_id, trade.quantity) for trade in trades] == [
        (first.order_id, 1.0),
        (second.order_id, 0.5),
    ]


def test_reduce_order_full_removal_and_invalid_reductions_are_atomic():
    book = OrderBook("BTCUSD")
    order = book.submit_limit_order(OrderSide.SELL, 101, 2).order

    with pytest.raises(ValueError, match="cannot exceed"):
        book.reduce_order(order.order_id, 3)
    assert book.get_order(order.order_id).remaining_quantity == pytest.approx(2)

    assert book.reduce_order(order.order_id, 2) is True
    assert book.get_order(order.order_id) is None
    assert book.reduce_order(order.order_id, 1) is False

    for bad in (True, 0, -1, 1.5):
        with pytest.raises(ValueError, match="Order id"):
            book.reduce_order(bad, 1)
    for bad in (True, 0, -1, float("nan")):
        with pytest.raises(ValueError, match="quantity"):
            book.reduce_order(1, bad)


def test_recorded_reduction_round_trips_in_schema_v2_and_v1_still_loads():
    book = OrderBook("BTCUSD")
    log = book.start_recording()
    order = book.submit_limit_order(OrderSide.BUY, 100, 3).order
    assert book.reduce_order(order.order_id, 1.25) is True
    book.stop_recording()

    restored = EventLog.from_json(log.to_json())
    rebuilt = replay(restored)

    assert restored.schema_version == 2
    assert [event.op for event in restored.events] == ["submit", "reduce"]
    assert rebuilt.get_order(order.order_id).remaining_quantity == pytest.approx(1.75)

    legacy = EventLog.from_dict({"schema_version": 1, "symbol": "BTCUSD", "events": []})
    assert legacy.schema_version == 1
    with pytest.raises(ValueError, match="does not support reduce"):
        EventLog.from_dict(
            {
                "schema_version": 1,
                "symbol": "BTCUSD",
                "events": [{"op": "reduce", "order_id": 1, "quantity": 0.5}],
            }
        )


def test_replay_rejects_malformed_reduce_records():
    missing_quantity = EventLog("BTCUSD")
    missing_quantity.events = []
    book = OrderBook("BTCUSD")
    log = book.start_recording()
    order = book.submit_limit_order(OrderSide.BUY, 100, 1).order
    book.stop_recording()

    log.events.append(type(log.events[0])(op="reduce", order_id=order.order_id))
    with pytest.raises(ValueError, match="has no quantity"):
        replay(log)

    assert replay(missing_quantity).get_active_order_ids() == []


def test_normalized_reduce_event_updates_source_id_in_place():
    events = [
        MarketEvent("new", "BTCUSD", 10, OrderSide.BUY, price=100, quantity=2),
        MarketEvent("reduce", "BTCUSD", 10, quantity=0.75),
    ]

    result = replay_market_events(events)
    engine_id = result.resolve_order_id("BTCUSD", 10)

    assert result.reductions == 1
    assert engine_id is not None
    assert result.manager.get_order_book("BTCUSD").get_order(engine_id).remaining_quantity == 1.25
    with pytest.raises(MarketReplayError, match="positive quantity"):
        MarketEvent("reduce", "BTCUSD", 10, quantity=0)
