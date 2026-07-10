"""Deterministic record/replay tests.

Replaying a recorded event log reconstructs the identical sequence of trades and
the identical final book state, including across a JSON round-trip.
"""

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tracebook import EventLog, Order, OrderBook, OrderSide, replay
from tracebook.core.order import OrderType
from tracebook.core.replay import RecordedEvent

_PRICES = st.sampled_from([98.0, 99.0, 100.0, 101.0, 102.0])
_QUANTITIES = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_SIDES = st.sampled_from([OrderSide.BUY, OrderSide.SELL])
_TYPES = st.sampled_from(
    [OrderType.LIMIT, OrderType.LIMIT, OrderType.MARKET, OrderType.IOC, OrderType.FOK]
)


@st.composite
def _actions(draw):
    otype = draw(_TYPES)
    side = draw(_SIDES)
    qty = draw(_QUANTITIES)
    price = None if otype == OrderType.MARKET else draw(_PRICES)
    return (otype, side, price, qty)


def _apply(book, action):
    otype, side, price, qty = action
    if otype == OrderType.LIMIT:
        return book.submit_limit_order(side, price, qty)
    if otype == OrderType.MARKET:
        return book.submit_market_order(side, qty)
    if otype == OrderType.IOC:
        return book.submit_ioc_order(side, price, qty)
    if otype == OrderType.FOK:
        return book.submit_fok_order(side, price, qty)
    raise AssertionError(otype)


def _trade_key(trade):
    return (
        trade.buy_order_id,
        trade.sell_order_id,
        round(trade.price, 9),
        round(trade.quantity, 9),
    )


def _trade_keys(book):
    return [_trade_key(t) for t in book.matching_engine.trades]


@pytest.mark.parametrize("algorithm", ["fifo", "pro_rata"])
@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(actions=st.lists(_actions(), min_size=1, max_size=30))
def test_replay_reproduces_trades_and_book_state(algorithm, actions):
    live = OrderBook("BTCUSD", matching_algorithm=algorithm)
    log = live.start_recording()

    for action in actions:
        result = _apply(live, action)
        # Occasionally cancel a resting order to exercise cancel recording.
        if result.rested and result.order.order_id % 3 == 0:
            live.cancel_order(result.order.order_id)

    live.stop_recording()

    replayed = replay(log)

    # Identical trade sequence and identical resting book.
    assert _trade_keys(replayed) == _trade_keys(live)
    assert replayed.get_order_book_depth(50)["bids"] == live.get_order_book_depth(50)["bids"]
    assert replayed.get_order_book_depth(50)["asks"] == live.get_order_book_depth(50)["asks"]


def test_event_log_survives_json_round_trip():
    live = OrderBook("ETHUSD")
    live.start_recording()
    live.add_limit_order(OrderSide.BUY, 3000.0, 2.0)
    live.add_limit_order(OrderSide.SELL, 2999.0, 1.0)
    resting = live.submit_limit_order(OrderSide.SELL, 3005.0, 1.0)
    live.cancel_order(resting.order.order_id)
    log = live.stop_recording()

    restored = EventLog.from_json(log.to_json())
    replayed = replay(restored)

    assert restored.schema_version == 1
    assert _trade_keys(replayed) == _trade_keys(live)
    assert replayed.get_best_bid() == live.get_best_bid()
    assert replayed.get_best_ask() == live.get_best_ask()


def test_event_log_rejects_unknown_schema_version():
    with pytest.raises(ValueError, match="schema_version"):
        EventLog.from_dict({"schema_version": 999, "symbol": "BTCUSD"})


def test_unfillable_fok_replays_without_diverging():
    live = OrderBook("BTCUSD")
    live.start_recording()
    live.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    # FOK needs more than is available -> soft rejection, no fill, no rest.
    live.add_fok_order(OrderSide.BUY, 100.0, 5.0)
    log = live.stop_recording()

    replayed = replay(log)  # must not raise on the soft FOK outcome

    assert _trade_keys(replayed) == _trade_keys(live)
    assert replayed.get_order_book_depth(10)["asks"] == live.get_order_book_depth(10)["asks"]


def test_replay_raises_on_hard_rejection_from_malformed_log():
    # Two submit events reuse the same order id: the second is a hard rejection.
    log = EventLog("BTCUSD")
    log.events = [
        RecordedEvent(
            op="submit",
            order_id=7,
            side=int(OrderSide.BUY),
            order_type=int(OrderType.LIMIT),
            price=100.0,
            quantity=1.0,
            symbol="BTCUSD",
            timestamp=0,
        ),
        RecordedEvent(
            op="submit",
            order_id=7,
            side=int(OrderSide.BUY),
            order_type=int(OrderType.LIMIT),
            price=100.0,
            quantity=1.0,
            symbol="BTCUSD",
            timestamp=0,
        ),
    ]

    with pytest.raises(ValueError, match="Replay diverged"):
        replay(log)


def test_replay_raises_on_cancel_with_no_matching_order():
    log = EventLog("BTCUSD")
    log.events = [RecordedEvent(op="cancel", order_id=999)]

    with pytest.raises(ValueError, match="Replay diverged"):
        replay(log)


def test_recording_can_be_restarted_and_stopped():
    book = OrderBook("BTCUSD")
    assert book.stop_recording() is None

    log = book.start_recording()
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0)
    assert len(log) == 1

    # The resting bid means the book is no longer empty; restarting is refused
    # because a fresh log cannot capture that pre-existing liquidity.
    with pytest.raises(ValueError, match="empty book"):
        book.start_recording()

    # On an empty book, restarting drops the prior log.
    book.clear()
    fresh = book.start_recording()
    assert len(fresh) == 0
    book.add_limit_order(OrderSide.SELL, 101.0, 1.0)
    assert len(fresh) == 1
    assert book.stop_recording() is fresh


def test_start_recording_requires_an_empty_book():
    book = OrderBook("BTCUSD")
    book.add_limit_order(OrderSide.SELL, 100.0, 1.0)  # pre-existing resting liquidity

    with pytest.raises(ValueError, match="empty book"):
        book.start_recording()


def test_start_recording_requires_no_prior_non_resting_activity():
    book = OrderBook("BTCUSD")
    book.submit_market_order(OrderSide.BUY, 1.0)

    with pytest.raises(ValueError, match="pristine book"):
        book.start_recording()

    book.clear()
    assert len(book.start_recording()) == 0


def test_external_partial_remaining_quantity_replays_deterministically():
    live = OrderBook("BTCUSD")
    log = live.start_recording()
    partial = Order(50, "BTCUSD", OrderSide.BUY, OrderType.LIMIT, 100.0, 5.0, 10, -1)
    partial.remaining_quantity = 2.0

    assert live.submit_order(partial).accepted is True
    live.stop_recording()
    rebuilt = replay(EventLog.from_json(log.to_json()))

    assert log.events[0].remaining_quantity == pytest.approx(2.0)
    assert rebuilt.get_order_book_depth()["bids"] == [(100.0, 2.0, 1)]
