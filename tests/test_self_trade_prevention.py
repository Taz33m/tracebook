"""Self-trade prevention tests across both policies and algorithms."""

import pytest

from tracebook import EventLog, OrderBook, OrderSide, SelfTradePolicy, replay


def test_disabled_by_default_allows_self_trade():
    book = OrderBook("BTCUSD")  # policy NONE
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)
    trades = book.add_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)

    assert len(trades) == 1  # same owner trades freely when disabled
    assert book.get_statistics()["self_trades_prevented"] == 0


def test_anonymous_orders_are_never_prevented():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0)  # owner NO_OWNER
    trades = book.add_limit_order(OrderSide.SELL, 100.0, 1.0)  # owner NO_OWNER

    assert len(trades) == 1
    assert book.get_statistics()["self_trades_prevented"] == 0


def test_cancel_resting_removes_own_order_and_matches_others():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    own = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)
    other = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=2)

    # Owner 1 sells into a level holding [own bid, other bid].
    trades = book.add_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)

    # It does not trade with itself; it trades with owner 2 instead.
    assert len(trades) == 1
    assert trades[0].buy_order_id == other.order.order_id
    # Own resting bid was cancelled, not matched.
    assert book.get_order(own.order.order_id) is None
    assert book.get_statistics()["self_trades_prevented"] == 1


def test_cancel_incoming_drops_aggressor_remainder_and_never_rests():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_INCOMING)
    own = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)

    # Owner 1's sell would hit its own bid -> incoming cancelled, no trade.
    trades = book.add_limit_order(OrderSide.SELL, 100.0, 2.0, owner=1)

    assert trades == []
    # The resting own bid is left untouched; the aggressor did not rest.
    assert book.get_order(own.order.order_id) is not None
    assert book.get_active_order_ids(OrderSide.SELL) == []
    assert book.get_statistics()["self_trades_prevented"] == 1


def test_book_never_crosses_after_prevention():
    for policy in (SelfTradePolicy.CANCEL_RESTING, SelfTradePolicy.CANCEL_INCOMING):
        book = OrderBook("BTCUSD", self_trade_policy=policy)
        book.add_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)
        book.add_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)

        bid = book.get_best_bid()
        ask = book.get_best_ask()
        if bid is not None and ask is not None:
            assert bid < ask


def test_cancel_resting_continues_to_next_price_level():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    # Best ask is the aggressor's own order; a deeper ask belongs to another owner.
    own = book.submit_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)
    other = book.submit_limit_order(OrderSide.SELL, 101.0, 1.0, owner=2)

    # Owner 1 buys through 101: own 100 ask is cancelled, fills against 101.
    trades = book.add_limit_order(OrderSide.BUY, 101.0, 1.0, owner=1)

    assert len(trades) == 1
    assert trades[0].sell_order_id == other.order.order_id
    assert book.get_order(own.order.order_id) is None


def test_fok_not_reported_fillable_by_own_liquidity():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    # Only liquidity at the price is the owner's own order.
    own = book.submit_limit_order(OrderSide.SELL, 100.0, 5.0, owner=1)

    # FOK buy for owner 1 cannot be filled by its own resting sell -> killed.
    trades = book.add_fok_order(OrderSide.BUY, 100.0, 5.0, owner=1)

    assert trades == []
    # The FOK left the book untouched; the own order still rests.
    assert book.get_order(own.order.order_id) is not None


def test_pro_rata_excludes_own_order_from_allocation():
    book = OrderBook(
        "BTCUSD",
        matching_algorithm="pro_rata",
        self_trade_policy=SelfTradePolicy.CANCEL_RESTING,
    )
    own = book.submit_limit_order(OrderSide.BUY, 100.0, 2.0, owner=1)
    other = book.submit_limit_order(OrderSide.BUY, 100.0, 2.0, owner=2)

    trades = book.add_limit_order(OrderSide.SELL, 100.0, 2.0, owner=1)

    # All 2.0 goes to owner 2; none to the owner's own bid.
    assert sum(t.quantity for t in trades) == pytest.approx(2.0)
    assert all(t.buy_order_id == other.order.order_id for t in trades)
    assert book.get_order(own.order.order_id) is None


def test_owner_survives_record_replay():
    live = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING)
    live.start_recording()
    live.add_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)
    live.add_limit_order(OrderSide.BUY, 100.0, 1.0, owner=2)
    live.add_limit_order(OrderSide.SELL, 100.0, 1.0, owner=1)  # prevented vs own
    log = live.stop_recording()

    # Owner ids must round-trip so replay reproduces the same prevention.
    restored = EventLog.from_json(log.to_json())
    replayed = replay(restored)

    assert [t.buy_order_id for t in replayed.matching_engine.trades] == [
        t.buy_order_id for t in live.matching_engine.trades
    ]
    assert (
        replayed.get_statistics()["self_trades_prevented"]
        == live.get_statistics()["self_trades_prevented"]
        == 1
    )


def test_invalid_owner_is_rejected():
    book = OrderBook("BTCUSD")
    result = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0, owner="me")
    assert result.rejected_reason
    assert "owner" in result.rejected_reason
