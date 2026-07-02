"""Matching-semantics invariants that must hold for any order sequence.

These lock in observable behavior so later refactors (pricing consolidation,
integer ticks, API changes) cannot silently alter matching semantics. The
property-based cases drive random order flow through a fresh book and assert
book-wide invariants after *every* operation; the example cases pin the
specific semantics (FIFO time priority, FOK all-or-nothing, IOC non-resting).
"""

from collections import defaultdict

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from tracebook.core.order import OrderSide, OrderType
from tracebook.core.orderbook import OrderBook

EPS = 1e-9

# A small discrete price grid so random orders actually cross and trade.
_PRICES = st.sampled_from([98.0, 99.0, 100.0, 101.0, 102.0])
_QUANTITIES = st.floats(min_value=0.1, max_value=10.0, allow_nan=False, allow_infinity=False)
_SIDES = st.sampled_from([OrderSide.BUY, OrderSide.SELL])
# Weight LIMIT higher so the book actually builds up resting liquidity.
_TYPES = st.sampled_from(
    [
        OrderType.LIMIT,
        OrderType.LIMIT,
        OrderType.LIMIT,
        OrderType.MARKET,
        OrderType.IOC,
        OrderType.FOK,
    ]
)


@st.composite
def _order_actions(draw):
    otype = draw(_TYPES)
    side = draw(_SIDES)
    qty = draw(_QUANTITIES)
    price = None if otype == OrderType.MARKET else draw(_PRICES)
    return (otype, side, price, qty)


def _submit(book, action):
    """Submit an action through the structured API and return the OrderResult."""
    otype, side, price, qty = action
    if otype == OrderType.LIMIT:
        return book.submit_limit_order(side, price, qty)
    if otype == OrderType.MARKET:
        return book.submit_market_order(side, qty)
    if otype == OrderType.IOC:
        return book.submit_ioc_order(side, price, qty)
    if otype == OrderType.FOK:
        return book.submit_fok_order(side, price, qty)
    raise AssertionError(f"unhandled order type {otype!r}")


def _assert_side_consistent(manager):
    """Internal-consistency invariants for one side's price-level manager."""
    # sorted_ticks mirrors the price-level keys (both integer ticks), no dups.
    assert set(manager.sorted_ticks) == set(manager.price_levels.keys())
    assert len(manager.sorted_ticks) == len(set(manager.sorted_ticks))

    # Best tick is at index 0: buy side descending, sell side ascending.
    if manager.is_buy_side:
        assert manager.sorted_ticks == sorted(manager.sorted_ticks, reverse=True)
    else:
        assert manager.sorted_ticks == sorted(manager.sorted_ticks)

    for tick, level in manager.price_levels.items():
        assert not level.is_empty()
        assert level.order_count == len(level.orders)
        # The level's canonical price agrees with its tick key.
        assert manager.price_to_tick(level.price) == tick
        level_sum = 0.0
        for order_id in level.orders:
            order = manager.orders[order_id]
            # Fully filled orders are evicted, never left resting.
            assert order.remaining_quantity > EPS
            # No order rests for more than it was submitted with.
            assert order.remaining_quantity <= order.quantity + EPS
            # Every resting order sits on the level's canonical grid price.
            assert order.price == level.price
            level_sum += order.remaining_quantity
        # Cached level total stays in step with the resting orders.
        assert level.total_quantity == pytest.approx(level_sum, abs=1e-6)


@pytest.mark.parametrize("algorithm", ["fifo", "pro_rata"])
@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(actions=st.lists(_order_actions(), min_size=1, max_size=40))
def test_book_invariants_hold_after_every_operation(algorithm, actions):
    book = OrderBook("BTCUSD", matching_algorithm=algorithm)
    original_qty = {}
    filled = defaultdict(float)

    for action in actions:
        result = _submit(book, action)
        order = result.order
        assert order is not None  # grid inputs are always valid
        original_qty[order.order_id] = order.quantity

        for trade in result.trades:
            # Conservation: every trade credits one buy id and one sell id equally.
            filled[trade.buy_order_id] += trade.quantity
            filled[trade.sell_order_id] += trade.quantity
            assert trade.quantity > 0
            assert trade.price > 0

        otype = action[0]
        active_ids = set(book.get_active_order_ids())

        # Only LIMIT orders may rest; the others never linger on the book.
        if otype != OrderType.LIMIT:
            assert order.order_id not in active_ids
        # rested flag agrees with actual book membership.
        assert result.rested == (order.order_id in active_ids)

        # Book is never crossed after an operation settles.
        bid = book.get_best_bid()
        ask = book.get_best_ask()
        if bid is not None and ask is not None:
            assert bid < ask

        _assert_side_consistent(book.matching_engine.buy_side)
        _assert_side_consistent(book.matching_engine.sell_side)

    # No order is ever filled beyond the quantity it was submitted with.
    for order_id, qty_filled in filled.items():
        assert qty_filled <= original_qty[order_id] + EPS


# --- Example-based semantic pins -------------------------------------------


def test_fifo_time_priority_fills_oldest_resting_order_first():
    book = OrderBook("X", matching_algorithm="fifo")
    first = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    second = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)

    trades = book.add_limit_order(OrderSide.SELL, 100.0, 1.0)

    assert len(trades) == 1
    assert trades[0].buy_order_id == first.order.order_id
    assert book.get_order(first.order.order_id) is None
    assert book.get_order(second.order.order_id).remaining_quantity == pytest.approx(1.0)


def test_aggressive_order_sweeps_a_deep_level_in_fifo_order():
    book = OrderBook("X", matching_algorithm="fifo")
    # Six resting bids at one price, submitted in a known order.
    ids = [book.submit_limit_order(OrderSide.BUY, 100.0, 1.0).order.order_id for _ in range(6)]

    # A sell of 3.5 sweeps the level: fills the three oldest fully and the
    # fourth partially, in FIFO order, leaving the rest untouched.
    trades = book.add_limit_order(OrderSide.SELL, 100.0, 3.5)

    assert [t.buy_order_id for t in trades] == ids[:4]
    assert [t.quantity for t in trades] == pytest.approx([1.0, 1.0, 1.0, 0.5])
    for spent in ids[:3]:
        assert book.get_order(spent) is None
    assert book.get_order(ids[3]).remaining_quantity == pytest.approx(0.5)
    assert book.get_order(ids[4]).remaining_quantity == pytest.approx(1.0)
    assert book.get_order(ids[5]).remaining_quantity == pytest.approx(1.0)


def test_matches_execute_at_the_resting_order_price():
    book = OrderBook("X", matching_algorithm="fifo")
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0)

    # Aggressive sell crosses well below the resting bid; fill is at the bid.
    trades = book.add_limit_order(OrderSide.SELL, 98.0, 1.0)

    assert len(trades) == 1
    assert trades[0].price == pytest.approx(100.0)


def test_fok_is_all_or_nothing_and_leaves_book_unchanged_when_unfillable():
    book = OrderBook("X", matching_algorithm="fifo")
    book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    before = book.get_order_book_depth(10)

    trades = book.add_fok_order(OrderSide.BUY, 100.0, 5.0)

    assert trades == []
    after = book.get_order_book_depth(10)
    assert (after["bids"], after["asks"]) == (before["bids"], before["asks"])


def test_fok_fully_fills_when_liquidity_spans_multiple_levels():
    book = OrderBook("X", matching_algorithm="fifo")
    book.add_limit_order(OrderSide.SELL, 100.0, 3.0)
    book.add_limit_order(OrderSide.SELL, 101.0, 3.0)

    trades = book.add_fok_order(OrderSide.BUY, 101.0, 5.0)

    assert sum(t.quantity for t in trades) == pytest.approx(5.0)
    assert book.get_active_order_ids(OrderSide.BUY) == []


def test_ioc_fills_what_it_can_and_never_rests():
    book = OrderBook("X", matching_algorithm="fifo")
    book.add_limit_order(OrderSide.SELL, 100.0, 1.0)

    trades = book.add_ioc_order(OrderSide.BUY, 100.0, 5.0)

    assert sum(t.quantity for t in trades) == pytest.approx(1.0)
    assert book.get_active_order_ids(OrderSide.BUY) == []
