"""Tests for integer-tick price keying.

Price levels are keyed by integer ticks, so prices that round to the same tick
must share one level and resting orders must sit on the canonical grid price.
This is the correctness fix for float-identity dict keys.
"""

import pytest

from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook
from tracebook.core.price_level import PriceLevelManager, infer_price_decimals


def test_prices_rounding_to_the_same_tick_share_one_level():
    book = OrderBook("BTCUSD", tick_size=0.01)

    # Two prices that are equal on the 0.01 grid but differ in float bits.
    book.add_limit_order(OrderSide.BUY, 100.00, 1.0)
    book.add_limit_order(OrderSide.BUY, 100.00 + 1e-12, 2.0)

    buy_side = book.matching_engine.buy_side
    assert len(buy_side.sorted_ticks) == 1
    assert len(buy_side.price_levels) == 1
    depth = book.get_order_book_depth(10)
    # One aggregated level carrying both orders' quantity.
    assert len(depth["bids"]) == 1
    price, qty, count = depth["bids"][0]
    assert price == pytest.approx(100.00)
    assert qty == pytest.approx(3.0)
    assert count == 2


def test_resting_order_price_is_snapped_onto_the_grid():
    book = OrderBook("BTCUSD", tick_size=0.01)

    result = book.submit_limit_order(OrderSide.BUY, 100.017, 1.0)

    # 100.017 rounds to the 100.02 tick and the resting order carries that price.
    assert book.get_order(result.order.order_id).price == pytest.approx(100.02)
    assert book.get_best_bid() == pytest.approx(100.02)


def test_coarser_tick_size_merges_nearby_prices():
    book = OrderBook("XYZ", tick_size=0.5)

    book.add_limit_order(OrderSide.SELL, 100.1, 1.0)  # -> 100.0 tick
    book.add_limit_order(OrderSide.SELL, 100.4, 1.0)  # -> 100.5 tick
    book.add_limit_order(OrderSide.SELL, 100.2, 1.0)  # -> 100.0 tick

    depth = book.get_order_book_depth(10)
    prices = [lvl[0] for lvl in depth["asks"]]
    assert prices == pytest.approx([100.0, 100.5])
    # The two orders on the 100.0 tick aggregate.
    assert depth["asks"][0][1] == pytest.approx(2.0)


def test_matching_still_works_across_the_grid():
    book = OrderBook("BTCUSD", tick_size=0.01)
    book.add_limit_order(OrderSide.SELL, 100.00, 1.0)

    # Buy priced just under the next tick still crosses the 100.00 ask.
    trades = book.add_limit_order(OrderSide.BUY, 100.004, 1.0)

    assert len(trades) == 1
    assert trades[0].price == pytest.approx(100.00)
    assert book.get_best_ask() is None


def test_invalid_tick_size_is_rejected():
    for bad in (0.0, -0.01, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            OrderBook("BTCUSD", tick_size=bad)
    with pytest.raises(ValueError):
        PriceLevelManager(is_buy_side=True, tick_size=0.0)


def test_infer_price_decimals():
    assert infer_price_decimals(0.01) == 2
    assert infer_price_decimals(0.5) == 1
    assert infer_price_decimals(1.0) == 0
    assert infer_price_decimals(0.001) == 3


def test_infer_price_decimals_handles_fine_and_scientific_ticks():
    # Fine grids must not be truncated to whole numbers.
    assert infer_price_decimals(1e-6) == 6
    assert infer_price_decimals(1e-9) == 9
    assert infer_price_decimals(0.00025) == 5


def test_fine_grid_preserves_distinct_sub_cent_levels():
    book = OrderBook("XYZ", tick_size=1e-6)

    book.add_limit_order(OrderSide.SELL, 100.000001, 1.0)
    book.add_limit_order(OrderSide.SELL, 100.000002, 1.0)

    depth = book.get_order_book_depth(10)
    prices = [lvl[0] for lvl in depth["asks"]]
    # Two distinct levels survive instead of collapsing onto 100.0.
    assert prices == pytest.approx([100.000001, 100.000002])
