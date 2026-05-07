import numpy as np
import pytest

from tracebook.algorithms.fifo import fifo_batch_match, fifo_match_single
from tracebook.algorithms.pro_rata import (
    ProRataAnalyzer,
    calculate_pro_rata_allocation,
    pro_rata_batch_allocation,
    pro_rata_match_single,
)
from tracebook.core.order import OrderFactory, OrderSide


def test_fifo_single_match_preserves_decimal_execution_quantity():
    price, quantity, can_match = fifo_match_single(101.0, 0.75, 1, 100.0, 0.5, 2)

    assert can_match is True
    assert price == pytest.approx(101.0)
    assert quantity == pytest.approx(0.5)


def test_fifo_batch_match_returns_decimal_quantities():
    _, quantities, flags = fifo_batch_match(
        np.array([101.0, 99.0], dtype=np.float64),
        np.array([0.75, 0.25], dtype=np.float64),
        np.array([1, 2], dtype=np.int64),
        np.array([100.0, 100.0], dtype=np.float64),
        np.array([0.5, 0.25], dtype=np.float64),
        np.array([3, 4], dtype=np.int64),
    )

    assert quantities.dtype == np.float64
    assert quantities.tolist() == pytest.approx([0.5, 0.0])
    assert flags.tolist() == [True, False]


def test_pro_rata_allocation_preserves_fractional_quantities():
    allocation = calculate_pro_rata_allocation(0.25, 1.0, 0.5)

    assert allocation == pytest.approx(0.125)


def test_pro_rata_batch_allocation_preserves_decimal_shares():
    allocations = pro_rata_batch_allocation(
        np.array([0.25, 0.75], dtype=np.float64),
        0.5,
    )

    assert allocations.dtype == np.float64
    assert allocations.tolist() == pytest.approx([0.125, 0.375])


def test_pro_rata_single_match_does_not_truncate_sub_unit_fills():
    price, quantity, can_match = pro_rata_match_single(100.0, 0.5, 99.0, 0.25, 1.0)

    assert can_match is True
    assert price == pytest.approx(99.0)
    assert quantity == pytest.approx(0.0625)


def test_pro_rata_analyzer_uses_decimal_denominators_for_fill_rates():
    order = OrderFactory().create_limit_order("BTCUSD", OrderSide.SELL, 100.0, 0.25)
    analyzer = ProRataAnalyzer()

    analyzer.record_allocation([order], [0.125], available_quantity=0.5, timestamp=1)

    record = analyzer.allocation_history[0]
    assert record["fill_rate"] == pytest.approx(0.25)
    assert record["demand_ratio"] == pytest.approx(0.5)
    assert record["order_details"][0]["fill_rate"] == pytest.approx(0.5)
