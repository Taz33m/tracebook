import pytest

from tracebook.core.order import OrderSide
from tracebook.core.orderbook import OrderBook
from tracebook.simulation.order_generator import SimulationEvent, SimulationEventType
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine


def test_cancel_and_replace_events_record_counts_and_latency_metadata():
    engine = SimulationEngine(
        SimulationConfig(
            duration_seconds=0.0,
            target_throughput=1.0,
            enable_magic_trace=False,
            seed=23,
            warmup_seconds=0.0,
        )
    )
    order_book = OrderBook("BTCUSD")
    cancel_target = order_book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    replace_target = order_book.submit_limit_order(OrderSide.SELL, 102.0, 2.0)

    engine._process_event(
        order_book,
        SimulationEvent(
            SimulationEventType.CANCEL,
            "BTCUSD",
            order_id=cancel_target.order.order_id,
        ),
    )
    engine._process_event(
        order_book,
        SimulationEvent(
            SimulationEventType.REPLACE,
            "BTCUSD",
            order_id=replace_target.order.order_id,
            price=103.0,
            quantity=1.25,
        ),
    )

    assert engine.total_events_processed == 2
    assert engine.total_cancel_events == 1
    assert engine.total_replace_events == 1
    assert order_book.get_order(cancel_target.order.order_id) is None
    assert order_book.get_order(replace_target.order.order_id) is None

    event_samples = list(
        engine.performance_monitor.metrics_collector.metrics["order_event_latency_ms"]
    )
    assert [sample.metadata["event_type"] for sample in event_samples] == ["cancel", "replace"]
    assert all(sample.value >= 0 for sample in event_samples)

    summary = engine._generate_results()["summary_metrics"]
    assert summary["total_events_processed"] == 2
    assert summary["total_cancel_events"] == 1
    assert summary["total_replace_events"] == 1


def test_simulation_config_validates_lifecycle_ratios_and_normalizes_algorithm():
    config = SimulationConfig(matching_algorithm="pro_rata", enable_magic_trace=False)

    assert config.matching_algorithm == "PRO_RATA"

    with pytest.raises(ValueError, match="cancel_ratio \\+ replace_ratio"):
        SimulationConfig(cancel_ratio=0.7, replace_ratio=0.4)

    with pytest.raises(ValueError, match="target_throughput"):
        SimulationConfig(target_throughput=0)
