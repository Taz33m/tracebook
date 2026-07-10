from pathlib import Path

import pytest

from tracebook.benchmarks.runner import run_benchmarks, write_report
from tracebook.simulation.order_generator import (
    MarketParameters,
    OrderGenerationConfig,
    OrderPattern,
    SimulationEvent,
    SimulationEventType,
    SyntheticOrderStream,
)
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine


def test_order_stream_exposes_new_order_events():
    stream = SyntheticOrderStream(
        MarketParameters(symbol="BTCUSD"),
        OrderGenerationConfig(pattern=OrderPattern.RANDOM, batch_size=3, seed=7),
    )

    orders = stream.generators[0].generate_orders(3)
    with stream.queue_lock:
        stream.orders_queue.extend(orders)
        stream.events_queue.extend(
            SimulationEvent(SimulationEventType.NEW, "BTCUSD", order=order) for order in orders
        )

    events = stream.get_events()

    assert len(events) == 3
    assert stream.get_orders() == []
    assert all(event.event_type == SimulationEventType.NEW for event in events)


def test_order_generation_config_rejects_ambiguous_numbers_and_seed():
    with pytest.raises(ValueError, match="duration_seconds"):
        OrderGenerationConfig(duration_seconds="1")
    with pytest.raises(ValueError, match="seed"):
        OrderGenerationConfig(seed=-1)


def _order_signature(orders):
    return [
        (
            int(order.side),
            int(order.order_type),
            round(order.price, 8),
            round(order.quantity, 8),
        )
        for order in orders
    ]


def test_seeded_order_stream_generates_repeatable_order_batches():
    config = OrderGenerationConfig(pattern=OrderPattern.RANDOM, batch_size=5, seed=123)
    stream_a = SyntheticOrderStream(MarketParameters(symbol="BTCUSD"), config)
    stream_b = SyntheticOrderStream(MarketParameters(symbol="BTCUSD"), config)

    orders_a = stream_a.generators[0].generate_orders(5)
    orders_b = stream_b.generators[0].generate_orders(5)

    assert _order_signature(orders_a) == _order_signature(orders_b)


def test_market_making_pattern_emits_requested_batch_count_with_unique_ids():
    stream = SyntheticOrderStream(
        MarketParameters(symbol="BTCUSD"),
        OrderGenerationConfig(pattern=OrderPattern.MARKET_MAKING, batch_size=20, seed=7),
    )

    orders = stream.generators[0].generate_orders(20)

    assert len(orders) == 20
    assert len({order.order_id for order in orders}) == 20


def test_simulation_processes_cancel_and_replace_events_with_seed():
    config = SimulationConfig(
        duration_seconds=0.25,
        target_throughput=40.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        seed=11,
        cancel_ratio=0.25,
        replace_ratio=0.25,
        warmup_seconds=0.0,
    )
    engine = SimulationEngine(config)

    results = engine.run_simulation()
    summary = results["summary_metrics"]

    assert summary["total_orders_processed"] > 0
    assert summary["total_new_orders_processed"] > 0
    assert summary["total_events_processed"] >= summary["total_orders_processed"]
    assert summary["actual_throughput"] == summary["achieved_new_order_rate"]
    assert summary["achieved_event_rate"] >= summary["achieved_new_order_rate"]
    assert "order_generation_latency_ms" in results["performance_data"]["performance_metrics"]


def test_simulation_can_disable_metrics_collection():
    engine = SimulationEngine(
        SimulationConfig(
            duration_seconds=0.1,
            target_throughput=20.0,
            enable_profiling=False,
            enable_magic_trace=False,
            seed=31,
        )
    )

    results = engine.run_simulation()

    assert results["summary_metrics"]["total_new_orders_processed"] > 0
    assert results["performance_data"]["performance_metrics"] == {}


def test_simulation_algorithm_analysis_is_explicitly_not_collected():
    config = SimulationConfig(
        duration_seconds=0.05,
        target_throughput=20.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        seed=13,
        warmup_seconds=0.0,
    )
    engine = SimulationEngine(config)

    results = engine.run_simulation()
    analysis = results["algorithm_analysis"]["BTCUSD"]

    assert analysis["algorithm"] == "FIFO"
    assert analysis["status"] == "not_collected"
    assert "recommendations" not in analysis
    assert analysis["matches_observed"] >= 0


def test_simulation_and_market_parameters_normalize_symbols():
    config = SimulationConfig(
        duration_seconds=0.0,
        target_throughput=1.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        symbols=[" BTCUSD "],
    )
    engine = SimulationEngine(config)
    market_params = MarketParameters(symbol=" ETHUSD ")

    assert config.symbols == ["BTCUSD"]
    assert engine.order_book_manager.get_all_symbols() == ["BTCUSD"]
    assert list(engine.order_streams.keys()) == ["BTCUSD"]
    assert market_params.symbol == "ETHUSD"

    with pytest.raises(ValueError, match="non-empty string"):
        SimulationConfig(symbols=["   "])

    with pytest.raises(ValueError, match="duplicates"):
        SimulationConfig(symbols=["BTCUSD", " BTCUSD "])

    with pytest.raises(ValueError, match="non-empty string"):
        MarketParameters(symbol="")


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tick_size": 0}, "tick_size"),
        ({"min_quantity": 2, "max_quantity": 1}, "max_quantity"),
        ({"market_order_ratio": 1.5}, "market_order_ratio"),
        ({"trend_strength": -2}, "trend_strength"),
        ({"price_volatility": float("nan")}, "price_volatility"),
    ],
)
def test_market_parameters_reject_invalid_numeric_domains(kwargs, message):
    with pytest.raises(ValueError, match=message):
        MarketParameters(**kwargs)


def test_simulation_reports_missing_configured_order_book_clearly():
    config = SimulationConfig(
        duration_seconds=0.0,
        target_throughput=1.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        symbols=["BTCUSD"],
    )
    engine = SimulationEngine(config)
    engine.order_book_manager.remove_order_book("BTCUSD")

    try:
        engine.run_simulation()
    except RuntimeError as exc:
        assert "No order book configured for symbol 'BTCUSD'" in str(exc)
    else:
        raise AssertionError("Expected missing order book to raise RuntimeError")

    assert engine.is_running is False


def test_simulation_engine_is_explicitly_single_use():
    engine = SimulationEngine(
        SimulationConfig(
            duration_seconds=0.0,
            target_throughput=1.0,
            enable_magic_trace=False,
        )
    )
    completed = []
    engine.register_simulation_callback(completed.append)
    results = engine.run_simulation()

    assert completed == [results]

    with pytest.raises(RuntimeError, match="single-use"):
        engine.run_simulation()


def test_simulation_adapts_batch_size_for_low_throughput_short_runs():
    config = SimulationConfig(
        duration_seconds=0.1,
        target_throughput=10.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        seed=5,
        batch_size=100,
        warmup_seconds=0.0,
    )
    engine = SimulationEngine(config)

    stream = engine.order_streams["BTCUSD"]
    results = engine.run_simulation()

    assert stream.config.batch_size == 1
    assert results["summary_metrics"]["total_orders_processed"] <= 5


def test_benchmark_smoke_report_has_latency_schema(tmp_path: Path):
    report = run_benchmarks(
        ["smoke"],
        seed=17,
        warmup_seconds=0.0,
        duration_override=0.25,
        throughput_override=40.0,
    )

    scenario = report["scenarios"][0]
    latency = scenario["summary"]["latency_ms"]

    assert scenario["name"] == "smoke"
    assert latency["count"] > 0
    assert set(["mean", "p50", "p95", "p99"]).issubset(latency)

    output_path = tmp_path / "benchmark.json"
    assert write_report(report, str(output_path)) == str(output_path)
    assert output_path.exists()


def test_cancellation_mix_benchmark_handles_replace_ids_without_collisions():
    report = run_benchmarks(
        ["cancellation_mix"],
        seed=23,
        warmup_seconds=0.0,
        duration_override=0.25,
        throughput_override=40.0,
    )

    scenario = report["scenarios"][0]

    assert scenario["name"] == "cancellation_mix"
    assert scenario["summary"]["orders_processed"] > 0
    assert scenario["summary"]["events_processed"] >= scenario["summary"]["orders_processed"]
