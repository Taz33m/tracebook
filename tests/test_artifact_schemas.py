"""Schema/shape tests for the public JSON artifacts.

These lock the structure of the benchmark, simulation, and replay outputs so a
refactor cannot silently change or break the contract that downstream consumers
(and the CLIs) depend on. Each artifact is round-tripped through ``json`` the
way the CLIs write it, so the asserted shape is the shape a consumer reads back.
"""

import json
import numbers

from tracebook import EventLog, OrderBook, OrderSide, SelfTradePolicy
from tracebook.benchmarks import runner
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine


def _require_keys(obj, keys, ctx):
    assert isinstance(obj, dict), f"{ctx}: expected dict, got {type(obj).__name__}"
    missing = set(keys) - set(obj)
    assert not missing, f"{ctx}: missing keys {sorted(missing)}"


def _json_roundtrip(obj):
    """Serialize and reload the way the CLIs write artifacts (``default=str``)."""
    return json.loads(json.dumps(obj, default=str))


def test_benchmark_report_schema():
    report = runner.run_benchmarks(
        ["smoke"], seed=7, warmup_seconds=0.0, duration_override=0.2, throughput_override=20.0
    )
    report = _json_roundtrip(report)

    _require_keys(report, ["metadata", "generated_at", "scenarios"], "report")
    assert isinstance(report["generated_at"], numbers.Integral)
    _require_keys(
        report["metadata"],
        ["python", "platform", "processor", "machine", "dependency_versions"],
        "metadata",
    )
    _require_keys(
        report["metadata"]["dependency_versions"],
        ["tracebook", "numpy", "psutil", "dash", "plotly"],
        "dependency_versions",
    )

    assert len(report["scenarios"]) == 1
    scenario = report["scenarios"][0]
    _require_keys(
        scenario,
        ["name", "started_at", "completed_at", "config", "summary", "raw_result"],
        "scenario",
    )
    assert scenario["name"] == "smoke"
    _require_keys(
        scenario["config"],
        [
            "name",
            "duration_seconds",
            "target_throughput",
            "matching_algorithm",
            "cancel_ratio",
            "replace_ratio",
            "seed",
            "warmup_seconds",
        ],
        "config",
    )
    assert scenario["config"]["seed"] == 7  # seed + index(0)

    summary = scenario["summary"]
    _require_keys(
        summary,
        [
            "orders_processed",
            "events_processed",
            "cancel_events",
            "replace_events",
            "trades_executed",
            "throughput_orders_per_sec",
            "latency_ms",
            "event_latency_ms",
            "generation_latency_ms",
            "memory_mb",
            "monitoring_overhead_ns",
        ],
        "summary",
    )
    _require_keys(
        summary["latency_ms"], ["count", "mean", "p50", "p95", "p99", "max"], "latency_ms"
    )
    _require_keys(summary["event_latency_ms"], ["count", "mean", "p95", "p99"], "event_latency_ms")
    _require_keys(
        summary["generation_latency_ms"], ["count", "mean", "p95", "p99"], "generation_latency_ms"
    )

    # raw_result carries the full simulation results (checked in detail below).
    _require_keys(
        scenario["raw_result"],
        [
            "simulation_config",
            "summary_metrics",
            "performance_data",
            "order_book_statistics",
            "stream_statistics",
            "algorithm_analysis",
            "timestamp",
        ],
        "raw_result",
    )


def test_simulation_results_schema():
    config = SimulationConfig(
        duration_seconds=0.2,
        target_throughput=20.0,
        enable_magic_trace=False,
        seed=3,
        warmup_seconds=0.0,
    )
    results = _json_roundtrip(SimulationEngine(config).run_simulation())

    _require_keys(
        results,
        [
            "simulation_config",
            "summary_metrics",
            "performance_data",
            "order_book_statistics",
            "stream_statistics",
            "algorithm_analysis",
            "timestamp",
        ],
        "results",
    )
    _require_keys(
        results["simulation_config"],
        [
            "duration_seconds",
            "target_throughput",
            "actual_duration",
            "matching_algorithm",
            "symbols",
            "seed",
            "cancel_ratio",
            "replace_ratio",
            "warmup_seconds",
        ],
        "simulation_config",
    )
    _require_keys(
        results["summary_metrics"],
        [
            "total_orders_processed",
            "total_events_processed",
            "total_cancel_events",
            "total_replace_events",
            "total_trades_executed",
            "total_volume",
            "actual_throughput",
            "trade_ratio",
            "trades_per_order",
            "average_trade_size",
        ],
        "summary_metrics",
    )

    assert results["order_book_statistics"], "expected at least one symbol"
    for symbol, stats in results["order_book_statistics"].items():
        _require_keys(
            stats,
            [
                "matching_algorithm",
                "self_trade_policy",
                "total_orders_processed",
                "total_trades",
                "self_trades_prevented",
            ],
            f"order_book_statistics[{symbol}]",
        )

    for symbol, analysis in results["algorithm_analysis"].items():
        _require_keys(
            analysis,
            ["algorithm", "status", "reason", "matches_observed"],
            f"algorithm_analysis[{symbol}]",
        )
        assert analysis["status"] == "not_collected"  # honest documented stub


def test_event_log_schema_and_roundtrip():
    book = OrderBook("BTCUSD", self_trade_policy=SelfTradePolicy.CANCEL_RESTING, tick_size=0.01)
    book.start_recording()
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0, owner=1)
    book.add_limit_order(OrderSide.SELL, 100.0, 0.5, owner=2)
    resting = book.submit_limit_order(OrderSide.SELL, 101.0, 1.0)
    book.cancel_order(resting.order.order_id)
    log = book.stop_recording()

    payload = _json_roundtrip(log.to_dict())
    _require_keys(
        payload,
        ["symbol", "matching_algorithm", "tick_size", "self_trade_policy", "events"],
        "event_log",
    )
    assert payload["self_trade_policy"] == int(SelfTradePolicy.CANCEL_RESTING)
    assert payload["tick_size"] == 0.01

    submit_events = [e for e in payload["events"] if e["op"] == "submit"]
    cancel_events = [e for e in payload["events"] if e["op"] == "cancel"]
    assert submit_events and cancel_events
    _require_keys(
        submit_events[0],
        [
            "op",
            "order_id",
            "side",
            "order_type",
            "price",
            "quantity",
            "symbol",
            "timestamp",
            "owner",
        ],
        "submit_event",
    )
    _require_keys(cancel_events[0], ["op", "order_id"], "cancel_event")

    # The round-tripped payload reconstructs an equivalent log.
    restored = EventLog.from_dict(payload)
    assert len(restored) == len(log)
    assert restored.self_trade_policy == int(SelfTradePolicy.CANCEL_RESTING)
