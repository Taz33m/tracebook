"""Schema/shape tests for the public JSON artifacts.

These lock the structure of the benchmark, simulation, and replay outputs so a
refactor cannot silently change or break the contract that downstream consumers
(and the CLIs) depend on. Each artifact is round-tripped through ``json`` the
way the CLIs write it, so the asserted shape is the shape a consumer reads back.
"""

import json
import numbers
from pathlib import Path

from tracebook import (
    EventLog,
    MarketEvent,
    OrderBook,
    OrderSide,
    SelfTradePolicy,
    replay_market_events,
)
from tracebook.benchmarks import runner
from tracebook.corpus import benchmark_coinbase_corpus, compare_corpus_benchmarks
from tracebook.conformance import (
    ReferenceEngineAdapter,
    load_conformance_suite,
    run_campaign,
    run_conformance,
    run_conformance_suite,
)
from tracebook.events import CoinbaseL3Adapter
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine

ROOT = Path(__file__).parents[1]
COINBASE_CORPUS = ROOT / "src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1"
CONFORMANCE_SUITE = ROOT / "src/tracebook/conformance/fixtures/v1"


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

    _require_keys(
        report,
        ["schema_version", "measurement_model", "metadata", "generated_at", "scenarios"],
        "report",
    )
    assert report["schema_version"] == 1
    assert report["measurement_model"] == "paced_workload"
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
            "measurement_model",
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
            "achieved_input_rate_orders_per_sec",
            "achieved_event_rate_per_sec",
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
            "schema_version",
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
            "schema_version",
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
            "measurement_model",
            "target_throughput",
            "actual_duration",
            "matching_algorithm",
            "symbols",
            "seed",
            "cancel_ratio",
            "replace_ratio",
            "warmup_seconds",
            "enable_profiling",
            "enable_magic_trace",
            "batch_processing",
            "batch_size",
        ],
        "simulation_config",
    )
    _require_keys(
        results["summary_metrics"],
        [
            "total_orders_processed",
            "total_new_orders_processed",
            "total_events_processed",
            "total_cancel_events",
            "total_replace_events",
            "total_trades_executed",
            "total_volume",
            "actual_throughput",
            "achieved_new_order_rate",
            "achieved_event_rate",
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
    reduced = book.submit_limit_order(OrderSide.BUY, 99.0, 1.0)
    book.reduce_order(reduced.order.order_id, 0.25)
    book.cancel_order(resting.order.order_id)
    log = book.stop_recording()

    payload = _json_roundtrip(log.to_dict())
    _require_keys(
        payload,
        [
            "schema_version",
            "symbol",
            "matching_algorithm",
            "tick_size",
            "self_trade_policy",
            "events",
        ],
        "event_log",
    )
    assert payload["schema_version"] == 2
    assert payload["self_trade_policy"] == int(SelfTradePolicy.CANCEL_RESTING)
    assert payload["tick_size"] == 0.01

    submit_events = [e for e in payload["events"] if e["op"] == "submit"]
    reduce_events = [e for e in payload["events"] if e["op"] == "reduce"]
    cancel_events = [e for e in payload["events"] if e["op"] == "cancel"]
    assert submit_events and reduce_events and cancel_events
    _require_keys(
        submit_events[0],
        [
            "op",
            "order_id",
            "side",
            "order_type",
            "price",
            "quantity",
            "remaining_quantity",
            "symbol",
            "timestamp",
            "owner",
        ],
        "submit_event",
    )
    _require_keys(reduce_events[0], ["op", "order_id", "quantity"], "reduce_event")
    _require_keys(cancel_events[0], ["op", "order_id"], "cancel_event")

    # The round-tripped payload reconstructs an equivalent log.
    restored = EventLog.from_dict(payload)
    assert len(restored) == len(log)
    assert restored.self_trade_policy == int(SelfTradePolicy.CANCEL_RESTING)


def test_market_replay_summary_schema():
    events = [
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 10,
                "side": "SELL",
                "price": 100,
                "quantity": 1,
            }
        ),
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 20,
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 1,
            }
        ),
    ]
    payload = _json_roundtrip(replay_market_events(events).to_dict(include_trades=True))

    _require_keys(
        payload,
        [
            "schema_version",
            "replay_config",
            "input_events",
            "applied_events",
            "rejected_events",
            "submissions",
            "cancellations",
            "reductions",
            "replacements",
            "clears",
            "trades_executed",
            "trades_included",
            "trades",
            "rejections",
            "active_orders",
            "books",
        ],
        "market_replay",
    )
    _require_keys(
        payload["replay_config"],
        ["matching_algorithm", "tick_size", "self_trade_policy"],
        "market_replay.replay_config",
    )
    _require_keys(
        payload["trades"][0],
        [
            "event_index",
            "symbol",
            "buy_order_id",
            "sell_order_id",
            "engine_buy_order_id",
            "engine_sell_order_id",
            "price",
            "quantity",
            "timestamp_ns",
        ],
        "market_replay.trades[0]",
    )


def test_coinbase_adapter_summary_schema():
    maker_id = "11111111-1111-4111-8111-111111111111"
    snapshot = {
        "product_id": "BTC-USD",
        "sequence": 10,
        "bids": [],
        "asks": [["101", "1", maker_id]],
    }
    adapter = CoinbaseL3Adapter(snapshot, retain_id_map=True, retain_trades=True)
    list(
        adapter.iter_events(
            [
                {
                    "type": "match",
                    "product_id": "BTC-USD",
                    "sequence": 11,
                    "trade_id": 1,
                    "maker_order_id": maker_id,
                    "taker_order_id": "22222222-2222-4222-8222-222222222222",
                    "side": "sell",
                    "price": "101",
                    "size": "0.5",
                    "time": "2026-01-01T00:00:00Z",
                }
            ]
        )
    )
    payload = _json_roundtrip(adapter.to_dict(include_trades=True, include_id_map=True))

    _require_keys(
        payload,
        [
            "schema_version",
            "venue",
            "channel",
            "product_id",
            "snapshot_sequence",
            "final_sequence",
            "sequence_complete",
            "normalization_complete",
            "snapshot_orders",
            "messages_seen",
            "messages_sequenced",
            "messages_ignored",
            "normalized_events",
            "active_orders",
            "issues",
            "exchange_trades_observed",
            "exchange_trades_included",
            "exchange_trades",
            "id_map_included",
            "id_map",
        ],
        "coinbase_adapter",
    )
    assert payload["schema_version"] == 1
    _require_keys(
        payload["exchange_trades"][0],
        [
            "sequence",
            "product_id",
            "maker_order_id",
            "taker_order_id",
            "normalized_maker_order_id",
            "normalized_taker_order_id",
            "price",
            "quantity",
            "timestamp_ns",
            "trade_id",
            "maker_side",
        ],
        "coinbase_adapter.exchange_trade",
    )
    _require_keys(
        payload["id_map"][0],
        ["normalized_order_id", "venue_order_id"],
        "coinbase_adapter.id_map",
    )


def test_coinbase_corpus_and_benchmark_schemas():
    manifest = json.loads((COINBASE_CORPUS / "manifest.json").read_text(encoding="utf-8"))
    golden = json.loads((COINBASE_CORPUS / "golden.json").read_text(encoding="utf-8"))
    benchmark = benchmark_coinbase_corpus(COINBASE_CORPUS, iterations=1, warmups=0)
    comparison = compare_corpus_benchmarks(benchmark, benchmark)

    _require_keys(
        manifest,
        [
            "schema_version",
            "corpus_id",
            "created_at",
            "tool",
            "source",
            "rights",
            "sanitization",
            "capture",
            "replay",
            "files",
        ],
        "coinbase_corpus.manifest",
    )
    _require_keys(
        golden,
        ["schema_version", "venue", "product_id", "normalization", "events", "replay"],
        "coinbase_corpus.golden",
    )
    _require_keys(
        benchmark,
        [
            "schema_version",
            "measurement_model",
            "generated_at",
            "corpus",
            "config",
            "environment",
            "phases",
        ],
        "coinbase_corpus.benchmark",
    )
    assert set(benchmark["phases"]) == {"stream_import_replay", "replay_only"}
    for phase, summary in benchmark["phases"].items():
        _require_keys(
            summary,
            [
                "event_count",
                "samples_ns",
                "min_ns",
                "median_ns",
                "p95_ns",
                "max_ns",
                "events_per_second_median",
            ],
            f"coinbase_corpus.benchmark.{phase}",
        )
    _require_keys(
        comparison,
        [
            "schema_version",
            "corpus_id",
            "manifest_match",
            "baseline",
            "candidate",
            "environment_match",
            "environment_differences",
            "software_differences",
            "phases",
        ],
        "coinbase_corpus.comparison",
    )


def test_conformance_report_and_suite_schemas():
    suite = load_conformance_suite(CONFORMANCE_SUITE)
    events = suite.cases[0].load_events()
    report = _json_roundtrip(
        run_conformance(events, ReferenceEngineAdapter, trace_name="schema").to_dict()
    )
    suite_report = _json_roundtrip(run_conformance_suite(suite, ReferenceEngineAdapter))

    _require_keys(
        report,
        [
            "schema_version",
            "artifact_type",
            "protocol_version",
            "trace",
            "config",
            "reference_engine",
            "candidate_engine",
            "compared_events",
            "conformant",
            "final_state_hash",
            "divergence",
        ],
        "conformance.report",
    )
    _require_keys(report["trace"], ["name", "sha256", "event_count"], "conformance.trace")
    _require_keys(
        report["config"],
        [
            "matching_algorithm",
            "tick_size",
            "self_trade_policy",
            "quantity_decimal_places",
        ],
        "conformance.config",
    )
    _require_keys(
        suite_report,
        [
            "schema_version",
            "artifact_type",
            "suite_id",
            "suite_hash",
            "candidate_engine",
            "case_count",
            "conformant_cases",
            "conformant",
            "cases",
        ],
        "conformance.suite",
    )
    _require_keys(
        suite_report["cases"][0],
        ["name", "tags", "events_sha256", "report"],
        "conformance.suite.case",
    )


def test_conformance_campaign_schema():
    campaign = _json_roundtrip(
        run_campaign(
            ReferenceEngineAdapter,
            profile="fifo-limit-v1",
            seed=23,
            traces=2,
            events_per_trace=12,
            max_minimize_runs=10,
        ).to_dict()
    )

    _require_keys(
        campaign,
        [
            "schema_version",
            "artifact_type",
            "generator_version",
            "campaign_id",
            "profile",
            "seed",
            "requested_traces",
            "completed_traces",
            "events_per_trace",
            "generated_events",
            "max_minimize_runs",
            "candidate_runs",
            "candidate_engine",
            "stopped_at_first_divergence",
            "conformant",
            "traces",
            "failure",
        ],
        "conformance.campaign",
    )
    assert campaign["artifact_type"] == "tracebook.conformance.campaign"
    assert campaign["generator_version"] == 1
    assert campaign["completed_traces"] == 2
    assert campaign["generated_events"] == 24
    assert campaign["conformant"] is True
    assert campaign["failure"] is None
    _require_keys(
        campaign["profile"],
        ["name", "description", "config", "order_types", "symbols"],
        "conformance.campaign.profile",
    )
    _require_keys(
        campaign["traces"][0],
        [
            "index",
            "seed",
            "event_count",
            "trace_sha256",
            "compared_events",
            "conformant",
            "divergence",
        ],
        "conformance.campaign.trace",
    )
