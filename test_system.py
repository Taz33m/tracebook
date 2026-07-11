#!/usr/bin/env python3
"""Fast end-to-end smoke checks for a source checkout or installed wheel."""

import sys
import tempfile
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))


def check_matching() -> None:
    from tracebook import OrderBook, OrderSide

    book = OrderBook("BTCUSD")
    bid = book.submit_limit_order(OrderSide.BUY, 50_000.0, 1.0)
    sell = book.submit_limit_order(OrderSide.SELL, 49_999.0, 0.5)

    assert bid.rested is True
    assert len(sell.trades) == 1
    assert sell.trades[0].price == 50_000.0
    assert sell.trades[0].quantity == 0.5
    assert book.get_best_bid() == 50_000.0


def check_event_replay() -> None:
    from tracebook import replay_market_event_file

    result = replay_market_event_file(ROOT / "examples" / "data" / "sample_events.jsonl")
    summary = result.to_dict()

    assert summary["input_events"] == 5
    assert summary["applied_events"] == 5
    assert summary["rejected_events"] == 0
    assert set(summary["books"]) == {"BTCUSD", "ETHUSD"}


def check_coinbase_corpus() -> None:
    from tracebook.corpus import copy_bundled_coinbase_corpus, verify_coinbase_corpus

    with tempfile.TemporaryDirectory(prefix="tracebook-corpus-smoke-") as directory:
        corpus = Path(directory) / "sample"
        copy_bundled_coinbase_corpus(corpus)
        result = verify_coinbase_corpus(corpus)

        assert result["verified"] is True
        assert result["events_verified"] == 8
        assert result["final_sequence"] == 109


def check_performance_monitor() -> None:
    from tracebook.profiling.performance_monitor import PerformanceMonitor

    monitor = PerformanceMonitor(enable_magic_trace=False)
    monitor.system_monitor.sample_interval = 0.01
    monitor.start_monitoring()
    try:
        monitor.record_order_processing(1_000_000, 1)
        monitor.record_trade_execution(1, 50_000.0)
        time.sleep(0.03)
        summary = monitor.get_performance_summary()
        assert summary["schema_version"] == 1
        assert summary["session_metrics"]["orders_processed"] == 1
    finally:
        monitor.stop_monitoring()


def check_simulation() -> None:
    from tracebook.simulation.simulation_engine import run_benchmark_simulation

    with tempfile.TemporaryDirectory(prefix="tracebook-smoke-") as directory:
        output = Path(directory) / "simulation.json"
        results = run_benchmark_simulation(
            duration=0.1,
            throughput=20.0,
            algorithm="FIFO",
            enable_magic_trace=False,
            seed=1337,
            output_path=str(output),
        )

        assert results["schema_version"] == 1
        assert results["summary_metrics"]["total_orders_processed"] > 0
        assert output.is_file()


def main() -> int:
    checks = [
        ("matching and lifecycle", check_matching),
        ("normalized event replay", check_event_replay),
        ("verified Coinbase corpus", check_coinbase_corpus),
        ("performance monitoring", check_performance_monitor),
        ("paced simulation", check_simulation),
    ]

    print("tracebook system smoke")
    print("=" * 48)
    failures = 0
    for name, check in checks:
        try:
            check()
            print(f"PASS  {name}")
        except Exception:
            failures += 1
            print(f"FAIL  {name}")
            traceback.print_exc()

    print("=" * 48)
    print(f"{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
