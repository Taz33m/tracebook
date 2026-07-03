from pathlib import Path
import os
import subprocess
import threading
import time

import pytest

from tracebook.core.order import NO_OWNER, Order, OrderFactory, OrderSide, OrderType
from tracebook.core.orderbook import OrderBook
from tracebook.profiling.magic_trace_wrapper import (
    MagicTraceConfig,
    MagicTraceProfiler,
    MagicTraceSession,
)
from tracebook.profiling.performance_monitor import MetricsCollector, PerformanceMonitor
from tracebook.profiling.trace_visualizer import TraceVisualizer
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine


def test_order_factory_rejects_non_numeric_side_and_order_type():
    factory = OrderFactory()

    with pytest.raises(ValueError, match="Unsupported order side"):
        factory.create_limit_order("BTCUSD", "BUY", 100.0, 1.0)

    with pytest.raises(ValueError, match="Unsupported order side"):
        factory.create_limit_order("BTCUSD", True, 100.0, 1.0)

    with pytest.raises(ValueError, match="Unsupported order type"):
        factory.create_order("BTCUSD", OrderSide.BUY, "LIMIT", 100.0, 1.0)

    with pytest.raises(ValueError, match="Unsupported order type"):
        factory.create_order("BTCUSD", OrderSide.BUY, True, 100.0, 1.0)


def test_submit_apis_return_structured_rejections_for_non_numeric_inputs():
    book = OrderBook("BTCUSD")
    factory = OrderFactory()

    with pytest.raises(ValueError, match="Order price must be positive"):
        factory.create_limit_order("BTCUSD", OrderSide.BUY, "100", 1.0)

    with pytest.raises(ValueError, match="Order quantity must be positive"):
        factory.create_market_order("BTCUSD", OrderSide.BUY, "1")

    rejected_price = book.submit_limit_order(OrderSide.BUY, "100", 1.0)
    rejected_quantity = book.submit_market_order(OrderSide.BUY, "1")
    rejected_bool = book.submit_ioc_order(OrderSide.BUY, True, 1.0)

    assert "Order price must be positive" in rejected_price.rejected_reason
    assert "Order quantity must be positive" in rejected_quantity.rejected_reason
    assert "Order price must be positive" in rejected_bool.rejected_reason
    assert book.get_active_order_ids() == []


def test_order_factory_allocates_unique_ids_across_threads():
    factory = OrderFactory()
    ids = []
    ids_lock = threading.Lock()

    def create_orders():
        local_ids = [
            factory.create_limit_order("BTCUSD", OrderSide.BUY, 100.0, 1.0).order_id
            for _ in range(100)
        ]
        with ids_lock:
            ids.extend(local_ids)

    threads = [threading.Thread(target=create_orders) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(ids) == 800
    assert sorted(ids) == list(range(1, 801))


def test_symbols_are_non_empty_and_normalized():
    factory = OrderFactory()

    with pytest.raises(ValueError, match="non-empty string"):
        OrderBook("")

    with pytest.raises(ValueError, match="non-empty string"):
        factory.create_limit_order("   ", OrderSide.BUY, 100.0, 1.0)

    book = OrderBook(" BTCUSD ")
    order = factory.create_limit_order(" BTCUSD ", OrderSide.BUY, 100.0, 1.0)
    result = book.submit_order(order)

    assert book.symbol == "BTCUSD"
    assert order.symbol == "BTCUSD"
    assert result.rejected_reason is None
    assert result.rested is True

    manual_order = Order(
        99,
        " BTCUSD ",
        int(OrderSide.SELL),
        int(OrderType.LIMIT),
        101.0,
        1.0,
        time.time_ns(),
        NO_OWNER,
    )
    manual_result = book.submit_order(manual_order)

    assert manual_result.rejected_reason is None
    assert manual_order.symbol == "BTCUSD"


def test_reused_order_ids_are_rejected_after_cancel():
    book = OrderBook("BTCUSD")
    first = book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)

    assert book.cancel_order(first.order.order_id) is True

    duplicate = OrderFactory().create_limit_order("BTCUSD", OrderSide.SELL, 101.0, 1.0)
    rejected = book.submit_order(duplicate)
    next_order = book.submit_limit_order(OrderSide.BUY, 99.0, 1.0)

    assert "already been processed" in rejected.rejected_reason
    assert next_order.rejected_reason is None
    assert next_order.order.order_id == first.order.order_id + 1


def test_reused_order_ids_are_rejected_after_fills_and_immediate_cancels():
    filled_book = OrderBook("BTCUSD")
    resting = filled_book.submit_limit_order(OrderSide.SELL, 100.0, 1.0)
    taker = filled_book.submit_limit_order(OrderSide.BUY, 100.0, 1.0)
    duplicate_resting = OrderFactory().create_limit_order("BTCUSD", OrderSide.SELL, 101.0, 1.0)

    assert len(taker.trades) == 1
    assert filled_book.get_order(resting.order.order_id) is None
    assert "already been processed" in filled_book.submit_order(duplicate_resting).rejected_reason

    ioc_book = OrderBook("BTCUSD")
    ioc = ioc_book.submit_ioc_order(OrderSide.BUY, 100.0, 1.0)
    duplicate_ioc = OrderFactory().create_limit_order("BTCUSD", OrderSide.BUY, 99.0, 1.0)

    assert ioc.cancelled is True
    assert "already been processed" in ioc_book.submit_order(duplicate_ioc).rejected_reason

    fok_book = OrderBook("BTCUSD")
    fok = fok_book.submit_fok_order(OrderSide.BUY, 100.0, 1.0)
    duplicate_fok = OrderFactory().create_limit_order("BTCUSD", OrderSide.BUY, 99.0, 1.0)

    assert fok.rejected_reason == "FOK order could not be fully filled"
    assert "already been processed" in fok_book.submit_order(duplicate_fok).rejected_reason


def test_depth_and_recent_trade_inputs_do_not_slice_surprisingly():
    book = OrderBook("BTCUSD")
    book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0)

    assert book.get_recent_trades(0) == []
    assert book.get_recent_trades(-1) == []

    with pytest.raises(ValueError, match="levels must be non-negative"):
        book.get_order_book_depth(-1)


def test_snapshot_interval_must_be_positive():
    book = OrderBook("BTCUSD")

    with pytest.raises(ValueError, match="snapshot interval must be positive"):
        book.set_snapshot_interval(0)

    with pytest.raises(ValueError, match="snapshot interval must be positive"):
        book.set_snapshot_interval(-1)


def test_simulation_export_creates_parent_directories(tmp_path: Path):
    engine = SimulationEngine(
        SimulationConfig(duration_seconds=0, target_throughput=1, enable_magic_trace=False)
    )

    output_path = tmp_path / "nested" / "results" / "simulation.json"

    assert engine.export_results({"ok": True}, str(output_path)) == str(output_path)
    assert output_path.exists()


def test_performance_monitor_export_creates_parent_directories(tmp_path: Path):
    monitor = PerformanceMonitor(enable_magic_trace=False)
    output_path = tmp_path / "nested" / "metrics" / "performance.json"

    assert monitor.export_metrics(str(output_path)) == str(output_path)
    assert output_path.exists()


def test_metrics_collector_exposes_current_value_for_dashboard():
    collector = MetricsCollector()

    collector.record_metric("throughput_ops_per_sec", 10.0)
    collector.record_metric("throughput_ops_per_sec", 25.0)
    stats = collector.get_metric_statistics("throughput_ops_per_sec")

    assert stats["current"] == 25.0
    assert stats["latest_timestamp"] > 0


def test_trace_html_report_escapes_insights_and_embeds_plotly(tmp_path: Path):
    visualizer = TraceVisualizer(
        {
            "summary": {
                "total_function_calls": 1,
                "unique_functions": 1,
                "total_traced_time_ms": 0.1,
                "trace_overhead_percentage": 0.0,
            },
            "performance_insights": ["<script>alert('xss')</script>"],
        }
    )

    output_path = tmp_path / "reports" / "trace.html"

    assert visualizer.generate_html_report(str(output_path)) is True
    content = output_path.read_text(encoding="utf-8")

    assert "&lt;script&gt;alert(&#x27;xss&#x27;)&lt;/script&gt;" in content
    assert "<script>alert('xss')</script>" not in content
    assert "plotly-latest" not in content
    assert '<script src="https://cdn.plot.ly' not in content


def test_magic_trace_raw_artifact_analysis_is_honest(tmp_path: Path):
    config = MagicTraceConfig()
    config.output_dir = str(tmp_path)
    session = MagicTraceSession(config, "raw_trace")
    session.trace_file.write_bytes(b"trace-data")

    session._analyze_trace()

    content = session.analysis_file.read_text(encoding="utf-8")

    assert "raw_trace_collected" in content
    assert "would be performed here" not in content


def test_magic_trace_stop_reaps_self_exited_child_without_wedging(tmp_path: Path):
    config = MagicTraceConfig()
    config.output_dir = str(tmp_path)
    config.auto_analyze = False
    session = MagicTraceSession(config, "qa")
    # magic-trace runs with a duration cap, so it can self-exit before stop().
    session.process = subprocess.Popen(["sleep", "0.05"])
    session.is_active = True
    time.sleep(0.2)  # let the child exit

    assert session.stop() is True
    assert session.is_active is False
    assert session.process.returncode is not None  # reaped, not a zombie
    assert session.stop() is False  # already inactive, not wedged


def test_magic_trace_session_name_cannot_escape_output_directory(tmp_path: Path):
    config = MagicTraceConfig()
    config.output_dir = str(tmp_path / "traces")

    session = MagicTraceSession(config, "../outside/<script>")

    assert session.session_name == "script"
    for artifact_path in (session.trace_file, session.metadata_file, session.analysis_file):
        assert artifact_path.parent.resolve() == session.output_path.resolve()
        assert ".." not in artifact_path.name
        assert "/" not in artifact_path.name


def test_magic_trace_export_skips_symlinks(tmp_path: Path):
    config = MagicTraceConfig()
    config.output_dir = str(tmp_path / "traces")
    trace_dir = Path(config.output_dir)
    trace_dir.mkdir()
    (trace_dir / "real_trace.json").write_text("{}", encoding="utf-8")
    outside_file = tmp_path / "outside.json"
    outside_file.write_text('{"secret": true}', encoding="utf-8")

    try:
        os.symlink(outside_file, trace_dir / "linked_trace.json")
    except OSError:
        pytest.skip("symlink creation is unavailable on this platform")

    export_dir = tmp_path / "exports"
    profiler = MagicTraceProfiler(config)

    assert profiler.export_traces(str(export_dir)) is True
    assert (export_dir / "real_trace.json").exists()
    assert not (export_dir / "linked_trace.json").exists()


def test_magic_trace_uses_thread_safe_subprocess_options(monkeypatch, tmp_path: Path):
    config = MagicTraceConfig()
    config.output_dir = str(tmp_path)
    session = MagicTraceSession(config, "thread_safe_launch")
    popen_calls = []

    class FakeProcess:
        pid = 12345

    def fake_popen(cmd, **kwargs):
        popen_calls.append((cmd, kwargs))
        return FakeProcess()

    monkeypatch.setattr(session, "_check_magic_trace_available", lambda: True)
    monkeypatch.setattr(session, "_build_magic_trace_command", lambda: ["magic-trace", "attach"])
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    assert session.start() is True
    assert popen_calls

    _, kwargs = popen_calls[0]
    assert kwargs["start_new_session"] is True
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
