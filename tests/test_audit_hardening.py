from pathlib import Path

import pytest

from tracebook.core.order import OrderFactory, OrderSide
from tracebook.core.orderbook import OrderBook
from tracebook.profiling.magic_trace_wrapper import MagicTraceConfig, MagicTraceSession
from tracebook.profiling.performance_monitor import PerformanceMonitor
from tracebook.profiling.trace_visualizer import TraceVisualizer
from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine


def test_order_factory_rejects_non_numeric_side_and_order_type():
    factory = OrderFactory()

    with pytest.raises(ValueError, match="Unsupported order side"):
        factory.create_limit_order("BTCUSD", "BUY", 100.0, 1.0)

    with pytest.raises(ValueError, match="Unsupported order type"):
        factory.create_order("BTCUSD", OrderSide.BUY, "LIMIT", 100.0, 1.0)


def test_depth_and_recent_trade_inputs_do_not_slice_surprisingly():
    book = OrderBook("BTCUSD")
    book.add_limit_order(OrderSide.SELL, 100.0, 1.0)
    book.add_limit_order(OrderSide.BUY, 100.0, 1.0)

    assert book.get_recent_trades(0) == []
    assert book.get_recent_trades(-1) == []

    with pytest.raises(ValueError, match="levels must be non-negative"):
        book.get_order_book_depth(-1)


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
