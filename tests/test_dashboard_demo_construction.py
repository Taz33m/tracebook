import importlib
import sys
import types

import pytest


def test_dashboard_help_and_version_work_without_optional_dependencies(monkeypatch, capsys):
    sys.modules.pop("tracebook.visualization.dashboard", None)
    monkeypatch.setitem(sys.modules, "dash", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")

    try:
        with pytest.raises(RuntimeError, match="Dashboard dependencies"):
            dashboard_module.PerformanceDashboard()
        with pytest.raises(SystemExit) as version_exit:
            dashboard_module.main(["--version"])
        assert version_exit.value.code == 0
        assert "tracebook" in capsys.readouterr().out
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)


def _install_dashboard_dependency_stubs(monkeypatch):
    dash_module = types.ModuleType("dash")

    class FakeComponentFactory:
        def __getattr__(self, name):
            return lambda *args, **kwargs: (name, args, kwargs)

    class FakeDash:
        def __init__(self, *args, **kwargs):
            pass

        def callback(self, *args, **kwargs):
            return lambda func: func

    dash_module.Dash = FakeDash
    dash_module.dcc = FakeComponentFactory()
    dash_module.html = FakeComponentFactory()
    dash_module.Input = lambda *args, **kwargs: (args, kwargs)
    dash_module.Output = lambda *args, **kwargs: (args, kwargs)
    dash_module.callback = lambda *args, **kwargs: lambda func: func

    plotly_module = types.ModuleType("plotly")
    graph_objs_module = types.ModuleType("plotly.graph_objs")
    express_module = types.ModuleType("plotly.express")
    pandas_module = types.ModuleType("pandas")

    monkeypatch.setitem(sys.modules, "dash", dash_module)
    monkeypatch.setitem(sys.modules, "plotly", plotly_module)
    monkeypatch.setitem(sys.modules, "plotly.graph_objs", graph_objs_module)
    monkeypatch.setitem(sys.modules, "plotly.express", express_module)
    monkeypatch.setitem(sys.modules, "pandas", pandas_module)


def test_dashboard_demo_mode_constructs_engine_and_dashboard_without_running_server(
    monkeypatch,
):
    _install_dashboard_dependency_stubs(monkeypatch)
    sys.modules.pop("tracebook.visualization.dashboard", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")

    fake_simulation_module = types.ModuleType("tracebook.simulation.simulation_engine")

    class FakeSimulationConfig:
        last = None

        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            FakeSimulationConfig.last = self

    class FakeSimulationEngine:
        last = None

        def __init__(self, config):
            self.config = config
            self.performance_monitor = object()
            self.order_book_manager = object()
            self.run_called = False
            self.stop_called = False
            FakeSimulationEngine.last = self

        def run_simulation(self):
            self.run_called = True

        def stop(self):
            self.stop_called = True

    fake_simulation_module.SimulationConfig = FakeSimulationConfig
    fake_simulation_module.SimulationEngine = FakeSimulationEngine
    monkeypatch.setitem(
        sys.modules,
        "tracebook.simulation.simulation_engine",
        fake_simulation_module,
    )

    class FakeThread:
        last = None

        def __init__(self, target, daemon):
            self.target = target
            self.daemon = daemon
            self.started = False
            self.join_timeout = None
            FakeThread.last = self

        def start(self):
            self.started = True

        def join(self, timeout=None):
            self.join_timeout = timeout

    class FakeDashboard:
        def __init__(self):
            self.manager = None
            self.run_host = None
            self.allow_remote = None

        def set_order_book_manager(self, manager):
            self.manager = manager

        def run(self, host, allow_remote=False):
            self.run_host = host
            self.allow_remote = allow_remote

    created = {}

    def fake_create_dashboard(port, update_interval, performance_monitor):
        created["port"] = port
        created["update_interval"] = update_interval
        created["performance_monitor"] = performance_monitor
        created["dashboard"] = FakeDashboard()
        return created["dashboard"]

    monkeypatch.setattr(dashboard_module.threading, "Thread", FakeThread)
    monkeypatch.setattr(dashboard_module, "create_dashboard", fake_create_dashboard)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracebook-dashboard",
            "--port",
            "9001",
            "--host",
            "0.0.0.0",
            "--allow-remote",
            "--update-interval",
            "250",
            "--demo-simulation",
            "--demo-duration",
            "1.5",
            "--demo-throughput",
            "25.0",
            "--seed",
            "99",
        ],
    )

    try:
        exit_code = dashboard_module.main()
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)

    assert exit_code == 0
    assert FakeSimulationConfig.last.duration_seconds == 1.5
    assert FakeSimulationConfig.last.target_throughput == 25.0
    assert FakeSimulationConfig.last.matching_algorithm == "FIFO"
    assert FakeSimulationConfig.last.enable_magic_trace is False
    assert FakeSimulationConfig.last.seed == 99
    assert FakeSimulationConfig.last.cancel_ratio == 0.05
    assert FakeSimulationConfig.last.replace_ratio == 0.02
    assert created["port"] == 9001
    assert created["update_interval"] == 250
    assert created["performance_monitor"] is FakeSimulationEngine.last.performance_monitor
    assert created["dashboard"].manager is FakeSimulationEngine.last.order_book_manager
    assert created["dashboard"].run_host == "0.0.0.0"
    assert created["dashboard"].allow_remote is True
    assert FakeThread.last.daemon is True
    assert FakeThread.last.target == FakeSimulationEngine.last.run_simulation
    assert FakeThread.last.started is True
    assert FakeThread.last.join_timeout == 5.0
    assert FakeSimulationEngine.last.run_called is False
    assert FakeSimulationEngine.last.stop_called is True


def test_dashboard_rejects_non_loopback_host_without_explicit_override(monkeypatch):
    _install_dashboard_dependency_stubs(monkeypatch)
    sys.modules.pop("tracebook.visualization.dashboard", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tracebook-dashboard",
            "--host",
            "0.0.0.0",
        ],
    )

    try:
        with pytest.raises(SystemExit):
            dashboard_module.main()
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)


def test_dashboard_run_rejects_non_loopback_host_without_explicit_override(monkeypatch):
    _install_dashboard_dependency_stubs(monkeypatch)
    sys.modules.pop("tracebook.visualization.dashboard", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")

    try:
        dashboard = dashboard_module.PerformanceDashboard()
        with pytest.raises(ValueError, match="Non-loopback dashboard hosts require"):
            dashboard.run(host="0.0.0.0")
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"update_interval": 0}, "positive integer"),
        ({"update_interval": 1.5}, "positive integer"),
        ({"port": -1}, "between 0 and 65535"),
        ({"port": 65536}, "between 0 and 65535"),
    ],
)
def test_dashboard_rejects_invalid_server_configuration(monkeypatch, kwargs, message):
    _install_dashboard_dependency_stubs(monkeypatch)
    sys.modules.pop("tracebook.visualization.dashboard", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")

    try:
        with pytest.raises(ValueError, match=message):
            dashboard_module.PerformanceDashboard(**kwargs)
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)


def test_dashboard_data_uses_current_throughput_metric(monkeypatch):
    _install_dashboard_dependency_stubs(monkeypatch)
    sys.modules.pop("tracebook.visualization.dashboard", None)
    dashboard_module = importlib.import_module("tracebook.visualization.dashboard")

    try:
        with pytest.raises(ValueError, match="max_points"):
            dashboard_module.DashboardData(max_points=0)
        data = dashboard_module.DashboardData()
        data.update_metrics(
            {
                "performance_metrics": {
                    "throughput_ops_per_sec": {"current": 42.0, "mean": 10.0},
                    "order_processing_latency_ms": {"mean": 0.25, "p99": 0.75},
                },
                "system_resources": {"process_cpu_percent": 1.0, "process_memory_mb": 64.0},
                "session_metrics": {
                    "orders_processed": 10,
                    "trades_executed": 4,
                    "total_volume": 123.0,
                    "peak_throughput_ops_per_sec": 50.0,
                },
            }
        )

        assert data.get_time_series_data()["throughput"] == [42.0]
        assert data.get_summary_stats()["current_throughput"] == 42.0
    finally:
        sys.modules.pop("tracebook.visualization.dashboard", None)
