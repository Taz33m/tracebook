import importlib
import sys
import types


def _install_dashboard_dependency_stubs(monkeypatch):
    dash_module = types.ModuleType("dash")

    class FakeDash:
        pass

    dash_module.Dash = FakeDash
    dash_module.dcc = types.SimpleNamespace()
    dash_module.html = types.SimpleNamespace()
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
            FakeSimulationEngine.last = self

        def run_simulation(self):
            self.run_called = True

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
            FakeThread.last = self

        def start(self):
            self.started = True

    class FakeDashboard:
        def __init__(self):
            self.manager = None
            self.run_host = None

        def set_order_book_manager(self, manager):
            self.manager = manager

        def run(self, host):
            self.run_host = host

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
    assert FakeThread.last.daemon is True
    assert FakeThread.last.target == FakeSimulationEngine.last.run_simulation
    assert FakeThread.last.started is True
    assert FakeSimulationEngine.last.run_called is False
