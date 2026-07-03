"""Tests for the standalone live order-book web frontend."""

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest

from tracebook.simulation.simulation_engine import SimulationConfig, SimulationEngine
from tracebook.visualization import web_server


@pytest.fixture
def running_engine():
    config = SimulationConfig(
        duration_seconds=5.0,
        target_throughput=400.0,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        seed=7,
        symbols=["BTCUSD"],
        cancel_ratio=0.05,
        replace_ratio=0.02,
    )
    engine = SimulationEngine(config)
    threading.Thread(target=engine.run_simulation, daemon=True).start()
    # Wait (bounded) until some liquidity has built up so state is meaningful.
    book = engine.order_book_manager.get_order_book("BTCUSD")
    for _ in range(40):
        if book is not None and book.get_best_bid() is not None:
            break
        time.sleep(0.05)
    return engine


def test_build_state_schema_is_complete_and_json_serializable(running_engine):
    state = web_server.build_state(running_engine, "BTCUSD")

    for key in (
        "symbol",
        "tick_size",
        "best_bid",
        "best_ask",
        "mid",
        "spread",
        "bids",
        "asks",
        "trades",
        "stats",
        "timestamp",
    ):
        assert key in state, f"missing top-level key {key}"
    for key in (
        "orders",
        "trades",
        "throughput",
        "latency_mean_ms",
        "latency_p50_ms",
        "latency_p95_ms",
        "latency_p99_ms",
    ):
        assert key in state["stats"], f"missing stats key {key}"

    assert state["symbol"] == "BTCUSD"
    # Depth rows are [price, size, count] triples.
    for level in state["bids"] + state["asks"]:
        assert len(level) == 3
    json.dumps(state, default=str)  # must be serializable


def test_is_loopback_host():
    assert web_server.is_loopback_host("127.0.0.1")
    assert web_server.is_loopback_host("localhost")
    assert web_server.is_loopback_host("::1")
    assert not web_server.is_loopback_host("0.0.0.0")
    assert not web_server.is_loopback_host("example.com")


def test_non_loopback_host_requires_allow_remote(running_engine):
    with pytest.raises(ValueError, match="allow_remote"):
        web_server.create_server(running_engine, "BTCUSD", host="0.0.0.0", port=0)


def test_negative_depth_levels_is_rejected(running_engine):
    with pytest.raises(ValueError, match="non-negative"):
        web_server.create_server(running_engine, "BTCUSD", port=0, depth_levels=-1)


def test_ipv6_loopback_host_binds(running_engine):
    try:
        server = web_server.create_server(running_engine, "BTCUSD", host="::1", port=0)
    except OSError:
        pytest.skip("IPv6 not available in this environment")
    try:
        assert server.address_family == socket.AF_INET6
    finally:
        server.server_close()


def test_server_serves_api_state_and_static_assets(running_engine):
    server = web_server.create_server(running_engine, "BTCUSD", host="127.0.0.1", port=0)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{port}"
    try:
        time.sleep(0.1)
        api = json.loads(urllib.request.urlopen(f"{base}/api/state", timeout=3).read())
        assert api["symbol"] == "BTCUSD"

        html = urllib.request.urlopen(f"{base}/", timeout=3).read().decode()
        assert "<title>tracebook" in html
        assert len(urllib.request.urlopen(f"{base}/app.js", timeout=3).read()) > 0
        assert len(urllib.request.urlopen(f"{base}/styles.css", timeout=3).read()) > 0

        # Unknown / traversal paths are refused.
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(f"{base}/does-not-exist.txt", timeout=3)
    finally:
        server.shutdown()
