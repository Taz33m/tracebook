"""Standalone live order-book web frontend.

A dependency-free frontend for the order book: a static HTML/CSS/JS page served
by a stdlib HTTP server, backed by a background simulation. The page polls
``/api/state`` for a JSON snapshot (top of book, depth ladder, recent trades,
engine metrics). Distinct from the Dash dashboard and needs no dashboard extras.

    tracebook-web --port 8080

The server is unauthenticated, so it binds to loopback by default; a non-loopback
host requires ``--allow-remote``.
"""

import argparse
import ipaddress
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional

from .. import __version__

STATIC_DIR = Path(__file__).parent / "web"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "application/javascript; charset=utf-8",
}


def is_loopback_host(host: str) -> bool:
    """Return True if host is loopback (so an unauthenticated bind is safe)."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def build_state(
    engine, symbol: str, depth_levels: int = 12, trade_count: int = 30
) -> Dict[str, Any]:
    """Build the JSON snapshot the frontend renders from a running engine."""
    manager = engine.order_book_manager
    book = manager.get_order_book(symbol)
    if book is None:
        symbols = manager.get_all_symbols()
        if symbols:
            symbol = symbols[0]
            book = manager.get_order_book(symbol)

    empty: Dict[str, Any] = {
        "symbol": symbol,
        "tick_size": None,
        "best_bid": None,
        "best_ask": None,
        "mid": None,
        "spread": None,
        "bids": [],
        "asks": [],
        "trades": [],
        "stats": {},
        "timestamp": time.time_ns(),
    }
    if book is None:
        return empty

    depth = book.get_order_book_depth(levels=depth_levels)
    trades = [
        {"price": t.price, "quantity": t.quantity, "timestamp": t.timestamp}
        for t in book.get_recent_trades(trade_count)
    ]

    stats = book.get_statistics()
    summary = engine.performance_monitor.get_performance_summary()
    perf = summary.get("performance_metrics", {})
    latency = perf.get("order_processing_latency_ms", {})
    throughput = perf.get("throughput_ops_per_sec", {})

    return {
        "symbol": symbol,
        "tick_size": book.tick_size,
        "best_bid": book.get_best_bid(),
        "best_ask": book.get_best_ask(),
        "mid": book.get_mid_price(),
        "spread": book.get_spread(),
        "bids": [list(level) for level in depth["bids"]],
        "asks": [list(level) for level in depth["asks"]],
        "trades": trades,
        "stats": {
            "orders": stats.get("orders_added", 0),
            "trades": stats.get("total_trades", 0),
            "throughput": throughput.get("current", 0.0),
            "latency_mean_ms": latency.get("mean", 0.0),
            "latency_p50_ms": latency.get("median", 0.0),
            "latency_p95_ms": latency.get("p95", 0.0),
            "latency_p99_ms": latency.get("p99", 0.0),
        },
        "timestamp": time.time_ns(),
    }


def _make_handler(engine, symbol: str, depth_levels: int, trade_count: int):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):  # keep the console quiet
            return

        def do_GET(self):
            path = self.path.split("?", 1)[0]
            if path == "/api/state":
                self._send_json(build_state(engine, symbol, depth_levels, trade_count))
            else:
                self._serve_static(path)

        def _send_json(self, payload: Dict[str, Any]):
            body = json.dumps(payload, default=str).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _serve_static(self, path: str):
            # Path(...).name strips any directory components, so only files
            # directly in STATIC_DIR are reachable (no traversal).
            name = "index.html" if path in ("/", "") else Path(path).name
            target = STATIC_DIR / name
            if not target.is_file():
                self.send_error(404)
                return
            body = target.read_bytes()
            self.send_response(200)
            self.send_header(
                "Content-Type", _CONTENT_TYPES.get(target.suffix, "application/octet-stream")
            )
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def create_server(
    engine,
    symbol: str,
    host: str = "127.0.0.1",
    port: int = 8080,
    depth_levels: int = 12,
    trade_count: int = 30,
    allow_remote: bool = False,
) -> ThreadingHTTPServer:
    """Create (but do not start) the web server bound to host:port."""
    if depth_levels < 0:
        raise ValueError("depth_levels must be non-negative")
    if not is_loopback_host(host) and not allow_remote:
        raise ValueError(
            "Non-loopback host requires allow_remote=True because the web "
            "frontend has no authentication"
        )
    handler = _make_handler(engine, symbol, depth_levels, trade_count)
    # Match the address family to the host so an IPv6 host (e.g. ::1) binds.
    family = socket.AF_INET6 if ":" in host else socket.AF_INET

    class _Server(ThreadingHTTPServer):
        address_family = family

    return _Server((host, port), handler)


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="Serve the tracebook live order-book frontend.")
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--symbol", default="BTCUSD", help="Symbol to display.")
    parser.add_argument("--throughput", type=float, default=500.0, help="Simulated orders/sec.")
    parser.add_argument("--seed", type=int, default=1337, help="Deterministic order-flow seed.")
    parser.add_argument(
        "--duration",
        type=float,
        default=3600.0,
        help="How long the background simulation runs (seconds).",
    )
    parser.add_argument("--depth-levels", type=int, default=12, help="Depth levels per side.")
    parser.add_argument(
        "--allow-remote",
        action="store_true",
        help="Allow binding the unauthenticated frontend to a non-loopback host.",
    )
    args = parser.parse_args(argv)

    if not is_loopback_host(args.host) and not args.allow_remote:
        parser.error(
            "--host with a non-loopback address requires --allow-remote because "
            "the web frontend has no authentication"
        )
    if args.depth_levels < 0:
        parser.error("--depth-levels must be non-negative")

    from ..simulation.simulation_engine import SimulationConfig, SimulationEngine

    config = SimulationConfig(
        duration_seconds=args.duration,
        target_throughput=args.throughput,
        matching_algorithm="FIFO",
        enable_magic_trace=False,
        seed=args.seed,
        symbols=[args.symbol],
        cancel_ratio=0.06,
        replace_ratio=0.03,
    )
    engine = SimulationEngine(config)
    threading.Thread(target=engine.run_simulation, daemon=True).start()

    server = create_server(
        engine,
        symbol=args.symbol,
        host=args.host,
        port=args.port,
        depth_levels=args.depth_levels,
        allow_remote=args.allow_remote,
    )
    print(f"tracebook live book on http://{args.host}:{args.port}  (symbol {args.symbol})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
