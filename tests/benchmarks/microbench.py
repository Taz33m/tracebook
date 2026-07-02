"""Deterministic single-threaded microbenchmarks for core order-book operations.

Unlike the simulation benchmark runner (which is wall-clock and thread
scheduling dependent), these measure individual operations in isolation so a
data-structure change can be judged by ops/sec and ns/op. Run directly:

    python tests/benchmarks/microbench.py
    python tests/benchmarks/microbench.py --n 20000

The scenarios are chosen to expose specific hot paths:
- add_deep:    append many orders to a single price level (in-level insert)
- cancel_deep: cancel many orders on a single price level (in-level removal)
- add_wide:    add one order each across many price levels (level index insert)
- match:       cross an equal number of resting orders (matching loop)
"""

import argparse
import sys
import time
from pathlib import Path

# Allow running as a plain script from the repo root.
SRC = Path(__file__).resolve().parents[2] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tracebook.core.order import OrderSide  # noqa: E402
from tracebook.core.orderbook import OrderBook  # noqa: E402

TICK = 0.01
BASE_PRICE = 1000.0


def _time_ops(fn, ops: int) -> dict:
    """Run fn (which performs `ops` operations) and report throughput."""
    start = time.perf_counter_ns()
    fn()
    elapsed_ns = time.perf_counter_ns() - start
    ns_per_op = elapsed_ns / ops if ops else 0.0
    ops_per_sec = ops / (elapsed_ns / 1_000_000_000) if elapsed_ns else 0.0
    return {"ops": ops, "ns_per_op": ns_per_op, "ops_per_sec": ops_per_sec}


def bench_add_deep(n: int) -> dict:
    """Append n non-crossing orders onto a single price level."""
    book = OrderBook("BENCH")

    def run():
        for _ in range(n):
            book.add_limit_order(OrderSide.BUY, BASE_PRICE, 1.0)

    return _time_ops(run, n)


def bench_cancel_deep(n: int) -> dict:
    """Cancel n resting orders that all sit on one price level (FIFO order)."""
    book = OrderBook("BENCH")
    ids = [book.submit_limit_order(OrderSide.BUY, BASE_PRICE, 1.0).order.order_id for _ in range(n)]

    def run():
        for order_id in ids:
            book.cancel_order(order_id)

    return _time_ops(run, n)


def bench_add_wide(n: int) -> dict:
    """Add one non-crossing order each across n distinct price levels."""
    book = OrderBook("BENCH")
    # Bids strictly below the base so nothing crosses; distinct ticks.
    prices = [round(BASE_PRICE - (i + 1) * TICK, 2) for i in range(n)]

    def run():
        for price in prices:
            book.add_limit_order(OrderSide.BUY, price, 1.0)

    return _time_ops(run, n)


def bench_match(n: int) -> dict:
    """Cross n resting asks with n aggressive buys (one trade each)."""
    book = OrderBook("BENCH")
    for _ in range(n):
        book.add_limit_order(OrderSide.SELL, BASE_PRICE, 1.0)

    def run():
        for _ in range(n):
            book.add_limit_order(OrderSide.BUY, BASE_PRICE, 1.0)

    return _time_ops(run, n)


SCENARIOS = {
    "add_deep": bench_add_deep,
    "cancel_deep": bench_cancel_deep,
    "add_wide": bench_add_wide,
    "match": bench_match,
}


def run_all(n: int, warmup: int = 1000) -> dict:
    """Warm up (JIT compile + caches) then measure each scenario."""
    for fn in SCENARIOS.values():
        fn(warmup)  # discard: warms Numba/CPython caches

    return {name: fn(n) for name, fn in SCENARIOS.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Order-book operation microbenchmarks")
    parser.add_argument("--n", type=int, default=10000, help="operations per scenario")
    parser.add_argument("--warmup", type=int, default=1000, help="warmup operations")
    args = parser.parse_args()

    results = run_all(args.n, args.warmup)

    print(f"\nMicrobenchmark (n={args.n}, warmup={args.warmup})")
    print(f"{'scenario':<14}{'ops/sec':>16}{'ns/op':>14}")
    print("-" * 44)
    for name, stats in results.items():
        print(f"{name:<14}{stats['ops_per_sec']:>16,.0f}{stats['ns_per_op']:>14,.0f}")
    print()


if __name__ == "__main__":
    main()
