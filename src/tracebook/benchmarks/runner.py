"""CLI benchmark runner with warmup and JSON output."""

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import dataclass, asdict, replace
from importlib import metadata
from typing import Any, Dict, List, Optional, Tuple

from .. import __version__
from ..simulation.simulation_engine import run_benchmark_simulation


@dataclass(frozen=True)
class BenchmarkScenario:
    """A fixed benchmark scenario."""

    name: str
    duration_seconds: float
    target_throughput: float
    matching_algorithm: str
    cancel_ratio: float = 0.0
    replace_ratio: float = 0.0
    symbols: Tuple[str, ...] = ("BTCUSD",)


SCENARIOS = {
    "smoke": BenchmarkScenario("smoke", 1.0, 100.0, "FIFO"),
    "fifo_baseline": BenchmarkScenario("fifo_baseline", 5.0, 500.0, "FIFO"),
    "pro_rata_baseline": BenchmarkScenario("pro_rata_baseline", 5.0, 500.0, "PRO_RATA"),
    "cancellation_mix": BenchmarkScenario("cancellation_mix", 5.0, 500.0, "FIFO", 0.15, 0.05),
    # Deeper book: higher throughput with no cancels so resting liquidity builds up.
    "deep_book": BenchmarkScenario("deep_book", 5.0, 2000.0, "FIFO"),
    # Cancel-heavy churn beyond the standard cancellation_mix.
    "high_cancellation": BenchmarkScenario("high_cancellation", 5.0, 500.0, "FIFO", 0.35, 0.10),
    # Pro-rata path under lifecycle events (pro_rata_baseline has none).
    "pro_rata_cancellation": BenchmarkScenario(
        "pro_rata_cancellation", 5.0, 500.0, "PRO_RATA", 0.15, 0.05
    ),
    # Multiple symbols, splitting throughput across independent books.
    "multi_symbol": BenchmarkScenario(
        "multi_symbol", 5.0, 500.0, "FIFO", symbols=("BTCUSD", "ETHUSD", "SOLUSD")
    ),
}


def _dependency_versions() -> Dict[str, Optional[str]]:
    packages = ["tracebook", "numpy", "psutil", "dash", "plotly"]
    versions: Dict[str, Optional[str]] = {}
    for package in packages:
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _machine_metadata() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "dependency_versions": _dependency_versions(),
    }


def _metric_summary(results: Dict[str, Any]) -> Dict[str, Any]:
    perf = results.get("performance_data", {}).get("performance_metrics", {})
    latency = perf.get("order_processing_latency_ms", {})
    generation = perf.get("order_generation_latency_ms", {})
    event_latency = perf.get("order_event_latency_ms", {})
    resources = results.get("performance_data", {}).get("system_resources", {})
    overhead = results.get("performance_data", {}).get("collection_overhead", {})
    summary = results.get("summary_metrics", {})

    return {
        "orders_processed": summary.get("total_orders_processed", 0),
        "events_processed": summary.get("total_events_processed", 0),
        "cancel_events": summary.get("total_cancel_events", 0),
        "replace_events": summary.get("total_replace_events", 0),
        "trades_executed": summary.get("total_trades_executed", 0),
        "throughput_orders_per_sec": summary.get("actual_throughput", 0.0),
        "latency_ms": {
            "count": latency.get("count", 0),
            "mean": latency.get("mean", 0.0),
            "p50": latency.get("median", 0.0),
            "p95": latency.get("p95", 0.0),
            "p99": latency.get("p99", 0.0),
            "max": latency.get("max", 0.0),
        },
        "event_latency_ms": {
            "count": event_latency.get("count", 0),
            "mean": event_latency.get("mean", 0.0),
            "p95": event_latency.get("p95", 0.0),
            "p99": event_latency.get("p99", 0.0),
        },
        "generation_latency_ms": {
            "count": generation.get("count", 0),
            "mean": generation.get("mean", 0.0),
            "p95": generation.get("p95", 0.0),
            "p99": generation.get("p99", 0.0),
        },
        "memory_mb": resources.get("process_memory_mb", 0.0),
        "monitoring_overhead_ns": overhead,
    }


def run_scenario(
    scenario: BenchmarkScenario,
    seed: int,
    warmup_seconds: float,
) -> Dict[str, Any]:
    """Run one scenario and return a compact benchmark result."""
    started_at = time.time_ns()
    results = run_benchmark_simulation(
        duration=scenario.duration_seconds,
        throughput=scenario.target_throughput,
        algorithm=scenario.matching_algorithm,
        enable_magic_trace=False,
        seed=seed,
        cancel_ratio=scenario.cancel_ratio,
        replace_ratio=scenario.replace_ratio,
        warmup_seconds=warmup_seconds,
        symbols=list(scenario.symbols),
    )

    return {
        "name": scenario.name,
        "started_at": started_at,
        "completed_at": time.time_ns(),
        "config": {
            **asdict(scenario),
            "seed": seed,
            "warmup_seconds": warmup_seconds,
        },
        "summary": _metric_summary(results),
        "raw_result": results,
    }


def run_benchmarks(
    scenario_names: List[str],
    seed: int = 1337,
    warmup_seconds: float = 0.05,
    duration_override: Optional[float] = None,
    throughput_override: Optional[float] = None,
) -> Dict[str, Any]:
    """Run selected benchmark scenarios."""
    selected = []
    for name in scenario_names:
        scenario = SCENARIOS[name]
        if duration_override is not None:
            scenario = replace(scenario, duration_seconds=duration_override)
        if throughput_override is not None:
            scenario = replace(scenario, target_throughput=throughput_override)
        selected.append(scenario)

    return {
        "metadata": _machine_metadata(),
        "generated_at": time.time_ns(),
        "scenarios": [
            run_scenario(scenario, seed + index, warmup_seconds)
            for index, scenario in enumerate(selected)
        ],
    }


def write_report(report: Dict[str, Any], output_path: str) -> str:
    """Write a benchmark report to JSON."""
    directory = os.path.dirname(output_path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    return output_path


def main(argv: Optional[List[str]] = None) -> int:
    """Run benchmarks from the command line."""
    parser = argparse.ArgumentParser(description="Run reproducible tracebook benchmarks.")
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="smoke",
        help="Benchmark scenario to run.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Base random seed.")
    parser.add_argument(
        "--warmup-seconds", type=float, default=0.05, help="Warmup excluded from results."
    )
    parser.add_argument("--duration", type=float, default=None, help="Override scenario duration.")
    parser.add_argument(
        "--throughput", type=float, default=None, help="Override scenario target throughput."
    )
    parser.add_argument("--output", default=None, help="Optional JSON output path.")
    args = parser.parse_args(argv)

    scenario_names = list(SCENARIOS.keys()) if args.scenario == "all" else [args.scenario]
    report = run_benchmarks(
        scenario_names,
        seed=args.seed,
        warmup_seconds=args.warmup_seconds,
        duration_override=args.duration,
        throughput_override=args.throughput,
    )

    if args.output:
        output_path = write_report(report, args.output)
        print(f"Benchmark report written to: {output_path}")
    else:
        print(json.dumps(report, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
