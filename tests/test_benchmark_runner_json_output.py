import json

from tracebook.benchmarks import runner


def test_benchmark_main_writes_valid_json_report_with_event_summary(tmp_path, monkeypatch, capsys):
    calls = []

    def fake_run_benchmark_simulation(**kwargs):
        calls.append(kwargs)
        return {
            "summary_metrics": {
                "total_orders_processed": 7,
                "total_events_processed": 9,
                "total_cancel_events": 1,
                "total_replace_events": 1,
                "total_trades_executed": 3,
                "actual_throughput": 14.0,
            },
            "performance_data": {
                "performance_metrics": {
                    "order_processing_latency_ms": {
                        "count": 7,
                        "mean": 0.20,
                        "median": 0.18,
                        "p95": 0.35,
                        "p99": 0.40,
                        "max": 0.45,
                    },
                    "order_generation_latency_ms": {
                        "count": 2,
                        "mean": 0.10,
                        "p95": 0.15,
                        "p99": 0.16,
                    },
                    "order_event_latency_ms": {
                        "count": 2,
                        "mean": 0.05,
                        "p95": 0.08,
                        "p99": 0.09,
                    },
                },
                "system_resources": {"process_memory_mb": 42.5},
                "collection_overhead": {"total_samples": 11},
            },
        }

    monkeypatch.setattr(runner, "run_benchmark_simulation", fake_run_benchmark_simulation)
    output_path = tmp_path / "nested" / "benchmark.json"

    exit_code = runner.main(
        [
            "--scenario",
            "cancellation_mix",
            "--seed",
            "41",
            "--warmup-seconds",
            "0.25",
            "--duration",
            "0.5",
            "--throughput",
            "12.5",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert calls == [
        {
            "duration": 0.5,
            "throughput": 12.5,
            "algorithm": "FIFO",
            "enable_magic_trace": False,
            "seed": 41,
            "cancel_ratio": 0.15,
            "replace_ratio": 0.05,
            "warmup_seconds": 0.25,
            "symbols": ["BTCUSD"],
        }
    ]
    assert capsys.readouterr().out.strip() == f"Benchmark report written to: {output_path}"

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    scenario = payload["scenarios"][0]

    assert payload["metadata"]["python"]
    assert scenario["name"] == "cancellation_mix"
    assert scenario["config"]["duration_seconds"] == 0.5
    assert scenario["config"]["target_throughput"] == 12.5
    assert scenario["summary"]["orders_processed"] == 7
    assert scenario["summary"]["events_processed"] == 9
    assert scenario["summary"]["cancel_events"] == 1
    assert scenario["summary"]["replace_events"] == 1
    assert scenario["summary"]["event_latency_ms"] == {
        "count": 2,
        "mean": 0.05,
        "p95": 0.08,
        "p99": 0.09,
    }
