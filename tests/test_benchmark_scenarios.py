"""Tests for the benchmark scenario catalog."""

from tracebook.benchmarks import runner


def test_new_scenarios_are_registered():
    for name in ("deep_book", "high_cancellation", "pro_rata_cancellation", "multi_symbol"):
        assert name in runner.SCENARIOS, f"missing scenario {name}"


def test_high_cancellation_has_a_heavier_lifecycle_mix_than_cancellation_mix():
    heavy = runner.SCENARIOS["high_cancellation"]
    base = runner.SCENARIOS["cancellation_mix"]
    assert heavy.cancel_ratio > base.cancel_ratio
    assert heavy.replace_ratio > base.replace_ratio


def test_pro_rata_cancellation_uses_the_pro_rata_algorithm_with_events():
    scenario = runner.SCENARIOS["pro_rata_cancellation"]
    assert scenario.matching_algorithm == "PRO_RATA"
    assert scenario.cancel_ratio > 0.0


def test_multi_symbol_scenario_runs_independent_books_per_symbol():
    report = runner.run_benchmarks(
        ["multi_symbol"],
        seed=1,
        warmup_seconds=0.0,
        duration_override=0.15,
        throughput_override=40.0,
    )
    scenario = report["scenarios"][0]
    stats = scenario["raw_result"]["order_book_statistics"]

    assert set(stats) == {"BTCUSD", "ETHUSD", "SOLUSD"}
    assert list(scenario["config"]["symbols"]) == ["BTCUSD", "ETHUSD", "SOLUSD"]


def test_single_symbol_scenarios_default_to_one_book():
    report = runner.run_benchmarks(
        ["deep_book"],
        seed=1,
        warmup_seconds=0.0,
        duration_override=0.15,
        throughput_override=40.0,
    )
    stats = report["scenarios"][0]["raw_result"]["order_book_statistics"]
    assert set(stats) == {"BTCUSD"}
