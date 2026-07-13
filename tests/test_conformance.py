import io
import json
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from tracebook import OrderBook, OrderSide
from tracebook.conformance import (
    AdapterProtocolError,
    BookSnapshot,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    ExternalProcessAdapter,
    Observation,
    Outcome,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    ReferenceEngineAdapter,
    TradeFill,
    canonical_decimal,
    copy_bundled_conformance_suite,
    load_conformance_suite,
    minimize_failing_trace,
    run_conformance,
    run_conformance_suite,
    serve_stdio,
)
from tracebook.conformance.cli import main
from tracebook.events import MarketEvent, load_market_events

ROOT = Path(__file__).parents[1]
SUITE = ROOT / "src/tracebook/conformance/fixtures/v1"
EXAMPLE_ADAPTER = ROOT / "examples/conformance_adapter.py"
FAULTY_ADAPTER = ROOT / "tests/fixtures/faulty_conformance_adapter.py"


def _event(**overrides):
    values = {
        "op": "new",
        "symbol": "TEST",
        "order_id": 1,
        "side": "BUY",
        "price": 100,
        "quantity": 1,
    }
    values.update(overrides)
    return MarketEvent.from_mapping(values)


class _DroppingAdapter:
    """In-process fault that omits one source order from canonical state."""

    def __init__(self, config, target_order_id=77):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata("dropping-adapter", "1", "Python")
        self._target_order_id = target_order_id
        self._fault_active = False

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if event.op == "new" and event.order_id == self._target_order_id:
            self._fault_active = True
        state = self.snapshot()
        return Observation(
            index,
            observation.outcome,
            observation.trades,
            state.digest(),
            state.order_count,
        )

    def snapshot(self):
        state = self._inner.snapshot()
        if not self._fault_active:
            return state
        return BookState(
            tuple(
                BookSnapshot(
                    book.symbol,
                    tuple(order for order in book.bids if order.order_id != self._target_order_id),
                    tuple(order for order in book.asks if order.order_id != self._target_order_id),
                )
                for book in state.books
            )
        )

    def close(self):
        self._inner.close()


class _ObservationFaultAdapter:
    def __init__(self, config, fault):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata(f"{fault}-fault", "1", "Python")
        self._fault = fault

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if index != 2:
            return observation
        if self._fault == "outcome":
            return Observation(
                index,
                Outcome("rejected", "INVALID_ORDER", "fault"),
                observation.trades,
                observation.state_hash,
                observation.resting_order_count,
            )
        return Observation(
            index,
            observation.outcome,
            (),
            observation.state_hash,
            observation.resting_order_count,
        )

    def snapshot(self):
        return self._inner.snapshot()

    def close(self):
        self._inner.close()


def test_resting_order_snapshot_preserves_queue_priority_and_is_detached():
    book = OrderBook("TEST")
    first = book.submit_limit_order(OrderSide.BUY, 100, 1)
    second = book.submit_limit_order(OrderSide.BUY, 100, 1)
    replacement = book.replace_order(first.order.order_id, quantity=1)

    resting = book.get_resting_orders(OrderSide.BUY)

    assert [order.order_id for order in resting] == [
        second.order.order_id,
        replacement.order.order_id,
    ]
    resting[0].remaining_quantity = 999
    assert book.get_order(second.order.order_id).remaining_quantity == pytest.approx(1)
    with pytest.raises(ValueError, match="Unsupported order side"):
        book.get_resting_orders(0)


def test_canonical_decimal_and_config_wire_contract_are_stable():
    assert canonical_decimal(0.30000000000000004, 12) == "0.3"
    assert canonical_decimal("1.2300") == "1.23"
    assert canonical_decimal(-0.0) == "0"
    config = ConformanceConfig.from_dict(
        {
            "matching_algorithm": "PRO_RATA",
            "tick_size": "0.25",
            "self_trade_policy": "CANCEL_INCOMING",
            "quantity_decimal_places": 8,
        }
    )
    assert config.to_dict() == {
        "matching_algorithm": "pro_rata",
        "tick_size": "0.25",
        "self_trade_policy": "CANCEL_INCOMING",
        "quantity_decimal_places": 8,
    }
    with pytest.raises(ConformanceError, match="finite"):
        canonical_decimal(float("nan"))
    with pytest.raises(ConformanceError, match="canonical"):
        TradeFill("TEST", 1, 2, "100.0", "1")


def test_reference_adapter_emits_stable_rejection_codes_and_source_ids():
    adapter = ReferenceEngineAdapter(ConformanceConfig())

    rejected = adapter.apply(
        MarketEvent.from_mapping({"op": "cancel", "symbol": "TEST", "order_id": 999}),
        1,
    )
    resting = adapter.apply(_event(order_id=10), 2)

    assert rejected.outcome.status == "rejected"
    assert rejected.outcome.reason == "ORDER_NOT_ACTIVE"
    assert resting.outcome.status == "applied"
    assert adapter.snapshot().books[0].bids[0].order_id == 10
    adapter.close()


def test_reference_candidate_conforms_and_report_schema_is_serializable():
    events = load_market_events(SUITE / "fifo-lifecycle.jsonl")

    report = run_conformance(events, ReferenceEngineAdapter, trace_name="fifo-lifecycle")
    payload = json.loads(json.dumps(report.to_dict(), allow_nan=False))

    assert report.conformant is True
    assert payload["artifact_type"] == "tracebook.conformance.report"
    assert payload["protocol_version"] == 1
    assert payload["trace"]["event_count"] == len(events)
    assert payload["trace"]["sha256"].startswith("sha256:")
    assert len(payload["final_state_hash"]) == 64
    assert payload["divergence"] is None


def test_semantic_diff_localizes_a_missing_resting_order():
    events = [
        _event(order_id=1, symbol="OTHER"),
        _event(order_id=77, symbol="FAULT", price=1),
        _event(order_id=2, symbol="OTHER", price=99),
    ]

    report = run_conformance(events, _DroppingAdapter)

    assert report.conformant is False
    assert report.compared_events == 2
    assert report.divergence.category == "book_state"
    assert report.divergence.event_index == 2
    assert report.divergence.path.startswith("$.state.books")
    assert report.divergence.event["order_id"] == 77


@pytest.mark.parametrize("fault, category", [("outcome", "outcome"), ("trades", "trades")])
def test_semantic_diff_checks_outcomes_before_trades(fault, category):
    events = [
        _event(order_id=1, side="SELL", quantity=1),
        _event(order_id=2, side="BUY", quantity=1),
    ]

    report = run_conformance(events, lambda config: _ObservationFaultAdapter(config, fault))

    assert report.conformant is False
    assert report.divergence.event_index == 2
    assert report.divergence.category == category


def test_in_process_adapter_must_return_the_current_observation_index():
    class WrongIndexAdapter(ReferenceEngineAdapter):
        def apply(self, event, index):
            observation = super().apply(event, index)
            return Observation(
                index + 1,
                observation.outcome,
                observation.trades,
                observation.state_hash,
                observation.resting_order_count,
            )

    report = run_conformance([_event()], WrongIndexAdapter)

    assert report.conformant is False
    assert report.divergence.category == "protocol"
    assert "does not match" in report.divergence.message


def test_delta_debugging_reduces_to_a_one_event_reproducer():
    events = [
        _event(order_id=1, symbol="OTHER"),
        _event(order_id=2, symbol="OTHER", price=99),
        _event(order_id=77, symbol="FAULT", price=1),
        _event(order_id=3, symbol="OTHER", price=98),
    ]

    result = minimize_failing_trace(events, _DroppingAdapter, max_runs=30)
    payload = result.to_dict()

    assert [event.order_id for event in result.events] == [77]
    assert result.report.conformant is False
    assert result.report.divergence.category == "book_state"
    assert payload["original_event_count"] == 4
    assert payload["minimized_event_count"] == 1
    assert payload["reduction_percent"] == pytest.approx(75.0)
    assert payload["runs"] <= 30
    assert payload["one_minimal"] is True
    assert payload["budget_exhausted"] is False

    budget_limited = minimize_failing_trace(events, _DroppingAdapter, max_runs=1)
    assert budget_limited.one_minimal is False
    assert budget_limited.budget_exhausted is True
    assert budget_limited.report.trace_hash == budget_limited.to_dict()["minimized_trace_sha256"]


def test_external_stdio_adapter_runs_incrementally_and_verifies_final_snapshot():
    events = load_market_events(SUITE / "order-instructions.jsonl")

    report = run_conformance(
        events,
        lambda config: ExternalProcessAdapter(
            [sys.executable, str(EXAMPLE_ADAPTER)], config, timeout_seconds=2
        ),
    )

    assert report.conformant is True
    assert report.candidate_engine.name == "example-python-adapter"
    assert report.compared_events == len(events)


def test_stdio_server_protocol_transcript_is_complete():
    event = _event(order_id=10)
    config = ConformanceConfig()
    messages = [
        {
            "type": "hello",
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "config": config.to_dict(),
        },
        {"type": "event", "index": 1, "event": event.to_dict()},
        {"type": "snapshot", "index": 1},
        {"type": "finish", "event_count": 1},
    ]
    source = io.StringIO("".join(json.dumps(message) + "\n" for message in messages))
    sink = io.StringIO()

    assert serve_stdio(ReferenceEngineAdapter, source, sink) == 0
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]

    assert [response["type"] for response in responses] == [
        "ready",
        "observation",
        "snapshot",
        "complete",
    ]
    assert responses[1]["resting_order_count"] == 1
    assert responses[2]["state"]["books"][0]["bids"][0]["order_id"] == 10


def test_stdio_server_returns_a_protocol_error_frame():
    source = io.StringIO(json.dumps({"type": "event"}) + "\n")
    sink = io.StringIO()

    assert serve_stdio(ReferenceEngineAdapter, source, sink) == 2
    response = json.loads(sink.getvalue())
    assert response["type"] == "error"
    assert response["code"] == "PROTOCOL_ERROR"


def test_external_adapter_reports_invalid_stdout_and_timeout():
    with pytest.raises(AdapterProtocolError, match="not valid JSON"):
        ExternalProcessAdapter(
            [sys.executable, "-c", "print('not-json', flush=True)"],
            ConformanceConfig(),
            timeout_seconds=1,
        )
    with pytest.raises(AdapterProtocolError, match="timed out"):
        ExternalProcessAdapter(
            [sys.executable, "-c", "import time; time.sleep(2)"],
            ConformanceConfig(),
            timeout_seconds=0.05,
        )


def test_external_adapter_rejects_an_invalid_complete_handshake():
    script = """
import json
import sys

hello = json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": "tracebook.conformance",
    "protocol_version": 1,
    "engine": {"name": "bad-close", "version": "1", "language": "Python"},
}), flush=True)
snapshot = json.loads(sys.stdin.readline())
print(json.dumps({"type": "snapshot", "index": 0, "state": {"books": []}}), flush=True)
finish = json.loads(sys.stdin.readline())
print(json.dumps({"type": "complete", "event_count": 999}), flush=True)
"""

    report = run_conformance(
        [],
        lambda config: ExternalProcessAdapter(
            [sys.executable, "-c", script], config, timeout_seconds=1
        ),
    )

    assert report.conformant is False
    assert report.divergence.category == "protocol"
    assert report.divergence.kind == "adapter_close_error"


def test_bundled_suite_is_hash_locked_copyable_and_fully_conformant(tmp_path):
    copied = copy_bundled_conformance_suite(tmp_path / "suite")
    report = run_conformance_suite(copied, ReferenceEngineAdapter)

    assert copied.suite_id == "tracebook-conformance-v1"
    assert copied.suite_hash.startswith("sha256:")
    assert len(copied.cases) == 8
    assert all(path.suffix in {".json", ".jsonl"} for path in copied.root.iterdir())
    assert report["conformant"] is True
    assert report["conformant_cases"] == 8
    assert {tag for case in copied.cases for tag in case.tags} >= {
        "deep-book",
        "cancellation-storm",
        "crossed-input",
        "multi-symbol",
        "pro-rata",
    }

    first_case = copied.cases[0]
    with first_case.events_path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ConformanceError, match="sha256 mismatch"):
        load_conformance_suite(copied.root)

    config_tamper = copy_bundled_conformance_suite(tmp_path / "config-tamper")
    manifest_path = config_tamper.root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["cases"][0]["config"]["tick_size"] = "1"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ConformanceError, match="suite_hash mismatch"):
        load_conformance_suite(config_tamper.root)


def test_cli_sample_suite_and_minimize_workflows(tmp_path, capsys):
    suite_path = tmp_path / "suite"
    suite_report = tmp_path / "suite-report.json"
    assert main(["sample", str(suite_path)]) == 0
    assert (
        main(
            [
                "suite",
                str(suite_path),
                "--output",
                str(suite_report),
                "--candidate",
                sys.executable,
                str(EXAMPLE_ADAPTER),
            ]
        )
        == 0
    )
    assert json.loads(suite_report.read_text(encoding="utf-8"))["conformant"] is True
    case_path = suite_path / "fifo-lifecycle.jsonl"
    case_before = case_path.read_bytes()
    assert (
        main(
            [
                "suite",
                str(suite_path),
                "--output",
                str(case_path),
                "--candidate",
                sys.executable,
                str(EXAMPLE_ADAPTER),
            ]
        )
        == 2
    )
    assert case_path.read_bytes() == case_before

    source = tmp_path / "failing.jsonl"
    source.write_text(
        "\n".join(
            json.dumps(event.to_dict())
            for event in [
                _event(order_id=1, symbol="OTHER"),
                _event(order_id=2, symbol="OTHER", price=99),
                _event(order_id=77, symbol="FAULT", price=1),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    minimized = tmp_path / "minimized.jsonl"
    minimization_report = tmp_path / "minimization.json"
    assert (
        main(
            [
                "minimize",
                str(source),
                "--events-output",
                str(minimized),
                "--output",
                str(minimization_report),
                "--max-runs",
                "30",
                "--candidate",
                sys.executable,
                str(FAULTY_ADAPTER),
            ]
        )
        == 0
    )
    assert [event.order_id for event in load_market_events(minimized)] == [77]
    assert json.loads(minimization_report.read_text(encoding="utf-8"))["minimized_event_count"] == 1
    assert (
        main(
            [
                "minimize",
                str(source),
                "--events-output",
                str(source),
                "--candidate",
                sys.executable,
                str(FAULTY_ADAPTER),
            ]
        )
        == 2
    )
    assert "Cases: 8" in capsys.readouterr().out


@settings(max_examples=25, deadline=None)
@given(
    st.lists(
        st.tuples(
            st.sampled_from(["BUY", "SELL"]),
            st.integers(min_value=9_900, max_value=10_100),
            st.integers(min_value=1, max_value=10),
        ),
        min_size=0,
        max_size=15,
    )
)
def test_generated_prefixes_find_the_exact_first_fault(rows):
    events = [
        _event(
            order_id=index,
            symbol="GENERATED",
            side=side,
            price=price / 100,
            quantity=quantity / 10,
        )
        for index, (side, price, quantity) in enumerate(rows, 1)
    ]
    events.append(_event(order_id=77, symbol="FAULT", price=1))

    report = run_conformance(events, _DroppingAdapter)

    assert report.conformant is False
    assert report.divergence.event_index == len(events)
    assert report.divergence.category == "book_state"
