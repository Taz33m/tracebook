import io
import json
import sys
from decimal import Decimal, localcontext
from pathlib import Path

import pytest

from tracebook.book_replay import (
    PROFILE_NAME,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookReplayConfig,
    BookReplayError,
    BookReplayEvent,
    BookReplayObservation,
    BookReplayProtocolError,
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    ExternalBookReplayAdapter,
    ReferenceBookReplayAdapter,
    run_book_replay,
    serve_book_replay_stdio,
)
from tracebook.book_replay.cli import main
from tracebook.book_replay.model import Outcome, SimulatedFill

ROOT = Path(__file__).parents[1]
PROFILE_TRACE = ROOT / "src" / "tracebook" / "book_replay" / "fixtures" / f"{PROFILE_NAME}.jsonl"
EXAMPLE_ADAPTER = ROOT / "examples" / "book_replay_adapter.py"


def _serve_test(factory, *messages, hello_overrides=None):
    hello = dict(type="hello", protocol=PROTOCOL_NAME, protocol_version=PROTOCOL_VERSION)
    hello.update(hello_overrides or {})
    source = io.StringIO("".join(json.dumps(item) + "\n" for item in (hello, *messages)))
    sink = io.StringIO()
    status = serve_book_replay_stdio(factory, source, sink)
    return status, [json.loads(line) for line in sink.getvalue().splitlines()]


def test_server_close_failure_is_one_terminal_adapter_error():
    instances = []

    class BrokenClose(ReferenceBookReplayAdapter):
        def __init__(self, config):
            super().__init__(config)
            self.close_calls = 0
            instances.append(self)

        def close(self):
            self.close_calls += 1
            raise ValueError("shutdown failed")

    status, messages = _serve_test(BrokenClose, dict(type="finish", event_count=0))
    assert status == 2
    assert [item["type"] for item in messages] == ["ready", "error"]
    assert messages[-1]["code"] == "ADAPTER_ERROR"
    assert instances[0].close_calls == 1


@pytest.mark.parametrize("stage", ["factory", "apply", "snapshot"])
@pytest.mark.parametrize("exception", [TypeError, ValueError])
def test_server_adapter_exceptions_are_not_client_errors(stage, exception):
    closed = []

    class BrokenAdapter(ReferenceBookReplayAdapter):
        def __init__(self, config):
            if stage == "factory":
                raise exception("adapter bug")
            super().__init__(config)

        def apply(self, event, index):
            if stage == "apply":
                raise exception("adapter bug")
            return super().apply(event, index)

        def snapshot(self):
            raise exception("adapter bug")

        def close(self):
            closed.append(True)

    request = (
        dict(type="event", index=1, event=_event().to_dict())
        if stage == "apply"
        else dict(type="snapshot", index=0)
    )
    status, messages = _serve_test(BrokenAdapter, request)
    assert status == 2
    assert messages[-1]["code"] == "ADAPTER_ERROR"
    assert len(closed) == (0 if stage == "factory" else 1)


@pytest.mark.parametrize("kind,field", [("snapshot", "index"), ("finish", "event_count")])
@pytest.mark.parametrize("value", [None, False, True, 0.0, "0", -1, "missing"])
def test_server_requires_integer_session_counters(kind, field, value):
    request = {"type": kind}
    if value != "missing":
        request[field] = value
    status, messages = _serve_test(ReferenceBookReplayAdapter, request)
    assert status == 2
    assert [item["type"] for item in messages] == ["ready", "error"]
    assert messages[-1]["code"] == "PROTOCOL_ERROR"


def test_server_invalid_payload_is_a_client_error():
    status, messages = _serve_test(
        ReferenceBookReplayAdapter, dict(type="event", index=1, event=None)
    )
    assert status == 2
    assert messages[-1]["code"] == "PROTOCOL_ERROR"
    status, messages = _serve_test(ReferenceBookReplayAdapter, hello_overrides={"config": None})
    assert status == 2
    assert messages[0]["code"] == "PROTOCOL_ERROR"


@pytest.mark.parametrize("field,value", [("outcome", {}), ("fills", (object(),)), ("fills", None)])
def test_invalid_in_process_observation_is_classified_as_protocol_failure(field, value):
    class Malformed(ReferenceBookReplayAdapter):
        def apply(self, event, index):
            fields = dict(
                index=index,
                outcome=Outcome("applied"),
                fills=(),
                state_hash=self.snapshot().digest(),
                order_count=0,
            )
            fields[field] = value
            return BookReplayObservation(**fields)

    report = run_book_replay([_event()], Malformed)
    assert report.operational_failure
    assert report.divergence.category == "protocol"
    assert report.divergence.kind == "adapter_error"


@pytest.mark.parametrize("precision", [3, 28, 60])
def test_probe_subtraction_and_canonical_values_are_exact_at_any_context_precision(precision):
    tiny = "0.0000000000000000000000000000000000000001"
    leftover = "0." + "9" * 40
    with localcontext() as context:
        context.prec = precision
        adapter = ReferenceBookReplayAdapter(BookReplayConfig())
        adapter.apply(_event(order_id=1, quantity=tiny), 1)
        adapter.apply(_event(order_id=2, quantity="1"), 2)
        before = adapter.snapshot()
        observation = adapter.apply(_event("probe", quantity="1"), 3)
        assert observation.fills == (SimulatedFill("100", tiny), SimulatedFill("100", leftover))
        assert adapter.snapshot() == before
    with localcontext() as context:
        context.prec = 100
        assert sum(Decimal(fill.quantity) for fill in observation.fills) == Decimal(1)


def test_external_timeout_overflow_is_a_public_validation_error():
    with pytest.raises(BookReplayError, match="positive finite"):
        ExternalBookReplayAdapter(["never-started"], BookReplayConfig(), timeout_seconds=10**400)


def test_external_invalid_metadata_is_a_protocol_error_and_child_is_reaped(monkeypatch):
    import subprocess

    processes = []
    popen = subprocess.Popen

    def capture_process(*args, **kwargs):
        process = popen(*args, **kwargs)
        processes.append(process)
        return process

    monkeypatch.setattr(subprocess, "Popen", capture_process)
    script = (
        "import json,sys; hello=json.loads(sys.stdin.readline()); "
        "hello.update(type='ready',engine={}); print(json.dumps(hello),flush=True); "
        "sys.stdin.read()"
    )
    with pytest.raises(BookReplayProtocolError, match="invalid ready engine metadata"):
        ExternalBookReplayAdapter([sys.executable, "-c", script], BookReplayConfig())
    assert len(processes) == 1
    assert processes[0].poll() is not None


def _event(op="add", **overrides):
    values = {
        "op": op,
        "symbol": "TEST",
        "order_id": 1,
        "side": "BUY",
        "price": "100",
        "quantity": "2",
    }
    if op == "delete":
        values = {"op": op, "symbol": "TEST", "order_id": 1}
    elif op == "probe":
        values = {
            "op": op,
            "symbol": "TEST",
            "side": "SELL",
            "price": "100",
            "quantity": "2",
        }
    values.update(overrides)
    return BookReplayEvent.from_mapping(values)


def test_event_contract_is_canonical_and_strict():
    event = _event(price="100.00", quantity=2.500)

    assert event.to_dict() == {
        "op": "add",
        "symbol": "TEST",
        "order_id": 1,
        "side": "BUY",
        "price": "100",
        "quantity": "2.5",
    }
    with pytest.raises(BookReplayError, match="unsupported fields"):
        BookReplayEvent.from_mapping({**event.to_dict(), "owner": 9})
    with pytest.raises(BookReplayError, match="contain only"):
        BookReplayEvent("delete", "TEST", 1, "BUY")
    with pytest.raises(BookReplayError, match="must be positive"):
        _event(quantity="0")


def test_same_level_size_increase_retains_fifo_and_probe_does_not_mutate():
    adapter = ReferenceBookReplayAdapter(BookReplayConfig())
    adapter.apply(_event(order_id=1, quantity="2"), 1)
    adapter.apply(_event(order_id=2, quantity="3"), 2)
    updated = adapter.apply(_event("update", order_id=1, quantity="4"), 3)
    before_probe = adapter.snapshot()

    observation = adapter.apply(_event("probe", quantity="6"), 4)

    assert [order.order_id for order in before_probe.books[0].bids] == [1, 2]
    assert [fill.to_dict() for fill in observation.fills] == [
        {"price": "100", "quantity": "4"},
        {"price": "100", "quantity": "2"},
    ]
    assert adapter.snapshot() == before_probe
    assert observation.state_hash == updated.state_hash


def test_price_change_requeues_and_delete_is_keyed_by_order_id():
    adapter = ReferenceBookReplayAdapter(BookReplayConfig())
    adapter.apply(_event(order_id=1, price="99"), 1)
    adapter.apply(_event(order_id=2, price="99"), 2)
    adapter.apply(_event("update", order_id=1, price="100", quantity="2"), 3)
    adapter.apply(_event("update", order_id=1, price="99", quantity="2"), 4)

    assert [order.order_id for order in adapter.snapshot().books[0].bids] == [2, 1]
    deleted = adapter.apply(_event("delete", order_id=2), 5)
    rejected = adapter.apply(_event("delete", order_id=999), 6)
    assert deleted.outcome.status == "applied"
    assert rejected.outcome.to_dict()["reason"] == "ORDER_NOT_ACTIVE"


def test_probe_obeys_limit_price_and_price_time_order():
    adapter = ReferenceBookReplayAdapter(BookReplayConfig())
    events = [
        _event(order_id=10, side="SELL", price="102", quantity="5"),
        _event(order_id=11, side="SELL", price="101", quantity="1"),
        _event(order_id=12, side="SELL", price="101", quantity="2"),
    ]
    for index, event in enumerate(events, 1):
        adapter.apply(event, index)

    observation = adapter.apply(
        _event("probe", side="BUY", price="101", quantity="5"),
        4,
    )

    assert [fill.to_dict() for fill in observation.fills] == [
        {"price": "101", "quantity": "1"},
        {"price": "101", "quantity": "2"},
    ]
    assert observation.order_count == 3


class _ReorderingAdapter:
    def __init__(self, config):
        self._inner = ReferenceBookReplayAdapter(config)
        self.metadata = EngineMetadata("reordering-test-adapter", "1", "Python")
        self._reorder = False

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if event.op == "update" and event.order_id == 1:
            self._reorder = True
        state = self.snapshot()
        return BookReplayObservation(
            index,
            observation.outcome,
            observation.fills,
            state.digest(),
            state.order_count,
        )

    def snapshot(self):
        state = self._inner.snapshot()
        if not self._reorder:
            return state
        books = []
        for book in state.books:
            bids = list(book.bids)
            if book.symbol == "TEST" and len(bids) >= 2:
                bids[0], bids[1] = bids[1], bids[0]
            books.append(BookReplaySnapshot(book.symbol, tuple(bids), book.asks))
        return BookReplayState(tuple(books))

    def close(self):
        self._inner.close()


def test_report_localizes_same_level_requeue_as_book_state_divergence():
    events = [
        _event(order_id=1),
        _event(order_id=2),
        _event("update", order_id=1, quantity="4"),
    ]

    report = run_book_replay(events, _ReorderingAdapter, trace_name="priority")

    assert report.conformant is False
    assert report.compared_events == 3
    assert report.divergence is not None
    assert report.divergence.category == "book_state"
    assert report.divergence.path.endswith("order_id")
    assert report.to_dict()["artifact_type"] == "tracebook.book-replay.report"
    assert report.to_dict()["config"] == {"profile": PROFILE_NAME}


def test_reference_candidate_conforms_on_bundled_profile():
    events = [
        BookReplayEvent.from_mapping(json.loads(line))
        for line in PROFILE_TRACE.read_text(encoding="utf-8").splitlines()
    ]

    report = run_book_replay(events, ReferenceBookReplayAdapter, trace_name=PROFILE_NAME)

    assert report.conformant is True
    assert report.event_count == 17
    assert report.compared_events == 17
    assert (
        report.final_state_hash
        == "9e0af6be935dce940a87497788c7a9a799c71f05e34a0204c9d294fce611b002"
    )


def test_stdio_transcript_uses_separate_protocol_identity():
    event = _event()
    messages = [
        {
            "type": "hello",
            "protocol": PROTOCOL_NAME,
            "protocol_version": PROTOCOL_VERSION,
            "config": BookReplayConfig().to_dict(),
        },
        {"type": "event", "index": 1, "event": event.to_dict()},
        {"type": "snapshot", "index": 1},
        {"type": "finish", "event_count": 1},
    ]
    source = io.StringIO("".join(json.dumps(message) + "\n" for message in messages))
    sink = io.StringIO()

    assert serve_book_replay_stdio(ReferenceBookReplayAdapter, source, sink) == 0
    responses = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [response["type"] for response in responses] == [
        "ready",
        "observation",
        "snapshot",
        "complete",
    ]
    assert responses[0]["protocol"] == "tracebook.book-replay"
    assert responses[2]["state"]["books"][0]["bids"][0]["order_id"] == 1


def test_external_example_adapter_conforms():
    events = [_event(order_id=1), _event("probe", quantity="1")]

    report = run_book_replay(
        events,
        lambda config: ExternalBookReplayAdapter(
            [sys.executable, str(EXAMPLE_ADAPTER)],
            config,
            timeout_seconds=10,
        ),
    )

    assert report.conformant is True
    assert report.candidate_engine.name == "example-book-replay-adapter"


def _shutdown_test_adapter(delay, exit_code, *, handle_terminate=False):
    script = """
import json
import signal
import sys
import time

delay, exit_code, handle_terminate = json.loads(sys.argv[1])
if handle_terminate:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
hello = json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": hello["protocol"],
    "protocol_version": hello["protocol_version"],
    "engine": {"name": "shutdown-test", "version": "1", "language": "Python"},
}), flush=True)
finish = json.loads(sys.stdin.readline())
print(json.dumps({"type": "complete", "event_count": finish["event_count"]}), flush=True)
time.sleep(delay)
sys.exit(exit_code)
"""
    return ExternalBookReplayAdapter(
        [
            sys.executable,
            "-I",
            "-u",
            "-c",
            script,
            json.dumps([delay, exit_code, handle_terminate]),
        ],
        BookReplayConfig(),
        timeout_seconds=10,
    )


def test_external_close_allows_configured_time_for_clean_exit():
    adapter = _shutdown_test_adapter(0.75, 0)

    adapter.close()
    adapter.close()

    assert adapter._process.returncode == 0


def test_external_close_rejects_nonzero_exit_after_complete():
    adapter = _shutdown_test_adapter(0, 7)

    with pytest.raises(BookReplayProtocolError, match="exited with code 7 after complete"):
        adapter.close()


def test_external_close_rejects_timeout_even_when_termination_exits_zero():
    adapter = _shutdown_test_adapter(60, 0, handle_terminate=True)
    # Startup has its own generous budget; now isolate the exit timeout.
    adapter.timeout_seconds = 0.2

    with pytest.raises(BookReplayProtocolError, match="timed out exiting after complete"):
        adapter.close()

    assert adapter._process.returncode is not None


def test_cli_sample_and_external_run(tmp_path, capsys):
    sample_dir = tmp_path / "sample"
    assert main(["sample", str(sample_dir)]) == 0
    copied = sample_dir / f"{PROFILE_NAME}.jsonl"
    assert copied.read_bytes() == PROFILE_TRACE.read_bytes()

    output = tmp_path / "report.json"
    assert (
        main(
            [
                "run",
                str(copied),
                "--output",
                str(output),
                "--candidate-cmd",
                f"{sys.executable} {EXAMPLE_ADAPTER}",
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["conformant"] is True
    assert payload["artifact_type"] == "tracebook.book-replay.report"
    assert "Book-replay trace copied" in capsys.readouterr().out


def test_cli_rejects_overwriting_input(tmp_path, capsys):
    trace = tmp_path / "trace.jsonl"
    trace.write_text(json.dumps(_event().to_dict()) + "\n", encoding="utf-8")

    exit_code = main(
        [
            "run",
            str(trace),
            "--output",
            str(trace),
            "--candidate-cmd",
            f"{sys.executable} {EXAMPLE_ADAPTER}",
        ]
    )

    assert exit_code == 2
    assert "input and output paths must be distinct" in capsys.readouterr().err
