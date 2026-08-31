import io
import json
import sys
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
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    ExternalBookReplayAdapter,
    ReferenceBookReplayAdapter,
    run_book_replay,
    serve_book_replay_stdio,
)
from tracebook.book_replay.cli import main

ROOT = Path(__file__).parents[1]
PROFILE_TRACE = ROOT / "src" / "tracebook" / "book_replay" / "fixtures" / f"{PROFILE_NAME}.jsonl"
EXAMPLE_ADAPTER = ROOT / "examples" / "book_replay_adapter.py"


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
            timeout_seconds=2,
        ),
    )

    assert report.conformant is True
    assert report.candidate_engine.name == "example-book-replay-adapter"


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
