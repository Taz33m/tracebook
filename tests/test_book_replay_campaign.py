import json
import sys
from pathlib import Path

import pytest

from tracebook.book_replay import (
    BOOK_REPLAY_CAPABILITIES,
    BookReplayError,
    BookReplayEvent,
    BookReplayObservation,
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    ReferenceBookReplayAdapter,
    generate_book_replay_trace,
    measure_book_replay_coverage,
    minimize_book_replay_failure,
    run_book_replay_campaign,
)
from tracebook.book_replay.cli import main
from tracebook.book_replay.model import trace_sha256

ROOT = Path(__file__).parents[1]
FAULTY_ADAPTER = ROOT / "tests" / "fixtures" / "faulty_book_replay_adapter.py"


class _ReorderingAdapter:
    def __init__(self, config):
        self._inner = ReferenceBookReplayAdapter(config)
        self._reorder = False
        self.metadata = EngineMetadata("reordering-test-adapter", "1", "Python")

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


def test_generator_is_deterministic_and_scaffold_covers_every_capability():
    first = generate_book_replay_trace(123, 40)
    repeated = generate_book_replay_trace(123, 40)
    different = generate_book_replay_trace(124, 40)

    assert first == repeated
    assert first != different
    assert len(first) == 40
    assert trace_sha256(first) == (
        "4199853cf0496a08beb964283932a6880bd23e534a5e1f72952a4d2d616d0fad"
    )
    assert trace_sha256(first[:18]) == (
        "25180672acc45a7487cf3f87abcdf9db5426bdefa82a0d2a00efb259581e41ee"
    )
    coverage = measure_book_replay_coverage([first[:18]])
    assert coverage.complete is True
    assert coverage.covered == BOOK_REPLAY_CAPABILITIES


def test_reference_campaign_is_reproducible_and_complete():
    result = run_book_replay_campaign(
        ReferenceBookReplayAdapter,
        seed=20260831,
        traces=3,
        events_per_trace=60,
    )

    assert result.conformant is True
    assert result.campaign_id == (
        "sha256:ae989f8ddd45385471c3e6b4b7736ec89410947bd028aadfc1c46f9e294aa876"
    )
    assert result.coverage.complete is True
    assert result.failure_id is None
    assert [trace.seed for trace in result.traces] == [
        10864503699890266417,
        10644828502866079881,
        3445208458078199137,
    ]


def test_minimizer_produces_one_minimal_priority_failure():
    events = generate_book_replay_trace(123, 40)

    result = minimize_book_replay_failure(events, _ReorderingAdapter)

    assert [event.to_dict() for event in result.events] == [
        {
            "op": "add",
            "symbol": "TEST",
            "order_id": 1,
            "side": "BUY",
            "price": "100",
            "quantity": "2",
        },
        {
            "op": "add",
            "symbol": "TEST",
            "order_id": 2,
            "side": "BUY",
            "price": "100",
            "quantity": "3",
        },
        {
            "op": "update",
            "symbol": "TEST",
            "order_id": 1,
            "side": "BUY",
            "price": "100",
            "quantity": "4",
        },
    ]
    assert result.target_category == "book_state"
    assert result.one_minimal is True
    assert result.budget_exhausted is False
    assert result.report.divergence is not None
    assert result.report.divergence.event_index == 3


def test_minimizer_reports_an_exhausted_budget_without_claiming_minimality():
    result = minimize_book_replay_failure(
        generate_book_replay_trace(123, 18),
        _ReorderingAdapter,
        max_runs=1,
    )

    assert result.runs == 1
    assert result.one_minimal is False
    assert result.budget_exhausted is True
    assert len(result.events) == 3


def test_faulty_campaign_stops_and_minimizes_first_divergence():
    result = run_book_replay_campaign(
        _ReorderingAdapter,
        seed=20260831,
        traces=10,
        events_per_trace=100,
    )

    assert result.conformant is False
    assert result.failure is not None
    assert len(result.traces) == 1
    assert result.traces[0].report.compared_events == 3
    assert len(result.failure.minimization.events) == 3
    assert result.failure.minimization.one_minimal is True
    assert result.failure_id == "failure-dfe5c23848b63211b655"
    assert trace_sha256(result.failure.minimization.events) == (
        "d9eeb771c7c469c0762c67be7dc76a63a09f36d46c1dd2cda1dbb3cbb47c0b29"
    )
    assert result.coverage.covered == (
        "l3-add",
        "same-level-fifo",
        "same-level-upsize",
    )


def test_cli_campaign_writes_report_and_reduced_trace(tmp_path):
    report_path = tmp_path / "campaign.json"
    reduced_path = tmp_path / "reduced.jsonl"

    exit_code = main(
        [
            "campaign",
            "--output",
            str(report_path),
            "--reduced-events-output",
            str(reduced_path),
            "--seed",
            "20260831",
            "--traces",
            "10",
            "--events-per-trace",
            "100",
            "--candidate-cmd",
            f"{sys.executable} {FAULTY_ADAPTER}",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    reduced = [
        BookReplayEvent.from_mapping(json.loads(line))
        for line in reduced_path.read_text(encoding="utf-8").splitlines()
    ]
    assert exit_code == 1
    assert report["artifact_type"] == "tracebook.book-replay.campaign"
    assert report["conformant"] is False
    assert report["failure"]["minimized_event_count"] == 3
    assert len(reduced) == 3


def test_cli_minimize_writes_a_semantic_failure_artifact(tmp_path):
    trace_path = tmp_path / "trace.jsonl"
    report_path = tmp_path / "minimization.json"
    reduced_path = tmp_path / "reduced.jsonl"
    trace_path.write_text(
        "".join(
            json.dumps(event.to_dict(), sort_keys=True) + "\n"
            for event in generate_book_replay_trace(123, 18)
        ),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "minimize",
            str(trace_path),
            "--events-output",
            str(reduced_path),
            "--output",
            str(report_path),
            "--candidate-cmd",
            f"{sys.executable} {FAULTY_ADAPTER}",
        ]
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 1
    assert report["artifact_type"] == "tracebook.book-replay.minimization"
    assert report["target_category"] == "book_state"
    assert report["minimized_event_count"] == 3
    assert report["one_minimal"] is True
    assert len(reduced_path.read_text(encoding="utf-8").splitlines()) == 3


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("seed", -1, "seed must be an integer"),
        ("traces", 0, "traces must be a positive integer"),
        ("events_per_trace", 0, "events_per_trace must be a positive integer"),
        ("max_minimize_runs", 0, "max_minimize_runs must be a positive integer"),
    ],
)
def test_campaign_rejects_invalid_bounds(keyword, value, message):
    arguments = {keyword: value}
    with pytest.raises(BookReplayError, match=message):
        run_book_replay_campaign(ReferenceBookReplayAdapter, **arguments)
