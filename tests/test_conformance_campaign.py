import json
import sys
from pathlib import Path

import pytest

from tracebook.conformance import (
    CAMPAIGN_GENERATOR_VERSION,
    BookSnapshot,
    BookState,
    CampaignProfile,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
    ReferenceEngineAdapter,
    campaign_profile_names,
    generate_campaign_trace,
    run_campaign,
    write_campaign_artifacts,
)
from tracebook.conformance.cli import main
from tracebook.conformance.model import trace_sha256
from tracebook.core.order import OrderType
from tracebook.events import load_market_events

ROOT = Path(__file__).parents[1]
EXAMPLE_ADAPTER = ROOT / "examples" / "conformance_adapter.py"
FAULTY_ADAPTER = ROOT / "tests" / "fixtures" / "faulty_campaign_adapter.py"


class _DroppingFirstAdapter:
    def __init__(self, config):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata("dropping-first", "1", "Python")
        self._target = None

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if self._target is None and event.op == "new":
            for book in self._inner.snapshot().books:
                if any(order.order_id == event.order_id for order in book.bids + book.asks):
                    self._target = (event.symbol, event.order_id)
                    break
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
        if self._target is None:
            return state
        target_symbol, target_order_id = self._target
        return BookState(
            tuple(
                BookSnapshot(
                    book.symbol,
                    tuple(
                        order
                        for order in book.bids
                        if (book.symbol, order.order_id) != (target_symbol, target_order_id)
                    ),
                    tuple(
                        order
                        for order in book.asks
                        if (book.symbol, order.order_id) != (target_symbol, target_order_id)
                    ),
                )
                for book in state.books
            )
        )

    def close(self):
        self._inner.close()


def test_campaign_generator_is_versioned_deterministic_and_stateful():
    first = generate_campaign_trace("fifo-limit-v1", seed=42, event_count=80)
    second = generate_campaign_trace("fifo-limit-v1", seed=42, event_count=80)

    assert CAMPAIGN_GENERATOR_VERSION == 1
    assert campaign_profile_names() == ("fifo-full-v1", "fifo-limit-v1")
    assert first == second
    assert trace_sha256(first) == (
        "sha256:c8511fc16949aa79a0c343d2157e2f45348aaa560038df9dd2390f18b39a4dae"
    )
    assert {event.symbol for event in first} == {"ALPHA", "BETA"}
    assert {event.op for event in first} >= {"new", "cancel", "reduce", "replace", "clear"}
    assert {event.order_type for event in first if event.op == "new"} == {OrderType.LIMIT}


def test_full_fifo_profile_generates_every_instruction_type():
    events = generate_campaign_trace("fifo-full-v1", seed=7, event_count=200)

    generated_types = {event.order_type for event in events if event.op == "new"}
    assert generated_types == {
        OrderType.LIMIT,
        OrderType.MARKET,
        OrderType.IOC,
        OrderType.FOK,
    }


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"seed": -1}, "seed"),
        ({"event_count": 0}, "event_count"),
        ({"profile": "missing"}, "unknown campaign profile"),
    ],
)
def test_campaign_generator_rejects_invalid_configuration(kwargs, message):
    values = {"profile": "fifo-limit-v1", "seed": 1, "event_count": 10}
    values.update(kwargs)

    with pytest.raises(ConformanceError, match=message):
        generate_campaign_trace(**values)


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"order_types": ()}, "start with LIMIT"),
        ({"order_types": (OrderType.IOC,)}, "start with LIMIT"),
        ({"symbols": ()}, "non-empty tuple"),
        ({"symbols": ("ALPHA", "ALPHA")}, "unique non-empty strings"),
    ],
)
def test_campaign_profile_rejects_unsafe_custom_shapes(kwargs, message):
    values = {
        "name": "custom-v1",
        "description": "A custom profile.",
        "config": ConformanceConfig(),
        "order_types": (OrderType.LIMIT,),
        "symbols": ("ALPHA",),
    }
    values.update(kwargs)

    with pytest.raises(ConformanceError, match=message):
        CampaignProfile(**values)


def test_conformant_campaign_has_stable_identity_and_atomic_artifact(tmp_path):
    result = run_campaign(
        ReferenceEngineAdapter,
        profile="fifo-limit-v1",
        seed=123,
        traces=3,
        events_per_trace=25,
        max_minimize_runs=10,
    )
    destination = tmp_path / "campaign"
    report_path = write_campaign_artifacts(result, destination)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.conformant is True
    assert len(result.traces) == 3
    assert payload["artifact_type"] == "tracebook.conformance.campaign"
    assert payload["campaign_id"] == result.campaign_id
    assert payload["completed_traces"] == 3
    assert payload["generated_events"] == 75
    assert payload["candidate_runs"] == 3
    assert payload["failure"] is None
    assert [trace["seed"] for trace in payload["traces"]] == [trace.seed for trace in result.traces]

    with pytest.raises(ConformanceError, match="already exists"):
        write_campaign_artifacts(result, destination)


def test_campaign_automatically_minimizes_and_persists_first_failure(tmp_path):
    result = run_campaign(
        _DroppingFirstAdapter,
        seed=9,
        traces=5,
        events_per_trace=40,
        max_minimize_runs=20,
    )
    destination = tmp_path / "failed-campaign"
    report_path = write_campaign_artifacts(result, destination)
    payload = json.loads(report_path.read_text(encoding="utf-8"))

    assert result.conformant is False
    assert len(result.traces) == 1
    assert result.failure is not None
    assert len(result.failure.minimization.events) == 1
    assert result.failure.minimization.one_minimal is True
    assert payload["failure"]["target_category"] == "book_state"
    assert payload["failure"]["minimized_event_count"] == 1
    assert len(load_market_events(destination / "failure" / "original.jsonl")) == 40
    assert len(load_market_events(destination / "failure" / "minimized.jsonl")) == 1
    assert (destination / "failure" / "original-report.json").is_file()
    assert (destination / "failure" / "minimization.json").is_file()


def test_campaign_cli_returns_one_and_writes_reproducer_for_divergence(tmp_path, capsys):
    output = tmp_path / "cli-campaign"

    exit_code = main(
        [
            "campaign",
            "--output-dir",
            str(output),
            "--seed",
            "9",
            "--traces",
            "2",
            "--events-per-trace",
            "20",
            "--max-minimize-runs",
            "20",
            "--candidate",
            sys.executable,
            str(FAULTY_ADAPTER),
        ]
    )

    assert exit_code == 1
    assert "First divergence reduced" in capsys.readouterr().out
    assert (output / "campaign.json").is_file()
    assert len(load_market_events(output / "failure" / "minimized.jsonl")) == 1


def test_campaign_cli_runs_external_conformant_adapter(tmp_path):
    output = tmp_path / "external-campaign"

    exit_code = main(
        [
            "campaign",
            "--output-dir",
            str(output),
            "--seed",
            "11",
            "--traces",
            "2",
            "--events-per-trace",
            "10",
            "--candidate",
            sys.executable,
            str(EXAMPLE_ADAPTER),
        ]
    )

    assert exit_code == 0
    assert json.loads((output / "campaign.json").read_text())["conformant"] is True


def test_campaign_is_wired_into_docs_and_ci():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    conformance_docs = (ROOT / "docs" / "conformance.md").read_text(encoding="utf-8")
    ci_workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    external_workflow = (ROOT / ".github" / "workflows" / "python-matching-engine.yml").read_text(
        encoding="utf-8"
    )

    assert "## Differential Campaigns" in readme
    assert "failure/minimized.jsonl" in readme
    assert "Generator version 1 specifies SplitMix64" in conformance_docs
    assert "tracebook-conformance campaign" in ci_workflow
    assert "--profile fifo-limit-v1" in external_workflow
    assert "--events-per-trace 50" in external_workflow
