import json
import shutil
import sys
import threading
from pathlib import Path

import pytest
import tracebook.conformance.campaign as campaign_module

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
    assert not (destination / ".tracebook-campaign-reservation").exists()

    with pytest.raises(ConformanceError, match="already exists"):
        write_campaign_artifacts(result, destination)
    assert not (destination / ".tracebook-campaign-reservation").exists()


def test_campaign_artifact_writer_rejects_an_active_output_reservation(tmp_path):
    result = run_campaign(
        ReferenceEngineAdapter,
        traces=1,
        events_per_trace=1,
        max_minimize_runs=1,
    )
    destination = tmp_path / "campaign"
    marker = destination / ".tracebook-campaign-reservation"

    with campaign_module._CampaignOutputReservation(destination):
        assert marker.is_file()
        with pytest.raises(ConformanceError, match="already exists"):
            write_campaign_artifacts(result, destination)

    assert marker.is_file()


def test_campaign_artifact_writer_validates_before_reserving_output(tmp_path):
    destination = tmp_path / "campaign"

    with pytest.raises(ConformanceError, match="result must be"):
        write_campaign_artifacts(None, destination)

    assert not destination.exists()


def test_campaign_artifact_writer_serializes_concurrent_writers(tmp_path, monkeypatch):
    result = run_campaign(
        ReferenceEngineAdapter,
        traces=1,
        events_per_trace=1,
        max_minimize_runs=1,
    )
    destination = tmp_path / "campaign"
    entered_write = threading.Event()
    release_write = threading.Event()
    writer_errors = []
    original_write_json = campaign_module._write_json

    def blocking_write(path, payload):
        if path.name == "campaign.json":
            entered_write.set()
            release_write.wait(timeout=5)
        original_write_json(path, payload)

    def first_writer():
        try:
            write_campaign_artifacts(result, destination)
        except Exception as exc:  # pragma: no cover - asserted in the parent thread
            writer_errors.append(exc)

    monkeypatch.setattr(campaign_module, "_write_json", blocking_write)
    writer = threading.Thread(target=first_writer)
    writer.start()
    try:
        assert entered_write.wait(timeout=5)
        with pytest.raises(ConformanceError, match="already exists"):
            write_campaign_artifacts(result, destination)
    finally:
        release_write.set()
        writer.join(timeout=5)

    assert not writer.is_alive()
    assert writer_errors == []
    assert (destination / "campaign.json").is_file()
    assert not (destination / ".tracebook-campaign-reservation").exists()


def test_campaign_writer_never_replaces_a_changed_reservation(tmp_path):
    result = run_campaign(
        ReferenceEngineAdapter,
        traces=1,
        events_per_trace=1,
        max_minimize_runs=1,
    )
    destination = tmp_path / "campaign"

    with campaign_module._CampaignOutputReservation(destination) as reservation:
        shutil.rmtree(destination)
        destination.mkdir()
        with pytest.raises(ConformanceError, match="reservation changed"):
            reservation.write(result)

    assert destination.is_dir()
    assert tuple(destination.iterdir()) == ()


def test_campaign_commit_stays_bound_to_reserved_inode_during_symlink_swap(tmp_path, monkeypatch):
    if not campaign_module._DIRECTORY_FD_SUPPORTED:
        pytest.skip("descriptor-relative directory operations are unavailable")

    result = run_campaign(
        ReferenceEngineAdapter,
        traces=1,
        events_per_trace=1,
        max_minimize_runs=1,
    )
    destination = tmp_path / "campaign"
    displaced = tmp_path / "displaced-reservation"
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    sentinel = unrelated / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    original_unlink = campaign_module._unlink_relative

    def swap_then_unlink(name, directory_fd, directory):
        destination.rename(displaced)
        destination.symlink_to(unrelated, target_is_directory=True)
        original_unlink(name, directory_fd, directory)

    monkeypatch.setattr(campaign_module, "_unlink_relative", swap_then_unlink)

    with campaign_module._CampaignOutputReservation(destination) as reservation:
        with pytest.raises(ConformanceError, match="changed during commit"):
            reservation.write(result)

    assert destination.is_symlink()
    assert tuple(path.name for path in unrelated.iterdir()) == ("keep.txt",)
    assert sentinel.read_text(encoding="utf-8") == "keep"
    assert (displaced / "campaign.json").is_file()
    assert (displaced / ".tracebook-campaign-reservation").is_file()


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


def test_campaign_rejects_engine_identity_changes_before_minimized_artifacts():
    class ChangesAfterCampaignTrace:
        def __init__(self):
            self.calls = 0

        def __call__(self, config):
            self.calls += 1
            adapter = _DroppingFirstAdapter(config)
            version = "campaign" if self.calls == 1 else "minimization"
            adapter.metadata = EngineMetadata("changing-adapter", version, "Python")
            return adapter

    with pytest.raises(ConformanceError, match="metadata changed during minimization"):
        run_campaign(
            ChangesAfterCampaignTrace(),
            traces=1,
            events_per_trace=10,
            max_minimize_runs=10,
        )


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


def test_campaign_cli_rejects_existing_output_before_running(tmp_path, monkeypatch, capsys):
    output = tmp_path / "existing"
    output.mkdir()

    def unexpected_run(*args, **kwargs):
        pytest.fail("campaign ran before output preflight")

    monkeypatch.setattr("tracebook.conformance.cli.run_campaign", unexpected_run)

    exit_code = main(
        [
            "campaign",
            "--output-dir",
            str(output),
            "--candidate",
            "unused-adapter",
        ]
    )

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err


def test_campaign_cli_reserves_output_before_candidate_work(tmp_path, monkeypatch):
    result = run_campaign(
        ReferenceEngineAdapter,
        traces=1,
        events_per_trace=1,
        max_minimize_runs=1,
    )
    output = tmp_path / "campaign"
    run_started = threading.Event()
    release_run = threading.Event()
    run_calls = []
    first_exit_codes = []

    def blocking_run(*args, **kwargs):
        run_calls.append(True)
        run_started.set()
        release_run.wait(timeout=5)
        return result

    def first_cli():
        first_exit_codes.append(
            main(
                [
                    "campaign",
                    "--output-dir",
                    str(output),
                    "--candidate",
                    "unused-adapter",
                ]
            )
        )

    monkeypatch.setattr("tracebook.conformance.cli.run_campaign", blocking_run)
    first = threading.Thread(target=first_cli)
    first.start()
    try:
        assert run_started.wait(timeout=5)
        assert (output / ".tracebook-campaign-reservation").is_file()
        second_exit_code = main(
            [
                "campaign",
                "--output-dir",
                str(output),
                "--candidate",
                "unused-adapter",
            ]
        )
    finally:
        release_run.set()
        first.join(timeout=5)

    assert not first.is_alive()
    assert first_exit_codes == [0]
    assert second_exit_code == 2
    assert run_calls == [True]
    assert (output / "campaign.json").is_file()
    assert not (output / ".tracebook-campaign-reservation").exists()


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
