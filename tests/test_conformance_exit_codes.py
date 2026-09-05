import json
import sys
from xml.etree import ElementTree

import pytest

import tracebook.conformance.cli as conformance_cli
from tracebook.conformance import (
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
    Outcome,
    ReferenceEngineAdapter,
    copy_bundled_conformance_suite,
    minimize_failing_trace,
    run_conformance,
    run_reproduction,
)
from tracebook.conformance.cli import main
from tracebook.conformance.exit_codes import exit_code_for_artifact
from tracebook.conformance.junit import render_junit
from tracebook.events import MarketEvent


def _event(order_id=1) -> MarketEvent:
    return MarketEvent.from_mapping(
        {
            "op": "new",
            "symbol": "TEST",
            "order_id": order_id,
            "side": "BUY",
            "price": 100,
            "quantity": 1,
        }
    )


def _write_trace(path, events=None) -> None:
    selected = [_event()] if events is None else events
    path.write_text(
        "".join(json.dumps(event.to_dict()) + "\n" for event in selected),
        encoding="utf-8",
    )


class _WrappedReferenceAdapter:
    def __init__(self, config):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata("exit-code-test-adapter", "1", "Python")

    def apply(self, event, index):
        return self._inner.apply(event, index)

    def snapshot(self):
        return self._inner.snapshot()

    def close(self):
        self._inner.close()


class _SemanticDivergenceAdapter(_WrappedReferenceAdapter):
    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        return Observation(
            index,
            Outcome("rejected", "INJECTED_FAULT"),
            observation.trades,
            observation.state_hash,
            observation.resting_order_count,
        )


class _LateSemanticDivergenceAdapter(_WrappedReferenceAdapter):
    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if index < 2:
            return observation
        return Observation(
            index,
            Outcome("rejected", "INJECTED_FAULT"),
            observation.trades,
            observation.state_hash,
            observation.resting_order_count,
        )


class _ApplyFailureAdapter(_WrappedReferenceAdapter):
    def apply(self, event, index):
        raise RuntimeError("apply failed")


class _SnapshotFailureAdapter(_WrappedReferenceAdapter):
    def snapshot(self):
        raise RuntimeError("snapshot failed")


class _InvalidStateHashAdapter(_WrappedReferenceAdapter):
    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        return Observation(
            index,
            observation.outcome,
            observation.trades,
            "0" * 64,
            observation.resting_order_count,
        )


class _CloseFailureAdapter(_WrappedReferenceAdapter):
    def close(self):
        self._inner.close()
        raise RuntimeError("close failed")


class _SemanticAndCloseFailureAdapter(_SemanticDivergenceAdapter):
    def close(self):
        self._inner.close()
        raise RuntimeError("close failed after semantic divergence")


class _ApplyAndCloseFailureAdapter(_ApplyFailureAdapter):
    def close(self):
        self._inner.close()
        raise RuntimeError("close failed after apply failure")


class _SequencedAdapterFactory:
    def __init__(self, *adapter_types):
        self._adapter_types = iter(adapter_types)

    def __call__(self, config):
        try:
            adapter_type = next(self._adapter_types)
        except StopIteration as exc:
            raise AssertionError("candidate factory received an unexpected extra call") from exc
        return adapter_type(config)


@pytest.mark.parametrize(
    ("adapter", "expected_kind"),
    [
        (_ApplyFailureAdapter, "adapter_error"),
        (_SnapshotFailureAdapter, "adapter_error"),
        (_InvalidStateHashAdapter, "invalid_state_hash"),
        (_CloseFailureAdapter, "adapter_close_error"),
    ],
)
def test_report_exit_classification_treats_adapter_failures_as_operational(adapter, expected_kind):
    report = run_conformance([_event()], adapter)

    assert report.divergence is not None
    assert report.divergence.category == "protocol"
    assert report.divergence.kind == expected_kind
    assert report.operational_failure is True
    assert exit_code_for_artifact(report.to_dict()) == 2


def test_report_exit_classification_keeps_semantic_divergence_at_one():
    report = run_conformance([_event()], _SemanticDivergenceAdapter)

    assert report.divergence is not None
    assert report.divergence.category == "outcome"
    assert report.operational_failure is False
    assert exit_code_for_artifact(report.to_dict()) == 1


def test_close_failure_preserves_first_semantic_divergence_and_exit_two():
    report = run_conformance([_event()], _SemanticAndCloseFailureAdapter)

    assert report.divergence is not None
    assert report.divergence.category == "outcome"
    assert report.divergence.kind == "type_mismatch"
    assert report.divergence.close_error == (
        "candidate close failed: close failed after semantic divergence"
    )
    assert report.operational_failure is True
    assert exit_code_for_artifact(report.to_dict()) == 2


def test_close_failure_preserves_first_protocol_divergence():
    report = run_conformance([_event()], _ApplyAndCloseFailureAdapter)

    assert report.divergence is not None
    assert report.divergence.category == "protocol"
    assert report.divergence.kind == "adapter_error"
    assert report.divergence.message == "apply failed"
    assert report.divergence.close_error == (
        "candidate close failed: close failed after apply failure"
    )
    assert report.operational_failure is True
    assert exit_code_for_artifact(report.to_dict()) == 2


def test_junit_marks_protocol_minimization_as_failed():
    result = minimize_failing_trace(
        [_event()],
        _SnapshotFailureAdapter,
        max_runs=1,
    )
    payload = result.to_dict()

    junit = ElementTree.fromstring(render_junit(payload))
    failure = junit.find("./testcase/failure")
    assert exit_code_for_artifact(payload) == 2
    assert junit.attrib["failures"] == "1"
    assert failure is not None
    assert failure.attrib["type"] == "protocol"


def test_junit_marks_protocol_reproduction_as_failed_even_without_metadata():
    result = run_reproduction(
        [_event()],
        _SnapshotFailureAdapter,
        ConformanceConfig(),
    )
    payload = result.to_dict()

    junit = ElementTree.fromstring(render_junit(payload))
    failure = junit.find("./testcase/failure")
    assert payload["reproduced"] is False
    assert exit_code_for_artifact(payload) == 2
    assert junit.attrib["failures"] == "1"
    assert failure is not None
    assert failure.attrib["type"] == "protocol"


def test_cli_reproduction_reports_protocol_failure_as_mismatch(tmp_path, monkeypatch, capsys):
    events = tmp_path / "events.jsonl"
    report_path = tmp_path / "reproduction.json"
    junit_path = tmp_path / "reproduction.xml"
    _write_trace(events)
    monkeypatch.setattr(
        conformance_cli,
        "_candidate_factory",
        lambda _args: _SnapshotFailureAdapter,
    )

    exit_code = main(
        [
            "reproduce",
            str(events),
            "--output",
            str(report_path),
            "--junit-output",
            str(junit_path),
            "--candidate",
            "unused",
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    junit = ElementTree.parse(junit_path).getroot()
    stdout = capsys.readouterr().out
    assert exit_code == 2
    assert payload["reproduced"] is False
    assert payload["conformance_report"]["divergence"]["category"] == "protocol"
    assert junit.attrib["failures"] == "1"
    assert "Reproduction: mismatch" in stdout
    assert "Reproduction: exact match" not in stdout


def test_cli_returns_two_for_malformed_mid_run_stdout(tmp_path):
    events = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.json"
    adapter = tmp_path / "malformed_adapter.py"
    _write_trace(events)
    adapter.write_text(
        """
import json
import sys

json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": "tracebook.conformance",
    "protocol_version": 1,
    "engine": {"name": "malformed", "version": "1", "language": "Python"},
}), flush=True)
json.loads(sys.stdin.readline())
print("not-json", flush=True)
""".lstrip(),
        encoding="utf-8",
    )

    exit_code = main(
        [
            "run",
            str(events),
            "--output",
            str(report_path),
            "--candidate",
            sys.executable,
            str(adapter),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["divergence"]["category"] == "protocol"
    assert "not valid JSON" in payload["divergence"]["message"]


def test_cli_returns_two_for_mid_run_timeout(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    report_path = tmp_path / "report.json"
    adapter = tmp_path / "timeout_adapter.py"
    _write_trace(events)
    adapter.write_text(
        """
import json
import sys
import time

json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": "tracebook.conformance",
    "protocol_version": 1,
    "engine": {"name": "timeout", "version": "1", "language": "Python"},
}), flush=True)
json.loads(sys.stdin.readline())
time.sleep(2)
""".lstrip(),
        encoding="utf-8",
    )

    class MidRunTimeoutFactory(conformance_cli.ExternalProcessAdapterFactory):
        def __call__(self, config):
            candidate = super().__call__(config)
            # Exercise the response timeout after ready, not interpreter startup
            # on a busy host. Startup failures intentionally have no run report.
            candidate.timeout_seconds = 0.2
            return candidate

    monkeypatch.setattr(conformance_cli, "ExternalProcessAdapterFactory", MidRunTimeoutFactory)

    exit_code = main(
        [
            "run",
            str(events),
            "--output",
            str(report_path),
            "--timeout",
            "10",
            "--candidate",
            sys.executable,
            str(adapter),
        ]
    )

    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert exit_code == 2
    assert payload["divergence"]["category"] == "protocol"
    assert "timed out waiting for 'observation'" in payload["divergence"]["message"]


@pytest.mark.parametrize(
    "command",
    ["run", "suite", "minimize", "campaign", "qualify", "reproduce"],
)
def test_cli_uses_protocol_exit_classification_for_every_result_shape(
    command, tmp_path, monkeypatch
):
    events = tmp_path / "events.jsonl"
    _write_trace(events)
    monkeypatch.setattr(
        conformance_cli,
        "_candidate_factory",
        lambda _args: _SnapshotFailureAdapter,
    )

    if command == "run":
        arguments = ["run", str(events), "--candidate", "unused"]
    elif command == "suite":
        suite = copy_bundled_conformance_suite(tmp_path / "suite", suite_version="v1")
        arguments = [
            "suite",
            str(suite.root),
            "--output",
            str(tmp_path / "suite.json"),
            "--candidate",
            "unused",
        ]
    elif command == "minimize":
        arguments = [
            "minimize",
            str(events),
            "--events-output",
            str(tmp_path / "reduced.jsonl"),
            "--output",
            str(tmp_path / "minimization.json"),
            "--max-runs",
            "1",
            "--candidate",
            "unused",
        ]
    elif command == "campaign":
        arguments = [
            "campaign",
            "--output-dir",
            str(tmp_path / "campaign"),
            "--traces",
            "1",
            "--events-per-trace",
            "1",
            "--max-minimize-runs",
            "1",
            "--candidate",
            "unused",
        ]
    elif command == "qualify":
        arguments = [
            "qualify",
            "--output-dir",
            str(tmp_path / "qualification"),
            "--suite-version",
            "v1",
            "--traces",
            "1",
            "--events-per-trace",
            "1",
            "--max-minimize-runs",
            "1",
            "--candidate",
            "unused",
        ]
    else:
        arguments = [
            "reproduce",
            str(events),
            "--output",
            str(tmp_path / "reproduction.json"),
            "--candidate",
            "unused",
        ]

    assert main(arguments) == 2


def test_cli_keeps_semantic_divergence_at_one(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    _write_trace(events)
    monkeypatch.setattr(
        conformance_cli,
        "_candidate_factory",
        lambda _args: _SemanticDivergenceAdapter,
    )

    assert main(["run", str(events), "--candidate", "unused"]) == 1


def test_minimizer_rejects_protocol_failure_after_initial_semantic_divergence():
    factory = _SequencedAdapterFactory(
        _LateSemanticDivergenceAdapter,
        _SnapshotFailureAdapter,
    )

    with pytest.raises(
        ConformanceError,
        match="candidate protocol failure during semantic minimization run 2",
    ):
        minimize_failing_trace(
            [_event(1), _event(2)],
            factory,
            max_runs=3,
        )


def test_minimizer_rejects_transient_close_failure_during_semantic_reduction():
    factory = _SequencedAdapterFactory(
        _LateSemanticDivergenceAdapter,
        _SemanticAndCloseFailureAdapter,
        _LateSemanticDivergenceAdapter,
    )

    with pytest.raises(
        ConformanceError,
        match=(
            "candidate protocol failure during semantic minimization run 2: "
            "adapter_close_error: candidate close failed"
        ),
    ):
        minimize_failing_trace(
            [_event(1), _event(2)],
            factory,
            max_runs=3,
        )


def test_cli_minimize_returns_two_for_protocol_failure_during_reduction(tmp_path, monkeypatch):
    events = tmp_path / "events.jsonl"
    _write_trace(events, [_event(1), _event(2)])
    factory = _SequencedAdapterFactory(
        _LateSemanticDivergenceAdapter,
        _SnapshotFailureAdapter,
    )
    monkeypatch.setattr(conformance_cli, "_candidate_factory", lambda _args: factory)

    exit_code = main(
        [
            "minimize",
            str(events),
            "--events-output",
            str(tmp_path / "reduced.jsonl"),
            "--output",
            str(tmp_path / "minimization.json"),
            "--max-runs",
            "3",
            "--candidate",
            "unused",
        ]
    )

    assert exit_code == 2
    assert not (tmp_path / "reduced.jsonl").exists()
    assert not (tmp_path / "minimization.json").exists()


def test_cli_campaign_returns_two_for_protocol_failure_during_reduction(tmp_path, monkeypatch):
    factory = _SequencedAdapterFactory(
        _LateSemanticDivergenceAdapter,
        _LateSemanticDivergenceAdapter,
        _SnapshotFailureAdapter,
    )
    monkeypatch.setattr(conformance_cli, "_candidate_factory", lambda _args: factory)
    output_dir = tmp_path / "campaign"

    exit_code = main(
        [
            "campaign",
            "--output-dir",
            str(output_dir),
            "--traces",
            "1",
            "--events-per-trace",
            "2",
            "--max-minimize-runs",
            "3",
            "--candidate",
            "unused",
        ]
    )

    assert exit_code == 2
    assert {path.name for path in output_dir.iterdir()} == {".tracebook-campaign-reservation"}


def test_cli_qualify_returns_two_for_protocol_failure_during_reduction(tmp_path, monkeypatch):
    factory = _SequencedAdapterFactory(
        _WrappedReferenceAdapter,
        _WrappedReferenceAdapter,
        _WrappedReferenceAdapter,
        _LateSemanticDivergenceAdapter,
        _LateSemanticDivergenceAdapter,
        _SnapshotFailureAdapter,
    )
    monkeypatch.setattr(conformance_cli, "_candidate_factory", lambda _args: factory)
    output_dir = tmp_path / "qualification"

    exit_code = main(
        [
            "qualify",
            "--output-dir",
            str(output_dir),
            "--suite-version",
            "v1",
            "--traces",
            "1",
            "--events-per-trace",
            "2",
            "--max-minimize-runs",
            "3",
            "--candidate",
            "unused",
        ]
    )

    assert exit_code == 2
    assert {path.name for path in output_dir.iterdir()} == {".tracebook-campaign-reservation"}
