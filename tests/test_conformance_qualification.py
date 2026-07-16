import json
import shlex
import sys
from pathlib import Path
from xml.etree import ElementTree

import pytest

from tracebook.conformance import (
    QUALIFICATION_VERSION,
    ConformanceError,
    EngineMetadata,
    ExternalProcessAdapterFactory,
    ReferenceEngineAdapter,
    qualification_case_names,
    run_qualification,
    write_qualification_artifacts,
)
from tracebook.conformance.cli import main

ROOT = Path(__file__).parents[1]
EXAMPLE_ADAPTER = ROOT / "examples" / "conformance_adapter.py"
FAULTY_ADAPTER = ROOT / "tests" / "fixtures" / "faulty_campaign_adapter.py"


def test_qualification_v1_selects_only_cases_inside_the_declared_profile():
    assert QUALIFICATION_VERSION == 1
    assert qualification_case_names("fifo-limit-v1") == (
        "fifo-lifecycle",
        "tick-grid",
        "deep-cancellation",
    )
    assert qualification_case_names("fifo-full-v1") == (
        "fifo-lifecycle",
        "order-instructions",
        "multi-symbol",
        "tick-grid",
        "deep-cancellation",
    )


def test_reference_engine_earns_a_complete_deterministic_qualification(tmp_path):
    result = run_qualification(
        ReferenceEngineAdapter,
        profile="fifo-limit-v1",
        seed=42,
        traces=1,
        events_per_trace=200,
    )
    destination = tmp_path / "qualification"
    report_path = write_qualification_artifacts(result, destination)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    junit = ElementTree.parse(destination / "qualification.xml").getroot()

    assert result.qualified is True
    assert payload["artifact_type"] == "tracebook.conformance.qualification"
    assert payload["qualification_version"] == 1
    assert payload["qualification_id"].startswith("sha256:")
    assert payload["checks"]["fixed_cases"] == {
        "passed": 3,
        "total": 3,
        "complete": True,
    }
    assert payload["checks"]["semantic_coverage"]["complete"] is True
    assert payload["candidate_runs"] == 4
    assert (destination / "suite.json").is_file()
    assert (destination / "campaign.json").is_file()
    assert junit.attrib["tests"] == "5"
    assert junit.attrib["failures"] == "0"


def test_qualification_fails_when_generated_evidence_does_not_cover_the_profile():
    result = run_qualification(
        ReferenceEngineAdapter,
        profile="fifo-limit-v1",
        seed=42,
        traces=1,
        events_per_trace=1,
    )

    assert result.suite_report["conformant"] is True
    assert result.campaign.conformant is True
    assert result.coverage_complete is False
    assert result.qualified is False
    assert result.to_dict()["checks"]["semantic_coverage"]["uncovered"]


def test_qualification_rejects_candidate_identity_drift_between_phases():
    class ChangingFactory:
        def __init__(self):
            self.calls = 0

        def __call__(self, config):
            self.calls += 1
            adapter = ReferenceEngineAdapter(config)
            version = "suite" if self.calls <= 3 else "campaign"
            adapter.metadata = EngineMetadata("changing", version, "Python")
            return adapter

    with pytest.raises(ConformanceError, match="between suite and campaign"):
        run_qualification(
            ChangingFactory(),
            traces=1,
            events_per_trace=1,
        )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ({"seed": -1}, "seed"),
        ({"seed": True}, "seed"),
        ({"traces": 0}, "traces"),
        ({"events_per_trace": 0}, "events_per_trace"),
        ({"max_minimize_runs": 0}, "max_minimize_runs"),
    ],
)
def test_qualification_rejects_invalid_campaign_controls_before_candidate_work(arguments, message):
    def unexpected_candidate(_config):
        pytest.fail("candidate started before qualification argument validation")

    with pytest.raises(ConformanceError, match=message):
        run_qualification(unexpected_candidate, **arguments)


def test_qualification_cli_produces_a_public_adapter_evidence_bundle(tmp_path, capsys):
    destination = tmp_path / "external-qualification"
    candidate_cmd = shlex.join([sys.executable, str(EXAMPLE_ADAPTER)])

    exit_code = main(
        [
            "qualify",
            "--output-dir",
            str(destination),
            "--profile",
            "fifo-limit-v1",
            "--seed",
            "42",
            "--traces",
            "1",
            "--events-per-trace",
            "200",
            "--candidate-cmd",
            candidate_cmd,
        ]
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Fixed cases: 3/3" in output
    assert "Semantic coverage: 10/10 capabilities" in output
    assert "Qualification: PASS" in output
    assert json.loads((destination / "qualification.json").read_text())["qualified"] is True


def test_divergent_qualification_commits_the_reduced_failure_with_its_reports(tmp_path):
    result = run_qualification(
        ExternalProcessAdapterFactory([sys.executable, str(FAULTY_ADAPTER)]),
        profile="fifo-limit-v1",
        seed=9,
        traces=1,
        events_per_trace=10,
        max_minimize_runs=10,
    )
    destination = tmp_path / "failed-qualification"
    report_path = write_qualification_artifacts(result, destination)
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    junit = ElementTree.parse(destination / "qualification.xml").getroot()

    assert result.qualified is False
    assert payload["paths"]["reduced"] == "reduced.jsonl"
    assert (destination / "reduced.jsonl").is_file()
    assert (destination / "failure" / "minimization.json").is_file()
    assert int(junit.attrib["failures"]) > 0
    assert not (destination / ".tracebook-campaign-reservation").exists()


def test_qualification_cli_rejects_existing_output_before_candidate_work(
    tmp_path, monkeypatch, capsys
):
    destination = tmp_path / "existing"
    destination.mkdir()

    def unexpected_run(*args, **kwargs):
        pytest.fail("qualification ran before output preflight")

    monkeypatch.setattr("tracebook.conformance.cli.run_qualification", unexpected_run)
    exit_code = main(
        [
            "qualify",
            "--output-dir",
            str(destination),
            "--candidate",
            "unused-adapter",
        ]
    )

    assert exit_code == 2
    assert "already exists" in capsys.readouterr().err


@pytest.mark.parametrize(
    "junit_relative",
    [Path("."), Path("campaign.json"), Path("nested/report.xml")],
)
def test_qualification_cli_rejects_junit_inside_bundle_before_candidate_work(
    tmp_path, monkeypatch, capsys, junit_relative
):
    destination = tmp_path / "qualification"

    def unexpected_run(*args, **kwargs):
        pytest.fail("qualification ran before JUnit path preflight")

    monkeypatch.setattr("tracebook.conformance.cli.run_qualification", unexpected_run)
    exit_code = main(
        [
            "qualify",
            "--output-dir",
            str(destination),
            "--junit-output",
            str(destination / junit_relative),
            "--candidate",
            "unused-adapter",
        ]
    )

    assert exit_code == 2
    assert "outside artifact directories" in capsys.readouterr().err
    assert not destination.exists()
