import hashlib
import json
import sys
from pathlib import Path

import pytest

from tracebook.conformance.campaign import get_campaign_profile
from tracebook.conformance.cli import main
from tracebook.conformance.evidence import (
    MANIFEST_FILENAME,
    candidate_snapshot_id,
    prepare_evidence_workspace,
    verify_evidence_plan,
    write_evidence_manifest,
)
from tracebook.conformance.external import AdapterProtocolError, ExternalProcessAdapter
from tracebook.conformance.junit import render_junit
from tracebook.conformance.model import (
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    PinnedCandidateIdentity,
)


def _artifact_id(payload: dict) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _candidate(tmp_path: Path) -> Path:
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "README.md").write_text("candidate\n", encoding="utf-8")
    (candidate / "src").mkdir()
    (candidate / "src" / "engine.txt").write_text("price-time\n", encoding="utf-8")
    (candidate / ".git").mkdir()
    (candidate / ".git" / "HEAD").write_text("ignored\n", encoding="utf-8")
    return candidate


def _plan(tmp_path: Path) -> tuple[Path, dict]:
    candidate = _candidate(tmp_path)
    plan_path = prepare_evidence_workspace(
        candidate,
        tmp_path / "evidence",
        candidate_name="example/matcher",
        candidate_revision="abc123",
    )
    return plan_path, json.loads(plan_path.read_text(encoding="utf-8"))


def _qualification(plan: dict) -> dict:
    identity = plan["candidate"]
    parameters = plan["qualification"]
    metadata = {
        **identity,
        "version": "1.2.3",
        "language": "Rust",
    }
    capabilities = parameters["capabilities"]
    profile = get_campaign_profile(parameters["profile"]).to_dict()
    coverage = {
        "schema_version": 1,
        "basis": "reference observations for candidate-compared events",
        "expected_capabilities": capabilities,
        "covered_capabilities": capabilities,
        "uncovered_capabilities": [],
        "covered_count": len(capabilities),
        "expected_count": len(capabilities),
        "coverage_ratio": 1.0,
        "evidence": {name: 1 for name in capabilities},
        "operations": {},
        "order_types": {},
        "applied_order_types": {},
        "outcomes": {},
        "rejection_reasons": {},
        "symbols": ["ALPHA", "BETA"],
        "compared_events": 5000,
        "trade_events": 1,
        "trades": 1,
        "partial_fill_events": 1,
        "queue_priority_probes": 1,
    }
    traces = [
        {
            "index": index + 1,
            "seed": index,
            "trace_sha256": f"sha256:{index:064x}",
            "event_count": parameters["events_per_trace"],
            "compared_events": parameters["events_per_trace"],
            "conformant": True,
            "divergence": None,
        }
        for index in range(parameters["traces"])
    ]
    cases = [
        {
            "name": name,
            "events_sha256": "sha256:" + f"{index + 10:064x}",
            "report": {
                "artifact_type": "tracebook.conformance.report",
                "schema_version": 1,
                "candidate_engine": metadata,
                "conformant": True,
                "divergence": None,
            },
        }
        for index, name in enumerate(parameters["fixed_cases"])
    ]
    suite_report = {
        "artifact_type": "tracebook.conformance.suite_report",
        "schema_version": 1,
        "suite_id": parameters["suite_id"],
        "suite_hash": "sha256:" + "1" * 64,
        "candidate_engine": metadata,
        "case_count": len(cases),
        "conformant_cases": len(cases),
        "conformant": True,
        "cases": cases,
    }
    campaign = {
        "artifact_type": "tracebook.conformance.campaign",
        "schema_version": 1,
        "candidate_engine": metadata,
        "profile": profile,
        "generator_version": 2,
        "seed": parameters["seed"],
        "requested_traces": parameters["traces"],
        "completed_traces": parameters["traces"],
        "events_per_trace": parameters["events_per_trace"],
        "generated_events": parameters["traces"] * parameters["events_per_trace"],
        "max_minimize_runs": parameters["max_minimize_runs"],
        "candidate_runs": parameters["traces"],
        "conformant": True,
        "failure": None,
        "stopped_at_first_divergence": True,
        "semantic_coverage": coverage,
        "traces": traces,
    }
    campaign["campaign_id"] = _artifact_id(
        {
            "generator_version": 2,
            "profile": profile,
            "seed": parameters["seed"],
            "requested_traces": parameters["traces"],
            "events_per_trace": parameters["events_per_trace"],
        }
    )
    checks = {
        "fixed_cases": {
            "passed": len(cases),
            "total": len(cases),
            "complete": True,
        },
        "generated_campaign": {
            "completed_traces": parameters["traces"],
            "requested_traces": parameters["traces"],
            "conformant": True,
        },
        "semantic_coverage": {
            "covered": len(capabilities),
            "expected": len(capabilities),
            "uncovered": [],
            "complete": True,
        },
    }
    qualification = {
        "artifact_type": "tracebook.conformance.qualification",
        "schema_version": 1,
        "qualification_version": 1,
        "qualified": True,
        "candidate_engine": metadata,
        "profile": profile,
        "suite": {
            "suite_id": parameters["suite_id"],
            "suite_hash": suite_report["suite_hash"],
            "selection_version": 1,
            "selected_cases": parameters["fixed_cases"],
            "report": suite_report,
        },
        "campaign": campaign,
        "checks": checks,
        "candidate_runs": len(cases) + parameters["traces"],
        "paths": {
            "suite": "suite.json",
            "campaign": "campaign.json",
            "junit": "qualification.xml",
            "reduced": None,
        },
    }
    qualification["qualification_id"] = _artifact_id(
        {
            "qualification_version": 1,
            "profile": profile,
            "suite_id": parameters["suite_id"],
            "suite_hash": suite_report["suite_hash"],
            "selected_cases": parameters["fixed_cases"],
            "suite_report": suite_report,
            "campaign": campaign,
        }
    )
    return qualification


def _write_bundle(root: Path, payload: dict) -> None:
    root.mkdir()
    (root / "qualification.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "campaign.json").write_text(
        json.dumps(payload["campaign"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "suite.json").write_text(
        json.dumps(payload["suite"]["report"], indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (root / "qualification.xml").write_text(render_junit(payload), encoding="utf-8")


def _write_pair(plan_path: Path, plan: dict, payload: dict) -> None:
    for run in plan["runs"]:
        _write_bundle(plan_path.parent / run["qualification_dir"], payload)


def test_prepare_creates_two_clean_roots_and_pins_snapshot(tmp_path):
    candidate = _candidate(tmp_path)
    snapshot = candidate_snapshot_id(candidate)

    plan_path = prepare_evidence_workspace(
        candidate,
        tmp_path / "evidence",
        candidate_name="example/matcher",
        candidate_revision="abc123",
        expected_snapshot=snapshot,
    )
    plan = json.loads(plan_path.read_text(encoding="utf-8"))

    assert plan["candidate"] == {
        "name": "example/matcher",
        "revision": "abc123",
        "snapshot_id": snapshot,
    }
    assert plan["qualification"]["profile"] == "fifo-limit-v1"
    assert plan["qualification"]["seed"] == 42
    assert plan["qualification"]["traces"] == 25
    assert plan["qualification"]["events_per_trace"] == 200
    assert [run["run_id"] for run in plan["runs"]] == ["run-1", "run-2"]
    for run in plan["runs"]:
        root = plan_path.parent / "runs" / run["run_id"]
        assert candidate_snapshot_id(root / "candidate") == snapshot
        assert not (root / "candidate" / ".git").exists()
        assert (root / "adapter").is_dir()
        assert (root / "build").is_dir()
        assert (root / "cache").is_dir()
        assert not (root / "qualification").exists()


def test_prepare_refuses_snapshot_mismatch_and_nested_workspace(tmp_path):
    candidate = _candidate(tmp_path)

    with pytest.raises(ConformanceError, match="does not match"):
        prepare_evidence_workspace(
            candidate,
            tmp_path / "wrong",
            candidate_name="example/matcher",
            candidate_revision="abc123",
            expected_snapshot="sha256:wrong",
        )
    with pytest.raises(ConformanceError, match="outside candidate source"):
        prepare_evidence_workspace(
            candidate,
            candidate / "evidence",
            candidate_name="example/matcher",
            candidate_revision="abc123",
        )


def test_prepare_refuses_a_symlinked_candidate_root(tmp_path):
    candidate = _candidate(tmp_path)
    alias = tmp_path / "candidate-link"
    alias.symlink_to(candidate, target_is_directory=True)

    with pytest.raises(ConformanceError, match="real directory"):
        candidate_snapshot_id(alias)
    with pytest.raises(ConformanceError, match="real directory"):
        prepare_evidence_workspace(
            alias,
            tmp_path / "evidence",
            candidate_name="example/matcher",
            candidate_revision="abc123",
        )


def test_verify_writes_grading_manifest_for_identical_canonical_pair(tmp_path):
    plan_path, plan = _plan(tmp_path)
    payload = _qualification(plan)
    _write_pair(plan_path, plan, payload)

    manifest_path = write_evidence_manifest(plan_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest_path.name == MANIFEST_FILENAME
    assert manifest["status"] == "qualified"
    assert manifest["candidate"] == plan["candidate"]
    assert manifest["equality"] == {
        "terminal_result": True,
        "candidate_metadata": True,
        "campaign_id": True,
        "qualification_id": True,
        "counts_and_coverage": True,
        "artifact_bytes": True,
    }
    assert len(manifest["runs"]) == 2
    assert manifest["runs"][0]["files"] == manifest["runs"][1]["files"]


def test_verify_refuses_unpinned_metadata_and_changed_candidate(tmp_path):
    plan_path, plan = _plan(tmp_path)
    payload = _qualification(plan)
    payload["candidate_engine"]["revision"] = "wrong"
    _write_pair(plan_path, plan, payload)

    with pytest.raises(ConformanceError, match="candidate revision"):
        verify_evidence_plan(plan_path)

    for run in plan["runs"]:
        qualification = plan_path.parent / run["qualification_dir"]
        for child in qualification.iterdir():
            child.unlink()
        qualification.rmdir()
    payload = _qualification(plan)
    _write_pair(plan_path, plan, payload)
    candidate_file = plan_path.parent / plan["runs"][0]["candidate_root"] / "README.md"
    candidate_file.write_text("mutated\n", encoding="utf-8")

    with pytest.raises(ConformanceError, match="candidate tree changed"):
        verify_evidence_plan(plan_path)


def test_verify_refuses_noncanonical_or_different_bundle_bytes(tmp_path):
    plan_path, plan = _plan(tmp_path)
    payload = _qualification(plan)
    _write_pair(plan_path, plan, payload)
    first = plan_path.parent / plan["runs"][0]["qualification_dir"] / "campaign.json"
    first.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ConformanceError, match="not the canonical artifact"):
        verify_evidence_plan(plan_path)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("campaign-result", "terminal result"),
        ("capability-list", "semantic coverage"),
        ("case-inventory", "case inventory"),
        ("candidate-runs", "candidate-run count"),
    ),
)
def test_verify_refuses_incomplete_or_substituted_claims(tmp_path, mutation, message):
    plan_path, plan = _plan(tmp_path)
    payload = _qualification(plan)
    if mutation == "campaign-result":
        payload["campaign"]["conformant"] = False
    elif mutation == "capability-list":
        payload["campaign"]["semantic_coverage"]["covered_capabilities"] = []
    elif mutation == "case-inventory":
        payload["suite"]["report"]["cases"][0]["name"] = "substituted"
    else:
        payload["candidate_runs"] -= 1
    _write_pair(plan_path, plan, payload)

    with pytest.raises(ConformanceError, match=message):
        verify_evidence_plan(plan_path)


def test_manifest_output_stays_at_workspace_root_and_plan_must_be_real(tmp_path):
    plan_path, plan = _plan(tmp_path)
    _write_pair(plan_path, plan, _qualification(plan))

    with pytest.raises(ConformanceError, match="workspace root"):
        write_evidence_manifest(plan_path, "nested/manifest.json")

    custom = write_evidence_manifest(plan_path, "review.json")
    assert custom == plan_path.parent / "review.json"

    plan_link = tmp_path / "plan-link.json"
    plan_link.symlink_to(plan_path)
    with pytest.raises(ConformanceError, match="real evidence-plan.json"):
        verify_evidence_plan(plan_link)


def test_verify_requires_all_prepared_roots_to_remain_real_directories(tmp_path):
    plan_path, plan = _plan(tmp_path)
    _write_pair(plan_path, plan, _qualification(plan))
    cache = plan_path.parent / plan["runs"][0]["cache_root"]
    cache.rmdir()

    with pytest.raises(ConformanceError, match="cache_root is not a real directory"):
        verify_evidence_plan(plan_path)


def test_engine_metadata_can_bind_and_validate_task_identity():
    identity = PinnedCandidateIdentity("example/matcher", "abc123", "sha256:tree")
    metadata = EngineMetadata(
        "example/matcher",
        "1.2.3",
        "Rust",
        revision="abc123",
        snapshot_id="sha256:tree",
    )

    identity.validate(metadata)
    assert EngineMetadata.from_dict(metadata.to_dict()) == metadata
    with pytest.raises(ConformanceError, match="candidate revision"):
        identity.validate(
            EngineMetadata(
                "example/matcher",
                "1.2.3",
                "Rust",
                revision="wrong",
                snapshot_id="sha256:tree",
            )
        )


def test_external_adapter_rejects_unpinned_ready_metadata_before_events():
    script = """
import json
import sys

json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": "tracebook.conformance",
    "protocol_version": 1,
    "engine": {"name": "example/matcher", "version": "1", "language": "Python"},
}), flush=True)
"""
    identity = PinnedCandidateIdentity("example/matcher", "abc123", "sha256:tree")

    with pytest.raises(AdapterProtocolError, match="candidate revision"):
        ExternalProcessAdapter(
            [sys.executable, "-c", script],
            ConformanceConfig(),
            timeout_seconds=1,
            expected_identity=identity,
        )


def test_cli_requires_all_identity_flags_and_drives_evidence_commands(tmp_path, capsys):
    candidate = _candidate(tmp_path)
    workspace = tmp_path / "evidence"
    assert (
        main(
            [
                "evidence-init",
                str(candidate),
                "--workspace",
                str(workspace),
                "--candidate-name",
                "example/matcher",
                "--candidate-revision",
                "abc123",
            ]
        )
        == 0
    )
    assert "Candidate snapshot: sha256:" in capsys.readouterr().out

    plan_path = workspace / "evidence-plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    _write_pair(plan_path, plan, _qualification(plan))
    assert main(["evidence-verify", str(plan_path)]) == 0
    assert "Evidence pair: PASS" in capsys.readouterr().out

    events = tmp_path / "empty.jsonl"
    events.write_text("", encoding="utf-8")
    assert (
        main(
            [
                "run",
                str(events),
                "--candidate-name",
                "example/matcher",
                "--candidate-cmd",
                f"{sys.executable} -c 'raise SystemExit(99)'",
            ]
        )
        == 2
    )
    assert "must be provided together" in capsys.readouterr().err
