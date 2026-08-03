"""Machine-enforced preparation and verification for qualification evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

from .campaign import (
    CAMPAIGN_GENERATOR_VERSION,
    _validated_campaign_parameters,
    get_campaign_profile,
)
from .junit import render_junit
from .model import ARTIFACT_SCHEMA_VERSION, ConformanceError, PinnedCandidateIdentity
from .qualification import QUALIFICATION_VERSION, qualification_case_names
from .suite import BUNDLED_SUITE_VERSIONS

EVIDENCE_PLAN_VERSION = 1
EVIDENCE_MANIFEST_VERSION = 1
PLAN_FILENAME = "evidence-plan.json"
MANIFEST_FILENAME = "evidence-manifest.json"
_RUN_IDS = ("run-1", "run-2")
_BUNDLE_FILES = (
    "campaign.json",
    "qualification.json",
    "qualification.xml",
    "suite.json",
)
_REQUIRED_PROFILE = "fifo-limit-v1"
_REQUIRED_SUITE_VERSION = "v2"
_REQUIRED_SEED = 42
_REQUIRED_TRACES = 25
_REQUIRED_EVENTS_PER_TRACE = 200
_REQUIRED_MAX_MINIMIZE_RUNS = 100


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    ).encode("utf-8")


def _content_id(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _required_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConformanceError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha256_id(value: Any, field_name: str) -> str:
    text = _required_text(value, field_name)
    if len(text) != 71 or not text.startswith("sha256:"):
        raise ConformanceError(f"{field_name} must be a SHA-256 identifier")
    try:
        int(text.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise ConformanceError(f"{field_name} must be a SHA-256 identifier") from exc
    return text


def _safe_relative_path(value: Any, field_name: str) -> Path:
    text = _required_text(value, field_name)
    path = Path(text)
    if path.is_absolute() or path == Path(".") or ".." in path.parts:
        raise ConformanceError(f"{field_name} must be a safe relative path")
    return path


def _inside(root: Path, path: Path, field_name: str) -> Path:
    lexical = Path(os.path.abspath(path.expanduser()))
    resolved = lexical.resolve()
    if resolved != root and root not in resolved.parents:
        raise ConformanceError(f"{field_name} must stay inside evidence workspace")
    if resolved != lexical:
        raise ConformanceError(f"{field_name} must not traverse symbolic links")
    return resolved


def _iter_snapshot_entries(root: Path) -> Iterable[Tuple[str, Path, os.stat_result]]:
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        directory_names[:] = sorted(name for name in directory_names if name != ".git")
        file_names = sorted(name for name in file_names if name != ".git")
        for name in directory_names + file_names:
            path = directory_path / name
            relative = path.relative_to(root).as_posix()
            yield relative, path, path.lstat()


def candidate_snapshot_id(candidate_root: str | Path) -> str:
    """Return the origin-stripped, content-addressed ID for one candidate tree."""
    source = Path(candidate_root).expanduser()
    if source.is_symlink():
        raise ConformanceError("candidate source must be a real directory")
    root = source.resolve()
    if not root.is_dir():
        raise ConformanceError("candidate source must be a real directory")
    digest = hashlib.sha256(b"tracebook-candidate-snapshot-v1\0")
    for relative, path, metadata in _iter_snapshot_entries(root):
        encoded_path = relative.encode("utf-8")
        if stat.S_ISDIR(metadata.st_mode):
            kind = b"directory"
            content = b""
        elif stat.S_ISREG(metadata.st_mode):
            kind = b"file-executable" if metadata.st_mode & 0o111 else b"file"
            content = path.read_bytes()
        elif stat.S_ISLNK(metadata.st_mode):
            target_text = os.readlink(path)
            try:
                target = path.resolve(strict=True)
            except OSError as exc:
                raise ConformanceError(f"candidate symlink is broken: {relative}") from exc
            if target != root and root not in target.parents:
                raise ConformanceError(f"candidate symlink escapes source root: {relative}")
            kind = b"symlink"
            content = target_text.encode("utf-8")
        else:
            raise ConformanceError(f"candidate source contains a special file: {relative}")
        digest.update(kind)
        digest.update(b"\0")
        digest.update(encoded_path)
        digest.update(b"\0")
        digest.update(str(len(content)).encode("ascii"))
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return "sha256:" + digest.hexdigest()


def _copy_candidate(source: Path, destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {".git"} if ".git" in names else set()

    shutil.copytree(source, destination, symlinks=True, ignore=ignore)


def prepare_evidence_workspace(
    candidate_source: str | Path,
    workspace: str | Path,
    *,
    candidate_name: str,
    candidate_revision: str,
    expected_snapshot: Optional[str] = None,
) -> Path:
    """Create two clean, captured roots and their immutable evidence plan."""
    unresolved_source = Path(candidate_source).expanduser()
    if unresolved_source.is_symlink():
        raise ConformanceError("candidate source must be a real directory")
    source = unresolved_source.resolve()
    destination = Path(workspace).expanduser().resolve()
    if not source.is_dir():
        raise ConformanceError("candidate source must be a real directory")
    if destination == source or source in destination.parents:
        raise ConformanceError("evidence workspace must be outside candidate source")
    if destination.exists() or destination.is_symlink():
        raise ConformanceError(f"evidence workspace already exists: {destination}")

    identity = PinnedCandidateIdentity(
        name=candidate_name,
        revision=candidate_revision,
        snapshot_id=candidate_snapshot_id(source),
    )
    if expected_snapshot is not None and identity.snapshot_id != expected_snapshot.strip():
        raise ConformanceError(
            "candidate snapshot does not match --expected-snapshot: "
            f"expected {expected_snapshot.strip()!r}, observed {identity.snapshot_id!r}"
        )
    suite_version = _REQUIRED_SUITE_VERSION
    if suite_version not in BUNDLED_SUITE_VERSIONS:
        choices = ", ".join(BUNDLED_SUITE_VERSIONS)
        raise ConformanceError(
            f"unknown bundled suite version {suite_version!r}; expected one of: {choices}"
        )
    seed, traces, events_per_trace, max_minimize_runs = _validated_campaign_parameters(
        _REQUIRED_SEED,
        _REQUIRED_TRACES,
        _REQUIRED_EVENTS_PER_TRACE,
        _REQUIRED_MAX_MINIMIZE_RUNS,
    )
    selected_profile = get_campaign_profile(_REQUIRED_PROFILE)
    selected_cases = qualification_case_names(selected_profile)
    parameters = {
        "profile": selected_profile.name,
        "suite_version": suite_version,
        "suite_id": f"tracebook-conformance-{suite_version}",
        "seed": seed,
        "traces": traces,
        "events_per_trace": events_per_trace,
        "max_minimize_runs": max_minimize_runs,
        "fixed_cases": list(selected_cases),
        "capabilities": list(selected_profile.capabilities),
    }
    runs = []
    created = False
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.mkdir(mode=0o700)
        created = True
        for run_id in _RUN_IDS:
            run_root = destination / "runs" / run_id
            run_root.mkdir(parents=True)
            candidate_root = run_root / "candidate"
            _copy_candidate(source, candidate_root)
            copied_snapshot = candidate_snapshot_id(candidate_root)
            if copied_snapshot != identity.snapshot_id:
                raise ConformanceError(f"{run_id} candidate copy changed snapshot identity")
            for name in ("adapter", "build", "cache"):
                (run_root / name).mkdir()
            runs.append(
                {
                    "run_id": run_id,
                    "candidate_root": f"runs/{run_id}/candidate",
                    "adapter_root": f"runs/{run_id}/adapter",
                    "build_root": f"runs/{run_id}/build",
                    "cache_root": f"runs/{run_id}/cache",
                    "qualification_dir": f"runs/{run_id}/qualification",
                }
            )
        plan_without_id: Dict[str, Any] = {
            "artifact_type": "tracebook.conformance.evidence-plan",
            "schema_version": EVIDENCE_PLAN_VERSION,
            "candidate": identity.to_dict(),
            "qualification": parameters,
            "runs": runs,
        }
        plan = {**plan_without_id, "plan_id": _content_id(plan_without_id)}
        plan_path = destination / PLAN_FILENAME
        with plan_path.open("xb") as handle:
            handle.write(_json_bytes(plan))
        return plan_path
    except Exception:
        if created:
            shutil.rmtree(destination, ignore_errors=True)
        raise


def _load_json_object(path: Path, description: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ConformanceError(f"could not read {description} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConformanceError(f"{description} must be a JSON object: {path}")
    return payload


def load_evidence_plan(path: str | Path) -> Tuple[Path, dict]:
    """Load and structurally validate one evidence plan."""
    unresolved_plan = Path(path).expanduser()
    if unresolved_plan.is_symlink():
        raise ConformanceError(f"evidence plan must be a real {PLAN_FILENAME} file")
    plan_path = unresolved_plan.resolve()
    if plan_path.name != PLAN_FILENAME or not plan_path.is_file():
        raise ConformanceError(f"evidence plan must be a real {PLAN_FILENAME} file")
    plan = _load_json_object(plan_path, "evidence plan")
    if plan.get("artifact_type") != "tracebook.conformance.evidence-plan":
        raise ConformanceError("evidence plan has the wrong artifact_type")
    if plan.get("schema_version") != EVIDENCE_PLAN_VERSION:
        raise ConformanceError("evidence plan has an unsupported schema_version")
    plan_id = _required_text(plan.get("plan_id"), "plan_id")
    unsigned = dict(plan)
    del unsigned["plan_id"]
    if plan_id != _content_id(unsigned):
        raise ConformanceError("evidence plan content does not match plan_id")
    candidate = plan.get("candidate")
    if not isinstance(candidate, dict):
        raise ConformanceError("evidence plan candidate must be an object")
    pinned_candidate = PinnedCandidateIdentity.from_dict(candidate)
    _sha256_id(pinned_candidate.snapshot_id, "candidate.snapshot_id")
    qualification = plan.get("qualification")
    if not isinstance(qualification, dict):
        raise ConformanceError("evidence plan qualification must be an object")
    selected_profile = get_campaign_profile(_REQUIRED_PROFILE)
    expected_qualification = {
        "profile": selected_profile.name,
        "suite_version": _REQUIRED_SUITE_VERSION,
        "suite_id": f"tracebook-conformance-{_REQUIRED_SUITE_VERSION}",
        "seed": _REQUIRED_SEED,
        "traces": _REQUIRED_TRACES,
        "events_per_trace": _REQUIRED_EVENTS_PER_TRACE,
        "max_minimize_runs": _REQUIRED_MAX_MINIMIZE_RUNS,
        "fixed_cases": list(qualification_case_names(selected_profile)),
        "capabilities": list(selected_profile.capabilities),
    }
    if qualification != expected_qualification:
        raise ConformanceError("evidence plan qualification contract is not canonical")
    runs = plan.get("runs")
    expected_runs = [
        {
            "run_id": run_id,
            "candidate_root": f"runs/{run_id}/candidate",
            "adapter_root": f"runs/{run_id}/adapter",
            "build_root": f"runs/{run_id}/build",
            "cache_root": f"runs/{run_id}/cache",
            "qualification_dir": f"runs/{run_id}/qualification",
        }
        for run_id in _RUN_IDS
    ]
    if runs != expected_runs:
        raise ConformanceError("evidence plan does not use the canonical two-run layout")
    return plan_path, plan


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ConformanceError(f"{field_name} must be a positive integer")
    return value


def _validate_candidate_metadata(
    payload: Any, expected: PinnedCandidateIdentity, path: str
) -> dict:
    if not isinstance(payload, dict):
        raise ConformanceError(f"{path} candidate metadata must be an object")
    for field_name, expected_value in expected.to_dict().items():
        if payload.get(field_name) != expected_value:
            raise ConformanceError(
                f"{path} candidate {field_name} does not match evidence plan: "
                f"expected {expected_value!r}, observed {payload.get(field_name)!r}"
            )
    _required_text(payload.get("version"), f"{path}.candidate.version")
    _required_text(payload.get("language"), f"{path}.candidate.language")
    return payload


def _validate_qualification(
    payload: dict,
    expected: PinnedCandidateIdentity,
    parameters: Mapping[str, Any],
    run_id: str,
) -> dict:
    prefix = f"{run_id} qualification"
    if payload.get("artifact_type") != "tracebook.conformance.qualification":
        raise ConformanceError(f"{prefix} has the wrong artifact_type")
    if (
        payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or payload.get("qualification_version") != QUALIFICATION_VERSION
    ):
        raise ConformanceError(f"{prefix} has an unsupported schema or contract version")
    if payload.get("qualified") is not True:
        raise ConformanceError(f"{prefix} is not qualified")
    metadata = _validate_candidate_metadata(payload.get("candidate_engine"), expected, prefix)
    expected_profile = get_campaign_profile(str(parameters.get("profile"))).to_dict()
    profile = payload.get("profile")
    if profile != expected_profile:
        raise ConformanceError(f"{prefix} profile does not match evidence plan")
    suite = payload.get("suite")
    if not isinstance(suite, dict):
        raise ConformanceError(f"{prefix} suite must be an object")
    if suite.get("selection_version") != 1:
        raise ConformanceError(f"{prefix} selection version must be 1")
    if suite.get("suite_id") != parameters.get("suite_id"):
        raise ConformanceError(f"{prefix} suite ID does not match evidence plan")
    if suite.get("selected_cases") != parameters.get("fixed_cases"):
        raise ConformanceError(f"{prefix} fixed-case selection does not match evidence plan")
    suite_hash = _sha256_id(suite.get("suite_hash"), f"{prefix}.suite.suite_hash")
    campaign = payload.get("campaign")
    if not isinstance(campaign, dict):
        raise ConformanceError(f"{prefix} campaign must be an object")
    if (
        campaign.get("artifact_type") != "tracebook.conformance.campaign"
        or campaign.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or campaign.get("generator_version") != CAMPAIGN_GENERATOR_VERSION
    ):
        raise ConformanceError(f"{prefix} campaign has an unsupported artifact version")
    expected_campaign = {
        "seed": parameters.get("seed"),
        "requested_traces": parameters.get("traces"),
        "events_per_trace": parameters.get("events_per_trace"),
        "max_minimize_runs": parameters.get("max_minimize_runs"),
    }
    for field_name, expected_value in expected_campaign.items():
        if campaign.get(field_name) != expected_value:
            raise ConformanceError(f"{prefix} campaign {field_name} does not match evidence plan")
    if campaign.get("profile") != expected_profile:
        raise ConformanceError(f"{prefix} campaign profile does not match evidence plan")
    if (
        campaign.get("completed_traces") != parameters.get("traces")
        or campaign.get("candidate_runs") != parameters.get("traces")
        or campaign.get("conformant") is not True
        or campaign.get("failure") is not None
        or campaign.get("stopped_at_first_divergence") is not True
    ):
        raise ConformanceError(f"{prefix} campaign terminal result is incomplete")
    expected_events = _positive_integer(parameters.get("traces"), "qualification.traces") * (
        _positive_integer(parameters.get("events_per_trace"), "qualification.events_per_trace")
    )
    if campaign.get("generated_events") != expected_events:
        raise ConformanceError(f"{prefix} generated-event count is incomplete")
    traces = campaign.get("traces")
    if not isinstance(traces, list) or len(traces) != parameters.get("traces"):
        raise ConformanceError(f"{prefix} trace inventory is incomplete")
    for expected_index, trace in enumerate(traces, 1):
        if (
            not isinstance(trace, dict)
            or trace.get("index") != expected_index
            or trace.get("conformant") is not True
            or trace.get("divergence") is not None
            or trace.get("event_count") != parameters.get("events_per_trace")
            or trace.get("compared_events") != parameters.get("events_per_trace")
        ):
            raise ConformanceError(f"{prefix} contains an incomplete generated trace")
        _sha256_id(trace.get("trace_sha256"), f"{prefix}.trace[{expected_index}].trace_sha256")
    checks = payload.get("checks")
    if not isinstance(checks, dict):
        raise ConformanceError(f"{prefix} checks must be an object")
    fixed = checks.get("fixed_cases")
    generated = checks.get("generated_campaign")
    coverage = checks.get("semantic_coverage")
    expected_fixed = len(parameters.get("fixed_cases", []))
    if fixed != {"passed": expected_fixed, "total": expected_fixed, "complete": True}:
        raise ConformanceError(f"{prefix} fixed-case counts are incomplete")
    expected_traces = parameters.get("traces")
    if generated != {
        "completed_traces": expected_traces,
        "requested_traces": expected_traces,
        "conformant": True,
    }:
        raise ConformanceError(f"{prefix} generated campaign is incomplete")
    if not isinstance(coverage, dict) or coverage.get("complete") is not True:
        raise ConformanceError(f"{prefix} semantic coverage is incomplete")
    expected_capabilities = len(parameters.get("capabilities", []))
    if (
        coverage.get("covered") != expected_capabilities
        or coverage.get("expected") != expected_capabilities
        or coverage.get("uncovered") != []
    ):
        raise ConformanceError(f"{prefix} semantic coverage counts are incomplete")
    campaign_coverage = campaign.get("semantic_coverage")
    if not isinstance(campaign_coverage, dict):
        raise ConformanceError(f"{prefix} campaign semantic coverage is missing")
    if (
        campaign_coverage.get("covered_count") != expected_capabilities
        or campaign_coverage.get("expected_count") != expected_capabilities
        or campaign_coverage.get("expected_capabilities") != parameters.get("capabilities")
        or campaign_coverage.get("covered_capabilities") != parameters.get("capabilities")
        or campaign_coverage.get("uncovered_capabilities") != []
        or campaign_coverage.get("compared_events") != expected_events
    ):
        raise ConformanceError(f"{prefix} campaign semantic coverage is incomplete")
    suite_report = suite.get("report")
    if (
        not isinstance(suite_report, dict)
        or suite_report.get("artifact_type") != "tracebook.conformance.suite_report"
        or suite_report.get("schema_version") != ARTIFACT_SCHEMA_VERSION
        or suite_report.get("suite_id") != parameters.get("suite_id")
        or suite_report.get("suite_hash") != suite_hash
        or suite_report.get("conformant") is not True
        or suite_report.get("conformant_cases") != expected_fixed
        or suite_report.get("case_count") != expected_fixed
        or not isinstance(suite_report.get("cases"), list)
        or len(suite_report["cases"]) != expected_fixed
    ):
        raise ConformanceError(f"{prefix} suite report is incomplete")
    if [case.get("name") for case in suite_report["cases"] if isinstance(case, dict)] != list(
        parameters.get("fixed_cases", [])
    ):
        raise ConformanceError(f"{prefix} suite case inventory does not match evidence plan")
    for case in suite_report["cases"]:
        if (
            not isinstance(case, dict)
            or not isinstance(case.get("report"), dict)
            or case["report"].get("artifact_type") != "tracebook.conformance.report"
            or case["report"].get("schema_version") != ARTIFACT_SCHEMA_VERSION
            or case["report"].get("conformant") is not True
            or case["report"].get("divergence") is not None
            or case["report"].get("candidate_engine") != metadata
        ):
            raise ConformanceError(f"{prefix} contains a nonconformant fixed case")
        _sha256_id(case.get("events_sha256"), f"{prefix}.case.events_sha256")
    expected_candidate_runs = expected_fixed + _positive_integer(
        parameters.get("traces"), "qualification.traces"
    )
    if payload.get("candidate_runs") != expected_candidate_runs:
        raise ConformanceError(f"{prefix} candidate-run count is incomplete")
    if payload.get("paths") != {
        "suite": "suite.json",
        "campaign": "campaign.json",
        "junit": "qualification.xml",
        "reduced": None,
    }:
        raise ConformanceError(f"{prefix} artifact paths are not canonical")
    campaign_identity = {
        "generator_version": CAMPAIGN_GENERATOR_VERSION,
        "profile": expected_profile,
        "seed": parameters.get("seed"),
        "requested_traces": parameters.get("traces"),
        "events_per_trace": parameters.get("events_per_trace"),
    }
    campaign_id = _sha256_id(campaign.get("campaign_id"), f"{prefix}.campaign_id")
    if campaign_id != _content_id(campaign_identity):
        raise ConformanceError(f"{prefix} campaign content does not match campaign_id")
    qualification_identity = {
        "qualification_version": QUALIFICATION_VERSION,
        "profile": expected_profile,
        "suite_id": parameters.get("suite_id"),
        "suite_hash": suite_hash,
        "selected_cases": parameters.get("fixed_cases"),
        "suite_report": suite_report,
        "campaign": campaign,
    }
    qualification_id = _sha256_id(payload.get("qualification_id"), f"{prefix}.qualification_id")
    if qualification_id != _content_id(qualification_identity):
        raise ConformanceError(f"{prefix} content does not match qualification_id")
    for nested_path, nested in (
        ("campaign", campaign.get("candidate_engine")),
        ("suite.report", suite_report.get("candidate_engine")),
    ):
        if nested != metadata:
            raise ConformanceError(f"{prefix} {nested_path} candidate metadata changed")
    return {
        "qualification_id": qualification_id,
        "campaign_id": campaign_id,
        "candidate_engine": metadata,
        "fixed_cases": fixed,
        "generated_campaign": generated,
        "semantic_coverage": coverage,
    }


def _bundle_record(bundle: Path) -> Tuple[dict, Dict[str, bytes]]:
    if not bundle.is_dir() or bundle.is_symlink():
        raise ConformanceError(f"qualification bundle is not a real directory: {bundle}")
    observed = sorted(path.name for path in bundle.iterdir())
    if observed != sorted(_BUNDLE_FILES):
        raise ConformanceError(
            f"qualification bundle must contain exactly {', '.join(_BUNDLE_FILES)}: {bundle}"
        )
    contents: Dict[str, bytes] = {}
    hashes: Dict[str, str] = {}
    for name in _BUNDLE_FILES:
        path = bundle / name
        if not path.is_file() or path.is_symlink():
            raise ConformanceError(f"qualification artifact must be a real file: {path}")
        contents[name] = path.read_bytes()
        hashes[name] = "sha256:" + hashlib.sha256(contents[name]).hexdigest()
    return {"files": hashes}, contents


def verify_evidence_plan(path: str | Path) -> dict:
    """Verify two captured qualification bundles and return a grading manifest."""
    plan_path, plan = load_evidence_plan(path)
    root = plan_path.parent.resolve()
    expected = PinnedCandidateIdentity.from_dict(plan["candidate"])
    parameters = plan["qualification"]
    run_records = []
    bundle_contents = []
    qualification_summaries = []
    for run in plan["runs"]:
        run_id = run["run_id"]
        prepared_roots = {}
        for field_name in ("candidate_root", "adapter_root", "build_root", "cache_root"):
            prepared = _inside(
                root,
                root / _safe_relative_path(run[field_name], field_name),
                field_name,
            )
            if not prepared.is_dir():
                raise ConformanceError(f"{run_id} {field_name} is not a real directory")
            prepared_roots[field_name] = prepared
        candidate_root = prepared_roots["candidate_root"]
        if candidate_snapshot_id(candidate_root) != expected.snapshot_id:
            raise ConformanceError(f"{run_id} candidate tree changed after evidence preparation")
        qualification_dir = _inside(
            root,
            root / _safe_relative_path(run["qualification_dir"], "qualification_dir"),
            "qualification_dir",
        )
        bundle_record, contents = _bundle_record(qualification_dir)
        qualification = _load_json_object(
            qualification_dir / "qualification.json", f"{run_id} qualification"
        )
        suite = qualification.get("suite")
        campaign_payload = qualification.get("campaign")
        if not isinstance(campaign_payload, dict):
            raise ConformanceError(f"{run_id} qualification campaign must be an object")
        if not isinstance(suite, dict) or not isinstance(suite.get("report"), dict):
            raise ConformanceError(f"{run_id} qualification suite report must be an object")
        suite_report_payload = suite["report"]
        canonical_artifacts = {
            "qualification.json": _json_bytes(qualification),
            "campaign.json": _json_bytes(campaign_payload),
            "suite.json": _json_bytes(suite_report_payload),
            "qualification.xml": render_junit(qualification).encode("utf-8"),
        }
        for name, canonical in canonical_artifacts.items():
            if contents[name] != canonical:
                raise ConformanceError(
                    f"{run_id} {name} is not the canonical artifact for qualification.json"
                )
        summary = _validate_qualification(qualification, expected, parameters, run_id)
        run_records.append(
            {
                "run_id": run_id,
                "candidate_root": run["candidate_root"],
                "adapter_root": run["adapter_root"],
                "build_root": run["build_root"],
                "cache_root": run["cache_root"],
                "qualification_dir": run["qualification_dir"],
                "candidate_snapshot_id": expected.snapshot_id,
                **bundle_record,
                **summary,
            }
        )
        bundle_contents.append(contents)
        qualification_summaries.append(summary)
    if qualification_summaries[0] != qualification_summaries[1]:
        raise ConformanceError("qualification terminal result or deterministic identifiers differ")
    for name in _BUNDLE_FILES:
        if bundle_contents[0][name] != bundle_contents[1][name]:
            raise ConformanceError(f"qualification artifact bytes differ between runs: {name}")
    unsigned: Dict[str, Any] = {
        "artifact_type": "tracebook.conformance.evidence-manifest",
        "schema_version": EVIDENCE_MANIFEST_VERSION,
        "status": "qualified",
        "plan_id": plan["plan_id"],
        "candidate": plan["candidate"],
        "qualification": parameters,
        "equality": {
            "terminal_result": True,
            "candidate_metadata": True,
            "campaign_id": True,
            "qualification_id": True,
            "counts_and_coverage": True,
            "artifact_bytes": True,
        },
        "runs": run_records,
    }
    return {**unsigned, "manifest_id": _content_id(unsigned)}


def write_evidence_manifest(
    plan_path: str | Path,
    output: Optional[str | Path] = None,
) -> Path:
    """Verify one plan and exclusively write its grading-ready manifest."""
    resolved_plan, _ = load_evidence_plan(plan_path)
    root = resolved_plan.parent.resolve()
    destination = (
        root / MANIFEST_FILENAME
        if output is None
        else _inside(
            root,
            Path(output) if Path(output).is_absolute() else root / Path(output),
            "manifest output",
        )
    )
    if destination.parent != root:
        raise ConformanceError("evidence manifest must be written at workspace root")
    if destination == resolved_plan:
        raise ConformanceError("evidence manifest must not overwrite the evidence plan")
    manifest = verify_evidence_plan(resolved_plan)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as handle:
            handle.write(_json_bytes(manifest))
    except FileExistsError as exc:
        raise ConformanceError(f"evidence manifest already exists: {destination}") from exc
    except OSError as exc:
        raise ConformanceError(f"could not write evidence manifest {destination}: {exc}") from exc
    return destination
