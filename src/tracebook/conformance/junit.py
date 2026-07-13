"""JUnit XML rendering for conformance commands."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

# ElementTree is only a serializer here; no XML input is parsed.
from xml.etree import ElementTree  # nosec B405

from .model import ConformanceError


def _failure_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)


def _add_case(
    suite: ElementTree.Element,
    name: str,
    failed: bool,
    details: Mapping[str, Any] | None = None,
) -> None:
    case = ElementTree.SubElement(
        suite,
        "testcase",
        {"classname": "tracebook.conformance", "name": name},
    )
    if failed:
        details = details or {}
        failure = ElementTree.SubElement(
            case,
            "failure",
            {
                "type": str(details.get("category", "semantic_divergence")),
                "message": str(details.get("message", "matching semantics diverged")),
            },
        )
        failure.text = _failure_text(details)


def _report_case(payload: Mapping[str, Any]) -> tuple[str, bool, Mapping[str, Any] | None]:
    trace = payload.get("trace", {})
    divergence = payload.get("divergence")
    name = str(trace.get("name") or trace.get("sha256") or "trace")
    return name, not bool(payload.get("conformant")), divergence


def render_junit(payload: Mapping[str, Any]) -> str:
    """Render one JSON artifact as deterministic JUnit XML."""
    if not isinstance(payload, Mapping):
        raise ConformanceError("JUnit payload must be an object")
    artifact_type = payload.get("artifact_type")
    suite = ElementTree.Element(
        "testsuite",
        {"name": str(artifact_type or "tracebook.conformance")},
    )
    failures = 0
    tests = 0

    if artifact_type == "tracebook.conformance.report":
        name, failed, details = _report_case(payload)
        _add_case(suite, name, failed, details)
        tests = 1
        failures = int(failed)
    elif artifact_type == "tracebook.conformance.suite_report":
        for case_payload in payload.get("cases", []):
            report = case_payload["report"]
            failed = not bool(report.get("conformant"))
            _add_case(suite, str(case_payload["name"]), failed, report.get("divergence"))
            tests += 1
            failures += int(failed)
    elif artifact_type == "tracebook.conformance.campaign":
        failure_payload = payload.get("failure") or {}
        coverage = payload.get("semantic_coverage")
        if isinstance(coverage, Mapping):
            properties = ElementTree.SubElement(suite, "properties")
            for key in ("coverage_ratio", "covered_count", "expected_count", "compared_events"):
                ElementTree.SubElement(
                    properties,
                    "property",
                    {"name": f"semantic_coverage.{key}", "value": str(coverage.get(key))},
                )
        for trace in payload.get("traces", []):
            failed = not bool(trace.get("conformant"))
            details = dict(trace.get("divergence") or {})
            if failed and failure_payload:
                details["failure_class"] = failure_payload.get("failure_class")
                details["failure_id"] = failure_payload.get("failure_id")
            _add_case(suite, f"trace-{trace.get('index')}", failed, details)
            tests += 1
            failures += int(failed)
    elif artifact_type == "tracebook.conformance.minimization":
        _add_case(suite, "trace-minimization", False)
        tests = 1
    elif artifact_type == "tracebook.conformance.reproduction":
        failed = not bool(payload.get("reproduced"))
        details = {
            "category": "reproduction_mismatch",
            "message": "stored failure did not reproduce exactly",
            "expected": payload.get("expected"),
            "observed": payload.get("observed"),
        }
        _add_case(suite, str(payload.get("failure_id") or "reproducer"), failed, details)
        tests = 1
        failures = int(failed)
    else:
        raise ConformanceError(f"unsupported JUnit artifact type: {artifact_type!r}")

    suite.set("tests", str(tests))
    suite.set("failures", str(failures))
    suite.set("errors", "0")
    return ElementTree.tostring(suite, encoding="unicode", xml_declaration=True) + "\n"


def write_junit(payload: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a JUnit report, creating only its parent directories."""
    path = Path(destination).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_junit(payload), encoding="utf-8")
    return path
