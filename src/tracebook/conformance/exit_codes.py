"""Central CLI exit-code classification for conformance artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .model import ConformanceError

SUCCESS = 0
SEMANTIC_DIVERGENCE = 1
OPERATIONAL_FAILURE = 2


def _divergence_exit_code(divergence: Any) -> int:
    if not isinstance(divergence, Mapping):
        return SUCCESS
    if (
        divergence.get("category") == "protocol"
        or divergence.get("snapshot_error")
        or divergence.get("close_error")
    ):
        return OPERATIONAL_FAILURE
    return SEMANTIC_DIVERGENCE


def _conformance_report_exit_code(artifact: Mapping[str, Any]) -> int:
    divergence_code = _divergence_exit_code(artifact.get("divergence"))
    if divergence_code != SUCCESS:
        return divergence_code
    return SUCCESS if artifact.get("conformant") is True else SEMANTIC_DIVERGENCE


def _suite_exit_code(artifact: Mapping[str, Any]) -> int:
    case_codes = []
    for case in artifact.get("cases", ()):
        if isinstance(case, Mapping) and isinstance(case.get("report"), Mapping):
            case_codes.append(_conformance_report_exit_code(case["report"]))
    if OPERATIONAL_FAILURE in case_codes:
        return OPERATIONAL_FAILURE
    return SUCCESS if artifact.get("conformant") is True else SEMANTIC_DIVERGENCE


def _campaign_exit_code(artifact: Mapping[str, Any]) -> int:
    trace_codes = []
    for trace in artifact.get("traces", ()):
        if isinstance(trace, Mapping):
            trace_codes.append(_divergence_exit_code(trace.get("divergence")))
    if OPERATIONAL_FAILURE in trace_codes:
        return OPERATIONAL_FAILURE
    return SUCCESS if artifact.get("conformant") is True else SEMANTIC_DIVERGENCE


def _qualification_exit_code(artifact: Mapping[str, Any]) -> int:
    suite = artifact.get("suite")
    campaign = artifact.get("campaign")
    nested_codes = []
    if isinstance(suite, Mapping) and isinstance(suite.get("report"), Mapping):
        nested_codes.append(_suite_exit_code(suite["report"]))
    if isinstance(campaign, Mapping):
        nested_codes.append(_campaign_exit_code(campaign))
    if OPERATIONAL_FAILURE in nested_codes:
        return OPERATIONAL_FAILURE
    return SUCCESS if artifact.get("qualified") is True else SEMANTIC_DIVERGENCE


def _reproduction_exit_code(artifact: Mapping[str, Any]) -> int:
    report = artifact.get("conformance_report")
    if isinstance(report, Mapping):
        report_code = _conformance_report_exit_code(report)
        if report_code == OPERATIONAL_FAILURE:
            return OPERATIONAL_FAILURE
    return SUCCESS if artifact.get("reproduced") is True else SEMANTIC_DIVERGENCE


def _minimization_exit_code(artifact: Mapping[str, Any]) -> int:
    report = artifact.get("conformance_report")
    if isinstance(report, Mapping):
        report_code = _conformance_report_exit_code(report)
        if report_code == OPERATIONAL_FAILURE:
            return OPERATIONAL_FAILURE
    return SUCCESS


def exit_code_for_artifact(artifact: Mapping[str, Any]) -> int:
    """Return the documented CLI status for one generated artifact.

    Protocol and adapter failures always take precedence over semantic result
    status. Reproducing or minimizing an expected semantic divergence remains a
    successful operation, while an adapter failure cannot count as success.
    """
    if not isinstance(artifact, Mapping):
        raise ConformanceError("exit-code classification requires an artifact object")
    artifact_type = artifact.get("artifact_type")
    classifiers = {
        "tracebook.conformance.report": _conformance_report_exit_code,
        "tracebook.conformance.suite_report": _suite_exit_code,
        "tracebook.conformance.campaign": _campaign_exit_code,
        "tracebook.conformance.qualification": _qualification_exit_code,
        "tracebook.conformance.reproduction": _reproduction_exit_code,
        "tracebook.conformance.minimization": _minimization_exit_code,
    }
    if not isinstance(artifact_type, str):
        raise ConformanceError(f"cannot classify artifact type {artifact_type!r}")
    try:
        classifier = classifiers[artifact_type]
    except KeyError as exc:
        raise ConformanceError(f"cannot classify artifact type {artifact_type!r}") from exc
    return classifier(artifact)
