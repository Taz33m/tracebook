"""Profile-scoped qualification for external matching engines."""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping, Tuple

from .campaign import (
    CampaignProfile,
    CampaignResult,
    _CampaignOutputReservation,
    _validated_campaign_parameters,
    get_campaign_profile,
    run_campaign,
)
from .junit import render_junit
from .model import ARTIFACT_SCHEMA_VERSION, ConformanceError
from .protocol import AdapterFactory
from .suite import (
    BUNDLED_SUITE_VERSIONS,
    LATEST_BUNDLED_SUITE_VERSION,
    ConformanceSuite,
    copy_bundled_conformance_suite,
    run_conformance_suite,
)

QUALIFICATION_VERSION = 1

# Qualification is profile-scoped. STP and pro-rata remain available through the
# complete fixed suite but cannot fail an engine that claims only a FIFO profile.
_PROFILE_CASES: Mapping[str, Tuple[str, ...]] = {
    "fifo-limit-v1": (
        "fifo-lifecycle",
        "tick-grid",
        "deep-cancellation",
    ),
    "fifo-full-v1": (
        "fifo-lifecycle",
        "order-instructions",
        "multi-symbol",
        "tick-grid",
        "deep-cancellation",
    ),
    "fifo-partial-fill-v1": (
        "fifo-lifecycle",
        "tick-grid",
        "deep-cancellation",
    ),
}


def qualification_case_names(profile: str | CampaignProfile) -> Tuple[str, ...]:
    """Return qualification-v1 fixed cases for one declared profile."""
    selected = get_campaign_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected, CampaignProfile):
        raise ConformanceError("profile must be a campaign profile or profile name")
    try:
        return _PROFILE_CASES[selected.name]
    except KeyError as exc:
        raise ConformanceError(
            f"campaign profile {selected.name!r} has no qualification-v1 contract"
        ) from exc


def _selected_suite(suite: ConformanceSuite, case_names: Tuple[str, ...]) -> ConformanceSuite:
    by_name = {case.name: case for case in suite.cases}
    missing = [name for name in case_names if name not in by_name]
    if missing:
        raise ConformanceError(
            f"suite {suite.suite_id!r} is missing qualification cases: {', '.join(missing)}"
        )
    return replace(suite, cases=tuple(by_name[name] for name in case_names))


@dataclass(frozen=True)
class QualificationResult:
    """One fixed-plus-generated profile qualification artifact."""

    profile: CampaignProfile
    suite_id: str
    suite_hash: str
    selected_cases: Tuple[str, ...]
    suite_report: dict
    campaign: CampaignResult

    @property
    def fixed_cases_passed(self) -> int:
        return int(self.suite_report["conformant_cases"])

    @property
    def coverage_complete(self) -> bool:
        return not self.campaign.semantic_coverage.uncovered_capabilities

    @property
    def qualified(self) -> bool:
        return bool(self.suite_report["conformant"] and self.campaign.conformant) and (
            self.coverage_complete
        )

    def _identity_payload(self) -> dict:
        return {
            "qualification_version": QUALIFICATION_VERSION,
            "profile": self.profile.to_dict(),
            "suite_id": self.suite_id,
            "suite_hash": self.suite_hash,
            "selected_cases": list(self.selected_cases),
            "suite_report": self.suite_report,
            "campaign": self.campaign.to_dict(),
        }

    @property
    def qualification_id(self) -> str:
        encoded = json.dumps(
            self._identity_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict:
        coverage = self.campaign.semantic_coverage
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.qualification",
            "qualification_version": QUALIFICATION_VERSION,
            "qualification_id": self.qualification_id,
            "qualified": self.qualified,
            "candidate_engine": self.campaign.candidate_engine.to_dict(),
            "profile": self.profile.to_dict(),
            "suite": {
                "suite_id": self.suite_id,
                "suite_hash": self.suite_hash,
                "selection_version": QUALIFICATION_VERSION,
                "selected_cases": list(self.selected_cases),
                "report": self.suite_report,
            },
            "campaign": self.campaign.to_dict(),
            "checks": {
                "fixed_cases": {
                    "passed": self.fixed_cases_passed,
                    "total": len(self.selected_cases),
                    "complete": bool(self.suite_report["conformant"]),
                },
                "generated_campaign": {
                    "completed_traces": len(self.campaign.traces),
                    "requested_traces": self.campaign.requested_traces,
                    "conformant": self.campaign.conformant,
                },
                "semantic_coverage": {
                    "covered": len(coverage.covered_capabilities),
                    "expected": len(coverage.expected_capabilities),
                    "uncovered": list(coverage.uncovered_capabilities),
                    "complete": self.coverage_complete,
                },
            },
            "candidate_runs": len(self.selected_cases)
            + int(self.campaign.to_dict()["candidate_runs"]),
            "paths": {
                "suite": "suite.json",
                "campaign": "campaign.json",
                "junit": "qualification.xml",
                "reduced": "reduced.jsonl" if self.campaign.failure is not None else None,
            },
        }


def run_qualification(
    candidate_factory: AdapterFactory,
    profile: str | CampaignProfile = "fifo-limit-v1",
    *,
    suite_version: str = LATEST_BUNDLED_SUITE_VERSION,
    seed: int = 1337,
    traces: int = 25,
    events_per_trace: int = 200,
    max_minimize_runs: int = 100,
) -> QualificationResult:
    """Run profile-relevant fixed cases followed by a generated campaign."""
    selected_profile = get_campaign_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected_profile, CampaignProfile):
        raise ConformanceError("profile must be a campaign profile or profile name")
    if suite_version not in BUNDLED_SUITE_VERSIONS:
        versions = ", ".join(BUNDLED_SUITE_VERSIONS)
        raise ConformanceError(
            f"unknown bundled suite version {suite_version!r}; expected one of: {versions}"
        )
    case_names = qualification_case_names(selected_profile)
    seed, traces, events_per_trace, max_minimize_runs = _validated_campaign_parameters(
        seed,
        traces,
        events_per_trace,
        max_minimize_runs,
    )

    with tempfile.TemporaryDirectory(prefix="tracebook-qualification-") as temporary:
        suite = copy_bundled_conformance_suite(
            Path(temporary) / suite_version,
            suite_version=suite_version,
        )
        suite_report = run_conformance_suite(
            _selected_suite(suite, case_names),
            candidate_factory,
        )

    campaign = run_campaign(
        candidate_factory,
        profile=selected_profile,
        seed=seed,
        traces=traces,
        events_per_trace=events_per_trace,
        max_minimize_runs=max_minimize_runs,
    )
    if suite_report["candidate_engine"] != campaign.candidate_engine.to_dict():
        raise ConformanceError("candidate engine metadata changed between suite and campaign")
    return QualificationResult(
        profile=selected_profile,
        suite_id=suite.suite_id,
        suite_hash=suite.suite_hash,
        selected_cases=case_names,
        suite_report=suite_report,
        campaign=campaign,
    )


def _json_bytes(payload: dict) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


class _QualificationOutputReservation:
    """Reserve a qualification directory before candidate processes start."""

    def __init__(self, destination: str | Path) -> None:
        self.target = Path(destination).expanduser()
        self._campaign = _CampaignOutputReservation(self.target)

    def __enter__(self) -> "_QualificationOutputReservation":
        self._campaign.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._campaign.__exit__(exc_type, exc_value, traceback)

    def write(self, result: QualificationResult) -> Path:
        """Commit qualification, suite, campaign, JUnit, and failure artifacts."""
        if not isinstance(result, QualificationResult):
            raise ConformanceError("result must be a QualificationResult")
        payload = result.to_dict()
        self._campaign.write(
            result.campaign,
            extra_files={
                "qualification.json": _json_bytes(payload),
                "suite.json": _json_bytes(result.suite_report),
                "qualification.xml": render_junit(payload).encode("utf-8"),
            },
        )
        return self.target / "qualification.json"


def write_qualification_artifacts(
    result: QualificationResult,
    destination: str | Path,
) -> Path:
    """Atomically commit one profile qualification evidence bundle."""
    if not isinstance(result, QualificationResult):
        raise ConformanceError("result must be a QualificationResult")
    with _QualificationOutputReservation(destination) as reservation:
        return reservation.write(result)
