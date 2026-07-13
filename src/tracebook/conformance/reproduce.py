"""Deterministic replay of saved conformance failures."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from ..events import MarketEvent
from .classification import classify_failure
from .compare import ConformanceReport, run_conformance
from .model import ARTIFACT_SCHEMA_VERSION, ConformanceConfig, ConformanceError, trace_sha256
from .protocol import AdapterFactory


@dataclass(frozen=True)
class ReproductionResult:
    """Observed replay result compared with optional corpus expectations."""

    events: Sequence[MarketEvent]
    report: ConformanceReport
    failure_class: str
    reproduced: bool
    expected: Optional[Mapping[str, Any]] = None

    def to_dict(self) -> dict:
        expected = self.expected or {}
        divergence = self.report.divergence.to_dict() if self.report.divergence else None
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.reproduction",
            "failure_id": expected.get("failure_id"),
            "campaign_seed": expected.get("campaign_seed"),
            "campaign_id": expected.get("campaign_id"),
            "original_divergence_event": expected.get("original_divergence_event"),
            "original_event_count": expected.get("original_event_count"),
            "reduced_event_count": len(self.events),
            "reduced_trace_sha256": trace_sha256(self.events),
            "failure_class": self.failure_class,
            "reproduced": self.reproduced,
            "expected": {
                "failure_class": expected.get("failure_class"),
                "divergence": expected.get("expected_reduced_divergence"),
            },
            "observed": {
                "failure_class": self.failure_class,
                "divergence": divergence,
            },
            "conformance_report": self.report.to_dict(),
        }


def load_failure_metadata(path: str | Path) -> Mapping[str, Any]:
    """Load and minimally validate a corpus ``failure.json`` file."""
    metadata_path = Path(path).expanduser()
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ConformanceError(f"failure metadata not found: {metadata_path}") from exc
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"invalid failure metadata JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ConformanceError("failure metadata must be a JSON object")
    if payload.get("artifact_type") != "tracebook.conformance.failure":
        raise ConformanceError("failure metadata has an unsupported artifact_type")
    if payload.get("schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ConformanceError("failure metadata has an unsupported schema_version")
    return payload


def discover_failure_metadata(events_path: str | Path) -> Optional[Mapping[str, Any]]:
    """Load sibling failure metadata when replaying a corpus trace."""
    candidate = Path(events_path).expanduser().parent / "failure.json"
    return load_failure_metadata(candidate) if candidate.is_file() else None


def reproduction_config(
    metadata: Optional[Mapping[str, Any]],
    fallback: ConformanceConfig,
) -> ConformanceConfig:
    """Use the corpus config when metadata is available."""
    if metadata is None:
        return fallback
    config = metadata.get("config")
    if not isinstance(config, Mapping):
        raise ConformanceError("failure metadata config must be an object")
    return ConformanceConfig.from_dict(config)


def run_reproduction(
    events: Sequence[MarketEvent],
    candidate_factory: AdapterFactory,
    config: ConformanceConfig,
    expected: Optional[Mapping[str, Any]] = None,
    trace_name: Optional[str] = None,
) -> ReproductionResult:
    """Replay a reduced trace and require its stored first divergence exactly."""
    event_hash = trace_sha256(events)
    if expected is not None:
        expected_hash = expected.get("reduced_trace_sha256")
        expected_count = expected.get("reduced_event_count")
        if expected_hash != event_hash:
            raise ConformanceError("reduced trace hash does not match failure metadata")
        if isinstance(expected_count, bool) or not isinstance(expected_count, int):
            raise ConformanceError("failure metadata reduced_event_count must be an integer")
        if expected_count != len(events):
            raise ConformanceError("reduced event count does not match failure metadata")
    report = run_conformance(events, candidate_factory, config=config, trace_name=trace_name)
    failure_class = classify_failure(events, report)
    divergence = report.divergence.to_dict() if report.divergence else None
    if expected is None:
        reproduced = divergence is not None
    else:
        expected_class = expected.get("failure_class")
        expected_divergence = expected.get("expected_reduced_divergence")
        if not isinstance(expected_class, str) or not expected_class:
            raise ConformanceError("failure metadata requires failure_class")
        if not isinstance(expected_divergence, Mapping):
            raise ConformanceError("failure metadata requires expected_reduced_divergence")
        reproduced = failure_class == expected_class and divergence == dict(expected_divergence)
    return ReproductionResult(
        events=tuple(events),
        report=report,
        failure_class=failure_class,
        reproduced=reproduced,
        expected=expected,
    )
