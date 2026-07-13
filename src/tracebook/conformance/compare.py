"""Run one trace against Tracebook and a candidate engine, then localize drift."""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ..events import MarketEvent
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    PROTOCOL_VERSION,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
    first_difference,
    trace_sha256,
)
from .protocol import AdapterFactory, EngineAdapter
from .reference import ReferenceEngineAdapter


@dataclass(frozen=True)
class Divergence:
    """First observable disagreement in a trace."""

    event_index: int
    category: str
    kind: str
    path: str
    message: str
    event: Optional[dict]
    reference: Any
    candidate: Any
    snapshot_error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "event_index": self.event_index,
            "category": self.category,
            "kind": self.kind,
            "path": self.path,
            "message": self.message,
            "event": self.event,
            "reference": self.reference,
            "candidate": self.candidate,
            "snapshot_error": self.snapshot_error,
        }


@dataclass(frozen=True)
class ConformanceReport:
    """Stable JSON artifact for one candidate/reference comparison."""

    config: ConformanceConfig
    trace_name: Optional[str]
    trace_hash: str
    event_count: int
    compared_events: int
    reference_engine: EngineMetadata
    candidate_engine: EngineMetadata
    conformant: bool
    final_state_hash: Optional[str]
    divergence: Optional[Divergence]

    def to_dict(self) -> dict:
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.report",
            "protocol_version": PROTOCOL_VERSION,
            "trace": {
                "name": self.trace_name,
                "sha256": self.trace_hash,
                "event_count": self.event_count,
            },
            "config": self.config.to_dict(),
            "reference_engine": self.reference_engine.to_dict(),
            "candidate_engine": self.candidate_engine.to_dict(),
            "compared_events": self.compared_events,
            "conformant": self.conformant,
            "final_state_hash": self.final_state_hash,
            "divergence": self.divergence.to_dict() if self.divergence else None,
        }


def _simple_divergence(
    event: MarketEvent,
    index: int,
    category: str,
    difference: dict,
    message: str,
) -> Divergence:
    return Divergence(
        event_index=index,
        category=category,
        kind=difference["kind"],
        path=difference["path"],
        message=message,
        event=event.to_dict(),
        reference=difference.get("reference"),
        candidate=difference.get("candidate"),
    )


def _compare_observations(
    event: MarketEvent,
    index: int,
    reference: Observation,
    candidate: Observation,
) -> Optional[Divergence]:
    reference_outcome = {
        "status": reference.outcome.status,
        "reason": reference.outcome.reason,
    }
    candidate_outcome = {
        "status": candidate.outcome.status,
        "reason": candidate.outcome.reason,
    }
    difference = first_difference(reference_outcome, candidate_outcome, "$.observation.outcome")
    if difference is not None:
        return _simple_divergence(
            event,
            index,
            "outcome",
            difference,
            "candidate applied/rejected semantics differ from the reference",
        )

    reference_trades = [trade.to_dict() for trade in reference.trades]
    candidate_trades = [trade.to_dict() for trade in candidate.trades]
    difference = first_difference(reference_trades, candidate_trades, "$.observation.trades")
    if difference is not None:
        return _simple_divergence(
            event,
            index,
            "trades",
            difference,
            "candidate executions differ from the reference",
        )

    if reference.resting_order_count != candidate.resting_order_count:
        return Divergence(
            event_index=index,
            category="book_state",
            kind="value_mismatch",
            path="$.observation.resting_order_count",
            message="candidate resting-order count differs from the reference",
            event=event.to_dict(),
            reference=reference.resting_order_count,
            candidate=candidate.resting_order_count,
        )
    if reference.state_hash != candidate.state_hash:
        return Divergence(
            event_index=index,
            category="book_state",
            kind="value_mismatch",
            path="$.observation.state_hash",
            message="candidate resting state differs from the reference",
            event=event.to_dict(),
            reference=reference.state_hash,
            candidate=candidate.state_hash,
        )
    return None


def _localize_state_divergence(
    divergence: Divergence,
    reference: ReferenceEngineAdapter,
    candidate: EngineAdapter,
    candidate_observation: Optional[Observation],
) -> Divergence:
    try:
        reference_state = reference.snapshot()
        candidate_state = candidate.snapshot()
        if not isinstance(candidate_state, BookState):
            raise ConformanceError("candidate snapshot must return BookState")
    except Exception as exc:
        return Divergence(
            **{
                **divergence.__dict__,
                "snapshot_error": f"candidate snapshot failed: {exc}",
            }
        )

    candidate_digest = candidate_state.digest()
    if candidate_observation is not None and candidate_digest != candidate_observation.state_hash:
        return Divergence(
            event_index=divergence.event_index,
            category="protocol",
            kind="invalid_state_hash",
            path="$.observation.state_hash",
            message="candidate state_hash does not describe its snapshot",
            event=divergence.event,
            reference=candidate_digest,
            candidate=candidate_observation.state_hash,
        )

    difference = first_difference(reference_state.to_dict(), candidate_state.to_dict(), "$.state")
    if difference is None:
        return divergence
    return Divergence(
        event_index=divergence.event_index,
        category="book_state",
        kind=difference["kind"],
        path=difference["path"],
        message="candidate resting orders or queue priority differ from the reference",
        event=divergence.event,
        reference=difference.get("reference"),
        candidate=difference.get("candidate"),
    )


def _protocol_divergence(event: MarketEvent, index: int, exc: Exception) -> Divergence:
    return Divergence(
        event_index=index,
        category="protocol",
        kind="adapter_error",
        path="$",
        message=str(exc),
        event=event.to_dict(),
        reference="valid observation",
        candidate=None,
    )


def run_conformance(
    events: Iterable[MarketEvent],
    candidate_factory: AdapterFactory,
    config: Optional[ConformanceConfig] = None,
    trace_name: Optional[str] = None,
) -> ConformanceReport:
    """Compare a candidate adapter to Tracebook and stop at the first drift."""
    normalized_events = list(events)
    if any(not isinstance(event, MarketEvent) for event in normalized_events):
        raise ConformanceError("conformance traces must contain only MarketEvent values")
    config = config or ConformanceConfig()
    reference = ReferenceEngineAdapter(config)
    candidate = candidate_factory(config)
    candidate_metadata = getattr(candidate, "metadata", None)
    if not isinstance(candidate_metadata, EngineMetadata):
        with suppress(Exception):
            candidate.close()
        reference.close()
        raise ConformanceError("candidate adapter metadata must be EngineMetadata")
    report_hash = trace_sha256(normalized_events)
    divergence: Optional[Divergence] = None
    compared_events = 0
    final_state_hash: Optional[str] = None
    last_candidate_observation: Optional[Observation] = None
    close_error: Optional[Exception] = None

    try:
        for index, event in enumerate(normalized_events, 1):
            reference_observation = reference.apply(event, index)
            try:
                candidate_observation = candidate.apply(event, index)
                if not isinstance(candidate_observation, Observation):
                    raise ConformanceError("candidate apply() must return Observation")
                if candidate_observation.index != index:
                    raise ConformanceError(
                        f"candidate observation index {candidate_observation.index} "
                        f"does not match event {index}"
                    )
            except Exception as exc:
                divergence = _protocol_divergence(event, index, exc)
                compared_events = index
                break
            last_candidate_observation = candidate_observation
            compared_events = index
            divergence = _compare_observations(
                event, index, reference_observation, candidate_observation
            )
            if divergence is not None:
                if divergence.category == "book_state":
                    divergence = _localize_state_divergence(
                        divergence,
                        reference,
                        candidate,
                        candidate_observation,
                    )
                break

        if divergence is None:
            reference_state = reference.snapshot()
            final_state_hash = reference_state.digest()
            try:
                candidate_state = candidate.snapshot()
                if not isinstance(candidate_state, BookState):
                    raise ConformanceError("candidate snapshot must return BookState")
                candidate_digest = candidate_state.digest()
                if (
                    last_candidate_observation is not None
                    and candidate_digest != last_candidate_observation.state_hash
                ):
                    hashed_event = normalized_events[-1]
                    divergence = Divergence(
                        event_index=len(normalized_events),
                        category="protocol",
                        kind="invalid_state_hash",
                        path="$.observation.state_hash",
                        message="candidate final state_hash does not describe its snapshot",
                        event=hashed_event.to_dict(),
                        reference=candidate_digest,
                        candidate=last_candidate_observation.state_hash,
                    )
                else:
                    difference = first_difference(
                        reference_state.to_dict(), candidate_state.to_dict(), "$.state"
                    )
                    if difference is not None:
                        state_event = normalized_events[-1] if normalized_events else None
                        divergence = Divergence(
                            event_index=len(normalized_events),
                            category="book_state",
                            kind=difference["kind"],
                            path=difference["path"],
                            message="candidate final state differs from the reference",
                            event=state_event.to_dict() if state_event is not None else None,
                            reference=difference.get("reference"),
                            candidate=difference.get("candidate"),
                        )
            except Exception as exc:
                failed_event = normalized_events[-1] if normalized_events else None
                divergence = Divergence(
                    event_index=len(normalized_events),
                    category="protocol",
                    kind="adapter_error",
                    path="$.state",
                    message=f"candidate final snapshot failed: {exc}",
                    event=failed_event.to_dict() if failed_event is not None else None,
                    reference="valid final snapshot",
                    candidate=None,
                )
    finally:
        try:
            candidate.close()
        except Exception as exc:
            close_error = exc
        reference.close()

    if close_error is not None and divergence is None:
        last_event = normalized_events[-1] if normalized_events else None
        divergence = Divergence(
            event_index=len(normalized_events),
            category="protocol",
            kind="adapter_close_error",
            path="$",
            message=f"candidate close failed: {close_error}",
            event=last_event.to_dict() if last_event is not None else None,
            reference="clean adapter shutdown",
            candidate=None,
        )

    return ConformanceReport(
        config=config,
        trace_name=trace_name,
        trace_hash=report_hash,
        event_count=len(normalized_events),
        compared_events=compared_events,
        reference_engine=reference.metadata,
        candidate_engine=candidate_metadata,
        conformant=divergence is None,
        final_state_hash=final_state_hash,
        divergence=divergence,
    )
