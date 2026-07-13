"""Deterministically reduce a failing trace while preserving its failure class."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Iterable, Optional, Tuple

from ..events import MarketEvent
from .compare import ConformanceReport, run_conformance
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    trace_sha256,
)
from .protocol import AdapterFactory


@dataclass(frozen=True)
class MinimizationResult:
    """A reduced event trace and the report proving it still fails."""

    events: Tuple[MarketEvent, ...]
    original_event_count: int
    runs: int
    target_category: str
    one_minimal: bool
    budget_exhausted: bool
    report: ConformanceReport

    def to_dict(self) -> dict:
        minimized_count = len(self.events)
        reduction_percent = (
            100.0 * (self.original_event_count - minimized_count) / self.original_event_count
            if self.original_event_count
            else 0.0
        )
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.minimization",
            "original_event_count": self.original_event_count,
            "minimized_event_count": minimized_count,
            "reduction_percent": reduction_percent,
            "runs": self.runs,
            "target_category": self.target_category,
            "one_minimal": self.one_minimal,
            "budget_exhausted": self.budget_exhausted,
            "minimized_trace_sha256": trace_sha256(self.events),
            "conformance_report": self.report.to_dict(),
        }


def minimize_failing_trace(
    events: Iterable[MarketEvent],
    candidate_factory: AdapterFactory,
    config: Optional[ConformanceConfig] = None,
    max_runs: int = 100,
    trace_name: Optional[str] = None,
    expected_candidate_engine: Optional[EngineMetadata] = None,
) -> MinimizationResult:
    """Use delta debugging to reduce a failure and report minimality honestly."""
    if isinstance(max_runs, bool) or not isinstance(max_runs, Integral) or max_runs <= 0:
        raise ConformanceError("max_runs must be a positive integer")
    max_runs = int(max_runs)
    config = config or ConformanceConfig()
    original = tuple(events)
    initial = run_conformance(original, candidate_factory, config, trace_name=trace_name)
    runs = 1
    if initial.conformant or initial.divergence is None:
        raise ConformanceError("trace is conformant and cannot be minimized")
    candidate_engine = expected_candidate_engine or initial.candidate_engine
    if initial.candidate_engine != candidate_engine:
        raise ConformanceError("candidate engine metadata changed during minimization")

    target = initial.divergence.category
    prefix_length = initial.divergence.event_index
    current = original[:prefix_length] if prefix_length > 0 else original
    current_report = (
        replace(
            initial,
            trace_name=None,
            trace_hash=trace_sha256(current),
            event_count=len(current),
        )
        if current != original
        else initial
    )
    granularity = 2
    one_minimal = len(current) == 0

    while len(current) >= 2 and runs < max_runs:
        chunk_size = int(math.ceil(len(current) / granularity))
        reduced = False
        completed_round = True
        for start in range(0, len(current), chunk_size):
            if runs >= max_runs:
                completed_round = False
                break
            trial = current[:start] + current[start + chunk_size :]
            if not trial:
                continue
            report = run_conformance(trial, candidate_factory, config)
            runs += 1
            if report.candidate_engine != candidate_engine:
                raise ConformanceError("candidate engine metadata changed during minimization")
            if (
                not report.conformant
                and report.divergence is not None
                and report.divergence.category == target
            ):
                current = trial
                current_report = report
                granularity = max(2, granularity - 1)
                reduced = True
                break
        if reduced:
            one_minimal = len(current) == 0
            continue
        if not completed_round:
            break
        if granularity >= len(current):
            one_minimal = True
            break
        granularity = min(len(current), granularity * 2)

    if len(current) == 1 and not one_minimal and runs < max_runs:
        empty_report = run_conformance((), candidate_factory, config)
        runs += 1
        if empty_report.candidate_engine != candidate_engine:
            raise ConformanceError("candidate engine metadata changed during minimization")
        if (
            not empty_report.conformant
            and empty_report.divergence is not None
            and empty_report.divergence.category == target
        ):
            current = ()
            current_report = empty_report
        one_minimal = True

    return MinimizationResult(
        events=current,
        original_event_count=len(original),
        runs=runs,
        target_category=target,
        one_minimal=one_minimal,
        budget_exhausted=runs >= max_runs and not one_minimal,
        report=current_report,
    )
