"""Deterministically reduce a failing trace while preserving its failure class."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from numbers import Integral
from typing import Iterable, Optional, Tuple, cast

from ..events import MarketEvent, replay_market_events
from .classification import (
    FailureSignature,
    failure_signature,
    preserves_failure_signature,
)
from .compare import ConformanceReport, Divergence, run_conformance
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    trace_sha256,
)
from .protocol import AdapterFactory

DEFAULT_MINIMIZE_RUNS = 100


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
    max_runs: int | None = None

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
            **({"max_runs": self.max_runs} if self.max_runs is not None else {}),
            "target_category": self.target_category,
            "one_minimal": self.one_minimal,
            "budget_exhausted": self.budget_exhausted,
            "minimized_trace_sha256": trace_sha256(self.events),
            "conformance_report": self.report.to_dict(),
        }


def _reject_protocol_failure_during_semantic_minimization(
    report: ConformanceReport,
    target: FailureSignature,
    observed: FailureSignature | None,
    run_number: int,
) -> None:
    """Do not mistake an operational trial failure for a removable semantic subset."""
    divergence = report.divergence
    if not target.operational and observed is not None and observed.operational:
        if divergence is None:
            raise ConformanceError("operational failure has no divergence")
        kind = "adapter_close_error" if divergence.close_error is not None else divergence.kind
        message = divergence.close_error or divergence.message
        raise ConformanceError(
            "candidate protocol failure during semantic minimization "
            f"run {run_number}: {kind}: {message}"
        )


def _order_lifecycles(
    events: tuple[MarketEvent, ...], mismatch: Divergence, config: ConformanceConfig
) -> tuple[MarketEvent, ...]:
    """Select the disputed orders and their direct reference-trade counterparties."""
    trigger = events[-1]
    disputed_ids = {mismatch.reference, mismatch.candidate}
    related_ids = disputed_ids | {trigger.order_id}
    replay = replay_market_events(
        events,
        strict=False,
        matching_algorithm=config.matching_algorithm,
        tick_size=config.tick_size,
        self_trade_policy=config.self_trade_policy,
    )
    for trade in replay.trades:
        if trade.symbol == trigger.symbol and (
            trade.buy_order_id in disputed_ids or trade.sell_order_id in disputed_ids
        ):
            related_ids.update((trade.buy_order_id, trade.sell_order_id))
    return tuple(
        event
        for event in events
        if event.symbol == trigger.symbol and (event.op == "clear" or event.order_id in related_ids)
    )


def minimize_failing_trace(
    events: Iterable[MarketEvent],
    candidate_factory: AdapterFactory,
    config: Optional[ConformanceConfig] = None,
    max_runs: int = DEFAULT_MINIMIZE_RUNS,
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

    target = cast(FailureSignature, failure_signature(original, initial))
    _reject_protocol_failure_during_semantic_minimization(initial, target, target, runs)
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

    def check_trial(trial: tuple[MarketEvent, ...]) -> ConformanceReport | None:
        nonlocal runs
        report = run_conformance(trial, candidate_factory, config)
        runs += 1
        if report.candidate_engine != candidate_engine:
            raise ConformanceError("candidate engine metadata changed during minimization")
        observed = failure_signature(trial, report)
        _reject_protocol_failure_during_semantic_minimization(report, target, observed, runs)
        return report if preserves_failure_signature(target, observed) else None

    mismatch = initial.divergence
    if (
        runs < max_runs
        and not target.operational
        and mismatch.category == "trades"
        and mismatch.path.endswith((".buy_order_id", ".sell_order_id"))
    ):
        trial = _order_lifecycles(current, mismatch, config)
        if len(trial) < len(current):
            report = check_trial(trial)
            if report is not None and report.divergence is not None:
                observed_drift = report.divergence
                # Removing unrelated fills can shift the trade index within this event.
                if (
                    observed_drift.event_index == len(trial)
                    and observed_drift.path.rsplit(".", 1)[-1] == mismatch.path.rsplit(".", 1)[-1]
                    and (observed_drift.reference, observed_drift.candidate)
                    == (mismatch.reference, mismatch.candidate)
                ):
                    current, current_report = trial, report

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
            report = check_trial(trial)
            if report is not None:
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
        empty_report = check_trial(())
        if empty_report is not None:
            current = ()
            current_report = empty_report
        one_minimal = True

    return MinimizationResult(
        events=current,
        original_event_count=len(original),
        runs=runs,
        max_runs=max_runs,
        target_category=target.category,
        one_minimal=one_minimal,
        budget_exhausted=runs >= max_runs and not one_minimal,
        report=current_report,
    )
