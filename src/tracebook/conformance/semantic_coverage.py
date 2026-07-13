"""Candidate-independent semantic coverage for generated campaigns."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Mapping, Sequence, Tuple

from ..events import MarketEvent
from .classification import is_queue_priority_probe
from .model import ConformanceConfig
from .reference import ReferenceEngineAdapter


@dataclass(frozen=True)
class SemanticCoverage:
    """Evidence that a campaign exercised a declared capability profile."""

    expected_capabilities: Tuple[str, ...]
    evidence: Mapping[str, int]
    operations: Mapping[str, int]
    order_types: Mapping[str, int]
    applied_order_types: Mapping[str, int]
    outcomes: Mapping[str, int]
    rejection_reasons: Mapping[str, int]
    symbols: Tuple[str, ...]
    compared_events: int
    trade_events: int
    trades: int
    partial_fill_events: int
    queue_priority_probes: int

    @property
    def covered_capabilities(self) -> Tuple[str, ...]:
        return tuple(name for name in self.expected_capabilities if self.evidence.get(name, 0) > 0)

    @property
    def uncovered_capabilities(self) -> Tuple[str, ...]:
        return tuple(name for name in self.expected_capabilities if self.evidence.get(name, 0) == 0)

    def to_dict(self) -> dict:
        expected_count = len(self.expected_capabilities)
        covered_count = len(self.covered_capabilities)
        return {
            "schema_version": 1,
            "basis": "reference observations for candidate-compared events",
            "expected_capabilities": list(self.expected_capabilities),
            "covered_capabilities": list(self.covered_capabilities),
            "uncovered_capabilities": list(self.uncovered_capabilities),
            "covered_count": covered_count,
            "expected_count": expected_count,
            "coverage_ratio": covered_count / expected_count if expected_count else 1.0,
            "evidence": dict(sorted(self.evidence.items())),
            "operations": dict(sorted(self.operations.items())),
            "order_types": dict(sorted(self.order_types.items())),
            "applied_order_types": dict(sorted(self.applied_order_types.items())),
            "outcomes": dict(sorted(self.outcomes.items())),
            "rejection_reasons": dict(sorted(self.rejection_reasons.items())),
            "symbols": list(self.symbols),
            "compared_events": self.compared_events,
            "trade_events": self.trade_events,
            "trades": self.trades,
            "partial_fill_events": self.partial_fill_events,
            "queue_priority_probes": self.queue_priority_probes,
        }


def measure_semantic_coverage(
    traces: Sequence[Sequence[MarketEvent]],
    config: ConformanceConfig,
    expected_capabilities: Sequence[str],
) -> SemanticCoverage:
    """Measure semantic evidence without consulting candidate behavior."""
    operations: Counter[str] = Counter()
    applied_operations: Counter[str] = Counter()
    order_types: Counter[str] = Counter()
    applied_order_types: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    applied_symbols: set[str] = set()
    trade_events = 0
    trade_count = 0
    partial_fill_events = 0
    queue_priority_probes = 0

    for events in traces:
        reference = ReferenceEngineAdapter(config)
        try:
            for index, event in enumerate(events, 1):
                operations[event.op] += 1
                if event.op == "new":
                    order_types[event.order_type.name] += 1
                observation = reference.apply(event, index)
                outcomes[observation.outcome.status] += 1
                if observation.outcome.status == "applied":
                    applied_operations[event.op] += 1
                    applied_symbols.add(event.symbol)
                    if event.op == "new":
                        applied_order_types[event.order_type.name] += 1
                elif observation.outcome.reason is not None:
                    rejection_reasons[observation.outcome.reason] += 1
                if observation.trades:
                    trade_events += 1
                    trade_count += len(observation.trades)
                    state = reference.snapshot()
                    active_ids = {
                        (book.symbol, order.order_id)
                        for book in state.books
                        for order in book.bids + book.asks
                    }
                    traded_ids = {
                        (trade.symbol, order_id)
                        for trade in observation.trades
                        for order_id in (trade.buy_order_id, trade.sell_order_id)
                    }
                    if active_ids & traded_ids:
                        partial_fill_events += 1
            queue_priority_probes += sum(
                1
                for end_index in range(5, len(events) + 1)
                if is_queue_priority_probe(events, end_index)
            )
        finally:
            reference.close()

    evidence = {
        "limit-orders": applied_order_types["LIMIT"],
        "fifo-price-time-priority": queue_priority_probes,
        "partial-fills": partial_fill_events,
        "cancellation": applied_operations["cancel"],
        "reduction": applied_operations["reduce"],
        "replacement": applied_operations["replace"],
        "book-clear": applied_operations["clear"],
        "duplicate-active-order-id": rejection_reasons["DUPLICATE_ORDER_ID"],
        "inactive-lifecycle-request": rejection_reasons["ORDER_NOT_ACTIVE"],
        "multiple-symbols": len(applied_symbols) if len(applied_symbols) >= 2 else 0,
        "market-orders": applied_order_types["MARKET"],
        "immediate-or-cancel": applied_order_types["IOC"],
        "fill-or-kill": applied_order_types["FOK"],
    }
    return SemanticCoverage(
        expected_capabilities=tuple(expected_capabilities),
        evidence=evidence,
        operations=operations,
        order_types=order_types,
        applied_order_types=applied_order_types,
        outcomes=outcomes,
        rejection_reasons=rejection_reasons,
        symbols=tuple(sorted(applied_symbols)),
        compared_events=sum(len(events) for events in traces),
        trade_events=trade_events,
        trades=trade_count,
        partial_fill_events=partial_fill_events,
        queue_priority_probes=queue_priority_probes,
    )
