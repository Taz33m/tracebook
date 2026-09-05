"""Deterministic stateful campaigns for L3 book-replay adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from decimal import Decimal
from numbers import Integral
from typing import Dict, List, Optional, Sequence, Tuple, TypeVar

from .compare import BookReplayReport, run_book_replay
from .minimize import BookReplayMinimizationResult, minimize_book_replay_failure
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    PROFILE_NAME,
    BookReplayConfig,
    BookReplayError,
    BookReplayEvent,
    BookReplayState,
    EngineMetadata,
    trace_sha256,
)
from .protocol import BookReplayAdapterFactory
from .reference import ReferenceBookReplayAdapter

BOOK_REPLAY_GENERATOR_VERSION = 1
BOOK_REPLAY_CAPABILITIES = (
    "l3-add",
    "same-level-fifo",
    "same-level-upsize",
    "price-relocation",
    "l3-delete",
    "duplicate-active-order-id",
    "inactive-lifecycle-request",
    "buy-fill-probe",
    "sell-fill-probe",
    "multi-level-depth",
    "multiple-symbols",
)
_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_CHOICE = TypeVar("_CHOICE")


class _SplitMix64:
    """Specified PRNG whose output is independent of Python's random module."""

    def __init__(self, seed: int) -> None:
        self._state = seed & _MASK_64

    def next_u64(self) -> int:
        self._state = (self._state + _GOLDEN_GAMMA) & _MASK_64
        value = self._state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK_64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK_64
        return (value ^ (value >> 31)) & _MASK_64

    def randbelow(self, bound: int) -> int:
        if bound <= 0:
            raise ValueError("bound must be positive")
        return self.next_u64() % bound

    def choice(self, values: Sequence[_CHOICE]) -> _CHOICE:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self.randbelow(len(values))]


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise BookReplayError(f"{field_name} must be a positive integer")
    return int(value)


def _campaign_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value <= _MASK_64:
        raise BookReplayError("seed must be an integer between 0 and 2^64-1")
    return int(value)


def _trace_seed(seed: int, trace_index: int) -> int:
    mixer = _SplitMix64(seed ^ ((trace_index * _GOLDEN_GAMMA) & _MASK_64))
    return mixer.next_u64()


def _semantic_scaffold() -> Tuple[BookReplayEvent, ...]:
    """Guarantee every v1 capability before random exploration starts."""
    rows = (
        ("add", "TEST", 1, "BUY", "100", "2"),
        ("add", "TEST", 2, "BUY", "100", "3"),
        ("update", "TEST", 1, "BUY", "100", "4"),
        ("add", "TEST", 3, "BUY", "99", "5"),
        ("add", "TEST", 10, "SELL", "101", "1.5"),
        ("add", "TEST", 11, "SELL", "101", "2.5"),
        ("add", "TEST", 12, "SELL", "102", "4"),
        ("probe", "TEST", None, "BUY", "102", "5"),
        ("probe", "TEST", None, "SELL", "99", "8"),
        ("update", "TEST", 2, "BUY", "99", "3"),
        ("probe", "TEST", None, "SELL", "99", "12"),
        ("delete", "TEST", 1, None, None, None),
        ("delete", "TEST", 10, None, None, None),
        ("probe", "TEST", None, "BUY", "101", "3"),
        ("add", "ALT", 1, "SELL", "50", "7"),
        ("probe", "ALT", None, "BUY", "50", "2"),
        ("delete", "TEST", 999, None, None, None),
        ("add", "TEST", 2, "BUY", "99", "1"),
    )
    return tuple(
        BookReplayEvent(
            op=op,
            symbol=symbol,
            order_id=order_id,
            side=side,
            price=price,
            quantity=quantity,
        )
        for op, symbol, order_id, side, price, quantity in rows
    )


@dataclass(frozen=True)
class _ActiveOrder:
    symbol: str
    order_id: int
    side: str
    price: str
    quantity: str


def _active_orders(state: BookReplayState) -> Tuple[_ActiveOrder, ...]:
    active = []
    for book in state.books:
        for side, orders in (("BUY", book.bids), ("SELL", book.asks)):
            for order in orders:
                active.append(
                    _ActiveOrder(
                        book.symbol,
                        order.order_id,
                        side,
                        order.price,
                        order.quantity,
                    )
                )
    return tuple(sorted(active, key=lambda item: (item.symbol, item.order_id)))


def _new_event(
    rng: _SplitMix64,
    next_ids: Dict[str, int],
) -> BookReplayEvent:
    symbol = rng.choice(tuple(next_ids))
    order_id = next_ids[symbol]
    next_ids[symbol] += 1
    return BookReplayEvent(
        "add",
        symbol,
        order_id,
        rng.choice(("BUY", "SELL")),
        rng.choice(("98", "99", "100", "101", "102")),
        rng.choice(("0.5", "1", "2", "3", "5")),
    )


def _generated_event(
    rng: _SplitMix64,
    next_ids: Dict[str, int],
    active: Tuple[_ActiveOrder, ...],
) -> BookReplayEvent:
    if not active:
        return _new_event(rng, next_ids)
    roll = rng.randbelow(100)
    selected = rng.choice(active)
    if roll < 35:
        return _new_event(rng, next_ids)
    if roll < 60:
        mode = rng.randbelow(4)
        side = selected.side if mode < 3 else ("SELL" if selected.side == "BUY" else "BUY")
        price = selected.price if mode in {0, 1} else rng.choice(("98", "99", "100", "101", "102"))
        quantity = (
            str(Decimal(selected.quantity) + Decimal("1"))
            if mode == 0
            else rng.choice(("0.5", "1", "2", "3", "5"))
        )
        return BookReplayEvent(
            "update",
            selected.symbol,
            selected.order_id,
            side,
            price,
            quantity,
        )
    if roll < 74:
        return BookReplayEvent("delete", selected.symbol, selected.order_id)
    if roll < 91:
        return BookReplayEvent(
            "probe",
            rng.choice(tuple(next_ids)),
            side=rng.choice(("BUY", "SELL")),
            price=rng.choice(("98", "99", "100", "101", "102")),
            quantity=rng.choice(("0.5", "1", "2", "4", "8")),
        )
    if roll < 95:
        return BookReplayEvent(
            "add",
            selected.symbol,
            selected.order_id,
            selected.side,
            selected.price,
            selected.quantity,
        )
    symbol = rng.choice(tuple(next_ids))
    unknown_id = 1_000_000 + next_ids[symbol] + rng.randbelow(10_000)
    if roll < 98:
        return BookReplayEvent("delete", symbol, unknown_id)
    return BookReplayEvent(
        "update",
        symbol,
        unknown_id,
        rng.choice(("BUY", "SELL")),
        rng.choice(("98", "99", "100", "101", "102")),
        rng.choice(("1", "2", "3")),
    )


def generate_book_replay_trace(
    seed: int,
    event_count: int,
) -> Tuple[BookReplayEvent, ...]:
    """Generate one stateful trace without consulting candidate behavior."""
    seed = _campaign_seed(seed)
    event_count = _positive_int(event_count, "event_count")
    rng = _SplitMix64(seed)
    scaffold = _semantic_scaffold()
    next_ids = {"ALPHA": 1, "BETA": 1}
    reference = ReferenceBookReplayAdapter(BookReplayConfig())
    events: List[BookReplayEvent] = []
    active: Tuple[_ActiveOrder, ...] = ()
    try:
        for index in range(1, event_count + 1):
            event = scaffold[index - 1] if index <= len(scaffold) else None
            if event is None:
                event = _generated_event(rng, next_ids, active)
            events.append(event)
            reference.apply(event, index)
            active = _active_orders(reference.snapshot())
    finally:
        reference.close()
    return tuple(events)


@dataclass(frozen=True)
class BookReplayCoverage:
    """Source-event capability coverage reached by compared campaign prefixes."""

    covered: Tuple[str, ...]

    @property
    def uncovered(self) -> Tuple[str, ...]:
        return tuple(name for name in BOOK_REPLAY_CAPABILITIES if name not in self.covered)

    @property
    def complete(self) -> bool:
        return not self.uncovered

    def to_dict(self) -> dict:
        return {
            "covered": len(self.covered),
            "expected": len(BOOK_REPLAY_CAPABILITIES),
            "capabilities": list(self.covered),
            "uncovered": list(self.uncovered),
            "complete": self.complete,
        }


def measure_book_replay_coverage(
    traces: Sequence[Sequence[BookReplayEvent]],
) -> BookReplayCoverage:
    """Measure which v1 operations occurred before comparison stopped."""
    covered = set()
    symbols = set()
    for trace in traces:
        active: Dict[Tuple[str, int], BookReplayEvent] = {}
        for event in trace:
            symbols.add(event.symbol)
            key = (event.symbol, event.order_id or 0)
            if event.op == "add":
                covered.add("l3-add")
                if key in active:
                    covered.add("duplicate-active-order-id")
                    continue
                peers = [
                    order
                    for order in active.values()
                    if order.symbol == event.symbol and order.side == event.side
                ]
                if any(order.price == event.price for order in peers):
                    covered.add("same-level-fifo")
                if peers and any(order.price != event.price for order in peers):
                    covered.add("multi-level-depth")
                active[key] = event
            elif event.op == "update":
                previous = active.get(key)
                if previous is None:
                    covered.add("inactive-lifecycle-request")
                    continue
                if previous.side == event.side and previous.price == event.price:
                    if Decimal(event.quantity or "0") > Decimal(previous.quantity or "0"):
                        covered.add("same-level-upsize")
                elif previous.price != event.price:
                    covered.add("price-relocation")
                active[key] = event
            elif event.op == "delete":
                if key in active:
                    covered.add("l3-delete")
                    del active[key]
                else:
                    covered.add("inactive-lifecycle-request")
            elif event.side == "BUY":
                covered.add("buy-fill-probe")
            else:
                covered.add("sell-fill-probe")
    if len(symbols) > 1:
        covered.add("multiple-symbols")
    return BookReplayCoverage(tuple(name for name in BOOK_REPLAY_CAPABILITIES if name in covered))


@dataclass(frozen=True)
class BookReplayCampaignTrace:
    index: int
    seed: int
    events: Tuple[BookReplayEvent, ...]
    report: BookReplayReport

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "seed": self.seed,
            "event_count": len(self.events),
            "trace_sha256": trace_sha256(self.events),
            "compared_events": self.report.compared_events,
            "conformant": self.report.conformant,
            "divergence": self.report.divergence.to_dict() if self.report.divergence else None,
        }


@dataclass(frozen=True)
class BookReplayCampaignFailure:
    trace: BookReplayCampaignTrace
    original_events: Tuple[BookReplayEvent, ...]
    original_report: BookReplayReport
    minimization: BookReplayMinimizationResult

    def to_dict(self) -> dict:
        divergence = self.original_report.divergence
        return {
            "trace_index": self.trace.index,
            "failure_class": (
                f"{divergence.category}:{divergence.kind}" if divergence else "unknown"
            ),
            "original_divergence_event": divergence.event_index if divergence else None,
            "original_event_count": len(self.original_events),
            "original_trace_sha256": trace_sha256(self.original_events),
            "original_events": [event.to_dict() for event in self.original_events],
            "original_report": self.original_report.to_dict(),
            "minimized_event_count": len(self.minimization.events),
            "minimized_trace_sha256": trace_sha256(self.minimization.events),
            "minimized_events": [event.to_dict() for event in self.minimization.events],
            "minimization": self.minimization.to_dict(),
            "one_minimal": self.minimization.one_minimal,
            "budget_exhausted": self.minimization.budget_exhausted,
        }


@dataclass(frozen=True)
class BookReplayCampaignResult:
    seed: int
    requested_traces: int
    events_per_trace: int
    max_minimize_runs: int
    candidate_engine: EngineMetadata
    traces: Tuple[BookReplayCampaignTrace, ...]
    failure: Optional[BookReplayCampaignFailure]
    coverage: BookReplayCoverage

    @property
    def conformant(self) -> bool:
        return self.failure is None

    @property
    def campaign_id(self) -> str:
        payload = {
            "generator_version": BOOK_REPLAY_GENERATOR_VERSION,
            "profile": PROFILE_NAME,
            "seed": self.seed,
            "requested_traces": self.requested_traces,
            "events_per_trace": self.events_per_trace,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @property
    def failure_id(self) -> Optional[str]:
        if self.failure is None:
            return None
        divergence = self.failure.minimization.report.divergence
        if divergence is None:
            raise BookReplayError("campaign failure has no reduced divergence")
        payload = {
            "campaign_id": self.campaign_id,
            "trace_index": self.failure.trace.index,
            "original_trace_sha256": trace_sha256(self.failure.original_events),
            "reduced_trace_sha256": trace_sha256(self.failure.minimization.events),
            "reduced_divergence": divergence.to_dict(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return "failure-" + hashlib.sha256(encoded).hexdigest()[:20]

    def to_dict(self) -> dict:
        minimization_runs = self.failure.minimization.runs if self.failure else 0
        failure = self.failure.to_dict() if self.failure else None
        if failure is not None:
            failure["failure_id"] = self.failure_id
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.book-replay.campaign",
            "generator_version": BOOK_REPLAY_GENERATOR_VERSION,
            "campaign_id": self.campaign_id,
            "profile": PROFILE_NAME,
            "config": BookReplayConfig().to_dict(),
            "seed": self.seed,
            "requested_traces": self.requested_traces,
            "completed_traces": len(self.traces),
            "events_per_trace": self.events_per_trace,
            "generated_events": sum(len(trace.events) for trace in self.traces),
            "max_minimize_runs": self.max_minimize_runs,
            "candidate_runs": len(self.traces) + minimization_runs,
            "candidate_engine": self.candidate_engine.to_dict(),
            "stopped_at_first_divergence": True,
            "conformant": self.conformant,
            "semantic_coverage": self.coverage.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "failure": failure,
        }


def run_book_replay_campaign(
    candidate_factory: BookReplayAdapterFactory,
    seed: int = 1337,
    traces: int = 25,
    events_per_trace: int = 100,
    max_minimize_runs: int = 100,
) -> BookReplayCampaignResult:
    """Run generated traces until completion or the first minimized divergence."""
    seed = _campaign_seed(seed)
    traces = _positive_int(traces, "traces")
    events_per_trace = _positive_int(events_per_trace, "events_per_trace")
    max_minimize_runs = _positive_int(max_minimize_runs, "max_minimize_runs")
    config = BookReplayConfig()
    results = []
    candidate_engine: Optional[EngineMetadata] = None
    failure: Optional[BookReplayCampaignFailure] = None
    for trace_index in range(1, traces + 1):
        generated_seed = _trace_seed(seed, trace_index)
        events = generate_book_replay_trace(generated_seed, events_per_trace)
        trace_name = f"campaign:{PROFILE_NAME}:{trace_index}"
        report = run_book_replay(events, candidate_factory, config, trace_name=trace_name)
        if candidate_engine is None:
            candidate_engine = report.candidate_engine
        elif candidate_engine != report.candidate_engine:
            raise BookReplayError("candidate engine metadata changed between campaign traces")
        trace_result = BookReplayCampaignTrace(trace_index, generated_seed, events, report)
        results.append(trace_result)
        if not report.conformant:
            if report.divergence is None:
                raise BookReplayError("non-conformant campaign report has no divergence")
            original_events = events[: report.divergence.event_index]
            original_report = replace(
                report,
                trace_hash=trace_sha256(original_events),
                event_count=len(original_events),
            )
            minimization = minimize_book_replay_failure(
                original_events,
                candidate_factory,
                config=config,
                max_runs=max_minimize_runs,
                trace_name=trace_name,
                expected_candidate_engine=candidate_engine,
            )
            failure = BookReplayCampaignFailure(
                trace_result,
                original_events,
                original_report,
                minimization,
            )
            break
    if candidate_engine is None:
        raise BookReplayError("campaign completed without candidate metadata")
    compared = tuple(trace.events[: trace.report.compared_events] for trace in results)
    return BookReplayCampaignResult(
        seed,
        traces,
        events_per_trace,
        max_minimize_runs,
        candidate_engine,
        tuple(results),
        failure,
        measure_book_replay_coverage(compared),
    )


__all__ = [
    "BOOK_REPLAY_CAPABILITIES",
    "BOOK_REPLAY_GENERATOR_VERSION",
    "BookReplayCampaignFailure",
    "BookReplayCampaignResult",
    "BookReplayCampaignTrace",
    "BookReplayCoverage",
    "generate_book_replay_trace",
    "measure_book_replay_coverage",
    "run_book_replay_campaign",
]
