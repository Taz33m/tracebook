"""Deterministic stateful campaigns for differential matching-engine testing."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from decimal import Decimal
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

from ..core.order import OrderSide, OrderType
from ..events import MarketEvent
from .classification import classify_failure
from .compare import ConformanceReport, run_conformance
from .minimize import MinimizationResult, minimize_failing_trace
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    canonical_decimal,
    trace_sha256,
)
from .protocol import AdapterFactory
from .reference import ReferenceEngineAdapter
from .semantic_coverage import SemanticCoverage, measure_semantic_coverage

CAMPAIGN_GENERATOR_VERSION = 2
_MASK_64 = (1 << 64) - 1
_GOLDEN_GAMMA = 0x9E3779B97F4A7C15
_CHOICES = TypeVar("_CHOICES")


@dataclass(frozen=True)
class CampaignProfile:
    """A versioned semantic surface used to generate campaign traces."""

    name: str
    description: str
    config: ConformanceConfig
    order_types: Tuple[OrderType, ...]
    capabilities: Tuple[str, ...] = ("limit-orders",)
    symbols: Tuple[str, ...] = ("ALPHA", "BETA")

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConformanceError("campaign profile name must be a non-empty string")
        if not isinstance(self.description, str) or not self.description.strip():
            raise ConformanceError("campaign profile description must be a non-empty string")
        if not isinstance(self.config, ConformanceConfig):
            raise ConformanceError("campaign profile config must be a ConformanceConfig")
        if not isinstance(self.order_types, tuple) or not self.order_types:
            raise ConformanceError("campaign profile order_types must start with LIMIT")
        if self.order_types[0] is not OrderType.LIMIT:
            raise ConformanceError("campaign profile order_types must start with LIMIT")
        if any(not isinstance(order_type, OrderType) for order_type in self.order_types):
            raise ConformanceError("campaign profile order_types must be unique OrderType values")
        if len(set(self.order_types)) != len(self.order_types):
            raise ConformanceError("campaign profile order_types must be unique OrderType values")
        if not isinstance(self.capabilities, tuple) or not self.capabilities:
            raise ConformanceError("campaign profile capabilities must be a non-empty tuple")
        if any(not isinstance(name, str) or not name for name in self.capabilities):
            raise ConformanceError("campaign profile capabilities must be non-empty strings")
        if len(set(self.capabilities)) != len(self.capabilities):
            raise ConformanceError("campaign profile capabilities must be unique")
        if not isinstance(self.symbols, tuple) or not self.symbols:
            raise ConformanceError("campaign profile symbols must be a non-empty tuple")
        if any(not isinstance(symbol, str) or not symbol.strip() for symbol in self.symbols):
            raise ConformanceError("campaign profile symbols must be unique non-empty strings")
        if len(set(self.symbols)) != len(self.symbols):
            raise ConformanceError("campaign profile symbols must be unique non-empty strings")

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "config": self.config.to_dict(),
            "order_types": [order_type.name for order_type in self.order_types],
            "capabilities": list(self.capabilities),
            "symbols": list(self.symbols),
        }


_PROFILES: Mapping[str, CampaignProfile] = {
    "fifo-limit-v1": CampaignProfile(
        name="fifo-limit-v1",
        description=(
            "FIFO limit orders, cancellation, reduction, replacement, clear, "
            "duplicate IDs, inactive lifecycle requests, and multiple symbols."
        ),
        config=ConformanceConfig(matching_algorithm="fifo"),
        order_types=(OrderType.LIMIT,),
        capabilities=(
            "limit-orders",
            "fifo-price-time-priority",
            "partial-fills",
            "cancellation",
            "reduction",
            "replacement",
            "book-clear",
            "duplicate-active-order-id",
            "inactive-lifecycle-request",
            "multiple-symbols",
        ),
    ),
    "fifo-full-v1": CampaignProfile(
        name="fifo-full-v1",
        description=("The FIFO lifecycle profile plus MARKET, IOC, and FOK instructions."),
        config=ConformanceConfig(matching_algorithm="fifo"),
        order_types=(OrderType.LIMIT, OrderType.MARKET, OrderType.IOC, OrderType.FOK),
        capabilities=(
            "limit-orders",
            "fifo-price-time-priority",
            "partial-fills",
            "cancellation",
            "reduction",
            "replacement",
            "book-clear",
            "duplicate-active-order-id",
            "inactive-lifecycle-request",
            "multiple-symbols",
            "market-orders",
            "immediate-or-cancel",
            "fill-or-kill",
        ),
    ),
}


def campaign_profile_names() -> Tuple[str, ...]:
    """Return the stable names accepted by the campaign generator."""
    return tuple(sorted(_PROFILES))


def get_campaign_profile(name: str) -> CampaignProfile:
    """Resolve one built-in campaign profile by its versioned name."""
    if not isinstance(name, str):
        raise ConformanceError("campaign profile must be a string")
    try:
        return _PROFILES[name.strip().lower()]
    except KeyError as exc:
        choices = ", ".join(campaign_profile_names())
        raise ConformanceError(f"unknown campaign profile {name!r}; choose {choices}") from exc


class _SplitMix64:
    """Small specified PRNG so campaign traces do not depend on Python internals."""

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

    def choice(self, values: Sequence[_CHOICES]) -> _CHOICES:
        if not values:
            raise ValueError("cannot choose from an empty sequence")
        return values[self.randbelow(len(values))]


@dataclass(frozen=True)
class _ActiveOrder:
    symbol: str
    order_id: int
    side: OrderSide
    price: Decimal
    remaining_quantity: Decimal
    owner: int


def _positive_int(value: int, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ConformanceError(f"{field_name} must be a positive integer")
    return int(value)


def _campaign_seed(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0 or value > _MASK_64:
        raise ConformanceError("seed must be an integer between 0 and 2^64-1")
    return int(value)


def _trace_seed(seed: int, trace_index: int) -> int:
    mixer = _SplitMix64(seed ^ ((trace_index * _GOLDEN_GAMMA) & _MASK_64))
    return mixer.next_u64()


def _price(rng: _SplitMix64, profile: CampaignProfile) -> Decimal:
    tick = Decimal(canonical_decimal(profile.config.tick_size))
    return Decimal("100") + tick * Decimal(rng.choice((-2, -1, 0, 1, 2)))


def _quantity(rng: _SplitMix64) -> Decimal:
    return rng.choice((Decimal("0.5"), Decimal("1"), Decimal("2"), Decimal("3"), Decimal("5")))


def _active_orders(state: BookState) -> Tuple[_ActiveOrder, ...]:
    active: List[_ActiveOrder] = []
    for book in state.books:
        for side, orders in ((OrderSide.BUY, book.bids), (OrderSide.SELL, book.asks)):
            for order in orders:
                active.append(
                    _ActiveOrder(
                        symbol=book.symbol,
                        order_id=order.order_id,
                        side=side,
                        price=Decimal(order.price),
                        remaining_quantity=Decimal(order.remaining_quantity),
                        owner=order.owner,
                    )
                )
    return tuple(sorted(active, key=lambda order: (order.symbol, order.order_id)))


def _new_event(
    rng: _SplitMix64,
    profile: CampaignProfile,
    next_ids: Dict[str, int],
    index: int,
    force_limit: bool = False,
) -> MarketEvent:
    symbol = rng.choice(profile.symbols)
    order_id = next_ids[symbol]
    next_ids[symbol] += 1
    if force_limit or len(profile.order_types) == 1:
        order_type = OrderType.LIMIT
    else:
        roll = rng.randbelow(10)
        additional_types = profile.order_types[1:]
        order_type = (
            OrderType.LIMIT if roll < 7 else additional_types[(roll - 7) % len(additional_types)]
        )
    price = None if order_type == OrderType.MARKET else float(_price(rng, profile))
    return MarketEvent(
        op="new",
        symbol=symbol,
        order_id=order_id,
        side=rng.choice((OrderSide.BUY, OrderSide.SELL)),
        order_type=order_type,
        price=price,
        quantity=float(_quantity(rng)),
        owner=1 + rng.randbelow(4),
        timestamp_ns=index,
    )


def _generated_event(
    rng: _SplitMix64,
    profile: CampaignProfile,
    next_ids: Dict[str, int],
    active: Tuple[_ActiveOrder, ...],
    index: int,
) -> MarketEvent:
    if not active:
        return _new_event(rng, profile, next_ids, index, force_limit=True)

    roll = rng.randbelow(100)
    selected = rng.choice(active)
    if roll < 52:
        return _new_event(rng, profile, next_ids, index)
    if roll < 64:
        return MarketEvent(
            op="cancel",
            symbol=selected.symbol,
            order_id=selected.order_id,
            timestamp_ns=index,
        )
    if roll < 75:
        reduction = (
            selected.remaining_quantity / 2
            if selected.remaining_quantity > Decimal("0.25")
            else selected.remaining_quantity
        )
        return MarketEvent(
            op="reduce",
            symbol=selected.symbol,
            order_id=selected.order_id,
            quantity=float(reduction),
            timestamp_ns=index,
        )
    if roll < 87:
        mode = rng.randbelow(3)
        if mode == 0:
            quantity = (
                selected.remaining_quantity / 2
                if selected.remaining_quantity > Decimal("0.25")
                else selected.remaining_quantity + Decimal("0.5")
            )
            return MarketEvent(
                op="replace",
                symbol=selected.symbol,
                order_id=selected.order_id,
                price=float(selected.price),
                quantity=float(quantity),
                timestamp_ns=index,
            )
        if mode == 1:
            return MarketEvent(
                op="replace",
                symbol=selected.symbol,
                order_id=selected.order_id,
                price=float(_price(rng, profile)),
                timestamp_ns=index,
            )
        return MarketEvent(
            op="replace",
            symbol=selected.symbol,
            order_id=selected.order_id,
            quantity=float(selected.remaining_quantity + _quantity(rng)),
            timestamp_ns=index,
        )
    if roll < 92:
        symbol = rng.choice(profile.symbols)
        return MarketEvent(
            op="cancel",
            symbol=symbol,
            order_id=1_000_000 + next_ids[symbol] + rng.randbelow(10_000),
            timestamp_ns=index,
        )
    if roll < 96:
        return MarketEvent(
            op="new",
            symbol=selected.symbol,
            order_id=selected.order_id,
            side=selected.side,
            order_type=OrderType.LIMIT,
            price=float(selected.price),
            quantity=1.0,
            owner=selected.owner,
            timestamp_ns=index,
        )
    return MarketEvent(
        op="clear",
        symbol=selected.symbol,
        timestamp_ns=index,
    )


_QUEUE_PROBE_SYMBOL = "FIFO-PRIORITY-PROBE"
_QUEUE_PROBE_FIRST_MAKER = 9_000_000_001
_QUEUE_PROBE_SECOND_MAKER = 9_000_000_002
_QUEUE_PROBE_TAKER = 9_000_000_003


def _queue_priority_probe_end(seed: int, event_count: int) -> Optional[int]:
    if event_count < 5:
        return None
    return min(event_count, 133 + seed % 43)


def _queue_priority_probe(seed: int, event_count: int) -> Mapping[int, MarketEvent]:
    """Place a deterministic five-event FIFO probe inside one isolated book."""
    end = _queue_priority_probe_end(seed, event_count)
    if end is None:
        return {}
    start = end - 4
    return {
        start: MarketEvent(
            op="new",
            symbol=_QUEUE_PROBE_SYMBOL,
            order_id=_QUEUE_PROBE_FIRST_MAKER,
            side=OrderSide.SELL,
            price=100.0,
            quantity=5.0,
            owner=101,
            timestamp_ns=start,
        ),
        start
        + 1: MarketEvent(
            op="new",
            symbol=_QUEUE_PROBE_SYMBOL,
            order_id=_QUEUE_PROBE_SECOND_MAKER,
            side=OrderSide.SELL,
            price=100.0,
            quantity=5.0,
            owner=102,
            timestamp_ns=start + 1,
        ),
        start
        + 2: MarketEvent(
            op="reduce",
            symbol=_QUEUE_PROBE_SYMBOL,
            order_id=_QUEUE_PROBE_FIRST_MAKER,
            quantity=1.0,
            timestamp_ns=start + 2,
        ),
        start
        + 3: MarketEvent(
            op="replace",
            symbol=_QUEUE_PROBE_SYMBOL,
            order_id=_QUEUE_PROBE_FIRST_MAKER,
            price=100.0,
            quantity=4.0,
            timestamp_ns=start + 3,
        ),
        end: MarketEvent(
            op="new",
            symbol=_QUEUE_PROBE_SYMBOL,
            order_id=_QUEUE_PROBE_TAKER,
            side=OrderSide.BUY,
            price=100.0,
            quantity=1.0,
            owner=103,
            timestamp_ns=end,
        ),
    }


def generate_campaign_trace(
    profile: str | CampaignProfile,
    seed: int,
    event_count: int,
) -> Tuple[MarketEvent, ...]:
    """Generate one reproducible trace without consulting candidate behavior."""
    selected_profile = get_campaign_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected_profile, CampaignProfile):
        raise ConformanceError("profile must be a campaign profile or profile name")
    seed = _campaign_seed(seed)
    event_count = _positive_int(event_count, "event_count")
    rng = _SplitMix64(seed)
    next_ids = {symbol: 1 for symbol in selected_profile.symbols}
    priority_probe = (
        _queue_priority_probe(seed, event_count)
        if "fifo-price-time-priority" in selected_profile.capabilities
        else {}
    )
    reference = ReferenceEngineAdapter(selected_profile.config)
    events: List[MarketEvent] = []
    active: Tuple[_ActiveOrder, ...] = ()
    try:
        for index in range(1, event_count + 1):
            event = priority_probe.get(index)
            if event is None:
                event = _generated_event(
                    rng,
                    selected_profile,
                    next_ids,
                    active,
                    index,
                )
            events.append(event)
            reference.apply(event, index)
            active = _active_orders(reference.snapshot())
    finally:
        reference.close()
    return tuple(events)


@dataclass(frozen=True)
class CampaignTraceResult:
    """One generated trace and its candidate comparison."""

    index: int
    seed: int
    events: Tuple[MarketEvent, ...]
    report: ConformanceReport

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "seed": self.seed,
            "event_count": len(self.events),
            "trace_sha256": trace_sha256(self.events),
            "compared_events": self.report.compared_events,
            "conformant": self.report.conformant,
            "divergence": (
                self.report.divergence.to_dict() if self.report.divergence is not None else None
            ),
        }


@dataclass(frozen=True)
class CampaignFailure:
    """The first divergent campaign trace and its automatic reduction."""

    trace: CampaignTraceResult
    original_events: Tuple[MarketEvent, ...]
    original_report: ConformanceReport
    minimization: MinimizationResult
    failure_class: str

    def to_dict(self) -> dict:
        minimized = self.minimization
        divergence = self.original_report.divergence
        return {
            "trace_index": self.trace.index,
            "failure_class": self.failure_class,
            "original_divergence_event": divergence.event_index if divergence else None,
            "target_category": minimized.target_category,
            "original_event_count": len(self.original_events),
            "original_trace_sha256": trace_sha256(self.original_events),
            "original_events": "failure/original.jsonl",
            "original_report": "failure/original-report.json",
            "minimized_event_count": len(minimized.events),
            "minimized_trace_sha256": trace_sha256(minimized.events),
            "minimized_events": "failure/minimized.jsonl",
            "reduced_events": "reduced.jsonl",
            "minimization_report": "failure/minimization.json",
            "one_minimal": minimized.one_minimal,
            "budget_exhausted": minimized.budget_exhausted,
        }


@dataclass(frozen=True)
class CampaignResult:
    """A deterministic campaign result and optional first-failure bundle."""

    profile: CampaignProfile
    seed: int
    requested_traces: int
    events_per_trace: int
    max_minimize_runs: int
    candidate_engine: EngineMetadata
    traces: Tuple[CampaignTraceResult, ...]
    failure: Optional[CampaignFailure]
    semantic_coverage: SemanticCoverage

    @property
    def conformant(self) -> bool:
        return self.failure is None

    @property
    def campaign_id(self) -> str:
        payload = {
            "generator_version": CAMPAIGN_GENERATOR_VERSION,
            "profile": self.profile.to_dict(),
            "seed": self.seed,
            "requested_traces": self.requested_traces,
            "events_per_trace": self.events_per_trace,
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "sha256:" + hashlib.sha256(encoded).hexdigest()

    @property
    def failure_id(self) -> Optional[str]:
        if self.failure is None:
            return None
        divergence = self.failure.minimization.report.divergence
        if divergence is None:
            raise ConformanceError("campaign failure has no reduced divergence")
        payload = {
            "campaign_id": self.campaign_id,
            "trace_index": self.failure.trace.index,
            "failure_class": self.failure.failure_class,
            "original_trace_sha256": trace_sha256(self.failure.original_events),
            "reduced_trace_sha256": trace_sha256(self.failure.minimization.events),
            "reduced_divergence": divergence.to_dict(),
        }
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return "failure-" + hashlib.sha256(encoded).hexdigest()[:20]

    @property
    def bundle_id(self) -> str:
        return self.failure_id or "campaign-" + self.campaign_id.removeprefix("sha256:")[:20]

    def failure_bundle_dict(self) -> dict:
        if self.failure is None:
            raise ConformanceError("a conformant campaign has no failure bundle")
        original_divergence = self.failure.original_report.divergence
        if original_divergence is None:
            raise ConformanceError("campaign failure has no original divergence")
        divergence = self.failure.minimization.report.divergence
        if divergence is None:
            raise ConformanceError("campaign failure has no reduced divergence")
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.failure",
            "failure_id": self.failure_id,
            "failure_class": self.failure.failure_class,
            "profile": self.profile.name,
            "config": self.profile.config.to_dict(),
            "generator_version": CAMPAIGN_GENERATOR_VERSION,
            "campaign_seed": self.seed,
            "campaign_id": self.campaign_id,
            "trace_index": self.failure.trace.index,
            "trace_seed": self.failure.trace.seed,
            "original_divergence_event": original_divergence.event_index,
            "original_event_count": len(self.failure.original_events),
            "original_trace_sha256": trace_sha256(self.failure.original_events),
            "reduced_event_count": len(self.failure.minimization.events),
            "reduced_trace_sha256": trace_sha256(self.failure.minimization.events),
            "target_category": self.failure.minimization.target_category,
            "one_minimal": self.failure.minimization.one_minimal,
            "budget_exhausted": self.failure.minimization.budget_exhausted,
            "candidate_engine": self.candidate_engine.to_dict(),
            "expected_reduced_divergence": divergence.to_dict(),
            "paths": {
                "original": "original.jsonl",
                "reduced": "reduced.jsonl",
                "campaign": "campaign.json",
                "minimization": "failure/minimization.json",
            },
        }

    def to_dict(self) -> dict:
        minimization_runs = self.failure.minimization.runs if self.failure else 0
        failure = self.failure.to_dict() if self.failure else None
        if failure is not None:
            failure["failure_id"] = self.failure_id
        return {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "artifact_type": "tracebook.conformance.campaign",
            "generator_version": CAMPAIGN_GENERATOR_VERSION,
            "campaign_id": self.campaign_id,
            "profile": self.profile.to_dict(),
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
            "semantic_coverage": self.semantic_coverage.to_dict(),
            "traces": [trace.to_dict() for trace in self.traces],
            "failure": failure,
        }


def run_campaign(
    candidate_factory: AdapterFactory,
    profile: str | CampaignProfile = "fifo-limit-v1",
    seed: int = 1337,
    traces: int = 25,
    events_per_trace: int = 100,
    max_minimize_runs: int = 100,
) -> CampaignResult:
    """Run generated traces until completion or the first minimized divergence."""
    selected_profile = get_campaign_profile(profile) if isinstance(profile, str) else profile
    if not isinstance(selected_profile, CampaignProfile):
        raise ConformanceError("profile must be a campaign profile or profile name")
    seed = _campaign_seed(seed)
    traces = _positive_int(traces, "traces")
    events_per_trace = _positive_int(events_per_trace, "events_per_trace")
    max_minimize_runs = _positive_int(max_minimize_runs, "max_minimize_runs")

    results: List[CampaignTraceResult] = []
    candidate_engine: Optional[EngineMetadata] = None
    failure: Optional[CampaignFailure] = None
    for trace_index in range(1, traces + 1):
        generated_seed = _trace_seed(seed, trace_index)
        events = generate_campaign_trace(selected_profile, generated_seed, events_per_trace)
        trace_name = f"campaign:{selected_profile.name}:{trace_index}"
        report = run_conformance(
            events,
            candidate_factory,
            config=selected_profile.config,
            trace_name=trace_name,
        )
        if candidate_engine is None:
            candidate_engine = report.candidate_engine
        elif candidate_engine != report.candidate_engine:
            raise ConformanceError("candidate engine metadata changed between campaign traces")
        trace_result = CampaignTraceResult(trace_index, generated_seed, events, report)
        results.append(trace_result)
        if not report.conformant:
            if report.divergence is None:
                raise ConformanceError("non-conformant campaign report has no divergence")
            original_events = events[: report.divergence.event_index]
            original_report = replace(
                report,
                trace_hash=trace_sha256(original_events),
                event_count=len(original_events),
            )
            minimization = minimize_failing_trace(
                original_events,
                candidate_factory,
                config=selected_profile.config,
                max_runs=max_minimize_runs,
                trace_name=trace_name,
                expected_candidate_engine=candidate_engine,
            )
            if minimization.report.candidate_engine != candidate_engine:
                raise ConformanceError(
                    "candidate engine metadata changed during campaign minimization"
                )
            failure = CampaignFailure(
                trace=trace_result,
                original_events=original_events,
                original_report=original_report,
                minimization=minimization,
                failure_class=classify_failure(original_events, original_report),
            )
            break

    if candidate_engine is None:
        raise ConformanceError("campaign completed without candidate metadata")
    compared_traces = tuple(trace.events[: trace.report.compared_events] for trace in results)
    return CampaignResult(
        profile=selected_profile,
        seed=seed,
        requested_traces=traces,
        events_per_trace=events_per_trace,
        max_minimize_runs=max_minimize_runs,
        candidate_engine=candidate_engine,
        traces=tuple(results),
        failure=failure,
        semantic_coverage=measure_semantic_coverage(
            compared_traces,
            selected_profile.config,
            selected_profile.capabilities,
        ),
    )


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_events(path: Path, events: Sequence[MarketEvent]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


_RESERVATION_MARKER = ".tracebook-campaign-reservation"
_DIRECTORY_OPEN_FLAGS = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0)
_EXCLUSIVE_FILE_FLAGS = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
_DIRECTORY_FD_SUPPORTED = (
    os.open in os.supports_dir_fd
    and os.mkdir in os.supports_dir_fd
    and os.unlink in os.supports_dir_fd
    and os.listdir in os.supports_fd
)


def _open_exclusive_file(name: str, directory_fd: int) -> int:
    return os.open(name, _EXCLUSIVE_FILE_FLAGS, 0o600, dir_fd=directory_fd)


def _write_exclusive_bytes(
    name: str,
    payload: bytes,
    directory_fd: int,
) -> None:
    file_descriptor = _open_exclusive_file(name, directory_fd)
    with os.fdopen(file_descriptor, "wb") as handle:
        handle.write(payload)


def _copy_exclusive_file(
    source: Path,
    name: str,
    directory_fd: int,
) -> None:
    file_descriptor = _open_exclusive_file(name, directory_fd)
    with source.open("rb") as source_handle, os.fdopen(file_descriptor, "wb") as target_handle:
        shutil.copyfileobj(source_handle, target_handle)


def _read_relative_bytes(name: str, directory_fd: int) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    file_descriptor = os.open(name, flags, dir_fd=directory_fd)
    with os.fdopen(file_descriptor, "rb") as handle:
        return handle.read()


def _relative_names(directory_fd: int) -> set[str]:
    return set(os.listdir(directory_fd))


def _make_relative_directory(name: str, directory_fd: int) -> int:
    os.mkdir(name, mode=0o700, dir_fd=directory_fd)
    return os.open(name, _DIRECTORY_OPEN_FLAGS, dir_fd=directory_fd)


def _unlink_relative(name: str, directory_fd: int) -> None:
    os.unlink(name, dir_fd=directory_fd)


class _CampaignOutputReservation:
    """Own one exact output path until its complete bundle is committed."""

    def __init__(self, destination: str | Path) -> None:
        self.target = Path(destination).expanduser()
        self._token = uuid.uuid4().hex
        self._identity: Optional[Tuple[int, int]] = None
        self._directory_fd: Optional[int] = None
        self._active = False
        self._committed = False

    def __enter__(self) -> "_CampaignOutputReservation":
        if not _DIRECTORY_FD_SUPPORTED:
            raise ConformanceError(
                "campaign artifact commits require descriptor-relative "
                "directory operations on this platform"
            )
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self.target.mkdir()
        except FileExistsError as exc:
            raise ConformanceError(f"campaign output already exists: {self.target}") from exc
        except OSError as exc:
            raise ConformanceError(
                f"could not reserve campaign output {self.target}: {exc}"
            ) from exc

        try:
            directory_fd = os.open(self.target, _DIRECTORY_OPEN_FLAGS)
            self._directory_fd = directory_fd
            stat = os.fstat(directory_fd)
            self._identity = (stat.st_dev, stat.st_ino)
            _write_exclusive_bytes(
                _RESERVATION_MARKER,
                self._token.encode("ascii"),
                directory_fd,
            )
            self._active = True
        except OSError as exc:
            self.close()
            raise ConformanceError(
                f"could not reserve campaign output {self.target}: {exc}"
            ) from exc
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _same_directory(self) -> bool:
        try:
            stat = self.target.stat()
        except OSError:
            return False
        return not self.target.is_symlink() and (stat.st_dev, stat.st_ino) == self._identity

    def _owns_clean_reservation(self) -> bool:
        directory_fd = self._directory_fd
        if directory_fd is None or not self._active or not self._same_directory():
            return False
        try:
            return _read_relative_bytes(
                _RESERVATION_MARKER,
                directory_fd,
            ) == self._token.encode(
                "ascii"
            ) and _relative_names(directory_fd) == {_RESERVATION_MARKER}
        except OSError:
            return False

    def _restore_marker(self) -> None:
        directory_fd = self._directory_fd
        if directory_fd is None:
            return
        try:
            _write_exclusive_bytes(
                _RESERVATION_MARKER,
                self._token.encode("ascii"),
                directory_fd,
            )
        except OSError:
            pass

    def close(self) -> None:
        if self._directory_fd is not None:
            os.close(self._directory_fd)
            self._directory_fd = None
        self._active = False

    def write(self, result: CampaignResult) -> Path:
        """Commit a complete campaign bundle over this exact reservation."""
        if not isinstance(result, CampaignResult):
            raise ConformanceError("result must be a CampaignResult")
        directory_fd = self._directory_fd
        if directory_fd is None or not self._active or self._committed:
            raise ConformanceError("campaign output reservation is not active")

        temporary = Path(
            tempfile.mkdtemp(prefix=f".{self.target.name}.", dir=str(self.target.parent))
        )
        failure_fd: Optional[int] = None
        try:
            if result.failure is not None:
                failure_dir = temporary / "failure"
                failure_dir.mkdir()
                _write_events(failure_dir / "original.jsonl", result.failure.original_events)
                _write_json(
                    failure_dir / "original-report.json",
                    result.failure.original_report.to_dict(),
                )
                _write_events(
                    failure_dir / "minimized.jsonl",
                    result.failure.minimization.events,
                )
                _write_json(
                    failure_dir / "minimization.json",
                    result.failure.minimization.to_dict(),
                )
                _write_events(temporary / "original.jsonl", result.failure.original_events)
                _write_events(temporary / "reduced.jsonl", result.failure.minimization.events)
                _write_json(temporary / "failure.json", result.failure_bundle_dict())
            _write_json(temporary / "campaign.json", result.to_dict())
            if not self._owns_clean_reservation():
                raise ConformanceError(
                    f"campaign output reservation changed before commit: {self.target}"
                )

            try:
                expected_top_level = {_RESERVATION_MARKER, "campaign.json"}
                expected_failure_files: Optional[set[str]] = None
                if result.failure is not None:
                    expected_top_level.update({"failure.json", "original.jsonl", "reduced.jsonl"})
                    failure_fd = _make_relative_directory("failure", directory_fd)
                    expected_top_level.add("failure")
                    expected_failure_files = {
                        "original.jsonl",
                        "original-report.json",
                        "minimized.jsonl",
                        "minimization.json",
                    }
                    for name in sorted(expected_failure_files):
                        _copy_exclusive_file(
                            temporary / "failure" / name,
                            name,
                            failure_fd,
                        )

                    for name in ("failure.json", "original.jsonl", "reduced.jsonl"):
                        _copy_exclusive_file(
                            temporary / name,
                            name,
                            directory_fd,
                        )

                _copy_exclusive_file(
                    temporary / "campaign.json",
                    "campaign.json",
                    directory_fd,
                )
                reservation_unchanged = (
                    self._same_directory()
                    and _relative_names(directory_fd) == expected_top_level
                    and (
                        expected_failure_files is None
                        or (
                            failure_fd is not None
                            and _relative_names(failure_fd) == expected_failure_files
                        )
                    )
                )
                if not reservation_unchanged:
                    raise ConformanceError(
                        f"campaign output reservation changed before commit: {self.target}"
                    )

                _unlink_relative(_RESERVATION_MARKER, directory_fd)
                committed_names = expected_top_level - {_RESERVATION_MARKER}
                commit_visible = (
                    self._same_directory()
                    and _relative_names(directory_fd) == committed_names
                    and (
                        expected_failure_files is None
                        or (
                            failure_fd is not None
                            and _relative_names(failure_fd) == expected_failure_files
                        )
                    )
                )
                if not commit_visible:
                    self._restore_marker()
                    raise ConformanceError(
                        f"campaign output reservation changed during commit: {self.target}"
                    )
            except ConformanceError:
                raise
            except OSError as exc:
                self._restore_marker()
                raise ConformanceError(
                    f"could not commit campaign output {self.target}: {exc}"
                ) from exc
            self._committed = True
            return self.target / "campaign.json"
        finally:
            if failure_fd is not None:
                os.close(failure_fd)
            shutil.rmtree(temporary, ignore_errors=True)


def write_campaign_artifacts(result: CampaignResult, destination: str | Path) -> Path:
    """Commit a bundle into a reserved directory, removing its marker last."""
    if not isinstance(result, CampaignResult):
        raise ConformanceError("result must be a CampaignResult")
    with _CampaignOutputReservation(destination) as reservation:
        return reservation.write(result)


def write_campaign_corpus(result: CampaignResult, corpus_dir: str | Path) -> Path:
    """Commit a campaign under its deterministic corpus bundle identifier."""
    if not isinstance(result, CampaignResult):
        raise ConformanceError("result must be a CampaignResult")
    destination = Path(corpus_dir).expanduser() / result.bundle_id
    return write_campaign_artifacts(result, destination)
