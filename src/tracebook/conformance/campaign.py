"""Deterministic stateful campaigns for differential matching-engine testing."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, TypeVar

from ..core.order import OrderSide, OrderType
from ..events import MarketEvent
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

CAMPAIGN_GENERATOR_VERSION = 1
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
    ),
    "fifo-full-v1": CampaignProfile(
        name="fifo-full-v1",
        description=("The FIFO lifecycle profile plus MARKET, IOC, and FOK instructions."),
        config=ConformanceConfig(matching_algorithm="fifo"),
        order_types=(OrderType.LIMIT, OrderType.MARKET, OrderType.IOC, OrderType.FOK),
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
    reference = ReferenceEngineAdapter(selected_profile.config)
    events: List[MarketEvent] = []
    active: Tuple[_ActiveOrder, ...] = ()
    try:
        for index in range(1, event_count + 1):
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
    minimization: MinimizationResult

    def to_dict(self) -> dict:
        minimized = self.minimization
        return {
            "trace_index": self.trace.index,
            "target_category": minimized.target_category,
            "original_event_count": len(self.trace.events),
            "original_trace_sha256": trace_sha256(self.trace.events),
            "original_events": "failure/original.jsonl",
            "original_report": "failure/original-report.json",
            "minimized_event_count": len(minimized.events),
            "minimized_trace_sha256": trace_sha256(minimized.events),
            "minimized_events": "failure/minimized.jsonl",
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

    def to_dict(self) -> dict:
        minimization_runs = self.failure.minimization.runs if self.failure else 0
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
            "traces": [trace.to_dict() for trace in self.traces],
            "failure": self.failure.to_dict() if self.failure else None,
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
            minimization = minimize_failing_trace(
                events,
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
            failure = CampaignFailure(trace_result, minimization)
            break

    if candidate_engine is None:
        raise ConformanceError("campaign completed without candidate metadata")
    return CampaignResult(
        profile=selected_profile,
        seed=seed,
        requested_traces=traces,
        events_per_trace=events_per_trace,
        max_minimize_runs=max_minimize_runs,
        candidate_engine=candidate_engine,
        traces=tuple(results),
        failure=failure,
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


def _path_occupied(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def write_campaign_artifacts(result: CampaignResult, destination: str | Path) -> Path:
    """Atomically persist a campaign summary and its optional failure bundle."""
    if not isinstance(result, CampaignResult):
        raise ConformanceError("result must be a CampaignResult")
    target = Path(destination).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    lock = target.with_name(f".{target.name}.lock")
    try:
        lock.mkdir()
    except FileExistsError as exc:
        raise ConformanceError(f"campaign output is locked: {target}") from exc
    except OSError as exc:
        raise ConformanceError(f"could not reserve campaign output {target}: {exc}") from exc

    temporary: Optional[Path] = None
    try:
        if _path_occupied(target):
            raise ConformanceError(f"campaign output already exists: {target}")
        temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}.", dir=str(target.parent)))
        if result.failure is not None:
            failure_dir = temporary / "failure"
            failure_dir.mkdir()
            _write_events(failure_dir / "original.jsonl", result.failure.trace.events)
            _write_json(
                failure_dir / "original-report.json",
                result.failure.trace.report.to_dict(),
            )
            _write_events(
                failure_dir / "minimized.jsonl",
                result.failure.minimization.events,
            )
            _write_json(
                failure_dir / "minimization.json",
                result.failure.minimization.to_dict(),
            )
        _write_json(temporary / "campaign.json", result.to_dict())
        if _path_occupied(target):
            raise ConformanceError(f"campaign output already exists: {target}")
        try:
            temporary.rename(target)
        except OSError as exc:
            raise ConformanceError(f"could not commit campaign output {target}: {exc}") from exc
        temporary = None
    finally:
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)
        shutil.rmtree(lock, ignore_errors=True)
    return target / "campaign.json"
