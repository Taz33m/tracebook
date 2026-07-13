"""Versioned data contracts for matching-engine conformance checks."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_EVEN, localcontext
from numbers import Integral, Real
from typing import Any, Mapping, Optional, Sequence, Tuple, cast

from ..core.order import SelfTradePolicy

PROTOCOL_NAME = "tracebook.conformance"
PROTOCOL_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
_REASON_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
_HASH_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_DECIMAL_PATTERN = re.compile(r"^(?:0|[1-9]\d*)(?:\.\d*[1-9])?$")


class ConformanceError(ValueError):
    """Raised when a conformance contract or adapter is invalid."""


def canonical_decimal(value: Any, decimal_places: Optional[int] = None) -> str:
    """Return a finite number as a stable, non-exponent decimal string."""
    if isinstance(value, bool):
        raise ConformanceError(f"numeric value cannot be boolean: {value!r}")
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ConformanceError(f"numeric value is invalid: {value!r}") from exc
    if not number.is_finite():
        raise ConformanceError(f"numeric value must be finite: {value!r}")
    if decimal_places is not None:
        if not isinstance(decimal_places, int) or isinstance(decimal_places, bool):
            raise ConformanceError("decimal_places must be an integer")
        if not 0 <= decimal_places <= 18:
            raise ConformanceError("decimal_places must be between 0 and 18")
        quantum = Decimal(1).scaleb(-decimal_places)
        integer_digits = max(number.adjusted() + 1, 1)
        with localcontext() as context:
            context.prec = max(
                28,
                len(number.as_tuple().digits),
                integer_digits + decimal_places + 2,
            )
            try:
                number = number.quantize(quantum, rounding=ROUND_HALF_EVEN)
            except InvalidOperation as exc:
                raise ConformanceError(f"numeric value cannot be normalized: {value!r}") from exc
    if number == 0:
        return "0"
    return format(number.normalize(), "f")


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ConformanceError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ConformanceError(f"{field_name} must be a non-negative integer")
    return int(value)


def _canonical_decimal_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or len(value) > 400 or not _DECIMAL_PATTERN.fullmatch(value):
        raise ConformanceError(f"{field_name} must be a canonical decimal string")
    normalized = canonical_decimal(value)
    if value != normalized:
        raise ConformanceError(f"{field_name} must be canonical {normalized!r}, not {value!r}")
    return normalized


def _positive_decimal_string(value: Any, field_name: str) -> str:
    normalized = _canonical_decimal_string(value, field_name)
    if Decimal(normalized) <= 0:
        raise ConformanceError(f"{field_name} must be positive")
    return normalized


@dataclass(frozen=True)
class ConformanceConfig:
    """Matching semantics and numeric normalization used by a comparison."""

    matching_algorithm: str = "fifo"
    tick_size: float = 0.01
    self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE
    quantity_decimal_places: int = 12

    def __post_init__(self) -> None:
        if not isinstance(self.matching_algorithm, str):
            raise ConformanceError("matching_algorithm must be 'fifo' or 'pro_rata'")
        algorithm = self.matching_algorithm.strip().lower()
        if algorithm not in {"fifo", "pro_rata"}:
            raise ConformanceError("matching_algorithm must be 'fifo' or 'pro_rata'")
        if isinstance(self.tick_size, bool) or not isinstance(self.tick_size, Real):
            raise ConformanceError("tick_size must be a positive finite number")
        tick_size = float(self.tick_size)
        if not math.isfinite(tick_size) or tick_size <= 0:
            raise ConformanceError("tick_size must be a positive finite number")
        if isinstance(self.self_trade_policy, bool):
            raise ConformanceError("self_trade_policy must be a SelfTradePolicy value")
        try:
            policy = SelfTradePolicy(self.self_trade_policy)
        except (TypeError, ValueError) as exc:
            raise ConformanceError("self_trade_policy must be a SelfTradePolicy value") from exc
        if (
            isinstance(self.quantity_decimal_places, bool)
            or not isinstance(self.quantity_decimal_places, Integral)
            or not 0 <= self.quantity_decimal_places <= 18
        ):
            raise ConformanceError("quantity_decimal_places must be between 0 and 18")
        object.__setattr__(self, "matching_algorithm", algorithm)
        object.__setattr__(self, "tick_size", tick_size)
        object.__setattr__(self, "self_trade_policy", policy)
        object.__setattr__(self, "quantity_decimal_places", int(self.quantity_decimal_places))

    def to_dict(self) -> dict:
        return {
            "matching_algorithm": self.matching_algorithm,
            "tick_size": canonical_decimal(self.tick_size),
            "self_trade_policy": self.self_trade_policy.name,
            "quantity_decimal_places": self.quantity_decimal_places,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ConformanceConfig":
        if not isinstance(data, Mapping):
            raise ConformanceError("config must be an object")
        tick_value = data.get("tick_size", 0.01)
        try:
            tick_size = float(tick_value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ConformanceError("tick_size must be a positive finite number") from exc
        policy_value = data.get("self_trade_policy", "NONE")
        if isinstance(policy_value, str):
            try:
                policy = SelfTradePolicy[policy_value.strip().upper()]
            except KeyError as exc:
                raise ConformanceError(
                    "self_trade_policy must be NONE, CANCEL_RESTING, or CANCEL_INCOMING"
                ) from exc
        else:
            try:
                policy = SelfTradePolicy(policy_value)
            except (TypeError, ValueError) as exc:
                raise ConformanceError(
                    "self_trade_policy must be NONE, CANCEL_RESTING, or CANCEL_INCOMING"
                ) from exc
        return cls(
            matching_algorithm=data.get("matching_algorithm", "fifo"),
            tick_size=tick_size,
            self_trade_policy=policy,
            quantity_decimal_places=data.get("quantity_decimal_places", 12),
        )


@dataclass(frozen=True)
class EngineMetadata:
    """Identity reported by one engine adapter."""

    name: str
    version: str
    language: str

    def __post_init__(self) -> None:
        for field_name in ("name", "version", "language"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ConformanceError(f"engine {field_name} must be a non-empty string")
            object.__setattr__(self, field_name, value.strip())

    def to_dict(self) -> dict:
        return {"name": self.name, "version": self.version, "language": self.language}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EngineMetadata":
        if not isinstance(data, Mapping):
            raise ConformanceError("engine metadata must be an object")
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            language=data.get("language", ""),
        )


@dataclass(frozen=True)
class Outcome:
    """Canonical applied/rejected result for one source event."""

    status: str
    reason: Optional[str] = None
    message: Optional[str] = None

    def __post_init__(self) -> None:
        if self.status not in {"applied", "rejected"}:
            raise ConformanceError("outcome status must be 'applied' or 'rejected'")
        if self.status == "applied" and self.reason is not None:
            raise ConformanceError("an applied outcome cannot have a rejection reason")
        if self.status == "rejected":
            if not isinstance(self.reason, str) or not _REASON_PATTERN.fullmatch(self.reason):
                raise ConformanceError("a rejected outcome requires an uppercase reason code")
        if self.message is not None and not isinstance(self.message, str):
            raise ConformanceError("outcome message must be a string or null")

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Outcome":
        if not isinstance(data, Mapping):
            raise ConformanceError("outcome must be an object")
        return cls(
            status=data.get("status", ""),
            reason=data.get("reason"),
            message=data.get("message"),
        )


@dataclass(frozen=True)
class TradeFill:
    """Trade fields that are semantically comparable across engines."""

    symbol: str
    buy_order_id: int
    sell_order_id: int
    price: str
    quantity: str

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ConformanceError("trade symbol must be a non-empty string")
        object.__setattr__(self, "buy_order_id", _positive_int(self.buy_order_id, "buy_order_id"))
        object.__setattr__(
            self, "sell_order_id", _positive_int(self.sell_order_id, "sell_order_id")
        )
        object.__setattr__(self, "price", _positive_decimal_string(self.price, "trade price"))
        object.__setattr__(
            self,
            "quantity",
            _positive_decimal_string(self.quantity, "trade quantity"),
        )

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "buy_order_id": self.buy_order_id,
            "sell_order_id": self.sell_order_id,
            "price": self.price,
            "quantity": self.quantity,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "TradeFill":
        if not isinstance(data, Mapping):
            raise ConformanceError("trade must be an object")
        return cls(
            symbol=data.get("symbol", ""),
            buy_order_id=cast(int, data.get("buy_order_id")),
            sell_order_id=cast(int, data.get("sell_order_id")),
            price=cast(str, data.get("price")),
            quantity=cast(str, data.get("quantity")),
        )


@dataclass(frozen=True)
class RestingOrder:
    """One source order in its current queue position."""

    order_id: int
    price: str
    remaining_quantity: str
    owner: int
    order_type: str = "LIMIT"

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _positive_int(self.order_id, "order_id"))
        object.__setattr__(
            self,
            "price",
            _positive_decimal_string(self.price, "resting order price"),
        )
        object.__setattr__(
            self,
            "remaining_quantity",
            _positive_decimal_string(self.remaining_quantity, "resting order remaining_quantity"),
        )
        if isinstance(self.owner, bool) or not isinstance(self.owner, Integral):
            raise ConformanceError("order owner must be an integer")
        object.__setattr__(self, "owner", int(self.owner))
        if self.order_type != "LIMIT":
            raise ConformanceError("only LIMIT orders may appear in resting state")

    def to_dict(self) -> dict:
        return {
            "order_id": self.order_id,
            "price": self.price,
            "remaining_quantity": self.remaining_quantity,
            "owner": self.owner,
            "order_type": self.order_type,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RestingOrder":
        if not isinstance(data, Mapping):
            raise ConformanceError("resting order must be an object")
        return cls(
            order_id=cast(int, data.get("order_id")),
            price=cast(str, data.get("price")),
            remaining_quantity=cast(str, data.get("remaining_quantity")),
            owner=cast(int, data.get("owner")),
            order_type=data.get("order_type", "LIMIT"),
        )


@dataclass(frozen=True)
class BookSnapshot:
    """Both matching-priority queues for one symbol."""

    symbol: str
    bids: Tuple[RestingOrder, ...] = ()
    asks: Tuple[RestingOrder, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.symbol, str) or not self.symbol:
            raise ConformanceError("book symbol must be a non-empty string")
        seen = set()
        for order in self.bids + self.asks:
            if order.order_id in seen:
                raise ConformanceError(
                    f"book {self.symbol} contains duplicate order id {order.order_id}"
                )
            seen.add(order.order_id)

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bids": [order.to_dict() for order in self.bids],
            "asks": [order.to_dict() for order in self.asks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookSnapshot":
        if not isinstance(data, Mapping):
            raise ConformanceError("book snapshot must be an object")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not isinstance(bids, list) or not isinstance(asks, list):
            raise ConformanceError("book bids and asks must be arrays")
        return cls(
            symbol=data.get("symbol", ""),
            bids=tuple(RestingOrder.from_dict(item) for item in bids),
            asks=tuple(RestingOrder.from_dict(item) for item in asks),
        )


@dataclass(frozen=True)
class BookState:
    """Canonical full resting state, including queue priority."""

    books: Tuple[BookSnapshot, ...] = ()

    def __post_init__(self) -> None:
        symbols = [book.symbol for book in self.books]
        if symbols != sorted(symbols):
            raise ConformanceError("book snapshots must be sorted by symbol")
        if len(symbols) != len(set(symbols)):
            raise ConformanceError("book state contains duplicate symbols")

    @property
    def order_count(self) -> int:
        return sum(len(book.bids) + len(book.asks) for book in self.books)

    def to_dict(self) -> dict:
        return {"books": [book.to_dict() for book in self.books]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookState":
        if not isinstance(data, Mapping):
            raise ConformanceError("state must be an object")
        books = data.get("books", [])
        if not isinstance(books, list):
            raise ConformanceError("state books must be an array")
        return cls(tuple(BookSnapshot.from_dict(item) for item in books))

    def digest(self) -> str:
        payload = json.dumps(
            self.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class Observation:
    """Compact result emitted after one event."""

    index: int
    outcome: Outcome
    trades: Tuple[TradeFill, ...]
    state_hash: str
    resting_order_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _positive_int(self.index, "observation index"))
        if not isinstance(self.state_hash, str) or not _HASH_PATTERN.fullmatch(self.state_hash):
            raise ConformanceError("state_hash must be a lowercase SHA-256 hex digest")
        object.__setattr__(
            self,
            "resting_order_count",
            _nonnegative_int(self.resting_order_count, "resting_order_count"),
        )

    def to_dict(self, include_type: bool = False) -> dict:
        payload = {
            "index": self.index,
            "outcome": self.outcome.to_dict(),
            "trades": [trade.to_dict() for trade in self.trades],
            "state_hash": self.state_hash,
            "resting_order_count": self.resting_order_count,
        }
        if include_type:
            payload = {"type": "observation", **payload}
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Observation":
        if not isinstance(data, Mapping):
            raise ConformanceError("observation must be an object")
        trades = data.get("trades", [])
        if not isinstance(trades, list):
            raise ConformanceError("observation trades must be an array")
        return cls(
            index=cast(int, data.get("index")),
            outcome=Outcome.from_dict(data.get("outcome", {})),
            trades=tuple(TradeFill.from_dict(item) for item in trades),
            state_hash=data.get("state_hash", ""),
            resting_order_count=cast(int, data.get("resting_order_count")),
        )


def trace_sha256(events: Sequence[Any]) -> str:
    """Hash normalized events using the canonical JSONL representation."""
    digest = hashlib.sha256()
    for event in events:
        try:
            payload = event.to_dict()
        except AttributeError as exc:
            raise ConformanceError("trace entries must provide to_dict()") from exc
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\n")
    return "sha256:" + digest.hexdigest()


def first_difference(reference: Any, candidate: Any, path: str = "$") -> Optional[dict]:
    """Return the first deterministic structural difference between two values."""
    if type(reference) is not type(candidate):
        return {
            "kind": "type_mismatch",
            "path": path,
            "reference": reference,
            "candidate": candidate,
        }
    if isinstance(reference, dict):
        for key in sorted(set(reference) | set(candidate)):
            child_path = f"{path}.{key}"
            if key not in reference:
                return {
                    "kind": "unexpected_field",
                    "path": child_path,
                    "reference": None,
                    "candidate": candidate[key],
                }
            if key not in candidate:
                return {
                    "kind": "missing_field",
                    "path": child_path,
                    "reference": reference[key],
                    "candidate": None,
                }
            difference = first_difference(reference[key], candidate[key], child_path)
            if difference is not None:
                return difference
        return None
    if isinstance(reference, list):
        common = min(len(reference), len(candidate))
        for index in range(common):
            difference = first_difference(reference[index], candidate[index], f"{path}[{index}]")
            if difference is not None:
                return difference
        if len(reference) != len(candidate):
            return {
                "kind": "length_mismatch",
                "path": path,
                "reference": len(reference),
                "candidate": len(candidate),
            }
        return None
    if reference != candidate:
        return {
            "kind": "value_mismatch",
            "path": path,
            "reference": reference,
            "candidate": candidate,
        }
    return None
