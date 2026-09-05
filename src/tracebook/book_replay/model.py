"""Versioned contracts for deterministic L3 order-book replay checks."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from numbers import Integral
from typing import Any, Iterable, Mapping, Optional, Tuple, cast

from ..conformance.model import EngineMetadata, Outcome, canonical_decimal

PROTOCOL_NAME = "tracebook.book-replay"
PROTOCOL_VERSION = 1
ARTIFACT_SCHEMA_VERSION = 1
PROFILE_NAME = "l3-book-replay-v1"
_HASH_LENGTH = 64


class BookReplayError(ValueError):
    """Raised when a book-replay trace, contract, or adapter is invalid."""


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise BookReplayError(f"{field_name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise BookReplayError(f"{field_name} must be a non-negative integer")
    return int(value)


def _positive_decimal(value: Any, field_name: str) -> str:
    try:
        normalized = canonical_decimal(value)
    except ValueError as exc:
        raise BookReplayError(str(exc)) from exc
    if Decimal(normalized) <= 0:
        raise BookReplayError(f"{field_name} must be positive")
    return normalized


def _symbol(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise BookReplayError("symbol must be a non-empty string")
    return value.strip()


def _side(value: Any) -> str:
    if not isinstance(value, str):
        raise BookReplayError("side must be BUY or SELL")
    normalized = value.strip().upper()
    if normalized not in {"BUY", "SELL"}:
        raise BookReplayError("side must be BUY or SELL")
    return normalized


@dataclass(frozen=True)
class BookReplayConfig:
    """The immutable semantic profile selected for one comparison."""

    profile: str = PROFILE_NAME

    def __post_init__(self) -> None:
        if self.profile != PROFILE_NAME:
            raise BookReplayError(f"profile must be {PROFILE_NAME!r}")

    def to_dict(self) -> dict:
        return {"profile": self.profile}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookReplayConfig":
        if not isinstance(data, Mapping):
            raise BookReplayError("config must be an object")
        return cls(profile=data.get("profile", PROFILE_NAME))


@dataclass(frozen=True)
class BookReplayEvent:
    """One L3 delta or non-mutating simulated-fill probe."""

    op: str
    symbol: str
    order_id: Optional[int] = None
    side: Optional[str] = None
    price: Optional[str] = None
    quantity: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.op, str):
            raise BookReplayError("op must be add, update, delete, or probe")
        op = self.op.strip().lower()
        if op not in {"add", "update", "delete", "probe"}:
            raise BookReplayError("op must be add, update, delete, or probe")
        object.__setattr__(self, "op", op)
        object.__setattr__(self, "symbol", _symbol(self.symbol))

        if op in {"add", "update", "delete"}:
            object.__setattr__(self, "order_id", _positive_int(self.order_id, "order_id"))
        elif self.order_id is not None:
            raise BookReplayError("probe events cannot contain order_id")

        if op in {"add", "update", "probe"}:
            object.__setattr__(self, "side", _side(self.side))
            object.__setattr__(self, "price", _positive_decimal(self.price, "price"))
            object.__setattr__(
                self,
                "quantity",
                _positive_decimal(self.quantity, "quantity"),
            )
        elif any(value is not None for value in (self.side, self.price, self.quantity)):
            raise BookReplayError("delete events contain only op, symbol, and order_id")

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {"op": self.op, "symbol": self.symbol}
        if self.order_id is not None:
            payload["order_id"] = self.order_id
        if self.side is not None:
            payload["side"] = self.side
        if self.price is not None:
            payload["price"] = self.price
        if self.quantity is not None:
            payload["quantity"] = self.quantity
        return payload

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "BookReplayEvent":
        if not isinstance(data, Mapping):
            raise BookReplayError("book-replay event must be an object")
        op = data.get("op", "")
        normalized_op = op.strip().lower() if isinstance(op, str) else op
        allowed = {
            "add": {"op", "symbol", "order_id", "side", "price", "quantity"},
            "update": {"op", "symbol", "order_id", "side", "price", "quantity"},
            "delete": {"op", "symbol", "order_id"},
            "probe": {"op", "symbol", "side", "price", "quantity"},
        }.get(normalized_op)
        if allowed is not None:
            unknown = set(data) - allowed
            if unknown:
                raise BookReplayError(
                    f"{normalized_op} event contains unsupported fields: {sorted(unknown)!r}"
                )
        return cls(
            op=cast(str, op),
            symbol=cast(str, data.get("symbol", "")),
            order_id=cast(Optional[int], data.get("order_id")),
            side=cast(Optional[str], data.get("side")),
            price=cast(Optional[str], data.get("price")),
            quantity=cast(Optional[str], data.get("quantity")),
        )


@dataclass(frozen=True)
class SimulatedFill:
    """One price/quantity segment returned by a non-mutating probe."""

    price: str
    quantity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "price", _positive_decimal(self.price, "fill price"))
        object.__setattr__(self, "quantity", _positive_decimal(self.quantity, "fill quantity"))

    def to_dict(self) -> dict:
        return {"price": self.price, "quantity": self.quantity}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SimulatedFill":
        if not isinstance(data, Mapping):
            raise BookReplayError("simulated fill must be an object")
        return cls(price=cast(str, data.get("price")), quantity=cast(str, data.get("quantity")))


@dataclass(frozen=True)
class RestingBookOrder:
    """One mirrored L3 order in queue position."""

    order_id: int
    price: str
    quantity: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "order_id", _positive_int(self.order_id, "order_id"))
        object.__setattr__(self, "price", _positive_decimal(self.price, "order price"))
        object.__setattr__(self, "quantity", _positive_decimal(self.quantity, "order quantity"))

    def to_dict(self) -> dict:
        return {"order_id": self.order_id, "price": self.price, "quantity": self.quantity}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RestingBookOrder":
        if not isinstance(data, Mapping):
            raise BookReplayError("resting order must be an object")
        return cls(
            order_id=cast(int, data.get("order_id")),
            price=cast(str, data.get("price")),
            quantity=cast(str, data.get("quantity")),
        )


@dataclass(frozen=True)
class BookReplaySnapshot:
    """Both price-time queues for one mirrored symbol."""

    symbol: str
    bids: Tuple[RestingBookOrder, ...] = ()
    asks: Tuple[RestingBookOrder, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _symbol(self.symbol))
        for name in ("bids", "asks"):
            orders = getattr(self, name)
            if not isinstance(orders, tuple) or any(
                not isinstance(order, RestingBookOrder) for order in orders
            ):
                raise BookReplayError(f"{name} must be a tuple of RestingBookOrder values")
        order_ids = [order.order_id for order in self.bids + self.asks]
        if len(order_ids) != len(set(order_ids)):
            raise BookReplayError(f"book {self.symbol!r} contains duplicate order ids")

    def to_dict(self) -> dict:
        return {
            "symbol": self.symbol,
            "bids": [order.to_dict() for order in self.bids],
            "asks": [order.to_dict() for order in self.asks],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookReplaySnapshot":
        if not isinstance(data, Mapping):
            raise BookReplayError("book snapshot must be an object")
        bids = data.get("bids", [])
        asks = data.get("asks", [])
        if not isinstance(bids, list) or not isinstance(asks, list):
            raise BookReplayError("book bids and asks must be arrays")
        return cls(
            symbol=cast(str, data.get("symbol", "")),
            bids=tuple(RestingBookOrder.from_dict(item) for item in bids),
            asks=tuple(RestingBookOrder.from_dict(item) for item in asks),
        )


@dataclass(frozen=True)
class BookReplayState:
    """Canonical mirrored state, including L3 queue order."""

    books: Tuple[BookReplaySnapshot, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.books, tuple) or any(
            not isinstance(book, BookReplaySnapshot) for book in self.books
        ):
            raise BookReplayError("books must be a tuple of BookReplaySnapshot values")
        symbols = [book.symbol for book in self.books]
        if symbols != sorted(symbols) or len(symbols) != len(set(symbols)):
            raise BookReplayError("book snapshots must have unique symbols in sorted order")

    @property
    def order_count(self) -> int:
        return sum(len(book.bids) + len(book.asks) for book in self.books)

    def to_dict(self) -> dict:
        return {"books": [book.to_dict() for book in self.books]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookReplayState":
        if not isinstance(data, Mapping):
            raise BookReplayError("state must be an object")
        books = data.get("books", [])
        if not isinstance(books, list):
            raise BookReplayError("state books must be an array")
        return cls(tuple(BookReplaySnapshot.from_dict(item) for item in books))

    def digest(self) -> str:
        return hashlib.sha256(_canonical_json(self.to_dict())).hexdigest()


@dataclass(frozen=True)
class BookReplayObservation:
    """Comparable result emitted after one delta or probe."""

    index: int
    outcome: Outcome
    fills: Tuple[SimulatedFill, ...]
    state_hash: str
    order_count: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "index", _positive_int(self.index, "observation index"))
        if not isinstance(self.outcome, Outcome):
            raise BookReplayError("observation outcome must be an Outcome")
        if not isinstance(self.fills, tuple) or any(
            not isinstance(fill, SimulatedFill) for fill in self.fills
        ):
            raise BookReplayError("observation fills must be a tuple of SimulatedFill values")
        if (
            not isinstance(self.state_hash, str)
            or len(self.state_hash) != _HASH_LENGTH
            or any(character not in "0123456789abcdef" for character in self.state_hash)
        ):
            raise BookReplayError("state_hash must be a lowercase SHA-256 hex digest")
        object.__setattr__(
            self,
            "order_count",
            _nonnegative_int(self.order_count, "order_count"),
        )

    def to_dict(self, include_type: bool = False) -> dict:
        payload = {
            "index": self.index,
            "outcome": self.outcome.to_dict(),
            "fills": [fill.to_dict() for fill in self.fills],
            "state_hash": self.state_hash,
            "order_count": self.order_count,
        }
        if include_type:
            payload = {"type": "observation", **payload}
        return payload

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "BookReplayObservation":
        if not isinstance(data, Mapping):
            raise BookReplayError("observation must be an object")
        fills = data.get("fills", [])
        if not isinstance(fills, list):
            raise BookReplayError("observation fills must be an array")
        return cls(
            index=cast(int, data.get("index")),
            outcome=Outcome.from_dict(data.get("outcome", {})),
            fills=tuple(SimulatedFill.from_dict(item) for item in fills),
            state_hash=cast(str, data.get("state_hash")),
            order_count=cast(int, data.get("order_count")),
        )


def load_book_replay_events(path: str) -> list[BookReplayEvent]:
    """Load a newline-delimited JSON book-replay trace."""
    events = []
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise BookReplayError(f"invalid JSON on line {line_number}: {exc.msg}") from exc
            try:
                events.append(BookReplayEvent.from_mapping(payload))
            except (TypeError, ValueError) as exc:
                raise BookReplayError(f"invalid event on line {line_number}: {exc}") from exc
    return events


def trace_sha256(events: Iterable[BookReplayEvent]) -> str:
    """Hash normalized events independently of input whitespace."""
    digest = hashlib.sha256()
    for event in events:
        digest.update(_canonical_json(event.to_dict()))
        digest.update(b"\n")
    return digest.hexdigest()


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROFILE_NAME",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "BookReplayConfig",
    "BookReplayError",
    "BookReplayEvent",
    "BookReplayObservation",
    "BookReplaySnapshot",
    "BookReplayState",
    "EngineMetadata",
    "Outcome",
    "RestingBookOrder",
    "SimulatedFill",
    "load_book_replay_events",
    "trace_sha256",
]
