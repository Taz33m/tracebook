"""Normalize Coinbase Exchange L3 snapshots and feed messages.

The adapter is deliberately offline and dependency-free: callers provide a
REST level-3 snapshot and an iterable of previously captured ``full`` channel
objects or compact ``level3`` arrays. The resulting ``MarketEvent`` stream can
be replayed directly without retaining the full source feed in memory.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, cast

from ..core.order import OrderSide, normalize_symbol
from .market_replay import MarketEvent, MarketReplayError

_BOOK_MESSAGE_TYPES = {"received", "open", "done", "match", "change", "noop"}
_CONTROL_MESSAGE_TYPES = {
    "subscriptions",
    "heartbeat",
    "status",
    "ticker",
    "ticker_batch",
    "last_match",
    "rfq_match",
    "activate",
}
_L3_REQUIRED_FIELDS = {
    "open": {
        "type",
        "product_id",
        "sequence",
        "order_id",
        "side",
        "price",
        "size",
        "time",
    },
    "done": {"type", "product_id", "sequence", "order_id", "time"},
    "match": {
        "type",
        "product_id",
        "sequence",
        "maker_order_id",
        "taker_order_id",
        "price",
        "size",
        "time",
    },
    "change": {"type", "product_id", "sequence", "order_id", "price", "size", "time"},
    "noop": {"type", "product_id", "sequence", "time"},
}
_RFC3339_PATTERN = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"
    r"(?:\.(?P<fraction>\d{1,9}))?(?P<zone>Z|[+-]\d{2}:\d{2})$"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_MAX_SOURCE_ID = (1 << 63) - 1

RawCoinbaseMessage = Mapping[str, Any] | Sequence[Any]


class CoinbaseL3Error(MarketReplayError):
    """Raised when Coinbase L3 input cannot be normalized safely."""


def _text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoinbaseL3Error(f"{field_name} must be a non-empty string")
    return value.strip()


def _integer(value: Any, field_name: str, required: bool = True) -> Optional[int]:
    if value is None:
        if required:
            raise CoinbaseL3Error(f"{field_name} is required")
        return None
    if isinstance(value, bool):
        raise CoinbaseL3Error(f"{field_name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    raise CoinbaseL3Error(f"{field_name} must be an integer: {value!r}")


def _decimal(
    value: Any, field_name: str, *, allow_zero: bool = False, required: bool = True
) -> Optional[Decimal]:
    if value is None:
        if required:
            raise CoinbaseL3Error(f"{field_name} is required")
        return None
    if isinstance(value, bool):
        raise CoinbaseL3Error(f"{field_name} must be numeric")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CoinbaseL3Error(f"{field_name} must be numeric: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0 or (not allow_zero and parsed == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise CoinbaseL3Error(f"{field_name} must be finite and {qualifier}: {value!r}")
    return parsed


def _float(value: Decimal, field_name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise CoinbaseL3Error(f"{field_name} cannot be represented as a positive float")
    return parsed


def _side(value: Any) -> OrderSide:
    normalized = _text(value, "side").lower()
    if normalized == "buy":
        return OrderSide.BUY
    if normalized == "sell":
        return OrderSide.SELL
    raise CoinbaseL3Error(f"side must be buy or sell: {value!r}")


def _product_id(value: Any) -> str:
    return normalize_symbol(_text(value, "product_id")).upper()


def _timestamp_ns(value: Any) -> Optional[int]:
    if value is None:
        return None
    raw = _text(value, "time")
    match = _RFC3339_PATTERN.fullmatch(raw)
    if match is None:
        raise CoinbaseL3Error(f"time must be an RFC3339 timestamp: {raw!r}")
    zone = "+00:00" if match.group("zone") == "Z" else match.group("zone")
    try:
        base = datetime.fromisoformat(match.group("date") + zone).astimezone(timezone.utc)
    except ValueError as exc:
        raise CoinbaseL3Error(f"time must be an RFC3339 timestamp: {raw!r}") from exc
    delta = base - _EPOCH
    whole_seconds = delta.days * 86_400 + delta.seconds
    fraction = (match.group("fraction") or "").ljust(9, "0")
    return whole_seconds * 1_000_000_000 + (int(fraction) if fraction else 0)


def coinbase_order_id(product_id: str, venue_order_id: str) -> int:
    """Return a deterministic positive 63-bit id for a Coinbase order id."""
    product = _product_id(product_id)
    venue_id = _text(venue_order_id, "order_id")
    digest = hashlib.blake2b(
        f"{product}\0{venue_id}".encode("utf-8"),
        digest_size=8,
        person=b"tracebook-l3",
    ).digest()
    normalized = int.from_bytes(digest, "big") & _MAX_SOURCE_ID
    return normalized or 1


@dataclass(frozen=True)
class CoinbaseAdapterIssue:
    """One feed problem retained by lenient normalization."""

    message_index: int
    kind: str
    reason: str
    sequence: Optional[int] = None


@dataclass(frozen=True)
class CoinbaseExchangeTrade:
    """An observed Coinbase match, separate from simulated engine trades."""

    sequence: int
    product_id: str
    maker_order_id: str
    taker_order_id: str
    normalized_maker_order_id: int
    normalized_taker_order_id: int
    price: float
    quantity: float
    timestamp_ns: Optional[int]
    trade_id: Optional[int] = None
    maker_side: Optional[str] = None


@dataclass
class _ActiveOrder:
    venue_order_id: str
    normalized_order_id: int
    side: OrderSide
    price: Decimal
    remaining: Decimal
    timestamp_ns: Optional[int]


class CoinbaseL3Adapter:
    """Stateful, single-pass Coinbase Exchange L3 normalizer."""

    def __init__(
        self,
        snapshot: Mapping[str, Any],
        product_id: Optional[str] = None,
        *,
        strict: bool = True,
        retain_id_map: bool = False,
        retain_trades: bool = False,
    ) -> None:
        if not isinstance(snapshot, Mapping):
            raise CoinbaseL3Error("snapshot must be a JSON object")
        if not isinstance(strict, bool):
            raise CoinbaseL3Error("strict must be a boolean")
        if not isinstance(retain_id_map, bool):
            raise CoinbaseL3Error("retain_id_map must be a boolean")
        if not isinstance(retain_trades, bool):
            raise CoinbaseL3Error("retain_trades must be a boolean")

        source_product = product_id if product_id is not None else snapshot.get("product_id")
        self.product_id = _product_id(source_product)
        sequence = _integer(snapshot.get("sequence"), "snapshot.sequence")
        if sequence is None or sequence < 0:
            raise CoinbaseL3Error("snapshot.sequence must be non-negative")

        self.strict = strict
        self.retain_id_map = retain_id_map
        self.retain_trades = retain_trades
        self.snapshot_sequence = sequence
        self.final_sequence = sequence
        self.snapshot_orders = 0
        self.messages_seen = 0
        self.messages_sequenced = 0
        self.messages_ignored = 0
        self.normalized_events = 0
        self.sequence_complete = True
        self.issues: List[CoinbaseAdapterIssue] = []
        self.exchange_trades_observed = 0
        self.exchange_trades: List[CoinbaseExchangeTrade] = []
        self._active: Dict[str, _ActiveOrder] = {}
        self._normalized_active: Dict[int, str] = {}
        self._id_map: Dict[int, str] = {}
        self._schema: Optional[Dict[str, Tuple[str, ...]]] = None
        self._started = False
        self._finished = False

        if snapshot.get("auction_mode") is True:
            raise CoinbaseL3Error("auction-mode snapshots are not supported")
        snapshot_time = _timestamp_ns(snapshot.get("time"))
        self._load_snapshot_side(snapshot.get("bids"), OrderSide.BUY, snapshot_time)
        self._load_snapshot_side(snapshot.get("asks"), OrderSide.SELL, snapshot_time)
        self._validate_uncrossed_snapshot()

    @property
    def active_order_count(self) -> int:
        return len(self._active)

    @property
    def normalization_complete(self) -> bool:
        return self._finished

    def iter_events(self, messages: Iterable[RawCoinbaseMessage]) -> Iterator[MarketEvent]:
        """Yield snapshot seed events followed by normalized feed events once."""
        if self._started:
            raise CoinbaseL3Error("a CoinbaseL3Adapter can only normalize one feed")
        self._started = True

        for order in self._active.values():
            yield self._count_event(
                MarketEvent(
                    op="new",
                    symbol=self.product_id,
                    order_id=order.normalized_order_id,
                    side=order.side,
                    price=_float(order.price, "snapshot price"),
                    quantity=_float(order.remaining, "snapshot size"),
                    timestamp_ns=order.timestamp_ns,
                )
            )

        for message_index, raw in enumerate(messages, 1):
            self.messages_seen += 1
            try:
                decoded = self._decode_message(raw)
                if decoded is None:
                    self.messages_ignored += 1
                    continue
                mapping, compact = decoded
                events = self._process_message(mapping, compact, message_index)
            except (CoinbaseL3Error, TypeError, ValueError) as exc:
                if self.strict:
                    raise CoinbaseL3Error(f"Message {message_index}: {exc}") from exc
                self.sequence_complete = False
                self.issues.append(CoinbaseAdapterIssue(message_index, "invalid_message", str(exc)))
                self.messages_ignored += 1
                continue

            for event in events:
                yield self._count_event(event)

        self._finished = True

    def to_dict(self, *, include_trades: bool = False, include_id_map: bool = False) -> dict:
        """Return a strict-JSON-ready normalization summary."""
        if not isinstance(include_trades, bool) or not isinstance(include_id_map, bool):
            raise CoinbaseL3Error("include_trades and include_id_map must be booleans")
        if include_id_map and not self.retain_id_map:
            raise CoinbaseL3Error("id map retention was not enabled for this adapter")
        if include_trades and not self.retain_trades:
            raise CoinbaseL3Error("trade retention was not enabled for this adapter")
        return {
            "schema_version": 1,
            "venue": "coinbase_exchange",
            "channel": "full_or_level3",
            "product_id": self.product_id,
            "snapshot_sequence": self.snapshot_sequence,
            "final_sequence": self.final_sequence,
            "sequence_complete": self.sequence_complete,
            "normalization_complete": self.normalization_complete,
            "snapshot_orders": self.snapshot_orders,
            "messages_seen": self.messages_seen,
            "messages_sequenced": self.messages_sequenced,
            "messages_ignored": self.messages_ignored,
            "normalized_events": self.normalized_events,
            "active_orders": self.active_order_count,
            "issues": [asdict(issue) for issue in self.issues],
            "exchange_trades_observed": self.exchange_trades_observed,
            "exchange_trades_included": include_trades,
            "exchange_trades": (
                [asdict(trade) for trade in self.exchange_trades] if include_trades else []
            ),
            "id_map_included": include_id_map,
            "id_map": (
                [
                    {"normalized_order_id": normalized, "venue_order_id": venue}
                    for normalized, venue in sorted(self._id_map.items())
                ]
                if include_id_map
                else []
            ),
        }

    def _count_event(self, event: MarketEvent) -> MarketEvent:
        self.normalized_events += 1
        return event

    def _load_snapshot_side(self, rows: Any, side: OrderSide, timestamp_ns: Optional[int]) -> None:
        field_name = "bids" if side == OrderSide.BUY else "asks"
        if not isinstance(rows, list):
            raise CoinbaseL3Error(f"snapshot.{field_name} must be an array")
        for row_index, row in enumerate(rows, 1):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise CoinbaseL3Error(
                    f"snapshot.{field_name}[{row_index}] must be [price, size, order_id]"
                )
            if not isinstance(row[2], str):
                raise CoinbaseL3Error(
                    f"snapshot.{field_name}[{row_index}] is not level 3; "
                    "the third value must be an order id"
                )
            price = cast(Decimal, _decimal(row[0], f"snapshot.{field_name}[{row_index}].price"))
            size = cast(Decimal, _decimal(row[1], f"snapshot.{field_name}[{row_index}].size"))
            self._add_active(_text(row[2], "order_id"), side, price, size, timestamp_ns)
            self.snapshot_orders += 1

    def _validate_uncrossed_snapshot(self) -> None:
        bids = [order.price for order in self._active.values() if order.side == OrderSide.BUY]
        asks = [order.price for order in self._active.values() if order.side == OrderSide.SELL]
        if bids and asks and max(bids) >= min(asks):
            raise CoinbaseL3Error("snapshot is crossed or locked; auction books are unsupported")

    def _decode_message(self, raw: RawCoinbaseMessage) -> Optional[Tuple[Dict[str, Any], bool]]:
        if isinstance(raw, Mapping):
            mapping = dict(raw)
            if mapping.get("type") == "level3" and "schema" in mapping:
                self._set_schema(mapping["schema"])
                return None
            return mapping, False
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            if self._schema is None:
                raise CoinbaseL3Error("compact level3 message arrived before its schema")
            if not raw:
                raise CoinbaseL3Error("compact level3 message cannot be empty")
            message_type = _text(raw[0], "type").lower()
            fields = self._schema.get(message_type)
            if fields is None:
                raise CoinbaseL3Error(f"compact schema has no fields for {message_type!r}")
            if len(raw) != len(fields):
                raise CoinbaseL3Error(
                    f"compact {message_type} has {len(raw)} values; expected {len(fields)}"
                )
            return dict(zip(fields, raw)), True
        raise CoinbaseL3Error("feed messages must be JSON objects or compact arrays")

    def _set_schema(self, raw_schema: Any) -> None:
        if not isinstance(raw_schema, Mapping):
            raise CoinbaseL3Error("level3 schema must be an object")
        parsed: Dict[str, Tuple[str, ...]] = {}
        for message_type, raw_fields in raw_schema.items():
            if not isinstance(message_type, str) or not isinstance(raw_fields, list):
                raise CoinbaseL3Error("level3 schema entries must map names to field arrays")
            fields = tuple(_text(field, "schema field") for field in raw_fields)
            if not fields or fields[0] != "type":
                raise CoinbaseL3Error(
                    f"level3 schema for {message_type!r} must begin with the type field"
                )
            if len(set(fields)) != len(fields):
                raise CoinbaseL3Error(f"level3 schema for {message_type!r} has duplicate fields")
            parsed[message_type.lower()] = fields
        for message_type, required in _L3_REQUIRED_FIELDS.items():
            field_set = set(parsed.get(message_type, ()))
            missing = required - field_set
            if missing:
                raise CoinbaseL3Error(
                    f"level3 schema for {message_type!r} is missing {sorted(missing)}"
                )
        self._schema = parsed

    def _process_message(
        self, mapping: Mapping[str, Any], compact: bool, message_index: int
    ) -> List[MarketEvent]:
        message_type = _text(mapping.get("type"), "type").lower()
        if message_type == "error":
            raise CoinbaseL3Error(f"feed returned an error: {mapping.get('message', 'unknown')}")
        if message_type in _CONTROL_MESSAGE_TYPES:
            self.messages_ignored += 1
            return []

        known_book_message = message_type in _BOOK_MESSAGE_TYPES
        product_value = mapping.get("product_id")
        if product_value is not None:
            product = _product_id(product_value)
            if product != self.product_id:
                self.messages_ignored += 1
                return []
        if product_value is None:
            if not known_book_message:
                self.messages_ignored += 1
                return []
            raise CoinbaseL3Error("product_id is required for book messages")
        if not known_book_message and mapping.get("sequence") is None:
            self.messages_ignored += 1
            return []

        sequence = cast(int, _integer(mapping.get("sequence"), "sequence"))
        if sequence <= self.snapshot_sequence or sequence <= self.final_sequence:
            self.messages_ignored += 1
            return []
        expected = self.final_sequence + 1
        if sequence != expected:
            self.sequence_complete = False
            reason = f"sequence gap: expected {expected}, received {sequence}"
            self.issues.append(
                CoinbaseAdapterIssue(message_index, "sequence_gap", reason, sequence)
            )
            if self.strict:
                raise CoinbaseL3Error(reason)
        self.final_sequence = sequence
        self.messages_sequenced += 1

        if not known_book_message or message_type in {"received", "noop"}:
            self.messages_ignored += 1
            return []
        if message_type == "open":
            return [self._open(mapping, compact)]
        if message_type == "match":
            return [self._match(mapping, compact, sequence)]
        if message_type == "done":
            return self._done(mapping, compact)
        if message_type == "change":
            return self._change(mapping, compact)
        raise CoinbaseL3Error(f"unhandled full-channel message type: {message_type!r}")

    def _open(self, mapping: Mapping[str, Any], compact: bool) -> MarketEvent:
        venue_id = _text(mapping.get("order_id"), "order_id")
        if venue_id in self._active:
            raise CoinbaseL3Error(f"order {venue_id} is already open")
        side = _side(mapping.get("side"))
        price = cast(Decimal, _decimal(mapping.get("price"), "price"))
        size_field = "size" if compact else "remaining_size"
        size = cast(Decimal, _decimal(mapping.get(size_field), size_field))
        timestamp_ns = _timestamp_ns(mapping.get("time"))
        order = self._add_active(venue_id, side, price, size, timestamp_ns)
        return MarketEvent(
            op="new",
            symbol=self.product_id,
            order_id=order.normalized_order_id,
            side=side,
            price=_float(price, "price"),
            quantity=_float(size, size_field),
            timestamp_ns=timestamp_ns,
        )

    def _match(self, mapping: Mapping[str, Any], compact: bool, sequence: int) -> MarketEvent:
        maker_id = _text(mapping.get("maker_order_id"), "maker_order_id")
        taker_id = _text(mapping.get("taker_order_id"), "taker_order_id")
        order = self._active.get(maker_id)
        if order is None:
            raise CoinbaseL3Error(f"match references non-resting maker order {maker_id}")
        price = cast(Decimal, _decimal(mapping.get("price"), "price"))
        size = cast(Decimal, _decimal(mapping.get("size"), "size"))
        if price != order.price:
            raise CoinbaseL3Error(f"match price {price} does not match maker price {order.price}")
        if size > order.remaining:
            raise CoinbaseL3Error(
                f"match size {size} exceeds maker remaining size {order.remaining}"
            )
        if not compact and mapping.get("side") is not None:
            self._validate_side(mapping.get("side"), order)
        timestamp_ns = _timestamp_ns(mapping.get("time"))
        trade_id = _integer(mapping.get("trade_id"), "trade_id", required=False)
        self.exchange_trades_observed += 1
        if self.retain_trades:
            self.exchange_trades.append(
                CoinbaseExchangeTrade(
                    sequence=sequence,
                    trade_id=trade_id,
                    product_id=self.product_id,
                    maker_order_id=maker_id,
                    taker_order_id=taker_id,
                    normalized_maker_order_id=order.normalized_order_id,
                    normalized_taker_order_id=coinbase_order_id(self.product_id, taker_id),
                    maker_side=order.side.name.lower(),
                    price=_float(price, "price"),
                    quantity=_float(size, "size"),
                    timestamp_ns=timestamp_ns,
                )
            )
        order.remaining -= size
        event = MarketEvent(
            op="reduce",
            symbol=self.product_id,
            order_id=order.normalized_order_id,
            quantity=_float(size, "size"),
            timestamp_ns=timestamp_ns,
        )
        if order.remaining == 0:
            self._remove_active(maker_id)
        return event

    def _done(self, mapping: Mapping[str, Any], compact: bool) -> List[MarketEvent]:
        venue_id = _text(mapping.get("order_id"), "order_id")
        order = self._active.get(venue_id)
        if order is None:
            self.messages_ignored += 1
            return []
        if not compact:
            reason = _text(mapping.get("reason"), "reason").lower()
            if reason == "filled":
                raise CoinbaseL3Error(
                    f"filled order {venue_id} is still active; a match message is missing"
                )
            if reason != "canceled":
                raise CoinbaseL3Error(f"unsupported done reason: {reason!r}")
            if mapping.get("remaining_size") is not None:
                remaining = _decimal(
                    mapping.get("remaining_size"), "remaining_size", allow_zero=True
                )
                if remaining != order.remaining:
                    raise CoinbaseL3Error(
                        f"done remaining size {remaining} does not match {order.remaining}"
                    )
            if mapping.get("side") is not None:
                self._validate_side(mapping.get("side"), order)
        timestamp_ns = _timestamp_ns(mapping.get("time"))
        self._remove_active(venue_id)
        return [
            MarketEvent(
                op="cancel",
                symbol=self.product_id,
                order_id=order.normalized_order_id,
                timestamp_ns=timestamp_ns,
            )
        ]

    def _change(self, mapping: Mapping[str, Any], compact: bool) -> List[MarketEvent]:
        venue_id = _text(mapping.get("order_id"), "order_id")
        order = self._active.get(venue_id)
        if order is None:
            self.messages_ignored += 1
            return []
        if mapping.get("side") is not None:
            self._validate_side(mapping.get("side"), order)

        size_field = "size" if compact else "new_size"
        new_size = cast(Decimal, _decimal(mapping.get(size_field), size_field, allow_zero=True))
        if not compact and mapping.get("old_size") is not None:
            old_size = _decimal(mapping.get("old_size"), "old_size", allow_zero=True)
            if old_size != order.remaining:
                raise CoinbaseL3Error(
                    f"change old size {old_size} does not match {order.remaining}"
                )

        raw_price = mapping.get("price") if compact else mapping.get("new_price")
        if raw_price is None:
            raw_price = mapping.get("price")
        new_price = _decimal(raw_price, "price", required=False)
        if new_price is None:
            new_price = order.price
        if not compact and mapping.get("old_price") is not None:
            old_price = _decimal(mapping.get("old_price"), "old_price")
            if old_price != order.price:
                raise CoinbaseL3Error(f"change old price {old_price} does not match {order.price}")

        timestamp_ns = _timestamp_ns(mapping.get("time"))
        if new_size == 0:
            reduction = order.remaining
            self._remove_active(venue_id)
            return [
                MarketEvent(
                    op="reduce",
                    symbol=self.product_id,
                    order_id=order.normalized_order_id,
                    quantity=_float(reduction, "reduction"),
                    timestamp_ns=timestamp_ns,
                )
            ]
        if new_price == order.price and new_size < order.remaining:
            reduction = order.remaining - new_size
            order.remaining = new_size
            return [
                MarketEvent(
                    op="reduce",
                    symbol=self.product_id,
                    order_id=order.normalized_order_id,
                    quantity=_float(reduction, "reduction"),
                    timestamp_ns=timestamp_ns,
                )
            ]
        if new_price == order.price and new_size == order.remaining:
            self.messages_ignored += 1
            return []

        order.price = new_price
        order.remaining = new_size
        order.timestamp_ns = timestamp_ns
        return [
            MarketEvent(
                op="replace",
                symbol=self.product_id,
                order_id=order.normalized_order_id,
                price=_float(new_price, "price"),
                quantity=_float(new_size, size_field),
                timestamp_ns=timestamp_ns,
            )
        ]

    def _add_active(
        self,
        venue_id: str,
        side: OrderSide,
        price: Decimal,
        remaining: Decimal,
        timestamp_ns: Optional[int],
    ) -> _ActiveOrder:
        if venue_id in self._active:
            raise CoinbaseL3Error(f"duplicate active order id: {venue_id}")
        normalized = coinbase_order_id(self.product_id, venue_id)
        collision = self._normalized_active.get(normalized)
        if collision is not None and collision != venue_id:
            raise CoinbaseL3Error(
                f"normalized order-id collision between {collision!r} and {venue_id!r}"
            )
        if self.retain_id_map:
            prior = self._id_map.get(normalized)
            if prior is not None and prior != venue_id:
                raise CoinbaseL3Error(
                    f"normalized order-id collision between {prior!r} and {venue_id!r}"
                )
            self._id_map[normalized] = venue_id
        order = _ActiveOrder(venue_id, normalized, side, price, remaining, timestamp_ns)
        self._active[venue_id] = order
        self._normalized_active[normalized] = venue_id
        return order

    def _remove_active(self, venue_id: str) -> _ActiveOrder:
        order = self._active.pop(venue_id)
        self._normalized_active.pop(order.normalized_order_id, None)
        return order

    @staticmethod
    def _validate_side(value: Any, order: _ActiveOrder) -> None:
        if _side(value) != order.side:
            raise CoinbaseL3Error(
                f"message side does not match resting order {order.venue_order_id}"
            )


def load_coinbase_l3_snapshot(path: str | Path) -> Mapping[str, Any]:
    """Load one Coinbase REST level-3 snapshot JSON object."""
    source = Path(path)
    if not source.is_file():
        raise CoinbaseL3Error(f"Snapshot file not found: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoinbaseL3Error(f"Invalid snapshot JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CoinbaseL3Error("snapshot JSON must contain an object")
    return payload


def iter_coinbase_l3_messages(path: str | Path) -> Iterator[RawCoinbaseMessage]:
    """Stream Coinbase feed messages from JSON, JSONL, or NDJSON."""
    source = Path(path)
    if not source.is_file():
        raise CoinbaseL3Error(f"Feed file not found: {source}")
    suffix = source.suffix.lower()
    if suffix in {".jsonl", ".ndjson"}:
        try:
            with source.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, 1):
                    if not line.strip():
                        continue
                    try:
                        payload = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise CoinbaseL3Error(
                            f"Invalid feed JSON on line {line_number}: {exc}"
                        ) from exc
                    if not isinstance(payload, (dict, list)):
                        raise CoinbaseL3Error(
                            f"Feed line {line_number} must contain an object or array"
                        )
                    yield payload
        except OSError as exc:
            raise CoinbaseL3Error(f"Unable to read feed file: {exc}") from exc
        return
    if suffix != ".json":
        raise CoinbaseL3Error("Feed file must end in .json, .jsonl, or .ndjson")
    try:
        payload = json.loads(source.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoinbaseL3Error(f"Invalid feed JSON: {exc}") from exc
    if isinstance(payload, dict):
        payload = payload.get("messages", [payload])
    elif isinstance(payload, list):
        if not payload:
            return
        if not isinstance(payload[0], (dict, list)):
            payload = [payload]
    else:
        raise CoinbaseL3Error("Feed JSON must be a message or a messages array")
    if not isinstance(payload, list):
        raise CoinbaseL3Error("Feed JSON messages must be an array")
    for index, message in enumerate(payload, 1):
        if not isinstance(message, (dict, list)):
            raise CoinbaseL3Error(f"Feed message {index} must be an object or array")
        yield message


def normalize_coinbase_l3(
    snapshot: Mapping[str, Any],
    messages: Iterable[RawCoinbaseMessage],
    product_id: Optional[str] = None,
    *,
    strict: bool = True,
    retain_id_map: bool = False,
    retain_trades: bool = False,
) -> Tuple[CoinbaseL3Adapter, List[MarketEvent]]:
    """Convenience API that collects a Coinbase event stream in memory."""
    adapter = CoinbaseL3Adapter(
        snapshot,
        product_id,
        strict=strict,
        retain_id_map=retain_id_map,
        retain_trades=retain_trades,
    )
    return adapter, list(adapter.iter_events(messages))
