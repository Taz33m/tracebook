"""Load normalized order events and replay them through real order books.

The importer intentionally targets an explicit order-level schema rather than
pretending an aggregated L2 snapshot can recover queue priority. Adapters for
exchange-specific feeds can normalize into ``MarketEvent`` and reuse the same
validated replay path.
"""

from __future__ import annotations

import csv
import json
import math
import re
from dataclasses import asdict, dataclass, field
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, cast

from ..core.order import NO_OWNER, OrderSide, OrderType, SelfTradePolicy, Trade, normalize_symbol
from ..core.orderbook import OrderBook, OrderBookManager

_INTEGER_PATTERN = re.compile(r"^[+-]?\d+$")
_OPS = {"new", "cancel", "replace", "clear"}


class MarketReplayError(ValueError):
    """Raised when a normalized market event cannot be parsed or applied."""


def _optional(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str) and not value.strip():
        return None
    return value


def _integer(value: Any, field_name: str, required: bool = True) -> Optional[int]:
    value = _optional(value)
    if value is None:
        if required:
            raise MarketReplayError(f"{field_name} is required")
        return None
    if isinstance(value, bool):
        raise MarketReplayError(f"{field_name} must be an integer")
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, str) and _INTEGER_PATTERN.fullmatch(value.strip()):
        return int(value)
    raise MarketReplayError(f"{field_name} must be an integer: {value!r}")


def _number(value: Any, field_name: str, required: bool = True) -> Optional[float]:
    value = _optional(value)
    if value is None:
        if required:
            raise MarketReplayError(f"{field_name} is required")
        return None
    if isinstance(value, bool):
        raise MarketReplayError(f"{field_name} must be numeric")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MarketReplayError(f"{field_name} must be numeric: {value!r}") from exc
    if not math.isfinite(parsed):
        raise MarketReplayError(f"{field_name} must be finite: {value!r}")
    return parsed


def _side(value: Any) -> OrderSide:
    value = _optional(value)
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in OrderSide.__members__:
            return OrderSide[normalized]
        if _INTEGER_PATTERN.fullmatch(normalized):
            value = int(normalized)
    if isinstance(value, bool):
        raise MarketReplayError("side must be BUY, SELL, 1, or -1")
    try:
        return OrderSide(value)
    except (TypeError, ValueError) as exc:
        raise MarketReplayError(f"side must be BUY, SELL, 1, or -1: {value!r}") from exc


def _order_type(value: Any) -> OrderType:
    value = "LIMIT" if _optional(value) is None else value
    if isinstance(value, str):
        normalized = value.strip().upper()
        if normalized in OrderType.__members__:
            return OrderType[normalized]
        if _INTEGER_PATTERN.fullmatch(normalized):
            value = int(normalized)
    if isinstance(value, bool):
        raise MarketReplayError("order_type must be LIMIT, MARKET, IOC, or FOK")
    try:
        return OrderType(value)
    except (TypeError, ValueError) as exc:
        raise MarketReplayError(
            f"order_type must be LIMIT, MARKET, IOC, or FOK: {value!r}"
        ) from exc


@dataclass(frozen=True)
class MarketEvent:
    """One normalized order-level event, processed in source-file order."""

    op: str
    symbol: str
    order_id: Optional[int] = None
    side: Optional[OrderSide] = None
    order_type: OrderType = OrderType.LIMIT
    price: Optional[float] = None
    quantity: Optional[float] = None
    owner: int = NO_OWNER
    timestamp_ns: Optional[int] = None

    def __post_init__(self) -> None:
        """Normalize direct construction to the same contract as file parsing."""
        raw_op = str(self.op).strip().lower()
        op = "new" if raw_op == "submit" else raw_op
        if op not in _OPS:
            raise MarketReplayError(f"op must be one of {sorted(_OPS)}: {raw_op!r}")

        if not isinstance(self.symbol, str):
            raise MarketReplayError(f"symbol must be a non-empty string: {self.symbol!r}")
        try:
            symbol = normalize_symbol(self.symbol)
        except ValueError as exc:
            raise MarketReplayError(str(exc)) from exc

        timestamp_ns = _integer(self.timestamp_ns, "timestamp_ns", required=False)
        if timestamp_ns is not None and timestamp_ns < 0:
            raise MarketReplayError("timestamp_ns must be non-negative")

        object.__setattr__(self, "op", op)
        object.__setattr__(self, "symbol", symbol)
        object.__setattr__(self, "timestamp_ns", timestamp_ns)
        object.__setattr__(self, "order_type", _order_type(self.order_type))

        owner = _integer(self.owner, "owner")
        object.__setattr__(self, "owner", NO_OWNER if owner is None else owner)

        order_id = _integer(self.order_id, "order_id", required=False)
        side = None if _optional(self.side) is None else _side(self.side)
        price = _number(self.price, "price", required=False)
        quantity = _number(self.quantity, "quantity", required=False)
        object.__setattr__(self, "order_id", order_id)
        object.__setattr__(self, "side", side)
        object.__setattr__(self, "price", price)
        object.__setattr__(self, "quantity", quantity)

        if op == "clear":
            return

        if order_id is None or order_id <= 0:
            raise MarketReplayError("order_id must be positive")

        if op == "cancel":
            return

        if op == "replace":
            if price is None and quantity is None:
                raise MarketReplayError("replace requires price and/or quantity")
            return

        if side is None:
            raise MarketReplayError("side is required")
        order_type = _order_type(self.order_type)
        if quantity is None:
            raise MarketReplayError("quantity is required")
        if order_type == OrderType.MARKET:
            price = 0.0 if price is None else price
        elif price is None:
            raise MarketReplayError("price is required")

        object.__setattr__(self, "side", side)
        object.__setattr__(self, "order_type", order_type)
        object.__setattr__(self, "price", price)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "MarketEvent":
        """Parse and validate one JSON/CSV-style mapping."""
        return cls(
            op=data.get("op", ""),
            symbol=cast(str, data.get("symbol")),
            order_id=data.get("order_id"),
            side=data.get("side"),
            order_type=data.get("order_type", OrderType.LIMIT),
            price=data.get("price"),
            quantity=data.get("quantity"),
            owner=data.get("owner", NO_OWNER),
            timestamp_ns=data.get("timestamp_ns"),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["side"] = self.side.name if self.side is not None else None
        data["order_type"] = self.order_type.name
        return data


@dataclass(frozen=True)
class ReplayRejection:
    event_index: int
    op: str
    symbol: str
    reason: str


@dataclass(frozen=True)
class ReplayTrade:
    """A trade annotated with source and engine order identifiers."""

    event_index: int
    symbol: str
    buy_order_id: int
    sell_order_id: int
    engine_buy_order_id: int
    engine_sell_order_id: int
    price: float
    quantity: float
    timestamp_ns: int


@dataclass
class MarketReplayResult:
    """Outcome and reconstructed books from an event replay."""

    manager: OrderBookManager
    matching_algorithm: str = "fifo"
    tick_size: float = 0.01
    self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE
    input_events: int = 0
    applied_events: int = 0
    submissions: int = 0
    cancellations: int = 0
    replacements: int = 0
    clears: int = 0
    trades: List[ReplayTrade] = field(default_factory=list)
    rejections: List[ReplayRejection] = field(default_factory=list)
    active_order_ids: Dict[str, Dict[int, int]] = field(default_factory=dict)
    _engine_to_source_ids: Dict[str, Dict[int, int]] = field(default_factory=dict, repr=False)

    def resolve_order_id(self, symbol: str, source_order_id: int) -> Optional[int]:
        """Return the active engine id for a source order id, if one exists."""
        symbol = normalize_symbol(symbol)
        normalized_id = _integer(source_order_id, "source_order_id")
        if normalized_id is None or normalized_id <= 0:
            raise MarketReplayError("source_order_id must be positive")
        return self.active_order_ids.get(symbol, {}).get(normalized_id)

    def to_dict(self, depth_levels: int = 5, include_trades: bool = False) -> dict:
        """Return a JSON-serializable replay summary."""
        if (
            isinstance(depth_levels, bool)
            or not isinstance(depth_levels, Integral)
            or depth_levels < 0
        ):
            raise MarketReplayError("depth_levels must be a non-negative integer")
        if not isinstance(include_trades, bool):
            raise MarketReplayError("include_trades must be a boolean")
        depth_levels = int(depth_levels)

        books = {}
        for symbol, book in self.manager.get_all_order_books().items():
            state = book.get_state_snapshot(levels=depth_levels, trade_count=0)
            books[symbol] = {
                "bids": [list(level) for level in state["bids"]],
                "asks": [list(level) for level in state["asks"]],
                "statistics": state["statistics"],
            }
        return {
            "schema_version": 1,
            "replay_config": {
                "matching_algorithm": self.matching_algorithm,
                "tick_size": self.tick_size,
                "self_trade_policy": self.self_trade_policy.name,
            },
            "input_events": self.input_events,
            "applied_events": self.applied_events,
            "rejected_events": len(self.rejections),
            "submissions": self.submissions,
            "cancellations": self.cancellations,
            "replacements": self.replacements,
            "clears": self.clears,
            "trades_executed": len(self.trades),
            "trades_included": include_trades,
            "trades": [asdict(trade) for trade in self.trades] if include_trades else [],
            "rejections": [asdict(rejection) for rejection in self.rejections],
            "active_orders": {
                symbol: [
                    {
                        "source_order_id": source_id,
                        "engine_order_id": engine_id,
                    }
                    for source_id, engine_id in sorted(order_ids.items())
                ]
                for symbol, order_ids in sorted(self.active_order_ids.items())
            },
            "books": books,
        }


def load_market_events(path: str | Path) -> List[MarketEvent]:
    """Load normalized events from JSON, JSONL/NDJSON, or CSV."""
    source = Path(path)
    if not source.is_file():
        raise MarketReplayError(f"Event file not found: {source}")

    suffix = source.suffix.lower()
    mappings: List[Mapping[str, Any]] = []
    if suffix in {".jsonl", ".ndjson"}:
        for line_number, line in enumerate(source.read_text(encoding="utf-8-sig").splitlines(), 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise MarketReplayError(f"Invalid JSON on line {line_number}: {exc}") from exc
            if not isinstance(item, dict):
                raise MarketReplayError(f"Line {line_number} must contain a JSON object")
            mappings.append(item)
    elif suffix == ".json":
        try:
            payload = json.loads(source.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as exc:
            raise MarketReplayError(f"Invalid JSON: {exc}") from exc
        if isinstance(payload, dict):
            payload = payload.get("events")
        if not isinstance(payload, list) or not all(isinstance(item, dict) for item in payload):
            raise MarketReplayError("JSON input must be an event array or an object with events[]")
        mappings = payload
    elif suffix == ".csv":
        with source.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise MarketReplayError("CSV input requires a header row")
            mappings = list(reader)
    else:
        raise MarketReplayError("Event file must end in .json, .jsonl, .ndjson, or .csv")

    events = []
    for index, mapping in enumerate(mappings, 1):
        try:
            events.append(MarketEvent.from_mapping(mapping))
        except (TypeError, ValueError) as exc:
            raise MarketReplayError(f"Invalid event {index}: {exc}") from exc
    return events


def replay_market_events(
    events: Iterable[MarketEvent],
    matching_algorithm: str = "fifo",
    tick_size: float = 0.01,
    strict: bool = True,
    self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
) -> MarketReplayResult:
    """Replay normalized events through independent books per symbol."""
    if not isinstance(matching_algorithm, str):
        raise MarketReplayError("matching_algorithm must be 'fifo' or 'pro_rata'")
    matching_algorithm = matching_algorithm.strip().lower()
    if matching_algorithm not in {"fifo", "pro_rata"}:
        raise MarketReplayError("matching_algorithm must be 'fifo' or 'pro_rata'")
    if isinstance(tick_size, bool) or not isinstance(tick_size, Real):
        raise MarketReplayError("tick_size must be a positive finite number")
    tick_size = float(tick_size)
    if not math.isfinite(tick_size) or tick_size <= 0:
        raise MarketReplayError("tick_size must be a positive finite number")
    if not isinstance(strict, bool):
        raise MarketReplayError("strict must be a boolean")
    if isinstance(self_trade_policy, bool):
        raise MarketReplayError("self_trade_policy must be a SelfTradePolicy value")
    try:
        self_trade_policy = SelfTradePolicy(self_trade_policy)
    except (TypeError, ValueError) as exc:
        raise MarketReplayError("self_trade_policy must be a SelfTradePolicy value") from exc

    manager = OrderBookManager()
    result = MarketReplayResult(
        manager=manager,
        matching_algorithm=matching_algorithm,
        tick_size=tick_size,
        self_trade_policy=self_trade_policy,
    )

    for index, event in enumerate(events, 1):
        result.input_events += 1

        if not isinstance(event, MarketEvent):
            reason = f"Expected MarketEvent, got {type(event).__name__}"
            rejection = ReplayRejection(index, "unknown", "", reason)
            result.rejections.append(rejection)
            if strict:
                raise MarketReplayError(f"Event {index} rejected: {reason}")
            continue

        try:
            book = manager.get_order_book(event.symbol)
            if book is None:
                if event.op in {"cancel", "replace"}:
                    raise MarketReplayError(f"Order {event.order_id} is not active")
                book = manager.create_order_book(
                    event.symbol,
                    matching_algorithm=matching_algorithm,
                    tick_size=tick_size,
                    self_trade_policy=self_trade_policy,
                )
            _apply_event(book, event, index, result)
            result.applied_events += 1
        except (TypeError, ValueError) as exc:
            rejection = ReplayRejection(index, event.op, event.symbol, str(exc))
            result.rejections.append(rejection)
            if strict:
                raise MarketReplayError(
                    f"Event {index} ({event.op} {event.symbol}) rejected: {exc}"
                ) from exc

    return result


def _apply_event(
    book: OrderBook, event: MarketEvent, event_index: int, result: MarketReplayResult
) -> None:
    active_ids = result.active_order_ids.setdefault(event.symbol, {})
    engine_to_source = result._engine_to_source_ids.setdefault(event.symbol, {})

    if event.op == "clear":
        book.clear()
        active_ids.clear()
        engine_to_source.clear()
        result.clears += 1
        return

    if event.order_id is None:
        raise MarketReplayError("order_id is required")

    if event.op == "cancel":
        engine_order_id = active_ids.get(event.order_id)
        if engine_order_id is None or not book.cancel_order(engine_order_id):
            raise MarketReplayError(f"Order {event.order_id} is not active")
        del active_ids[event.order_id]
        engine_to_source.pop(engine_order_id, None)
        result.cancellations += 1
        return

    if event.op == "replace":
        engine_order_id = active_ids.get(event.order_id)
        if engine_order_id is None:
            raise MarketReplayError(f"Order {event.order_id} is not active")
        replaced = book.replace_order(
            engine_order_id,
            price=event.price,
            quantity=event.quantity,
            timestamp=(event.timestamp_ns if event.timestamp_ns is not None else event_index),
        )
        if not replaced.accepted:
            raise MarketReplayError(replaced.rejected_reason or "replacement was rejected")
        if replaced.order is None:
            raise MarketReplayError("replacement returned no order")
        replacement_id = replaced.order.order_id
        engine_to_source.pop(engine_order_id, None)
        engine_to_source[replacement_id] = event.order_id
        _record_trades(result, event, event_index, replaced.trades)
        if replaced.rested:
            active_ids[event.order_id] = replacement_id
        else:
            active_ids.pop(event.order_id, None)
        _prune_inactive_orders(
            book,
            active_ids,
            engine_to_source,
            replaced.trades,
            result.self_trade_policy,
        )
        if not replaced.rested:
            engine_to_source.pop(replacement_id, None)
        result.replacements += 1
        return

    if event.side is None or event.price is None or event.quantity is None:
        raise MarketReplayError("new events require side, price, and quantity")
    if event.order_id in active_ids:
        raise MarketReplayError(f"Order {event.order_id} is already active")

    timestamp = event.timestamp_ns if event.timestamp_ns is not None else event_index
    order = book.order_factory.create_order(
        event.symbol,
        event.side,
        event.order_type,
        event.price,
        event.quantity,
        event.owner,
    )
    order.timestamp = timestamp
    order.priority = timestamp
    engine_to_source[order.order_id] = event.order_id
    submitted = book.submit_order(order)
    if not submitted.accepted:
        engine_to_source.pop(order.order_id, None)
        raise MarketReplayError(submitted.rejected_reason or "submission was rejected")
    if submitted.order is None:
        engine_to_source.pop(order.order_id, None)
        raise MarketReplayError("submission returned no order")
    if submitted.rested:
        active_ids[event.order_id] = submitted.order.order_id
    _record_trades(result, event, event_index, submitted.trades)
    _prune_inactive_orders(
        book,
        active_ids,
        engine_to_source,
        submitted.trades,
        result.self_trade_policy,
    )
    if not submitted.rested:
        engine_to_source.pop(submitted.order.order_id, None)
    result.submissions += 1


def _record_trades(
    result: MarketReplayResult,
    event: MarketEvent,
    event_index: int,
    trades: List[Trade],
) -> None:
    """Attach source ids and event context to engine trade records."""
    id_map = result._engine_to_source_ids[event.symbol]
    for trade in trades:
        result.trades.append(
            ReplayTrade(
                event_index=event_index,
                symbol=event.symbol,
                buy_order_id=id_map.get(trade.buy_order_id, trade.buy_order_id),
                sell_order_id=id_map.get(trade.sell_order_id, trade.sell_order_id),
                engine_buy_order_id=trade.buy_order_id,
                engine_sell_order_id=trade.sell_order_id,
                price=trade.price,
                quantity=trade.quantity,
                timestamp_ns=trade.timestamp,
            )
        )


def _prune_inactive_orders(
    book: OrderBook,
    active_ids: Dict[int, int],
    engine_to_source: Dict[int, int],
    trades: List[Trade],
    self_trade_policy: SelfTradePolicy,
) -> None:
    """Remove source ids spent by fills or self-trade prevention."""
    if self_trade_policy == SelfTradePolicy.CANCEL_RESTING:
        # CANCEL_RESTING can remove orders without emitting a trade, so this
        # policy requires a full active-map reconciliation.
        candidates = list(active_ids.items())
    else:
        engine_ids = {
            engine_id for trade in trades for engine_id in (trade.buy_order_id, trade.sell_order_id)
        }
        candidates = [
            (engine_to_source[engine_id], engine_id)
            for engine_id in engine_ids
            if engine_id in engine_to_source
        ]

    for source_id, engine_id in candidates:
        if active_ids.get(source_id) == engine_id and book.get_order(engine_id) is None:
            del active_ids[source_id]
            engine_to_source.pop(engine_id, None)


def replay_market_event_file(
    path: str | Path,
    matching_algorithm: str = "fifo",
    tick_size: float = 0.01,
    strict: bool = True,
    self_trade_policy: SelfTradePolicy = SelfTradePolicy.NONE,
) -> MarketReplayResult:
    """Load and replay one normalized event file."""
    return replay_market_events(
        load_market_events(path),
        matching_algorithm=matching_algorithm,
        tick_size=tick_size,
        strict=strict,
        self_trade_policy=self_trade_policy,
    )
