"""Deterministic record and replay for order books.

A recorder captures every mutating operation (submissions, reductions, cancels, and clears)
as a serializable event log. Replaying that log against a fresh book reconstructs
the identical sequence of trades -- by ``(buy_order_id, sell_order_id, price,
quantity)`` -- and the identical final book state.

Determinism note: matching does not depend on wall-clock time (execution price
is the resting price and FIFO priority is insertion order), so a fixed event log
always reproduces the same trades and resting book. Wall-clock ``timestamp``
fields on individual trades and orders are metadata and are excluded from the
guarantee; replay reuses the recorded order timestamps for order objects but
trade timestamps are stamped fresh.
"""

import json
from dataclasses import asdict, dataclass, field
from typing import List, Optional

from .order import NO_OWNER, Order, SelfTradePolicy


@dataclass
class RecordedEvent:
    """One recorded mutating operation against a book."""

    op: str  # "submit" | "reduce" | "cancel" | "clear"
    order_id: int
    side: Optional[int] = None
    order_type: Optional[int] = None
    price: Optional[float] = None
    quantity: Optional[float] = None
    remaining_quantity: Optional[float] = None
    symbol: Optional[str] = None
    timestamp: Optional[int] = None
    owner: int = NO_OWNER

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "RecordedEvent":
        return cls(**data)


@dataclass
class EventLog:
    """An ordered, serializable log of operations for one book."""

    symbol: str
    matching_algorithm: str = "fifo"
    tick_size: float = 0.01
    self_trade_policy: int = int(SelfTradePolicy.NONE)
    events: List[RecordedEvent] = field(default_factory=list)
    schema_version: int = 2

    def record_submit(self, order: Order) -> None:
        """Capture an accepted order as submitted (pre-matching values)."""
        self.events.append(
            RecordedEvent(
                op="submit",
                order_id=int(order.order_id),
                side=int(order.side),
                order_type=int(order.order_type),
                price=float(order.price),
                quantity=float(order.quantity),
                remaining_quantity=float(order.remaining_quantity),
                symbol=order.symbol,
                timestamp=int(order.timestamp),
                owner=int(order.owner),
            )
        )

    def record_cancel(self, order_id: int) -> None:
        """Capture a successful cancellation."""
        self.events.append(RecordedEvent(op="cancel", order_id=int(order_id)))

    def record_reduce(self, order_id: int, quantity: float) -> None:
        """Capture a priority-preserving reduction in remaining quantity."""
        self.events.append(
            RecordedEvent(op="reduce", order_id=int(order_id), quantity=float(quantity))
        )

    def record_clear(self) -> None:
        """Capture a full book reset."""
        self.events.append(RecordedEvent(op="clear", order_id=0))

    def to_dict(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "symbol": self.symbol,
            "matching_algorithm": self.matching_algorithm,
            "tick_size": self.tick_size,
            "self_trade_policy": int(self.self_trade_policy),
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventLog":
        schema_version = int(data.get("schema_version", 1))
        if schema_version not in {1, 2}:
            raise ValueError(f"Unsupported event log schema_version: {schema_version}")
        log = cls(
            symbol=data["symbol"],
            matching_algorithm=data.get("matching_algorithm", "fifo"),
            tick_size=data.get("tick_size", 0.01),
            self_trade_policy=int(data.get("self_trade_policy", int(SelfTradePolicy.NONE))),
            schema_version=schema_version,
        )
        log.events = [RecordedEvent.from_dict(event) for event in data.get("events", [])]
        if schema_version == 1 and any(event.op == "reduce" for event in log.events):
            raise ValueError("Event log schema_version 1 does not support reduce operations")
        return log

    def to_json(self, **kwargs) -> str:
        kwargs.setdefault("allow_nan", False)
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> "EventLog":
        return cls.from_dict(json.loads(text))

    def __len__(self) -> int:
        return len(self.events)


def replay(event_log: EventLog):
    """Replay an event log against a fresh book and return the reconstructed book.

    Every recorded submission is re-applied exactly as captured. A soft outcome
    such as an unfillable FOK is reproduced faithfully (it is a normal result,
    not a divergence), so the reconstructed trades and book state match the
    original run. A hard rejection (an order that fails validation) or a cancel
    that finds no order signals that the log is malformed or incompatible; both
    raise ValueError so replay fails fast instead of diverging silently.
    """
    from .orderbook import OrderBook  # local import to avoid an import cycle

    book = OrderBook(
        event_log.symbol,
        matching_algorithm=event_log.matching_algorithm,
        tick_size=event_log.tick_size,
        self_trade_policy=SelfTradePolicy(event_log.self_trade_policy),
    )

    for event in event_log.events:
        if event.op == "submit":
            order = Order(
                order_id=event.order_id,
                symbol=event.symbol,
                side=event.side,
                order_type=event.order_type,
                price=event.price,
                quantity=event.quantity,
                timestamp=event.timestamp or 0,
                owner=event.owner if event.owner is not None else NO_OWNER,
            )
            if event.remaining_quantity is not None:
                order.remaining_quantity = event.remaining_quantity
            result = book.submit_order(order)
            # A soft rejection (e.g. unfillable FOK) keeps accepted=True and is
            # a faithful outcome; only a hard validation rejection diverges.
            if not result.accepted:
                raise ValueError(
                    f"Replay diverged: submission of order {event.order_id} "
                    f"was rejected: {result.rejected_reason}"
                )
        elif event.op == "cancel":
            # Only successful cancels are ever recorded, so a faithful replay
            # always finds the order; a failure means the log has diverged.
            if not book.cancel_order(event.order_id):
                raise ValueError(
                    f"Replay diverged: cancel of order {event.order_id} "
                    "found no matching resting order"
                )
        elif event.op == "reduce":
            if event.quantity is None:
                raise ValueError(
                    f"Replay diverged: reduction of order {event.order_id} has no quantity"
                )
            if not book.reduce_order(event.order_id, event.quantity):
                raise ValueError(
                    f"Replay diverged: reduction of order {event.order_id} "
                    "found no matching resting order"
                )
        elif event.op == "clear":
            book.clear()
        else:
            raise ValueError(f"Unknown recorded op: {event.op!r}")

    return book
