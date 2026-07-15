"""Stable failure classes for human-readable conformance artifacts."""

from __future__ import annotations

from typing import Sequence

from ..core.order import OrderSide
from ..events import MarketEvent
from .compare import ConformanceReport

QUEUE_PRIORITY_DRIFT = "queue-priority drift"


def is_queue_priority_probe(events: Sequence[MarketEvent], end_index: int | None = None) -> bool:
    """Return whether a five-event FIFO priority probe ends at ``end_index``."""
    end = len(events) if end_index is None else end_index
    if end < 5 or end > len(events):
        return False
    first, second, reduction, replacement, taker = events[end - 5 : end]
    if not all(event.symbol == first.symbol for event in events[end - 5 : end]):
        return False
    if first.op != "new" or second.op != "new" or taker.op != "new":
        return False
    if reduction.op != "reduce" or replacement.op != "replace":
        return False
    if first.order_id is None or second.order_id is None or taker.order_id is None:
        return False
    if len({first.order_id, second.order_id, taker.order_id}) != 3:
        return False
    if reduction.order_id != first.order_id or replacement.order_id != first.order_id:
        return False
    if first.side is None or second.side != first.side or taker.side is None:
        return False
    if taker.side == first.side:
        return False
    if first.price is None or second.price != first.price or replacement.price != first.price:
        return False
    if taker.price is None:
        return False
    crosses = (taker.side == OrderSide.BUY and taker.price >= first.price) or (
        taker.side == OrderSide.SELL and taker.price <= first.price
    )
    return crosses


def is_partial_fill_priority_probe(
    events: Sequence[MarketEvent], end_index: int | None = None
) -> bool:
    """Return whether a four-event partial-fill continuation probe ends here."""
    end = len(events) if end_index is None else end_index
    if end < 4 or end > len(events):
        return False
    first, second, first_taker, second_taker = events[end - 4 : end]
    window = (first, second, first_taker, second_taker)
    if not all(event.op == "new" and event.symbol == first.symbol for event in window):
        return False
    if any(event.order_id is None for event in window):
        return False
    if len({event.order_id for event in window}) != 4:
        return False
    if first.side is None or second.side != first.side:
        return False
    if first_taker.side is None or second_taker.side != first_taker.side:
        return False
    if first_taker.side == first.side:
        return False
    if first.price is None or any(event.price != first.price for event in window[1:]):
        return False
    if (
        first.quantity is None
        or second.quantity is None
        or first_taker.quantity is None
        or second_taker.quantity is None
    ):
        return False
    return (
        0 < first_taker.quantity < first.quantity
        and second.quantity > 0
        and second_taker.quantity > 0
    )


def classify_failure(events: Sequence[MarketEvent], report: ConformanceReport) -> str:
    """Map a first divergence to a stable, user-facing failure class."""
    divergence = report.divergence
    if divergence is None:
        return "conformant"
    is_priority_probe = is_queue_priority_probe(
        events, divergence.event_index
    ) or is_partial_fill_priority_probe(events, divergence.event_index)
    if divergence.category == "trades" and is_priority_probe:
        return QUEUE_PRIORITY_DRIFT
    return {
        "outcome": "order-outcome drift",
        "trades": "execution drift",
        "book_state": "book-state drift",
        "protocol": "adapter protocol failure",
    }.get(divergence.category, "semantic drift")
