"""Stable failure classes for human-readable conformance artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

from ..core.order import OrderSide
from ..events import MarketEvent

if TYPE_CHECKING:
    from .compare import ConformanceReport

QUEUE_PRIORITY_DRIFT = "queue-priority drift"


@dataclass(frozen=True)
class FailureSignature:
    """Failure identity that a reduction must preserve."""

    operational: bool
    category: str
    failure_class: str


def is_operational_divergence(
    category: object,
    *,
    snapshot_failed: bool = False,
    close_failed: bool = False,
) -> bool:
    """Return whether a divergence represents an adapter/protocol failure."""
    return category == "protocol" or snapshot_failed or close_failed


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
    if first.price is None or second.price != first.price:
        return False
    if (
        first.quantity is None
        or second.quantity is None
        or first_taker.quantity is None
        or second_taker.quantity is None
    ):
        return False
    if first_taker.price is None or second_taker.price is None:
        return False
    first_crosses = (first_taker.side == OrderSide.BUY and first_taker.price >= first.price) or (
        first_taker.side == OrderSide.SELL and first_taker.price <= first.price
    )
    second_crosses = (second_taker.side == OrderSide.BUY and second_taker.price >= first.price) or (
        second_taker.side == OrderSide.SELL and second_taker.price <= first.price
    )
    return (
        first_crosses
        and second_crosses
        and 0 < first_taker.quantity < first.quantity
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


def failure_signature(
    events: Sequence[MarketEvent], report: ConformanceReport
) -> FailureSignature | None:
    """Return the stable failure identity for one conformance result."""
    divergence = report.divergence
    if divergence is None:
        return None
    return FailureSignature(
        operational=report.operational_failure,
        category=divergence.category,
        failure_class=classify_failure(events, report),
    )


def preserves_failure_signature(
    target: FailureSignature, observed: FailureSignature | None
) -> bool:
    """Return whether a reduction preserves or narrows the original failure."""
    if observed is None:
        return False
    if target.operational != observed.operational or target.category != observed.category:
        return False
    return target.failure_class == observed.failure_class or (
        target.failure_class == "execution drift" and observed.failure_class == QUEUE_PRIORITY_DRIFT
    )
