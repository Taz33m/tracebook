"""Tracebook's inspectable matching engine exposed as a conformance adapter."""

from __future__ import annotations

from typing import Dict, List

from .._version import __version__
from ..core.order import OrderSide, OrderType
from ..core.orderbook import OrderBookManager
from ..events.market_replay import (
    MarketEvent,
    MarketReplayError,
    MarketReplayResult,
    ReplayRejection,
    _apply_event,
)
from .model import (
    BookSnapshot,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
    Outcome,
    RestingOrder,
    TradeFill,
    canonical_decimal,
)


def rejection_code(message: str, op: str = "") -> str:
    """Map reference-engine messages onto stable cross-language reason codes."""
    lowered = message.lower()
    if "is not active" in lowered or "not found" in lowered:
        return "ORDER_NOT_ACTIVE"
    if "already active" in lowered or "duplicate order id" in lowered:
        return "DUPLICATE_ORDER_ID"
    if op == "replace" or "replace" in lowered or "replacement" in lowered:
        return "INVALID_REPLACEMENT"
    if op == "cancel" or "cancel" in lowered:
        return "INVALID_CANCEL"
    return "INVALID_ORDER"


class ReferenceEngineAdapter:
    """Incrementally apply ``MarketEvent`` values to Tracebook's engine."""

    def __init__(
        self,
        config: ConformanceConfig,
        engine_name: str = "tracebook-reference",
        engine_version: str = __version__,
    ) -> None:
        self.config = config
        self.metadata = EngineMetadata(engine_name, engine_version, "Python")
        self._result = MarketReplayResult(
            manager=OrderBookManager(),
            matching_algorithm=config.matching_algorithm,
            tick_size=config.tick_size,
            self_trade_policy=config.self_trade_policy,
        )

    def apply(self, event: MarketEvent, index: int) -> Observation:
        """Apply one event and return its canonical observable outcome."""
        if not isinstance(event, MarketEvent):
            raise ConformanceError("reference adapter requires MarketEvent values")
        self._result.input_events += 1
        trade_start = len(self._result.trades)
        try:
            book = self._result.manager.get_order_book(event.symbol)
            if book is None:
                if event.op in {"cancel", "reduce", "replace"}:
                    raise MarketReplayError(f"Order {event.order_id} is not active")
                book = self._result.manager.create_order_book(
                    event.symbol,
                    matching_algorithm=self.config.matching_algorithm,
                    tick_size=self.config.tick_size,
                    self_trade_policy=self.config.self_trade_policy,
                )
            _apply_event(book, event, index, self._result)
            self._result.applied_events += 1
            outcome = Outcome("applied")
        except (TypeError, ValueError) as exc:
            message = str(exc)
            self._result.rejections.append(ReplayRejection(index, event.op, event.symbol, message))
            outcome = Outcome("rejected", rejection_code(message, event.op), message)

        trades = tuple(
            TradeFill(
                symbol=trade.symbol,
                buy_order_id=trade.buy_order_id,
                sell_order_id=trade.sell_order_id,
                price=canonical_decimal(trade.price),
                quantity=canonical_decimal(trade.quantity, self.config.quantity_decimal_places),
            )
            for trade in self._result.trades[trade_start:]
        )
        state = self.snapshot()
        return Observation(index, outcome, trades, state.digest(), state.order_count)

    def snapshot(self) -> BookState:
        """Return every resting source order in matching-priority order."""
        snapshots: List[BookSnapshot] = []
        books = self._result.manager.get_all_order_books()
        for symbol in sorted(books):
            book = books[symbol]
            engine_to_source = self._result._engine_to_source_ids.get(symbol, {})
            bids = self._resting_side(book.get_resting_orders(OrderSide.BUY), engine_to_source)
            asks = self._resting_side(book.get_resting_orders(OrderSide.SELL), engine_to_source)
            snapshots.append(BookSnapshot(symbol, bids, asks))
        return BookState(tuple(snapshots))

    def _resting_side(self, orders, engine_to_source: Dict[int, int]):
        resting = []
        for order in orders:
            source_id = engine_to_source.get(order.order_id)
            if source_id is None:
                raise RuntimeError(
                    f"reference state has no source id for engine order {order.order_id}"
                )
            resting.append(
                RestingOrder(
                    order_id=source_id,
                    price=canonical_decimal(order.price),
                    remaining_quantity=canonical_decimal(
                        order.remaining_quantity,
                        self.config.quantity_decimal_places,
                    ),
                    owner=order.owner,
                    order_type=OrderType(order.order_type).name,
                )
            )
        return tuple(resting)

    def close(self) -> None:
        """Reference adapters own no external resources."""
