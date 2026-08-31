"""Inspectable reference semantics for L3 book replay."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Dict, List, Tuple, cast

from .._version import __version__
from .model import (
    BookReplayConfig,
    BookReplayError,
    BookReplayEvent,
    BookReplayObservation,
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    Outcome,
    RestingBookOrder,
    SimulatedFill,
    canonical_decimal,
)


@dataclass
class _Order:
    order_id: int
    side: str
    price: Decimal
    quantity: Decimal


class _Book:
    def __init__(self) -> None:
        self.levels: Dict[str, Dict[Decimal, List[_Order]]] = {"BUY": {}, "SELL": {}}
        self.orders: Dict[int, _Order] = {}


class ReferenceBookReplayAdapter:
    """Mirror L3 deltas and simulate fills without crossing submitted orders."""

    def __init__(
        self,
        config: BookReplayConfig,
        engine_name: str = "tracebook-book-replay-reference",
        engine_version: str = __version__,
    ) -> None:
        if not isinstance(config, BookReplayConfig):
            raise BookReplayError("reference adapter requires BookReplayConfig")
        self.config = config
        self.metadata = EngineMetadata(engine_name, engine_version, "Python")
        self._books: Dict[str, _Book] = {}

    def apply(self, event: BookReplayEvent, index: int) -> BookReplayObservation:
        """Apply one delta or probe and emit its canonical observation."""
        if not isinstance(event, BookReplayEvent):
            raise BookReplayError("reference adapter requires BookReplayEvent values")

        fills: Tuple[SimulatedFill, ...] = ()
        outcome = Outcome("applied")
        if event.op == "add":
            outcome = self._add(event)
        elif event.op == "update":
            outcome = self._update(event)
        elif event.op == "delete":
            outcome = self._delete(event)
        else:
            fills = self._probe(event)

        state = self.snapshot()
        return BookReplayObservation(index, outcome, fills, state.digest(), state.order_count)

    def _add(self, event: BookReplayEvent) -> Outcome:
        book = self._books.setdefault(event.symbol, _Book())
        order_id = cast(int, event.order_id)
        if order_id in book.orders:
            return Outcome(
                "rejected",
                "DUPLICATE_ORDER_ID",
                f"Order {order_id} is already present in {event.symbol}",
            )
        side = cast(str, event.side)
        price = Decimal(cast(str, event.price))
        order = _Order(order_id, side, price, Decimal(cast(str, event.quantity)))
        book.orders[order_id] = order
        book.levels[side].setdefault(price, []).append(order)
        return Outcome("applied")

    def _update(self, event: BookReplayEvent) -> Outcome:
        book = self._books.get(event.symbol)
        order_id = cast(int, event.order_id)
        if book is None or order_id not in book.orders:
            return Outcome(
                "rejected",
                "ORDER_NOT_ACTIVE",
                f"Order {order_id} is not present in {event.symbol}",
            )

        order = book.orders[order_id]
        new_side = cast(str, event.side)
        new_price = Decimal(cast(str, event.price))
        new_quantity = Decimal(cast(str, event.quantity))
        if order.side == new_side and order.price == new_price:
            # A mirrored update does not decide venue priority. Like NautilusTrader's
            # BookLevel update, replacing the value under the same key keeps its slot.
            order.quantity = new_quantity
            return Outcome("applied")

        self._remove_from_level(book, order)
        order.side = new_side
        order.price = new_price
        order.quantity = new_quantity
        book.levels[new_side].setdefault(new_price, []).append(order)
        return Outcome("applied")

    def _delete(self, event: BookReplayEvent) -> Outcome:
        book = self._books.get(event.symbol)
        order_id = cast(int, event.order_id)
        if book is None or order_id not in book.orders:
            return Outcome(
                "rejected",
                "ORDER_NOT_ACTIVE",
                f"Order {order_id} is not present in {event.symbol}",
            )
        order = book.orders.pop(order_id)
        self._remove_from_level(book, order)
        return Outcome("applied")

    @staticmethod
    def _remove_from_level(book: _Book, order: _Order) -> None:
        levels = book.levels[order.side]
        level = levels[order.price]
        level.remove(order)
        if not level:
            del levels[order.price]

    def _probe(self, event: BookReplayEvent) -> Tuple[SimulatedFill, ...]:
        book = self._books.get(event.symbol)
        if book is None:
            return ()

        incoming_side = cast(str, event.side)
        resting_side = "SELL" if incoming_side == "BUY" else "BUY"
        limit_price = Decimal(cast(str, event.price))
        remaining = Decimal(cast(str, event.quantity))
        levels = book.levels[resting_side]
        prices = sorted(levels, reverse=resting_side == "BUY")
        fills = []
        for price in prices:
            if incoming_side == "BUY" and price > limit_price:
                break
            if incoming_side == "SELL" and price < limit_price:
                break
            for order in levels[price]:
                quantity = min(order.quantity, remaining)
                fills.append(
                    SimulatedFill(
                        price=canonical_decimal(price),
                        quantity=canonical_decimal(quantity),
                    )
                )
                remaining -= quantity
                if remaining == 0:
                    return tuple(fills)
        return tuple(fills)

    def snapshot(self) -> BookReplayState:
        """Return every mirrored order in price-time priority order."""
        snapshots = []
        for symbol in sorted(self._books):
            book = self._books[symbol]
            snapshots.append(
                BookReplaySnapshot(
                    symbol=symbol,
                    bids=self._snapshot_side(book, "BUY"),
                    asks=self._snapshot_side(book, "SELL"),
                )
            )
        return BookReplayState(tuple(snapshots))

    @staticmethod
    def _snapshot_side(book: _Book, side: str) -> Tuple[RestingBookOrder, ...]:
        orders = []
        for price in sorted(book.levels[side], reverse=side == "BUY"):
            for order in book.levels[side][price]:
                orders.append(
                    RestingBookOrder(
                        order_id=order.order_id,
                        price=canonical_decimal(order.price),
                        quantity=canonical_decimal(order.quantity),
                    )
                )
        return tuple(orders)

    def close(self) -> None:
        """Reference adapters own no external resources."""


__all__ = ["ReferenceBookReplayAdapter"]
