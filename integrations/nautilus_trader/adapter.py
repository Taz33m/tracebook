#!/usr/bin/env python3
"""NautilusTrader v2 L3 book adapter for Tracebook's book-replay protocol."""

from __future__ import annotations

from importlib import metadata as importlib_metadata

from tracebook.book_replay import (
    BookReplayObservation,
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    Outcome,
    RestingBookOrder,
    SimulatedFill,
    serve_book_replay_stdio,
)
from tracebook.conformance import canonical_decimal

UPSTREAM_REPOSITORY = "https://github.com/nautechsystems/nautilus_trader.git"
UPSTREAM_TAG = "v2.0.0rc3"
UPSTREAM_COMMIT = "648970ce64a304d93da0a29320cb6e19b905fa39"
UPSTREAM_VERSION = "2.0.0rc3"


class NautilusTraderBookReplayAdapter:
    """Translate the normalized profile onto NautilusTrader's native L3 book."""

    def __init__(self, config):
        from nautilus_trader.model import (  # pylint: disable=import-outside-toplevel
            BookOrder,
            BookType,
            InstrumentId,
            OrderBook,
            OrderSide,
            Price,
            Quantity,
        )

        installed_version = importlib_metadata.version("nautilus-trader")
        if installed_version != UPSTREAM_VERSION:
            raise RuntimeError(
                f"NautilusTrader adapter requires {UPSTREAM_VERSION}, found {installed_version}"
            )
        self.config = config
        self.metadata = EngineMetadata(
            "NautilusTrader L3 OrderBook",
            installed_version,
            "Rust/Python",
        )
        self._BookOrder = BookOrder
        self._BookType = BookType
        self._InstrumentId = InstrumentId
        self._OrderBook = OrderBook
        self._OrderSide = OrderSide
        self._Price = Price
        self._Quantity = Quantity
        self._books = {}
        self._orders = {}

    def _book(self, symbol):
        book = self._books.get(symbol)
        if book is None:
            instrument_id = self._InstrumentId.from_str(f"{symbol}.TRACEBOOK")
            book = self._OrderBook(instrument_id, self._BookType.L3_MBO)
            self._books[symbol] = book
            self._orders[symbol] = {}
        return book

    def _native_order(self, event):
        side = self._OrderSide.BUY if event.side == "BUY" else self._OrderSide.SELL
        return self._BookOrder(
            side,
            self._Price.from_str(event.price),
            self._Quantity.from_str(event.quantity),
            event.order_id or 0,
        )

    def apply(self, event, index):
        fills: tuple[SimulatedFill, ...] = ()
        outcome = Outcome("applied")
        known_orders = self._orders.get(event.symbol, {})
        if event.op == "add":
            if event.order_id in known_orders:
                outcome = Outcome(
                    "rejected",
                    "DUPLICATE_ORDER_ID",
                    f"Order {event.order_id} is already present in {event.symbol}",
                )
            else:
                book = self._book(event.symbol)
                native_order = self._native_order(event)
                book.add(native_order, 0, index, index)
                self._orders[event.symbol][event.order_id] = native_order
        elif event.op == "update":
            if event.order_id not in known_orders:
                outcome = Outcome(
                    "rejected",
                    "ORDER_NOT_ACTIVE",
                    f"Order {event.order_id} is not present in {event.symbol}",
                )
            else:
                book = self._books[event.symbol]
                old_order = known_orders[event.order_id]
                native_order = self._native_order(event)
                self._apply_native_update(book, old_order, native_order, index)
                known_orders[event.order_id] = native_order
        elif event.op == "delete":
            if event.order_id not in known_orders:
                outcome = Outcome(
                    "rejected",
                    "ORDER_NOT_ACTIVE",
                    f"Order {event.order_id} is not present in {event.symbol}",
                )
            else:
                native_order = known_orders.pop(event.order_id)
                self._books[event.symbol].delete(native_order, 0, index, index)
        else:
            book = self._books.get(event.symbol)
            if book is not None:
                probe = self._native_order(event)
                fills = tuple(
                    SimulatedFill(
                        price=canonical_decimal(str(price)),
                        quantity=canonical_decimal(str(quantity)),
                    )
                    for price, quantity in book.simulate_fills(probe)
                )

        state = self.snapshot()
        return BookReplayObservation(
            index=index,
            outcome=outcome,
            fills=fills,
            state_hash=state.digest(),
            order_count=state.order_count,
        )

    @staticmethod
    def _apply_native_update(book, old_order, native_order, index):
        if old_order.side != native_order.side:
            # Native OrderBook.update selects one ladder from the new side. A
            # normalized side move is therefore delete+add at this boundary.
            book.delete(old_order, 0, index, index)
            book.add(native_order, 0, index, index)
        else:
            book.update(native_order, 0, index, index)

    def snapshot(self):
        snapshots = []
        for symbol in sorted(self._books):
            book = self._books[symbol]
            snapshots.append(
                BookReplaySnapshot(
                    symbol=symbol,
                    bids=self._snapshot_levels(book.bids()),
                    asks=self._snapshot_levels(book.asks()),
                )
            )
        return BookReplayState(tuple(snapshots))

    @staticmethod
    def _snapshot_levels(levels):
        return tuple(
            RestingBookOrder(
                order_id=order.order_id,
                price=canonical_decimal(str(order.price)),
                quantity=canonical_decimal(str(order.size)),
            )
            for level in levels
            for order in level.get_orders()
        )

    def close(self):
        """The native book objects own no external resources."""


if __name__ == "__main__":
    raise SystemExit(serve_book_replay_stdio(NautilusTraderBookReplayAdapter))
