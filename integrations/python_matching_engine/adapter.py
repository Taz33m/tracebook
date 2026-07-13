#!/usr/bin/env python3
"""Tracebook stdio adapter for PythonMatchingEngine's FIFO order book."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Tuple

from tracebook.conformance import (
    BookSnapshot,
    BookState,
    ConformanceConfig,
    EngineMetadata,
    Observation,
    Outcome,
    RestingOrder,
    TradeFill,
    canonical_decimal,
    serve_stdio,
)
from tracebook.core.order import NO_OWNER, OrderSide, OrderType
from tracebook.events import MarketEvent

UPSTREAM_REPOSITORY = "https://github.com/Surbeivol/PythonMatchingEngine.git"
UPSTREAM_COMMIT = "f94150294a85d7b415ca4518590b5a661d6f9958"
UPSTREAM_PATH_ENV = "PYTHON_MATCHING_ENGINE_PATH"


def _load_orderbook_class():
    upstream_path = os.environ.get(UPSTREAM_PATH_ENV)
    if upstream_path:
        resolved = Path(upstream_path).expanduser().resolve()
        if not resolved.is_dir():
            raise RuntimeError(f"{UPSTREAM_PATH_ENV} is not a directory: {resolved}")
        sys.path.insert(0, str(resolved))
    try:
        from marketsimulator.orderbook import Orderbook
    except ImportError as exc:
        raise RuntimeError(
            "PythonMatchingEngine is unavailable; clone the pinned upstream commit and "
            f"set {UPSTREAM_PATH_ENV} to its checkout"
        ) from exc
    return Orderbook


class PythonMatchingEngineAdapter:
    """Expose PythonMatchingEngine's native FIFO/LIMIT subset to Tracebook."""

    metadata = EngineMetadata(
        name="PythonMatchingEngine FIFO/LIMIT",
        version=UPSTREAM_COMMIT[:12],
        language="Python",
    )

    def __init__(self, config: ConformanceConfig) -> None:
        self.config = config
        self._orderbook_class = _load_orderbook_class()
        self._books: Dict[str, Any] = {}
        self._owners: Dict[str, Dict[int, int]] = {}

    def _book(self, symbol: str):
        book = self._books.get(symbol)
        if book is None:
            # The ticker selects an upstream MiFID band. With resilience disabled,
            # it does not alter submitted prices or Tracebook's source symbol.
            book = self._orderbook_class(ticker="san", resilience=0)
            self._books[symbol] = book
            self._owners[symbol] = {}
        return book

    @staticmethod
    def _is_active(book, order_id: int) -> bool:
        try:
            return bool(book.get(order_id)["active"])
        except KeyError:
            return False

    def apply(self, event: MarketEvent, index: int) -> Observation:
        book = self._books.get(event.symbol)
        if book is None and event.op in {"cancel", "reduce", "replace"}:
            outcome = Outcome("rejected", "ORDER_NOT_ACTIVE", "order is not active")
            return self._observation(index, outcome, ())

        book = self._book(event.symbol)
        trade_start = int(book.ntrds)
        try:
            outcome = self._apply_event(book, event, index)
        except Exception as exc:
            outcome = Outcome("rejected", "EXTERNAL_ENGINE_ERROR", str(exc))
        trades = self._trades(book, event.symbol, trade_start)
        return self._observation(index, outcome, trades)

    def _apply_event(self, book, event: MarketEvent, index: int) -> Outcome:
        if event.op == "clear":
            book.reset_ob(reset_all=True)
            self._owners[event.symbol].clear()
            return Outcome("applied")

        if event.order_id is None:
            return Outcome("rejected", "INVALID_ORDER", "order_id is required")

        if event.op == "cancel":
            if not self._is_active(book, event.order_id):
                return Outcome("rejected", "ORDER_NOT_ACTIVE", "order is not active")
            book.cancel(event.order_id)
            return Outcome("applied")

        if event.op == "reduce":
            if not self._is_active(book, event.order_id):
                return Outcome("rejected", "ORDER_NOT_ACTIVE", "order is not active")
            book.modif(event.order_id, event.quantity)
            return Outcome("applied")

        if event.op == "replace":
            return self._replace(book, event, index)

        if event.order_type != OrderType.LIMIT:
            return Outcome(
                "rejected",
                "UNSUPPORTED_ORDER_TYPE",
                "PythonMatchingEngine natively accepts limit orders only",
            )
        if self._is_active(book, event.order_id):
            return Outcome("rejected", "DUPLICATE_ORDER_ID", "order is already active")
        if event.side is None or event.price is None or event.quantity is None:
            return Outcome("rejected", "INVALID_ORDER", "new order fields are incomplete")

        self._owners[event.symbol][event.order_id] = event.owner
        book.send(
            uid=event.order_id,
            is_buy=event.side == OrderSide.BUY,
            qty=event.quantity,
            price=event.price,
            timestamp=self._timestamp(event, index),
        )
        return Outcome("applied")

    def _replace(self, book, event: MarketEvent, index: int) -> Outcome:
        if event.order_id is None or not self._is_active(book, event.order_id):
            return Outcome("rejected", "ORDER_NOT_ACTIVE", "order is not active")
        existing = book.get(event.order_id)
        price = existing["price"] if event.price is None else event.price
        quantity = existing["leavesqty"] if event.quantity is None else event.quantity
        if price is None or quantity is None or price <= 0 or quantity <= 0:
            return Outcome("rejected", "INVALID_REPLACEMENT", "price and quantity must be positive")

        book.cancel(event.order_id)
        book.send(
            uid=event.order_id,
            is_buy=bool(existing["is_buy"]),
            qty=quantity,
            price=price,
            timestamp=self._timestamp(event, index),
        )
        return Outcome("applied")

    @staticmethod
    def _timestamp(event: MarketEvent, index: int) -> datetime:
        timestamp_ns = event.timestamp_ns if event.timestamp_ns is not None else index
        return datetime(1970, 1, 1) + timedelta(microseconds=timestamp_ns // 1_000)

    def _trades(self, book, symbol: str, start: int) -> Tuple[TradeFill, ...]:
        trades = []
        for position in range(start, int(book.ntrds)):
            aggressor = int(book.trades["agg_ord"][position])
            passive = int(book.trades["pas_ord"][position])
            buy_initiated = bool(book.trades["buy_init"][position])
            trades.append(
                TradeFill(
                    symbol=symbol,
                    buy_order_id=aggressor if buy_initiated else passive,
                    sell_order_id=passive if buy_initiated else aggressor,
                    price=canonical_decimal(book.trades["price"][position]),
                    quantity=canonical_decimal(
                        book.trades["vol"][position],
                        self.config.quantity_decimal_places,
                    ),
                )
            )
        return tuple(trades)

    def _observation(
        self,
        index: int,
        outcome: Outcome,
        trades: Tuple[TradeFill, ...],
    ) -> Observation:
        state = self.snapshot()
        return Observation(index, outcome, trades, state.digest(), state.order_count)

    def snapshot(self) -> BookState:
        snapshots = []
        for symbol in sorted(self._books):
            book = self._books[symbol]
            owners = self._owners[symbol]
            snapshots.append(
                BookSnapshot(
                    symbol=symbol,
                    bids=self._resting_orders(book._bids, owners),
                    asks=self._resting_orders(book._asks, owners),
                )
            )
        return BookState(tuple(snapshots))

    def _resting_orders(self, side, owners: Dict[int, int]) -> Tuple[RestingOrder, ...]:
        resting = []
        level = side.head_pricelevel
        while level is not None:
            order = level.head_order
            while order is not None:
                resting.append(
                    RestingOrder(
                        order_id=int(order.uid),
                        price=canonical_decimal(order.price),
                        remaining_quantity=canonical_decimal(
                            order.leavesqty,
                            self.config.quantity_decimal_places,
                        ),
                        owner=owners.get(int(order.uid), NO_OWNER),
                    )
                )
                order = order.next
            level = level.next
        return tuple(resting)

    def close(self) -> None:
        """The wrapped in-process engine owns no external resources."""


def main() -> int:
    return serve_stdio(PythonMatchingEngineAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
