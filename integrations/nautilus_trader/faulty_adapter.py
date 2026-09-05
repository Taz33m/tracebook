#!/usr/bin/env python3
"""Negative control that deliberately requeues same-level Nautilus upsizes."""

from __future__ import annotations

from decimal import Decimal

from tracebook.book_replay import EngineMetadata, serve_book_replay_stdio

from integrations.nautilus_trader.adapter import NautilusTraderBookReplayAdapter


class FaultyNautilusTraderBookReplayAdapter(NautilusTraderBookReplayAdapter):
    """Inject one documented queue-priority fault into the real native book."""

    def __init__(self, config):
        super().__init__(config)
        self.metadata = EngineMetadata(
            "NautilusTrader L3 negative control: upsize requeue",
            self.metadata.version,
            self.metadata.language,
        )

    @staticmethod
    def _apply_native_update(book, old_order, native_order, index):
        same_level = old_order.side == native_order.side and old_order.price == native_order.price
        is_upsize = Decimal(str(native_order.size)) > Decimal(str(old_order.size))
        if same_level and is_upsize:
            book.delete(old_order, 0, index, index)
            book.add(native_order, 0, index, index)
            return
        NautilusTraderBookReplayAdapter._apply_native_update(
            book,
            old_order,
            native_order,
            index,
        )


if __name__ == "__main__":
    raise SystemExit(serve_book_replay_stdio(FaultyNautilusTraderBookReplayAdapter))
