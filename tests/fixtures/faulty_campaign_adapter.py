#!/usr/bin/env python3
"""Test adapter that hides the first generated resting order."""

from tracebook.conformance import (
    BookSnapshot,
    BookState,
    EngineMetadata,
    Observation,
    ReferenceEngineAdapter,
    serve_stdio,
)


class FaultyCampaignAdapter:
    def __init__(self, config):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata("faulty-campaign-adapter", "1", "Python")
        self._target = None

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if self._target is None and event.op == "new":
            for book in self._inner.snapshot().books:
                orders = book.bids + book.asks
                if any(order.order_id == event.order_id for order in orders):
                    self._target = (event.symbol, event.order_id)
                    break
        state = self.snapshot()
        return Observation(
            index=observation.index,
            outcome=observation.outcome,
            trades=observation.trades,
            state_hash=state.digest(),
            resting_order_count=state.order_count,
        )

    def snapshot(self):
        state = self._inner.snapshot()
        if self._target is None:
            return state
        target_symbol, target_order_id = self._target
        return BookState(
            tuple(
                BookSnapshot(
                    symbol=book.symbol,
                    bids=tuple(
                        order
                        for order in book.bids
                        if (book.symbol, order.order_id) != (target_symbol, target_order_id)
                    ),
                    asks=tuple(
                        order
                        for order in book.asks
                        if (book.symbol, order.order_id) != (target_symbol, target_order_id)
                    ),
                )
                for book in state.books
            )
        )

    def close(self):
        self._inner.close()


def main() -> int:
    return serve_stdio(FaultyCampaignAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
