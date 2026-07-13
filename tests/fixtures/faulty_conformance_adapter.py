#!/usr/bin/env python3
"""Test-only adapter that hides source order 77 from its reported state."""

from tracebook.conformance import (
    BookSnapshot,
    BookState,
    EngineMetadata,
    Observation,
    ReferenceEngineAdapter,
    serve_stdio,
)


class FaultyAdapter:
    def __init__(self, config):
        self._inner = ReferenceEngineAdapter(config)
        self.metadata = EngineMetadata("faulty-test-adapter", "1", "Python")
        self._fault_active = False

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if event.op == "new" and event.order_id == 77:
            self._fault_active = True
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
        if not self._fault_active:
            return state
        books = []
        for book in state.books:
            books.append(
                BookSnapshot(
                    symbol=book.symbol,
                    bids=tuple(order for order in book.bids if order.order_id != 77),
                    asks=tuple(order for order in book.asks if order.order_id != 77),
                )
            )
        return BookState(tuple(books))

    def close(self):
        self._inner.close()


def main() -> int:
    return serve_stdio(FaultyAdapter)


if __name__ == "__main__":
    raise SystemExit(main())
