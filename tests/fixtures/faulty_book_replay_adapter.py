#!/usr/bin/env python3
"""Protocol fixture that visibly requeues one same-level size increase."""

from tracebook.book_replay import (
    BookReplayObservation,
    BookReplaySnapshot,
    BookReplayState,
    EngineMetadata,
    ReferenceBookReplayAdapter,
    serve_book_replay_stdio,
)


class FaultyBookReplayAdapter:
    def __init__(self, config):
        self._inner = ReferenceBookReplayAdapter(config)
        self._reorder = False
        self.metadata = EngineMetadata(
            "faulty-book-replay-test-adapter",
            "1",
            "Python",
        )

    def apply(self, event, index):
        observation = self._inner.apply(event, index)
        if event.op == "update" and event.order_id == 1:
            self._reorder = True
        state = self.snapshot()
        return BookReplayObservation(
            index,
            observation.outcome,
            observation.fills,
            state.digest(),
            state.order_count,
        )

    def snapshot(self):
        state = self._inner.snapshot()
        if not self._reorder:
            return state
        books = []
        for book in state.books:
            bids = list(book.bids)
            if book.symbol == "TEST" and len(bids) >= 2:
                bids[0], bids[1] = bids[1], bids[0]
            books.append(BookReplaySnapshot(book.symbol, tuple(bids), book.asks))
        return BookReplayState(tuple(books))

    def close(self):
        self._inner.close()


if __name__ == "__main__":
    raise SystemExit(serve_book_replay_stdio(FaultyBookReplayAdapter))
