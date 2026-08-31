#!/usr/bin/env python3
"""Minimal Python adapter for the isolated L3 book-replay protocol."""

from tracebook.book_replay import (
    EngineMetadata,
    ReferenceBookReplayAdapter,
    serve_book_replay_stdio,
)


class ExampleBookReplayAdapter(ReferenceBookReplayAdapter):
    """Replace this reference-backed body with calls into a candidate L3 book."""

    def __init__(self, config):
        super().__init__(config)
        self.metadata = EngineMetadata("example-book-replay-adapter", "1", "Python")


if __name__ == "__main__":
    raise SystemExit(serve_book_replay_stdio(ExampleBookReplayAdapter))
