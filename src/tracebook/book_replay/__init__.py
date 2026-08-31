"""A separate, explicitly non-CLOB L3 book-replay conformance surface."""

from .compare import BookReplayDivergence, BookReplayReport, run_book_replay
from .external import (
    BookReplayProtocolError,
    ExternalBookReplayAdapter,
    ExternalBookReplayAdapterFactory,
)
from .model import (
    ARTIFACT_SCHEMA_VERSION,
    PROFILE_NAME,
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
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
    load_book_replay_events,
)
from .protocol import BookReplayAdapter, BookReplayAdapterFactory, serve_book_replay_stdio
from .reference import ReferenceBookReplayAdapter

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "PROFILE_NAME",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "BookReplayAdapter",
    "BookReplayAdapterFactory",
    "BookReplayConfig",
    "BookReplayDivergence",
    "BookReplayError",
    "BookReplayEvent",
    "BookReplayObservation",
    "BookReplayProtocolError",
    "BookReplayReport",
    "BookReplaySnapshot",
    "BookReplayState",
    "EngineMetadata",
    "ExternalBookReplayAdapter",
    "ExternalBookReplayAdapterFactory",
    "Outcome",
    "ReferenceBookReplayAdapter",
    "RestingBookOrder",
    "SimulatedFill",
    "load_book_replay_events",
    "run_book_replay",
    "serve_book_replay_stdio",
]
