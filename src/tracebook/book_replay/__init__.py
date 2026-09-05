"""A separate, explicitly non-CLOB L3 book-replay conformance surface."""

from .campaign import (
    BOOK_REPLAY_CAPABILITIES,
    BOOK_REPLAY_GENERATOR_VERSION,
    BookReplayCampaignFailure,
    BookReplayCampaignResult,
    BookReplayCampaignTrace,
    BookReplayCoverage,
    generate_book_replay_trace,
    measure_book_replay_coverage,
    run_book_replay_campaign,
)
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
from .minimize import BookReplayMinimizationResult, minimize_book_replay_failure
from .protocol import BookReplayAdapter, BookReplayAdapterFactory, serve_book_replay_stdio
from .reference import ReferenceBookReplayAdapter

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "BOOK_REPLAY_CAPABILITIES",
    "BOOK_REPLAY_GENERATOR_VERSION",
    "PROFILE_NAME",
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "BookReplayAdapter",
    "BookReplayAdapterFactory",
    "BookReplayCampaignFailure",
    "BookReplayCampaignResult",
    "BookReplayCampaignTrace",
    "BookReplayConfig",
    "BookReplayCoverage",
    "BookReplayDivergence",
    "BookReplayError",
    "BookReplayEvent",
    "BookReplayObservation",
    "BookReplayMinimizationResult",
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
    "generate_book_replay_trace",
    "measure_book_replay_coverage",
    "minimize_book_replay_failure",
    "run_book_replay",
    "run_book_replay_campaign",
    "serve_book_replay_stdio",
]
