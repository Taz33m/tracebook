"""Matching-engine conformance, semantic diffing, and trace reduction."""

from .compare import ConformanceReport, Divergence, run_conformance
from .external import (
    AdapterProtocolError,
    ExternalProcessAdapter,
    ExternalProcessAdapterFactory,
)
from .minimize import MinimizationResult, minimize_failing_trace
from .model import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookSnapshot,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
    Outcome,
    RestingOrder,
    TradeFill,
    canonical_decimal,
)
from .protocol import AdapterFactory, EngineAdapter, serve_stdio
from .reference import ReferenceEngineAdapter, rejection_code
from .suite import (
    ConformanceSuite,
    SuiteCase,
    copy_bundled_conformance_suite,
    load_conformance_suite,
    run_conformance_suite,
)

__all__ = [
    "PROTOCOL_NAME",
    "PROTOCOL_VERSION",
    "AdapterFactory",
    "AdapterProtocolError",
    "BookSnapshot",
    "BookState",
    "ConformanceConfig",
    "ConformanceError",
    "ConformanceReport",
    "ConformanceSuite",
    "Divergence",
    "EngineAdapter",
    "EngineMetadata",
    "ExternalProcessAdapter",
    "ExternalProcessAdapterFactory",
    "MinimizationResult",
    "Observation",
    "Outcome",
    "ReferenceEngineAdapter",
    "RestingOrder",
    "SuiteCase",
    "TradeFill",
    "canonical_decimal",
    "copy_bundled_conformance_suite",
    "load_conformance_suite",
    "minimize_failing_trace",
    "rejection_code",
    "run_conformance",
    "run_conformance_suite",
    "serve_stdio",
]
