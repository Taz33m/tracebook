"""Matching-engine conformance, semantic diffing, and trace reduction."""

from .campaign import (
    CAMPAIGN_GENERATOR_VERSION,
    CampaignFailure,
    CampaignProfile,
    CampaignResult,
    CampaignTraceResult,
    campaign_profile_names,
    generate_campaign_trace,
    get_campaign_profile,
    run_campaign,
    write_campaign_artifacts,
    write_campaign_corpus,
)
from .classification import (
    QUEUE_PRIORITY_DRIFT,
    classify_failure,
    is_partial_fill_priority_probe,
    is_queue_priority_probe,
)
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
from .reproduce import ReproductionResult, run_reproduction
from .semantic_coverage import SemanticCoverage, measure_semantic_coverage
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
    "CAMPAIGN_GENERATOR_VERSION",
    "AdapterFactory",
    "AdapterProtocolError",
    "BookSnapshot",
    "BookState",
    "CampaignFailure",
    "CampaignProfile",
    "CampaignResult",
    "CampaignTraceResult",
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
    "ReproductionResult",
    "RestingOrder",
    "SuiteCase",
    "SemanticCoverage",
    "TradeFill",
    "campaign_profile_names",
    "classify_failure",
    "canonical_decimal",
    "copy_bundled_conformance_suite",
    "generate_campaign_trace",
    "get_campaign_profile",
    "is_queue_priority_probe",
    "is_partial_fill_priority_probe",
    "load_conformance_suite",
    "minimize_failing_trace",
    "measure_semantic_coverage",
    "QUEUE_PRIORITY_DRIFT",
    "rejection_code",
    "run_campaign",
    "run_conformance",
    "run_conformance_suite",
    "run_reproduction",
    "serve_stdio",
    "write_campaign_artifacts",
    "write_campaign_corpus",
]
