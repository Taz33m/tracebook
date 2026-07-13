import sys
from pathlib import Path

import pytest

from integrations.python_matching_engine.adapter import (
    UPSTREAM_PATH_ENV,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    PythonMatchingEngineAdapter,
)
from tracebook.conformance import (
    AdapterProtocolError,
    ConformanceConfig,
    ExternalProcessAdapter,
    ReferenceEngineAdapter,
    run_conformance,
)
from tracebook.events import load_market_events

ROOT = Path(__file__).resolve().parents[1]
INTEGRATION = ROOT / "integrations" / "python_matching_engine"


def test_python_matching_engine_integration_is_commit_pinned():
    assert UPSTREAM_REPOSITORY == "https://github.com/Surbeivol/PythonMatchingEngine.git"
    assert UPSTREAM_COMMIT == "f94150294a85d7b415ca4518590b5a661d6f9958"
    assert PythonMatchingEngineAdapter.metadata.version == UPSTREAM_COMMIT[:12]


def test_python_matching_engine_compatibility_trace_is_reference_conformant():
    events = load_market_events(INTEGRATION / "fifo-compatible.jsonl")
    report = run_conformance(events, ReferenceEngineAdapter, trace_name="fifo-compatible")

    assert len(events) == 13
    assert report.conformant is True
    assert report.compared_events == 13
    assert (
        report.final_state_hash
        == "21a9606e7c77c3b239259f5032245c6330ddcd1d3f7fa25394612d9818becee3"
    )


def test_python_matching_engine_adapter_explains_a_missing_upstream(monkeypatch, tmp_path):
    monkeypatch.setenv(UPSTREAM_PATH_ENV, str(tmp_path / "missing"))

    with pytest.raises(AdapterProtocolError, match="is not a directory"):
        ExternalProcessAdapter(
            [sys.executable, str(INTEGRATION / "adapter.py")],
            ConformanceConfig(),
        )


def test_030_integration_documentation_and_ci_template_are_complete():
    workflow = (ROOT / "examples" / "github-actions" / "conformance.yml").read_text(
        encoding="utf-8"
    )
    release_notes = (ROOT / "docs" / "releases" / "0.3.0.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'python -m pip install "tracebook-sim==0.3.0"' in workflow
    assert "tracebook-conformance suite" in workflow
    assert "if: always()" in workflow
    assert "Version 0.3.0 changes the category of the project." in release_notes
    assert "## Real-Engine Demo" in readme
    assert 'A <--> E["External engine"]' in readme
