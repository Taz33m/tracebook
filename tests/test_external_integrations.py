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
RUST_INTEGRATION = ROOT / "integrations" / "orderbook_rs"


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


def test_orderbook_rs_integration_pins_engine_toolchain_and_dependency_graph():
    cargo = (RUST_INTEGRATION / "Cargo.toml").read_text(encoding="utf-8")
    lock = (RUST_INTEGRATION / "Cargo.lock").read_text(encoding="utf-8")
    toolchain = (RUST_INTEGRATION / "rust-toolchain.toml").read_text(encoding="utf-8")
    readme = (RUST_INTEGRATION / "README.md").read_text(encoding="utf-8")

    assert 'orderbook-rs = "=0.10.4"' in cargo
    assert 'pricelevel = "=0.8.4"' in cargo
    assert 'name = "orderbook-rs"\nversion = "0.10.4"' in lock
    assert 'channel = "1.88.0"' in toolchain
    assert "92db5927ac59bf5f68ebdea011e6d7fe9a8ecb64" in readme


def test_orderbook_rs_compatibility_trace_is_reference_conformant():
    events = load_market_events(RUST_INTEGRATION / "fifo-compatible.jsonl")
    report = run_conformance(events, ReferenceEngineAdapter, trace_name="fifo-compatible")

    assert len(events) == 13
    assert report.conformant is True
    assert report.compared_events == 13
    assert (
        report.final_state_hash
        == "21a9606e7c77c3b239259f5032245c6330ddcd1d3f7fa25394612d9818becee3"
    )


def test_orderbook_rs_documentation_and_ci_lock_the_proof_profile():
    workflow = (ROOT / ".github" / "workflows" / "orderbook-rs.yml").read_text(encoding="utf-8")
    readme = (RUST_INTEGRATION / "README.md").read_text(encoding="utf-8")
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "cargo clippy --locked --all-targets -- -D warnings" in workflow
    assert 'failed == {"pro-rata-allocation"}' in workflow
    assert "--profile fifo-full-v1" in workflow
    assert "--events-per-trace 100" in workflow
    assert "sha256:3042184192ea03c666dd2120d8b8acc728b2805678c5fb5fdd849bf97a00925d" in workflow
    assert "--test-fault=drop-first-trade" in workflow
    assert 'assert report["divergence"]["event_index"] == 3' in workflow
    assert 'T["Tracebook runner"]' in readme
    assert "7/8" in root_readme


def test_source_manifest_includes_native_integration_files():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "recursive-include integrations/orderbook_rs/src *.rs" in manifest
    assert "include integrations/orderbook_rs/Cargo.lock" in manifest
    assert "prune integrations/orderbook_rs/target" in manifest


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
