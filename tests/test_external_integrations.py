import importlib.util
import sys
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from integrations.python_matching_engine.adapter import (
    UPSTREAM_PATH_ENV,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    PythonMatchingEngineAdapter,
)
from integrations.nautilus_trader.adapter import (
    UPSTREAM_COMMIT as NAUTILUS_COMMIT,
    UPSTREAM_REPOSITORY as NAUTILUS_REPOSITORY,
    UPSTREAM_TAG as NAUTILUS_TAG,
    UPSTREAM_VERSION as NAUTILUS_VERSION,
)
from tracebook.book_replay import (
    ExternalBookReplayAdapter,
    load_book_replay_events,
    run_book_replay,
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
GOCRONX_INTEGRATION = ROOT / "integrations" / "gocronx_matcher"
NAUTILUS_INTEGRATION = ROOT / "integrations" / "nautilus_trader"
INTREPID_INTEGRATION = ROOT / "integrations" / "intrepid_orderbook"
RUST_PROTOCOL = ROOT / "integrations" / "rust_protocol"


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


def test_nautilus_book_replay_integration_is_version_and_source_pinned():
    readme = (NAUTILUS_INTEGRATION / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "nautilus-trader-book-replay.yml").read_text(
        encoding="utf-8"
    )

    assert NAUTILUS_REPOSITORY == "https://github.com/nautechsystems/nautilus_trader.git"
    assert NAUTILUS_TAG == "v2.0.0rc3"
    assert NAUTILUS_COMMIT == "648970ce64a304d93da0a29320cb6e19b905fa39"
    assert NAUTILUS_VERSION == "2.0.0rc3"
    assert "LGPL-3.0-only" in readme
    assert "does not claim" in readme
    assert '"nautilus-trader==2.0.0rc3"' in workflow
    assert "actions/setup-python@v7" in workflow
    assert 'python -m pip install -e . "nautilus-trader==2.0.0rc3" "pytest>=9.1.1"' in workflow
    assert "python -m pytest" in workflow
    assert "9e0af6be935dce940a87497788c7a9a799c71f05e34a0204c9d294fce611b002" in workflow
    assert "sha256:09b1599eaeb474d98617acc7869ace26759ed5ab8350803f06197ef572864bab" in workflow
    assert "failure-dfe5c23848b63211b655" in readme
    assert (NAUTILUS_INTEGRATION / "faulty_adapter.py").is_file()
    assert (NAUTILUS_INTEGRATION / "regressions" / "upsize-requeue-reduced.jsonl").is_file()
    assert (NAUTILUS_INTEGRATION / "regressions" / "upsize-requeue-failure.json").is_file()


@pytest.mark.skipif(
    importlib.util.find_spec("nautilus_trader") is None,
    reason="the optional pinned NautilusTrader runtime is not installed",
)
def test_pinned_nautilus_runtime_conforms_to_book_replay_profile():
    events = load_book_replay_events(
        ROOT / "src" / "tracebook" / "book_replay" / "fixtures" / "l3-book-replay-v1.jsonl"
    )

    report = run_book_replay(
        events,
        lambda config: ExternalBookReplayAdapter(
            [sys.executable, str(NAUTILUS_INTEGRATION / "adapter.py")],
            config,
            timeout_seconds=10,
        ),
    )

    assert report.conformant is True
    assert report.compared_events == 17
    assert report.candidate_engine.version == NAUTILUS_VERSION
    assert (
        report.final_state_hash
        == "9e0af6be935dce940a87497788c7a9a799c71f05e34a0204c9d294fce611b002"
    )


def test_orderbook_rs_integration_pins_engine_toolchain_and_dependency_graph():
    cargo = (RUST_INTEGRATION / "Cargo.toml").read_text(encoding="utf-8")
    lock = (RUST_INTEGRATION / "Cargo.lock").read_text(encoding="utf-8")
    toolchain = (RUST_INTEGRATION / "rust-toolchain.toml").read_text(encoding="utf-8")
    readme = (RUST_INTEGRATION / "README.md").read_text(encoding="utf-8")

    assert 'orderbook-rs = { version = "=0.12.0", optional = true }' in cargo
    assert 'pricelevel = { version = "=0.9.1", optional = true }' in cargo
    assert 'uuid = { version = "1.23", features = ["v5"], optional = true }' in cargo
    assert 'name = "orderbook-rs-issue-88-adapter"' in cargo
    assert 'rev = "53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf"' in cargo
    assert 'pricelevel", version = "=0.7.0"' in cargo
    assert 'name = "faulty-orderbook-adapter"' in cargo
    assert 'name = "orderbook-rs"\nversion = "0.8.0"' in lock
    assert "53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf" in lock
    assert 'name = "orderbook-rs"\nversion = "0.12.0"' in lock
    assert 'channel = "1.88.0"' in toolchain
    assert "0e44b5b2334a6878c6a7e57491c4dfb7e2df4d72" in readme


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
    assert 'failed == {"pro-rata-allocation", "stp-cancel-resting-deep"}' in workflow
    assert 'stp["path"] == "$.state.books[0].asks"' in workflow
    assert "--profile fifo-full-v1" in workflow
    assert "--events-per-trace 100" in workflow
    assert "sha256:95c3dac9d27b770a5cccebe9ff16b6e71af443001d633b640983f02f3e04b3c9" in workflow
    assert "sha256:c04a07cbdb9931232162ef1a05740221311f3b2fa21b9b552093a2706a12c488" in workflow
    assert "--test-fault=drop-first-trade" in workflow
    assert 'assert report["divergence"]["event_index"] == 3' in workflow
    assert "failure-bc8b19d3e0e3441a98db" in workflow
    assert 'assert failure["original_divergence_event"] == 173' in workflow
    assert 'assert failure["reduced_event_count"] == 5' in workflow
    assert "--profile fifo-partial-fill-v1" in workflow
    assert "failure-7dd023c684cdb2d0fc0e" in workflow
    assert 'assert failure["reduced_event_count"] == 4' in workflow
    assert "historical orderbook-rs issue 88 adapter" in workflow
    assert "matching-engine-benchmark" in readme
    assert "four-event JSONL reproducer" in root_readme
    assert 'T["Tracebook runner"]' in readme
    assert "OrderBook-rs/issues/203" in readme
    assert "queue-consumption order even after an in-place upsize" in readme
    assert "7/9" in root_readme


def test_gocronx_matcher_integration_is_pinned_qualified_and_honest_about_assumptions():
    cargo = (GOCRONX_INTEGRATION / "Cargo.toml").read_text(encoding="utf-8")
    lock = (GOCRONX_INTEGRATION / "Cargo.lock").read_text(encoding="utf-8")
    readme = (GOCRONX_INTEGRATION / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "gocronx-matcher.yml").read_text(encoding="utf-8")

    assert 'rev = "b8d48356c8a2677e0d8a1965d754e3c4884bb947"' in cargo
    assert "b8d48356c8a2677e0d8a1965d754e3c4884bb947" in lock
    assert "snapshot format version 1" in readme
    assert "issue #7" in readme
    assert "awaiting an upstream" in readme
    assert "stable public inspection contract" in readme
    assert "tracebook-conformance qualify" in workflow
    assert "sha256:1636620f5f0f6b65fd425a6021424e9c7678bec5215555556c70a929c5e0b144" in workflow
    assert "actions/upload-artifact@v7" in workflow


def test_intrepid_orderbook_is_pinned_qualified_and_retains_the_fok_boundary():
    go_mod = (INTREPID_INTEGRATION / "go.mod").read_text(encoding="utf-8")
    go_sum = (INTREPID_INTEGRATION / "go.sum").read_text(encoding="utf-8")
    readme = (INTREPID_INTEGRATION / "README.md").read_text(encoding="utf-8")
    workflow = (ROOT / ".github" / "workflows" / "intrepid-orderbook.yml").read_text(
        encoding="utf-8"
    )
    root_readme = (ROOT / "README.md").read_text(encoding="utf-8")
    architecture = (ROOT / "docs" / "architecture.md").read_text(encoding="utf-8")

    assert "github.com/intrepidkarthi/orderbook v0.26.0" in go_mod
    assert "github.com/intrepidkarthi/orderbook v0.26.0 h1:" in go_sum
    assert "actions/setup-python@v7" in workflow
    assert "51d480cdb68b9989febb0b075d291cf891f425b3" in readme
    assert "submitted-order versus submitted-order" in readme
    assert "Inputs requiring more than six" in readme
    assert "non-zero decimal places are explicitly rejected" in readme
    assert "rounds output quantities only" in readme
    assert "maps to a distinct native account" in readme
    assert "incoming quantity times each potentially crossed maker's" in readme
    assert "6/9" in readme
    assert "failure-1ce2954857800b3a068d" in readme
    assert "sha256:f8db775baabe7e665d45bbb920a67e43d405acf445de4951617df68ffa24eb69" in workflow
    assert "sha256:08bf641bfa57da1892d744c08bca0be00f5ac1ade580ae8fcf7f195f0fcad6eb" in workflow
    assert "sha256:81a095f9804852ff7035f4c3d428a51ae3d0e8e4b81ec335128c6af15b6f65bd" in workflow
    assert "$.observation.outcome.reason" in workflow
    assert "intrepidkarthi/orderbook` v0.26.0" in root_readme
    assert "integrations/intrepid_orderbook/" in architecture
    assert (INTREPID_INTEGRATION / "adapter_test.go").is_file()
    assert (INTREPID_INTEGRATION / "regressions" / "fok-rejection-reduced.jsonl").is_file()
    assert (INTREPID_INTEGRATION / "regressions" / "fok-rejection-failure.json").is_file()


def test_native_adapters_share_one_rust_protocol_contract():
    shared_source = (RUST_PROTOCOL / "src" / "lib.rs").read_text(encoding="utf-8")
    shared_server = (RUST_PROTOCOL / "src" / "server.rs").read_text(encoding="utf-8")

    for integration in (RUST_INTEGRATION, GOCRONX_INTEGRATION):
        cargo = (integration / "Cargo.toml").read_text(encoding="utf-8")
        assert 'tracebook-conformance-protocol = { path = "../rust_protocol" }' in cargo
        assert not (integration / "src" / "wire.rs").exists()
        assert len((integration / "src" / "server.rs").read_text().splitlines()) < 40

    assert "pub trait EngineAdapter" in shared_server
    assert "event indexes must be contiguous and start at 1" in shared_server
    assert "pub fn canonical_json" in shared_source
    assert "pub fn write_frame" in shared_source


def test_source_manifest_includes_native_integration_files():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    with (ROOT / "pyproject.toml").open("rb") as stream:
        package_data = tomllib.load(stream)["tool"]["setuptools"]["package-data"]

    assert "recursive-include integrations/orderbook_rs/src *.rs" in manifest
    assert package_data["tracebook.conformance.fixtures.v2"] == ["*.json", "*.jsonl"]
    assert "recursive-include integrations *.py *.md *.json *.jsonl" in manifest
    assert "recursive-include integrations/intrepid_orderbook *.go" in manifest
    assert "include integrations/intrepid_orderbook/go.mod" in manifest
    assert "include integrations/intrepid_orderbook/go.sum" in manifest
    assert "recursive-include src/tracebook/book_replay/fixtures *.jsonl" in manifest
    assert "include integrations/orderbook_rs/Cargo.lock" in manifest
    assert "prune integrations/orderbook_rs/target" in manifest
    assert "recursive-include integrations/gocronx_matcher/src *.rs" in manifest
    assert "include integrations/gocronx_matcher/Cargo.lock" in manifest
    assert "prune integrations/gocronx_matcher/target" in manifest
    assert "recursive-include integrations/rust_protocol/src *.rs" in manifest
    assert "include integrations/rust_protocol/Cargo.lock" in manifest
    assert "prune integrations/rust_protocol/target" in manifest
    assert "recursive-include experiments *.py *.json" in manifest
    assert (RUST_INTEGRATION / "src" / "bin" / "faulty_orderbook_adapter.rs").is_file()
    assert (RUST_INTEGRATION / "src" / "bin" / "orderbook_rs_issue_88_adapter.rs").is_file()
    assert (RUST_INTEGRATION / "regressions" / "issue-88-reduced.jsonl").is_file()
    assert (RUST_INTEGRATION / "regressions" / "flash-issue-88-reduced.jsonl").is_file()


def test_030_integration_documentation_and_ci_template_are_complete():
    workflow = (ROOT / "examples" / "github-actions" / "conformance.yml").read_text(
        encoding="utf-8"
    )
    ci_docs = (ROOT / "docs" / "ci.md").read_text(encoding="utf-8")
    release_notes = (ROOT / "docs" / "releases" / "0.3.0.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert 'python -m pip install "tracebook-conformance==0.6.0"' in workflow
    assert 'python -m pip install "tracebook-conformance==0.6.0"' in ci_docs
    assert "tracebook-conformance qualify" in workflow
    assert "--output-dir artifacts/qualification" in workflow
    assert "if: always()" in workflow
    assert "Version 0.3.0 changes the category of the project." in release_notes
    assert "## Engine Adapters" in readme
    assert 'A <--> E["Candidate engine"]' in readme


def test_040_release_notes_pin_the_public_failure_story():
    notes = (ROOT / "docs" / "releases" / "0.4.0.md").read_text(encoding="utf-8")

    assert "original event 173" in notes
    assert "five causal events" in notes
    assert "one-minimal" in notes
    assert "tracebook-conformance reproduce" in notes
    assert "semantic capability coverage" in notes
    assert "external validation, not another feature expansion" in notes


def test_041_release_notes_record_external_validation_without_overclaiming():
    notes = (ROOT / "docs" / "releases" / "0.4.1.md").read_text(encoding="utf-8")

    assert "orderbook-rs` issue #203" in notes
    assert "orderbook-rs` PR #204" in notes
    assert "PriceLevel` PR #110" in notes
    assert "independent review" in notes
    assert "automatic Tracebook campaign" in notes
    assert "Conformance protocol: version 1, unchanged" in notes


def test_050_release_notes_publish_the_qualification_surface():
    notes = (ROOT / "docs" / "releases" / "0.5.0.md").read_text(encoding="utf-8")

    assert "tracebook-conformance qualify" in notes
    assert "tracebook-sim==0.5.0" in notes
    assert "public package" in notes
    assert "not production certification" in notes
