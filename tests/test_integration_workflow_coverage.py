"""Keep native CI attached to the reference/runtime code it actually executes."""

import ast
import os
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = {
    "intrepid-orderbook": "intrepid_orderbook",
    "nautilus-trader-book-replay": "nautilus_trader",
    "orderbook-rs": "orderbook_rs",
    "gocronx-matcher": "gocronx_matcher",
    "python-matching-engine": "python_matching_engine",
}


def _pull_request_paths(workflow):
    source = (ROOT / ".github" / "workflows" / f"{workflow}.yml").read_text(encoding="utf-8")
    # Read this deliberately simple YAML sequence without adding PyYAML to the
    # dependency-free package's test requirements. Do not search job bodies,
    # where a mentioned path cannot cause the workflow to run.
    match = re.search(r"^  pull_request:\n    paths:\n((?:      - .+\n)+)", source, re.M)
    assert match is not None, f"{workflow} must declare pull_request.paths"
    patterns = [ast.literal_eval(line.removeprefix("      - ")) for line in match[1].splitlines()]
    assert all(isinstance(pattern, str) and not pattern.startswith("!") for pattern in patterns)
    return patterns


def _matches(path, pattern):
    # The checked workflows use only literal paths and positive * / ** globs.
    # Unlike fnmatch, a single * must not match a directory separator.
    expression = re.escape(pattern).replace(r"\*\*", ".*").replace(r"\*", "[^/]*")
    return re.fullmatch(expression, path) is not None


def _require_covered(workflow, paths):
    patterns = _pull_request_paths(workflow)
    missing = sorted(path for path in paths if not any(_matches(path, p) for p in patterns))
    assert not missing, f"{workflow} misses direct runtime inputs: {missing}"


def _python_sources(relative_root):
    return [path.relative_to(ROOT).as_posix() for path in (ROOT / relative_root).rglob("*.py")]


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_integration_workflows_cover_installed_reference_runtime(workflow):
    # Even L3 imports conformance.model and tracebook's package initializer.
    # Every CLOB proof directly runs the shared Python reference matching engine.
    paths = [
        "pyproject.toml",
        "MANIFEST.in",
        "src/tracebook/__init__.py",
        "src/tracebook/_version.py",
        f".github/workflows/{workflow}.yml",
        "tests/test_integration_workflow_coverage.py",
    ]
    for package in ("core", "events", "conformance"):
        paths.extend(_python_sources(f"src/tracebook/{package}"))
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "src/tracebook/conformance/fixtures").rglob("*")
        if path.suffix in {".json", ".jsonl"}
    )
    _require_covered(workflow, paths)


@pytest.mark.parametrize("workflow", WORKFLOWS)
def test_integration_workflows_cover_own_source_pins_and_retained_evidence(workflow):
    integration = ROOT / "integrations" / WORKFLOWS[workflow]
    paths = []
    for directory, directories, filenames in os.walk(integration):
        directories[:] = [name for name in directories if name not in {"target", "__pycache__"}]
        for name in filenames:
            path = Path(directory) / name
            if path.suffix in {".go", ".rs", ".py", ".json", ".jsonl"} or name in {
                "go.mod",
                "go.sum",
                "Cargo.toml",
                "Cargo.lock",
                "rust-toolchain.toml",
            }:
                paths.append(path.relative_to(ROOT).as_posix())
    assert paths
    _require_covered(workflow, paths)


@pytest.mark.parametrize("workflow", ["orderbook-rs", "gocronx-matcher"])
def test_rust_workflows_cover_the_shared_protocol_crate(workflow):
    paths = ["integrations/rust_protocol/Cargo.toml", "integrations/rust_protocol/Cargo.lock"]
    paths.extend(
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "integrations/rust_protocol/src").rglob("*.rs")
    )
    _require_covered(workflow, paths)


def test_nautilus_workflow_covers_book_replay_and_executed_test_helpers():
    paths = _python_sources("src/tracebook/book_replay")
    paths.extend(
        [
            "src/tracebook/book_replay/fixtures/l3-book-replay-v1.jsonl",
            "examples/book_replay_adapter.py",
            "integrations/__init__.py",
            "integrations/python_matching_engine/__init__.py",
            "integrations/python_matching_engine/adapter.py",
            "tests/conftest.py",
            "tests/test_book_replay.py",
            "tests/test_book_replay_campaign.py",
            "tests/test_book_replay_artifacts.py",
            "tests/fixtures/faulty_book_replay_adapter.py",
            "tests/test_external_integrations.py",
        ]
    )
    _require_covered("nautilus-trader-book-replay", paths)


def test_orderbook_rs_workflow_covers_the_executed_flash_bridge():
    _require_covered(
        "orderbook-rs",
        [
            "integrations/flash_benchmark/bridge.py",
            "integrations/flash_benchmark/artifacts/orderbook-rs-issue-88-divergence.json",
        ],
    )


def test_path_glob_matching_keeps_single_star_inside_one_directory():
    assert _matches("src/tracebook/core/order.py", "src/tracebook/core/**")
    assert _matches("tests/test_book_replay_campaign.py", "tests/test_book_replay*.py")
    assert not _matches("tests/nested/test_book_replay.py", "tests/*.py")
