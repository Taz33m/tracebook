from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import tracebook
from tracebook._version import __version__

ROOT = Path(__file__).resolve().parents[1]
DEVELOPMENT_EXTRAS = ("dev", "dashboard", "analysis", "capture")


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_runtime_version_has_single_source_of_truth():
    metadata = _pyproject()

    assert tracebook.__version__ == __version__
    assert __version__ == "0.5.0"
    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "tracebook._version.__version__"
    }


def test_distribution_name_cli_and_typing_metadata_are_release_ready():
    metadata = _pyproject()
    project = metadata["project"]
    scripts = project["scripts"]
    package_data = metadata["tool"]["setuptools"]["package-data"]

    assert project["name"] == "tracebook-sim"
    assert scripts["tracebook-replay"] == "tracebook.events.cli:main"
    assert scripts["tracebook-coinbase"] == "tracebook.events.coinbase_cli:main"
    assert scripts["tracebook-corpus"] == "tracebook.corpus.cli:main"
    assert scripts["tracebook-conformance"] == "tracebook.conformance.cli:main"
    assert package_data["tracebook.corpus.fixtures"] == ["coinbase-btcusd-synthetic-v1/*"]
    assert package_data["tracebook.conformance.fixtures.v1"] == ["*.json", "*.jsonl"]
    assert (ROOT / "src" / "tracebook" / "py.typed").is_file()
    assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


def test_contributor_requirements_delegate_to_package_extras():
    active_lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    extras = _pyproject()["project"]["optional-dependencies"]

    assert active_lines == [f"-e .[{','.join(DEVELOPMENT_EXTRAS)}]"]
    assert set(extras) == set(DEVELOPMENT_EXTRAS)
    assert not (ROOT / "setup.py").exists()


def test_dependency_groups_do_not_repeat_packages_internally():
    metadata = _pyproject()["project"]

    for group_name, requirements in {
        "runtime": metadata["dependencies"],
        **metadata["optional-dependencies"],
    }.items():
        normalized = [
            requirement.split(";", 1)[0].split("[", 1)[0].lower() for requirement in requirements
        ]
        assert len(normalized) == len(set(normalized)), f"duplicate dependency in {group_name}"


def test_release_gate_covers_research_and_integration_code():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "black --check src tests examples integrations experiments" in workflow
    assert "flake8 src tests examples integrations experiments" in workflow
    assert "mypy --python-version 3.13 src/tracebook experiments" in workflow
    assert "bandit -q -r src integrations" in workflow
    assert "compileall -q src tests examples integrations experiments" in workflow


def test_sdist_excludes_local_navigation_material():
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for directive in (
        "exclude docs/openwiki.md",
        "exclude docs/graphify.md",
        "exclude AGENTS.md",
        "exclude CLAUDE.md",
        "prune openwiki",
        "prune graphify-out",
        "prune .local-tools",
    ):
        assert directive in manifest


def test_citation_metadata_tracks_the_public_release():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'version: "0.5.0"' in citation
    assert "date-released: 2026-07-19" in citation
    assert '- name: "Taz33m"' in citation
    assert "family-names:" not in citation
    assert 'repository-code: "https://github.com/Taz33m/tracebook"' in citation
    assert "include CITATION.cff" in manifest


def test_research_docs_keep_injected_and_historical_reducers_distinct():
    release_guide = (ROOT / "docs" / "release.md").read_text(encoding="utf-8")
    field_note = (ROOT / "docs" / "field-notes" / "001-failure-forensics.md").read_text(
        encoding="utf-8"
    )

    assert "seed-42 faulty campaign" in release_guide
    assert "five-event reduced trace" in release_guide
    assert "reduced 15,739 messages to four events" in field_note
    assert "integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter" in field_note
    assert "integrations/orderbook_rs/target/release/tracebook-orderbook-rs" in field_note


def test_engine_qualification_form_captures_adoption_evidence():
    form = (ROOT / ".github" / "ISSUE_TEMPLATE" / "engine_qualification.yml").read_text(
        encoding="utf-8"
    )

    for field_id in (
        "engine",
        "revision",
        "relationship",
        "profile",
        "package_version",
        "time",
        "adapter_size",
        "failed_attempts",
        "questions",
        "result",
        "evidence",
        "ci",
        "friction",
    ):
        assert f"id: {field_id}" in form

    assert "not production certification" in form
    assert "I removed secrets, proprietary traces, and private source" in form
