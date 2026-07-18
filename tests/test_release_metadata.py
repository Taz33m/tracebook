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
    assert __version__ == "0.4.1"
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
