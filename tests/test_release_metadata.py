import ast
import re
from pathlib import Path

import tracebook
from tracebook._version import __version__

ROOT = Path(__file__).resolve().parents[1]
SETUP_REQUIREMENT_GROUPS = (
    "CORE_REQUIREMENTS",
    "DASHBOARD_REQUIREMENTS",
    "ANALYSIS_REQUIREMENTS",
    "CAPTURE_REQUIREMENTS",
    "DEV_REQUIREMENTS",
)


def _normalize_requirement(requirement: str) -> str:
    return re.sub(r"\s+", "", requirement).lower().replace("_", "-")


def _requirement_name(requirement: str) -> str:
    name = re.split(r"[<>=!~;\\[]", requirement, maxsplit=1)[0]
    return name.strip().lower().replace("_", "-")


def _setup_requirements() -> dict[str, list[str]]:
    tree = ast.parse((ROOT / "setup.py").read_text(encoding="utf-8"))
    groups: dict[str, list[str]] = {}

    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        name = node.targets[0].id
        if name in SETUP_REQUIREMENT_GROUPS:
            groups[name] = ast.literal_eval(node.value)

    return groups


def _requirements_txt() -> dict[str, str]:
    requirements: dict[str, str] = {}
    for raw_line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = raw_line.strip()
        if not requirement or requirement.startswith("#") or requirement.startswith("-"):
            continue
        requirements[_requirement_name(requirement)] = requirement
    return requirements


def test_runtime_version_has_single_source_of_truth():
    assert tracebook.__version__ == __version__
    assert __version__ == "0.4.0"


def test_distribution_name_cli_and_typing_metadata_are_release_ready():
    setup_text = (ROOT / "setup.py").read_text(encoding="utf-8")

    assert 'name="tracebook-sim"' in setup_text
    assert "tracebook-replay=tracebook.events.cli:main" in setup_text
    assert "tracebook-coinbase=tracebook.events.coinbase_cli:main" in setup_text
    assert "tracebook-corpus=tracebook.corpus.cli:main" in setup_text
    assert "tracebook-conformance=tracebook.conformance.cli:main" in setup_text
    assert '"tracebook.corpus.fixtures": ["coinbase-btcusd-synthetic-v1/*"]' in setup_text
    assert '"tracebook.conformance.fixtures.v1": ["*.json", "*.jsonl"]' in setup_text
    assert (ROOT / "src" / "tracebook" / "py.typed").is_file()
    assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


def test_setup_dependencies_match_requirements_txt_bounds():
    requirements = _requirements_txt()

    for group_name, group_requirements in _setup_requirements().items():
        for setup_requirement in group_requirements:
            package_name = _requirement_name(setup_requirement)

            assert package_name in requirements, (
                f"{group_name} declares {setup_requirement!r}, "
                "but requirements.txt does not include it"
            )
            assert _normalize_requirement(setup_requirement) == _normalize_requirement(
                requirements[package_name]
            )
