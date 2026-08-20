import json
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import tracebook
from tracebook._version import __version__

ROOT = Path(__file__).resolve().parents[1]
ROOT_DEVELOPMENT_EXTRAS = ("dev",)
SIMULATOR_EXTRAS = ("analysis", "capture", "dashboard")


def _pyproject() -> dict:
    with (ROOT / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def _sim_pyproject() -> dict:
    with (ROOT / "packaging" / "tracebook-sim" / "pyproject.toml").open("rb") as stream:
        return tomllib.load(stream)


def test_runtime_version_has_single_source_of_truth():
    metadata = _pyproject()

    assert tracebook.__version__ == __version__
    assert __version__ == "0.6.0"
    assert metadata["project"]["dynamic"] == ["version"]
    assert metadata["tool"]["setuptools"]["dynamic"]["version"] == {
        "attr": "tracebook._version.__version__"
    }


def test_distribution_name_cli_and_typing_metadata_are_release_ready():
    metadata = _pyproject()
    project = metadata["project"]
    scripts = project["scripts"]
    package_data = metadata["tool"]["setuptools"]["package-data"]

    assert project["name"] == "tracebook-conformance"
    assert scripts == {
        "tracebook-conformance": "tracebook.conformance.cli:main",
    }
    assert package_data["tracebook.corpus.fixtures"] == ["coinbase-btcusd-synthetic-v1/*"]
    assert package_data["tracebook.conformance.fixtures.v1"] == ["*.json", "*.jsonl"]
    assert (ROOT / "src" / "tracebook" / "py.typed").is_file()
    assert (ROOT / ".github" / "workflows" / "release.yml").is_file()


def test_distribution_platform_classifier_matches_the_release_gate():
    classifiers = _pyproject()["project"]["classifiers"]

    assert "Operating System :: POSIX :: Linux" in classifiers
    assert "Operating System :: OS Independent" not in classifiers


def test_contributor_requirements_delegate_to_package_extras():
    active_lines = [
        line.strip()
        for line in (ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    root_project = _pyproject()["project"]
    sim_project = _sim_pyproject()["project"]

    assert active_lines == [
        f"-e .[{','.join(ROOT_DEVELOPMENT_EXTRAS)}]",
        "-e ./packaging/tracebook-sim[dashboard,analysis,capture]",
    ]
    assert root_project["dependencies"] == []
    assert set(root_project["optional-dependencies"]) == set(ROOT_DEVELOPMENT_EXTRAS)
    assert set(sim_project["optional-dependencies"]) == set(SIMULATOR_EXTRAS)
    assert not (ROOT / "setup.py").exists()


def test_dependency_groups_do_not_repeat_packages_internally():
    for distribution, metadata in {
        "conformance": _pyproject()["project"],
        "simulator": _sim_pyproject()["project"],
    }.items():
        for group_name, requirements in {
            "runtime": metadata["dependencies"],
            **metadata["optional-dependencies"],
        }.items():
            normalized = [
                requirement.split(";", 1)[0].split("[", 1)[0].lower()
                for requirement in requirements
            ]
            assert len(normalized) == len(
                set(normalized)
            ), f"duplicate dependency in {distribution}:{group_name}"


def test_simulator_facade_has_exact_version_dependency_and_no_packages():
    metadata = _sim_pyproject()
    project = metadata["project"]
    setuptools = metadata["tool"]["setuptools"]
    build_requirements = ["setuptools==84.0.0", "wheel==0.47.0"]

    assert _pyproject()["build-system"]["requires"] == build_requirements
    assert metadata["build-system"]["requires"] == build_requirements
    assert project["name"] == "tracebook-sim"
    assert project["version"] == "0.6.0"
    assert project["dependencies"] == [
        "tracebook-conformance==0.6.0",
        "numpy>=2.2.6",
        "psutil>=7.2.2",
    ]
    assert set(project["scripts"]) == {
        "tracebook-sim",
        "tracebook-benchmark",
        "tracebook-dashboard",
        "tracebook-web",
        "tracebook-replay",
        "tracebook-coinbase",
        "tracebook-corpus",
    }
    assert "tracebook-conformance" not in project["scripts"]
    assert setuptools["packages"] == []
    assert setuptools["py-modules"] == []


def test_release_gate_covers_research_and_integration_code():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "black --check src tests examples integrations experiments tools" in workflow
    assert "flake8 src tests examples integrations experiments tools" in workflow
    assert "mypy --python-version 3.13 src/tracebook experiments tools" in workflow
    assert "bandit -q -r src integrations tools" in workflow
    assert "compileall -q src tests examples integrations experiments tools" in workflow
    assert "CITATION.cff must identify the release version" in workflow
    assert "SECURITY.md must contain" in workflow


def test_release_gate_proves_conformance_without_simulation_dependencies():
    ci_workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    release_workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")
    smoke = (ROOT / "tools" / "smoke_conformance_wheel.py").read_text(encoding="utf-8")
    decision = (ROOT / "packaging" / "lightweight-conformance.md").read_text(encoding="utf-8")
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    invocation = "python tools/smoke_conformance_wheel.py dist/*.whl"
    split_invocation = "python tools/verify_distribution_split.py"
    sdist_invocation = "tools/verify_sdist_wheel_agreement.py"
    assert invocation.replace("dist/*.whl", "dist/conformance/*.whl") in ci_workflow
    assert invocation.replace("dist/*.whl", "dist/conformance/*.whl") in release_workflow
    assert split_invocation in ci_workflow
    assert split_invocation in release_workflow
    assert sdist_invocation in ci_workflow
    assert sdist_invocation in release_workflow
    assert sdist_invocation in makefile
    assert "--resolver-runtime-checks" in ci_workflow
    assert "--resolver-runtime-checks" in release_workflow
    assert "--legacy-wheel dist/legacy/tracebook_sim-0.5.0-py3-none-any.whl" in (release_workflow)
    assert "d190e1c2af83e5d853b0734b4d9627b1a8f6707e0fbab391015d2d94437cd4da" in (release_workflow)
    assert release_workflow.count("python -m pip --isolated download") == 2
    assert release_workflow.count("--index-url https://pypi.org/simple") == 2
    assert "SOURCE_DATE_EPOCH=$(git show -s --format=%ct HEAD)" in release_workflow
    assert "wheel is not reproducible" in release_workflow
    assert '"numpy", "psutil"' in smoke
    assert '"--no-deps"' not in smoke
    assert '"--traces",\n                "25"' in smoke
    assert '"--events-per-trace",\n                "200"' in smoke
    assert "`tracebook-conformance` owns the `tracebook` Python package" in decision
    assert "`tracebook-sim` is a package-less compatibility facade" in decision
    assert "Uninstall the legacy owner" in decision
    assert "--force-reinstall --no-deps" in decision


def test_release_publishes_and_verifies_the_source_owner_before_the_facade():
    workflow = (ROOT / ".github" / "workflows" / "release.yml").read_text(encoding="utf-8")

    assert "publish-simulator:\n    needs: publish-conformance" in workflow
    assert "name: tracebook-conformance-distributions" in workflow
    assert "Verify the exact conformance wheel is available from PyPI" in workflow
    assert "hashlib.sha256" in workflow
    assert "published conformance wheel differs from the built release artifact" in workflow
    assert workflow.index("Publish conformance distribution first") < workflow.index(
        "Publish simulator facade after dependency availability"
    )


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
        "recursive-include packaging *.md",
        "include packaging/tracebook-sim/pyproject.toml",
        "include packaging/tracebook-sim/LICENSE",
        "recursive-include tools *.py",
    ):
        assert directive in manifest


def test_public_install_surfaces_warn_before_the_legacy_ownership_handoff():
    install_surfaces = (
        ROOT / "README.md",
        ROOT / "QUICKSTART.md",
        ROOT / "docs" / "commands.md",
        ROOT / "docs" / "corpora.md",
        ROOT / "docs" / "event-replay.md",
        ROOT / "docs" / "releases" / "0.6.0.md",
        ROOT / "packaging" / "lightweight-conformance.md",
        ROOT / "packaging" / "tracebook-sim" / "README.md",
    )

    for path in install_surfaces:
        text = path.read_text(encoding="utf-8")
        uninstall = "python -m pip uninstall -y tracebook-sim"
        pinned_installs = (
            'python -m pip install "tracebook-conformance==0.6.0"',
            'python -m pip install "tracebook-sim==0.6.0"',
            'python -m pip install "tracebook-sim[capture]==0.6.0"',
        )
        first_install = min(text.index(command) for command in pinned_installs if command in text)
        assert uninstall in text, f"{path} omits the 0.5.x ownership handoff"
        assert text.index(uninstall) < first_install


def test_citation_metadata_tracks_the_public_release():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "cff-version: 1.2.0" in citation
    assert 'version: "0.6.0"' in citation
    assert "date-released: 2026-07-27" in citation
    assert '- name: "Taz33m"' in citation
    assert "family-names:" not in citation
    assert 'repository-code: "https://github.com/Taz33m/tracebook"' in citation
    assert 'url: "https://pypi.org/project/tracebook-conformance/0.6.0/"' in citation
    assert "| `0.6.x` | Yes |" in security
    assert "| `< 0.6` | No |" in security
    assert "| `0.5.x` | Yes |" not in security
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


def test_historical_field_note_pins_the_exact_reduced_divergence():
    field_note = (ROOT / "docs" / "field-notes" / "001-failure-forensics.md").read_text(
        encoding="utf-8"
    )
    metadata_path = ROOT / "integrations" / "orderbook_rs" / "regressions" / "issue-88-failure.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    divergence = metadata["expected_reduced_divergence"]
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    assert "--metadata" in field_note
    assert "integrations/orderbook_rs/regressions/issue-88-failure.json" in field_note
    assert metadata["artifact_type"] == "tracebook.conformance.failure"
    assert metadata["failure_id"] == "failure-7dd023c684cdb2d0fc0e"
    assert metadata["failure_class"] == "queue-priority drift"
    assert "paths" not in metadata
    assert metadata["reduced_event_count"] == 4
    assert metadata["reduced_trace_sha256"] == (
        "sha256:c7b3f3132e230e74734c442a798df614691491f9ca58b8eeee49d1555bd68f76"
    )
    assert divergence["path"] == "$.observation.trades[0].sell_order_id"
    assert divergence["reference"] == 9100000001
    assert divergence["candidate"] == 9100000002
    assert metadata_path.with_name("issue-88-reduced.jsonl").is_file()
    assert "recursive-include integrations *.py *.md *.json *.jsonl" in manifest


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
