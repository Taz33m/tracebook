"""Release archives must not inherit private files or stale source inventories."""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.verify_distribution_privacy import PrivacyError, main, verify_artifact

ROOT = Path(__file__).resolve().parents[1]
PRIVATE_PATHS = (
    "experiments/private/agent-qualification-v2/gold.json",
    "experiments/private/agent-qualification-v2/execution/controller.py",
    ".agents/skills/local/SKILL.md",
    ".codex/settings.json",
    "openwiki/quickstart.md",
    "graphify-out/graph.json",
    ".local-tools/helper.py",
    "docs/openwiki.md",
    "docs/graphify.md",
    "AGENTS.md",
    "CLAUDE.md",
    "skills-lock.json",
    "examples/AGENTS.md",
)


def _write_archive(path: Path, files: dict[str, bytes]) -> None:
    if path.name.endswith(".whl"):
        with zipfile.ZipFile(path, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name, data in files.items():
                member = tarfile.TarInfo(f"privacy_fixture-1.0/{name}")
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))


@pytest.mark.parametrize("suffix", (".whl", ".tar.gz"))
@pytest.mark.parametrize(
    "private_path",
    (
        *PRIVATE_PATHS,
        "tracebook.data/data/experiments/private/gold.json",
        "src/tracebook/.agents/x.py",
    ),
)
def test_privacy_gate_rejects_private_archive_members(tmp_path, suffix, private_path):
    artifact = tmp_path / f"contaminated{suffix}"
    _write_archive(artifact, {private_path: b"synthetic private sentinel\n"})

    with pytest.raises(PrivacyError, match="private path"):
        verify_artifact(artifact)


@pytest.mark.parametrize("suffix", (".whl", ".tar.gz"))
def test_privacy_gate_rejects_stale_sources_without_private_members(tmp_path, suffix):
    artifact = tmp_path / f"stale-inventory{suffix}"
    _write_archive(
        artifact,
        {
            "pyproject.toml": b"# synthetic public file\n",
            "src/privacy_fixture.egg-info/SOURCES.txt": (
                b"pyproject.toml\nexperiments/private/synthetic-gold.json\n"
            ),
        },
    )

    with pytest.raises(PrivacyError, match="SOURCES.txt.*experiments/private/synthetic-gold.json"):
        verify_artifact(artifact)


def test_privacy_gate_cli_checks_every_artifact_and_fails_closed(tmp_path, capsys):
    clean = tmp_path / "clean.whl"
    leaked = tmp_path / "leaked.tar.gz"
    _write_archive(clean, {"tracebook/__init__.py": b"# synthetic public source\n"})
    _write_archive(leaked, {"experiments/private/gold.json": b"{}\n"})

    assert main([str(clean), str(leaked)]) == 1
    assert "leaked.tar.gz" in capsys.readouterr().err
    assert main([str(clean)]) == 0
    assert "verified for 1 artifacts" in capsys.readouterr().out
    assert main([str(tmp_path / "missing.whl")]) == 1


def test_real_manifest_excludes_synthetic_private_tree_and_stale_cache(tmp_path):
    """Build only synthetic files with the real manifest, never the private tree."""

    project = tmp_path / "source"
    project.mkdir()
    (project / "pyproject.toml").write_text(
        """\
[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "privacy-fixture"
version = "1.0"
readme = "README.md"

[tool.setuptools]
package-dir = { "" = "src" }
include-package-data = true

[tool.setuptools.packages.find]
where = ["src"]
""",
        encoding="utf-8",
    )
    (project / "MANIFEST.in").write_text(
        (ROOT / "MANIFEST.in").read_text(encoding="utf-8"), encoding="utf-8"
    )
    public_files = {
        "README.md": "Synthetic build regression.\n",
        "LICENSE": "MIT\n",
        "src/tracebook/__init__.py": "# synthetic public package\n",
        "src/tracebook/py.typed": "",
        "experiments/public-evidence.json": '{"synthetic": true}\n',
        "experiments/public_runner.py": "# synthetic public runner\n",
        "integrations/intrepid_orderbook/main.go": "package main\n",
        "integrations/intrepid_orderbook/go.mod": "module privacy-fixture\n",
        "integrations/intrepid_orderbook/go.sum": "synthetic public lockfile\n",
    }
    for path, content in public_files.items():
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    for path in PRIVATE_PATHS:
        target = project / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("synthetic private sentinel\n", encoding="utf-8")

    # setuptools reads an existing inventory before applying MANIFEST.in.
    # Seed paths that ordinary recursive includes would not otherwise discover.
    cache = project / "src/privacy_fixture.egg-info/SOURCES.txt"
    cache.parent.mkdir(parents=True)
    cache.write_text("\n".join((*public_files, *PRIVATE_PATHS)) + "\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PIP_NO_INDEX"] = "1"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(tmp_path / "dist"),
            str(project),
        ],
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    artifacts = sorted((tmp_path / "dist").iterdir())
    assert len(artifacts) == 2
    for artifact in artifacts:
        verify_artifact(artifact)

    sdist = next(path for path in artifacts if path.name.endswith(".tar.gz"))
    with tarfile.open(sdist, "r:gz") as archive:
        members = {name.partition("/")[2] for name in archive.getnames()}
    assert set(public_files) <= members
    assert not set(PRIVATE_PATHS) & members
    cached_paths = set(cache.read_text(encoding="utf-8").splitlines())
    assert not set(PRIVATE_PATHS) & cached_paths
    assert all((project / path).is_file() for path in PRIVATE_PATHS)
