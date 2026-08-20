import csv
import io
import os
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

from tools.verify_distribution_split import (
    CONFORMANCE_COMMANDS,
    CONFORMANCE_DISTRIBUTION,
    EXPECTED_VERSION,
    SIM_COMMANDS,
    SIM_DISTRIBUTION,
    VerificationError,
    inspect_wheel,
)
from tools.verify_sdist_wheel_agreement import (
    main,
    safely_extract_sdist,
    verify_wheel_agreement,
)


def _fixture_wheel(
    root: Path,
    *,
    distribution: str = CONFORMANCE_DISTRIBUTION,
    version: str = EXPECTED_VERSION,
    requirements: tuple[str, ...] = (),
    commands: dict[str, str] | None = None,
    payload: dict[str, bytes] | None = None,
    summary: str = "test fixture",
) -> Path:
    root.mkdir(parents=True)
    normalized = distribution.replace("-", "_")
    dist_info = f"{normalized}-{version}.dist-info"
    wheel_path = root / f"{normalized}-{version}-py3-none-any.whl"
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        f"Summary: {summary}",
        *(f"Requires-Dist: {requirement}" for requirement in requirements),
        "",
        "",
    ]
    files = dict(payload or {})
    files[f"{dist_info}/METADATA"] = "\n".join(metadata).encode("utf-8")
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: test fixture\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
        "\n"
    ).encode("utf-8")
    if commands:
        entries = ["[console_scripts]"]
        entries.extend(f"{name} = {target}" for name, target in sorted(commands.items()))
        files[f"{dist_info}/entry_points.txt"] = ("\n".join(entries) + "\n").encode("utf-8")
    record_path = f"{dist_info}/RECORD"
    record = io.StringIO()
    writer = csv.writer(record, lineterminator="\n")
    for path in sorted((*files, record_path)):
        writer.writerow((path, "", ""))
    files[record_path] = record.getvalue().encode("utf-8")

    with zipfile.ZipFile(wheel_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in sorted(files.items()):
            archive.writestr(path, content)
    return wheel_path


def _write_tar_member(
    archive: tarfile.TarFile,
    name: str,
    content: bytes = b"",
    *,
    member_type: bytes | None = None,
    link_name: str = "",
) -> None:
    member = tarfile.TarInfo(name)
    member.size = len(content)
    if member_type is not None:
        member.type = member_type
        member.linkname = link_name
        member.size = 0
    archive.addfile(member, io.BytesIO(content) if content else None)


@pytest.mark.parametrize(
    "unsafe_name",
    (
        "../escape.txt",
        "/absolute.txt",
        "project/../../escape.txt",
        r"C:\escape.txt",
    ),
)
def test_safe_sdist_extraction_rejects_traversal_before_writing(tmp_path, unsafe_name):
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_tar_member(archive, unsafe_name, b"escaped")

    with pytest.raises(VerificationError, match="unsafe path"):
        safely_extract_sdist(archive_path, tmp_path / "extracted")

    assert not (tmp_path / "escape.txt").exists()


def test_safe_sdist_extraction_rejects_links(tmp_path):
    archive_path = tmp_path / "linked.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        _write_tar_member(archive, "project")
        _write_tar_member(
            archive,
            "project/redirect",
            member_type=tarfile.SYMTYPE,
            link_name="../../outside",
        )

    with pytest.raises(VerificationError, match="unsupported type"):
        safely_extract_sdist(archive_path, tmp_path / "extracted")


def test_agreement_rejects_metadata_drift(tmp_path):
    original = inspect_wheel(
        _fixture_wheel(
            tmp_path / "original",
            requirements=("example-dependency>=1",),
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
        )
    )
    rebuilt = inspect_wheel(
        _fixture_wheel(
            tmp_path / "rebuilt",
            requirements=("example-dependency>=2",),
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
        )
    )

    with pytest.raises(VerificationError, match="requirements differs"):
        verify_wheel_agreement(original, rebuilt, label="test")


def test_agreement_rejects_unselected_dist_info_metadata_drift(tmp_path):
    original = inspect_wheel(
        _fixture_wheel(
            tmp_path / "original",
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
            summary="original summary",
        )
    )
    rebuilt = inspect_wheel(
        _fixture_wheel(
            tmp_path / "rebuilt",
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
            summary="drifted summary",
        )
    )

    with pytest.raises(VerificationError, match="wheel member bytes differ"):
        verify_wheel_agreement(original, rebuilt, label="test")


def test_agreement_rejects_duplicate_wheel_archive_members(tmp_path):
    original = inspect_wheel(
        _fixture_wheel(
            tmp_path / "original",
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
        )
    )
    rebuilt_path = _fixture_wheel(
        tmp_path / "rebuilt",
        commands=CONFORMANCE_COMMANDS,
        payload={"tracebook/__init__.py": b"payload\n"},
    )
    with pytest.warns(UserWarning, match="Duplicate name"):
        with zipfile.ZipFile(rebuilt_path, "a") as archive:
            archive.writestr("tracebook/__init__.py", b"duplicate\n")
    rebuilt = inspect_wheel(rebuilt_path)

    with pytest.raises(VerificationError, match="duplicate wheel archive members"):
        verify_wheel_agreement(original, rebuilt, label="test")


def test_agreement_rejects_wheel_filename_or_tag_drift(tmp_path):
    original = inspect_wheel(
        _fixture_wheel(
            tmp_path / "original",
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
        )
    )
    rebuilt_path = _fixture_wheel(
        tmp_path / "rebuilt",
        commands=CONFORMANCE_COMMANDS,
        payload={"tracebook/__init__.py": b"payload\n"},
    )
    rebuilt_path = rebuilt_path.rename(
        rebuilt_path.with_name(f"tracebook_conformance-{EXPECTED_VERSION}-1-py3-none-any.whl")
    )
    rebuilt = inspect_wheel(rebuilt_path)

    with pytest.raises(VerificationError, match="wheel filename differs"):
        verify_wheel_agreement(original, rebuilt, label="test")


@pytest.mark.parametrize(
    ("rebuilt_payload", "message"),
    (
        ({"tracebook/__init__.py": b"changed\n"}, "payload bytes differ"),
        ({"tracebook/other.py": b"payload\n"}, "payload paths differ"),
    ),
)
def test_agreement_rejects_payload_drift(tmp_path, rebuilt_payload, message):
    original = inspect_wheel(
        _fixture_wheel(
            tmp_path / "original",
            commands=CONFORMANCE_COMMANDS,
            payload={"tracebook/__init__.py": b"payload\n"},
        )
    )
    rebuilt = inspect_wheel(
        _fixture_wheel(
            tmp_path / "rebuilt",
            commands=CONFORMANCE_COMMANDS,
            payload=rebuilt_payload,
        )
    )

    with pytest.raises(VerificationError, match=message):
        verify_wheel_agreement(original, rebuilt, label="test")


def _write_project(project: Path, *, simulator: bool) -> None:
    project.mkdir()
    commands = SIM_COMMANDS if simulator else CONFORMANCE_COMMANDS
    dependencies = (
        [
            f'"{CONFORMANCE_DISTRIBUTION}=={EXPECTED_VERSION}"',
            '"numpy>=2.2.6"',
            '"psutil>=7.2.2"',
        ]
        if simulator
        else []
    )
    project_name = SIM_DISTRIBUTION if simulator else CONFORMANCE_DISTRIBUTION
    command_lines = "\n".join(f'{name} = "{target}"' for name, target in commands.items())
    dependencies_toml = ", ".join(dependencies)
    setuptools_config = (
        "packages = []\npy-modules = []\n"
        if simulator
        else 'package-dir = {"" = "src"}\n\n' '[tool.setuptools.packages.find]\nwhere = ["src"]\n'
    )
    (project / "pyproject.toml").write_text(
        f"""
[build-system]
requires = ["setuptools==84.0.0", "wheel==0.48.0"]
build-backend = "setuptools.build_meta"

[project]
name = "{project_name}"
version = "{EXPECTED_VERSION}"
description = "artifact agreement fixture"
requires-python = ">=3.10"
dependencies = [{dependencies_toml}]

[project.scripts]
{command_lines}

[tool.setuptools]
{setuptools_config}
""".lstrip(),
        encoding="utf-8",
    )
    if not simulator:
        package = project / "src" / "tracebook"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text(
            f'__version__ = "{EXPECTED_VERSION}"\n',
            encoding="utf-8",
        )


def _build_project(project: Path, output: Path) -> tuple[Path, Path]:
    environment = os.environ.copy()
    environment["PIP_NO_INDEX"] = "1"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "build",
            "--sdist",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output),
            str(project),
        ],
        check=True,
        env=environment,
        capture_output=True,
        text=True,
    )
    return next(output.glob("*.whl")), next(output.glob("*.tar.gz"))


def test_four_artifact_cli_rebuilds_and_verifies_both_distributions(tmp_path, capsys):
    conformance_project = tmp_path / "conformance-project"
    sim_project = tmp_path / "sim-project"
    _write_project(conformance_project, simulator=False)
    _write_project(sim_project, simulator=True)
    conformance_wheel, conformance_sdist = _build_project(
        conformance_project,
        tmp_path / "conformance-dist",
    )
    sim_wheel, sim_sdist = _build_project(
        sim_project,
        tmp_path / "sim-dist",
    )

    result = main(
        [
            "--conformance-wheel",
            str(conformance_wheel),
            "--conformance-sdist",
            str(conformance_sdist),
            "--sim-wheel",
            str(sim_wheel),
            "--sim-sdist",
            str(sim_sdist),
            "--expected-version",
            EXPECTED_VERSION,
        ]
    )

    assert result == 0
    assert "agreement verified for both distributions" in capsys.readouterr().out
