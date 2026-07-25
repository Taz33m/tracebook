#!/usr/bin/env python3
"""Verify that both release sdists rebuild to their companion wheels.

The release builds each distribution twice in effect: once directly from the
checkout and once from the published source archive.  This gate proves that
those routes agree on public metadata, package payload, and every logical wheel
member byte while ignoring ZIP container timestamps.  It also applies the
coordinated wheel-ownership contract to both the original and rebuilt pairs.

Source archives are unpacked manually.  Absolute paths, parent traversal,
links, special files, duplicate normalized paths, and file/directory
collisions are rejected before anything is written.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess  # nosec B404
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

if __package__:
    from .verify_distribution_split import (
        EXPECTED_VERSION,
        VerificationError,
        WheelArtifact,
        inspect_wheel,
        verify_artifact_pair,
    )
else:
    from verify_distribution_split import (  # type: ignore[no-redef]
        EXPECTED_VERSION,
        VerificationError,
        WheelArtifact,
        inspect_wheel,
        verify_artifact_pair,
    )


def _safe_archive_parts(member_name: str) -> tuple[str, ...]:
    """Return a safe, normalized relative archive path."""

    if not member_name or "\x00" in member_name or "\\" in member_name:
        raise VerificationError(f"sdist contains unsafe path {member_name!r}")

    raw_parts = member_name.split("/")
    if raw_parts[-1] == "":
        raw_parts.pop()
    if not raw_parts or any(part in {"", ".", ".."} for part in raw_parts):
        raise VerificationError(f"sdist contains unsafe path {member_name!r}")

    posix_path = PurePosixPath(member_name)
    windows_path = PureWindowsPath(member_name)
    if posix_path.is_absolute() or windows_path.is_absolute() or bool(windows_path.drive):
        raise VerificationError(f"sdist contains unsafe path {member_name!r}")
    return tuple(raw_parts)


def safely_extract_sdist(sdist: Path, destination: Path) -> Path:
    """Extract one tar sdist and return its single project root.

    ``destination`` must not exist.  Requiring a new extraction directory and
    rejecting every link prevents an archive from redirecting later writes.
    """

    source_path = sdist.resolve()
    if not source_path.is_file():
        raise VerificationError(f"sdist not found: {source_path}")

    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)

    with tarfile.open(source_path, mode="r:*") as archive:
        validated: list[tuple[tarfile.TarInfo, tuple[str, ...]]] = []
        path_kinds: dict[tuple[str, ...], str] = {}

        for member in archive.getmembers():
            parts = _safe_archive_parts(member.name)
            if member.isdir():
                kind = "directory"
            elif member.isreg():
                kind = "file"
            else:
                raise VerificationError(
                    f"sdist member {member.name!r} has unsupported type; "
                    "only regular files and directories are allowed"
                )
            if parts in path_kinds:
                raise VerificationError(
                    f"sdist contains duplicate normalized path {'/'.join(parts)!r}"
                )
            path_kinds[parts] = kind
            validated.append((member, parts))

        if not validated:
            raise VerificationError("sdist is empty")

        top_level = {parts[0] for _member, parts in validated}
        if len(top_level) != 1:
            raise VerificationError(
                "sdist must contain exactly one top-level project directory; "
                f"found {sorted(top_level)!r}"
            )

        for parts, kind in path_kinds.items():
            for index in range(1, len(parts)):
                ancestor = parts[:index]
                if path_kinds.get(ancestor) == "file":
                    raise VerificationError(
                        "sdist contains a file/directory collision at " f"{'/'.join(ancestor)!r}"
                    )
            if kind == "file" and any(
                len(other) > len(parts) and other[: len(parts)] == parts for other in path_kinds
            ):
                raise VerificationError(
                    f"sdist file path {'/'.join(parts)!r} is also used as a directory"
                )

        for _member, parts in sorted(
            (item for item in validated if item[0].isdir()),
            key=lambda item: len(item[1]),
        ):
            destination.joinpath(*parts).mkdir(parents=True, exist_ok=False)

        for member, parts in (item for item in validated if item[0].isreg()):
            target = destination.joinpath(*parts)
            target.parent.mkdir(parents=True, exist_ok=True)
            extracted = archive.extractfile(member)
            if extracted is None:
                raise VerificationError(f"could not read sdist member {member.name!r}")
            with extracted, target.open("xb") as output:
                shutil.copyfileobj(extracted, output)

    project_root = destination / next(iter(top_level))
    if not project_root.is_dir() or not (project_root / "pyproject.toml").is_file():
        raise VerificationError("sdist's single top-level directory must contain pyproject.toml")
    return project_root


def _run_build(command: Sequence[str]) -> None:
    print(f"+ {subprocess.list2cmdline(command)}", flush=True)
    environment = os.environ.copy()
    for variable in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "__PYVENV_LAUNCHER__",
    ):
        environment.pop(variable, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PIP_NO_INDEX"] = "1"
    environment["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    subprocess.run(command, check=True, env=environment)  # nosec B603


def rebuild_wheel_from_sdist(sdist: Path, work_directory: Path) -> Path:
    """Safely unpack an sdist and rebuild exactly one wheel without isolation."""

    work_directory.mkdir(parents=True, exist_ok=False)
    project_root = safely_extract_sdist(sdist, work_directory / "source")
    wheel_directory = work_directory / "wheel"
    wheel_directory.mkdir()
    _run_build(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(wheel_directory),
            str(project_root),
        ]
    )
    wheels = sorted(wheel_directory.glob("*.whl"))
    if len(wheels) != 1:
        raise VerificationError(
            f"sdist rebuild must produce exactly one wheel; found "
            f"{[path.name for path in wheels]!r}"
        )
    return wheels[0].resolve()


def _wheel_member_bytes(artifact: WheelArtifact) -> dict[str, bytes]:
    with zipfile.ZipFile(artifact.path) as archive:
        return {
            member_path: archive.read(member_path) for member_path in sorted(artifact.record_paths)
        }


def verify_wheel_agreement(
    original: WheelArtifact,
    rebuilt: WheelArtifact,
    *,
    label: str,
) -> None:
    """Compare public metadata, payload, and every logical wheel member byte."""

    metadata_fields = (
        ("distribution", original.distribution, rebuilt.distribution),
        ("version", original.version, rebuilt.version),
        ("requirements", original.requirements, rebuilt.requirements),
        (
            "console scripts",
            dict(original.console_scripts),
            dict(rebuilt.console_scripts),
        ),
    )
    for field, expected, actual in metadata_fields:
        if expected != actual:
            raise VerificationError(
                f"{label} sdist-rebuilt wheel {field} differs from the original wheel; "
                f"original={expected!r}, rebuilt={actual!r}"
            )

    original_members = _wheel_member_bytes(original)
    rebuilt_members = _wheel_member_bytes(rebuilt)
    original_payload = {path: original_members[path] for path in original.payload_paths}
    rebuilt_payload = {path: rebuilt_members[path] for path in rebuilt.payload_paths}
    original_paths = set(original_payload)
    rebuilt_paths = set(rebuilt_payload)
    if original_paths != rebuilt_paths:
        raise VerificationError(
            f"{label} sdist-rebuilt wheel payload paths differ from the original wheel; "
            f"missing={sorted(original_paths - rebuilt_paths)!r}, "
            f"extra={sorted(rebuilt_paths - original_paths)!r}"
        )

    changed = sorted(
        path for path in original_paths if original_payload[path] != rebuilt_payload[path]
    )
    if changed:
        raise VerificationError(
            f"{label} sdist-rebuilt wheel payload bytes differ from the original wheel "
            f"at {changed!r}"
        )

    original_member_paths = set(original_members)
    rebuilt_member_paths = set(rebuilt_members)
    if original_member_paths != rebuilt_member_paths:
        raise VerificationError(
            f"{label} sdist-rebuilt wheel member paths differ from the original wheel; "
            f"missing={sorted(original_member_paths - rebuilt_member_paths)!r}, "
            f"extra={sorted(rebuilt_member_paths - original_member_paths)!r}"
        )

    changed_members = sorted(
        path for path in original_member_paths if original_members[path] != rebuilt_members[path]
    )
    if changed_members:
        raise VerificationError(
            f"{label} sdist-rebuilt wheel member bytes differ from the original wheel "
            f"at {changed_members!r}"
        )


def verify_release_artifacts(
    conformance_wheel: Path,
    conformance_sdist: Path,
    sim_wheel: Path,
    sim_sdist: Path,
    *,
    expected_version: str = EXPECTED_VERSION,
) -> None:
    """Verify direct wheels, sdist rebuilds, agreement, and pair ownership."""

    original_conformance = inspect_wheel(conformance_wheel)
    original_sim = inspect_wheel(sim_wheel)
    verify_artifact_pair(
        original_conformance,
        original_sim,
        expected_version=expected_version,
    )

    with tempfile.TemporaryDirectory(prefix="tracebook-sdist-agreement-") as temporary:
        root = Path(temporary)
        rebuilt_conformance = inspect_wheel(
            rebuild_wheel_from_sdist(conformance_sdist, root / "conformance")
        )
        rebuilt_sim = inspect_wheel(rebuild_wheel_from_sdist(sim_sdist, root / "sim"))

        verify_wheel_agreement(
            original_conformance,
            rebuilt_conformance,
            label="tracebook-conformance",
        )
        verify_wheel_agreement(
            original_sim,
            rebuilt_sim,
            label="tracebook-sim",
        )
        verify_artifact_pair(
            rebuilt_conformance,
            rebuilt_sim,
            expected_version=expected_version,
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild both Tracebook wheels from their sdists and verify artifact "
            "agreement plus coordinated ownership."
        )
    )
    parser.add_argument("--conformance-wheel", required=True, type=Path)
    parser.add_argument("--conformance-sdist", required=True, type=Path)
    parser.add_argument("--sim-wheel", required=True, type=Path)
    parser.add_argument("--sim-sdist", required=True, type=Path)
    parser.add_argument("--expected-version", default=EXPECTED_VERSION)
    args = parser.parse_args(argv)

    try:
        verify_release_artifacts(
            args.conformance_wheel,
            args.conformance_sdist,
            args.sim_wheel,
            args.sim_sdist,
            expected_version=args.expected_version,
        )
        print(
            "Sdist/wheel agreement verified for both distributions: metadata, "
            "scripts, every logical wheel member path and byte, and coordinated "
            "ownership all match."
        )
        return 0
    except (
        OSError,
        subprocess.CalledProcessError,
        tarfile.TarError,
        VerificationError,
        zipfile.BadZipFile,
    ) as error:
        print(f"sdist/wheel agreement verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
