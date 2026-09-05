#!/usr/bin/env python3
"""Reject private experiment and local navigation paths in release artifacts.

Checks archive member names and setuptools' SOURCES.txt inventories without
extracting artifacts. The latter prevents publishing stale local path metadata
even when the referenced private files are absent from the archive.
"""

from __future__ import annotations

import argparse
import sys
import tarfile
import zipfile
import zlib
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Sequence

PRIVATE_ROOTS = (
    "experiments/private",
    ".agents",
    ".codex",
    "openwiki",
    "graphify-out",
    ".local-tools",
)
PRIVATE_FILES = {"docs/openwiki.md", "docs/graphify.md"}
LOCAL_POINTER_NAMES = {"AGENTS.md", "CLAUDE.md", "skills-lock.json"}


class PrivacyError(ValueError):
    """A release artifact contains private paths or cannot be checked."""


def _check_path(path: str, *, context: str) -> None:
    normalized = path.rstrip("/")
    parts = normalized.split("/")
    if (
        not normalized
        or "\\" in normalized
        or "\x00" in normalized
        or any(part in {"", ".", ".."} for part in parts)
        or PureWindowsPath(normalized).drive
    ):
        raise PrivacyError(f"unsafe path in {context}: {path!r}")
    if (
        any(normalized == path or normalized.endswith("/" + path) for path in PRIVATE_FILES)
        or PurePosixPath(normalized).name in LOCAL_POINTER_NAMES
        or any(
            parts[index : index + len(root.split("/"))] == root.split("/")
            for root in PRIVATE_ROOTS
            for index in range(len(parts))
        )
    ):
        raise PrivacyError(f"private path in {context}: {normalized}")


def _check_source_inventory(data: bytes, *, context: str) -> None:
    try:
        inventory = data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise PrivacyError(f"cannot decode source inventory in {context}") from error
    for path in inventory.splitlines():
        if path:
            _check_path(path, context=context)


def verify_artifact(artifact: Path) -> None:
    """Check one built wheel or gzip source distribution, without extraction."""

    if artifact.name.endswith(".whl"):
        with zipfile.ZipFile(artifact) as wheel:
            for member in wheel.infolist():
                _check_path(member.filename, context=artifact.name)
                if PurePosixPath(member.filename).name == "SOURCES.txt" and not member.is_dir():
                    _check_source_inventory(
                        wheel.read(member), context=f"{artifact.name}:{member.filename}"
                    )
    elif artifact.name.endswith(".tar.gz"):
        with tarfile.open(artifact, "r:gz") as archive:
            roots = set()
            for source_member in archive:
                _check_path(source_member.name, context=artifact.name)
                root, separator, relative = source_member.name.rstrip("/").partition("/")
                roots.add(root)
                if not separator:
                    if not source_member.isdir():
                        raise PrivacyError(
                            f"sdist member has no project root: {source_member.name!r}"
                        )
                    continue
                _check_path(relative, context=artifact.name)
                if PurePosixPath(relative).name == "SOURCES.txt":
                    source = archive.extractfile(source_member) if source_member.isreg() else None
                    if source is None:
                        raise PrivacyError(
                            f"cannot inspect source inventory: {source_member.name!r}"
                        )
                    with source:
                        _check_source_inventory(
                            source.read(), context=f"{artifact.name}:{relative}"
                        )
            if len(roots) != 1:
                raise PrivacyError(f"sdist must contain one project root: {artifact.name}")
    else:
        raise PrivacyError(f"expected a wheel or .tar.gz sdist: {artifact}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, nargs="+")
    args = parser.parse_args(argv)
    try:
        for artifact in args.artifacts:
            verify_artifact(artifact)
    except (PrivacyError, OSError, tarfile.TarError, zipfile.BadZipFile, zlib.error) as error:
        print(f"distribution privacy check failed: {error}", file=sys.stderr)
        return 1
    print(f"Distribution privacy verified for {len(args.artifacts)} artifacts.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
