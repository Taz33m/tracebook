#!/usr/bin/env python3
"""Build and verify Tracebook's coordinated two-distribution release.

``tracebook-conformance`` is the sole owner of the importable ``tracebook``
package and the conformance command.  The package-less ``tracebook-sim``
compatibility facade owns the remaining historical commands and depends on the
exact same conformance version.

The runtime migration check intentionally models the supported 0.5.0 upgrade:
uninstall the old monolithic ``tracebook-sim`` distribution first, then install
the coordinated pair.  A naive in-place upgrade is not treated as supported
because the old distribution owns paths that move to ``tracebook-conformance``.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
import hashlib
import io
import json
import os
import re
import subprocess  # nosec B404
import sys
import tempfile
import venv
import zipfile
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable, Mapping, Sequence

EXPECTED_VERSION = "0.6.0"
LEGACY_VERSION = "0.5.0"
OFFICIAL_LEGACY_WHEEL_SHA256 = "d190e1c2af83e5d853b0734b4d9627b1a8f6707e0fbab391015d2d94437cd4da"
CONFORMANCE_DISTRIBUTION = "tracebook-conformance"
SIM_DISTRIBUTION = "tracebook-sim"

CONFORMANCE_COMMANDS = {
    "tracebook-conformance": "tracebook.conformance.cli:main",
}
SIM_COMMANDS = {
    "tracebook-sim": "tracebook.simulation.simulation_engine:main",
    "tracebook-benchmark": "tracebook.benchmarks.runner:main",
    "tracebook-dashboard": "tracebook.visualization.dashboard:main",
    "tracebook-web": "tracebook.visualization.web_server:main",
    "tracebook-replay": "tracebook.events.cli:main",
    "tracebook-coinbase": "tracebook.events.coinbase_cli:main",
    "tracebook-corpus": "tracebook.corpus.cli:main",
}
ALL_COMMANDS = {**CONFORMANCE_COMMANDS, **SIM_COMMANDS}
FACADE_COMMAND_PROBES = {
    "tracebook-sim": ("--version",),
    "tracebook-benchmark": ("--version",),
    "tracebook-dashboard": ("--version",),
    "tracebook-web": ("--version",),
    "tracebook-replay": ("--help",),
    "tracebook-coinbase": ("--help",),
    "tracebook-corpus": ("--help",),
}
SIM_RUNTIME_DEPENDENCIES = {
    CONFORMANCE_DISTRIBUTION,
    "numpy",
    "psutil",
}


class VerificationError(RuntimeError):
    """Raised when a distribution-split invariant is not satisfied."""


class _CaseSensitiveConfigParser(configparser.ConfigParser):
    def optionxform(self, optionstr: str) -> str:
        return optionstr


def _canonicalize_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _requirement_name(requirement: str) -> str:
    requirement_without_marker = requirement.split(";", 1)[0].strip()
    match = re.match(r"([A-Za-z0-9][A-Za-z0-9._-]*)", requirement_without_marker)
    if match is None:
        raise VerificationError(f"cannot parse requirement name: {requirement!r}")
    return _canonicalize_name(match.group(1))


def _is_optional_requirement(requirement: str) -> bool:
    marker = requirement.partition(";")[2]
    return bool(re.search(r"\bextra\s*={2,3}", marker, flags=re.IGNORECASE))


def _mandatory_requirements(requirements: Iterable[str]) -> tuple[str, ...]:
    return tuple(
        requirement for requirement in requirements if not _is_optional_requirement(requirement)
    )


def _is_dist_info_path(path: str) -> bool:
    first_component = path.split("/", 1)[0]
    return first_component.endswith(".dist-info")


@dataclass(frozen=True)
class WheelArtifact:
    """The release-contract metadata extracted from one wheel."""

    path: Path
    distribution: str
    version: str
    requirements: tuple[str, ...]
    console_scripts: Mapping[str, str]
    record_paths: frozenset[str]
    payload_paths: frozenset[str]
    dist_info: str


def inspect_wheel(path: Path) -> WheelArtifact:
    """Read and structurally validate one wheel without installing it."""

    wheel = path.resolve()
    if not wheel.is_file():
        raise VerificationError(f"wheel not found: {wheel}")

    with zipfile.ZipFile(wheel) as archive:
        members = tuple(name for name in archive.namelist() if not name.endswith("/"))
        metadata_members = tuple(name for name in members if name.endswith(".dist-info/METADATA"))
        if len(metadata_members) != 1:
            raise VerificationError(
                f"{wheel.name} must contain exactly one .dist-info/METADATA; "
                f"found {len(metadata_members)}"
            )

        metadata_path = metadata_members[0]
        dist_info = metadata_path.split("/", 1)[0]
        message = BytesParser(policy=policy.default).parsebytes(archive.read(metadata_path))
        distribution = message.get("Name")
        version = message.get("Version")
        if not distribution or not version:
            raise VerificationError(f"{wheel.name} METADATA must declare Name and Version")

        entry_points_path = f"{dist_info}/entry_points.txt"
        console_scripts: dict[str, str] = {}
        if entry_points_path in members:
            parser = _CaseSensitiveConfigParser(interpolation=None, strict=True)
            parser.read_string(archive.read(entry_points_path).decode("utf-8"))
            if parser.has_section("console_scripts"):
                console_scripts = dict(parser.items("console_scripts"))

        record_path = f"{dist_info}/RECORD"
        if record_path not in members:
            raise VerificationError(f"{wheel.name} does not contain {record_path}")
        record_rows = tuple(csv.reader(io.StringIO(archive.read(record_path).decode("utf-8"))))
        if any(not row for row in record_rows):
            raise VerificationError(f"{wheel.name} contains an empty RECORD row")
        record_names = tuple(row[0] for row in record_rows)
        if len(record_names) != len(set(record_names)):
            raise VerificationError(f"{wheel.name} RECORD contains duplicate paths")
        if set(record_names) != set(members):
            missing = sorted(set(members) - set(record_names))
            stale = sorted(set(record_names) - set(members))
            raise VerificationError(
                f"{wheel.name} RECORD does not match archive members; "
                f"missing={missing!r}, stale={stale!r}"
            )
        for recorded_path in record_names:
            path_parts = Path(recorded_path).parts
            if Path(recorded_path).is_absolute() or ".." in path_parts:
                raise VerificationError(
                    f"{wheel.name} RECORD contains unsafe path {recorded_path!r}"
                )

    records = frozenset(record_names)
    return WheelArtifact(
        path=wheel,
        distribution=distribution,
        version=version,
        requirements=tuple(message.get_all("Requires-Dist", [])),
        console_scripts=console_scripts,
        record_paths=records,
        payload_paths=frozenset(path for path in records if not _is_dist_info_path(path)),
        dist_info=dist_info,
    )


def _require_exact_conformance_pin(requirements: Sequence[str], version: str) -> None:
    matching = [
        requirement
        for requirement in requirements
        if _requirement_name(requirement) == CONFORMANCE_DISTRIBUTION
    ]
    if len(matching) != 1:
        raise VerificationError(
            "tracebook-sim must have exactly one mandatory tracebook-conformance requirement"
        )
    normalized = re.sub(r"\s+", "", matching[0]).lower()
    expected = f"{CONFORMANCE_DISTRIBUTION}=={version}".lower()
    if normalized != expected:
        raise VerificationError(
            "tracebook-sim must pin the exact coordinated dependency "
            f"{expected!r}; found {matching[0]!r}"
        )


def verify_artifact_pair(
    conformance: WheelArtifact,
    simulator: WheelArtifact,
    *,
    expected_version: str = EXPECTED_VERSION,
) -> None:
    """Enforce metadata, dependency, command, and file-ownership invariants."""

    conformance_name = _canonicalize_name(conformance.distribution)
    simulator_name = _canonicalize_name(simulator.distribution)
    if conformance_name != CONFORMANCE_DISTRIBUTION:
        raise VerificationError(
            f"source-owning wheel must be {CONFORMANCE_DISTRIBUTION!r}, "
            f"found {conformance.distribution!r}"
        )
    if simulator_name != SIM_DISTRIBUTION:
        raise VerificationError(
            f"compatibility wheel must be {SIM_DISTRIBUTION!r}, "
            f"found {simulator.distribution!r}"
        )

    versions = {conformance.version, simulator.version, expected_version}
    if len(versions) != 1:
        raise VerificationError(
            "both distributions must use the coordinated version "
            f"{expected_version}; found conformance={conformance.version}, "
            f"sim={simulator.version}"
        )

    conformance_mandatory = _mandatory_requirements(conformance.requirements)
    if conformance_mandatory:
        raise VerificationError(
            "tracebook-conformance must have no mandatory runtime dependencies; "
            f"found {conformance_mandatory!r}"
        )

    simulator_mandatory = _mandatory_requirements(simulator.requirements)
    dependency_name_sequence = tuple(_requirement_name(item) for item in simulator_mandatory)
    duplicate_dependency_names = sorted(
        name for name in set(dependency_name_sequence) if dependency_name_sequence.count(name) > 1
    )
    if duplicate_dependency_names:
        raise VerificationError(
            "tracebook-sim contains duplicate mandatory dependency names: "
            f"{duplicate_dependency_names!r}"
        )
    dependency_names = set(dependency_name_sequence)
    if dependency_names != SIM_RUNTIME_DEPENDENCIES:
        raise VerificationError(
            "tracebook-sim mandatory dependency names must be exactly "
            f"{sorted(SIM_RUNTIME_DEPENDENCIES)!r}; found {sorted(dependency_names)!r}"
        )
    _require_exact_conformance_pin(simulator_mandatory, expected_version)

    if dict(conformance.console_scripts) != CONFORMANCE_COMMANDS:
        raise VerificationError(
            "tracebook-conformance console scripts differ from the expected ownership: "
            f"{dict(conformance.console_scripts)!r}"
        )
    if dict(simulator.console_scripts) != SIM_COMMANDS:
        raise VerificationError(
            "tracebook-sim console scripts differ from the expected ownership: "
            f"{dict(simulator.console_scripts)!r}"
        )
    command_overlap = set(conformance.console_scripts) & set(simulator.console_scripts)
    if command_overlap:
        raise VerificationError(
            f"console entry-point ownership overlaps: {sorted(command_overlap)!r}"
        )
    command_union = {**conformance.console_scripts, **simulator.console_scripts}
    if command_union != ALL_COMMANDS:
        raise VerificationError(
            f"coordinated command union is incomplete or unexpected: {command_union!r}"
        )

    record_overlap = conformance.record_paths & simulator.record_paths
    if record_overlap:
        raise VerificationError(f"wheel RECORD ownership overlaps: {sorted(record_overlap)!r}")
    if not any(path.startswith("tracebook/") for path in conformance.payload_paths):
        raise VerificationError("tracebook-conformance does not own the tracebook package")
    if simulator.payload_paths:
        raise VerificationError(
            "tracebook-sim must be package-less and own no wheel payload paths; "
            f"found {sorted(simulator.payload_paths)!r}"
        )


def _run(command: Sequence[str]) -> None:
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
    subprocess.run(command, check=True, env=environment)  # nosec B603


def build_wheel(project: Path, output_dir: Path, distribution: str, version: str) -> Path:
    """Build one wheel and return the uniquely matching artifact."""

    project = project.resolve()
    if not (project / "pyproject.toml").is_file():
        raise VerificationError(f"project does not contain pyproject.toml: {project}")
    output_dir.mkdir(parents=True, exist_ok=True)
    _run(
        [
            sys.executable,
            "-m",
            "build",
            "--wheel",
            "--no-isolation",
            "--outdir",
            str(output_dir),
            str(project),
        ]
    )

    matches: list[Path] = []
    for candidate in output_dir.glob("*.whl"):
        artifact = inspect_wheel(candidate)
        if (
            _canonicalize_name(artifact.distribution) == distribution
            and artifact.version == version
        ):
            matches.append(candidate)
    if len(matches) != 1:
        raise VerificationError(
            f"expected exactly one {distribution} {version} wheel in {output_dir}; "
            f"found {[path.name for path in matches]!r}"
        )
    return matches[0].resolve()


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _environment_script(environment: Path, command: str) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    return environment / ("Scripts" if os.name == "nt" else "bin") / f"{command}{suffix}"


def _pip(python: Path, *arguments: str) -> None:
    _run(
        [
            str(python),
            "-m",
            "pip",
            "--disable-pip-version-check",
            *arguments,
        ]
    )


def _resolver_install_arguments(
    conformance: WheelArtifact,
    simulator: WheelArtifact,
) -> tuple[str, ...]:
    """Return the dependency-resolving pip arguments for the coordinated pair."""

    return ("install", str(conformance.path), str(simulator.path))


def _assert_distribution_state(python: Path, expected: Mapping[str, str | None]) -> None:
    probe = """
import importlib.metadata
import json
import sys

expected = json.loads(sys.argv[1])
actual = {}
for distribution, wanted in expected.items():
    try:
        actual[distribution] = importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        actual[distribution] = None
if actual != expected:
    raise SystemExit(f"distribution state mismatch: expected {expected!r}, got {actual!r}")
"""
    _run([str(python), "-c", probe, json.dumps(expected, sort_keys=True)])


def _assert_command_state(environment: Path, present: Iterable[str]) -> None:
    expected_present = set(present)
    for command in ALL_COMMANDS:
        exists = _environment_script(environment, command).is_file()
        if exists != (command in expected_present):
            state = "present" if exists else "absent"
            raise VerificationError(
                f"expected command {command!r} to be "
                f"{'present' if command in expected_present else 'absent'}, found {state}"
            )


def _assert_tracebook_import(python: Path, *, expected: bool) -> None:
    probe = """
import importlib.util
import sys

found = importlib.util.find_spec("tracebook") is not None
wanted = sys.argv[1] == "1"
if found != wanted:
    raise SystemExit(f"tracebook import state mismatch: expected {wanted}, got {found}")
"""
    _run([str(python), "-c", probe, "1" if expected else "0"])


def _assert_conformance_is_lightweight(python: Path) -> None:
    probe = """
import importlib.util
import tracebook.conformance

for dependency in ("numpy", "psutil"):
    if importlib.util.find_spec(dependency) is not None:
        raise SystemExit(f"{dependency} unexpectedly installed with conformance")
"""
    _run([str(python), "-c", probe])


def _assert_simulator_dependencies(python: Path, *, expected: bool) -> None:
    probe = """
import importlib.metadata
import importlib.util
import sys

wanted = sys.argv[1] == "1"
for dependency in ("numpy", "psutil"):
    importable = importlib.util.find_spec(dependency) is not None
    try:
        importlib.metadata.version(dependency)
        installed = True
    except importlib.metadata.PackageNotFoundError:
        installed = False
    if importable != wanted or installed != wanted:
        raise SystemExit(
            f"{dependency} state mismatch: expected {wanted}, "
            f"importable={importable}, installed={installed}"
        )
"""
    _run([str(python), "-c", probe, "1" if expected else "0"])


def _wheel_hash(data: bytes) -> str:
    digest = hashlib.sha256(data).digest()
    encoded = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return f"sha256={encoded}"


def _entry_points_bytes(commands: Mapping[str, str]) -> bytes:
    lines = ["[console_scripts]"]
    lines.extend(f"{name} = {target}" for name, target in sorted(commands.items()))
    return ("\n".join(lines) + "\n").encode("utf-8")


def _record_bytes(files: Mapping[str, bytes], record_path: str) -> bytes:
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for path, data in sorted(files.items()):
        writer.writerow((path, _wheel_hash(data), len(data)))
    writer.writerow((record_path, "", ""))
    return output.getvalue().encode("utf-8")


def build_ownership_equivalent_legacy_wheel(
    conformance: WheelArtifact,
    output_dir: Path,
    *,
    legacy_version: str = LEGACY_VERSION,
) -> Path:
    """Create a local 0.5-style monolith for the uninstall-first migration test.

    The fixture copies the new source-owning wheel payload and gives the old
    ``tracebook-sim`` distribution ownership of it plus the historical command
    union.  It is never a release artifact and is not used to claim that a
    naive in-place upgrade is safe.
    """

    output_dir.mkdir(parents=True, exist_ok=True)
    normalized = SIM_DISTRIBUTION.replace("-", "_")
    dist_info = f"{normalized}-{legacy_version}.dist-info"
    wheel_path = output_dir / f"{normalized}-{legacy_version}-py3-none-any.whl"

    files: dict[str, bytes] = {}
    with zipfile.ZipFile(conformance.path) as source:
        for payload_path in conformance.payload_paths:
            data = source.read(payload_path)
            if payload_path == "tracebook/_version.py":
                data = (
                    '"""Package version metadata."""\n\n' f'__version__ = "{legacy_version}"\n'
                ).encode("utf-8")
            files[payload_path] = data

    files[f"{dist_info}/METADATA"] = (
        "Metadata-Version: 2.4\n"
        f"Name: {SIM_DISTRIBUTION}\n"
        f"Version: {legacy_version}\n"
        "Summary: Ownership-equivalent local fixture for migration verification\n"
        "Requires-Python: >=3.10\n"
        "Requires-Dist: numpy>=2.2.6\n"
        "Requires-Dist: psutil>=7.2.2\n"
        "\n"
    ).encode("utf-8")
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: tracebook split verifier\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
        "\n"
    ).encode("utf-8")
    files[f"{dist_info}/entry_points.txt"] = _entry_points_bytes(ALL_COMMANDS)
    files[f"{dist_info}/top_level.txt"] = b"tracebook\n"
    record_path = f"{dist_info}/RECORD"
    files[record_path] = _record_bytes(files, record_path)

    with zipfile.ZipFile(wheel_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(files.items()):
            archive.writestr(path, data)
    return wheel_path


def validate_legacy_wheel(
    path: Path,
    expected_sha256: str,
    *,
    legacy_version: str = LEGACY_VERSION,
) -> Path:
    """Validate an explicitly supplied legacy wheel before migration testing."""

    normalized_hash = expected_sha256.strip().lower()
    if not re.fullmatch(r"[0-9a-f]{64}", normalized_hash):
        raise VerificationError("legacy wheel SHA256 must contain exactly 64 hexadecimal digits")
    wheel = path.resolve()
    if not wheel.is_file():
        raise VerificationError(f"legacy wheel not found: {wheel}")
    actual_hash = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if actual_hash != normalized_hash:
        raise VerificationError(
            f"legacy wheel SHA256 mismatch: expected {normalized_hash}, found {actual_hash}"
        )

    artifact = inspect_wheel(wheel)
    if _canonicalize_name(artifact.distribution) != SIM_DISTRIBUTION:
        raise VerificationError(
            f"legacy wheel must be {SIM_DISTRIBUTION!r}, found {artifact.distribution!r}"
        )
    if artifact.version != legacy_version:
        raise VerificationError(
            f"legacy wheel must be version {legacy_version}, found {artifact.version}"
        )
    if dict(artifact.console_scripts) != ALL_COMMANDS:
        raise VerificationError(
            "legacy wheel must own the complete pre-split command set; "
            f"found {dict(artifact.console_scripts)!r}"
        )
    if "tracebook/__init__.py" not in artifact.payload_paths:
        raise VerificationError("legacy wheel must own the monolithic tracebook package")
    return wheel


def _new_environment(root: Path, name: str) -> tuple[Path, Path]:
    environment = root / name
    venv.EnvBuilder(with_pip=True, clear=True).create(environment)
    return environment, _environment_python(environment)


def verify_runtime_installation(
    conformance: WheelArtifact,
    simulator: WheelArtifact,
    *,
    legacy_version: str = LEGACY_VERSION,
    legacy_wheel: Path | None = None,
    legacy_wheel_sha256: str | None = None,
) -> None:
    """Check clean installs, safe uninstalls, and uninstall-first migration."""

    supplied_legacy_wheel: Path | None = None
    if legacy_wheel is not None:
        if legacy_wheel_sha256 is None:
            raise VerificationError("an explicit legacy wheel requires its expected SHA256")
        supplied_legacy_wheel = validate_legacy_wheel(
            legacy_wheel,
            legacy_wheel_sha256,
            legacy_version=legacy_version,
        )

    with tempfile.TemporaryDirectory(prefix="tracebook-split-runtime-") as temporary:
        root = Path(temporary)
        environment, python = _new_environment(root, "environment")

        _pip(python, "install", "--no-deps", str(conformance.path))
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: conformance.version,
                SIM_DISTRIBUTION: None,
            },
        )
        _assert_command_state(environment, CONFORMANCE_COMMANDS)
        _assert_conformance_is_lightweight(python)
        _run(
            [
                str(_environment_script(environment, "tracebook-conformance")),
                "--help",
            ]
        )
        _pip(python, "uninstall", "--yes", CONFORMANCE_DISTRIBUTION)
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: None,
                SIM_DISTRIBUTION: None,
            },
        )
        _assert_command_state(environment, ())
        _assert_tracebook_import(python, expected=False)

        _pip(
            python,
            "install",
            "--no-deps",
            str(conformance.path),
            str(simulator.path),
        )
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: conformance.version,
                SIM_DISTRIBUTION: simulator.version,
            },
        )
        _assert_command_state(environment, ALL_COMMANDS)
        _assert_tracebook_import(python, expected=True)
        _pip(python, "uninstall", "--yes", SIM_DISTRIBUTION)
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: conformance.version,
                SIM_DISTRIBUTION: None,
            },
        )
        _assert_command_state(environment, CONFORMANCE_COMMANDS)
        _assert_tracebook_import(python, expected=True)
        _pip(python, "uninstall", "--yes", CONFORMANCE_DISTRIBUTION)
        _assert_command_state(environment, ())
        _assert_tracebook_import(python, expected=False)

        # The coordinated 0.6.0 wheels also remain independently owned if the
        # source distribution is removed first.  The facade stays installed
        # (and correctly lacks its dependency) until conformance is restored.
        _pip(
            python,
            "install",
            "--no-deps",
            str(conformance.path),
            str(simulator.path),
        )
        _pip(python, "uninstall", "--yes", CONFORMANCE_DISTRIBUTION)
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: None,
                SIM_DISTRIBUTION: simulator.version,
            },
        )
        _assert_command_state(environment, SIM_COMMANDS)
        _assert_tracebook_import(python, expected=False)
        _pip(python, "install", "--no-deps", str(conformance.path))
        _assert_command_state(environment, ALL_COMMANDS)
        _assert_tracebook_import(python, expected=True)
        _pip(
            python,
            "uninstall",
            "--yes",
            SIM_DISTRIBUTION,
            CONFORMANCE_DISTRIBUTION,
        )
        _assert_command_state(environment, ())
        _assert_tracebook_import(python, expected=False)

        if supplied_legacy_wheel is None:
            selected_legacy_wheel = build_ownership_equivalent_legacy_wheel(
                conformance,
                root / "legacy-artifact",
                legacy_version=legacy_version,
            )
        else:
            selected_legacy_wheel = supplied_legacy_wheel
        _pip(python, "install", "--no-deps", str(selected_legacy_wheel))
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: None,
                SIM_DISTRIBUTION: legacy_version,
            },
        )
        _assert_command_state(environment, ALL_COMMANDS)
        _assert_tracebook_import(python, expected=True)

        # This order is the migration contract.  Installing 0.6.0 while the
        # 0.5.0 owner remains is deliberately not tested or advertised.
        _pip(python, "uninstall", "--yes", SIM_DISTRIBUTION)
        _assert_command_state(environment, ())
        _assert_tracebook_import(python, expected=False)
        _pip(
            python,
            "install",
            "--no-deps",
            str(conformance.path),
            str(simulator.path),
        )
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: conformance.version,
                SIM_DISTRIBUTION: simulator.version,
            },
        )
        _assert_command_state(environment, ALL_COMMANDS)
        _assert_tracebook_import(python, expected=True)


def verify_resolver_installation(
    conformance: WheelArtifact,
    simulator: WheelArtifact,
) -> None:
    """Run the opt-in, network-capable simulator installation proof."""

    with tempfile.TemporaryDirectory(prefix="tracebook-resolver-runtime-") as temporary:
        environment, python = _new_environment(Path(temporary), "environment")
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: None,
                SIM_DISTRIBUTION: None,
            },
        )
        _assert_command_state(environment, ())
        _assert_tracebook_import(python, expected=False)
        _assert_simulator_dependencies(python, expected=False)

        # Deliberately omit --no-deps: pip must resolve the facade's complete
        # mandatory dependency graph while using these exact two local wheels.
        _pip(python, *_resolver_install_arguments(conformance, simulator))
        _pip(python, "check")
        _assert_distribution_state(
            python,
            {
                CONFORMANCE_DISTRIBUTION: conformance.version,
                SIM_DISTRIBUTION: simulator.version,
            },
        )
        _assert_command_state(environment, ALL_COMMANDS)
        _assert_tracebook_import(python, expected=True)
        _assert_simulator_dependencies(python, expected=True)
        for command, arguments in FACADE_COMMAND_PROBES.items():
            _run([str(_environment_script(environment, command)), *arguments])


def _resolve_artifacts(args: argparse.Namespace, output_root: Path) -> tuple[Path, Path]:
    if bool(args.conformance_wheel) != bool(args.sim_wheel):
        raise VerificationError("--conformance-wheel and --sim-wheel must be supplied together")
    if args.conformance_wheel:
        return args.conformance_wheel.resolve(), args.sim_wheel.resolve()

    conformance_wheel = build_wheel(
        args.root_project,
        output_root / "tracebook-conformance",
        CONFORMANCE_DISTRIBUTION,
        args.expected_version,
    )
    sim_wheel = build_wheel(
        args.sim_project,
        output_root / "tracebook-sim",
        SIM_DISTRIBUTION,
        args.expected_version,
    )
    return conformance_wheel, sim_wheel


def main(argv: Sequence[str] | None = None) -> int:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Build and verify Tracebook's coordinated distribution split."
    )
    parser.add_argument("--root-project", type=Path, default=repository)
    parser.add_argument(
        "--sim-project",
        type=Path,
        default=repository / "packaging" / "tracebook-sim",
    )
    parser.add_argument("--expected-version", default=EXPECTED_VERSION)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="retain built wheels here; otherwise a temporary directory is used",
    )
    parser.add_argument("--conformance-wheel", type=Path)
    parser.add_argument("--sim-wheel", type=Path)
    parser.add_argument(
        "--legacy-wheel",
        type=Path,
        help=(
            "use this local tracebook-sim 0.5.0 wheel for migration verification; "
            "the verifier never downloads it"
        ),
    )
    parser.add_argument(
        "--legacy-wheel-sha256",
        help=(
            "expected SHA256 for --legacy-wheel; defaults to the official 0.5.0 "
            f"wheel hash {OFFICIAL_LEGACY_WHEEL_SHA256}"
        ),
    )
    parser.add_argument(
        "--skip-runtime-checks",
        action="store_true",
        help="skip the default offline clean virtualenv and migration checks",
    )
    parser.add_argument(
        "--resolver-runtime-checks",
        action="store_true",
        help=(
            "also create a fresh environment, resolve simulator dependencies from "
            "the package index, run pip check, and invoke all facade commands"
        ),
    )
    args = parser.parse_args(argv)

    try:
        if args.legacy_wheel_sha256 and not args.legacy_wheel:
            raise VerificationError("--legacy-wheel-sha256 requires --legacy-wheel")
        if args.legacy_wheel and args.skip_runtime_checks:
            raise VerificationError("--legacy-wheel cannot be used with --skip-runtime-checks")
        legacy_hash = None
        if args.legacy_wheel:
            legacy_hash = args.legacy_wheel_sha256 or OFFICIAL_LEGACY_WHEEL_SHA256

        if args.output_dir:
            output_root = args.output_dir.resolve()
            output_root.mkdir(parents=True, exist_ok=True)
            conformance_path, sim_path = _resolve_artifacts(args, output_root)
            conformance = inspect_wheel(conformance_path)
            simulator = inspect_wheel(sim_path)
            verify_artifact_pair(
                conformance,
                simulator,
                expected_version=args.expected_version,
            )
            if not args.skip_runtime_checks:
                verify_runtime_installation(
                    conformance,
                    simulator,
                    legacy_wheel=args.legacy_wheel,
                    legacy_wheel_sha256=legacy_hash,
                )
            if args.resolver_runtime_checks:
                verify_resolver_installation(conformance, simulator)
        else:
            with tempfile.TemporaryDirectory(prefix="tracebook-split-build-") as temporary:
                conformance_path, sim_path = _resolve_artifacts(args, Path(temporary))
                conformance = inspect_wheel(conformance_path)
                simulator = inspect_wheel(sim_path)
                verify_artifact_pair(
                    conformance,
                    simulator,
                    expected_version=args.expected_version,
                )
                if not args.skip_runtime_checks:
                    verify_runtime_installation(
                        conformance,
                        simulator,
                        legacy_wheel=args.legacy_wheel,
                        legacy_wheel_sha256=legacy_hash,
                    )
                if args.resolver_runtime_checks:
                    verify_resolver_installation(conformance, simulator)

        verified = (
            f"coordinated {args.expected_version} metadata, disjoint RECORD and "
            "entry-point ownership, and the complete command union"
        )
        if args.skip_runtime_checks and not args.resolver_runtime_checks:
            print(f"Distribution split artifact inspection passed: {verified}.")
            print("Runtime install, uninstall, and migration checks were explicitly skipped.")
        else:
            runtime_proofs: list[str] = []
            if not args.skip_runtime_checks:
                runtime_proofs.extend(
                    (
                        "clean install/uninstall",
                        "uninstall-first 0.5.0 migration",
                    )
                )
            if args.resolver_runtime_checks:
                runtime_proofs.append(
                    "fresh resolver install with pip check and seven facade command probes"
                )
            print(f"Distribution split verified: {verified}, and " f"{'; '.join(runtime_proofs)}.")
            if args.skip_runtime_checks:
                print("The default offline install/uninstall and migration checks were skipped.")
        if not args.skip_runtime_checks:
            print(
                "Naive in-place migration from the monolithic 0.5.0 distribution remains "
                f"unsupported; uninstall tracebook-sim 0.5.0 before installing "
                f"{args.expected_version}."
            )
        return 0
    except (OSError, subprocess.CalledProcessError, VerificationError, zipfile.BadZipFile) as error:
        print(f"distribution split verification failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
