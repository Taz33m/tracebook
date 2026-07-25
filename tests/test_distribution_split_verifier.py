import hashlib
import zipfile
from pathlib import Path

import pytest
import tools.verify_distribution_split as verifier

from tools.verify_distribution_split import (
    ALL_COMMANDS,
    CONFORMANCE_COMMANDS,
    CONFORMANCE_DISTRIBUTION,
    EXPECTED_VERSION,
    FACADE_COMMAND_PROBES,
    SIM_COMMANDS,
    SIM_DISTRIBUTION,
    VerificationError,
    _entry_points_bytes,
    _record_bytes,
    _resolver_install_arguments,
    build_ownership_equivalent_legacy_wheel,
    inspect_wheel,
    validate_legacy_wheel,
    verify_artifact_pair,
    verify_runtime_installation,
)


def _build_fixture_wheel(
    root: Path,
    *,
    distribution: str,
    version: str = EXPECTED_VERSION,
    requirements: tuple[str, ...] = (),
    commands: dict[str, str],
    payload: dict[str, bytes] | None = None,
) -> Path:
    normalized_name = distribution.replace("-", "_")
    dist_info = f"{normalized_name}-{version}.dist-info"
    wheel = root / f"{normalized_name}-{version}-py3-none-any.whl"
    metadata = [
        "Metadata-Version: 2.4",
        f"Name: {distribution}",
        f"Version: {version}",
        "Requires-Python: >=3.10",
    ]
    metadata.extend(f"Requires-Dist: {requirement}" for requirement in requirements)
    metadata.extend(("", ""))

    files = dict(payload or {})
    files[f"{dist_info}/METADATA"] = "\n".join(metadata).encode("utf-8")
    files[f"{dist_info}/WHEEL"] = (
        "Wheel-Version: 1.0\n"
        "Generator: test fixture\n"
        "Root-Is-Purelib: true\n"
        "Tag: py3-none-any\n"
        "\n"
    ).encode("utf-8")
    files[f"{dist_info}/entry_points.txt"] = _entry_points_bytes(commands)
    record_path = f"{dist_info}/RECORD"
    files[record_path] = _record_bytes(files, record_path)

    with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path, data in sorted(files.items()):
            archive.writestr(path, data)
    return wheel


def _valid_pair(
    tmp_path: Path,
    *,
    conformance_version: str = EXPECTED_VERSION,
    sim_version: str = EXPECTED_VERSION,
    sim_requirements: tuple[str, ...] | None = None,
    conformance_commands: dict[str, str] | None = None,
    sim_commands: dict[str, str] | None = None,
    sim_payload: dict[str, bytes] | None = None,
):
    root_payload = {
        "tracebook/__init__.py": b'__version__ = "0.6.0"\n',
        "tracebook/conformance/__init__.py": b"",
        "tracebook/conformance/cli.py": b"def main():\n    return 0\n",
    }
    conformance_wheel = _build_fixture_wheel(
        tmp_path,
        distribution=CONFORMANCE_DISTRIBUTION,
        version=conformance_version,
        commands=conformance_commands or CONFORMANCE_COMMANDS,
        payload=root_payload,
    )
    sim_wheel = _build_fixture_wheel(
        tmp_path,
        distribution=SIM_DISTRIBUTION,
        version=sim_version,
        requirements=sim_requirements
        or (
            f"{CONFORMANCE_DISTRIBUTION}=={EXPECTED_VERSION}",
            "numpy>=2.2.6",
            "psutil>=7.2.2",
            'pandas>=2.3.3; extra == "analysis"',
        ),
        commands=sim_commands or SIM_COMMANDS,
        payload=sim_payload,
    )
    return inspect_wheel(conformance_wheel), inspect_wheel(sim_wheel)


def test_valid_split_has_equal_versions_dependencies_and_disjoint_ownership(tmp_path):
    conformance, simulator = _valid_pair(tmp_path)

    verify_artifact_pair(conformance, simulator)

    assert conformance.version == simulator.version == EXPECTED_VERSION
    assert not (conformance.record_paths & simulator.record_paths)
    assert not (set(conformance.console_scripts) & set(simulator.console_scripts))
    assert {**conformance.console_scripts, **simulator.console_scripts} == ALL_COMMANDS
    assert simulator.payload_paths == frozenset()


def test_split_rejects_version_or_exact_dependency_drift(tmp_path):
    conformance, simulator = _valid_pair(
        tmp_path,
        sim_version="0.6.1",
        sim_requirements=(
            f"{CONFORMANCE_DISTRIBUTION}>=0.6.0",
            "numpy>=2.2.6",
            "psutil>=7.2.2",
        ),
    )

    with pytest.raises(VerificationError, match="coordinated version"):
        verify_artifact_pair(conformance, simulator)

    conformance, simulator = _valid_pair(
        tmp_path,
        sim_requirements=(
            f"{CONFORMANCE_DISTRIBUTION}>=0.6.0",
            "numpy>=2.2.6",
            "psutil>=7.2.2",
        ),
    )
    with pytest.raises(VerificationError, match="exact coordinated dependency"):
        verify_artifact_pair(conformance, simulator)


def test_split_rejects_shared_record_or_entry_point_ownership(tmp_path):
    conformance, simulator = _valid_pair(
        tmp_path,
        sim_payload={"tracebook/__init__.py": b"# forbidden duplicate\n"},
    )
    with pytest.raises(VerificationError, match="RECORD ownership overlaps"):
        verify_artifact_pair(conformance, simulator)

    overlapping_sim_commands = dict(SIM_COMMANDS)
    overlapping_sim_commands.update(CONFORMANCE_COMMANDS)
    conformance, simulator = _valid_pair(
        tmp_path,
        sim_commands=overlapping_sim_commands,
    )
    with pytest.raises(VerificationError, match="console scripts differ"):
        verify_artifact_pair(conformance, simulator)


def test_split_rejects_missing_or_extra_mandatory_facade_dependency(tmp_path):
    conformance, simulator = _valid_pair(
        tmp_path,
        sim_requirements=(
            f"{CONFORMANCE_DISTRIBUTION}=={EXPECTED_VERSION}",
            "numpy>=2.2.6",
        ),
    )

    with pytest.raises(VerificationError, match="dependency names must be exactly"):
        verify_artifact_pair(conformance, simulator)


def test_runtime_checks_clean_uninstalls_and_uninstall_first_migration(tmp_path, monkeypatch):
    conformance, simulator = _valid_pair(tmp_path)
    verify_artifact_pair(conformance, simulator)

    leaked_path = tmp_path / "inherited-python-path"
    (leaked_path / "numpy").mkdir(parents=True)
    (leaked_path / "numpy" / "__init__.py").write_text("", encoding="utf-8")
    (leaked_path / "psutil").mkdir()
    (leaked_path / "psutil" / "__init__.py").write_text("", encoding="utf-8")
    leaked_metadata = leaked_path / "tracebook_sim-0.5.0.dist-info"
    leaked_metadata.mkdir()
    (leaked_metadata / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: tracebook-sim\nVersion: 0.5.0\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("PYTHONPATH", str(leaked_path))

    verify_runtime_installation(conformance, simulator)


def test_resolver_check_uses_dependencies_and_safe_facade_probes(tmp_path):
    conformance, simulator = _valid_pair(tmp_path)

    arguments = _resolver_install_arguments(conformance, simulator)

    assert arguments == ("install", str(conformance.path), str(simulator.path))
    assert "--no-deps" not in arguments
    assert FACADE_COMMAND_PROBES == {
        "tracebook-sim": ("--version",),
        "tracebook-benchmark": ("--version",),
        "tracebook-dashboard": ("--version",),
        "tracebook-web": ("--version",),
        "tracebook-replay": ("--help",),
        "tracebook-coinbase": ("--help",),
        "tracebook-corpus": ("--help",),
    }


def test_resolver_cli_flag_routes_without_running_offline_or_network_checks(tmp_path, monkeypatch):
    conformance, simulator = _valid_pair(tmp_path)
    resolver_calls = []

    def artifact_for(path):
        if "conformance" in Path(path).name:
            return conformance
        return simulator

    monkeypatch.setattr(verifier, "inspect_wheel", artifact_for)
    monkeypatch.setattr(
        verifier,
        "verify_runtime_installation",
        lambda *_args, **_kwargs: pytest.fail("offline checks must be skipped"),
    )
    monkeypatch.setattr(
        verifier,
        "verify_resolver_installation",
        lambda root_artifact, sim_artifact: resolver_calls.append((root_artifact, sim_artifact)),
    )

    result = verifier.main(
        [
            "--conformance-wheel",
            str(conformance.path),
            "--sim-wheel",
            str(simulator.path),
            "--skip-runtime-checks",
            "--resolver-runtime-checks",
        ]
    )

    assert result == 0
    assert resolver_calls == [(conformance, simulator)]


def test_explicit_legacy_wheel_requires_matching_hash_and_monolith(tmp_path):
    conformance, _ = _valid_pair(tmp_path)
    legacy = build_ownership_equivalent_legacy_wheel(conformance, tmp_path / "legacy")
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()

    assert validate_legacy_wheel(legacy, digest) == legacy.resolve()
    with pytest.raises(VerificationError, match="SHA256 mismatch"):
        validate_legacy_wheel(legacy, "0" * 64)


def test_legacy_cli_flags_route_the_pinned_wheel_to_migration_check(tmp_path, monkeypatch):
    conformance, simulator = _valid_pair(tmp_path)
    legacy = build_ownership_equivalent_legacy_wheel(conformance, tmp_path / "legacy")
    digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
    runtime_calls = []

    def artifact_for(path):
        if "conformance" in Path(path).name:
            return conformance
        return simulator

    monkeypatch.setattr(verifier, "inspect_wheel", artifact_for)
    monkeypatch.setattr(
        verifier,
        "verify_runtime_installation",
        lambda root_artifact, sim_artifact, **kwargs: runtime_calls.append(
            (root_artifact, sim_artifact, kwargs)
        ),
    )

    result = verifier.main(
        [
            "--conformance-wheel",
            str(conformance.path),
            "--sim-wheel",
            str(simulator.path),
            "--legacy-wheel",
            str(legacy),
            "--legacy-wheel-sha256",
            digest,
        ]
    )

    assert result == 0
    assert runtime_calls == [
        (
            conformance,
            simulator,
            {
                "legacy_wheel": legacy,
                "legacy_wheel_sha256": digest,
            },
        )
    ]
