#!/usr/bin/env python3
"""Prove that the public conformance wheel is independently installable."""

from __future__ import annotations

import argparse
import json
import os
import subprocess  # nosec B404
import tempfile
import venv
from pathlib import Path
from typing import Sequence


def _run(command: Sequence[str]) -> None:
    rendered = subprocess.list2cmdline(command)
    print(f"+ {rendered}", flush=True)
    environment = os.environ.copy()
    for variable in (
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONUSERBASE",
        "__PYVENV_LAUNCHER__",
    ):
        environment.pop(variable, None)
    environment["PYTHONNOUSERSITE"] = "1"
    # Every command is an argument vector assembled by this release script; no
    # shell is involved.
    subprocess.run(command, check=True, env=environment)  # nosec B603


def _environment_python(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _environment_script(environment: Path, name: str) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / f"{name}.exe"
    return environment / "bin" / name


def _assert_lightweight_environment(python: Path) -> None:
    probe = """
import importlib.util

for package in ("numpy", "psutil"):
    if importlib.util.find_spec(package) is not None:
        raise SystemExit(f"{package} unexpectedly installed in isolated environment")

import tracebook.conformance
import tracebook.book_replay
import importlib.metadata
from importlib import resources

assert importlib.metadata.version("tracebook-conformance") == tracebook.__version__
assert resources.files("tracebook.book_replay.fixtures").joinpath(
    "l3-book-replay-v1.jsonl"
).is_file()
"""
    _run([str(python), "-c", probe])


def _assert_script_ownership(environment: Path) -> None:
    if not _environment_script(environment, "tracebook-conformance").is_file():
        raise RuntimeError("conformance environment is missing tracebook-conformance")
    simulator_commands = (
        "tracebook-sim",
        "tracebook-benchmark",
        "tracebook-dashboard",
        "tracebook-web",
        "tracebook-replay",
        "tracebook-coinbase",
        "tracebook-corpus",
    )
    unexpected = [
        command
        for command in simulator_commands
        if _environment_script(environment, command).exists()
    ]
    if unexpected:
        raise RuntimeError(
            "conformance-only install unexpectedly owns simulator commands: "
            + ", ".join(unexpected)
        )


def _assert_qualification(report_path: Path) -> None:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    fixed = report["checks"]["fixed_cases"]
    generated = report["checks"]["generated_campaign"]
    coverage = report["checks"]["semantic_coverage"]
    campaign = report["campaign"]

    expected_fixed = {"passed": 3, "total": 3, "complete": True}
    expected_generated = {
        "completed_traces": 25,
        "requested_traces": 25,
        "conformant": True,
    }
    expected_coverage = {
        "covered": 10,
        "expected": 10,
        "uncovered": [],
        "complete": True,
    }
    checks = {
        "qualified": (report["qualified"], True),
        "fixed cases": (fixed, expected_fixed),
        "generated campaign": (generated, expected_generated),
        "generated events": (campaign["generated_events"], 5_000),
        "semantic coverage": (coverage, expected_coverage),
    }
    for name, (actual, expected) in checks.items():
        if actual != expected:
            raise RuntimeError(f"unexpected {name}: expected {expected!r}, got {actual!r}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Smoke a Tracebook wheel in a dependency-empty virtual environment."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--adapter",
        type=Path,
        default=Path("examples/conformance_adapter.py"),
    )
    args = parser.parse_args(argv)

    wheel = args.wheel.resolve()
    adapter = args.adapter.resolve()
    if not wheel.is_file():
        parser.error(f"wheel not found: {wheel}")
    if not adapter.is_file():
        parser.error(f"adapter not found: {adapter}")

    with tempfile.TemporaryDirectory(prefix="tracebook-wheel-smoke-") as temporary:
        root = Path(temporary)
        environment = root / "venv"
        output = root / "qualification"
        # Match `python -m venv`: keep executable-relative runtime libraries
        # reachable on POSIX, while retaining the default Windows copy mode.
        venv.EnvBuilder(with_pip=True, clear=True, symlinks=os.name != "nt").create(environment)
        python = _environment_python(environment)
        conformance = _environment_script(environment, "tracebook-conformance")

        _run(
            [
                str(python),
                "-m",
                "pip",
                "--disable-pip-version-check",
                "install",
                str(wheel),
            ]
        )
        _assert_lightweight_environment(python)
        _assert_script_ownership(environment)
        _run(
            [
                str(conformance),
                "qualify",
                "--output-dir",
                str(output),
                "--profile",
                "fifo-limit-v1",
                "--seed",
                "42",
                "--traces",
                "25",
                "--events-per-trace",
                "200",
                "--candidate",
                str(python),
                str(adapter),
            ]
        )
        _assert_qualification(output / "qualification.json")

    print(
        "Lightweight conformance proof passed: "
        "3/3 fixed cases, 25/25 traces, 5,000 events, 10/10 capabilities."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
