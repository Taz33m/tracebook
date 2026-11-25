"""Rebuild the two pinned Rust candidates and reduce Flash's historical FIFO trace."""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import subprocess  # nosec B404
import tempfile
from importlib import metadata
from pathlib import Path

from ..events import load_market_events
from .compare import run_conformance
from .external import ExternalProcessAdapterFactory
from .minimize import minimize_failing_trace
from .model import ConformanceConfig, ConformanceError, canonical_event_jsonl_line

_PREFIX_HASH = "4e0bb497924a68dcca9575a5860954089bf5d137ab1f577618fba482a19877cd"
_REDUCED_HASH = "71ddb3f8d9c987b634bf0fed9a3fec688caaf7cb8a264a971037e5a8cf00d59d"
_AFFECTED = "orderbook-rs-issue-88-adapter"
_FIXED = "tracebook-orderbook-rs"


def _manifest() -> Path:
    relative = "share/tracebook/fifo-proof/orderbook_rs/Cargo.toml"
    for entry in metadata.files("tracebook-conformance") or ():
        if entry.match(relative):
            return Path(entry.locate())
    # Editable installations may omit wheel data-files from their RECORD.
    checkout = Path(__file__).resolve().parents[3] / "integrations/orderbook_rs/Cargo.toml"
    if checkout.is_file():
        return checkout
    raise ConformanceError("the installed distribution is missing the FIFO proof Rust sources")


def _build_adapters(
    scratch: Path,
) -> tuple[ExternalProcessAdapterFactory, ExternalProcessAdapterFactory]:
    rustup = shutil.which("rustup")
    if rustup is None:
        raise ConformanceError("FIFO proof requires rustup and the 1.88.0 toolchain")
    target = scratch / "target"
    build = [
        rustup,
        "run",
        "1.88.0",
        "cargo",
        "build",
        "--locked",
        "--release",
        "--manifest-path",
        str(_manifest()),
        "--target-dir",
        str(target),
    ]
    for binary, features in [
        (_AFFECTED, ["--no-default-features", "--features", "historical-issue-88"]),
        (_FIXED, []),
    ]:
        print(f"Building {binary} with Rust 1.88.0 ...", flush=True)
        try:
            subprocess.run(
                [*build, *features, "--bin", binary], cwd=scratch, check=True
            )  # nosec B603
        except subprocess.CalledProcessError as exc:
            raise ConformanceError(
                f"building {binary} failed (exit {exc.returncode}); see Cargo output"
            ) from exc
    return (
        ExternalProcessAdapterFactory([str(target / "release" / _AFFECTED)]),
        ExternalProcessAdapterFactory([str(target / "release" / _FIXED)]),
    )


def run_fifo_proof(output: Path) -> None:
    """Publish the reduced JSONL only after both real candidates verify it."""
    output = output.expanduser()
    if os.path.lexists(output):
        raise ConformanceError(f"proof output already exists: {output}")
    fixture = Path(__file__).parent / "fixtures/flash/issue-88-prefix.jsonl.gz"
    with gzip.open(fixture, "rb") as stream:
        prefix = stream.read()
    if hashlib.sha256(prefix).hexdigest() != _PREFIX_HASH:
        raise ConformanceError("Flash prefix differs from its recorded SHA-256")

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".fifo-proof-", dir=output.parent) as temporary:
        scratch = Path(temporary).resolve()
        source = scratch / "prefix.jsonl"
        source.write_bytes(prefix)
        events = load_market_events(source)
        affected, fixed = _build_adapters(scratch)
        config = ConformanceConfig(tick_size=1, quantity_decimal_places=0)
        original = run_conformance(events, affected, config)
        if (
            original.operational_failure
            or original.divergence is None
            or original.divergence.event_index != len(events)
            or len(events) != 15739
        ):
            raise ConformanceError("historical engine did not diverge at Flash event 15739")
        print(f"Reducing {len(events):,} events (200 candidate runs available) ...", flush=True)
        reduced = minimize_failing_trace(events, affected, config, max_runs=200)
        trace = b"".join(canonical_event_jsonl_line(event) for event in reduced.events)
        if (
            not reduced.one_minimal
            or reduced.budget_exhausted
            or hashlib.sha256(trace).hexdigest() != _REDUCED_HASH
        ):
            raise ConformanceError("reduction differs from the recorded four-event proof")
        replay = run_conformance(reduced.events, affected, config)
        drift = replay.divergence
        if (
            replay.operational_failure
            or drift is None
            or drift.event_index != 4
            or drift.path != "$.observation.trades[0].buy_order_id"
            or (drift.reference, drift.candidate) != (146075, 185199)
        ):
            raise ConformanceError("affected engine did not reproduce the expected maker mismatch")
        control = run_conformance(reduced.events, fixed, config)
        if not control.conformant or control.compared_events != len(reduced.events):
            raise ConformanceError("fixed engine did not pass the reduced trace")
        staged = scratch / "reduced.jsonl"
        staged.write_bytes(trace)
        os.link(staged, output)  # Publish complete bytes without replacing concurrent output.

    print(
        f"Reduced {len(events):,} events to {len(reduced.events)} in {reduced.runs} minimizer runs."
    )
    for i, event in enumerate(reduced.events, 1):
        print(
            f"  {i}. {event.to_dict()['side']} {event.order_type.name} {event.quantity:g} "
            f"@ {event.price:g} (order {event.order_id})"
        )
    print(
        f"Affected {replay.candidate_engine.version}: expected maker {drift.reference}, "
        f"filled {drift.candidate}."
    )
    print(f"Fixed {control.candidate_engine.version}: PASS ({control.compared_events} events).")
    print(f"Verified JSONL: {output}")
