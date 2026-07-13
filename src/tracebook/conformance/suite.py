"""Hash-locked adversarial traces for repeatable matching-engine conformance."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, List, Mapping, Tuple

from ..events import MarketEvent, load_market_events
from .compare import run_conformance
from .model import ARTIFACT_SCHEMA_VERSION, ConformanceConfig, ConformanceError
from .protocol import AdapterFactory


@dataclass(frozen=True)
class SuiteCase:
    """One trace and semantic configuration from a conformance suite."""

    name: str
    tags: Tuple[str, ...]
    config: ConformanceConfig
    events_path: Path
    events_sha256: str

    def load_events(self) -> List[MarketEvent]:
        return load_market_events(self.events_path)


@dataclass(frozen=True)
class ConformanceSuite:
    """Validated, immutable collection of standard conformance cases."""

    suite_id: str
    suite_hash: str
    description: str
    root: Path
    cases: Tuple[SuiteCase, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _manifest_hash(manifest: Mapping[str, Any]) -> str:
    payload = dict(manifest)
    payload.pop("suite_hash", None)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _safe_filename(value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConformanceError("suite event path must be a non-empty filename")
    path = Path(value)
    if path.name != value or path.is_absolute() or value in {".", ".."}:
        raise ConformanceError(f"unsafe suite event path: {value!r}")
    return value


def load_conformance_suite(path: str | Path) -> ConformanceSuite:
    """Load and hash-verify a conformance suite directory or manifest."""
    source = Path(path)
    manifest_path = source / "manifest.json" if source.is_dir() else source
    if not manifest_path.is_file():
        raise ConformanceError(f"conformance suite manifest not found: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConformanceError(f"invalid suite manifest JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise ConformanceError("suite manifest must be a JSON object")
    if manifest.get("schema_version") != 1:
        raise ConformanceError("suite schema_version must be 1")
    suite_id = manifest.get("suite_id")
    suite_hash = manifest.get("suite_hash")
    description = manifest.get("description")
    cases_data = manifest.get("cases")
    if not isinstance(suite_id, str) or not suite_id:
        raise ConformanceError("suite_id must be a non-empty string")
    if not isinstance(suite_hash, str) or len(suite_hash) != 71:
        raise ConformanceError("suite_hash must be a SHA-256 digest")
    actual_suite_hash = _manifest_hash(manifest)
    if suite_hash != actual_suite_hash:
        raise ConformanceError(
            f"suite_hash mismatch: expected {suite_hash}, found {actual_suite_hash}"
        )
    if not isinstance(description, str) or not description:
        raise ConformanceError("suite description must be a non-empty string")
    if not isinstance(cases_data, list) or not cases_data:
        raise ConformanceError("suite cases must be a non-empty array")

    cases: List[SuiteCase] = []
    seen_names = set()
    seen_paths = set()
    for position, item in enumerate(cases_data, 1):
        if not isinstance(item, Mapping):
            raise ConformanceError(f"suite case {position} must be an object")
        name = item.get("name")
        if not isinstance(name, str) or not name:
            raise ConformanceError(f"suite case {position} requires a name")
        if name in seen_names:
            raise ConformanceError(f"duplicate suite case name: {name}")
        seen_names.add(name)
        filename = _safe_filename(item.get("events"))
        if filename in seen_paths:
            raise ConformanceError(f"duplicate suite event path: {filename}")
        seen_paths.add(filename)
        expected_hash = item.get("sha256")
        if (
            not isinstance(expected_hash, str)
            or not expected_hash.startswith("sha256:")
            or len(expected_hash) != 71
        ):
            raise ConformanceError(f"suite case {name} requires a SHA-256 digest")
        tags = item.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) and tag for tag in tags):
            raise ConformanceError(f"suite case {name} tags must be strings")
        events_path = manifest_path.parent / filename
        if not events_path.is_file():
            raise ConformanceError(f"suite case {name} event file not found: {filename}")
        actual_hash = _sha256(events_path)
        if actual_hash != expected_hash:
            raise ConformanceError(
                f"suite case {name} sha256 mismatch: expected {expected_hash}, "
                f"found {actual_hash}"
            )
        case = SuiteCase(
            name=name,
            tags=tuple(tags),
            config=ConformanceConfig.from_dict(item.get("config", {})),
            events_path=events_path,
            events_sha256=actual_hash,
        )
        case.load_events()
        cases.append(case)

    return ConformanceSuite(
        suite_id=suite_id,
        suite_hash=suite_hash,
        description=description,
        root=manifest_path.parent,
        cases=tuple(cases),
    )


def copy_bundled_conformance_suite(destination: str | Path) -> ConformanceSuite:
    """Copy the bundled v1 suite to a new user-owned directory."""
    target = Path(destination)
    if target.exists():
        raise ConformanceError(f"destination already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    fixture_root = resources.files("tracebook.conformance.fixtures").joinpath("v1")
    created = False
    try:
        target.mkdir()
        created = True
        with resources.as_file(fixture_root) as source:
            for fixture in source.iterdir():
                if fixture.is_file() and fixture.suffix in {".json", ".jsonl"}:
                    shutil.copy2(fixture, target / fixture.name)
    except Exception:
        if created:
            shutil.rmtree(target)
        raise
    return load_conformance_suite(target)


def run_conformance_suite(
    suite: ConformanceSuite,
    candidate_factory: AdapterFactory,
) -> dict:
    """Run every case and return one machine-readable suite artifact."""
    case_results = []
    conformant_cases = 0
    candidate_engine = None
    for case in suite.cases:
        report = run_conformance(
            case.load_events(),
            candidate_factory,
            config=case.config,
            trace_name=case.name,
        )
        if report.conformant:
            conformant_cases += 1
        if candidate_engine is None:
            candidate_engine = report.candidate_engine.to_dict()
        elif candidate_engine != report.candidate_engine.to_dict():
            raise ConformanceError("candidate engine metadata changed between suite cases")
        case_results.append(
            {
                "name": case.name,
                "tags": list(case.tags),
                "events_sha256": case.events_sha256,
                "report": report.to_dict(),
            }
        )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "tracebook.conformance.suite_report",
        "suite_id": suite.suite_id,
        "suite_hash": suite.suite_hash,
        "candidate_engine": candidate_engine,
        "case_count": len(suite.cases),
        "conformant_cases": conformant_cases,
        "conformant": conformant_cases == len(suite.cases),
        "cases": case_results,
    }
