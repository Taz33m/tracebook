"""Frozen docs/skill treatment runner for the agent qualification evaluation.

The baseline runner is a scored, hash-pinned input and must remain unchanged.
This companion imports its isolation and evidence helpers while adding only the
pre-registered docs and native-skill treatment surfaces.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shlex
import shutil
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

if __package__:
    from . import agent_qualification as baseline
else:  # Support ``python experiments/agent_qualification_treatments.py``.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import agent_qualification as baseline  # type: ignore[no-redef]

EvaluationError = baseline.EvaluationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
PRIVATE_ROOT = baseline.PRIVATE_ROOT
TREATMENT_ROOT = PRIVATE_ROOT / "treatments"
DEFAULT_SKILL_PATH = TREATMENT_ROOT / "tracebook-qualify-matching-engine" / "SKILL.md"
DEFAULT_EXTERNAL_PINS_PATH = TREATMENT_ROOT / "EXTERNAL_PINS.json"
DEFAULT_PROTOCOL_AMENDMENT_PATH = TREATMENT_ROOT / "TREATMENT_PROTOCOL.md"
DEFAULT_AUTHORIZATION_PATH = TREATMENT_ROOT / "AUTHORIZATION.json"
DEFAULT_CLAUDE_WRAPPER_PATH = (
    TREATMENT_ROOT / "provider-wrappers" / "claude" / ".claude-plugin" / "plugin.json"
)
DEFAULT_PLAN_PATH = TREATMENT_ROOT / "master-plan.json"
DEFAULT_FREEZE_PATH = TREATMENT_ROOT / "freeze.json"
DEFAULT_SHAKEDOWN_VERDICT_PATH = TREATMENT_ROOT / "native-delivery-shakedown.json"
DEFAULT_BASELINE_PLAN_PATH = baseline.DEFAULT_PLAN_PATH
EVALUATION_ROOT = PRIVATE_ROOT / "evaluation"
DEFAULT_RUBRIC_PATH = EVALUATION_ROOT / "rubric.json"
DEFAULT_VALIDATOR_PATH = EVALUATION_ROOT / "scorecard.py"
DEFAULT_BASELINE_SCORES_PATH = EVALUATION_ROOT / "scores"
DEFAULT_RUNS_PATH = baseline.DEFAULT_RUNS_PATH
DEFAULT_TEST_PATH = REPOSITORY_ROOT / "tests" / "test_agent_qualification_treatments.py"

PROTOCOL_ID = baseline.PROTOCOL_ID
PLAN_KIND = "agent-qualification-treatment-master-plan-v1"
FREEZE_KIND = "agent-qualification-treatment-freeze-v1"
VERDICT_KIND = "agent-qualification-pre-score-run-verdict-v1"
SHAKEDOWN_KIND = "agent-qualification-native-skill-shakedown-v1"
TREATMENT_CONDITIONS = ("docs", "skill")
REPETITIONS = 3
RUNS_PER_CONDITION = 12
TOTAL_RUNS = 24
RANDOMIZATION_SEED = baseline.RANDOMIZATION_SEED
SKILL_NAME = "tracebook-qualify-matching-engine"
EXPECTED_MODELS = {
    "codex": "gpt-5.6-sol",
    "claude": "claude-opus-4-8",
}

MEASUREMENT_POLICY: dict[str, Any] = {
    "assigned_payload_is_intervention": False,
    "post_launch_evaluator_intervention": "I0",
    "elapsed_endpoint": {
        "name": "subprocess completion elapsed",
        "definition": (
            "monotonic wall-clock seconds from provider subprocess spawn " "until termination"
        ),
        "not_measured": "time-to-first-valid-evidence",
    },
    "timing_gate": {
        "minimum_matched_autonomous_safe_pass_pairs": 3,
        "both_cases_required": True,
        "skill_safe_pass_rate_must_not_be_lower": True,
        "paired_statistic": "median(skill_elapsed/docs_elapsed)",
        "pass_threshold": 0.70,
        "zero_c3_safety_violations_required": True,
        "otherwise": (
            "timing is exploratory/inestimable; skill value may pass only "
            "through at least +20 percentage points safe-pass rate"
        ),
    },
}


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _read_json_once(path: Path, description: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read {description} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvaluationError(f"{description} must contain one JSON object")
    return value, raw


def _path_label(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


def _reject_symlink_chain(path: Path, root: Path, description: str) -> None:
    lexical_path = Path(os.path.abspath(path))
    lexical_root = Path(os.path.abspath(root))
    try:
        relative = lexical_path.relative_to(lexical_root)
    except ValueError as exc:
        raise EvaluationError(f"{description} must stay inside {lexical_root}") from exc
    cursor = lexical_root
    if cursor.is_symlink():
        raise EvaluationError(f"{description} root must not be a symlink: {cursor}")
    for part in relative.parts:
        cursor /= part
        if cursor.is_symlink():
            raise EvaluationError(f"{description} must not use symlinks: {cursor}")


def _private_file(path: Path, description: str) -> Path:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    _reject_symlink_chain(candidate, PRIVATE_ROOT, description)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(PRIVATE_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"{description} must be an existing private file") from exc
    mode = resolved.stat().st_mode
    if not stat.S_ISREG(mode):
        raise EvaluationError(f"{description} must be a regular file: {resolved}")
    return resolved


def _private_output(path: Path, description: str) -> Path:
    candidate = path if path.is_absolute() else REPOSITORY_ROOT / path
    parent = candidate.parent
    _reject_symlink_chain(parent, PRIVATE_ROOT, description)
    parent.mkdir(parents=True, exist_ok=True)
    try:
        parent.resolve(strict=True).relative_to(PRIVATE_ROOT.resolve(strict=True))
    except (OSError, ValueError) as exc:
        raise EvaluationError(f"{description} must stay inside {PRIVATE_ROOT}") from exc
    if candidate.exists() or candidate.is_symlink():
        raise EvaluationError(f"refusing to overwrite {description}: {candidate}")
    return candidate


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": _path_label(path),
        "sha256": baseline._sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _tree_record(root: Path) -> dict[str, Any]:
    resolved = root.resolve(strict=True)
    inventory = baseline._tree_inventory(resolved)
    return {
        "path": _path_label(resolved),
        "sha256": baseline._snapshot_digest(resolved),
        "file_count": len(inventory),
        "files": inventory,
    }


def _resolve_record(
    record: Any,
    description: str,
    *,
    expected_path: Path | None = None,
    private: bool,
) -> tuple[Path, bytes]:
    if not isinstance(record, dict):
        raise EvaluationError(f"{description} record must be an object")
    raw_path = record.get("path")
    expected_hash = record.get("sha256")
    expected_bytes = record.get("bytes")
    if not isinstance(raw_path, str) or not isinstance(expected_hash, str):
        raise EvaluationError(f"{description} record must pin path and sha256")
    candidate = Path(raw_path)
    if private:
        path = _private_file(candidate, description)
    else:
        lexical_path = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
        _reject_symlink_chain(lexical_path, REPOSITORY_ROOT, description)
        path = lexical_path.resolve(strict=True)
        if not path.is_file():
            raise EvaluationError(f"{description} must be a regular non-symlink file")
    if expected_path is not None and path != expected_path.resolve(strict=True):
        raise EvaluationError(f"{description} path changed: expected {expected_path}, got {path}")
    raw = path.read_bytes()
    actual_hash = _sha256_bytes(raw)
    if actual_hash != expected_hash:
        raise EvaluationError(f"{description} changed: expected {expected_hash}, got {actual_hash}")
    if expected_bytes != len(raw):
        raise EvaluationError(f"{description} byte length changed")
    return path, raw


def _write_new_json(path: Path, payload: Mapping[str, Any], description: str) -> None:
    target = _private_output(path, description)
    target.write_bytes(_json_bytes(payload))


def _collect_strings(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        result: list[str] = []
        for nested in value.values():
            result.extend(_collect_strings(nested))
        return result
    if isinstance(value, list):
        result = []
        for nested in value:
            result.extend(_collect_strings(nested))
        return result
    return []


def _leakage_tokens(manifest: Mapping[str, Any]) -> set[str]:
    tokens: set[str] = set()
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return tokens
    for case in cases:
        if not isinstance(case, dict):
            continue
        selected: list[Any] = [
            case.get("id"),
            case.get("snapshot_id"),
            case.get("snapshot_sha256"),
            case.get("revision"),
            case.get("source"),
            case.get("gold_manifest"),
            case.get("gold_sha256"),
            case.get("origin"),
        ]
        for raw in _collect_strings(selected):
            normalized = raw.strip().casefold()
            if len(normalized) >= 8:
                tokens.add(normalized)
            if "://" in raw:
                parsed = urlparse(raw)
                for part in parsed.path.split("/"):
                    part = part.strip().casefold()
                    if len(part) >= 8:
                        tokens.add(part)
    return tokens


def _load_skill(skill_path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    path = _private_file(skill_path, "skill")
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise EvaluationError("skill must be UTF-8") from exc
    if text.encode("utf-8") != raw:
        raise EvaluationError("skill bytes must round-trip through UTF-8 exactly")
    if not text.startswith("---\n") or f"name: {SKILL_NAME}\n" not in text:
        raise EvaluationError(f"skill frontmatter must name {SKILL_NAME!r}")
    lowered = text.casefold()
    leaked = sorted(token for token in _leakage_tokens(manifest) if token in lowered)
    if leaked:
        raise EvaluationError(
            "skill leakage audit found held-out identifier(s): " + ", ".join(leaked)
        )
    return {
        **_file_record(path),
        "name": SKILL_NAME,
        "content": raw,
    }


def _ready_case_ids(manifest: Mapping[str, Any]) -> list[str]:
    return sorted(
        case_id
        for case_id, case in baseline._case_index(manifest).items()
        if case.get("status") == "ready"
    )


def _structural_cells(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = [
        {
            "case_id": case_id,
            "agent": agent,
            "repetition": repetition,
        }
        for case_id in _ready_case_ids(manifest)
        for agent in ("codex", "claude")
        for repetition in range(1, REPETITIONS + 1)
    ]
    random.Random(RANDOMIZATION_SEED).shuffle(cells)
    if len(cells) != RUNS_PER_CONDITION:
        raise EvaluationError(
            f"treatment protocol requires {RUNS_PER_CONDITION} cells, got {len(cells)}"
        )
    return cells


def _evidence_id(ordinal: int) -> str:
    material = f"{PROTOCOL_ID}|treatment|{RANDOMIZATION_SEED}|{ordinal}".encode()
    return "ev_" + _sha256_bytes(material)[:20]


def _master_entries(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    cells = _structural_cells(manifest)
    entries: list[dict[str, Any]] = []
    ordinal = 0
    for condition in TREATMENT_CONDITIONS:
        for cell in cells:
            ordinal += 1
            run_id = f"{condition}__{cell['case_id']}__{cell['agent']}__r{cell['repetition']}"
            entries.append(
                {
                    **cell,
                    "condition": condition,
                    "run_id": run_id,
                    "evidence_id": _evidence_id(ordinal),
                    "ordinal": ordinal,
                }
            )
    return entries


def render_prompt(case: Mapping[str, Any]) -> str:
    """Render the byte-identical user prompt used by both treatment arms."""

    return baseline.render_prompt(case, "docs")


def _binary_binding(agent: str) -> dict[str, Any]:
    binary = baseline.CODEX_BINARY if agent == "codex" else baseline.CLAUDE_BINARY
    if not binary.is_file():
        raise EvaluationError(f"{agent} binary is unavailable at {binary}")
    secret_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    clean_environment = {
        name: value
        for name, value in os.environ.items()
        if name not in baseline.SECRET_ENVIRONMENT_NAMES
        and not any(fragment in name.upper() for fragment in secret_fragments)
    }
    version = baseline._binary_version(binary, clean_environment)
    return {
        "configured_path": str(binary),
        "resolved_path": str(binary.resolve()),
        "sha256": baseline._sha256_file(binary),
        "bytes": binary.stat().st_size,
        "cli_version": version,
        "requested_model": EXPECTED_MODELS[agent],
        "required_reported_model": EXPECTED_MODELS[agent],
    }


def _load_external_pins(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = _private_file(path, "external pins")
    payload, _ = _read_json_once(resolved, "external pins")
    if payload.get("protocol") != PROTOCOL_ID:
        raise EvaluationError("external pins protocol changed")
    models = payload.get("models")
    release = payload.get("public_release")
    if not isinstance(models, dict) or not isinstance(release, dict):
        raise EvaluationError("external pins must contain models and public_release")
    for agent in ("codex", "claude"):
        expected = _binary_binding(agent)
        pinned = models.get(agent)
        if not isinstance(pinned, dict):
            raise EvaluationError(f"external pins are missing {agent}")
        checks = {
            "binary_path": expected["configured_path"],
            "binary_sha256": expected["sha256"],
            "cli_version": expected["cli_version"],
            "requested_model": expected["requested_model"],
            "required_reported_model": expected["required_reported_model"],
        }
        for field, expected_value in checks.items():
            if pinned.get(field) != expected_value:
                raise EvaluationError(
                    f"external {agent} pin {field!r} changed: "
                    f"expected {expected_value!r}, got {pinned.get(field)!r}"
                )
    git = release.get("git")
    pypi = release.get("pypi")
    if not isinstance(git, dict) or not isinstance(pypi, dict):
        raise EvaluationError("external release pins must contain git and pypi")
    required_release_values = {
        ("git", "tag"): "v0.6.0",
        ("git", "tag_commit"): "d7e20a345974f8fb57537512962e283f14ae51e3",
        ("pypi", "project"): "tracebook-conformance",
        ("pypi", "version"): "0.6.0",
    }
    sections = {"git": git, "pypi": pypi}
    for (section, field), expected_value in required_release_values.items():
        if sections[section].get(field) != expected_value:
            raise EvaluationError(
                f"public release pin {section}.{field} must be {expected_value!r}"
            )
    for section, fields in {
        "git": ("tagged_readme_sha256", "tagged_conformance_docs_sha256"),
        "pypi": (
            "wheel_filename",
            "wheel_sha256",
            "sdist_filename",
            "sdist_sha256",
        ),
    }.items():
        for field in fields:
            value = sections[section].get(field)
            if not isinstance(value, str) or not value:
                raise EvaluationError(f"public release pin {section}.{field} is missing")
    return payload, _file_record(resolved)


def _prompt_hashes(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    for case_id in _ready_case_ids(manifest):
        prompt = render_prompt(baseline._case_index(manifest)[case_id])
        prompt_hash = _sha256_bytes(prompt.encode())
        result[case_id] = {"docs": prompt_hash, "skill": prompt_hash}
    return result


def _plan_payload(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    skill: Mapping[str, Any],
    external_pins: Mapping[str, Any],
    external_pins_record: Mapping[str, Any],
    protocol_amendment_path: Path,
    authorization_path: Path,
    claude_wrapper_path: Path,
) -> dict[str, Any]:
    entries = _master_entries(manifest)
    prompt_hashes = _prompt_hashes(manifest)
    for entry in entries:
        entry["rendered_prompt_sha256"] = prompt_hashes[entry["case_id"]][entry["condition"]]
    return {
        "kind": PLAN_KIND,
        "protocol": PROTOCOL_ID,
        "created_at": baseline._utc_now(),
        "randomization_seed": RANDOMIZATION_SEED,
        "design": "docs-block-then-matched-skill-block",
        "repetitions": REPETITIONS,
        "runs_per_condition": RUNS_PER_CONDITION,
        "timeout_seconds": baseline.DEFAULT_TIMEOUT_SECONDS,
        "conditions": list(TREATMENT_CONDITIONS),
        "manifest": _file_record(manifest_path),
        "baseline_runner": _file_record(baseline.RUNNER_PATH),
        "treatment_runner": _file_record(RUNNER_PATH),
        "protocol_document": _file_record(baseline.PROTOCOL_PATH),
        "prompt_template": _file_record(baseline.PROMPT_PATH),
        "treatment_protocol": _file_record(protocol_amendment_path),
        "authorization": _file_record(authorization_path),
        "docs_treatment_sha256": _sha256_bytes(baseline.DOCS_TREATMENT.encode()),
        "skill": {key: value for key, value in skill.items() if key != "content"},
        "native_wrappers": {
            "codex": {
                "kind": "isolated-codex-home-skill-v1",
                "semantic_files": ["SKILL.md"],
            },
            "claude": {
                "kind": "isolated-claude-plugin-skill-v1",
                "manifest": _file_record(claude_wrapper_path),
                "source_tree": _tree_record(claude_wrapper_path.parents[1]),
                "semantic_files": ["SKILL.md"],
            },
        },
        "external_pins": dict(external_pins_record),
        "pinned_models": external_pins["models"],
        "public_release": external_pins["public_release"],
        "rendered_prompt_sha256": prompt_hashes,
        "measurement_policy": MEASUREMENT_POLICY,
        "entries": entries,
    }


def create_plan(
    manifest_path: Path,
    *,
    skill_path: Path,
    external_pins_path: Path,
    protocol_amendment_path: Path,
    authorization_path: Path,
    claude_wrapper_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    manifest = baseline.validate_manifest(manifest_path)
    skill = _load_skill(skill_path, manifest)
    external_pins, external_pins_record = _load_external_pins(external_pins_path)
    amendment = _private_file(protocol_amendment_path, "treatment protocol")
    authorization = _private_file(authorization_path, "authorization ledger")
    wrapper = _private_file(claude_wrapper_path, "Claude plugin wrapper")
    payload = _plan_payload(
        manifest_path,
        manifest,
        skill,
        external_pins,
        external_pins_record,
        amendment,
        authorization,
        wrapper,
    )
    _write_new_json(output_path, payload, "master plan")
    return payload


def _validate_plan_entry_shape(entries: Any) -> list[dict[str, Any]]:
    if not isinstance(entries, list):
        raise EvaluationError("master plan entries must be a list")
    if len(entries) != TOTAL_RUNS:
        raise EvaluationError(f"master plan must contain {TOTAL_RUNS} entries")
    result: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    seen_evidence: set[str] = set()
    for index, entry in enumerate(entries, start=1):
        if not isinstance(entry, dict):
            raise EvaluationError("every master plan entry must be an object")
        if entry.get("ordinal") != index:
            raise EvaluationError("master plan ordinals must be contiguous")
        run_id = entry.get("run_id")
        evidence_id = entry.get("evidence_id")
        if not isinstance(run_id, str) or run_id in seen_ids:
            raise EvaluationError("master plan run IDs must be unique strings")
        if not isinstance(evidence_id, str) or evidence_id in seen_evidence:
            raise EvaluationError("master plan evidence IDs must be unique opaque strings")
        if entry.get("condition") not in TREATMENT_CONDITIONS:
            raise EvaluationError("master plan contains an invalid condition")
        seen_ids.add(run_id)
        seen_evidence.add(evidence_id)
        result.append(entry)
    return result


def validate_plan(
    plan_path: Path,
    *,
    manifest_path: Path,
    skill_path: Path,
    external_pins_path: Path,
    protocol_amendment_path: Path,
    authorization_path: Path,
    claude_wrapper_path: Path,
) -> dict[str, Any]:
    path = _private_file(plan_path, "master plan")
    payload, raw = _read_json_once(path, "master plan")
    manifest_path = manifest_path.resolve(strict=True)
    manifest = baseline.validate_manifest(manifest_path)
    skill = _load_skill(skill_path, manifest)
    external_pins, external_record = _load_external_pins(external_pins_path)
    amendment = _private_file(protocol_amendment_path, "treatment protocol")
    authorization = _private_file(authorization_path, "authorization ledger")
    wrapper = _private_file(claude_wrapper_path, "Claude plugin wrapper")

    if payload.get("kind") != PLAN_KIND or payload.get("protocol") != PROTOCOL_ID:
        raise EvaluationError("master plan identity changed")
    if payload.get("randomization_seed") != RANDOMIZATION_SEED:
        raise EvaluationError("master plan randomization seed changed")
    if payload.get("conditions") != list(TREATMENT_CONDITIONS):
        raise EvaluationError("master plan condition order changed")
    if payload.get("repetitions") != REPETITIONS:
        raise EvaluationError("master plan repetition count changed")
    if payload.get("timeout_seconds") != baseline.DEFAULT_TIMEOUT_SECONDS:
        raise EvaluationError("master plan timeout changed")
    expected_records = {
        "manifest": _file_record(manifest_path),
        "baseline_runner": _file_record(baseline.RUNNER_PATH),
        "treatment_runner": _file_record(RUNNER_PATH),
        "protocol_document": _file_record(baseline.PROTOCOL_PATH),
        "prompt_template": _file_record(baseline.PROMPT_PATH),
        "treatment_protocol": _file_record(amendment),
        "authorization": _file_record(authorization),
        "external_pins": external_record,
    }
    for field, expected in expected_records.items():
        if payload.get(field) != expected:
            raise EvaluationError(f"master plan {field} binding changed")
    expected_skill = {key: value for key, value in skill.items() if key != "content"}
    if payload.get("skill") != expected_skill:
        raise EvaluationError("master plan skill binding changed")
    if payload.get("docs_treatment_sha256") != _sha256_bytes(baseline.DOCS_TREATMENT.encode()):
        raise EvaluationError("master plan docs treatment binding changed")
    if payload.get("pinned_models") != external_pins["models"]:
        raise EvaluationError("master plan model pins changed")
    if payload.get("public_release") != external_pins["public_release"]:
        raise EvaluationError("master plan public release pins changed")
    if payload.get("measurement_policy") != MEASUREMENT_POLICY:
        raise EvaluationError("master plan measurement policy changed")
    native_wrappers = payload.get("native_wrappers")
    if not isinstance(native_wrappers, dict):
        raise EvaluationError("master plan native wrapper bindings are missing")
    claude_wrapper = native_wrappers.get("claude")
    if not isinstance(claude_wrapper, dict) or claude_wrapper.get("manifest") != _file_record(
        wrapper
    ):
        raise EvaluationError("master plan Claude wrapper binding changed")
    if claude_wrapper.get("source_tree") != _tree_record(wrapper.parents[1]):
        raise EvaluationError("master plan Claude wrapper tree changed")
    if payload.get("rendered_prompt_sha256") != _prompt_hashes(manifest):
        raise EvaluationError("master plan rendered prompt hashes changed")

    entries = _validate_plan_entry_shape(payload.get("entries"))
    expected_entries = _master_entries(manifest)
    prompt_hashes = _prompt_hashes(manifest)
    for entry in expected_entries:
        entry["rendered_prompt_sha256"] = prompt_hashes[entry["case_id"]][entry["condition"]]
    if entries != expected_entries:
        raise EvaluationError("master plan entry order or identity changed")
    payload["_path"] = str(path)
    payload["_sha256"] = _sha256_bytes(raw)
    payload["_skill_content"] = skill["content"]
    return payload


def _run_scorecard_validator(*args: str) -> dict[str, Any] | None:
    completed = subprocess.run(
        (sys.executable, str(DEFAULT_VALIDATOR_PATH), *args),
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise EvaluationError(f"scorecard validation failed: {detail}")
    if not completed.stdout.strip().startswith("{"):
        return None
    value = json.loads(completed.stdout)
    if not isinstance(value, dict):
        raise EvaluationError("scorecard validator returned a non-object")
    return value


def _validate_baseline_evidence() -> dict[str, Any]:
    _run_scorecard_validator("check-rubric")
    plan_path = _private_file(DEFAULT_BASELINE_PLAN_PATH, "baseline plan")
    plan, _ = _read_json_once(plan_path, "baseline plan")
    manifest = baseline.validate_manifest(baseline.DEFAULT_MANIFEST_PATH)
    expected_entries = [
        {
            "run_id": (f"baseline__{cell['case_id']}__{cell['agent']}__r" f"{cell['repetition']}"),
            "condition": "baseline",
            **cell,
        }
        for cell in _structural_cells(manifest)
    ]
    expected_plan_fields = {
        "protocol": PROTOCOL_ID,
        "randomization_seed": RANDOMIZATION_SEED,
        "repetitions": REPETITIONS,
        "conditions": ["baseline"],
        "manifest_sha256": baseline._sha256_file(baseline.DEFAULT_MANIFEST_PATH),
        "runner_sha256": baseline._sha256_file(baseline.RUNNER_PATH),
    }
    for field, expected in expected_plan_fields.items():
        if plan.get(field) != expected:
            raise EvaluationError(f"baseline plan {field} binding changed")
    if plan.get("entries") != expected_entries:
        raise EvaluationError("baseline plan identity or order changed")

    paths = sorted(DEFAULT_BASELINE_SCORES_PATH.glob("baseline__*.scorecard.json"))
    if len(paths) != RUNS_PER_CONDITION:
        raise EvaluationError(
            f"expected {RUNS_PER_CONDITION} baseline scorecards, got {len(paths)}"
        )
    expected_run_ids = {entry["run_id"] for entry in expected_entries}
    records: list[dict[str, Any]] = []
    observed_run_ids: set[str] = set()
    for path in paths:
        summary = _run_scorecard_validator("validate", str(path))
        scorecard, _ = _read_json_once(path, "baseline scorecard")
        run_id = scorecard.get("run_id")
        if not isinstance(run_id, str) or run_id in observed_run_ids:
            raise EvaluationError("baseline scorecard run IDs must be unique")
        if summary is None or summary.get("protocol_valid") is not True:
            raise EvaluationError(f"baseline scorecard {run_id!r} is not protocol-valid")
        observed_run_ids.add(run_id)
        records.append(
            {
                "run_id": run_id,
                "case_id": scorecard.get("case_id"),
                "score_summary": summary,
                **_file_record(path),
            }
        )
    if observed_run_ids != expected_run_ids:
        raise EvaluationError("baseline scorecard identities do not match the baseline plan")
    return {
        "plan": _file_record(plan_path),
        "rubric": _file_record(DEFAULT_RUBRIC_PATH),
        "validator": _file_record(DEFAULT_VALIDATOR_PATH),
        "scorecards": records,
    }


def _validate_shakedown_verdict(path: Path) -> dict[str, Any]:
    resolved = _private_file(path, "native-delivery shakedown verdict")
    verdict, _ = _read_json_once(resolved, "native-delivery shakedown verdict")
    if verdict.get("kind") != SHAKEDOWN_KIND or verdict.get("protocol") != PROTOCOL_ID:
        raise EvaluationError("native-delivery shakedown identity changed")
    if verdict.get("runner") != _file_record(RUNNER_PATH):
        raise EvaluationError("native-delivery shakedown runner binding changed")
    if verdict.get("passed") is not True:
        raise EvaluationError("native-delivery shakedown did not pass")
    providers = verdict.get("providers")
    if not isinstance(providers, dict) or set(providers) != {"codex", "claude"}:
        raise EvaluationError("native-delivery shakedown must cover Codex and Claude")
    skill_hash = verdict.get("synthetic_skill_sha256")
    marker = verdict.get("marker")
    if not isinstance(skill_hash, str) or not isinstance(marker, str):
        raise EvaluationError("native-delivery shakedown is missing its canary binding")
    for agent, result in providers.items():
        if not isinstance(result, dict) or result.get("passed") is not True:
            raise EvaluationError(f"native-delivery shakedown failed for {agent}")
        if result.get("synthetic_skill_sha256") != skill_hash:
            raise EvaluationError(f"native-delivery skill bytes differed for {agent}")
        if result.get("treatment_result") != marker:
            raise EvaluationError(f"native-delivery marker was not observed for {agent}")
        if result.get("control_result") != "NO_NATIVE_DELIVERY":
            raise EvaluationError(f"native-delivery control was contaminated for {agent}")
        for field in ("control_transcript", "treatment_transcript"):
            record = result.get(field)
            _resolve_record(record, f"{agent} {field}", private=True)
    return {"verdict": verdict, "record": _file_record(resolved)}


def _freeze_payload(
    *,
    created_at: str,
    plan: Mapping[str, Any],
    shakedown: Mapping[str, Any],
) -> dict[str, Any]:
    authorization = _private_file(DEFAULT_AUTHORIZATION_PATH, "authorization ledger")
    wrapper = _private_file(DEFAULT_CLAUDE_WRAPPER_PATH, "Claude plugin wrapper")
    external_pins = _private_file(DEFAULT_EXTERNAL_PINS_PATH, "external pins")
    treatment_protocol = _private_file(DEFAULT_PROTOCOL_AMENDMENT_PATH, "treatment protocol")
    test_path = DEFAULT_TEST_PATH.resolve(strict=True)
    return {
        "kind": FREEZE_KIND,
        "protocol": PROTOCOL_ID,
        "created_at": created_at,
        "plan": _file_record(Path(str(plan["_path"]))),
        "baseline": _validate_baseline_evidence(),
        "frozen_inputs": {
            "baseline_runner": _file_record(baseline.RUNNER_PATH),
            "treatment_runner": _file_record(RUNNER_PATH),
            "treatment_runner_tests": _file_record(test_path),
            "protocol_document": _file_record(baseline.PROTOCOL_PATH),
            "prompt_template": _file_record(baseline.PROMPT_PATH),
            "case_manifest": _file_record(baseline.DEFAULT_MANIFEST_PATH),
            "treatment_protocol": _file_record(treatment_protocol),
            "authorization": _file_record(authorization),
            "external_pins": _file_record(external_pins),
            "claude_plugin_wrapper": _file_record(wrapper),
            "skill": dict(plan["skill"]),
        },
        "binary_bindings": {agent: _binary_binding(agent) for agent in ("codex", "claude")},
        "native_delivery_shakedown": dict(shakedown["record"]),
        "measurement_policy": MEASUREMENT_POLICY,
        "conditions": list(TREATMENT_CONDITIONS),
        "condition_order": "all docs verdicts, then all skill verdicts",
        "post_launch_scoring": (
            "No measured outcome grading until all 24 pre-score verdicts validate"
        ),
        "restart_policy": (
            "Quarantine and clean-restart only genuine technical interruptions; "
            "never retry a completed result, refusal, timeout, or semantic failure"
        ),
    }


def create_freeze(
    freeze_path: Path,
    *,
    plan_path: Path,
    shakedown_verdict_path: Path,
) -> dict[str, Any]:
    plan = validate_plan(
        plan_path,
        manifest_path=baseline.DEFAULT_MANIFEST_PATH,
        skill_path=DEFAULT_SKILL_PATH,
        external_pins_path=DEFAULT_EXTERNAL_PINS_PATH,
        protocol_amendment_path=DEFAULT_PROTOCOL_AMENDMENT_PATH,
        authorization_path=DEFAULT_AUTHORIZATION_PATH,
        claude_wrapper_path=DEFAULT_CLAUDE_WRAPPER_PATH,
    )
    shakedown = _validate_shakedown_verdict(shakedown_verdict_path)
    payload = _freeze_payload(
        created_at=baseline._utc_now(),
        plan=plan,
        shakedown=shakedown,
    )
    _write_new_json(freeze_path, payload, "treatment freeze")
    return payload


def validate_freeze(
    freeze_path: Path,
    *,
    plan_path: Path = DEFAULT_PLAN_PATH,
    shakedown_verdict_path: Path = DEFAULT_SHAKEDOWN_VERDICT_PATH,
) -> dict[str, Any]:
    path = _private_file(freeze_path, "treatment freeze")
    payload, raw = _read_json_once(path, "treatment freeze")
    if payload.get("kind") != FREEZE_KIND or payload.get("protocol") != PROTOCOL_ID:
        raise EvaluationError("treatment freeze identity changed")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise EvaluationError("treatment freeze is missing created_at")
    plan = validate_plan(
        plan_path,
        manifest_path=baseline.DEFAULT_MANIFEST_PATH,
        skill_path=DEFAULT_SKILL_PATH,
        external_pins_path=DEFAULT_EXTERNAL_PINS_PATH,
        protocol_amendment_path=DEFAULT_PROTOCOL_AMENDMENT_PATH,
        authorization_path=DEFAULT_AUTHORIZATION_PATH,
        claude_wrapper_path=DEFAULT_CLAUDE_WRAPPER_PATH,
    )
    shakedown = _validate_shakedown_verdict(shakedown_verdict_path)
    expected = _freeze_payload(
        created_at=created_at,
        plan=plan,
        shakedown=shakedown,
    )
    if payload != expected:
        raise EvaluationError("treatment freeze content changed")
    payload["_path"] = str(path)
    payload["_sha256"] = _sha256_bytes(raw)
    payload["_plan"] = plan
    return payload


def _claude_treatment_settings(
    workspace: Path,
    scratch: Path,
    plugin_root: Path,
) -> dict[str, Any]:
    settings = baseline._claude_settings(workspace, scratch)
    environment = settings["env"]
    environment.pop("HOME", None)
    zdotdir = scratch / "zsh-env"
    zdotdir.mkdir()
    zshenv = zdotdir / ".zshenv"
    zshenv.write_text(f"export HOME={shlex.quote(str(scratch / 'shell-home'))}\n")
    zshenv.chmod(0o444)
    environment["ZDOTDIR"] = str(zdotdir)
    environment["CLAUDE_CODE_SAFE_MODE"] = "0"
    permissions = settings["permissions"]
    plugin_rule = "//" + str(plugin_root).lstrip("/")
    zshenv_rule = "//" + str(zshenv).lstrip("/")
    permissions["deny"].append(f"Edit({plugin_rule}/**)")
    permissions["deny"].append(f"Edit({zshenv_rule})")
    sandbox = settings["sandbox"]
    sandbox["filesystem"]["denyWrite"].append(str(plugin_root))
    sandbox["filesystem"]["denyWrite"].append(str(zshenv))
    return settings


def _claude_treatment_command(
    settings_path: Path,
    plugin_root: Path,
) -> list[str]:
    return [
        str(baseline.CLAUDE_BINARY),
        "-p",
        "--no-session-persistence",
        "--no-chrome",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--settings",
        str(settings_path),
        "--plugin-dir",
        str(plugin_root),
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Glob,Grep,Edit,Write,Bash,WebSearch,WebFetch,Skill",
        "--model",
        EXPECTED_MODELS["claude"],
        "--effort",
        "high",
        "--max-budget-usd",
        "50",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def _codex_treatment_command(
    workspace: Path,
    scratch: Path,
    skills_root: Path,
) -> list[str]:
    command = baseline._codex_command(workspace, scratch)
    permission_index = next(
        index for index, value in enumerate(command) if value.startswith("permissions.agent-eval=")
    )
    permission_profile = command[permission_index]
    skill_read_entry = f'{json.dumps(str(skills_root))}="read",'
    command[permission_index] = permission_profile.replace(
        "filesystem={",
        "filesystem={" + skill_read_entry,
        1,
    )
    return command


def _prepare_native_surface(
    *,
    agent: str,
    condition: str,
    workspace: Path,
    external_run_root: Path,
    scratch: Path,
    environment: dict[str, str],
    skill_content: bytes,
    skill_name: str = SKILL_NAME,
) -> tuple[list[str], dict[str, Any], Path | None]:
    skill_hash = _sha256_bytes(skill_content)
    if condition not in TREATMENT_CONDITIONS:
        raise EvaluationError(f"unknown treatment condition {condition!r}")
    installed = condition == "skill"
    if agent == "codex":
        codex_home = baseline._prepare_codex_environment(external_run_root, scratch, environment)
        skills_root = codex_home / "skills"
        skills_root.mkdir(exist_ok=True)
        if installed:
            target = skills_root / skill_name / "SKILL.md"
            target.parent.mkdir()
            target.write_bytes(skill_content)
            target.chmod(0o444)
        surface = {
            "kind": "isolated-codex-home-skill-v1",
            "skill_installed": installed,
            "semantic_file_count": 1 if installed else 0,
            "skill_sha256": skill_hash if installed else None,
        }
        return _codex_treatment_command(workspace, scratch, skills_root), surface, None

    if agent != "claude":
        raise EvaluationError(f"unknown agent {agent!r}")
    plugin_source = DEFAULT_CLAUDE_WRAPPER_PATH.parents[1]
    plugin_root = scratch / "claude-plugin"
    shutil.copytree(plugin_source, plugin_root)
    if installed:
        target = plugin_root / "skills" / skill_name / "SKILL.md"
        target.parent.mkdir(parents=True)
        target.write_bytes(skill_content)
        target.chmod(0o444)
    surface = {
        "kind": "isolated-claude-plugin-skill-v1",
        "skill_installed": installed,
        "semantic_file_count": 1 if installed else 0,
        "skill_sha256": skill_hash if installed else None,
        "wrapper_manifest_sha256": baseline._sha256_file(
            plugin_root / ".claude-plugin" / "plugin.json"
        ),
    }
    return [], surface, plugin_root


def _audit_native_surface(
    *,
    agent: str,
    condition: str,
    external_run_root: Path,
    scratch: Path,
    skill_content: bytes,
    skill_name: str = SKILL_NAME,
) -> dict[str, Any]:
    if agent == "codex":
        path = external_run_root / "codex-home" / "skills" / skill_name / "SKILL.md"
    elif agent == "claude":
        path = scratch / "claude-plugin" / "skills" / skill_name / "SKILL.md"
    else:
        raise EvaluationError(f"unknown agent {agent!r}")
    expected = condition == "skill"
    present = path.is_file() and not path.is_symlink()
    actual_hash = baseline._sha256_file(path) if present else None
    expected_hash = _sha256_bytes(skill_content) if expected else None
    return {
        "expected_present": expected,
        "present": present,
        "path": str(path),
        "expected_sha256": expected_hash,
        "final_sha256": actual_hash,
        "mutated": present != expected or actual_hash != expected_hash,
    }


def _transcript_terminal_state(path: Path, agent: str) -> str:
    completed = False
    failed = False
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    failed = True
                    continue
                if not isinstance(event, dict):
                    continue
                if agent == "codex":
                    if event.get("type") == "turn.completed":
                        completed = True
                    if event.get("type") in {"turn.failed", "error"}:
                        failed = True
                else:
                    if event.get("type") == "result":
                        terminal_reason = event.get("terminal_reason")
                        subtype = event.get("subtype")
                        if terminal_reason == "completed" and subtype == "success":
                            completed = True
                        else:
                            failed = True
    except (OSError, UnicodeDecodeError):
        return "unreadable"
    if completed and not failed:
        return "completed"
    if completed:
        return "completed-with-error-events"
    if failed:
        return "failed"
    return "incomplete"


def _claude_catalog_audit(
    path: Path,
    *,
    condition: str,
    skill_name: str = SKILL_NAME,
) -> dict[str, Any]:
    init: dict[str, Any] | None = None
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if (
                isinstance(event, dict)
                and event.get("type") == "system"
                and event.get("subtype") == "init"
            ):
                init = event
                break
    if init is None:
        return {"valid": False, "reason": "missing system init event"}
    plugins = init.get("plugins")
    plugin_names = (
        [item.get("name") for item in plugins if isinstance(item, dict)]
        if isinstance(plugins, list)
        else []
    )
    raw_skills = init.get("skills")
    skills: list[Any] = raw_skills if isinstance(raw_skills, list) else []
    measured_name = f"agent-qualification-treatment:{skill_name}"
    measured_present = measured_name in skills
    expected_present = condition == "skill"
    errors: list[str] = []
    if plugin_names != ["agent-qualification-treatment"]:
        errors.append("unexpected plugin catalog")
    if init.get("mcp_servers") != []:
        errors.append("MCP catalog was not empty")
    if measured_present is not expected_present:
        errors.append("measured skill catalog presence changed")
    required_tools = {
        "Bash",
        "Edit",
        "Glob",
        "Grep",
        "Read",
        "Skill",
        "WebFetch",
        "WebSearch",
        "Write",
    }
    if set(init.get("tools", [])) != required_tools:
        errors.append("tool catalog changed")
    return {
        "valid": not errors,
        "errors": errors,
        "plugin_names": plugin_names,
        "mcp_servers": init.get("mcp_servers"),
        "measured_skill": measured_name,
        "measured_skill_expected": expected_present,
        "measured_skill_present": measured_present,
        "model": init.get("model"),
    }


def _entry_by_run_id(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise EvaluationError("validated plan lost its entries")
    matches = [entry for entry in entries if entry.get("run_id") == run_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise EvaluationError(f"run {run_id!r} is not one frozen plan entry")
    return matches[0]


def _run_root(run_id: str) -> Path:
    return DEFAULT_RUNS_PATH.resolve() / run_id


def _validate_prior_verdicts(
    plan: Mapping[str, Any],
    freeze: Mapping[str, Any],
    ordinal: int,
) -> None:
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise EvaluationError("validated plan lost its entries")
    for entry in entries[: ordinal - 1]:
        validate_run(str(entry["run_id"]), plan=plan, freeze=freeze)


def _expected_metadata_bindings(
    entry: Mapping[str, Any],
    *,
    freeze: Mapping[str, Any],
    plan: Mapping[str, Any],
    case: Mapping[str, Any],
) -> dict[str, Any]:
    agent = str(entry["agent"])
    binary_binding = freeze["binary_bindings"][agent]
    return {
        "protocol": PROTOCOL_ID,
        "run_id": entry["run_id"],
        "evidence_id": entry["evidence_id"],
        "ordinal": entry["ordinal"],
        "condition": entry["condition"],
        "case_id": entry["case_id"],
        "agent": agent,
        "repetition": entry["repetition"],
        "manifest_sha256": baseline._sha256_file(baseline.DEFAULT_MANIFEST_PATH),
        "gold_manifest_sha256": str(case["gold_sha256"]),
        "baseline_runner_sha256": baseline._sha256_file(baseline.RUNNER_PATH),
        "treatment_runner_sha256": baseline._sha256_file(RUNNER_PATH),
        "treatment_freeze_sha256": freeze["_sha256"],
        "master_plan_sha256": plan["_sha256"],
        "protocol_sha256": baseline._sha256_file(baseline.PROTOCOL_PATH),
        "prompt_template_sha256": baseline._sha256_file(baseline.PROMPT_PATH),
        "docs_treatment_sha256": _sha256_bytes(baseline.DOCS_TREATMENT.encode()),
        "rendered_prompt_sha256": entry["rendered_prompt_sha256"],
        "snapshot_sha256": str(case["snapshot_sha256"]),
        "requested_model": EXPECTED_MODELS[agent],
        "binary_sha256": binary_binding["sha256"],
        "cli_version": binary_binding["cli_version"],
        "timeout_seconds": baseline.DEFAULT_TIMEOUT_SECONDS,
        "assigned_payload_is_intervention": False,
        "post_launch_intervention_class": "I0",
    }


def _write_pre_score_verdict(
    run_root: Path,
    *,
    valid: bool,
    errors: Sequence[str],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    verdict = {
        "kind": VERDICT_KIND,
        "protocol": PROTOCOL_ID,
        "run_id": metadata.get("run_id"),
        "valid": valid,
        "errors": list(errors),
        "terminal_state": metadata.get("terminal_state"),
        "timed_out": metadata.get("timed_out"),
        "exit_code": metadata.get("exit_code"),
        "metadata_sha256": baseline._sha256_file(run_root / "metadata.json"),
        "transcript_sha256": metadata.get("transcript_sha256"),
        "created_at": baseline._utc_now(),
    }
    path = run_root / "pre-score-verdict.json"
    if path.exists():
        raise EvaluationError(f"refusing to overwrite run verdict {path}")
    path.write_bytes(_json_bytes(verdict))
    return verdict


def validate_run(
    run_id: str,
    *,
    plan: Mapping[str, Any] | None = None,
    freeze: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if freeze is None:
        freeze = validate_freeze(DEFAULT_FREEZE_PATH)
    if plan is None:
        raw_plan = freeze.get("_plan")
        if not isinstance(raw_plan, dict):
            raise EvaluationError("validated freeze lost its master plan")
        plan = raw_plan
    entry = _entry_by_run_id(plan, run_id)
    run_root = _run_root(run_id)
    metadata_path = _private_file(run_root / "metadata.json", "run metadata")
    metadata, _ = _read_json_once(metadata_path, "run metadata")
    manifest = baseline.validate_manifest(baseline.DEFAULT_MANIFEST_PATH)
    case = baseline._case_index(manifest)[str(entry["case_id"])]
    errors: list[str] = []
    for field, expected in _expected_metadata_bindings(
        entry, freeze=freeze, plan=plan, case=case
    ).items():
        if metadata.get(field) != expected:
            errors.append(f"metadata {field} binding changed")
    agent = str(entry["agent"])
    transcript_path = run_root / f"{agent}.jsonl"
    stderr_path = run_root / f"{agent}.stderr"
    for artifact in (transcript_path, stderr_path, run_root / "prompt.txt"):
        if not artifact.is_file() or artifact.is_symlink():
            errors.append(f"missing required run artifact {artifact.name}")
    if transcript_path.is_file():
        actual_transcript_hash = baseline._sha256_file(transcript_path)
        if metadata.get("transcript_sha256") != actual_transcript_hash:
            errors.append("transcript hash changed")
    if stderr_path.is_file():
        actual_stderr_hash = baseline._sha256_file(stderr_path)
        if metadata.get("stderr_sha256") != actual_stderr_hash:
            errors.append("stderr hash changed")
    prompt_path = run_root / "prompt.txt"
    if prompt_path.is_file() and baseline._sha256_file(prompt_path) != entry.get(
        "rendered_prompt_sha256"
    ):
        errors.append("rendered prompt artifact changed")
    if metadata.get("model_identifier") != EXPECTED_MODELS[agent]:
        errors.append("reported model identifier changed")
    expected_skill = entry["condition"] == "skill"
    if metadata.get("skill_injected") is not expected_skill:
        errors.append("skill injection flag changed")
    if metadata.get("frozen_skill_sha256") != plan["skill"]["sha256"]:
        errors.append("frozen skill metadata binding changed")
    surface = metadata.get("native_surface")
    if not isinstance(surface, dict):
        errors.append("native skill surface audit is missing")
    else:
        expected_kind = (
            "isolated-codex-home-skill-v1"
            if agent == "codex"
            else "isolated-claude-plugin-skill-v1"
        )
        if surface.get("kind") != expected_kind:
            errors.append("native skill surface kind changed")
        if surface.get("skill_installed") is not expected_skill:
            errors.append("native skill surface condition changed")
        if surface.get("semantic_file_count") != (1 if expected_skill else 0):
            errors.append("native skill semantic file count changed")
        expected_skill_hash = plan["skill"]["sha256"] if expected_skill else None
        if surface.get("skill_sha256") != expected_skill_hash:
            errors.append("native skill surface hash changed")
        final_audit = surface.get("final_audit")
        if not isinstance(final_audit, dict) or final_audit.get("mutated") is not False:
            errors.append("native skill surface final audit failed")
        if agent == "claude":
            catalog_audit = surface.get("provider_catalog_audit")
            if not isinstance(catalog_audit, dict) or catalog_audit.get("valid") is not True:
                errors.append("Claude provider catalog audit failed")
    terminal_state = metadata.get("terminal_state")
    timed_out = metadata.get("timed_out")
    if timed_out is not True and terminal_state != "completed":
        errors.append("provider subprocess did not produce a completed terminal event")
    fixtures = metadata.get("fixtures")
    if not isinstance(fixtures, dict):
        errors.append("fixture audit is missing")
    else:
        for name, fixture in fixtures.items():
            if not isinstance(fixture, dict):
                errors.append(f"fixture {name} audit is invalid")
                continue
            if fixture.get("kind") == "none":
                continue
            if fixture.get("mutated") is not False:
                errors.append(f"fixture {name} mutated")
            if fixture.get("final_tree_sha256") != fixture.get("initial_tree_sha256"):
                errors.append(f"fixture {name} final hash changed")
    verdict_path = run_root / "pre-score-verdict.json"
    if not verdict_path.is_file():
        raise EvaluationError(f"run {run_id!r} is missing its pre-score verdict")
    verdict, _ = _read_json_once(verdict_path, "pre-score verdict")
    if verdict.get("kind") != VERDICT_KIND or verdict.get("run_id") != run_id:
        errors.append("pre-score verdict identity changed")
    if verdict.get("metadata_sha256") != baseline._sha256_file(metadata_path):
        errors.append("pre-score verdict metadata hash changed")
    if verdict.get("transcript_sha256") != metadata.get("transcript_sha256"):
        errors.append("pre-score verdict transcript hash changed")
    if verdict.get("valid") is not True or verdict.get("errors") != []:
        errors.append("pre-score verdict did not record a valid run")
    if errors:
        raise EvaluationError("; ".join(errors))
    return verdict


def execute_run(
    run_id: str,
    *,
    timeout_seconds: int = baseline.DEFAULT_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if timeout_seconds != baseline.DEFAULT_TIMEOUT_SECONDS:
        raise EvaluationError(
            "official treatment timeout must remain " f"{baseline.DEFAULT_TIMEOUT_SECONDS} seconds"
        )
    freeze = validate_freeze(DEFAULT_FREEZE_PATH)
    raw_plan = freeze.get("_plan")
    if not isinstance(raw_plan, dict):
        raise EvaluationError("validated freeze lost its master plan")
    plan = raw_plan
    entry = _entry_by_run_id(plan, run_id)
    _validate_prior_verdicts(plan, freeze, int(entry["ordinal"]))

    manifest = baseline.validate_manifest(baseline.DEFAULT_MANIFEST_PATH)
    case = baseline._case_index(manifest)[str(entry["case_id"])]
    if case.get("status") != "ready":
        raise EvaluationError(f"case {entry['case_id']!r} is not ready")
    run_root = _run_root(run_id)
    if run_root.exists() or run_root.is_symlink():
        raise EvaluationError(f"refusing to overwrite existing run {run_root}")

    agent = str(entry["agent"])
    condition = str(entry["condition"])
    source = (REPOSITORY_ROOT / str(case["source"])).resolve(strict=True)
    work_root = baseline.CODEX_WORK_ROOT if agent == "codex" else baseline.CLAUDE_WORK_ROOT
    external_run_root = work_root / run_id
    if external_run_root.exists() or external_run_root.is_symlink():
        raise EvaluationError(f"refusing to reuse external workspace {external_run_root}")
    workspace = external_run_root / "workspace"
    scratch = external_run_root / "scratch"
    transcript_path = run_root / f"{agent}.jsonl"
    stderr_path = run_root / f"{agent}.stderr"

    run_root.mkdir(parents=True)
    scratch.mkdir(parents=True)
    baseline._initialize_snapshot(source, workspace)
    prompt = render_prompt(case)
    if _sha256_bytes(prompt.encode()) != entry["rendered_prompt_sha256"]:
        raise EvaluationError("rendered treatment prompt changed before launch")
    (run_root / "prompt.txt").write_text(prompt)
    environment = baseline._clean_agent_environment(scratch)
    fixtures = baseline._prepare_case_fixtures(case, scratch)
    skill_content = plan.get("_skill_content")
    if not isinstance(skill_content, bytes):
        raise EvaluationError("validated plan lost the frozen skill bytes")
    command, native_surface, plugin_root = _prepare_native_surface(
        agent=agent,
        condition=condition,
        workspace=workspace,
        external_run_root=external_run_root,
        scratch=scratch,
        environment=environment,
        skill_content=skill_content,
    )
    settings_path: Path | None = None
    if agent == "claude":
        if plugin_root is None:
            raise EvaluationError("Claude native plugin root was not prepared")
        settings_path = run_root / "claude-settings.json"
        baseline._write_json(
            settings_path,
            _claude_treatment_settings(workspace, scratch, plugin_root),
        )
        command = _claude_treatment_command(settings_path, plugin_root)

    binary = baseline.CODEX_BINARY if agent == "codex" else baseline.CLAUDE_BINARY
    cli_version = baseline._binary_version(binary, environment)
    started_at = baseline._utc_now()
    bindings = _expected_metadata_bindings(entry, freeze=freeze, plan=plan, case=case)
    metadata: dict[str, Any] = {
        **bindings,
        "started_at": started_at,
        "timeout_seconds": timeout_seconds,
        "command": command,
        "prompt_transport": "stdin",
        "cli_version": cli_version,
        "binary_sha256": baseline._sha256_file(binary),
        "removed_environment_names": sorted(baseline.SECRET_ENVIRONMENT_NAMES),
        "fixtures": fixtures,
        "native_surface": native_surface,
        "skill_injected": condition == "skill",
        "frozen_skill_sha256": plan["skill"]["sha256"],
        "treatment_payload_transport": "native local skill mechanism",
    }
    baseline._write_json(run_root / "metadata.json", metadata)

    exit_code, timed_out, elapsed_seconds = baseline._run_subprocess(
        command,
        cwd=workspace,
        environment=environment,
        prompt_stdin=prompt,
        stdout_path=transcript_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout_seconds,
    )
    native_surface["final_audit"] = _audit_native_surface(
        agent=agent,
        condition=condition,
        external_run_root=external_run_root,
        scratch=scratch,
        skill_content=skill_content,
    )
    baseline._write_workspace_evidence(workspace, run_root)
    baseline._record_final_fixture_state(fixtures, scratch)
    shutil.copytree(
        workspace,
        run_root / "workspace-final",
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", "bin", "obj", "target", "node_modules", ".eval-cache"
        ),
    )
    terminal_state = _transcript_terminal_state(transcript_path, agent)
    if agent == "claude":
        native_surface["provider_catalog_audit"] = _claude_catalog_audit(
            transcript_path,
            condition=condition,
        )
    metadata.update(
        {
            "completed_at": baseline._utc_now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed_seconds,
            "terminal_state": terminal_state,
            "transcript_sha256": baseline._sha256_file(transcript_path),
            "stderr_sha256": baseline._sha256_file(stderr_path),
            "fixtures": fixtures,
        }
    )
    model_identifier, model_identifier_source = baseline._reported_model(transcript_path, agent)
    metadata["model_identifier"] = model_identifier
    metadata["model_identifier_source"] = model_identifier_source
    baseline._write_json(run_root / "metadata.json", metadata)

    pre_errors: list[str] = []
    if model_identifier != EXPECTED_MODELS[agent]:
        pre_errors.append(f"reported model {model_identifier!r} != {EXPECTED_MODELS[agent]!r}")
    if timed_out is not True and terminal_state != "completed":
        pre_errors.append(f"provider terminal state was {terminal_state!r}")
    if native_surface["final_audit"]["mutated"] is not False:
        pre_errors.append("native skill surface mutated")
    if agent == "claude" and native_surface["provider_catalog_audit"]["valid"] is not True:
        pre_errors.append("Claude provider catalog isolation failed")
    for name, fixture in fixtures.items():
        if fixture.get("kind") != "none" and fixture.get("mutated") is not False:
            pre_errors.append(f"fixture {name} mutated")
    verdict = _write_pre_score_verdict(
        run_root,
        valid=not pre_errors,
        errors=pre_errors,
        metadata=metadata,
    )
    if pre_errors:
        raise EvaluationError(f"run {run_id!r} is technically invalid: " + "; ".join(pre_errors))
    validate_run(run_id, plan=plan, freeze=freeze)
    shutil.rmtree(external_run_root)
    return {"metadata": metadata, "verdict": verdict}


def _transcript_result_text(path: Path, agent: str) -> str:
    result = ""
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            event = json.loads(line)
            if not isinstance(event, dict):
                continue
            if agent == "codex" and event.get("type") == "item.completed":
                item = event.get("item")
                if isinstance(item, dict) and item.get("type") == "agent_message":
                    text = item.get("text")
                    if isinstance(text, str):
                        result = text
            if agent == "claude" and event.get("type") == "result":
                text = event.get("result")
                if isinstance(text, str):
                    result = text
    return result.strip()


def run_native_delivery_shakedown(
    *,
    output_path: Path = DEFAULT_SHAKEDOWN_VERDICT_PATH,
) -> dict[str, Any]:
    prior_root = PRIVATE_ROOT / "shakedowns" / "native-skill-delivery-20260729"
    skill_source = _private_file(
        prior_root / "skill-source" / "SKILL.md", "synthetic shakedown skill"
    )
    skill_content = skill_source.read_bytes()
    skill_hash = _sha256_bytes(skill_content)
    marker = "NATIVE_SKILL_PROBE_20260729_8C41D6"
    if marker.encode() not in skill_content:
        raise EvaluationError("synthetic shakedown skill lost its marker")
    verified_root = PRIVATE_ROOT / "shakedowns" / "native-skill-delivery-verified-20260729"
    if verified_root.exists() or verified_root.is_symlink():
        raise EvaluationError(f"refusing to reuse shakedown root {verified_root}")
    verified_root.mkdir(parents=True)
    prompt = (
        "Run the native-delivery-probe shakedown now. If its native skill is "
        "available, use it. If unavailable, reply exactly "
        "NO_NATIVE_DELIVERY. Do not search for skill files or inspect any path "
        "not supplied by the native skill catalog."
    )
    providers: dict[str, Any] = {}
    for agent in ("codex", "claude"):
        agent_result: dict[str, Any] = {
            "synthetic_skill_sha256": skill_hash,
        }
        for condition, label in (("docs", "control"), ("skill", "treatment")):
            root = verified_root / f"{agent}-{label}"
            workspace = root / "workspace"
            scratch = root / "scratch"
            workspace.mkdir(parents=True)
            scratch.mkdir(parents=True)
            (workspace / "README.md").write_text("Synthetic native skill delivery probe.\n")
            environment = baseline._clean_agent_environment(scratch)
            command, surface, plugin_root = _prepare_native_surface(
                agent=agent,
                condition=condition,
                workspace=workspace,
                external_run_root=root,
                scratch=scratch,
                environment=environment,
                skill_content=skill_content,
                skill_name="native-delivery-probe",
            )
            if agent == "claude":
                if plugin_root is None:
                    raise EvaluationError("Claude shakedown plugin was not prepared")
                settings_path = root / "claude-settings.json"
                baseline._write_json(
                    settings_path,
                    _claude_treatment_settings(workspace, scratch, plugin_root),
                )
                command = _claude_treatment_command(settings_path, plugin_root)
            transcript = root / f"{agent}.jsonl"
            stderr = root / f"{agent}.stderr"
            exit_code, timed_out, elapsed = baseline._run_subprocess(
                command,
                cwd=workspace,
                environment=environment,
                prompt_stdin=prompt,
                stdout_path=transcript,
                stderr_path=stderr,
                timeout_seconds=600,
            )
            surface["final_audit"] = _audit_native_surface(
                agent=agent,
                condition=condition,
                external_run_root=root,
                scratch=scratch,
                skill_content=skill_content,
                skill_name="native-delivery-probe",
            )
            if agent == "claude":
                surface["provider_catalog_audit"] = _claude_catalog_audit(
                    transcript,
                    condition=condition,
                    skill_name="native-delivery-probe",
                )
            terminal_state = _transcript_terminal_state(transcript, agent)
            result = _transcript_result_text(transcript, agent)
            model, model_source = baseline._reported_model(transcript, agent)
            expected_result = marker if condition == "skill" else "NO_NATIVE_DELIVERY"
            if exit_code != 0 or timed_out or terminal_state != "completed":
                raise EvaluationError(f"{agent} {label} shakedown did not complete cleanly")
            if surface["final_audit"]["mutated"] is not False:
                raise EvaluationError(f"{agent} {label} native skill surface mutated")
            if agent == "claude" and surface["provider_catalog_audit"]["valid"] is not True:
                raise EvaluationError(f"{agent} {label} provider catalog was not isolated")
            if result != expected_result:
                raise EvaluationError(
                    f"{agent} {label} shakedown returned {result!r}, "
                    f"expected {expected_result!r}"
                )
            if model != EXPECTED_MODELS[agent]:
                raise EvaluationError(f"{agent} shakedown model changed: {model!r}")
            agent_result[f"{label}_result"] = result
            agent_result[f"{label}_transcript"] = _file_record(transcript)
            agent_result[f"{label}_stderr"] = _file_record(stderr)
            agent_result[f"{label}_surface"] = surface
            agent_result[f"{label}_elapsed_seconds"] = elapsed
            agent_result[f"{label}_model"] = model
            agent_result[f"{label}_model_source"] = model_source
        agent_result["passed"] = True
        providers[agent] = agent_result
    verdict = {
        "kind": SHAKEDOWN_KIND,
        "protocol": PROTOCOL_ID,
        "created_at": baseline._utc_now(),
        "passed": True,
        "runner": _file_record(RUNNER_PATH),
        "marker": marker,
        "synthetic_skill": _file_record(skill_source),
        "synthetic_skill_sha256": skill_hash,
        "prompt_sha256": _sha256_bytes(prompt.encode()),
        "providers": providers,
    }
    _write_new_json(output_path, verdict, "native-delivery shakedown verdict")
    return verdict


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("shakedown", help="verify native skill delivery on a synthetic case")

    plan = subparsers.add_parser("plan", help="create the immutable 24-run plan")
    plan.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)

    freeze = subparsers.add_parser("freeze", help="freeze all treatment inputs")
    freeze.add_argument("--output", type=Path, default=DEFAULT_FREEZE_PATH)

    subparsers.add_parser("validate", help="validate the complete treatment freeze")

    run = subparsers.add_parser("run", help="execute the next frozen plan entry")
    run.add_argument("--run-id", required=True)
    run.add_argument("--timeout-seconds", type=int, default=baseline.DEFAULT_TIMEOUT_SECONDS)

    validate_run_parser = subparsers.add_parser(
        "validate-run", help="validate one completed pre-score run"
    )
    validate_run_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "shakedown":
            payload = run_native_delivery_shakedown()
        elif args.command == "plan":
            payload = create_plan(
                baseline.DEFAULT_MANIFEST_PATH,
                skill_path=DEFAULT_SKILL_PATH,
                external_pins_path=DEFAULT_EXTERNAL_PINS_PATH,
                protocol_amendment_path=DEFAULT_PROTOCOL_AMENDMENT_PATH,
                authorization_path=DEFAULT_AUTHORIZATION_PATH,
                claude_wrapper_path=DEFAULT_CLAUDE_WRAPPER_PATH,
                output_path=args.output,
            )
        elif args.command == "freeze":
            payload = create_freeze(
                args.output,
                plan_path=DEFAULT_PLAN_PATH,
                shakedown_verdict_path=DEFAULT_SHAKEDOWN_VERDICT_PATH,
            )
        elif args.command == "validate":
            payload = validate_freeze(DEFAULT_FREEZE_PATH)
        elif args.command == "run":
            payload = execute_run(args.run_id, timeout_seconds=args.timeout_seconds)
        elif args.command == "validate-run":
            payload = validate_run(args.run_id)
        else:  # pragma: no cover - argparse enforces the command.
            raise EvaluationError(f"unknown command {args.command!r}")
    except (EvaluationError, OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    printable = {
        key: value
        for key, value in payload.items()
        if not key.startswith("_") and not isinstance(value, bytes)
    }
    print(json.dumps(printable, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
