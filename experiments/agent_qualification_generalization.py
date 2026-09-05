"""Freeze and validate the v2 held-out agent-qualification cohort.

This module intentionally does not launch provider CLIs. Execution remains
blocked until the user explicitly authorizes transmission of the new public
candidate snapshots and frozen treatment material to each provider.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import stat
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "agent-qualification-generalization-v2.md"
PROMPT_PATH = REPOSITORY_ROOT / "experiments" / "agent_qualification_generalization_prompt.txt"
PRIVATE_ROOT = REPOSITORY_ROOT / "experiments" / "private" / "agent-qualification-v2"
DRAFT_PATH = PRIVATE_ROOT / "draft.json"
MANIFEST_PATH = PRIVATE_ROOT / "cases.json"
PLAN_PATH = PRIVATE_ROOT / "plan.json"
FREEZE_PATH = PRIVATE_ROOT / "freeze.json"
MANIFESTS_ROOT = PRIVATE_ROOT / "manifests"
SKILL_PATH = PRIVATE_ROOT / "treatments" / "tracebook-qualify-matching-engine" / "SKILL.md"

PROTOCOL_ID = "agent-qualification-generalization-v2"
RANDOMIZATION_SEED = 20260730
REPETITIONS = 3
AGENTS = ("codex", "claude")
CONDITIONS = ("docs", "skill")
DEPENDENCY_TARGETS = ("m2-repository", "m2-seed", "cargo-home", "cargo-vendor")

DOCS_TREATMENT = (
    "\n\nYou may evaluate tracebook-conformance==0.6.0 using only its public "
    "README and conformance documentation. Use it only if it fits the "
    "declared contract.\n"
)


class GeneralizationError(RuntimeError):
    """Raised when a frozen v2 precondition is not satisfied."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_digest(root: Path) -> str:
    """Hash relative file names, modes, links, and bytes deterministically."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative_path = path.relative_to(root)
        if ".git" in relative_path.parts:
            continue
        relative = relative_path.as_posix()
        if path.is_symlink():
            digest.update(b"L\0")
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(os.readlink(path).encode())
            digest.update(b"\0")
        elif path.is_file():
            digest.update(b"F\0")
            digest.update(relative.encode())
            digest.update(b"\0")
            digest.update(f"{path.stat().st_mode & 0o777:o}".encode())
            digest.update(b"\0")
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\0")
    return digest.hexdigest()


def _tree_inventory(root: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative_path = path.relative_to(root)
        if ".git" in relative_path.parts:
            continue
        relative = relative_path.as_posix()
        if path.is_symlink():
            raise GeneralizationError(f"frozen tree must not contain symlink {relative!r}")
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise GeneralizationError(f"frozen tree contains unsafe entry {relative!r}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise GeneralizationError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise GeneralizationError(f"{path} must contain one JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPOSITORY_ROOT.resolve()).as_posix()


def _private_path(raw_path: str, description: str) -> Path:
    path = (REPOSITORY_ROOT / raw_path).resolve()
    try:
        path.relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise GeneralizationError(f"{description} must stay inside {PRIVATE_ROOT}") from exc
    return path


def _cases(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise GeneralizationError("case payload must contain a non-empty cases list")
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise GeneralizationError("every case must be an object")
        case_id = raw.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise GeneralizationError("every case must have a non-empty id")
        if case_id in seen:
            raise GeneralizationError(f"duplicate case id {case_id!r}")
        seen.add(case_id)
        result.append(dict(raw))
    return result


def _freeze_tree(
    *,
    case_id: str,
    name: str,
    source_path: str,
    snapshot_sha256: str,
) -> dict[str, Any]:
    source = _private_path(source_path, f"{case_id} {name} source")
    if not source.is_dir():
        raise GeneralizationError(f"{case_id} {name} source is unavailable: {source}")
    inventory = _tree_inventory(source)
    tree_sha256 = _snapshot_digest(source)
    manifest = {
        "kind": "frozen-tree-v1",
        "case_id": case_id,
        "name": name,
        "snapshot_sha256": snapshot_sha256,
        "tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "total_bytes": sum(int(item["bytes"]) for item in inventory),
        "files": inventory,
    }
    manifest_path = MANIFESTS_ROOT / f"{case_id}--{name}.json"
    _write_json(manifest_path, manifest)
    return {
        "kind": "frozen-tree-v1",
        "source": _relative(source),
        "manifest": _relative(manifest_path),
        "manifest_sha256": _sha256_file(manifest_path),
        "tree_sha256": tree_sha256,
    }


def _validate_tree(
    case: Mapping[str, Any],
    declaration: Any,
    *,
    name: str,
) -> dict[str, Any]:
    if not isinstance(declaration, dict) or declaration.get("kind") != "frozen-tree-v1":
        raise GeneralizationError(f"case {case['id']!r} {name!r} must be frozen-tree-v1")
    for field in ("source", "manifest", "manifest_sha256", "tree_sha256"):
        if not isinstance(declaration.get(field), str):
            raise GeneralizationError(f"case {case['id']!r} {name!r} lacks {field!r}")
    source = _private_path(str(declaration["source"]), f"{case['id']} {name} source")
    manifest_path = _private_path(str(declaration["manifest"]), f"{case['id']} {name} manifest")
    if not source.is_dir() or not manifest_path.is_file():
        raise GeneralizationError(f"case {case['id']!r} {name!r} tree is unavailable")
    manifest_hash = _sha256_file(manifest_path)
    if manifest_hash != declaration["manifest_sha256"]:
        raise GeneralizationError(f"case {case['id']!r} {name!r} manifest drifted")
    manifest = _load_json(manifest_path)
    inventory = _tree_inventory(source)
    tree_sha256 = _snapshot_digest(source)
    expected = {
        "kind": "frozen-tree-v1",
        "case_id": case["id"],
        "name": name,
        "snapshot_sha256": case["snapshot_sha256"],
        "tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "total_bytes": sum(int(item["bytes"]) for item in inventory),
        "files": inventory,
    }
    if manifest != expected:
        raise GeneralizationError(f"case {case['id']!r} {name!r} inventory drifted")
    if declaration["tree_sha256"] != tree_sha256:
        raise GeneralizationError(f"case {case['id']!r} {name!r} tree drifted")
    return {
        "tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "total_bytes": expected["total_bytes"],
    }


def _dependency_target(value: Any) -> str:
    if value not in DEPENDENCY_TARGETS:
        raise GeneralizationError(
            "dependency_target must be one of " + ", ".join(DEPENDENCY_TARGETS)
        )
    return str(value)


def freeze() -> dict[str, Any]:
    """Create the immutable v2 case manifest, plan, and freeze binding."""

    existing = [path for path in (MANIFEST_PATH, PLAN_PATH, FREEZE_PATH) if path.exists()]
    if existing:
        raise GeneralizationError(
            "refusing to overwrite frozen v2 files: " + ", ".join(str(path) for path in existing)
        )
    draft = _load_json(DRAFT_PATH)
    if draft.get("protocol") != PROTOCOL_ID:
        raise GeneralizationError(f"draft protocol must be {PROTOCOL_ID!r}")
    if not PROTOCOL_PATH.is_file() or not PROMPT_PATH.is_file() or not SKILL_PATH.is_file():
        raise GeneralizationError("tracked protocol, prompt, and v2 skill must exist before freeze")

    frozen_cases: list[dict[str, Any]] = []
    for raw_case in _cases(draft):
        required = (
            "source",
            "revision",
            "commands",
            "declared_claim",
            "excluded_scope",
            "origin",
            "gold_manifest",
            "dependency_source",
            "dependency_target",
            "gold_evidence",
            "status",
        )
        missing = [field for field in required if field not in raw_case]
        if missing:
            raise GeneralizationError(
                f"case {raw_case['id']!r} is missing fields: {', '.join(missing)}"
            )
        source = _private_path(str(raw_case["source"]), f"{raw_case['id']} source")
        if not source.is_dir() or (source / ".git").exists():
            raise GeneralizationError(
                f"case {raw_case['id']!r} source must be an origin-stripped directory"
            )
        snapshot_sha256 = _snapshot_digest(source)
        gold_path = _private_path(str(raw_case["gold_manifest"]), f"{raw_case['id']} gold manifest")
        if not gold_path.is_file():
            raise GeneralizationError(f"case {raw_case['id']!r} gold is unavailable")
        evidence_specs = raw_case["gold_evidence"]
        if not isinstance(evidence_specs, list):
            raise GeneralizationError(f"case {raw_case['id']!r} gold_evidence must be a list")
        evidence: list[dict[str, Any]] = []
        for item in evidence_specs:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise GeneralizationError("gold evidence entries require name and source")
            evidence.append(
                {
                    "name": item["name"],
                    **_freeze_tree(
                        case_id=str(raw_case["id"]),
                        name=str(item["name"]),
                        source_path=str(item.get("source", "")),
                        snapshot_sha256=snapshot_sha256,
                    ),
                }
            )
        dependency = _freeze_tree(
            case_id=str(raw_case["id"]),
            name="dependency-cache",
            source_path=str(raw_case["dependency_source"]),
            snapshot_sha256=snapshot_sha256,
        )
        dependency["target"] = _dependency_target(raw_case["dependency_target"])
        frozen_cases.append(
            {
                "id": raw_case["id"],
                "source": _relative(source),
                "snapshot_id": f"sha256:{snapshot_sha256}",
                "snapshot_sha256": snapshot_sha256,
                "revision": raw_case["revision"],
                "commands": raw_case["commands"],
                "declared_claim": raw_case["declared_claim"],
                "excluded_scope": raw_case["excluded_scope"],
                "origin": raw_case["origin"],
                "gold_manifest": _relative(gold_path),
                "gold_sha256": _sha256_file(gold_path),
                "dependency_cache": dependency,
                "gold_evidence": evidence,
                "status": raw_case["status"],
            }
        )

    manifest = {
        "protocol": PROTOCOL_ID,
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "prompt_sha256": _sha256_file(PROMPT_PATH),
        "skill_sha256": _sha256_file(SKILL_PATH),
        "authorization": {
            "codex": "pending-explicit-case-specific-authorization",
            "claude": "pending-explicit-case-specific-authorization",
        },
        "cases": frozen_cases,
    }
    _write_json(MANIFEST_PATH, manifest)

    plan = create_plan(manifest)
    _write_json(PLAN_PATH, plan)
    freeze_payload = {
        "protocol": PROTOCOL_ID,
        "randomization_seed": RANDOMIZATION_SEED,
        "repetitions": REPETITIONS,
        "agents": list(AGENTS),
        "conditions": list(CONDITIONS),
        "bindings": {
            "draft": {"path": _relative(DRAFT_PATH), "sha256": _sha256_file(DRAFT_PATH)},
            "manifest": {
                "path": _relative(MANIFEST_PATH),
                "sha256": _sha256_file(MANIFEST_PATH),
            },
            "plan": {"path": _relative(PLAN_PATH), "sha256": _sha256_file(PLAN_PATH)},
            "protocol": {
                "path": _relative(PROTOCOL_PATH),
                "sha256": _sha256_file(PROTOCOL_PATH),
            },
            "prompt": {"path": _relative(PROMPT_PATH), "sha256": _sha256_file(PROMPT_PATH)},
            "runner": {"path": _relative(RUNNER_PATH), "sha256": _sha256_file(RUNNER_PATH)},
            "skill": {"path": _relative(SKILL_PATH), "sha256": _sha256_file(SKILL_PATH)},
        },
        "execution_authorized": False,
        "execution_blocker": (
            "Explicit provider authorization for the new C4/C5 public snapshots "
            "and frozen treatment material has not been recorded."
        ),
    }
    _write_json(FREEZE_PATH, freeze_payload)
    validate()
    return freeze_payload


def validate_manifest() -> dict[str, Any]:
    manifest = _load_json(MANIFEST_PATH)
    if manifest.get("protocol") != PROTOCOL_ID:
        raise GeneralizationError(f"manifest protocol must be {PROTOCOL_ID!r}")
    tracked = {
        "protocol_sha256": PROTOCOL_PATH,
        "prompt_sha256": PROMPT_PATH,
        "skill_sha256": SKILL_PATH,
    }
    for field, path in tracked.items():
        if manifest.get(field) != _sha256_file(path):
            raise GeneralizationError(f"{field} drifted after freeze")
    for case in _cases(manifest):
        for field in (
            "source",
            "snapshot_id",
            "snapshot_sha256",
            "revision",
            "commands",
            "declared_claim",
            "excluded_scope",
            "origin",
            "gold_manifest",
            "gold_sha256",
            "dependency_cache",
            "gold_evidence",
            "status",
        ):
            if field not in case:
                raise GeneralizationError(f"case {case['id']!r} is missing {field!r}")
        source = _private_path(str(case["source"]), f"{case['id']} source")
        if not source.is_dir() or (source / ".git").exists():
            raise GeneralizationError(f"case {case['id']!r} source is unavailable or not stripped")
        snapshot_sha256 = _snapshot_digest(source)
        if snapshot_sha256 != case["snapshot_sha256"]:
            raise GeneralizationError(f"case {case['id']!r} snapshot drifted")
        if case["snapshot_id"] != f"sha256:{snapshot_sha256}":
            raise GeneralizationError(f"case {case['id']!r} snapshot_id is inconsistent")
        gold_path = _private_path(str(case["gold_manifest"]), f"{case['id']} gold")
        if _sha256_file(gold_path) != case["gold_sha256"]:
            raise GeneralizationError(f"case {case['id']!r} gold drifted")
        _validate_tree(case, case["dependency_cache"], name="dependency-cache")
        dependency = case["dependency_cache"]
        if not isinstance(dependency, dict):
            raise GeneralizationError(f"case {case['id']!r} dependency cache is invalid")
        _dependency_target(dependency.get("target"))
        evidence = case["gold_evidence"]
        if not isinstance(evidence, list):
            raise GeneralizationError(f"case {case['id']!r} gold_evidence must be a list")
        names: set[str] = set()
        for item in evidence:
            if not isinstance(item, dict) or not isinstance(item.get("name"), str):
                raise GeneralizationError("gold evidence entries require a name")
            if item["name"] in names:
                raise GeneralizationError(f"duplicate gold evidence name {item['name']!r}")
            names.add(item["name"])
            _validate_tree(case, item, name=item["name"])
    return manifest


def create_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    ready = [case for case in _cases(manifest) if case.get("status") == "ready"]
    entries: list[dict[str, Any]] = []
    for case in ready:
        for agent in AGENTS:
            for condition in CONDITIONS:
                for repetition in range(1, REPETITIONS + 1):
                    entries.append(
                        {
                            "id": f"{condition}__{case['id']}__{agent}__r{repetition}",
                            "case_id": case["id"],
                            "agent": agent,
                            "condition": condition,
                            "repetition": repetition,
                        }
                    )
    random.Random(RANDOMIZATION_SEED).shuffle(entries)
    for index, entry in enumerate(entries, 1):
        entry["order"] = index
    return {
        "protocol": PROTOCOL_ID,
        "randomization_seed": RANDOMIZATION_SEED,
        "repetitions": REPETITIONS,
        "agents": list(AGENTS),
        "conditions": list(CONDITIONS),
        "entry_count": len(entries),
        "entries": entries,
    }


def validate_plan(manifest: Mapping[str, Any]) -> dict[str, Any]:
    plan = _load_json(PLAN_PATH)
    expected = create_plan(manifest)
    if plan != expected:
        raise GeneralizationError("seeded v2 plan drifted")
    return plan


def validate() -> dict[str, Any]:
    manifest = validate_manifest()
    plan = validate_plan(manifest)
    freeze_payload = _load_json(FREEZE_PATH)
    if freeze_payload.get("protocol") != PROTOCOL_ID:
        raise GeneralizationError(f"freeze protocol must be {PROTOCOL_ID!r}")
    bindings = freeze_payload.get("bindings")
    if not isinstance(bindings, dict):
        raise GeneralizationError("freeze bindings must be an object")
    expected_paths = {
        "draft": DRAFT_PATH,
        "manifest": MANIFEST_PATH,
        "plan": PLAN_PATH,
        "protocol": PROTOCOL_PATH,
        "prompt": PROMPT_PATH,
        "runner": RUNNER_PATH,
        "skill": SKILL_PATH,
    }
    for name, path in expected_paths.items():
        binding = bindings.get(name)
        if not isinstance(binding, dict):
            raise GeneralizationError(f"freeze lacks {name!r} binding")
        if binding.get("path") != _relative(path) or binding.get("sha256") != _sha256_file(path):
            raise GeneralizationError(f"freeze {name!r} binding drifted")
    if freeze_payload.get("execution_authorized") is not False:
        raise GeneralizationError("frozen cohort must remain execution-blocked")
    return {
        "protocol": PROTOCOL_ID,
        "cases": len(_cases(manifest)),
        "runs": plan["entry_count"],
        "execution_authorized": False,
    }


def render_prompt(case: Mapping[str, Any], condition: str) -> str:
    if condition not in CONDITIONS:
        raise GeneralizationError(f"condition must be one of {CONDITIONS!r}")
    prompt = PROMPT_PATH.read_text()
    substitutions = {
        "case_id": case["id"],
        "snapshot_id": case["snapshot_id"],
        "revision": case["revision"],
        "commands": case["commands"],
        "declared_claim": case["declared_claim"],
        "excluded_scope": case["excluded_scope"],
    }
    for name, value in substitutions.items():
        prompt = prompt.replace("{{" + name + "}}", str(value))
    if "{{" in prompt or "}}" in prompt:
        raise GeneralizationError("unresolved prompt placeholder")
    return prompt + DOCS_TREATMENT


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "validate", "summary"))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "freeze":
            payload = freeze()
        elif args.command == "validate":
            payload = validate()
        else:
            payload = validate()
            payload["authorization_blocker"] = _load_json(FREEZE_PATH).get("execution_blocker")
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    except GeneralizationError as exc:
        print(f"Generalization experiment error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
