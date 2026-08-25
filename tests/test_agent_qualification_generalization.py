import hashlib
import json
from pathlib import Path

import pytest

from experiments import agent_qualification_generalization as generalization


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_plan_is_seeded_complete_and_excludes_v1_cases():
    manifest = {
        "cases": [
            {"id": "c4-compatible", "status": "ready"},
            {"id": "c5-boundary", "status": "ready"},
        ]
    }

    first = generalization.create_plan(manifest)
    second = generalization.create_plan(manifest)

    assert first == second
    assert first["entry_count"] == 24
    assert [entry["order"] for entry in first["entries"]] == list(range(1, 25))
    assert {entry["agent"] for entry in first["entries"]} == {"codex", "claude"}
    assert {entry["condition"] for entry in first["entries"]} == {"docs", "skill"}
    assert {entry["case_id"] for entry in first["entries"]} == {
        "c4-compatible",
        "c5-boundary",
    }
    assert all("c1" not in entry["id"] and "c3" not in entry["id"] for entry in first["entries"])


def test_pending_cases_are_not_scheduled():
    manifest = {
        "cases": [
            {"id": "ready", "status": "ready"},
            {"id": "pending", "status": "pending-gold"},
        ]
    }

    plan = generalization.create_plan(manifest)

    assert plan["entry_count"] == 12
    assert {entry["case_id"] for entry in plan["entries"]} == {"ready"}


def test_docs_and_skill_prompts_are_text_identical_before_skill_injection(tmp_path, monkeypatch):
    prompt = tmp_path / "prompt.txt"
    prompt.write_text(
        "case={{case_id}}\n"
        "snapshot={{snapshot_id}}\n"
        "revision={{revision}}\n"
        "commands={{commands}}\n"
        "claim={{declared_claim}}\n"
        "excluded={{excluded_scope}}\n"
    )
    monkeypatch.setattr(generalization, "PROMPT_PATH", prompt)
    case = {
        "id": "heldout",
        "snapshot_id": "sha256:abc",
        "revision": "pinned",
        "commands": "native-test",
        "declared_claim": "price-time",
        "excluded_scope": "latency",
    }

    docs = generalization.render_prompt(case, "docs")
    skill = generalization.render_prompt(case, "skill")

    assert docs == skill
    assert "tracebook-conformance==0.6.0" in docs
    assert "{{" not in docs


def test_snapshot_digest_binds_file_mode_and_content(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    file_path = source / "engine.txt"
    file_path.write_text("price-time\n")
    first = generalization._snapshot_digest(source)

    file_path.write_text("changed\n")
    second = generalization._snapshot_digest(source)
    file_path.write_text("price-time\n")
    file_path.chmod(0o755)
    third = generalization._snapshot_digest(source)

    assert len({first, second, third}) == 3


def test_tree_inventory_uses_same_git_exclusion_as_snapshot_digest(tmp_path):
    source = tmp_path / "source"
    git = source / ".git"
    git.mkdir(parents=True)
    (source / "engine.txt").write_text("price-time\n")
    (git / "index").write_text("metadata\n")

    assert [item["path"] for item in generalization._tree_inventory(source)] == ["engine.txt"]


def test_dependency_target_is_explicit_and_restricted():
    assert generalization._dependency_target("m2-repository") == "m2-repository"
    assert generalization._dependency_target("cargo-home") == "cargo-home"

    with pytest.raises(generalization.GeneralizationError, match="dependency_target"):
        generalization._dependency_target("../operator-cache")


def test_validate_tree_rejects_inventory_drift(tmp_path, monkeypatch):
    private = tmp_path / "private"
    source = private / "cache"
    source.mkdir(parents=True)
    (source / "dependency.bin").write_bytes(b"frozen")
    manifests = private / "manifests"
    manifests.mkdir()
    snapshot_sha256 = "candidate-tree"
    inventory = generalization._tree_inventory(source)
    tree_sha256 = generalization._snapshot_digest(source)
    tree_manifest = {
        "kind": "frozen-tree-v1",
        "case_id": "case",
        "name": "dependency-cache",
        "snapshot_sha256": snapshot_sha256,
        "tree_sha256": tree_sha256,
        "file_count": len(inventory),
        "total_bytes": sum(item["bytes"] for item in inventory),
        "files": inventory,
    }
    manifest_path = manifests / "tree.json"
    manifest_path.write_text(json.dumps(tree_manifest))
    monkeypatch.setattr(generalization, "REPOSITORY_ROOT", tmp_path)
    monkeypatch.setattr(generalization, "PRIVATE_ROOT", private)
    case = {"id": "case", "snapshot_sha256": snapshot_sha256}
    declaration = {
        "kind": "frozen-tree-v1",
        "source": "private/cache",
        "manifest": "private/manifests/tree.json",
        "manifest_sha256": _sha256(manifest_path),
        "tree_sha256": tree_sha256,
    }

    generalization._validate_tree(case, declaration, name="dependency-cache")
    (source / "dependency.bin").write_bytes(b"drift")

    with pytest.raises(generalization.GeneralizationError, match="inventory drifted"):
        generalization._validate_tree(case, declaration, name="dependency-cache")


def test_freeze_refuses_to_overwrite_existing_bindings(tmp_path, monkeypatch):
    manifest = tmp_path / "cases.json"
    manifest.write_text("{}\n")
    monkeypatch.setattr(generalization, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(generalization, "PLAN_PATH", tmp_path / "plan.json")
    monkeypatch.setattr(generalization, "FREEZE_PATH", tmp_path / "freeze.json")

    with pytest.raises(generalization.GeneralizationError, match="refusing to overwrite"):
        generalization.freeze()
