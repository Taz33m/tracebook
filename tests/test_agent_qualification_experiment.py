import hashlib
import json
from pathlib import Path

import pytest

from experiments import agent_qualification


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path, *, status: str = "ready") -> Path:
    source = tmp_path / "source"
    source.mkdir(parents=True)
    (source / "engine.txt").write_text("price-time\n")
    gold = tmp_path / "gold.json"
    gold.write_text('{"terminal_class":"conformant"}\n')
    manifest = {
        "protocol": agent_qualification.PROTOCOL_ID,
        "protocol_sha256": _file_hash(agent_qualification.PROTOCOL_PATH),
        "prompt_sha256": _file_hash(agent_qualification.PROMPT_PATH),
        "cases": [
            {
                "id": "c1",
                "source": str(source),
                "snapshot_id": "snapshot-c1",
                "revision": "revision-c1",
                "commands": "test-engine",
                "declared_claim": "FIFO matching",
                "excluded_scope": "latency",
                "snapshot_sha256": agent_qualification._snapshot_digest(source),
                "gold_manifest": str(gold),
                "gold_sha256": _file_hash(gold),
                "dependency_cache": {"kind": "none"},
                "native_test_tool": {"kind": "none"},
                "status": status,
            }
        ],
    }
    path = tmp_path / "cases.json"
    path.write_text(json.dumps(manifest))
    return path


def _add_frozen_fixture(
    manifest_path: Path,
    fixture_root: Path,
    *,
    field_name: str,
    kind: str,
) -> None:
    source = fixture_root / "source"
    source.mkdir(parents=True)
    (source / "package.bin").write_bytes(b"frozen dependency")
    case = json.loads(manifest_path.read_text())["cases"][0]
    inventory = agent_qualification._tree_inventory(source)
    tree_hash = agent_qualification._snapshot_digest(source)
    fixture_manifest = {
        "kind": kind,
        "case_id": case["id"],
        "snapshot_sha256": case["snapshot_sha256"],
        "tree_sha256": tree_hash,
        "file_count": len(inventory),
        "total_bytes": sum(item["bytes"] for item in inventory),
        "files": inventory,
    }
    fixture_manifest_path = fixture_root / "manifest.json"
    fixture_manifest_path.write_text(json.dumps(fixture_manifest))
    payload = json.loads(manifest_path.read_text())
    payload["cases"][0][field_name] = {
        "kind": kind,
        "source": str(source),
        "manifest": str(fixture_manifest_path),
        "manifest_sha256": _file_hash(fixture_manifest_path),
        "tree_sha256": tree_hash,
    }
    manifest_path.write_text(json.dumps(payload))


def test_validate_manifest_rejects_snapshot_drift(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = agent_qualification.validate_manifest(manifest_path)
    assert manifest["protocol"] == agent_qualification.PROTOCOL_ID

    source = Path(manifest["cases"][0]["source"])
    (source / "engine.txt").write_text("changed\n")

    with pytest.raises(agent_qualification.EvaluationError, match="snapshot changed"):
        agent_qualification.validate_manifest(manifest_path)


def test_validate_manifest_rejects_gold_drift(tmp_path):
    manifest_path = _manifest(tmp_path)
    manifest = agent_qualification.validate_manifest(manifest_path)

    Path(manifest["cases"][0]["gold_manifest"]).write_text(
        '{"terminal_class":"candidate-defect"}\n'
    )

    with pytest.raises(agent_qualification.EvaluationError, match="gold manifest changed"):
        agent_qualification.validate_manifest(manifest_path)


def test_frozen_cache_copy_is_verified_and_isolated(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    _add_frozen_fixture(
        manifest_path,
        tmp_path / "fixtures" / "nuget",
        field_name="dependency_cache",
        kind="nuget-global-packages-v1",
    )
    monkeypatch.setattr(agent_qualification, "PRIVATE_ROOT", tmp_path)

    manifest = agent_qualification.validate_manifest(manifest_path)
    case = manifest["cases"][0]
    scratch = tmp_path / "scratch"
    agent_qualification._clean_agent_environment(scratch)
    prepared = agent_qualification._prepare_case_fixtures(case, scratch)

    assert (
        prepared["dependency_cache"]["initial_tree_sha256"]
        == case["dependency_cache"]["tree_sha256"]
    )
    copied = scratch / "nuget" / "package.bin"
    copied.write_bytes(b"mutated run copy")
    source = Path(case["dependency_cache"]["source"]) / "package.bin"
    assert source.read_bytes() == b"frozen dependency"
    agent_qualification._record_final_fixture_state(prepared, scratch)
    assert prepared["dependency_cache"]["mutated"] is True


def test_frozen_cache_rejects_tree_drift(tmp_path, monkeypatch):
    manifest_path = _manifest(tmp_path)
    fixture_root = tmp_path / "fixtures" / "nuget"
    _add_frozen_fixture(
        manifest_path,
        fixture_root,
        field_name="dependency_cache",
        kind="nuget-global-packages-v1",
    )
    monkeypatch.setattr(agent_qualification, "PRIVATE_ROOT", tmp_path)
    (fixture_root / "source" / "package.bin").write_bytes(b"drift")

    with pytest.raises(agent_qualification.EvaluationError, match="manifest binding"):
        agent_qualification.validate_manifest(manifest_path)


def test_render_prompt_keeps_baseline_blind():
    case = {
        "id": "c1",
        "snapshot_id": "snapshot-c1",
        "revision": "revision-c1",
        "commands": "test-engine",
        "declared_claim": "FIFO matching",
        "excluded_scope": "latency",
    }

    baseline = agent_qualification.render_prompt(case, "baseline")
    docs = agent_qualification.render_prompt(case, "docs")

    assert "tracebook" not in baseline.lower()
    assert "tracebook-conformance==0.6.0" in docs
    assert "{{" not in baseline
    assert "snapshot-c1" in baseline


def test_plan_is_seeded_and_excludes_pending_cases(tmp_path):
    manifest_path = _manifest(tmp_path)
    pending_path = _manifest(tmp_path / "pending", status="pending-gold")
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    pending_plan_path = tmp_path / "pending-plan.json"

    first = agent_qualification.create_plan(
        manifest_path, first_path, repetitions=3, conditions=("baseline",)
    )
    second = agent_qualification.create_plan(
        manifest_path, second_path, repetitions=3, conditions=("baseline",)
    )
    pending = agent_qualification.create_plan(
        pending_path,
        pending_plan_path,
        repetitions=3,
        conditions=("baseline",),
    )

    assert first["entries"] == second["entries"]
    assert len(first["entries"]) == 6
    assert {entry["agent"] for entry in first["entries"]} == {"codex", "claude"}
    assert pending["entries"] == []


def test_skill_condition_is_locked_before_baseline(tmp_path):
    manifest_path = _manifest(tmp_path)

    with pytest.raises(agent_qualification.EvaluationError, match="remain locked"):
        agent_qualification.execute_run(
            manifest_path,
            case_id="c1",
            agent="codex",
            repetition=1,
            condition="skill",
            runs_path=tmp_path / "runs",
            timeout_seconds=1,
        )


def test_claude_permissions_delegate_paths_to_fail_closed_sandbox(tmp_path):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()

    settings = agent_qualification._claude_settings(workspace, scratch)

    assert settings["permissions"]["allow"] == [
        f"Read(/{workspace}/**)",
        f"Edit(/{workspace}/**)",
        "Bash",
        "WebSearch",
        "WebFetch",
    ]
    assert f"Read(/{Path.home()}/**)" in settings["permissions"]["deny"]
    assert f"Edit(/{workspace}/.git/**)" in settings["permissions"]["deny"]
    assert str(workspace) in settings["sandbox"]["filesystem"]["allowWrite"]
    assert str(workspace / ".git") in settings["sandbox"]["filesystem"]["denyWrite"]
    assert str(Path.home()) in settings["sandbox"]["filesystem"]["denyRead"]
    for candidate in (
        Path("/usr/local/bin/dotnet"),
        Path("/usr/local/share/dotnet"),
        Path("/usr/local/opt"),
        Path("/usr/local/Cellar"),
    ):
        assert (str(candidate) in settings["sandbox"]["filesystem"]["allowRead"]) is (
            candidate.exists()
        )
    _, dotnet_launcher_dir, dotnet_root = agent_qualification._dotnet_layout()
    if dotnet_launcher_dir is None:
        assert "DOTNET_ROOT" not in settings["env"]
    else:
        assert settings["env"]["PATH"].startswith(str(dotnet_launcher_dir))
        assert settings["env"]["DOTNET_ROOT"] == str(dotnet_root)
    assert settings["env"]["NuGetAudit"] == "false"


def test_dotnet_environment_preseeds_nuget_migration_state(tmp_path):
    environment = agent_qualification._dotnet_environment(tmp_path)

    xdg_data_home = tmp_path / "xdg-data"
    assert environment["XDG_DATA_HOME"] == str(xdg_data_home)
    assert (xdg_data_home / "NuGet" / "Migrations" / "1").is_file()


def test_codex_command_uses_pure_permission_profile(tmp_path):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()

    command = agent_qualification._codex_command(workspace, scratch)
    permission_override = next(
        item for item in command if item.startswith("permissions.agent-eval=")
    )

    assert command[-1] == "-"
    assert "--sandbox" not in command
    assert "workspace-write" not in command
    assert command.index("exec") > command.index("--disable")
    assert 'default_permissions="agent-eval"' in command
    assert '":root"="deny"' in permission_override
    assert '":workspace_roots"={"."="write",".git"="read"}' in permission_override
    assert f'{scratch}"="write"' in permission_override
    for candidate in (
        Path("/usr/local/bin/dotnet"),
        Path("/usr/local/share/dotnet"),
        Path("/usr/local/opt"),
        Path("/usr/local/Cellar"),
    ):
        assert (f'"{candidate}"="read"' in permission_override) is candidate.exists()
    assert 'mode="limited"' in permission_override
    assert "allow_upstream_proxy=false" in permission_override
    assert "allow_local_binding=true" not in permission_override
    assert '"pypi.org"="allow"' in permission_override
    assert "nuget.org" not in permission_override
    assert 'shell_environment_policy.inherit="core"' in command
    assert any(item.startswith("shell_environment_policy.set=") for item in command)
    assert "--search" in command
    shell_environment = agent_qualification._codex_shell_environment(scratch)
    _, dotnet_launcher_dir, dotnet_root = agent_qualification._dotnet_layout()
    if dotnet_launcher_dir is None:
        assert "DOTNET_ROOT" not in shell_environment
    else:
        assert shell_environment["PATH"].startswith(str(dotnet_launcher_dir))
        assert shell_environment["DOTNET_ROOT"] == str(dotnet_root)


def test_clean_environment_removes_secret_shaped_names(tmp_path, monkeypatch):
    monkeypatch.setenv("AGENT_EVAL_SAFE_VALUE", "safe")
    monkeypatch.setenv("UNLISTED_PRIVATE_TOKEN", "secret")
    monkeypatch.setenv("SSH_AUTH_SOCK", "/private/tmp/agent.sock")

    environment = agent_qualification._clean_agent_environment(tmp_path)

    assert environment["AGENT_EVAL_SAFE_VALUE"] == "safe"
    assert "UNLISTED_PRIVATE_TOKEN" not in environment
    assert "SSH_AUTH_SOCK" not in environment


def test_codex_home_is_outside_agent_writable_scratch(tmp_path, monkeypatch):
    external_root = tmp_path / "run"
    scratch = external_root / "scratch"
    external_root.mkdir()
    scratch.mkdir()
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n")
    monkeypatch.setattr(agent_qualification, "CODEX_AUTH_PATH", auth)
    environment: dict[str, str] = {}

    codex_home = agent_qualification._prepare_codex_environment(external_root, scratch, environment)

    assert codex_home == external_root / "codex-home"
    assert scratch not in codex_home.parents
    assert (codex_home / "auth.json").is_symlink()
    assert Path(environment["HOME"]) == scratch / "shell-home"
    assert Path(environment["CODEX_HOME"]) == codex_home
    assert Path(environment["CODEX_SQLITE_HOME"]) == codex_home / "state"


def test_reported_model_uses_claude_transcript_and_codex_command(tmp_path):
    transcript = tmp_path / "events.jsonl"
    transcript.write_text(
        '{"type":"system","model":"claude-opus-exact"}\n' '{"type":"result","result":"done"}\n'
    )

    assert agent_qualification._reported_model(transcript, "claude") == (
        "claude-opus-exact",
        "transcript",
    )
    assert agent_qualification._reported_model(transcript, "codex") == (
        "gpt-5.6-sol",
        "command",
    )
