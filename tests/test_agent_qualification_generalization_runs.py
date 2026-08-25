import hashlib
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from experiments import agent_qualification_generalization_runs as execution


def test_plan_lookup_uses_exact_frozen_id():
    plan = {"entries": [{"id": "docs__case__codex__r1", "order": 1}]}

    assert execution._plan_entry(plan, "docs__case__codex__r1")["order"] == 1
    with pytest.raises(execution.ExecutionError, match="not one frozen plan entry"):
        execution._plan_entry(plan, "missing")


def test_authorization_is_provider_and_case_specific(tmp_path, monkeypatch):
    auth_root = tmp_path / "authorizations"
    statement = "I approve C4 and C5 for Codex."
    payload = {
        "kind": execution.AUTHORIZATION_KIND,
        "protocol": execution.PROTOCOL_ID,
        "provider": "codex",
        "case_ids": ["c4-compatible", "c5-boundary"],
        "materials": [
            "public repository snapshots",
            "frozen prompt",
            "docs-assisted treatment",
            "skill-assisted treatment",
        ],
        "scope": "all-preregistered-runs-and-genuine-technical-restarts",
        "user_statement": statement,
        "statement_sha256": hashlib.sha256(statement.encode()).hexdigest(),
        "recorded_at": "2026-08-01T00:00:00Z",
    }
    auth_root.mkdir()
    (auth_root / "codex.json").write_text(json.dumps(payload))
    monkeypatch.setattr(execution, "AUTHORIZATION_ROOT", auth_root)

    assert execution.validate_authorization("codex")["provider"] == "codex"
    with pytest.raises(execution.ExecutionError, match="cannot read claude authorization"):
        execution.validate_authorization("claude")


def test_authorization_rejects_statement_drift(tmp_path, monkeypatch):
    auth_root = tmp_path / "authorizations"
    auth_root.mkdir()
    payload = {
        "kind": execution.AUTHORIZATION_KIND,
        "protocol": execution.PROTOCOL_ID,
        "provider": "codex",
        "case_ids": ["c4-compatible", "c5-boundary"],
        "materials": [
            "public repository snapshots",
            "frozen prompt",
            "docs-assisted treatment",
            "skill-assisted treatment",
        ],
        "scope": "all-preregistered-runs-and-genuine-technical-restarts",
        "user_statement": "changed",
        "statement_sha256": "not-the-hash",
        "recorded_at": "2026-08-01T00:00:00Z",
    }
    (auth_root / "codex.json").write_text(json.dumps(payload))
    monkeypatch.setattr(execution, "AUTHORIZATION_ROOT", auth_root)

    with pytest.raises(execution.ExecutionError, match="statement binding drifted"):
        execution.validate_authorization("codex")


def test_copy_fixture_starts_from_frozen_hash(tmp_path, monkeypatch):
    repository = tmp_path / "repo"
    source = repository / "private" / "cache"
    source.mkdir(parents=True)
    (source / "artifact").write_text("frozen")
    tree_hash = execution.helpers._snapshot_digest(source)
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(execution, "REPOSITORY_ROOT", repository)
    case = {
        "id": "c5-boundary",
        "dependency_cache": {
            "source": "private/cache",
            "tree_sha256": tree_hash,
            "manifest_sha256": "manifest",
            "target": "cargo-home",
        },
    }

    fixture = execution._copy_fixture(case, scratch)

    assert fixture["initial_tree_sha256"] == tree_hash
    assert Path(fixture["target"]).name == "cargo-home"
    assert execution.helpers._snapshot_digest(Path(fixture["target"])) == tree_hash

    (Path(fixture["target"]) / "artifact").write_text("mutated")
    execution._record_fixture_final(fixture)
    assert fixture["mutated"] is True
    assert fixture["final_tree_sha256"] != fixture["initial_tree_sha256"]


def test_native_surface_has_no_skill_in_docs_condition(tmp_path, monkeypatch):
    root = tmp_path / "run"
    workspace = root / "workspace"
    scratch = root / "scratch"
    workspace.mkdir(parents=True)
    scratch.mkdir()
    monkeypatch.setattr(
        execution,
        "_configured_delivery",
        lambda: nullcontext(),
    )
    monkeypatch.setattr(
        execution.delivery,
        "_prepare_native_surface",
        lambda **kwargs: (
            ["provider", "permissions.agent-eval={filesystem={}}"],
            {
                "skill_installed": kwargs["condition"] == "skill",
                "semantic_file_count": 1 if kwargs["condition"] == "skill" else 0,
            },
            None,
        ),
    )

    _, surface, _ = execution._native_surface(
        provider="codex",
        condition="docs",
        workspace=workspace,
        run_root=root,
        scratch=scratch,
        environment={},
        skill=b"skill",
        skill_name="probe",
    )

    assert surface == {"skill_installed": False, "semantic_file_count": 0}


def test_codex_native_surface_can_read_declared_toolchain_roots(tmp_path, monkeypatch):
    root = tmp_path / "run"
    workspace = root / "workspace"
    scratch = root / "scratch"
    maven = tmp_path / "maven"
    java = tmp_path / "java"
    workspace.mkdir(parents=True)
    scratch.mkdir()
    (maven / "bin").mkdir(parents=True)
    (java / "bin").mkdir(parents=True)
    monkeypatch.setenv("TRACEBOOK_V2_MAVEN_HOME", str(maven))
    monkeypatch.setenv("TRACEBOOK_V2_JAVA_HOME", str(java))
    monkeypatch.setattr(execution, "_configured_delivery", lambda: nullcontext())
    monkeypatch.setattr(
        execution.delivery,
        "_prepare_native_surface",
        lambda **kwargs: (
            ["provider", "permissions.agent-eval={filesystem={}}"],
            {"skill_installed": False},
            None,
        ),
    )

    command, _, _ = execution._native_surface(
        provider="codex",
        condition="docs",
        workspace=workspace,
        run_root=root,
        scratch=scratch,
        environment={},
        skill=b"skill",
        skill_name="probe",
    )

    permission = next(value for value in command if value.startswith("permissions.agent-eval="))
    assert f'"{maven}"="read"' in permission
    assert f'"{java}"="read"' in permission


def test_validate_prior_runs_enforces_seeded_order(monkeypatch):
    seen = []
    monkeypatch.setattr(execution, "validate_run", seen.append)
    plan = {
        "entries": [
            {"id": "first"},
            {"id": "second"},
            {"id": "third"},
        ]
    }

    execution._validate_prior_runs(plan, 3)

    assert seen == ["first", "second"]


def test_validate_prior_runs_reuses_one_frozen_context(monkeypatch):
    seen = []
    freeze = {"frozen": True}
    primary = ({"cases": []}, {"entries": []})

    def validate(run_id, **kwargs):
        seen.append((run_id, kwargs["freeze"], kwargs["primary"]))

    monkeypatch.setattr(execution, "validate_run", validate)
    plan = {"entries": [{"id": "first"}, {"id": "second"}, {"id": "third"}]}

    execution._validate_prior_runs(plan, 3, freeze=freeze, primary=primary)

    assert seen == [("first", freeze, primary), ("second", freeze, primary)]


def test_execution_refuses_without_provider_authorization(monkeypatch):
    monkeypatch.setattr(execution, "validate_freeze", lambda: {})
    monkeypatch.setattr(
        execution,
        "_load_primary",
        lambda: (
            {"cases": []},
            {
                "entries": [
                    {
                        "id": "skill__c4-compatible__codex__r1",
                        "agent": "codex",
                        "case_id": "c4-compatible",
                        "condition": "skill",
                        "repetition": 1,
                        "order": 1,
                    }
                ]
            },
            {},
        ),
    )
    monkeypatch.setattr(
        execution,
        "validate_authorization",
        lambda provider: (_ for _ in ()).throw(execution.ExecutionError("authorization missing")),
    )

    with pytest.raises(execution.ExecutionError, match="authorization missing"):
        execution.execute_run("skill__c4-compatible__codex__r1")


def test_wrapper_is_deterministic_and_refuses_drift(tmp_path, monkeypatch):
    wrapper = tmp_path / ".claude-plugin" / "plugin.json"
    monkeypatch.setattr(execution, "CLAUDE_WRAPPER_PATH", wrapper)

    execution._write_wrapper()
    first = wrapper.read_bytes()
    execution._write_wrapper()
    wrapper.write_text("{}")

    assert json.loads(first) == execution._PLUGIN_MANIFEST
    with pytest.raises(execution.ExecutionError, match="wrapper drifted"):
        execution._write_wrapper()


def test_toolchain_environment_prepends_frozen_tools(tmp_path, monkeypatch):
    maven = tmp_path / "maven"
    java = tmp_path / "java"
    python = tmp_path / "python"
    uv = tmp_path / "uv"
    for root in (maven, java, python):
        (root / "bin").mkdir(parents=True)
    uv.write_text("binary")
    monkeypatch.setenv("TRACEBOOK_V2_MAVEN_HOME", str(maven))
    monkeypatch.setenv("TRACEBOOK_V2_JAVA_HOME", str(java))
    monkeypatch.setenv("TRACEBOOK_V2_PYTHON_HOME", str(python))
    monkeypatch.setenv("TRACEBOOK_V2_UV_BINARY", str(uv))
    monkeypatch.setenv("PATH", "/operator/private/bin")

    environment = execution._toolchain_environment()

    assert environment["JAVA_HOME"] == str(java)
    assert environment["PATH"].split(":")[:4] == [
        str(maven / "bin"),
        str(java / "bin"),
        str(python / "bin"),
        str(tmp_path),
    ]
    assert "/operator/private/bin" not in environment["PATH"]
    assert "/usr/bin" in environment["PATH"].split(":")


def test_provider_shells_receive_fresh_paths(tmp_path, monkeypatch):
    monkeypatch.delenv("TRACEBOOK_V2_MAVEN_HOME", raising=False)
    monkeypatch.delenv("TRACEBOOK_V2_JAVA_HOME", raising=False)
    monkeypatch.delenv("TRACEBOOK_V2_PYTHON_HOME", raising=False)
    monkeypatch.delenv("TRACEBOOK_V2_UV_BINARY", raising=False)

    values = execution._codex_shell_environment(tmp_path)

    assert values["FRESH_M2"] == str(tmp_path / "m2-repository")
    assert values["FRESH_CARGO_HOME"] == str(tmp_path / "cargo-home")
    assert values["FRESH_TARGET"] == str(tmp_path / "cargo-target")


def test_claude_binding_probes_an_explicit_first_party_route(tmp_path, monkeypatch):
    binary = tmp_path / "claude"
    binary.write_text("claude")
    observed = {}

    monkeypatch.setattr(execution, "_provider_binary", lambda provider: binary)
    monkeypatch.setattr(execution.helpers, "_binary_version", lambda *args: "2.1.220")

    def run(command, **kwargs):
        observed["command"] = command
        return execution.subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(
                {
                    "loggedIn": True,
                    "authMethod": "claude.ai",
                    "apiProvider": "firstParty",
                    "subscriptionType": "max",
                }
            ),
        )

    monkeypatch.setattr(execution.subprocess, "run", run)

    binding = execution._provider_binding("claude")

    command = observed["command"]
    assert command[1:4] == ("--setting-sources", "", "--settings")
    assert json.loads(command[4])["env"] == execution.CLAUDE_FIRST_PARTY_ENV
    assert command[5:] == ("auth", "status")
    assert binding["auth_route"]["apiProvider"] == "firstParty"


def test_claude_run_settings_disable_third_party_routes(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    plugin_root = tmp_path / "plugin"
    workspace.mkdir()
    scratch.mkdir()
    plugin_root.mkdir()
    maven = tmp_path / "maven"
    java = tmp_path / "java"
    (maven / "bin").mkdir(parents=True)
    (java / "bin").mkdir(parents=True)
    monkeypatch.setenv("TRACEBOOK_V2_MAVEN_HOME", str(maven))
    monkeypatch.setenv("TRACEBOOK_V2_JAVA_HOME", str(java))

    settings = execution._claude_settings(workspace, scratch, plugin_root)

    for name, value in execution.CLAUDE_FIRST_PARTY_ENV.items():
        assert settings["env"][name] == value
    assert str(maven) in settings["sandbox"]["filesystem"]["allowRead"]
    assert str(java) in settings["sandbox"]["filesystem"]["allowRead"]


def test_delivery_configuration_restores_imported_module_globals(tmp_path, monkeypatch):
    binary = tmp_path / "provider"
    binary.write_text("provider")
    monkeypatch.setattr(execution, "_provider_binary", lambda provider: binary)
    original = (
        execution.helpers.CODEX_BINARY,
        execution.helpers.CLAUDE_BINARY,
        execution.helpers.CODEX_AUTH_PATH,
        execution.delivery.DEFAULT_CLAUDE_WRAPPER_PATH,
        execution.delivery.EXPECTED_MODELS,
        execution.helpers._codex_shell_environment,
    )

    with execution._configured_delivery():
        assert execution.helpers.CODEX_BINARY == binary
        assert execution.helpers.CLAUDE_BINARY == binary
        assert execution.helpers._codex_shell_environment is execution._codex_shell_environment

    assert (
        execution.helpers.CODEX_BINARY,
        execution.helpers.CLAUDE_BINARY,
        execution.helpers.CODEX_AUTH_PATH,
        execution.delivery.DEFAULT_CLAUDE_WRAPPER_PATH,
        execution.delivery.EXPECTED_MODELS,
        execution.helpers._codex_shell_environment,
    ) == original


def test_interrupted_v2_run_is_quarantined_for_clean_restart(tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    external = tmp_path / "external"
    monkeypatch.setattr(execution, "RUNS_ROOT", runs)
    monkeypatch.setattr(execution, "EXTERNAL_ROOT", external)

    def fail(run_id, timeout_seconds):
        (runs / run_id).mkdir(parents=True)
        (external / run_id / "workspace").mkdir(parents=True)
        raise KeyboardInterrupt("interrupted")

    monkeypatch.setattr(execution, "_execute_run_once", fail)

    with pytest.raises(KeyboardInterrupt, match="interrupted"):
        execution.execute_run("docs__case__codex__r1")

    assert not (runs / "docs__case__codex__r1").exists()
    assert not (external / "docs__case__codex__r1").exists()
    assert len(list((runs / "interruptions").iterdir())) == 1


def test_shakedown_requires_first_party_claude_route(monkeypatch):
    def binding(provider):
        if provider == "claude":
            return {
                "auth_route": {
                    "loggedIn": True,
                    "authMethod": "third_party",
                    "apiProvider": "bedrock",
                }
            }
        return {"requested_model": execution.EXPECTED_MODELS[provider]}

    monkeypatch.setattr(execution, "_provider_binding", binding)

    with pytest.raises(execution.ExecutionError, match="first-party route"):
        execution._validated_provider_bindings()
