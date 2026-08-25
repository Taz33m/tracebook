import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

import pytest

from experiments import agent_qualification
from experiments import agent_qualification_treatments as treatments

BASELINE_RUNNER_SHA256 = "71300722b273e983dcd4e830358677337255a39f52b7f10e4c57479c8dc5cb0d"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _synthetic_manifest():
    return {
        "cases": [
            {
                "id": case_id,
                "status": "ready",
                "snapshot_id": f"sha256:{case_id}",
                "revision": f"revision-{case_id}",
                "commands": "native-test",
                "declared_claim": "price-time matching",
                "excluded_scope": "production readiness",
            }
            for case_id in ("case-alpha", "case-beta")
        ]
    }


def test_frozen_baseline_runner_is_byte_identical():
    assert _sha256(agent_qualification.RUNNER_PATH) == BASELINE_RUNNER_SHA256


def test_treatment_prompt_is_identical_between_docs_and_skill():
    manifest = _synthetic_manifest()
    for case in manifest["cases"]:
        if case["status"] != "ready":
            continue
        rendered = treatments.render_prompt(case)
        assert rendered == agent_qualification.render_prompt(case, "docs")
        assert rendered == agent_qualification.render_prompt(case, "skill")
        assert "tracebook-conformance==0.6.0" in rendered


def test_master_plan_is_two_matched_blocks_in_baseline_order():
    manifest = _synthetic_manifest()
    evidence_ids = [f"ev_{index:032x}" for index in range(24)]
    entries = treatments._master_entries(manifest, evidence_ids=evidence_ids)
    baseline_cells = [
        {
            "case_id": case["id"],
            "agent": agent,
            "repetition": repetition,
        }
        for case in manifest["cases"]
        for agent in ("codex", "claude")
        for repetition in range(1, treatments.REPETITIONS + 1)
    ]
    random.Random(agent_qualification.RANDOMIZATION_SEED).shuffle(baseline_cells)

    assert len(entries) == 24
    assert [entry["condition"] for entry in entries[:12]] == ["docs"] * 12
    assert [entry["condition"] for entry in entries[12:]] == ["skill"] * 12
    for offset in (0, 12):
        assert [
            {
                "case_id": entry["case_id"],
                "agent": entry["agent"],
                "repetition": entry["repetition"],
            }
            for entry in entries[offset : offset + 12]
        ] == baseline_cells
    assert len({entry["run_id"] for entry in entries}) == 24
    assert len({entry["evidence_id"] for entry in entries}) == 24
    assert [entry["evidence_id"] for entry in entries] == evidence_ids


def test_master_plan_preserves_manifest_case_order_before_seeded_shuffle():
    manifest = {
        "cases": [
            {"id": "z-last-lexically", "status": "ready"},
            {"id": "a-first-lexically", "status": "ready"},
        ]
    }
    evidence_ids = [f"ev_{index:032x}" for index in range(24)]
    entries = treatments._master_entries(manifest, evidence_ids=evidence_ids)
    expected = [
        {"case_id": case["id"], "agent": agent, "repetition": repetition}
        for case in manifest["cases"]
        for agent in ("codex", "claude")
        for repetition in range(1, treatments.REPETITIONS + 1)
    ]
    random.Random(agent_qualification.RANDOMIZATION_SEED).shuffle(expected)

    assert [
        {
            "case_id": entry["case_id"],
            "agent": entry["agent"],
            "repetition": entry["repetition"],
        }
        for entry in entries[:12]
    ] == expected


def test_evidence_ids_are_random_opaque_values():
    first = treatments._new_evidence_id()
    second = treatments._new_evidence_id()

    assert first != second
    assert len(first) == len("ev_") + 32
    assert int(first.removeprefix("ev_"), 16) >= 0


def test_skill_is_candidate_agnostic_and_exactly_one_measured_file(tmp_path, monkeypatch):
    monkeypatch.setattr(treatments, "PRIVATE_ROOT", tmp_path)
    skill_path = tmp_path / "skill" / "SKILL.md"
    skill_path.parent.mkdir()
    skill_path.write_text(
        "---\n"
        f"name: {treatments.SKILL_NAME}\n"
        "description: Candidate-agnostic qualification workflow.\n"
        "---\n\n"
        "Inspect the supplied candidate and preserve identity lifecycles.\n"
    )
    skill = treatments._load_skill(skill_path, _synthetic_manifest())

    assert skill["name"] == treatments.SKILL_NAME
    assert skill["sha256"] == _sha256(skill_path)
    assert b"heldout" not in skill["content"].lower()
    assert b"order-matcher" not in skill["content"].lower()
    assert b"matching-engine-rs" not in skill["content"].lower()


def test_skill_leakage_audit_rejects_heldout_identifier(tmp_path, monkeypatch):
    monkeypatch.setattr(treatments, "PRIVATE_ROOT", tmp_path)
    skill = tmp_path / "SKILL.md"
    skill.write_text(
        "---\n"
        f"name: {treatments.SKILL_NAME}\n"
        "description: synthetic\n"
        "---\n"
        "Target c1-heldout directly.\n"
    )
    manifest = {"cases": [{"id": "c1-heldout"}]}

    with pytest.raises(treatments.EvaluationError, match="leakage audit"):
        treatments._load_skill(skill, manifest)


def test_private_output_rejects_escape_before_creating_parent(tmp_path, monkeypatch):
    private_root = tmp_path / "private"
    private_root.mkdir()
    outside = tmp_path / "outside" / "result.json"
    monkeypatch.setattr(treatments, "PRIVATE_ROOT", private_root)

    with pytest.raises(treatments.EvaluationError, match="must stay inside"):
        treatments._private_output(outside, "test output")

    assert not outside.parent.exists()


def test_master_plan_shape_rejects_tampering():
    entries = [
        {
            "ordinal": index,
            "run_id": f"run-{index}",
            "evidence_id": f"ev_{index:032x}",
            "condition": "docs" if index <= 12 else "skill",
        }
        for index in range(1, 25)
    ]
    assert len(treatments._validate_plan_entry_shape(entries)) == 24
    entries[12]["evidence_id"] = entries[0]["evidence_id"]

    with pytest.raises(treatments.EvaluationError, match="evidence IDs"):
        treatments._validate_plan_entry_shape(entries)


def test_claude_treatment_command_is_skill_capable_and_customization_isolated(
    tmp_path,
):
    command = treatments._claude_treatment_command(tmp_path / "settings.json", tmp_path / "plugin")

    assert "--safe-mode" not in command
    assert "--disable-slash-commands" not in command
    assert command[command.index("--setting-sources") + 1] == ""
    assert command[command.index("--plugin-dir") + 1] == str(tmp_path / "plugin")
    assert "Skill" in command[command.index("--tools") + 1].split(",")
    assert command[command.index("--model") + 1] == "claude-opus-4-8"
    assert command[command.index("--mcp-config") + 1] == '{"mcpServers":{}}'


def test_claude_wrapper_tree_contains_no_semantic_payload(tmp_path):
    root = tmp_path / "claude-wrapper"
    manifest = root / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name":"agent-qualification-treatment","version":"1.0.0"}\n')
    record = treatments._tree_record(root)

    assert record["file_count"] == 1
    assert record["files"][0]["path"] == ".claude-plugin/plugin.json"


def test_claude_settings_make_plugin_read_only_to_agent(tmp_path):
    workspace = tmp_path / "workspace"
    scratch = tmp_path / "scratch"
    plugin = scratch / "plugin"
    workspace.mkdir()
    scratch.mkdir()
    plugin.mkdir()

    settings = treatments._claude_treatment_settings(workspace, scratch, plugin)

    assert "CLAUDE_CONFIG_DIR" not in settings["env"]
    assert "HOME" not in settings["env"]
    assert settings["env"]["ZDOTDIR"] == str(scratch / "zsh-env")
    assert (scratch / "zsh-env" / ".zshenv").read_text() == (
        f"export HOME={scratch / 'shell-home'}\n"
    )
    assert str(plugin) in settings["sandbox"]["filesystem"]["denyWrite"]
    assert f"Edit(/{plugin}/**)" in settings["permissions"]["deny"]


def test_codex_native_surface_differs_only_by_skill_file(tmp_path, monkeypatch):
    auth = tmp_path / "auth.json"
    auth.write_text("{}\n")
    monkeypatch.setattr(agent_qualification, "CODEX_AUTH_PATH", auth)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    skill = b"synthetic frozen skill\n"

    docs_root = tmp_path / "docs"
    docs_scratch = docs_root / "scratch"
    docs_root.mkdir()
    docs_scratch.mkdir()
    docs_environment: dict[str, str] = {}
    docs_command, docs_surface, _ = treatments._prepare_native_surface(
        agent="codex",
        condition="docs",
        workspace=workspace,
        external_run_root=docs_root,
        scratch=docs_scratch,
        environment=docs_environment,
        skill_content=skill,
    )

    skill_root = tmp_path / "skill"
    skill_scratch = skill_root / "scratch"
    skill_root.mkdir()
    skill_scratch.mkdir()
    skill_environment: dict[str, str] = {}
    skill_command, skill_surface, _ = treatments._prepare_native_surface(
        agent="codex",
        condition="skill",
        workspace=workspace,
        external_run_root=skill_root,
        scratch=skill_scratch,
        environment=skill_environment,
        skill_content=skill,
    )

    assert docs_surface["skill_installed"] is False
    assert skill_surface["skill_installed"] is True
    assert not (docs_root / "codex-home" / "skills" / treatments.SKILL_NAME).exists()
    installed = skill_root / "codex-home" / "skills" / treatments.SKILL_NAME / "SKILL.md"
    assert installed.read_bytes() == skill
    for command, root in ((docs_command, docs_root), (skill_command, skill_root)):
        permission = next(item for item in command if item.startswith("permissions.agent-eval="))
        assert f'"{root / "codex-home" / "skills"}"="read"' in permission


def test_native_surface_final_audit_detects_absence_and_drift(tmp_path):
    skill = b"synthetic frozen skill\n"
    external_root = tmp_path / "run"
    scratch = external_root / "scratch"
    scratch.mkdir(parents=True)

    docs = treatments._audit_native_surface(
        agent="claude",
        condition="docs",
        external_run_root=external_root,
        scratch=scratch,
        skill_content=skill,
    )
    assert docs["mutated"] is False

    path = scratch / "claude-plugin" / "skills" / treatments.SKILL_NAME / "SKILL.md"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"drift")
    treatment = treatments._audit_native_surface(
        agent="claude",
        condition="skill",
        external_run_root=external_root,
        scratch=scratch,
        skill_content=skill,
    )
    assert treatment["mutated"] is True


def test_transcript_terminal_state_requires_clean_completion(tmp_path):
    codex = tmp_path / "codex.jsonl"
    codex.write_text('{"type":"turn.completed"}\n')
    claude = tmp_path / "claude.jsonl"
    claude.write_text(
        '{"type":"result","terminal_reason":"completed","subtype":"success",' '"result":"done"}\n'
    )
    failed = tmp_path / "failed.jsonl"
    failed.write_text('{"type":"error","message":"network"}\n')

    assert treatments._transcript_terminal_state(codex, "codex") == "completed"
    assert treatments._transcript_terminal_state(claude, "claude") == "completed"
    assert treatments._transcript_terminal_state(failed, "codex") == "failed"


def test_claude_catalog_audit_requires_only_wrapper_and_expected_skill(tmp_path):
    transcript = tmp_path / "claude.jsonl"
    transcript.write_text(
        "provider banner that is not JSON\n"
        + json.dumps(
            {
                "type": "system",
                "subtype": "init",
                "model": "claude-opus-4-8",
                "plugins": [{"name": "agent-qualification-treatment"}],
                "skills": [f"agent-qualification-treatment:{treatments.SKILL_NAME}"],
                "mcp_servers": [],
                "tools": [
                    "Bash",
                    "Edit",
                    "Glob",
                    "Grep",
                    "Read",
                    "Skill",
                    "WebFetch",
                    "WebSearch",
                    "Write",
                ],
            }
        )
        + "\n"
    )

    assert treatments._claude_catalog_audit(transcript, condition="skill")["valid"] is True
    assert treatments._claude_catalog_audit(transcript, condition="docs")["valid"] is False
    assert treatments._transcript_result_text(transcript, "claude") == ""


def test_transcript_result_text_skips_non_json_lines(tmp_path):
    transcript = tmp_path / "claude-result.jsonl"
    transcript.write_text('banner\n{"type":"result","result":"done"}\n')

    assert treatments._transcript_result_text(transcript, "claude") == "done"


def test_measurement_policy_uses_conservative_paired_timing_gate():
    timing = treatments.MEASUREMENT_POLICY["timing_gate"]

    assert timing["minimum_matched_autonomous_safe_pass_pairs"] == 3
    assert timing["both_cases_required"] is True
    assert timing["skill_safe_pass_rate_must_not_be_lower"] is True
    assert timing["pass_threshold"] == 0.70


def test_official_run_rejects_timeout_drift_before_freeze_validation():
    with pytest.raises(treatments.EvaluationError, match="timeout must remain"):
        treatments.execute_run("not-a-run", timeout_seconds=1)


def test_main_converts_subprocess_failures_to_exit_two(monkeypatch, capsys):
    monkeypatch.setattr(
        treatments,
        "run_native_delivery_shakedown",
        lambda: (_ for _ in ()).throw(subprocess.CalledProcessError(1, ["git"])),
    )

    assert treatments.main(["shakedown"]) == 2
    assert "error:" in capsys.readouterr().err


def test_direct_script_entrypoint_smoke():
    completed = subprocess.run(
        [
            sys.executable,
            str(treatments.RUNNER_PATH),
            "--help",
        ],
        cwd=treatments.REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "create the immutable 24-run plan" in completed.stdout
