"""Execute the separately frozen v2 agent-qualification run matrix.

The primary v2 cohort is immutable. This companion freezes machine/provider
bindings, verifies native skill delivery on synthetic material, enforces
provider-specific authorization, and records technical evidence without
grading semantic outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

if __package__:
    from . import agent_qualification as helpers
    from . import agent_qualification_generalization as cohort
    from . import agent_qualification_treatments as delivery
else:  # Support direct execution from the repository root.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import agent_qualification as helpers  # type: ignore[no-redef]
    import agent_qualification_generalization as cohort  # type: ignore[no-redef]
    import agent_qualification_treatments as delivery  # type: ignore[no-redef]

ExecutionError = helpers.EvaluationError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
TEST_PATH = REPOSITORY_ROOT / "tests" / "test_agent_qualification_generalization_runs.py"
ADDENDUM_PATH = REPOSITORY_ROOT / "docs" / "agent-qualification-generalization-execution-v2.md"
PRIVATE_ROOT = cohort.PRIVATE_ROOT
EXECUTION_ROOT = PRIVATE_ROOT / "execution"
RUNS_ROOT = EXECUTION_ROOT / "runs"
AUTHORIZATION_ROOT = EXECUTION_ROOT / "authorizations"
FREEZE_PATH = EXECUTION_ROOT / "freeze.json"
SHAKEDOWN_PATH = EXECUTION_ROOT / "native-delivery-shakedown.json"
CLAUDE_WRAPPER_PATH = (
    EXECUTION_ROOT / "provider-wrappers" / "claude" / ".claude-plugin" / "plugin.json"
)
EXTERNAL_ROOT = Path(
    os.environ.get("TRACEBOOK_V2_EXTERNAL_ROOT", "/private/tmp/tracebook-agent-qualification-v2")
)

PROTOCOL_ID = cohort.PROTOCOL_ID
FREEZE_KIND = "agent-qualification-generalization-execution-freeze-v1"
SHAKEDOWN_KIND = "agent-qualification-generalization-native-skill-shakedown-v1"
VERDICT_KIND = "agent-qualification-generalization-pre-score-verdict-v1"
AUTHORIZATION_KIND = "agent-qualification-generalization-provider-authorization-v1"
SKILL_NAME = "tracebook-qualify-matching-engine"
SYNTHETIC_SKILL_NAME = "native-delivery-probe"
SYNTHETIC_MARKER = "NATIVE_SKILL_PROBE_V2_20260801_7F31C9"
DEFAULT_TIMEOUT_SECONDS = 120 * 60
EXPECTED_MODELS = {"codex": "gpt-5.6-sol", "claude": "claude-opus-4-8"}
PROVIDERS = tuple(EXPECTED_MODELS)
CLAUDE_FIRST_PARTY_ENV = {
    "CLAUDE_CODE_USE_BEDROCK": "0",
    "CLAUDE_CODE_USE_VERTEX": "0",
    "CLAUDE_CODE_USE_FOUNDRY": "0",
}
RESTRICTED_SYSTEM_PATH = (
    "/Library/Developer/CommandLineTools/usr/bin",
    "/usr/local/bin",
    "/opt/homebrew/bin",
    "/usr/bin",
    "/bin",
    "/usr/sbin",
    "/sbin",
)

_PLUGIN_MANIFEST = {
    "author": {"name": "Tracebook evaluator"},
    "description": "Isolated carrier for a preregistered agent evaluation treatment.",
    "name": "agent-qualification-treatment",
    "version": "2.0.0",
}
_SYNTHETIC_SKILL = (
    "---\n"
    "name: native-delivery-probe\n"
    "description: Synthetic probe used only to verify native skill delivery.\n"
    "---\n\n"
    "When explicitly asked to run this probe, reply with exactly " + SYNTHETIC_MARKER + ".\n"
).encode()

_ORIGINAL_CODEX_SHELL_ENVIRONMENT = helpers._codex_shell_environment


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _read_json(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExecutionError(f"cannot read {description} from {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ExecutionError(f"{description} must contain one JSON object")
    return value


def _write_new_json(path: Path, value: Mapping[str, Any], description: str) -> None:
    if path.exists() or path.is_symlink():
        raise ExecutionError(f"refusing to overwrite {description} {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_json_bytes(value))


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.resolve(strict=True)
    if not resolved.is_file():
        raise ExecutionError(f"required file is unavailable: {path}")
    try:
        label = resolved.relative_to(REPOSITORY_ROOT.resolve()).as_posix()
    except ValueError:
        label = str(resolved)
    return {
        "path": label,
        "sha256": helpers._sha256_file(resolved),
        "bytes": resolved.stat().st_size,
    }


def _validate_file_record(record: Any, description: str) -> Path:
    if not isinstance(record, dict):
        raise ExecutionError(f"{description} binding is missing")
    raw_path = record.get("path")
    if not isinstance(raw_path, str):
        raise ExecutionError(f"{description} path is missing")
    candidate = Path(raw_path)
    path = candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate
    if _file_record(path) != record:
        raise ExecutionError(f"{description} binding drifted")
    return path.resolve()


def _provider_binary(provider: str) -> Path:
    if provider not in PROVIDERS:
        raise ExecutionError(f"unknown provider {provider!r}")
    override = os.environ.get(f"TRACEBOOK_V2_{provider.upper()}_BINARY")
    candidates = [
        Path(override) if override else None,
        Path(shutil.which(provider) or "") if shutil.which(provider) else None,
        Path.home() / ".local" / "bin" / provider,
        Path("/Applications/ChatGPT.app/Contents/Resources/codex") if provider == "codex" else None,
    ]
    for candidate in candidates:
        if candidate is not None and candidate.is_file():
            return candidate.resolve()
    raise ExecutionError(f"{provider} CLI is unavailable")


def _clean_probe_environment() -> dict[str, str]:
    secret_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    return {
        name: value
        for name, value in os.environ.items()
        if name not in helpers.SECRET_ENVIRONMENT_NAMES
        and not any(fragment in name.upper() for fragment in secret_fragments)
    }


def _provider_binding(provider: str) -> dict[str, Any]:
    binary = _provider_binary(provider)
    version = helpers._binary_version(binary, _clean_probe_environment())
    binding: dict[str, Any] = {
        "configured_path": str(binary),
        "resolved_path": str(binary.resolve()),
        "sha256": helpers._sha256_file(binary),
        "bytes": binary.stat().st_size,
        "cli_version": version,
        "requested_model": EXPECTED_MODELS[provider],
    }
    if provider == "claude":
        route_settings = json.dumps(
            {"env": CLAUDE_FIRST_PARTY_ENV},
            sort_keys=True,
            separators=(",", ":"),
        )
        completed = subprocess.run(
            (
                str(binary),
                "--setting-sources",
                "",
                "--settings",
                route_settings,
                "auth",
                "status",
            ),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
        )
        try:
            auth = json.loads(completed.stdout)
        except json.JSONDecodeError:
            auth = {"raw_sha256": hashlib.sha256(completed.stdout.encode()).hexdigest()}
        if isinstance(auth, dict):
            # Do not persist account identifiers in the experiment archive.
            binding["auth_route"] = {
                key: auth.get(key)
                for key in ("loggedIn", "authMethod", "apiProvider", "subscriptionType")
                if key in auth
            }
    return binding


def _validated_provider_bindings() -> dict[str, dict[str, Any]]:
    bindings = {provider: _provider_binding(provider) for provider in PROVIDERS}
    route = bindings["claude"].get("auth_route")
    if (
        not isinstance(route, dict)
        or route.get("loggedIn") is not True
        or route.get("apiProvider") != "firstParty"
    ):
        raise ExecutionError(
            "Claude must use an authenticated first-party route before native-delivery "
            "shakedown or execution freeze"
        )
    return bindings


@contextmanager
def _configured_delivery() -> Iterator[None]:
    """Temporarily bind shared delivery helpers and always restore their globals."""

    original_codex_binary = helpers.CODEX_BINARY
    original_claude_binary = helpers.CLAUDE_BINARY
    original_codex_auth_path = helpers.CODEX_AUTH_PATH
    original_claude_wrapper_path = delivery.DEFAULT_CLAUDE_WRAPPER_PATH
    original_expected_models = delivery.EXPECTED_MODELS
    original_codex_shell_environment = helpers._codex_shell_environment
    try:
        helpers.CODEX_BINARY = _provider_binary("codex")
        helpers.CLAUDE_BINARY = _provider_binary("claude")
        helpers.CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
        delivery.DEFAULT_CLAUDE_WRAPPER_PATH = CLAUDE_WRAPPER_PATH
        delivery.EXPECTED_MODELS = dict(EXPECTED_MODELS)
        helpers._codex_shell_environment = _codex_shell_environment
        yield
    finally:
        helpers.CODEX_BINARY = original_codex_binary
        helpers.CLAUDE_BINARY = original_claude_binary
        helpers.CODEX_AUTH_PATH = original_codex_auth_path
        delivery.DEFAULT_CLAUDE_WRAPPER_PATH = original_claude_wrapper_path
        delivery.EXPECTED_MODELS = original_expected_models
        helpers._codex_shell_environment = original_codex_shell_environment


def _toolchain_read_paths() -> tuple[Path, ...]:
    paths: list[Path] = []
    for name in (
        "TRACEBOOK_V2_MAVEN_HOME",
        "TRACEBOOK_V2_JAVA_HOME",
        "TRACEBOOK_V2_PYTHON_HOME",
    ):
        raw = os.environ.get(name)
        if raw:
            paths.append(Path(raw).resolve())
    uv_binary = os.environ.get("TRACEBOOK_V2_UV_BINARY")
    if uv_binary:
        paths.append(Path(uv_binary).resolve().parent)
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        paths.append(cargo_bin.resolve())
    return tuple(dict.fromkeys(paths))


def _toolchain_environment() -> dict[str, str]:
    values: dict[str, str] = {}
    path_prefixes: list[str] = []
    for name in ("TRACEBOOK_V2_MAVEN_HOME", "TRACEBOOK_V2_JAVA_HOME"):
        raw = os.environ.get(name)
        if not raw:
            continue
        root = Path(raw).resolve()
        if not root.is_dir():
            raise ExecutionError(f"{name} is not an available directory: {root}")
        path_prefixes.append(str(root / "bin"))
        if name == "TRACEBOOK_V2_JAVA_HOME":
            values["JAVA_HOME"] = str(root)
    python_home = os.environ.get("TRACEBOOK_V2_PYTHON_HOME")
    if python_home:
        root = Path(python_home).resolve()
        if not root.is_dir():
            raise ExecutionError(f"TRACEBOOK_V2_PYTHON_HOME is not an available directory: {root}")
        path_prefixes.append(str(root / "bin"))
    uv_binary = os.environ.get("TRACEBOOK_V2_UV_BINARY")
    if uv_binary:
        path = Path(uv_binary).resolve()
        if not path.is_file():
            raise ExecutionError(f"TRACEBOOK_V2_UV_BINARY is not an available file: {path}")
        path_prefixes.append(str(path.parent))
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        path_prefixes.append(str(cargo_bin))
    values["PATH"] = ":".join(dict.fromkeys([*path_prefixes, *RESTRICTED_SYSTEM_PATH]))
    return values


def _codex_shell_environment(scratch: Path) -> dict[str, str]:
    values = _ORIGINAL_CODEX_SHELL_ENVIRONMENT(scratch)
    values.update(_toolchain_environment())
    values.update(_fresh_environment(scratch))
    return values


def _claude_settings(workspace: Path, scratch: Path, plugin_root: Path) -> dict[str, Any]:
    settings = delivery._claude_treatment_settings(workspace, scratch, plugin_root)
    settings["env"].update(CLAUDE_FIRST_PARTY_ENV)
    settings["env"].update(_toolchain_environment())
    settings["env"].update(_fresh_environment(scratch))
    allow_read = settings["sandbox"]["filesystem"]["allowRead"]
    for path in _toolchain_read_paths():
        if str(path) not in allow_read:
            allow_read.append(str(path))
    return settings


def _fresh_environment(scratch: Path) -> dict[str, str]:
    return {
        "FRESH_M2": str(scratch / "m2-repository"),
        "FRESH_CARGO_HOME": str(scratch / "cargo-home"),
        "FRESH_TARGET": str(scratch / "cargo-target"),
    }


def _write_wrapper() -> None:
    if CLAUDE_WRAPPER_PATH.exists():
        if _read_json(CLAUDE_WRAPPER_PATH, "Claude plugin wrapper") != _PLUGIN_MANIFEST:
            raise ExecutionError("Claude plugin wrapper drifted")
        return
    CLAUDE_WRAPPER_PATH.parent.mkdir(parents=True, exist_ok=True)
    CLAUDE_WRAPPER_PATH.write_bytes(_json_bytes(_PLUGIN_MANIFEST))


def _case_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ExecutionError("validated cohort lost its cases")
    return {
        str(case["id"]): dict(case)
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    }


def _plan_entry(plan: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise ExecutionError("validated cohort lost its plan entries")
    matches = [entry for entry in entries if isinstance(entry, dict) and entry.get("id") == run_id]
    if len(matches) != 1:
        raise ExecutionError(f"run {run_id!r} is not one frozen plan entry")
    return dict(matches[0])


def _load_primary() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cohort.validate()
    return (
        cohort._load_json(cohort.MANIFEST_PATH),
        cohort._load_json(cohort.PLAN_PATH),
        cohort._load_json(cohort.FREEZE_PATH),
    )


def _authorization_path(provider: str) -> Path:
    if provider not in PROVIDERS:
        raise ExecutionError(f"unknown provider {provider!r}")
    return AUTHORIZATION_ROOT / f"{provider}.json"


def record_authorization(provider: str, statement_file: Path) -> dict[str, Any]:
    """Record explicit consent already supplied by the user; never infer it."""

    validate_freeze()
    statement = statement_file.read_text().strip()
    if not statement:
        raise ExecutionError("authorization statement must not be empty")
    payload = {
        "kind": AUTHORIZATION_KIND,
        "protocol": PROTOCOL_ID,
        "provider": provider,
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
        "recorded_at": helpers._utc_now(),
    }
    _write_new_json(_authorization_path(provider), payload, f"{provider} authorization")
    return payload


def validate_authorization(provider: str) -> dict[str, Any]:
    payload = _read_json(_authorization_path(provider), f"{provider} authorization")
    expected = {
        "kind": AUTHORIZATION_KIND,
        "protocol": PROTOCOL_ID,
        "provider": provider,
        "case_ids": ["c4-compatible", "c5-boundary"],
        "materials": [
            "public repository snapshots",
            "frozen prompt",
            "docs-assisted treatment",
            "skill-assisted treatment",
        ],
        "scope": "all-preregistered-runs-and-genuine-technical-restarts",
    }
    for field, value in expected.items():
        if payload.get(field) != value:
            raise ExecutionError(f"{provider} authorization {field!r} is invalid")
    statement = payload.get("user_statement")
    if not isinstance(statement, str) or not statement.strip():
        raise ExecutionError(f"{provider} authorization statement is missing")
    if payload.get("statement_sha256") != hashlib.sha256(statement.encode()).hexdigest():
        raise ExecutionError(f"{provider} authorization statement binding drifted")
    if not isinstance(payload.get("recorded_at"), str):
        raise ExecutionError(f"{provider} authorization timestamp is missing")
    return payload


def _toolchain_versions() -> dict[str, Any]:
    result: dict[str, Any] = {}
    environment = dict(os.environ)
    environment.update(_toolchain_environment())
    for name, command in {
        "git": ("git", "--version"),
        "python": (sys.executable, "--version"),
        "java": ("java", "-version"),
        "maven": ("mvn", "--version"),
        "cargo": ("cargo", "--version"),
        "rustc": ("rustc", "--version"),
    }.items():
        binary = (
            shutil.which(command[0], path=environment["PATH"])
            if not Path(command[0]).is_absolute()
            else command[0]
        )
        if not binary:
            result[name] = {"available": False}
            continue
        completed = subprocess.run(
            (str(binary), *command[1:]),
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=30,
            env=environment,
        )
        result[name] = {
            "available": completed.returncode == 0,
            "path": str(Path(binary).resolve()),
            "version": completed.stdout.strip().splitlines()[0] if completed.stdout.strip() else "",
        }
    return result


def _native_surface(
    *,
    provider: str,
    condition: str,
    workspace: Path,
    run_root: Path,
    scratch: Path,
    environment: dict[str, str],
    skill: bytes,
    skill_name: str,
) -> tuple[list[str], dict[str, Any], Path | None]:
    with _configured_delivery():
        command, surface, plugin_root = delivery._prepare_native_surface(
            agent=provider,
            condition=condition,
            workspace=workspace,
            external_run_root=run_root,
            scratch=scratch,
            environment=environment,
            skill_content=skill,
            skill_name=skill_name,
        )
    if provider == "codex":
        permission_index = next(
            index
            for index, value in enumerate(command)
            if value.startswith("permissions.agent-eval=")
        )
        read_entries = "".join(
            f'{json.dumps(str(path))}="read",' for path in _toolchain_read_paths()
        )
        command[permission_index] = command[permission_index].replace(
            "filesystem={",
            "filesystem={" + read_entries,
            1,
        )
    return command, surface, plugin_root


def _claude_command(settings_path: Path, plugin_root: Path) -> list[str]:
    with _configured_delivery():
        return delivery._claude_treatment_command(settings_path, plugin_root)


def _run_delivery_probe(provider: str, condition: str, root: Path) -> dict[str, Any]:
    workspace = root / "workspace"
    scratch = root / "scratch"
    workspace.mkdir(parents=True)
    scratch.mkdir(parents=True)
    (workspace / "README.md").write_text("Synthetic native skill delivery probe.\n")
    environment = helpers._clean_agent_environment(scratch)
    environment.update(_toolchain_environment())
    command, surface, plugin_root = _native_surface(
        provider=provider,
        condition=condition,
        workspace=workspace,
        run_root=root,
        scratch=scratch,
        environment=environment,
        skill=_SYNTHETIC_SKILL,
        skill_name=SYNTHETIC_SKILL_NAME,
    )
    if provider == "claude":
        if plugin_root is None:
            raise ExecutionError("Claude probe plugin was not prepared")
        settings_path = root / "claude-settings.json"
        helpers._write_json(settings_path, _claude_settings(workspace, scratch, plugin_root))
        command = _claude_command(settings_path, plugin_root)
    transcript = root / f"{provider}.jsonl"
    stderr = root / f"{provider}.stderr"
    prompt = (
        "Run the native-delivery-probe shakedown. If its native skill is available, "
        "use it. If unavailable, reply exactly NO_NATIVE_DELIVERY. Do not search for "
        "skill files or inspect any path not supplied by the native skill catalog."
    )
    exit_code, timed_out, elapsed = helpers._run_subprocess(
        command,
        cwd=workspace,
        environment=environment,
        prompt_stdin=prompt,
        stdout_path=transcript,
        stderr_path=stderr,
        timeout_seconds=600,
    )
    audit = delivery._audit_native_surface(
        agent=provider,
        condition=condition,
        external_run_root=root,
        scratch=scratch,
        skill_content=_SYNTHETIC_SKILL,
        skill_name=SYNTHETIC_SKILL_NAME,
    )
    terminal = delivery._transcript_terminal_state(transcript, provider)
    result = delivery._transcript_result_text(transcript, provider)
    catalog_audit = None
    if provider == "claude":
        catalog_audit = delivery._claude_catalog_audit(
            transcript,
            condition=condition,
            skill_name=SYNTHETIC_SKILL_NAME,
        )
    expected = SYNTHETIC_MARKER if condition == "skill" else "NO_NATIVE_DELIVERY"
    errors = []
    if exit_code != 0 or timed_out or terminal != "completed":
        errors.append("provider did not complete cleanly")
    if audit.get("mutated") is not False:
        errors.append("native skill surface mutated")
    if provider == "claude" and catalog_audit is not None and catalog_audit["valid"] is not True:
        errors.append("Claude provider catalog isolation failed")
    if result != expected:
        errors.append(f"result {result!r} != {expected!r}")
    return {
        "valid": not errors,
        "errors": errors,
        "condition": condition,
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": elapsed,
        "terminal_state": terminal,
        "result": result,
        "surface": audit,
        "provider_catalog_audit": catalog_audit,
        "transcript": _file_record(transcript),
        "stderr": _file_record(stderr),
    }


def shakedown() -> dict[str, Any]:
    cohort.validate()
    _write_wrapper()
    provider_bindings = _validated_provider_bindings()
    if SHAKEDOWN_PATH.exists() or SHAKEDOWN_PATH.is_symlink():
        raise ExecutionError(f"refusing to overwrite shakedown {SHAKEDOWN_PATH}")
    root = EXECUTION_ROOT / "shakedown-work"
    if root.exists() or root.is_symlink():
        raise ExecutionError(f"refusing to reuse shakedown work root {root}")
    providers: dict[str, Any] = {}
    for provider in PROVIDERS:
        conditions: dict[str, Any] = {}
        for condition in ("docs", "skill"):
            result = _run_delivery_probe(provider, condition, root / provider / condition)
            if result["valid"] is not True:
                raise ExecutionError(
                    f"{provider} {condition} native-delivery probe failed: "
                    + "; ".join(result["errors"])
                )
            conditions[condition] = result
        providers[provider] = {"passed": True, "conditions": conditions}
    verdict = {
        "kind": SHAKEDOWN_KIND,
        "protocol": PROTOCOL_ID,
        "created_at": helpers._utc_now(),
        "passed": True,
        "runner": _file_record(RUNNER_PATH),
        "synthetic_skill_sha256": hashlib.sha256(_SYNTHETIC_SKILL).hexdigest(),
        "marker": SYNTHETIC_MARKER,
        "provider_bindings": provider_bindings,
        "providers": providers,
    }
    _write_new_json(SHAKEDOWN_PATH, verdict, "native-delivery shakedown")
    return verdict


def _validate_shakedown() -> dict[str, Any]:
    verdict = _read_json(SHAKEDOWN_PATH, "native-delivery shakedown")
    if verdict.get("kind") != SHAKEDOWN_KIND or verdict.get("protocol") != PROTOCOL_ID:
        raise ExecutionError("native-delivery shakedown identity changed")
    if verdict.get("passed") is not True or verdict.get("runner") != _file_record(RUNNER_PATH):
        raise ExecutionError("native-delivery shakedown binding changed")
    if verdict.get("synthetic_skill_sha256") != hashlib.sha256(_SYNTHETIC_SKILL).hexdigest():
        raise ExecutionError("native-delivery synthetic skill changed")
    if verdict.get("marker") != SYNTHETIC_MARKER:
        raise ExecutionError("native-delivery marker changed")
    if verdict.get("provider_bindings") != {
        provider: _provider_binding(provider) for provider in PROVIDERS
    }:
        raise ExecutionError("native-delivery provider bindings changed")
    providers = verdict.get("providers")
    if not isinstance(providers, dict) or set(providers) != set(PROVIDERS):
        raise ExecutionError("native-delivery provider coverage changed")
    for provider in PROVIDERS:
        conditions = (
            providers[provider].get("conditions") if isinstance(providers[provider], dict) else None
        )
        if not isinstance(conditions, dict) or set(conditions) != {"docs", "skill"}:
            raise ExecutionError(f"{provider} shakedown conditions changed")
        if any(
            result.get("valid") is not True
            for result in conditions.values()
            if isinstance(result, dict)
        ):
            raise ExecutionError(f"{provider} shakedown did not pass")
        for result in conditions.values():
            if not isinstance(result, dict):
                raise ExecutionError(f"{provider} shakedown result is invalid")
            _validate_file_record(result.get("transcript"), f"{provider} transcript")
            _validate_file_record(result.get("stderr"), f"{provider} stderr")
    return verdict


def _freeze_payload(created_at: str) -> dict[str, Any]:
    manifest, plan, primary_freeze = _load_primary()
    del manifest, plan, primary_freeze
    shakedown_verdict = _validate_shakedown()
    inputs = {
        "execution_addendum": _file_record(ADDENDUM_PATH),
        "execution_runner": _file_record(RUNNER_PATH),
        "execution_runner_tests": _file_record(TEST_PATH),
        "helper_runner": _file_record(helpers.RUNNER_PATH),
        "delivery_runner": _file_record(delivery.RUNNER_PATH),
        "primary_protocol": _file_record(cohort.PROTOCOL_PATH),
        "primary_prompt": _file_record(cohort.PROMPT_PATH),
        "primary_runner": _file_record(cohort.RUNNER_PATH),
        "primary_manifest": _file_record(cohort.MANIFEST_PATH),
        "primary_plan": _file_record(cohort.PLAN_PATH),
        "primary_freeze": _file_record(cohort.FREEZE_PATH),
        "skill": _file_record(cohort.SKILL_PATH),
        "claude_plugin_wrapper": _file_record(CLAUDE_WRAPPER_PATH),
        "native_delivery_shakedown": _file_record(SHAKEDOWN_PATH),
    }
    return {
        "kind": FREEZE_KIND,
        "protocol": PROTOCOL_ID,
        "created_at": created_at,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "inputs": inputs,
        "provider_bindings": shakedown_verdict["provider_bindings"],
        "toolchains": _toolchain_versions(),
        "run_count": 24,
        "order": "exact seeded primary plan order",
        "authorization": "provider-specific external records required after freeze",
        "restart_policy": (
            "archive and retry only genuine technical interruptions without a completed "
            "provider turn; never retry a timeout, refusal, or semantic outcome"
        ),
    }


def freeze_execution() -> dict[str, Any]:
    if FREEZE_PATH.exists() or FREEZE_PATH.is_symlink():
        raise ExecutionError(f"refusing to overwrite execution freeze {FREEZE_PATH}")
    payload = _freeze_payload(helpers._utc_now())
    _write_new_json(FREEZE_PATH, payload, "execution freeze")
    return payload


def validate_freeze() -> dict[str, Any]:
    cohort.validate()
    payload = _read_json(FREEZE_PATH, "execution freeze")
    if payload.get("kind") != FREEZE_KIND or payload.get("protocol") != PROTOCOL_ID:
        raise ExecutionError("execution freeze identity changed")
    created_at = payload.get("created_at")
    if not isinstance(created_at, str) or not created_at:
        raise ExecutionError("execution freeze timestamp is missing")
    if payload != _freeze_payload(created_at):
        raise ExecutionError("execution freeze content changed")
    return payload


def _copy_fixture(case: Mapping[str, Any], scratch: Path) -> dict[str, Any]:
    declaration = case.get("dependency_cache")
    if not isinstance(declaration, dict):
        raise ExecutionError("case dependency-cache declaration is missing")
    source = (REPOSITORY_ROOT / str(declaration["source"])).resolve(strict=True)
    target_name = cohort._dependency_target(declaration.get("target"))
    target = scratch / target_name
    helpers._copy_frozen_tree(source, target, str(declaration["tree_sha256"]))
    return {
        "source_manifest_sha256": declaration["manifest_sha256"],
        "initial_tree_sha256": declaration["tree_sha256"],
        "target": str(target),
    }


def _record_fixture_final(fixture: dict[str, Any]) -> None:
    target = Path(str(fixture["target"]))
    if not target.is_dir():
        fixture["final_state"] = "missing"
        fixture["mutated"] = True
        return
    try:
        final_hash = helpers._snapshot_digest(target)
    except OSError as exc:
        fixture["final_state"] = f"unreadable:{type(exc).__name__}"
        fixture["mutated"] = True
        return
    fixture["final_state"] = "present"
    fixture["final_tree_sha256"] = final_hash
    fixture["mutated"] = final_hash != fixture["initial_tree_sha256"]


def _run_root(run_id: str) -> Path:
    return RUNS_ROOT / run_id


def _external_run_root(run_id: str) -> Path:
    return EXTERNAL_ROOT / run_id


def _technical_verdict(
    run_root: Path, metadata: Mapping[str, Any], errors: list[str]
) -> dict[str, Any]:
    verdict = {
        "kind": VERDICT_KIND,
        "protocol": PROTOCOL_ID,
        "run_id": metadata.get("run_id"),
        "valid": not errors,
        "errors": errors,
        "terminal_state": metadata.get("terminal_state"),
        "timed_out": metadata.get("timed_out"),
        "exit_code": metadata.get("exit_code"),
        "metadata_sha256": helpers._sha256_file(run_root / "metadata.json"),
        "transcript_sha256": metadata.get("transcript_sha256"),
        "created_at": helpers._utc_now(),
    }
    _write_new_json(run_root / "pre-score-verdict.json", verdict, "pre-score verdict")
    return verdict


def validate_run(
    run_id: str,
    *,
    freeze: Mapping[str, Any] | None = None,
    primary: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if freeze is None:
        freeze = validate_freeze()
    if primary is None:
        loaded_manifest, loaded_plan, _ = _load_primary()
        manifest: Mapping[str, Any] = loaded_manifest
        plan: Mapping[str, Any] = loaded_plan
    else:
        manifest, plan = primary
    entry = _plan_entry(plan, run_id)
    run_root = _run_root(run_id)
    metadata = _read_json(run_root / "metadata.json", "run metadata")
    verdict = _read_json(run_root / "pre-score-verdict.json", "pre-score verdict")
    errors: list[str] = []
    expected = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "case_id": entry["case_id"],
        "provider": entry["agent"],
        "condition": entry["condition"],
        "repetition": entry["repetition"],
        "order": entry["order"],
        "primary_manifest_sha256": helpers._sha256_file(cohort.MANIFEST_PATH),
        "primary_plan_sha256": helpers._sha256_file(cohort.PLAN_PATH),
        "primary_freeze_sha256": helpers._sha256_file(cohort.FREEZE_PATH),
        "execution_freeze_sha256": helpers._sha256_file(FREEZE_PATH),
        "execution_runner_sha256": helpers._sha256_file(RUNNER_PATH),
    }
    for field, value in expected.items():
        if metadata.get(field) != value:
            errors.append(f"metadata {field} binding changed")
    case = _case_index(manifest)[str(entry["case_id"])]
    if metadata.get("snapshot_sha256") != case["snapshot_sha256"]:
        errors.append("candidate snapshot binding changed")
    if metadata.get("initial_workspace_sha256") != case["snapshot_sha256"]:
        errors.append("initial workspace binding changed")
    if metadata.get("frozen_skill_sha256") != helpers._sha256_file(cohort.SKILL_PATH):
        errors.append("frozen skill metadata binding changed")
    transcript = run_root / f"{entry['agent']}.jsonl"
    stderr = run_root / f"{entry['agent']}.stderr"
    prompt = run_root / "prompt.txt"
    for artifact in (transcript, stderr, prompt):
        if not artifact.is_file() or artifact.is_symlink():
            errors.append(f"missing required artifact {artifact.name}")
    if transcript.is_file() and metadata.get("transcript_sha256") != helpers._sha256_file(
        transcript
    ):
        errors.append("transcript hash changed")
    if stderr.is_file() and metadata.get("stderr_sha256") != helpers._sha256_file(stderr):
        errors.append("stderr hash changed")
    rendered = cohort.render_prompt(case, str(entry["condition"]))
    rendered_hash = hashlib.sha256(rendered.encode()).hexdigest()
    if metadata.get("rendered_prompt_sha256") != rendered_hash:
        errors.append("rendered prompt metadata binding changed")
    if prompt.is_file() and helpers._sha256_file(prompt) != rendered_hash:
        errors.append("rendered prompt artifact changed")
    frozen_provider = freeze["provider_bindings"][entry["agent"]]
    if metadata.get("provider_binding") != frozen_provider:
        errors.append("provider binding changed")
    if metadata.get("authorization_sha256") != helpers._sha256_file(
        _authorization_path(str(entry["agent"]))
    ):
        errors.append("provider authorization binding changed")
    if metadata.get("gold_manifest_sha256") != case["gold_sha256"]:
        errors.append("gold manifest binding changed")
    if metadata.get("model_identifier") != EXPECTED_MODELS[str(entry["agent"])]:
        errors.append("reported model identifier changed")
    if metadata.get("exit_code") != 0:
        errors.append("provider subprocess exit code changed")
    if metadata.get("terminal_state") != "completed" or metadata.get("timed_out") is not False:
        errors.append("provider did not complete a valid technical turn")
    surface = metadata.get("native_surface")
    expected_skill = entry["condition"] == "skill"
    if not isinstance(surface, dict) or surface.get("skill_installed") is not expected_skill:
        errors.append("native skill surface condition changed")
    else:
        expected_skill_hash = helpers._sha256_file(cohort.SKILL_PATH) if expected_skill else None
        if surface.get("semantic_file_count") != (1 if expected_skill else 0):
            errors.append("native skill semantic file count changed")
        if surface.get("skill_sha256") != expected_skill_hash:
            errors.append("native skill hash changed")
        if (
            not isinstance(surface.get("final_audit"), dict)
            or surface["final_audit"].get("mutated") is not False
        ):
            errors.append("native skill surface audit failed")
        if entry["agent"] == "claude":
            catalog_audit = surface.get("provider_catalog_audit")
            if not isinstance(catalog_audit, dict) or catalog_audit.get("valid") is not True:
                errors.append("Claude provider catalog audit failed")
    fixture = metadata.get("fixture")
    if not isinstance(fixture, dict):
        errors.append("dependency fixture audit is missing")
    else:
        if fixture.get("mutated") is not False:
            errors.append("dependency fixture mutated")
        if fixture.get("final_tree_sha256") != fixture.get("initial_tree_sha256"):
            errors.append("dependency fixture final hash changed")
    for artifact_name in (
        "git-status.txt",
        "workspace.patch",
        "workspace-files.json",
        "workspace-final",
    ):
        if not (run_root / artifact_name).exists():
            errors.append(f"missing required artifact {artifact_name}")
    if verdict.get("kind") != VERDICT_KIND or verdict.get("run_id") != run_id:
        errors.append("pre-score verdict identity changed")
    if verdict.get("metadata_sha256") != helpers._sha256_file(run_root / "metadata.json"):
        errors.append("pre-score verdict metadata binding changed")
    if verdict.get("transcript_sha256") != metadata.get("transcript_sha256"):
        errors.append("pre-score verdict transcript binding changed")
    if verdict.get("valid") is not True or verdict.get("errors") != []:
        errors.append("pre-score verdict is not valid")
    if errors:
        raise ExecutionError("; ".join(errors))
    return verdict


def _validate_prior_runs(
    plan: Mapping[str, Any],
    order: int,
    *,
    freeze: Mapping[str, Any] | None = None,
    primary: tuple[Mapping[str, Any], Mapping[str, Any]] | None = None,
) -> None:
    entries = plan.get("entries")
    if not isinstance(entries, list):
        raise ExecutionError("validated cohort lost its plan entries")
    for prior in entries[: order - 1]:
        if not isinstance(prior, dict):
            raise ExecutionError("plan entry shape changed")
        if freeze is None or primary is None:
            validate_run(str(prior["id"]))
        else:
            validate_run(str(prior["id"]), freeze=freeze, primary=primary)


def _execute_run_once(
    run_id: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
) -> dict[str, Any]:
    if timeout_seconds != DEFAULT_TIMEOUT_SECONDS:
        raise ExecutionError(f"official timeout must remain {DEFAULT_TIMEOUT_SECONDS} seconds")
    freeze = validate_freeze()
    manifest, plan, _ = _load_primary()
    entry = _plan_entry(plan, run_id)
    provider = str(entry["agent"])
    validate_authorization(provider)
    _validate_prior_runs(
        plan,
        int(entry["order"]),
        freeze=freeze,
        primary=(manifest, plan),
    )
    case = _case_index(manifest)[str(entry["case_id"])]
    run_root = _run_root(run_id)
    external_root = _external_run_root(run_id)
    if run_root.exists() or run_root.is_symlink():
        raise ExecutionError(f"refusing to overwrite existing run {run_root}")
    if external_root.exists() or external_root.is_symlink():
        raise ExecutionError(f"refusing to reuse external workspace {external_root}")

    source = (REPOSITORY_ROOT / str(case["source"])).resolve(strict=True)
    workspace = external_root / "workspace"
    scratch = external_root / "scratch"
    run_root.mkdir(parents=True)
    scratch.mkdir(parents=True)
    helpers._initialize_snapshot(source, workspace)
    initial_workspace_hash = helpers._snapshot_digest(workspace)
    if initial_workspace_hash != case["snapshot_sha256"]:
        raise ExecutionError("fresh workspace changed the frozen candidate snapshot")
    for relative in (".tracebook", ".eval-cache", "target"):
        if (workspace / relative).exists() or (workspace / relative).is_symlink():
            raise ExecutionError(f"fresh workspace unexpectedly contains {relative}")
    prompt = cohort.render_prompt(case, str(entry["condition"]))
    (run_root / "prompt.txt").write_text(prompt)
    environment = helpers._clean_agent_environment(scratch)
    environment.update(_toolchain_environment())
    fixture = _copy_fixture(case, scratch)
    environment.update(_fresh_environment(scratch))
    Path(environment["FRESH_TARGET"]).mkdir(exist_ok=True)
    skill = cohort.SKILL_PATH.read_bytes()
    command, surface, plugin_root = _native_surface(
        provider=provider,
        condition=str(entry["condition"]),
        workspace=workspace,
        run_root=external_root,
        scratch=scratch,
        environment=environment,
        skill=skill,
        skill_name=SKILL_NAME,
    )
    if provider == "claude":
        if plugin_root is None:
            raise ExecutionError("Claude plugin was not prepared")
        settings_path = run_root / "claude-settings.json"
        helpers._write_json(settings_path, _claude_settings(workspace, scratch, plugin_root))
        command = _claude_command(settings_path, plugin_root)

    transcript = run_root / f"{provider}.jsonl"
    stderr = run_root / f"{provider}.stderr"
    binary = _provider_binary(provider)
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "case_id": entry["case_id"],
        "provider": provider,
        "condition": entry["condition"],
        "repetition": entry["repetition"],
        "order": entry["order"],
        "started_at": helpers._utc_now(),
        "timeout_seconds": timeout_seconds,
        "primary_manifest_sha256": helpers._sha256_file(cohort.MANIFEST_PATH),
        "primary_plan_sha256": helpers._sha256_file(cohort.PLAN_PATH),
        "primary_freeze_sha256": helpers._sha256_file(cohort.FREEZE_PATH),
        "execution_freeze_sha256": helpers._sha256_file(FREEZE_PATH),
        "execution_runner_sha256": helpers._sha256_file(RUNNER_PATH),
        "snapshot_sha256": case["snapshot_sha256"],
        "initial_workspace_sha256": initial_workspace_hash,
        "gold_manifest_sha256": case["gold_sha256"],
        "rendered_prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "authorization_sha256": helpers._sha256_file(_authorization_path(provider)),
        "provider_binding": freeze["provider_bindings"][provider],
        "command": command,
        "requested_model": EXPECTED_MODELS[provider],
        "cli_version": helpers._binary_version(binary, environment),
        "binary_sha256": helpers._sha256_file(binary),
        "prompt_transport": "stdin",
        "fixture": fixture,
        "native_surface": surface,
        "skill_injected": entry["condition"] == "skill",
        "frozen_skill_sha256": helpers._sha256_file(cohort.SKILL_PATH),
    }
    helpers._write_json(run_root / "metadata.json", metadata)
    exit_code, timed_out, elapsed = helpers._run_subprocess(
        command,
        cwd=workspace,
        environment=environment,
        prompt_stdin=prompt,
        stdout_path=transcript,
        stderr_path=stderr,
        timeout_seconds=timeout_seconds,
    )
    surface["final_audit"] = delivery._audit_native_surface(
        agent=provider,
        condition=str(entry["condition"]),
        external_run_root=external_root,
        scratch=scratch,
        skill_content=skill,
        skill_name=SKILL_NAME,
    )
    helpers._write_workspace_evidence(workspace, run_root)
    _record_fixture_final(fixture)
    shutil.copytree(
        workspace,
        run_root / "workspace-final",
        symlinks=True,
        ignore=shutil.ignore_patterns(".git", "target", "node_modules", ".eval-cache"),
    )
    terminal = delivery._transcript_terminal_state(transcript, provider)
    if provider == "claude":
        surface["provider_catalog_audit"] = delivery._claude_catalog_audit(
            transcript,
            condition=str(entry["condition"]),
            skill_name=SKILL_NAME,
        )
    model, model_source = helpers._reported_model(transcript, provider)
    metadata.update(
        {
            "completed_at": helpers._utc_now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed,
            "terminal_state": terminal,
            "model_identifier": model,
            "model_identifier_source": model_source,
            "transcript_sha256": helpers._sha256_file(transcript),
            "stderr_sha256": helpers._sha256_file(stderr),
            "fixture": fixture,
            "native_surface": surface,
        }
    )
    helpers._write_json(run_root / "metadata.json", metadata)
    errors: list[str] = []
    if timed_out:
        errors.append("provider subprocess timed out")
    if terminal != "completed":
        errors.append(f"provider terminal state was {terminal!r}")
    if exit_code != 0:
        errors.append(f"provider exit code was {exit_code}")
    if model != EXPECTED_MODELS[provider]:
        errors.append(f"reported model {model!r} != {EXPECTED_MODELS[provider]!r}")
    if surface["final_audit"].get("mutated") is not False:
        errors.append("native skill surface mutated")
    if provider == "claude" and surface.get("provider_catalog_audit", {}).get("valid") is not True:
        errors.append("Claude provider catalog isolation failed")
    if fixture.get("mutated") is not False:
        errors.append("dependency fixture mutated")
    verdict = _technical_verdict(run_root, metadata, errors)
    if errors:
        raise ExecutionError(f"run {run_id!r} is technically invalid: " + "; ".join(errors))
    validate_run(run_id)
    shutil.rmtree(external_root)
    return {"metadata": metadata, "verdict": verdict}


def execute_run(run_id: str, timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS) -> dict[str, Any]:
    run_root = _run_root(run_id)
    external_root = _external_run_root(run_id)
    run_root_preexisting = run_root.exists() or run_root.is_symlink()
    external_root_preexisting = external_root.exists() or external_root.is_symlink()
    try:
        return _execute_run_once(run_id, timeout_seconds)
    except BaseException:
        if not run_root_preexisting and not external_root_preexisting:
            helpers._quarantine_interrupted_run(run_root, external_root)
        raise


def status() -> dict[str, Any]:
    cohort_summary = cohort.validate()
    _, plan, _ = _load_primary()
    completed = []
    for entry in plan["entries"]:
        run_id = str(entry["id"])
        if (_run_root(run_id) / "pre-score-verdict.json").is_file():
            try:
                validate_run(run_id)
            except ExecutionError:
                continue
            completed.append(run_id)
    authorizations = {provider: _authorization_path(provider).is_file() for provider in PROVIDERS}
    next_run = next(
        (entry["id"] for entry in plan["entries"] if entry["id"] not in completed), None
    )
    return {
        **cohort_summary,
        "execution_frozen": FREEZE_PATH.is_file(),
        "authorizations_recorded": authorizations,
        "completed_valid_runs": len(completed),
        "next_run": next_run,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("shakedown")
    commands.add_parser("freeze")
    commands.add_parser("validate")
    commands.add_parser("status")
    authorize = commands.add_parser("authorize")
    authorize.add_argument("--provider", required=True, choices=PROVIDERS)
    authorize.add_argument("--statement-file", required=True, type=Path)
    run = commands.add_parser("run")
    run.add_argument("--run-id", required=True)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    validate_run_parser = commands.add_parser("validate-run")
    validate_run_parser.add_argument("--run-id", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "shakedown":
            payload = shakedown()
        elif args.command == "freeze":
            payload = freeze_execution()
        elif args.command == "validate":
            payload = validate_freeze()
        elif args.command == "authorize":
            payload = record_authorization(args.provider, args.statement_file)
        elif args.command == "run":
            payload = execute_run(args.run_id, args.timeout_seconds)
        elif args.command == "validate-run":
            payload = validate_run(args.run_id)
        else:
            payload = status()
    except (ExecutionError, OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Execution experiment error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
