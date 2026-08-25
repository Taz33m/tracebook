"""Reproducible runner for the agentic matching-engine qualification evaluation.

The tracked protocol and prompt are public. Candidate snapshots, evaluator gold
manifests, raw transcripts, and workspaces live under ``experiments/private/``
and are intentionally ignored.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import random
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = Path(__file__).resolve()
PROTOCOL_PATH = REPOSITORY_ROOT / "docs" / "agent-qualification-evaluation.md"
PROMPT_PATH = REPOSITORY_ROOT / "experiments" / "agent_qualification_prompt.txt"
PRIVATE_ROOT = REPOSITORY_ROOT / "experiments" / "private" / "agent-qualification-v1"
DEFAULT_MANIFEST_PATH = PRIVATE_ROOT / "cases.json"
DEFAULT_PLAN_PATH = PRIVATE_ROOT / "plan.json"
DEFAULT_RUNS_PATH = PRIVATE_ROOT / "runs"
CODEX_WORK_ROOT = PRIVATE_ROOT / "work" / "codex"
CLAUDE_WORK_ROOT = Path("/private/tmp/tracebook-agent-eval")
PROTOCOL_ID = "agent-qualification-v1"
RANDOMIZATION_SEED = 20260728
DEFAULT_REPETITIONS = 3
DEFAULT_TIMEOUT_SECONDS = 120 * 60

CODEX_BINARY = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
CODEX_AUTH_PATH = Path.home() / ".codex" / "auth.json"
CLAUDE_BINARY = Path("/Users/tazeemmahashin/.local/bin/claude")
DOTNET_BINARY_CANDIDATES = (
    Path("/usr/local/opt/dotnet@8/bin/dotnet"),
    Path("/usr/local/bin/dotnet"),
)

DOCS_TREATMENT = (
    "\n\nYou may evaluate tracebook-conformance==0.6.0 using only its public "
    "README and conformance documentation. Use it only if it fits the "
    "declared contract.\n"
)

NETWORK_DOMAINS = (
    "pypi.org",
    "files.pythonhosted.org",
    "registry.npmjs.org",
    "crates.io",
    "static.crates.io",
    "index.crates.io",
    "proxy.golang.org",
    "sum.golang.org",
    "github.com",
    "api.github.com",
    "codeload.github.com",
    "raw.githubusercontent.com",
    "*.githubusercontent.com",
)

SECRET_ENVIRONMENT_NAMES = {
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_AUTH_TOKEN",
    "OPENAI_API_KEY",
    "CODEX_API_KEY",
    "CODEX_ACCESS_TOKEN",
    "GH_TOKEN",
    "GITHUB_TOKEN",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "GOOGLE_APPLICATION_CREDENTIALS",
    "DATABASE_URL",
    "SSH_AUTH_SOCK",
}


class EvaluationError(RuntimeError):
    """Raised when a frozen evaluation precondition is not satisfied."""


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot_digest(root: Path) -> str:
    """Hash relative names, modes, and bytes without following symlinks."""

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(root).as_posix()
        if ".git" in path.relative_to(root).parts:
            continue
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
    """Return a stable inventory while rejecting unsafe tree entries."""

    inventory: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        relative_path = path.relative_to(root)
        if ".git" in relative_path.parts:
            continue
        relative = relative_path.as_posix()
        if path.is_symlink():
            raise EvaluationError(f"frozen fixture must not contain symlink {relative!r}")
        mode = path.stat().st_mode
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise EvaluationError(f"frozen fixture must contain only regular files: {relative!r}")
        inventory.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return inventory


def _private_path(raw_path: str, description: str) -> Path:
    path = (REPOSITORY_ROOT / raw_path).resolve()
    try:
        path.relative_to(PRIVATE_ROOT.resolve())
    except ValueError as exc:
        raise EvaluationError(f"{description} must stay inside {PRIVATE_ROOT}") from exc
    return path


def _dotnet_layout() -> tuple[Path | None, Path | None, Path | None]:
    for configured_binary in DOTNET_BINARY_CANDIDATES:
        if not configured_binary.exists():
            continue
        launcher_directory = configured_binary.resolve().parent
        libexec = launcher_directory.parent / "libexec"
        root = libexec if libexec.is_dir() else launcher_directory
        return configured_binary, launcher_directory, root
    return None, None, None


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"cannot read JSON from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvaluationError(f"{path} must contain one JSON object")
    return payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _case_index(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        raise EvaluationError("case manifest must contain a non-empty cases list")
    result: dict[str, dict[str, Any]] = {}
    for raw_case in cases:
        if not isinstance(raw_case, dict):
            raise EvaluationError("every case must be an object")
        case_id = raw_case.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise EvaluationError("every case must have a non-empty id")
        if case_id in result:
            raise EvaluationError(f"duplicate case id {case_id!r}")
        result[case_id] = raw_case
    return result


def _validate_frozen_tree(
    case: Mapping[str, Any],
    declaration: Any,
    *,
    field_name: str,
    expected_kind: str,
) -> dict[str, Any]:
    if not isinstance(declaration, dict):
        raise EvaluationError(f"case {case['id']!r} must declare {field_name!r} as an object")
    kind = declaration.get("kind")
    if kind == "none":
        if set(declaration) != {"kind"}:
            raise EvaluationError(
                f"case {case['id']!r} {field_name!r} kind 'none' " "must not contain fixture fields"
            )
        return {"kind": "none"}
    if kind != expected_kind:
        raise EvaluationError(
            f"case {case['id']!r} {field_name!r} kind must be " f"{expected_kind!r} or 'none'"
        )

    required = (
        "source",
        "manifest",
        "manifest_sha256",
        "tree_sha256",
    )
    missing = [name for name in required if not isinstance(declaration.get(name), str)]
    if missing:
        raise EvaluationError(
            f"case {case['id']!r} {field_name!r} is missing string fields: " + ", ".join(missing)
        )
    source = _private_path(str(declaration["source"]), f"case {case['id']!r} {field_name!r} source")
    fixture_manifest_path = _private_path(
        str(declaration["manifest"]),
        f"case {case['id']!r} {field_name!r} manifest",
    )
    if not source.is_dir():
        raise EvaluationError(f"case {case['id']!r} {field_name!r} source is unavailable: {source}")
    if not fixture_manifest_path.is_file():
        raise EvaluationError(
            f"case {case['id']!r} {field_name!r} manifest is unavailable: "
            f"{fixture_manifest_path}"
        )
    actual_manifest_hash = _sha256_file(fixture_manifest_path)
    if declaration["manifest_sha256"] != actual_manifest_hash:
        raise EvaluationError(
            f"case {case['id']!r} {field_name!r} manifest changed: "
            f"expected {declaration['manifest_sha256']}, got {actual_manifest_hash}"
        )

    fixture_manifest = _load_json(fixture_manifest_path)
    inventory = _tree_inventory(source)
    tree_hash = _snapshot_digest(source)
    total_bytes = sum(int(item["bytes"]) for item in inventory)
    bindings = {
        "kind": expected_kind,
        "case_id": case["id"],
        "snapshot_sha256": case["snapshot_sha256"],
        "tree_sha256": tree_hash,
        "file_count": len(inventory),
        "total_bytes": total_bytes,
        "files": inventory,
    }
    for name, expected in bindings.items():
        if fixture_manifest.get(name) != expected:
            raise EvaluationError(
                f"case {case['id']!r} {field_name!r} manifest binding " f"{name!r} changed"
            )
    if declaration["tree_sha256"] != tree_hash:
        raise EvaluationError(
            f"case {case['id']!r} {field_name!r} tree changed: "
            f"expected {declaration['tree_sha256']}, got {tree_hash}"
        )
    return {
        "kind": expected_kind,
        "source": source,
        "manifest": fixture_manifest_path,
        "manifest_sha256": actual_manifest_hash,
        "tree_sha256": tree_hash,
        "file_count": len(inventory),
        "total_bytes": total_bytes,
    }


def validate_manifest(manifest_path: Path) -> dict[str, Any]:
    manifest = _load_json(manifest_path)
    if manifest.get("protocol") != PROTOCOL_ID:
        raise EvaluationError(
            f"manifest protocol must be {PROTOCOL_ID!r}, got {manifest.get('protocol')!r}"
        )
    expected_protocol_hash = manifest.get("protocol_sha256")
    actual_protocol_hash = _sha256_file(PROTOCOL_PATH)
    if expected_protocol_hash != actual_protocol_hash:
        raise EvaluationError(
            "tracked protocol hash changed after case freeze: "
            f"expected {expected_protocol_hash}, got {actual_protocol_hash}"
        )
    expected_prompt_hash = manifest.get("prompt_sha256")
    actual_prompt_hash = _sha256_file(PROMPT_PATH)
    if expected_prompt_hash != actual_prompt_hash:
        raise EvaluationError(
            "tracked prompt hash changed after case freeze: "
            f"expected {expected_prompt_hash}, got {actual_prompt_hash}"
        )

    for case in _case_index(manifest).values():
        required = (
            "source",
            "snapshot_id",
            "revision",
            "commands",
            "declared_claim",
            "excluded_scope",
            "snapshot_sha256",
            "status",
        )
        missing = [field for field in required if field not in case]
        if missing:
            raise EvaluationError(f"case {case['id']!r} is missing fields: {', '.join(missing)}")
        source = (REPOSITORY_ROOT / str(case["source"])).resolve()
        if not source.is_dir():
            raise EvaluationError(f"case {case['id']!r} source is not a directory")
        if (source / ".git").exists():
            raise EvaluationError(f"case {case['id']!r} source must be origin-stripped")
        actual_snapshot_hash = _snapshot_digest(source)
        if case["snapshot_sha256"] != actual_snapshot_hash:
            raise EvaluationError(
                f"case {case['id']!r} snapshot changed: "
                f"expected {case['snapshot_sha256']}, got {actual_snapshot_hash}"
            )
        if case.get("status") == "ready":
            gold_manifest = case.get("gold_manifest")
            gold_hash = case.get("gold_sha256")
            if not isinstance(gold_manifest, str) or not isinstance(gold_hash, str):
                raise EvaluationError(
                    f"ready case {case['id']!r} must freeze gold_manifest and gold_sha256"
                )
            gold_path = (REPOSITORY_ROOT / gold_manifest).resolve()
            if not gold_path.is_file():
                raise EvaluationError(
                    f"case {case['id']!r} gold manifest is unavailable: {gold_path}"
                )
            actual_gold_hash = _sha256_file(gold_path)
            if gold_hash != actual_gold_hash:
                raise EvaluationError(
                    f"case {case['id']!r} gold manifest changed: "
                    f"expected {gold_hash}, got {actual_gold_hash}"
                )
            _validate_frozen_tree(
                case,
                case.get("dependency_cache"),
                field_name="dependency_cache",
                expected_kind="nuget-global-packages-v1",
            )
            _validate_frozen_tree(
                case,
                case.get("native_test_tool"),
                field_name="native_test_tool",
                expected_kind="xunit-inproc-v1",
            )
    return manifest


def render_prompt(case: Mapping[str, Any], condition: str) -> str:
    prompt = PROMPT_PATH.read_text()
    substitutions = {
        "case_id": str(case["id"]),
        "snapshot_id": str(case["snapshot_id"]),
        "revision": str(case["revision"]),
        "commands": str(case["commands"]),
        "declared_claim": str(case["declared_claim"]),
        "excluded_scope": str(case["excluded_scope"]),
    }
    for name, value in substitutions.items():
        prompt = prompt.replace("{{" + name + "}}", value)
    if "{{" in prompt or "}}" in prompt:
        raise EvaluationError("unresolved prompt placeholder")
    if condition == "baseline":
        return prompt
    if condition in {"docs", "skill"}:
        return prompt.rstrip() + DOCS_TREATMENT
    raise EvaluationError(f"unknown treatment condition {condition!r}")


def create_plan(
    manifest_path: Path,
    plan_path: Path,
    repetitions: int,
    conditions: Sequence[str],
) -> dict[str, Any]:
    if repetitions < 1:
        raise EvaluationError("repetitions must be positive")
    manifest = validate_manifest(manifest_path)
    cases = _case_index(manifest)
    entries = [
        {
            "run_id": f"{condition}__{case_id}__{agent}__r{repetition}",
            "condition": condition,
            "case_id": case_id,
            "agent": agent,
            "repetition": repetition,
        }
        for condition in conditions
        for case_id, case in cases.items()
        if case.get("status") == "ready"
        for agent in ("codex", "claude")
        for repetition in range(1, repetitions + 1)
    ]
    random.Random(RANDOMIZATION_SEED).shuffle(entries)
    plan = {
        "protocol": PROTOCOL_ID,
        "created_at": _utc_now(),
        "randomization_seed": RANDOMIZATION_SEED,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_sha256": _sha256_file(manifest_path),
        "runner_sha256": _sha256_file(RUNNER_PATH),
        "repetitions": repetitions,
        "conditions": list(conditions),
        "entries": entries,
    }
    _write_json(plan_path, plan)
    return plan


def _clean_agent_environment(scratch: Path) -> dict[str, str]:
    secret_fragments = ("KEY", "TOKEN", "SECRET", "PASSWORD", "CREDENTIAL")
    secret_prefixes = ("GH_", "GITHUB_", "AWS_", "GOOGLE_")
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in SECRET_ENVIRONMENT_NAMES
        and not any(fragment in name.upper() for fragment in secret_fragments)
        and not name.upper().startswith(secret_prefixes)
    }

    cache_paths = {
        "TMPDIR": scratch / "tmp",
        "PIP_CACHE_DIR": scratch / "pip-cache",
        "UV_CACHE_DIR": scratch / "uv-cache",
        "CARGO_HOME": scratch / "cargo-home",
        "NUGET_PACKAGES": scratch / "nuget",
        "DOTNET_CLI_HOME": scratch / "dotnet-home",
        "npm_config_cache": scratch / "npm-cache",
        "CODEX_SQLITE_HOME": scratch / "codex-state",
    }
    for name, path in cache_paths.items():
        path.mkdir(parents=True, exist_ok=True)
        environment[name] = str(path)
    environment["GIT_TERMINAL_PROMPT"] = "0"
    environment["CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR"] = "1"
    environment["CLAUDE_CODE_SKIP_PROMPT_HISTORY"] = "1"
    rustup_home = Path.home() / ".rustup"
    if rustup_home.is_dir():
        environment["RUSTUP_HOME"] = str(rustup_home)
    return environment


def _copy_frozen_tree(source: Path, target: Path, expected_hash: str) -> None:
    if target.exists():
        if not target.is_dir() or any(target.iterdir()):
            raise EvaluationError(f"fixture destination must be an empty directory: {target}")
        target.rmdir()
    try:
        subprocess.run(
            ("/bin/cp", "-cR", str(source), str(target)),
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
    except subprocess.CalledProcessError:
        shutil.copytree(source, target)
    actual_hash = _snapshot_digest(target)
    if actual_hash != expected_hash:
        raise EvaluationError(
            f"fixture copy changed in transit: expected {expected_hash}, got {actual_hash}"
        )


def _prepare_case_fixtures(case: Mapping[str, Any], scratch: Path) -> dict[str, dict[str, Any]]:
    declarations = (
        (
            "dependency_cache",
            "nuget-global-packages-v1",
            scratch / "nuget",
        ),
        (
            "native_test_tool",
            "xunit-inproc-v1",
            scratch / "native-test-tool",
        ),
    )
    prepared: dict[str, dict[str, Any]] = {}
    for field_name, expected_kind, target in declarations:
        fixture = _validate_frozen_tree(
            case,
            case.get(field_name),
            field_name=field_name,
            expected_kind=expected_kind,
        )
        if fixture["kind"] == "none":
            prepared[field_name] = {"kind": "none"}
            continue
        _copy_frozen_tree(
            Path(fixture["source"]),
            target,
            str(fixture["tree_sha256"]),
        )
        prepared[field_name] = {
            "kind": expected_kind,
            "manifest_sha256": fixture["manifest_sha256"],
            "initial_tree_sha256": fixture["tree_sha256"],
            "file_count": fixture["file_count"],
            "total_bytes": fixture["total_bytes"],
            "target": str(target),
        }

    shell_home = scratch / "shell-home"
    shell_home.mkdir(parents=True, exist_ok=True)
    nuget_config = shell_home / ".nuget" / "NuGet" / "NuGet.Config"
    nuget_config.parent.mkdir(parents=True, exist_ok=True)
    nuget_config.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        "<configuration>\n"
        "  <packageSources><clear /></packageSources>\n"
        "  <auditSources><clear /></auditSources>\n"
        "</configuration>\n"
    )
    return prepared


def _record_final_fixture_state(fixtures: dict[str, dict[str, Any]], scratch: Path) -> None:
    targets = {
        "dependency_cache": scratch / "nuget",
        "native_test_tool": scratch / "native-test-tool",
    }
    for field_name, fixture in fixtures.items():
        if fixture.get("kind") == "none":
            continue
        target = targets[field_name]
        if not target.is_dir():
            fixture["final_state"] = "missing"
            fixture["mutated"] = True
            continue
        try:
            final_hash = _snapshot_digest(target)
        except OSError as exc:
            fixture["final_state"] = f"unreadable:{type(exc).__name__}"
            fixture["mutated"] = True
            continue
        fixture["final_state"] = "present"
        fixture["final_tree_sha256"] = final_hash
        fixture["mutated"] = final_hash != fixture["initial_tree_sha256"]


def _dotnet_environment(scratch: Path) -> dict[str, str]:
    _, _, dotnet_root = _dotnet_layout()
    xdg_data_home = scratch / "xdg-data"
    migration_sentinel = xdg_data_home / "NuGet" / "Migrations" / "1"
    migration_sentinel.parent.mkdir(parents=True, exist_ok=True)
    migration_sentinel.touch(exist_ok=True)
    values = {
        "DOTNET_EnableDiagnostics": "0",
        "DOTNET_CLI_TELEMETRY_OPTOUT": "1",
        "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1",
        "DOTNET_CLI_USE_MSBUILD_SERVER": "0",
        "MSBUILDUSESERVER": "0",
        "NUGET_CERT_REVOCATION_MODE": "offline",
        "NuGetAudit": "false",
        "HOME": str(scratch / "shell-home"),
        # NuGet otherwise creates a named mutex before checking its migration
        # state. Claude's macOS sandbox intentionally blocks that global IPC,
        # so pre-seed NuGet's documented empty completion marker instead.
        "XDG_DATA_HOME": str(xdg_data_home),
    }
    if dotnet_root is not None:
        values["DOTNET_ROOT"] = str(dotnet_root)
    return values


def _claude_settings(workspace: Path, scratch: Path) -> dict[str, Any]:
    user_home = str(Path.home())
    workspace_rule_path = "//" + str(workspace).lstrip("/")
    user_home_rule_path = "//" + user_home.lstrip("/")
    cargo_bin = Path.home() / ".cargo" / "bin"
    rustup = Path.home() / ".rustup"
    _, dotnet_launcher_dir, _ = _dotnet_layout()
    path_parts = [
        *([str(scratch / "native-test-tool")] if (scratch / "native-test-tool").is_dir() else []),
        *([str(dotnet_launcher_dir)] if dotnet_launcher_dir is not None else []),
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    allow_read = [str(workspace)]
    if cargo_bin.is_dir():
        path_parts.append(str(cargo_bin))
        allow_read.append(str(cargo_bin))
    if rustup.is_dir():
        allow_read.append(str(rustup))
    for dotnet_read_path in (
        Path("/usr/local/bin/dotnet"),
        Path("/usr/local/share/dotnet"),
        Path("/usr/local/opt"),
        Path("/usr/local/Cellar"),
    ):
        if dotnet_read_path.exists():
            allow_read.append(str(dotnet_read_path))

    environment = {
        "PATH": ":".join(path_parts),
        "TMPDIR": str(scratch / "tmp"),
        "PIP_CACHE_DIR": str(scratch / "pip-cache"),
        "UV_CACHE_DIR": str(scratch / "uv-cache"),
        "CARGO_HOME": str(scratch / "cargo-home"),
        "NUGET_PACKAGES": str(scratch / "nuget"),
        "DOTNET_CLI_HOME": str(scratch / "dotnet-home"),
        "npm_config_cache": str(scratch / "npm-cache"),
        "GIT_TERMINAL_PROMPT": "0",
        "CLAUDE_BASH_MAINTAIN_PROJECT_WORKING_DIR": "1",
    }
    environment.update(_dotnet_environment(scratch))

    return {
        "env": environment,
        "permissions": {
            "defaultMode": "dontAsk",
            # Claude's OS sandbox applies only to Bash and its children. Its
            # native file tools need a second, absolute-path allowlist. In
            # Claude rules ``//`` (represented by the leading slash added
            # above plus the path's own slash) anchors at the filesystem root;
            # a single slash would anchor at the settings-file directory.
            # Edit rules cover Edit, Write, and NotebookEdit.
            "allow": [
                f"Read({workspace_rule_path}/**)",
                f"Edit({workspace_rule_path}/**)",
                "Bash",
                "WebSearch",
                "WebFetch",
            ],
            "deny": [
                f"Read({user_home_rule_path}/**)",
                f"Edit({user_home_rule_path}/**)",
                f"Edit({workspace_rule_path}/.git/**)",
                "Bash(git commit *)",
                "Bash(git push *)",
                "Bash(gh *)",
                "Bash(ssh *)",
                "Bash(scp *)",
                "Bash(rsync *)",
            ],
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "filesystem": {
                "denyRead": [user_home, "/Volumes"],
                "allowRead": allow_read,
                "denyWrite": [
                    user_home,
                    "/Volumes",
                    str(workspace / ".git"),
                    str(scratch / "native-test-tool"),
                ],
                "allowWrite": [str(workspace), str(scratch)],
            },
            "network": {"allowedDomains": list(NETWORK_DOMAINS)},
        },
    }


def _codex_permission_profile(workspace: Path, scratch: Path) -> str:
    del workspace  # Runtime ``:workspace_roots`` supplies the exact workspace.
    filesystem = [
        '":root"="deny"',
        '":minimal"="read"',
        f'{json.dumps(str(scratch))}="write"',
        f'{json.dumps(str(scratch / "native-test-tool"))}="read"',
        '":workspace_roots"={"."="write",".git"="read"}',
    ]
    for read_path in (
        Path("/Library/Developer/CommandLineTools"),
        Path("/usr/local/bin/dotnet"),
        Path("/usr/local/share/dotnet"),
        Path("/usr/local/opt"),
        Path("/usr/local/Cellar"),
        Path.home() / ".cargo" / "bin",
        Path.home() / ".rustup",
    ):
        if read_path.exists():
            filesystem.append(f'{json.dumps(str(read_path))}="read"')
    domains = ",".join(f'{json.dumps(domain)}="allow"' for domain in NETWORK_DOMAINS)
    return (
        "permissions.agent-eval={"
        f"filesystem={{{','.join(filesystem)}}},"
        'network={enabled=true,mode="limited",allow_upstream_proxy=false,'
        f"domains={{{domains}}}}}"
        "}"
    )


def _codex_shell_environment(scratch: Path) -> dict[str, str]:
    _, dotnet_launcher_dir, _ = _dotnet_layout()
    path_parts = [
        *([str(scratch / "native-test-tool")] if (scratch / "native-test-tool").is_dir() else []),
        *([str(dotnet_launcher_dir)] if dotnet_launcher_dir is not None else []),
        "/Library/Developer/CommandLineTools/usr/bin",
        "/usr/local/bin",
        "/opt/homebrew/bin",
        "/usr/bin",
        "/bin",
        "/usr/sbin",
        "/sbin",
    ]
    cargo_bin = Path.home() / ".cargo" / "bin"
    if cargo_bin.is_dir():
        path_parts.append(str(cargo_bin))
    values = {
        "PATH": ":".join(path_parts),
        "TMPDIR": str(scratch / "tmp"),
        "PIP_CACHE_DIR": str(scratch / "pip-cache"),
        "UV_CACHE_DIR": str(scratch / "uv-cache"),
        "CARGO_HOME": str(scratch / "cargo-home"),
        "NUGET_PACKAGES": str(scratch / "nuget"),
        "DOTNET_CLI_HOME": str(scratch / "dotnet-home"),
        "npm_config_cache": str(scratch / "npm-cache"),
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_OPTIONAL_LOCKS": "0",
    }
    rustup_home = Path.home() / ".rustup"
    if rustup_home.is_dir():
        values["RUSTUP_HOME"] = str(rustup_home)
    values.update(_dotnet_environment(scratch))
    return values


def _codex_command(workspace: Path, scratch: Path) -> list[str]:
    permission_profile = _codex_permission_profile(workspace, scratch)
    shell_environment = ",".join(
        f"{json.dumps(name)}={json.dumps(value)}"
        for name, value in _codex_shell_environment(scratch).items()
    )
    secret_excludes = (
        'shell_environment_policy.exclude=["*KEY*","*TOKEN*","*SECRET*",'
        '"*PASSWORD*","*CREDENTIAL*","GH_*","GITHUB_*","AWS_*","GOOGLE_*",'
        '"DATABASE_URL","CODEX_HOME","SSH_AUTH_SOCK"]'
    )
    return [
        str(CODEX_BINARY),
        "--ask-for-approval",
        "never",
        "--search",
        "--disable",
        "apps",
        "--disable",
        "plugins",
        "--disable",
        "remote_plugin",
        "--disable",
        "skill_search",
        "--disable",
        "tool_suggest",
        "--disable",
        "plugin_sharing",
        "--disable",
        "workspace_dependencies",
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "--strict-config",
        "--skip-git-repo-check",
        "--json",
        "--color",
        "never",
        "--model",
        "gpt-5.6-sol",
        "--cd",
        str(workspace),
        "-c",
        "features.network_proxy.enabled=true",
        "-c",
        'default_permissions="agent-eval"',
        "-c",
        permission_profile,
        "-c",
        "project_doc_max_bytes=0",
        "-c",
        "project_doc_fallback_filenames=[]",
        "-c",
        'model_reasoning_effort="high"',
        "-c",
        "allow_login_shell=false",
        "-c",
        'history.persistence="none"',
        "-c",
        'shell_environment_policy.inherit="core"',
        "-c",
        f"shell_environment_policy.set={{{shell_environment}}}",
        "-c",
        secret_excludes,
        "-c",
        "features.apps=false",
        "-c",
        "features.remote_plugin=false",
        "-c",
        "features.multi_agent=false",
        "-c",
        "features.memories=false",
        "-c",
        "features.hooks=false",
        "-c",
        "features.skill_mcp_dependency_install=false",
        "-c",
        "features.goals=false",
        "-c",
        "agents.enabled=false",
        "-c",
        "check_for_update_on_startup=false",
        "-c",
        "feedback.enabled=false",
        "-c",
        "analytics.enabled=false",
        "-c",
        'personality="none"',
        "-",
    ]


def _prepare_codex_environment(
    external_run_root: Path,
    scratch: Path,
    environment: dict[str, str],
) -> Path:
    if not CODEX_AUTH_PATH.is_file():
        raise EvaluationError(f"Codex auth is unavailable at {CODEX_AUTH_PATH}")
    codex_home = external_run_root / "codex-home"
    codex_home.mkdir()
    codex_home.chmod(0o700)
    codex_state = codex_home / "state"
    codex_state.mkdir()
    (codex_home / "auth.json").symlink_to(CODEX_AUTH_PATH)
    shell_home = scratch / "shell-home"
    shell_home.mkdir(exist_ok=True)
    environment.update(_codex_shell_environment(scratch))
    environment["CODEX_HOME"] = str(codex_home)
    environment["CODEX_SQLITE_HOME"] = str(codex_state)
    return codex_home


def _claude_command(settings_path: Path) -> list[str]:
    return [
        str(CLAUDE_BINARY),
        "-p",
        "--safe-mode",
        "--no-session-persistence",
        "--no-chrome",
        "--disable-slash-commands",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--mcp-config",
        '{"mcpServers":{}}',
        "--settings",
        str(settings_path),
        "--permission-mode",
        "dontAsk",
        "--tools",
        "Read,Glob,Grep,Edit,Write,Bash,WebSearch,WebFetch",
        "--model",
        "opus",
        "--effort",
        "high",
        "--max-budget-usd",
        "50",
        "--output-format",
        "stream-json",
        "--verbose",
    ]


def _run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
    prompt_stdin: str | None,
    stdout_path: Path,
    stderr_path: Path,
    timeout_seconds: int,
) -> tuple[int, bool, float]:
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            list(command),
            cwd=cwd,
            env=dict(environment),
            stdin=subprocess.PIPE if prompt_stdin is not None else subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            process.communicate(
                prompt_stdin.encode() if prompt_stdin is not None else None,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    elapsed = time.monotonic() - started
    return process.returncode, timed_out, elapsed


def _run_checked(command: Sequence[str], cwd: Path) -> str:
    completed = subprocess.run(
        list(command),
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return completed.stdout


def _binary_version(binary: Path, environment: Mapping[str, str]) -> str:
    completed = subprocess.run(
        (str(binary), "--version"),
        env=dict(environment),
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        timeout=15,
    )
    return completed.stdout.strip()


def _reported_model(transcript_path: Path, agent: str) -> tuple[str, str]:
    if agent == "codex":
        # Codex's JSONL protocol does not currently repeat the selected model;
        # the CLI's exact, non-alias model argument is the authoritative value.
        return "gpt-5.6-sol", "command"
    with transcript_path.open() as handle:
        for line in handle:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(event, dict):
                continue
            model = event.get("model")
            if isinstance(model, str) and model:
                return model, "transcript"
            message = event.get("message")
            if isinstance(message, dict):
                model = message.get("model")
                if isinstance(model, str) and model:
                    return model, "transcript"
    return "opus", "command-fallback"


def _initialize_snapshot(source: Path, workspace: Path) -> None:
    shutil.copytree(source, workspace, symlinks=True)
    _run_checked(("git", "init", "-q"), workspace)
    _run_checked(("git", "config", "user.name", "Tracebook evaluator"), workspace)
    _run_checked(("git", "config", "user.email", "evaluator@tracebook.invalid"), workspace)
    _run_checked(("git", "config", "commit.gpgsign", "false"), workspace)
    _run_checked(("git", "add", "-A"), workspace)
    _run_checked(("git", "commit", "-q", "-m", "Frozen candidate snapshot"), workspace)


def _workspace_inventory(workspace: Path) -> list[dict[str, Any]]:
    inventory: list[dict[str, Any]] = []
    excluded_parts = {".git", ".eval-cache", "bin", "obj", "target", "node_modules"}
    for path in sorted(workspace.rglob("*"), key=lambda value: value.as_posix()):
        relative = path.relative_to(workspace)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.is_file() and not path.is_symlink():
            inventory.append(
                {
                    "path": relative.as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": _sha256_file(path),
                }
            )
    return inventory


def _write_workspace_evidence(workspace: Path, run_root: Path) -> None:
    """Capture tracked, untracked, and ignored candidate changes without mutating its index."""

    inventory = _workspace_inventory(workspace)
    temporary_index = run_root / ".workspace-evidence.index"
    environment = dict(os.environ)
    environment["GIT_INDEX_FILE"] = str(temporary_index)

    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ("git", *arguments),
            cwd=workspace,
            env=environment,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        return completed.stdout

    try:
        run_git("read-tree", "HEAD")
        run_git("add", "-u", "--", ".")
        paths = ["./" + str(item["path"]) for item in inventory]
        for start in range(0, len(paths), 200):
            run_git("add", "-f", "--", *paths[start : start + 200])
        (run_root / "git-status.txt").write_text(run_git("status", "--short"))
        (run_root / "workspace.patch").write_text(run_git("diff", "--cached", "--binary", "HEAD"))
    finally:
        temporary_index.unlink(missing_ok=True)
    _write_json(run_root / "workspace-files.json", {"files": inventory})


def _quarantine_interrupted_run(run_root: Path, external_run_root: Path) -> Path | None:
    """Preserve incomplete evidence under ``interruptions`` and unblock a clean retry."""

    if external_run_root.is_symlink():
        external_run_root.unlink()
    elif external_run_root.exists():
        shutil.rmtree(external_run_root)
    if not run_root.exists() and not run_root.is_symlink():
        return None
    if not run_root.is_symlink() and (run_root / "pre-score-verdict.json").is_file():
        return None
    interruptions = run_root.parent / "interruptions"
    interruptions.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    target = interruptions / f"{run_root.name}.{stamp}.{os.getpid()}"
    run_root.replace(target)
    return target


def _execute_run_once(
    manifest_path: Path,
    *,
    case_id: str,
    agent: str,
    repetition: int,
    condition: str,
    runs_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    manifest = validate_manifest(manifest_path)
    cases = _case_index(manifest)
    try:
        case = cases[case_id]
    except KeyError as exc:
        raise EvaluationError(f"unknown case {case_id!r}") from exc
    if case.get("status") != "ready":
        raise EvaluationError(f"case {case_id!r} is not ready")
    if agent not in {"codex", "claude"}:
        raise EvaluationError(f"unknown agent {agent!r}")
    if repetition < 1:
        raise EvaluationError("repetition must be positive")
    if condition == "skill":
        raise EvaluationError("skill-assisted runs remain locked until baseline grading")

    run_id = f"{condition}__{case_id}__{agent}__r{repetition}"
    run_root = runs_path.resolve() / run_id
    if run_root.exists():
        raise EvaluationError(f"refusing to overwrite existing run {run_root}")

    source = (REPOSITORY_ROOT / str(case["source"])).resolve()
    work_root = CODEX_WORK_ROOT if agent == "codex" else CLAUDE_WORK_ROOT
    external_run_root = work_root / run_id
    if external_run_root.exists():
        raise EvaluationError(f"refusing to reuse external workspace {external_run_root}")
    workspace = external_run_root / "workspace"
    scratch = external_run_root / "scratch"
    transcript_path = run_root / f"{agent}.jsonl"
    stderr_path = run_root / f"{agent}.stderr"
    run_root.mkdir(parents=True)
    scratch.mkdir(parents=True)
    _initialize_snapshot(source, workspace)

    prompt = render_prompt(case, condition)
    (run_root / "prompt.txt").write_text(prompt)
    environment = _clean_agent_environment(scratch)
    fixtures = _prepare_case_fixtures(case, scratch)
    settings_path: Path | None = None
    if agent == "codex":
        if not CODEX_BINARY.is_file():
            raise EvaluationError(f"Codex binary is unavailable at {CODEX_BINARY}")
        _prepare_codex_environment(external_run_root, scratch, environment)
        command = _codex_command(workspace, scratch)
        prompt_stdin = prompt
    else:
        if not CLAUDE_BINARY.is_file():
            raise EvaluationError(f"Claude binary is unavailable at {CLAUDE_BINARY}")
        settings_path = run_root / "claude-settings.json"
        _write_json(settings_path, _claude_settings(workspace, scratch))
        command = _claude_command(settings_path)
        prompt_stdin = prompt

    cli_version = _binary_version(CODEX_BINARY if agent == "codex" else CLAUDE_BINARY, environment)
    started_at = _utc_now()
    metadata: dict[str, Any] = {
        "protocol": PROTOCOL_ID,
        "run_id": run_id,
        "condition": condition,
        "case_id": case_id,
        "agent": agent,
        "repetition": repetition,
        "started_at": started_at,
        "timeout_seconds": timeout_seconds,
        "manifest_sha256": _sha256_file(manifest_path),
        "gold_manifest_sha256": str(case["gold_sha256"]),
        "runner_sha256": _sha256_file(RUNNER_PATH),
        "protocol_sha256": _sha256_file(PROTOCOL_PATH),
        "prompt_template_sha256": _sha256_file(PROMPT_PATH),
        "rendered_prompt_sha256": _sha256_bytes(prompt.encode()),
        "snapshot_sha256": _snapshot_digest(source),
        "command": command,
        "prompt_transport": "stdin",
        "cli_version": cli_version,
        "requested_model": "gpt-5.6-sol" if agent == "codex" else "opus",
        "removed_environment_names": sorted(SECRET_ENVIRONMENT_NAMES),
        "fixtures": fixtures,
    }
    _write_json(run_root / "metadata.json", metadata)

    exit_code, timed_out, elapsed_seconds = _run_subprocess(
        command,
        cwd=workspace,
        environment=environment,
        prompt_stdin=prompt_stdin,
        stdout_path=transcript_path,
        stderr_path=stderr_path,
        timeout_seconds=timeout_seconds,
    )
    _write_workspace_evidence(workspace, run_root)
    _record_final_fixture_state(fixtures, scratch)
    shutil.copytree(
        workspace,
        run_root / "workspace-final",
        symlinks=True,
        ignore=shutil.ignore_patterns(
            ".git", "bin", "obj", "target", "node_modules", ".eval-cache"
        ),
    )

    metadata.update(
        {
            "completed_at": _utc_now(),
            "exit_code": exit_code,
            "timed_out": timed_out,
            "elapsed_seconds": elapsed_seconds,
            "transcript_sha256": _sha256_file(transcript_path),
            "stderr_sha256": _sha256_file(stderr_path),
            "fixtures": fixtures,
        }
    )
    model_identifier, model_identifier_source = _reported_model(transcript_path, agent)
    metadata["model_identifier"] = model_identifier
    metadata["model_identifier_source"] = model_identifier_source
    _write_json(run_root / "metadata.json", metadata)
    shutil.rmtree(external_run_root)
    return metadata


def execute_run(
    manifest_path: Path,
    *,
    case_id: str,
    agent: str,
    repetition: int,
    condition: str,
    runs_path: Path,
    timeout_seconds: int,
) -> dict[str, Any]:
    run_id = f"{condition}__{case_id}__{agent}__r{repetition}"
    run_root = runs_path.resolve() / run_id
    work_root = CODEX_WORK_ROOT if agent == "codex" else CLAUDE_WORK_ROOT
    external_run_root = work_root / run_id
    run_root_preexisting = run_root.exists() or run_root.is_symlink()
    external_root_preexisting = external_run_root.exists() or external_run_root.is_symlink()
    try:
        return _execute_run_once(
            manifest_path,
            case_id=case_id,
            agent=agent,
            repetition=repetition,
            condition=condition,
            runs_path=runs_path,
            timeout_seconds=timeout_seconds,
        )
    except BaseException:
        if not run_root_preexisting and not external_root_preexisting:
            _quarantine_interrupted_run(run_root, external_run_root)
        raise


def _parse_conditions(raw: str) -> list[str]:
    conditions = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [item for item in conditions if item not in {"baseline", "docs", "skill"}]
    if unknown:
        raise EvaluationError(f"unknown conditions: {', '.join(unknown)}")
    if not conditions:
        raise EvaluationError("at least one condition is required")
    return conditions


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate", help="validate frozen inputs")
    validate.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)

    plan = subparsers.add_parser("plan", help="create a deterministic run order")
    plan.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    plan.add_argument("--output", type=Path, default=DEFAULT_PLAN_PATH)
    plan.add_argument("--repetitions", type=int, default=DEFAULT_REPETITIONS)
    plan.add_argument("--conditions", default="baseline")

    run = subparsers.add_parser("run", help="execute one isolated run")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    run.add_argument("--case", required=True)
    run.add_argument("--agent", choices=("codex", "claude"), required=True)
    run.add_argument("--repetition", type=int, required=True)
    run.add_argument("--condition", choices=("baseline", "docs", "skill"), default="baseline")
    run.add_argument("--runs-path", type=Path, default=DEFAULT_RUNS_PATH)
    run.add_argument("--timeout-seconds", type=int, default=DEFAULT_TIMEOUT_SECONDS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            manifest = validate_manifest(arguments.manifest)
            print(
                json.dumps(
                    {
                        "protocol": manifest["protocol"],
                        "cases": sorted(_case_index(manifest)),
                        "status": "valid",
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "plan":
            plan = create_plan(
                arguments.manifest,
                arguments.output,
                arguments.repetitions,
                _parse_conditions(arguments.conditions),
            )
            print(
                json.dumps(
                    {
                        "entries": len(plan["entries"]),
                        "output": str(arguments.output),
                        "status": "created",
                    },
                    sort_keys=True,
                )
            )
        elif arguments.command == "run":
            metadata = execute_run(
                arguments.manifest,
                case_id=arguments.case,
                agent=arguments.agent,
                repetition=arguments.repetition,
                condition=arguments.condition,
                runs_path=arguments.runs_path,
                timeout_seconds=arguments.timeout_seconds,
            )
            print(json.dumps(metadata, sort_keys=True))
        else:  # pragma: no cover - argparse enforces the command set
            parser.error(f"unsupported command {arguments.command!r}")
    except (EvaluationError, OSError, subprocess.CalledProcessError) as exc:
        print(f"agent qualification evaluation failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
