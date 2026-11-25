"""Build and qualify two independent captures of an explicitly pinned Git commit."""

from __future__ import annotations

import os
import shutil
import signal
import subprocess  # nosec B404
import tempfile
import threading
from contextlib import ExitStack
from pathlib import Path
from types import FrameType

from .evidence import (
    GIT_COMMIT_HASH,
    _copy_candidate,
    candidate_snapshot_id,
    load_evidence_plan,
    prepare_evidence_workspace,
    write_evidence_manifest,
)
from .exit_codes import exit_code_for_artifact
from .external import ExternalProcessAdapterFactory, _validate_timeout
from .model import ConformanceError, PinnedCandidateIdentity
from .qualification import _QualificationOutputReservation, run_qualification


def _checkout(repository: str, revision: str, tag: str | None, target: Path) -> None:
    """Fetch the pin into a fresh repository, independent of local working-tree edits."""
    if not GIT_COMMIT_HASH.fullmatch(revision):
        raise ConformanceError("--candidate-revision requires a full Git commit hash")
    local = Path(repository).expanduser()
    source = str(local.resolve()) if local.is_dir() else repository

    def git(*args: str) -> str:
        return subprocess.run(  # nosec B603 B607
            ["git", "-C", str(target), *args],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
            env=environment,
        ).stdout.strip()

    target.mkdir()
    try:
        # Git hooks export repository-local paths that would override -C in the capture.
        local_vars = subprocess.check_output(  # nosec B603 B607
            ["git", "rev-parse", "--local-env-vars"], text=True
        ).splitlines()
        environment = {key: value for key, value in os.environ.items() if key not in local_vars}
        git("init", "--quiet", *(["--object-format=sha256"] if len(revision) == 64 else []))
        git("remote", "add", "--", "origin", source)
        ref = revision
        if tag is not None:
            ref = f"refs/tags/{tag}"
            git("check-ref-format", ref)
        git("fetch", "--quiet", "--depth=1", "--no-tags", "origin", ref)
        observed = git("rev-parse", "--verify", "FETCH_HEAD^{commit}")
        if observed != revision:
            raise ConformanceError(
                f"Git revision mismatch: expected {revision}, fetched {observed}"
            )
        git("checkout", "--quiet", "--detach", revision)
        if (target / ".gitmodules").is_file():
            git("submodule", "update", "--init", "--recursive")
    except subprocess.CalledProcessError as exc:
        raise ConformanceError(
            f"could not capture Git revision {revision} (exit {exc.returncode})"
        ) from exc


def _require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ConformanceError(f"adapter requires an executable file at {path}")


def _build_adapter(
    directory: Path, environment: dict[str, str], timeout: float, run_id: str
) -> None:
    # A detached build needs a supervisor to relay cancellation to its descendants.
    supervise_group = (
        os.name == "posix"
        and threading.current_thread() is threading.main_thread()
        and signal.getsignal(signal.SIGTERM) == signal.SIG_DFL
    )

    def terminate(signum: int, _frame: FrameType | None) -> None:
        raise SystemExit(128 + signum)

    with ExitStack() as cleanup:
        if supervise_group:
            previous = signal.signal(signal.SIGTERM, terminate)
            cleanup.callback(signal.signal, signal.SIGTERM, previous)
        process = cleanup.enter_context(
            subprocess.Popen(  # nosec B603
                [str(directory / "build")],
                cwd=directory,
                env=environment,
                start_new_session=supervise_group,
            )
        )
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise ConformanceError(f"{run_id} adapter build timed out after {timeout:g}s") from exc
        finally:
            if process.returncode is None:
                if supervise_group:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                else:
                    process.kill()
    if status:
        raise ConformanceError(f"{run_id} adapter build failed (exit {status})")


def qualify_pair(
    repository: str,
    adapter: Path,
    workspace: Path,
    *,
    candidate_name: str,
    revision: str,
    tag: str | None = None,
    timeout: float = 10.0,
    build_timeout: float = 300.0,
) -> tuple[Path, int]:
    """Return the verified manifest, or the failing run's qualification and exit code."""
    _validate_timeout(timeout, "--timeout")
    _validate_timeout(build_timeout, "--build-timeout")
    revision = revision.lower()
    workspace = workspace.expanduser().absolute()
    if workspace.exists() or workspace.is_symlink():
        raise ConformanceError(f"evidence workspace already exists: {workspace}")
    adapter = adapter.expanduser().resolve(strict=True)
    if adapter.is_dir() and adapter in workspace.resolve().parents:
        raise ConformanceError("evidence workspace must be outside the adapter directory")
    with tempfile.TemporaryDirectory(prefix="tracebook-pair-") as temporary:
        staging = Path(temporary)
        if adapter.is_file():
            _require_executable(adapter)
            bundle = staging / "single-adapter"
            bundle.mkdir()
            shutil.copy2(adapter, bundle / "launch")
        else:
            bundle = adapter
        builder = (bundle / "build").is_file()
        _require_executable(bundle / ("build" if builder else "launch"))
        adapter_digest = candidate_snapshot_id(bundle)
        source = staging / "source"
        _checkout(repository, revision, tag, source)
        plan_path = prepare_evidence_workspace(
            source,
            workspace,
            candidate_name=candidate_name,
            candidate_revision=revision,
        )
        plan_path, plan = load_evidence_plan(plan_path)
        for run in plan["runs"]:
            copied = plan_path.parent / run["adapter_root"] / "source"
            _copy_candidate(bundle, copied)
            if candidate_snapshot_id(copied) != adapter_digest:
                raise ConformanceError("adapter source changed during capture")

    identity = PinnedCandidateIdentity.from_dict(plan["candidate"])
    params = plan["qualification"]
    for run in plan["runs"]:
        root = plan_path.parent
        adapter_dir = root / run["adapter_root"] / "source"
        build_dir = root / run["build_root"]
        cache_dir = root / run["cache_root"]
        environment = {
            **os.environ,
            "TRACEBOOK_CANDIDATE_DIR": str(root / run["candidate_root"]),
            "TRACEBOOK_CANDIDATE_NAME": identity.name,
            "TRACEBOOK_CANDIDATE_REVISION": identity.revision,
            "TRACEBOOK_CANDIDATE_SNAPSHOT": identity.snapshot_id,
            "TRACEBOOK_CANDIDATE_TAG": tag or "",
            "TRACEBOOK_BUILD_DIR": str(build_dir),
            "TRACEBOOK_CACHE_DIR": str(cache_dir),
            "XDG_CACHE_HOME": str(cache_dir),
        }
        with _QualificationOutputReservation(root / run["qualification_dir"]) as reservation:
            executable = adapter_dir / "launch"
            if builder:
                _build_adapter(adapter_dir, environment, build_timeout, run["run_id"])
                executable = build_dir / "adapter"
                if build_dir not in executable.resolve().parents:
                    raise ConformanceError(
                        f"{run['run_id']} adapter must be built inside its build directory"
                    )
            _require_executable(executable)
            factory = ExternalProcessAdapterFactory(
                [str(executable)],
                cwd=adapter_dir,
                env=environment,
                timeout_seconds=timeout,
                expected_identity=identity,
            )
            result = run_qualification(
                factory,
                profile=params["profile"],
                suite_version=params["suite_version"],
                seed=params["seed"],
                traces=params["traces"],
                events_per_trace=params["events_per_trace"],
                max_minimize_runs=params["max_minimize_runs"],
            )
            report = reservation.write(result)
        status = exit_code_for_artifact(result.to_dict())
        if status:
            return report, status
    return (
        write_evidence_manifest(
            plan_path, expected_revision=revision, expected_plan_id=plan["plan_id"]
        ),
        0,
    )
