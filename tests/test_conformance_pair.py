import json
import subprocess
import sys

import pytest

from tracebook.conformance.cli import main


@pytest.fixture
def candidate(tmp_path):
    repo = tmp_path / "engine"
    repo.mkdir()

    def git(*args):
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()

    git("init", "-q")
    (repo / "version").write_text("from-pin")
    git("add", "version")
    git("-c", "user.name=Test", "-c", "user.email=test@example.org", "commit", "-qm", "engine")
    revision = git("rev-parse", "HEAD")
    git("tag", "release")
    (repo / "version").write_text("dirty")
    (repo / "untracked").touch()
    return repo, revision


def adapter_script(path, *, stale=False, diverge=False):
    path.write_text(f"""#!{sys.executable}
import os
from dataclasses import replace
from pathlib import Path
from tracebook.conformance import EngineMetadata, Outcome, ReferenceEngineAdapter, serve_stdio

class Candidate(ReferenceEngineAdapter):
    def __init__(self, config):
        super().__init__(config)
        env = os.environ
        self.metadata = EngineMetadata(
            env["TRACEBOOK_CANDIDATE_NAME"],
            (Path(env["TRACEBOOK_CANDIDATE_DIR"]) / "version").read_text(),
            "Python",
            "stale" if {stale!r} else env["TRACEBOOK_CANDIDATE_REVISION"],
            env["TRACEBOOK_CANDIDATE_SNAPSHOT"],
        )

    def apply(self, event, index):
        observation = super().apply(event, index)
        if {diverge!r}:
            return replace(observation, outcome=Outcome("rejected", "INVALID_ORDER"))
        return observation

raise SystemExit(serve_stdio(Candidate))
""")
    path.chmod(0o755)


def pair_args(candidate, adapter, workspace):
    repo, revision = candidate
    return [
        "qualify-pair",
        str(repo),
        "--candidate-name",
        "example/matcher",
        "--candidate-revision",
        revision,
        "--adapter",
        str(adapter),
        "--workspace",
        str(workspace),
    ]


def test_pair_builds_twice_from_committed_source_and_rejects_stale_packet(
    candidate, tmp_path, monkeypatch, capsys
):
    bundle = tmp_path / "adapter"
    bundle.mkdir()
    adapter_script(bundle / "program")
    build = bundle / "build"
    build.write_text(
        "#!/bin/sh\nset -eu\ntest ! -e built && touch built\n"
        'test ! -e "$TRACEBOOK_CACHE_DIR/stamp" && touch "$TRACEBOOK_CACHE_DIR/stamp"\n'
        'cp -p program "$TRACEBOOK_BUILD_DIR/adapter"\n'
    )
    build.chmod(0o755)
    workspace = tmp_path / "results"
    monkeypatch.setenv("GIT_DIR", str(candidate[0] / ".git"))
    args = pair_args(candidate, bundle, workspace) + ["--tag", "release"]
    assert main(args) == 0
    manifest = json.loads((workspace / "evidence-manifest.json").read_bytes())
    assert manifest["candidate"]["revision"] == candidate[1]
    assert len(manifest["runs"]) == 2
    first, second = manifest["runs"]
    assert first["files"] == second["files"]
    for run in manifest["runs"]:
        assert run["candidate_engine"]["version"] == "from-pin"
        assert not (workspace / run["candidate_root"] / "untracked").exists()
        assert (workspace / run["cache_root"] / "stamp").is_file()
        assert (workspace / run["adapter_root"] / "source" / "built").is_file()
    assert not (bundle / "built").exists()
    assert (candidate[0] / "version").read_text() == "dirty"

    assert main(args) == 2
    assert "already exists" in capsys.readouterr().err
    assert (
        main(
            [
                "evidence-verify",
                str(workspace / "evidence-plan.json"),
                "--expected-revision",
                "a" * 40,
                "--output",
                "review.json",
            ]
        )
        == 2
    )
    assert "evidence revision does not match requested revision" in capsys.readouterr().err
    assert not (workspace / "review.json").exists()


@pytest.mark.parametrize(
    "revision, message", [("a" * 40, "Git revision mismatch"), ("HEAD", "full Git commit")]
)
def test_pair_rejects_wrong_pin_before_launch(candidate, tmp_path, revision, message, capsys):
    adapter = tmp_path / "unused"
    adapter_script(adapter)
    workspace = tmp_path / "results"
    assert main(pair_args((candidate[0], revision), adapter, workspace) + ["--tag", "release"]) == 2
    assert message in capsys.readouterr().err
    assert not workspace.exists()


@pytest.mark.parametrize("stale, expected", [(True, 2), (False, 1)])
def test_pair_stops_on_first_failing_run(candidate, tmp_path, stale, expected, capsys):
    adapter = tmp_path / "executable"
    adapter_script(adapter, stale=stale, diverge=not stale)
    workspace = tmp_path / "results"
    assert main(pair_args(candidate, adapter, workspace)) == expected
    report = workspace / "runs/run-1/qualification/qualification.json"
    if stale:
        assert "candidate revision does not match task-pinned identity" in capsys.readouterr().err
        assert not report.exists()
    else:
        assert json.loads(report.read_bytes())["qualified"] is False
    assert not (workspace / "runs/run-2/qualification").exists()
    assert not (workspace / "evidence-manifest.json").exists()


def test_pair_stops_at_failed_build(candidate, tmp_path, capsys):
    adapter = tmp_path / "bundle"
    adapter.mkdir()
    build = adapter / "build"
    build.write_text("#!/bin/sh\nexit 3\n")
    build.chmod(0o755)
    workspace = tmp_path / "results"
    assert main(pair_args(candidate, adapter, workspace)) == 2
    assert "run-1 adapter build failed (exit 3)" in capsys.readouterr().err
    assert not (workspace / "evidence-manifest.json").exists()
