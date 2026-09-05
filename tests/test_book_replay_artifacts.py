"""L3 output failures must not truncate prior evidence or run invalid targets."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from tracebook.book_replay import BookReplayEvent, ReferenceBookReplayAdapter
from tracebook.book_replay import artifacts, cli

EVENT = BookReplayEvent("add", "TEST", 1, "BUY", "100", "2")
PAYLOAD = {"artifact_type": "tracebook.book-replay.report", "conformant": True}


def _trace(path: Path) -> None:
    path.write_text(json.dumps(EVENT.to_dict()) + "\n", encoding="utf-8")


def _command(command: str, trace: Path, report: Path, reduced: Path) -> list[str]:
    arguments = [command]
    if command != "campaign":
        arguments.append(str(trace))
    arguments.extend(("--output", str(report)))
    if command == "campaign":
        arguments.extend(("--reduced-events-output", str(reduced)))
    elif command == "minimize":
        arguments.extend(("--events-output", str(reduced)))
    arguments.extend(("--candidate-cmd", "unused-adapter"))
    return arguments


def _forbid_candidate(monkeypatch) -> None:
    def fail(_args):
        pytest.fail("candidate factory reached despite invalid output paths")

    monkeypatch.setattr(cli, "_candidate_factory", fail)


def _stub_result(monkeypatch, command: str) -> None:
    monkeypatch.setattr(cli, "_candidate_factory", lambda _args: object())
    report = SimpleNamespace(conformant=True, operational_failure=False, to_dict=lambda: PAYLOAD)
    minimization = SimpleNamespace(events=[EVENT], report=report, to_dict=lambda: PAYLOAD)
    result = {
        "run": report,
        "minimize": minimization,
        "campaign": SimpleNamespace(
            failure=SimpleNamespace(minimization=minimization), to_dict=lambda: PAYLOAD
        ),
    }[command]
    function = {
        "run": "run_book_replay",
        "minimize": "minimize_book_replay_failure",
        "campaign": "run_book_replay_campaign",
    }[command]
    monkeypatch.setattr(cli, function, lambda *_args, **_kwargs: result)


@pytest.mark.parametrize("command", ("run", "minimize", "campaign"))
@pytest.mark.parametrize("existing_kind", ("file", "directory", "symlink", "dangling-symlink"))
def test_cli_refuses_existing_report_before_candidate(
    tmp_path, monkeypatch, command, existing_kind
):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    if existing_kind == "file":
        report.write_bytes(b"prior evidence\n")
    elif existing_kind == "directory":
        report.mkdir()
    else:
        report.symlink_to(tmp_path / "prior.json")
        if existing_kind == "symlink":
            (tmp_path / "prior.json").write_bytes(b"prior evidence\n")
    before = set(tmp_path.iterdir())
    _forbid_candidate(monkeypatch)

    assert cli.main(_command(command, trace, report, reduced)) == 2

    assert set(tmp_path.iterdir()) == before
    if existing_kind == "file":
        assert report.read_bytes() == b"prior evidence\n"
    if existing_kind == "symlink":
        assert report.is_symlink()
        assert report.read_bytes() == b"prior evidence\n"
    if existing_kind == "dangling-symlink":
        assert report.is_symlink()
        assert not (tmp_path / "prior.json").exists()


@pytest.mark.parametrize("command", ("minimize", "campaign"))
def test_cli_reserves_conditional_reduced_output_before_candidate(tmp_path, monkeypatch, command):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    reduced.write_bytes(b"prior reduced trace\n")
    _forbid_candidate(monkeypatch)

    assert cli.main(_command(command, trace, report, reduced)) == 2

    assert reduced.read_bytes() == b"prior reduced trace\n"
    assert set(tmp_path.iterdir()) == {trace, reduced}


@pytest.mark.parametrize("command", ("minimize", "campaign"))
@pytest.mark.parametrize(
    "collision", ("same", "nested", "reverse-nested", "sidecar", "reverse-sidecar")
)
def test_cli_rejects_colliding_output_paths_before_candidate(
    tmp_path, monkeypatch, command, collision
):
    trace, report = tmp_path / "trace.jsonl", tmp_path / "report.json"
    _trace(trace)
    reduced = {
        "same": tmp_path / "unused" / ".." / "report.json",
        "nested": report / "reduced.jsonl",
        "reverse-nested": tmp_path,
        "sidecar": tmp_path / ".report.json.tracebook-in-progress",
        "reverse-sidecar": report,
    }[collision]
    if collision == "reverse-sidecar":
        report = tmp_path / ".report.json.tracebook-in-progress"
    _forbid_candidate(monkeypatch)

    assert cli.main(_command(command, trace, report, reduced)) == 2

    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.parametrize("command", ("run", "minimize", "campaign"))
def test_cli_rejects_non_directory_parent_before_candidate(tmp_path, monkeypatch, command):
    trace, parent = tmp_path / "trace.jsonl", tmp_path / "not-a-directory"
    _trace(trace)
    parent.write_bytes(b"keep this file\n")
    _forbid_candidate(monkeypatch)

    assert cli.main(_command(command, trace, parent / "report.json", tmp_path / "reduced")) == 2
    assert parent.read_bytes() == b"keep this file\n"


def test_cli_refuses_another_active_reservation_before_candidate(tmp_path, monkeypatch):
    trace, report = tmp_path / "trace.jsonl", tmp_path / "report.json"
    _trace(trace)
    _forbid_candidate(monkeypatch)
    with artifacts.reserve_outputs(str(report)):
        before = set(tmp_path.iterdir())
        assert cli.main(_command("run", trace, report, tmp_path / "unused")) == 2
        assert set(tmp_path.iterdir()) == before
    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.parametrize("command", ("run", "minimize", "campaign", "sample"))
def test_partial_stage_write_never_publishes_an_artifact(tmp_path, monkeypatch, command):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    if command != "sample":
        _stub_result(monkeypatch, command)

    def partial_write(descriptor, payload):
        os.write(descriptor, payload[:5])
        raise OSError("injected partial write failure")

    monkeypatch.setattr(artifacts, "_write_payload", partial_write)
    arguments = (
        ["sample", str(tmp_path)]
        if command == "sample"
        else _command(command, trace, report, reduced)
    )

    assert cli.main(arguments) == 2
    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.parametrize("command", ("minimize", "campaign"))
def test_second_publication_failure_rolls_back_first_file(tmp_path, monkeypatch, command):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    _stub_result(monkeypatch, command)
    link = os.link
    published = []

    def fail_second(source, target, **kwargs):
        published.append(target)
        if len(published) == 2:
            raise OSError("injected report publication failure")
        link(source, target, **kwargs)

    monkeypatch.setattr(artifacts.os, "link", fail_second)
    # The runtime supports descriptor-relative links; the wrapper remains the
    # same operation for this injected failure.
    monkeypatch.setattr(artifacts.os, "supports_dir_fd", os.supports_dir_fd | {fail_second})

    assert cli.main(_command(command, trace, report, reduced)) == 2
    assert published == [reduced.name, report.name]
    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.parametrize("command", ["run", "sample"])
def test_cli_symlink_loop_is_controlled_before_candidate(tmp_path, monkeypatch, capsys, command):
    trace = tmp_path / "trace.jsonl"
    _trace(trace)
    loop = tmp_path / "loop"
    loop.symlink_to(loop)
    _forbid_candidate(monkeypatch)
    arguments = (
        ["sample", str(loop)]
        if command == "sample"
        else _command("run", trace, loop / "report.json", tmp_path / "unused")
    )
    assert cli.main(arguments) == 2
    assert "Traceback" not in capsys.readouterr().err
    assert set(tmp_path.iterdir()) == {trace, loop}


def test_late_destination_creation_is_never_overwritten(tmp_path, monkeypatch):
    trace, report = tmp_path / "trace.jsonl", tmp_path / "report.json"
    _trace(trace)
    _stub_result(monkeypatch, "run")
    link = os.link

    def race_link(source, target, **kwargs):
        report.write_bytes(b"another writer's evidence\n")
        link(source, target, **kwargs)

    monkeypatch.setattr(artifacts.os, "link", race_link)
    monkeypatch.setattr(artifacts.os, "supports_dir_fd", os.supports_dir_fd | {race_link})

    assert cli.main(_command("run", trace, report, tmp_path / "unused")) == 2
    assert report.read_bytes() == b"another writer's evidence\n"
    assert set(tmp_path.iterdir()) == {trace, report}


def test_replaced_output_parent_cannot_redirect_publication(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    output_parent, moved_parent, other_parent = (
        tmp_path / name for name in ("out", "moved", "other")
    )
    _trace(trace)
    other_parent.mkdir()
    _stub_result(monkeypatch, "run")

    def replace_parent(*_args, **_kwargs):
        output_parent.rename(moved_parent)
        output_parent.symlink_to(other_parent, target_is_directory=True)
        return SimpleNamespace(to_dict=lambda: PAYLOAD, conformant=True, operational_failure=False)

    monkeypatch.setattr(cli, "run_book_replay", replace_parent)

    assert cli.main(_command("run", trace, output_parent / "report.json", tmp_path / "unused")) == 2
    assert list(other_parent.iterdir()) == []
    assert list(moved_parent.iterdir()) == []
    assert output_parent.is_symlink()


def test_retargeted_parent_alias_cannot_publish_at_an_invisible_old_path(tmp_path, monkeypatch):
    trace = tmp_path / "trace.jsonl"
    original, other, alias = (tmp_path / name for name in ("original", "other", "alias"))
    _trace(trace)
    original.mkdir()
    other.mkdir()
    alias.symlink_to(original, target_is_directory=True)
    _stub_result(monkeypatch, "run")

    def retarget_alias(*_args, **_kwargs):
        alias.unlink()
        alias.symlink_to(other, target_is_directory=True)
        return SimpleNamespace(to_dict=lambda: PAYLOAD, conformant=True, operational_failure=False)

    monkeypatch.setattr(cli, "run_book_replay", retarget_alias)

    assert cli.main(_command("run", trace, alias / "report.json", tmp_path / "unused")) == 2
    assert list(original.iterdir()) == []
    assert list(other.iterdir()) == []
    assert alias.resolve() == other


def test_replaced_reduced_file_invalidates_and_rolls_back_the_report(tmp_path, monkeypatch):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    _stub_result(monkeypatch, "minimize")
    link = os.link

    def replace_reduced_before_report(source, target, **kwargs):
        if target == report.name:
            reduced.unlink()
            reduced.write_bytes(b"another writer's replacement\n")
        link(source, target, **kwargs)

    monkeypatch.setattr(artifacts.os, "link", replace_reduced_before_report)
    monkeypatch.setattr(
        artifacts.os, "supports_dir_fd", os.supports_dir_fd | {replace_reduced_before_report}
    )

    assert cli.main(_command("minimize", trace, report, reduced)) == 2
    assert not report.exists()
    assert reduced.read_bytes() == b"another writer's replacement\n"
    assert set(tmp_path.iterdir()) == {trace, reduced}


def test_interruption_after_report_link_rolls_back_both_published_files(tmp_path, monkeypatch):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    _stub_result(monkeypatch, "minimize")
    link = os.link

    def interrupt_after_link(source, target, **kwargs):
        link(source, target, **kwargs)
        if target == report.name:
            raise KeyboardInterrupt

    monkeypatch.setattr(artifacts.os, "link", interrupt_after_link)
    monkeypatch.setattr(
        artifacts.os, "supports_dir_fd", os.supports_dir_fd | {interrupt_after_link}
    )

    with pytest.raises(KeyboardInterrupt):
        cli.main(_command("minimize", trace, report, reduced))
    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.skipif(os.geteuid() == 0, reason="root can deliberately bypass directory permissions")
def test_stage_entry_cannot_be_replaced_between_validation_and_link(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    link = os.link

    def replace_stage(source, target, **kwargs):
        # The old implementation allowed this unlink, then linked the foreign
        # replacement and left it at report.json after its failed validation.
        os.unlink(source, dir_fd=kwargs["src_dir_fd"])
        descriptor = os.open(
            source, os.O_WRONLY | os.O_CREAT | os.O_EXCL, dir_fd=kwargs["src_dir_fd"]
        )
        os.write(descriptor, b"foreign bytes")
        os.close(descriptor)
        link(source, target, **kwargs)

    monkeypatch.setattr(artifacts.os, "link", replace_stage)
    monkeypatch.setattr(artifacts.os, "supports_dir_fd", os.supports_dir_fd | {replace_stage})
    with artifacts.reserve_outputs(str(report)) as outputs:
        with pytest.raises(PermissionError):
            artifacts.publish_outputs([(outputs[str(report)], b"owned bytes")])
    assert not report.exists()
    assert list(tmp_path.iterdir()) == []


def test_replacing_stage_directory_path_cannot_change_published_inode(tmp_path, monkeypatch):
    report = tmp_path / "report.json"
    link = os.link
    with artifacts.reserve_outputs(str(report)) as outputs:
        output = outputs[str(report)]
        stage = tmp_path / output._stage_directory_name
        moved = tmp_path / "moved-stage"

        def replace_directory(source, target, **kwargs):
            # macOS requires write permission on a directory being renamed;
            # emulate a writer that can rename it, then restore its protection.
            stage.chmod(0o700)
            stage.rename(moved)
            moved.chmod(0o500)
            stage.mkdir()
            (stage / source).write_bytes(b"foreign replacement")
            link(source, target, **kwargs)

        monkeypatch.setattr(artifacts.os, "link", replace_directory)
        artifacts.publish_outputs([(output, b"owned bytes")])
        assert report.read_bytes() == b"owned bytes"
    assert (stage / output._stage_name).read_bytes() == b"foreign replacement"
    assert list(moved.iterdir()) == []


def test_cleanup_failure_still_closes_all_descriptors_and_other_sidecars(tmp_path, monkeypatch):
    reservation = artifacts.OutputReservation(tmp_path / "report.json")
    reservation.__enter__()
    descriptors = (
        reservation._stage_fd,
        reservation._lock_fd,
        reservation._stage_directory_fd,
        reservation._parent_fd,
    )
    unlink = os.unlink

    def fail_stage_cleanup(name, **kwargs):
        if name == reservation._stage_name:
            raise OSError("injected temporary file cleanup failure")
        unlink(name, **kwargs)

    monkeypatch.setattr(artifacts.os, "unlink", fail_stage_cleanup)

    with pytest.raises(OSError, match="cleanup failure"):
        reservation.close()
    for descriptor in descriptors:
        with pytest.raises(OSError):
            os.fstat(descriptor)
    assert {path.name for path in tmp_path.iterdir()} == {reservation._stage_directory_name}
    assert (tmp_path / reservation._stage_directory_name / reservation._stage_name).is_file()


def test_trace_serialization_failure_precedes_any_artifact_write(tmp_path, monkeypatch):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    _stub_result(monkeypatch, "minimize")

    def bad_event():
        raise ValueError("injected event serialization error")

    result = SimpleNamespace(
        to_dict=lambda: PAYLOAD,
        events=[EVENT, SimpleNamespace(to_dict=bad_event)],
    )
    monkeypatch.setattr(cli, "minimize_book_replay_failure", lambda *_args, **_kwargs: result)
    monkeypatch.setattr(
        artifacts,
        "_write_payload",
        lambda *_args: pytest.fail("writing before serialization finished"),
    )

    assert cli.main(_command("minimize", trace, report, reduced)) == 2
    assert set(tmp_path.iterdir()) == {trace}


@pytest.mark.parametrize("command", ("run", "minimize", "campaign"))
def test_successful_publication_preserves_existing_json_bytes(tmp_path, monkeypatch, command):
    trace, report, reduced = (
        tmp_path / name for name in ("trace.jsonl", "report.json", "reduced.jsonl")
    )
    _trace(trace)
    _stub_result(monkeypatch, command)

    assert cli.main(_command(command, trace, report, reduced)) == (0 if command == "run" else 1)

    assert report.read_bytes() == (json.dumps(PAYLOAD, sort_keys=True, indent=2) + "\n").encode()
    if command != "run":
        assert (
            reduced.read_bytes()
            == (json.dumps(EVENT.to_dict(), sort_keys=True, separators=(",", ":")) + "\n").encode()
        )
    assert not list(tmp_path.glob(".*"))


def test_passing_campaign_removes_unused_reduced_reservation(tmp_path, monkeypatch):
    report, reduced = tmp_path / "campaign.json", tmp_path / "reduced.jsonl"
    monkeypatch.setattr(cli, "_candidate_factory", lambda _args: ReferenceBookReplayAdapter)

    assert (
        cli.main(
            [
                "campaign",
                "--output",
                str(report),
                "--reduced-events-output",
                str(reduced),
                "--traces",
                "1",
                "--events-per-trace",
                "18",
                "--candidate-cmd",
                "unused",
            ]
        )
        == 0
    )

    assert json.loads(report.read_text())["conformant"] is True
    assert set(tmp_path.iterdir()) == {report}
