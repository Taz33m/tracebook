"""CLI for the isolated L3 book-replay conformance surface."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from importlib import resources
from pathlib import Path
from typing import List, Optional

from .._version import __version__
from .compare import run_book_replay
from .external import ExternalBookReplayAdapterFactory
from .model import PROFILE_NAME, BookReplayConfig, BookReplayError, load_book_replay_events


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m tracebook.book_replay",
        description=(
            "Compare an external L3 order-book mirror without claiming matching-engine "
            "conformance."
        ),
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="Copy the bundled L3 replay trace.")
    sample.add_argument("destination", help="New directory for the profile trace.")

    run = commands.add_parser("run", help="Compare one L3 delta/probe JSONL trace.")
    run.add_argument("events")
    run.add_argument("--output")
    run.add_argument("--profile", choices=[PROFILE_NAME], default=PROFILE_NAME)
    run.add_argument("--timeout", type=float, default=5.0)
    candidate = run.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-cmd", help="Candidate command as one shell-style string.")
    candidate.add_argument(
        "--candidate",
        nargs=argparse.REMAINDER,
        help="Adapter command and arguments; this must be the final CLI option.",
    )
    return parser


def _candidate_factory(args) -> ExternalBookReplayAdapterFactory:
    command = (
        shlex.split(args.candidate_cmd) if args.candidate_cmd is not None else list(args.candidate)
    )
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise BookReplayError("--candidate requires a command")
    return ExternalBookReplayAdapterFactory(command, timeout_seconds=args.timeout)


def _copy_sample(destination: str) -> Path:
    root = Path(destination)
    target = root / f"{PROFILE_NAME}.jsonl"
    if root.exists() and not root.is_dir():
        raise BookReplayError("sample destination must be a directory")
    if target.exists():
        raise BookReplayError(f"sample trace already exists: {target}")
    root.mkdir(parents=True, exist_ok=True)
    source = resources.files("tracebook.book_replay.fixtures").joinpath(f"{PROFILE_NAME}.jsonl")
    target.write_bytes(source.read_bytes())
    return target


def _emit(payload: dict, output: Optional[str]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output is None:
        print(rendered, end="")
        return
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(rendered, encoding="utf-8")
    print(f"Report written: {path}")


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "sample":
            target = _copy_sample(args.destination)
            print(f"Book-replay trace copied to: {target}")
            print(f"Profile: {PROFILE_NAME}")
            return 0

        events_path = Path(args.events).expanduser().resolve()
        if args.output is not None and Path(args.output).expanduser().resolve() == events_path:
            raise BookReplayError("input and output paths must be distinct")
        report = run_book_replay(
            load_book_replay_events(str(events_path)),
            _candidate_factory(args),
            config=BookReplayConfig(args.profile),
            trace_name=str(events_path),
        )
        payload = report.to_dict()
        _emit(payload, args.output)
        if report.operational_failure:
            return 2
        return 0 if report.conformant else 1
    except (BookReplayError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
