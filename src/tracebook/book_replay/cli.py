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
from .artifacts import OutputReservation, publish_outputs, reserve_outputs
from .campaign import run_book_replay_campaign
from .compare import run_book_replay
from .external import ExternalBookReplayAdapterFactory
from .minimize import minimize_book_replay_failure
from .model import PROFILE_NAME, BookReplayConfig, BookReplayError, load_book_replay_events


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=5.0)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate-cmd", help="Candidate command as one shell-style string.")
    candidate.add_argument(
        "--candidate",
        nargs=argparse.REMAINDER,
        help="Adapter command and arguments; this must be the final CLI option.",
    )


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
    _add_candidate_arguments(run)

    minimize = commands.add_parser(
        "minimize",
        help="Reduce one divergent L3 trace while preserving its failure category.",
    )
    minimize.add_argument("events")
    minimize.add_argument("--events-output", required=True)
    minimize.add_argument("--output")
    minimize.add_argument("--max-runs", type=int, default=100)
    minimize.add_argument("--profile", choices=[PROFILE_NAME], default=PROFILE_NAME)
    _add_candidate_arguments(minimize)

    campaign = commands.add_parser(
        "campaign",
        help="Generate deterministic L3 traces and minimize the first divergence.",
    )
    campaign.add_argument("--output", required=True)
    campaign.add_argument("--reduced-events-output")
    campaign.add_argument("--seed", type=int, default=1337)
    campaign.add_argument("--traces", type=int, default=25)
    campaign.add_argument("--events-per-trace", type=int, default=100)
    campaign.add_argument("--max-minimize-runs", type=int, default=100)
    _add_candidate_arguments(campaign)
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
    source = resources.files("tracebook.book_replay.fixtures").joinpath(f"{PROFILE_NAME}.jsonl")
    with reserve_outputs(str(target)) as outputs:
        publish_outputs([(outputs[str(target)], source.read_bytes())])
    return target


def _emit(
    payload: dict,
    output: Optional[str],
    outputs: dict[str, OutputReservation],
    *,
    events=None,
    events_output: Optional[str] = None,
) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    files = []
    if events_output is not None and events is not None:
        event_bytes = "".join(
            json.dumps(event.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\n"
            for event in events
        ).encode("utf-8")
        files.append((outputs[events_output], event_bytes))
    if output is not None:
        files.append((outputs[output], rendered.encode("utf-8")))
    # Publish the report last so a successful report never precedes its trace.
    publish_outputs(files)
    if output is None:
        print(rendered, end="")
    else:
        print(f"Report written: {Path(output)}")


def _require_distinct_paths(*paths: Optional[str]) -> None:
    resolved = [Path(path).expanduser().resolve() for path in paths if path is not None]
    if len(resolved) != len(set(resolved)) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise BookReplayError(
            "input and output paths must be distinct and cannot contain one another"
        )


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "sample":
            target = _copy_sample(args.destination)
            print(f"Book-replay trace copied to: {target}")
            print(f"Profile: {PROFILE_NAME}")
            return 0

        if args.command == "campaign":
            _require_distinct_paths(args.output, args.reduced_events_output)
            with reserve_outputs(args.output, args.reduced_events_output) as outputs:
                campaign = run_book_replay_campaign(
                    _candidate_factory(args),
                    seed=args.seed,
                    traces=args.traces,
                    events_per_trace=args.events_per_trace,
                    max_minimize_runs=args.max_minimize_runs,
                )
                _emit(
                    campaign.to_dict(),
                    args.output,
                    outputs,
                    events=campaign.failure.minimization.events if campaign.failure else None,
                    events_output=args.reduced_events_output,
                )
                if campaign.failure is not None and args.reduced_events_output is not None:
                    print(f"Reduced events written: {Path(args.reduced_events_output)}")
            if campaign.failure is None:
                return 0
            return 2 if campaign.failure.minimization.report.operational_failure else 1

        events_path = Path(args.events).expanduser().resolve()
        events_output = args.events_output if args.command == "minimize" else None
        _require_distinct_paths(str(events_path), args.output, events_output)
        with reserve_outputs(args.output, events_output) as outputs:
            if args.command == "minimize":
                minimization = minimize_book_replay_failure(
                    load_book_replay_events(str(events_path)),
                    _candidate_factory(args),
                    config=BookReplayConfig(args.profile),
                    max_runs=args.max_runs,
                    trace_name=str(events_path),
                )
                _emit(
                    minimization.to_dict(),
                    args.output,
                    outputs,
                    events=minimization.events,
                    events_output=args.events_output,
                )
                print(f"Minimized events written: {Path(args.events_output)}")
                return 2 if minimization.report.operational_failure else 1
            report = run_book_replay(
                load_book_replay_events(str(events_path)),
                _candidate_factory(args),
                config=BookReplayConfig(args.profile),
                trace_name=str(events_path),
            )
            _emit(report.to_dict(), args.output, outputs)
        if report.operational_failure:
            return 2
        return 0 if report.conformant else 1
    except (BookReplayError, OSError, TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2


__all__ = ["main"]
