"""Command-line interface for external-engine conformance and trace reduction."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from contextlib import ExitStack
from pathlib import Path
from typing import List, Optional

from .._version import __version__
from ..core.order import SelfTradePolicy
from ..events import load_market_events
from .campaign import (
    _CampaignOutputReservation,
    campaign_profile_names,
    run_campaign,
    write_campaign_corpus,
)
from .compare import run_conformance
from .external import AdapterProtocolError, ExternalProcessAdapterFactory
from .junit import write_junit
from .minimize import minimize_failing_trace
from .model import ConformanceConfig, ConformanceError
from .reproduce import (
    discover_failure_metadata,
    load_failure_metadata,
    reproduction_config,
    run_reproduction,
)
from .suite import (
    BUNDLED_SUITE_VERSIONS,
    LATEST_BUNDLED_SUITE_VERSION,
    copy_bundled_conformance_suite,
    load_conformance_suite,
    run_conformance_suite,
)


def _add_config_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--algorithm", choices=["fifo", "pro_rata"], default="fifo")
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument(
        "--self-trade-policy",
        choices=[policy.name for policy in SelfTradePolicy],
        default=SelfTradePolicy.NONE.name,
    )
    parser.add_argument("--quantity-decimal-places", type=int, default=12)


def _add_candidate_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--timeout", type=float, default=5.0)
    candidate = parser.add_mutually_exclusive_group(required=True)
    candidate.add_argument(
        "--candidate-cmd",
        help="Candidate command as one shell-style string.",
    )
    candidate.add_argument(
        "--candidate",
        nargs=argparse.REMAINDER,
        help="Adapter command and arguments; this must be the final CLI option.",
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tracebook-conformance",
        description="Compare an external matching engine with Tracebook's reference semantics.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    sample = commands.add_parser("sample", help="Copy the bundled adversarial suite.")
    sample.add_argument("destination")
    sample.add_argument(
        "--suite-version",
        choices=BUNDLED_SUITE_VERSIONS,
        default=LATEST_BUNDLED_SUITE_VERSION,
        help=f"Bundled suite version to copy (default: {LATEST_BUNDLED_SUITE_VERSION}).",
    )

    run = commands.add_parser("run", help="Compare one normalized event trace.")
    run.add_argument("events")
    run.add_argument("--output")
    run.add_argument("--junit-output")
    _add_config_arguments(run)
    _add_candidate_arguments(run)

    suite = commands.add_parser("suite", help="Run every case in a suite directory.")
    suite.add_argument("suite")
    suite.add_argument("--output")
    suite.add_argument("--junit-output")
    _add_candidate_arguments(suite)

    minimize = commands.add_parser(
        "minimize", help="Reduce a divergent trace to a smaller reproducer."
    )
    minimize.add_argument("events")
    minimize.add_argument("--events-output", required=True)
    minimize.add_argument("--output")
    minimize.add_argument("--junit-output")
    minimize.add_argument("--max-runs", type=int, default=100)
    _add_config_arguments(minimize)
    _add_candidate_arguments(minimize)

    campaign = commands.add_parser(
        "campaign",
        help="Generate deterministic traces and minimize the first divergence.",
    )
    campaign_output = campaign.add_mutually_exclusive_group(required=True)
    campaign_output.add_argument("--output-dir")
    campaign_output.add_argument("--corpus-dir")
    campaign.add_argument(
        "--profile",
        choices=campaign_profile_names(),
        default="fifo-limit-v1",
    )
    campaign.add_argument("--seed", type=int, default=1337)
    campaign.add_argument("--traces", type=int, default=25)
    campaign.add_argument("--events-per-trace", type=int, default=100)
    campaign.add_argument("--max-minimize-runs", type=int, default=100)
    campaign.add_argument(
        "--stop-after-first",
        action="store_true",
        help="Stop after the first divergence (the current campaign contract).",
    )
    campaign.add_argument("--junit-output")
    _add_candidate_arguments(campaign)

    reproduce = commands.add_parser(
        "reproduce",
        help="Replay a saved reduced failure and verify its exact divergence.",
    )
    reproduce.add_argument("events")
    reproduce.add_argument("--metadata")
    reproduce.add_argument("--output")
    reproduce.add_argument("--junit-output")
    _add_config_arguments(reproduce)
    _add_candidate_arguments(reproduce)
    return parser


def _candidate_factory(args) -> ExternalProcessAdapterFactory:
    command = (
        shlex.split(args.candidate_cmd) if args.candidate_cmd is not None else list(args.candidate)
    )
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        raise ConformanceError("--candidate requires a command")
    return ExternalProcessAdapterFactory(command, timeout_seconds=args.timeout)


def _config(args) -> ConformanceConfig:
    return ConformanceConfig(
        matching_algorithm=args.algorithm,
        tick_size=args.tick_size,
        self_trade_policy=SelfTradePolicy[args.self_trade_policy],
        quantity_decimal_places=args.quantity_decimal_places,
    )


def _emit_report(report: dict, output: Optional[str]) -> None:
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(rendered, encoding="utf-8")
        print(f"Report written: {path}")
    else:
        print(rendered, end="")


def _emit_junit(report: dict, output: Optional[str]) -> None:
    if output:
        path = write_junit(report, output)
        print(f"JUnit report written: {path}")


def _write_events(events, output: str) -> None:
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(
                json.dumps(
                    event.to_dict(),
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            )


def _require_distinct_paths(*paths: Optional[str]) -> None:
    resolved = [Path(path).expanduser().resolve() for path in paths if path is not None]
    if len(resolved) != len(set(resolved)):
        raise ConformanceError("input and output paths must be distinct")


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "sample":
            suite = copy_bundled_conformance_suite(
                args.destination,
                suite_version=args.suite_version,
            )
            print(f"Conformance suite copied to: {suite.root}")
            print(f"Suite id: {suite.suite_id}")
            print(f"Cases: {len(suite.cases)}")
            return 0
        if args.command == "run":
            _require_distinct_paths(args.events, args.output, args.junit_output)
            single_report = run_conformance(
                load_market_events(args.events),
                _candidate_factory(args),
                config=_config(args),
                trace_name=str(Path(args.events)),
            )
            payload = single_report.to_dict()
            _emit_report(payload, args.output)
            _emit_junit(payload, args.junit_output)
            return 0 if single_report.conformant else 1
        if args.command == "suite":
            suite_path = Path(args.suite)
            suite_manifest = suite_path / "manifest.json" if suite_path.is_dir() else suite_path
            loaded_suite = load_conformance_suite(args.suite)
            _require_distinct_paths(
                str(suite_manifest),
                *(str(case.events_path) for case in loaded_suite.cases),
                args.output,
                args.junit_output,
            )
            suite_report = run_conformance_suite(loaded_suite, _candidate_factory(args))
            _emit_report(suite_report, args.output)
            _emit_junit(suite_report, args.junit_output)
            return 0 if suite_report["conformant"] else 1
        if args.command == "minimize":
            _require_distinct_paths(
                args.events,
                args.events_output,
                args.output,
                args.junit_output,
            )
            minimization_result = minimize_failing_trace(
                load_market_events(args.events),
                _candidate_factory(args),
                config=_config(args),
                max_runs=args.max_runs,
                trace_name=str(Path(args.events)),
            )
            _write_events(minimization_result.events, args.events_output)
            payload = minimization_result.to_dict()
            _emit_report(payload, args.output)
            _emit_junit(payload, args.junit_output)
            print(f"Minimized events written: {Path(args.events_output)}")
            return 0
        if args.command == "campaign":
            _require_distinct_paths(args.output_dir, args.corpus_dir, args.junit_output)
            candidate_factory = _candidate_factory(args)
            with ExitStack() as stack:
                reservation = (
                    stack.enter_context(_CampaignOutputReservation(args.output_dir))
                    if args.output_dir is not None
                    else None
                )
                campaign_result = run_campaign(
                    candidate_factory,
                    profile=args.profile,
                    seed=args.seed,
                    traces=args.traces,
                    events_per_trace=args.events_per_trace,
                    max_minimize_runs=args.max_minimize_runs,
                )
                if reservation is not None:
                    report_path = reservation.write(campaign_result)
                else:
                    report_path = write_campaign_corpus(campaign_result, args.corpus_dir)
            payload = campaign_result.to_dict()
            _emit_junit(payload, args.junit_output)
            print(f"Campaign report written: {report_path}")
            print(
                f"Traces completed: {len(campaign_result.traces)}/"
                f"{campaign_result.requested_traces}"
            )
            coverage = campaign_result.semantic_coverage
            print(
                f"Semantic coverage: {len(coverage.covered_capabilities)}/"
                f"{len(coverage.expected_capabilities)} capabilities"
            )
            if campaign_result.failure is not None:
                failure = campaign_result.failure
                reduced_path = report_path.parent / "reduced.jsonl"
                print(f"Divergence detected at original event {len(failure.original_events)}")
                print(f"Failure class: {failure.failure_class}")
                print(f"Original trace: {len(failure.original_events)} events")
                print(f"Reduced reproducer: {len(failure.minimization.events)} events")
                print(f"Campaign seed: {campaign_result.seed}")
                print(f"Campaign hash: {campaign_result.campaign_id}")
                print(f"Failure id: {campaign_result.failure_id}")
                print(f"Reduced trace: {reduced_path}")
            return 0 if campaign_result.conformant else 1
        if args.command == "reproduce":
            _require_distinct_paths(
                args.events,
                args.metadata,
                args.output,
                args.junit_output,
            )
            metadata = (
                load_failure_metadata(args.metadata)
                if args.metadata is not None
                else discover_failure_metadata(args.events)
            )
            events = load_market_events(args.events)
            reproduction_result = run_reproduction(
                events,
                _candidate_factory(args),
                config=reproduction_config(metadata, _config(args)),
                expected=metadata,
                trace_name=str(Path(args.events)),
            )
            payload = reproduction_result.to_dict()
            if args.output:
                _emit_report(payload, args.output)
            _emit_junit(payload, args.junit_output)
            original_event = payload.get("original_divergence_event")
            if original_event is not None:
                print(f"Divergence detected at original event {original_event}")
            print(f"Failure class: {reproduction_result.failure_class}")
            print(f"Reduced reproducer: {len(events)} events")
            if payload.get("campaign_seed") is not None:
                print(f"Campaign seed: {payload['campaign_seed']}")
            if payload.get("campaign_id") is not None:
                print(f"Campaign hash: {payload['campaign_id']}")
            print(
                "Reproduction: exact match"
                if reproduction_result.reproduced
                else "Reproduction: mismatch"
            )
            return 0 if reproduction_result.reproduced else 1
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("Conformance operation interrupted.", file=sys.stderr)
        return 130
    except (AdapterProtocolError, ConformanceError, OSError, ValueError) as exc:
        print(f"Conformance operation failed: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
