"""Command-line interface for external-engine conformance and trace reduction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .._version import __version__
from ..core.order import SelfTradePolicy
from ..events import load_market_events
from .campaign import (
    _CampaignOutputReservation,
    campaign_profile_names,
    run_campaign,
)
from .compare import run_conformance
from .external import AdapterProtocolError, ExternalProcessAdapterFactory
from .minimize import minimize_failing_trace
from .model import ConformanceConfig, ConformanceError
from .suite import (
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
    parser.add_argument(
        "--candidate",
        nargs=argparse.REMAINDER,
        required=True,
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

    run = commands.add_parser("run", help="Compare one normalized event trace.")
    run.add_argument("events")
    run.add_argument("--output")
    _add_config_arguments(run)
    _add_candidate_arguments(run)

    suite = commands.add_parser("suite", help="Run every case in a suite directory.")
    suite.add_argument("suite")
    suite.add_argument("--output")
    _add_candidate_arguments(suite)

    minimize = commands.add_parser(
        "minimize", help="Reduce a divergent trace to a smaller reproducer."
    )
    minimize.add_argument("events")
    minimize.add_argument("--events-output", required=True)
    minimize.add_argument("--output")
    minimize.add_argument("--max-runs", type=int, default=100)
    _add_config_arguments(minimize)
    _add_candidate_arguments(minimize)

    campaign = commands.add_parser(
        "campaign",
        help="Generate deterministic traces and minimize the first divergence.",
    )
    campaign.add_argument("--output-dir", required=True)
    campaign.add_argument(
        "--profile",
        choices=campaign_profile_names(),
        default="fifo-limit-v1",
    )
    campaign.add_argument("--seed", type=int, default=1337)
    campaign.add_argument("--traces", type=int, default=25)
    campaign.add_argument("--events-per-trace", type=int, default=100)
    campaign.add_argument("--max-minimize-runs", type=int, default=100)
    _add_candidate_arguments(campaign)
    return parser


def _candidate_factory(args) -> ExternalProcessAdapterFactory:
    command = list(args.candidate)
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
            suite = copy_bundled_conformance_suite(args.destination)
            print(f"Conformance suite copied to: {suite.root}")
            print(f"Suite id: {suite.suite_id}")
            print(f"Cases: {len(suite.cases)}")
            return 0
        if args.command == "run":
            _require_distinct_paths(args.events, args.output)
            single_report = run_conformance(
                load_market_events(args.events),
                _candidate_factory(args),
                config=_config(args),
                trace_name=str(Path(args.events)),
            )
            _emit_report(single_report.to_dict(), args.output)
            return 0 if single_report.conformant else 1
        if args.command == "suite":
            suite_path = Path(args.suite)
            suite_manifest = suite_path / "manifest.json" if suite_path.is_dir() else suite_path
            loaded_suite = load_conformance_suite(args.suite)
            _require_distinct_paths(
                str(suite_manifest),
                *(str(case.events_path) for case in loaded_suite.cases),
                args.output,
            )
            suite_report = run_conformance_suite(loaded_suite, _candidate_factory(args))
            _emit_report(suite_report, args.output)
            return 0 if suite_report["conformant"] else 1
        if args.command == "minimize":
            _require_distinct_paths(args.events, args.events_output, args.output)
            result = minimize_failing_trace(
                load_market_events(args.events),
                _candidate_factory(args),
                config=_config(args),
                max_runs=args.max_runs,
                trace_name=str(Path(args.events)),
            )
            _write_events(result.events, args.events_output)
            _emit_report(result.to_dict(), args.output)
            print(f"Minimized events written: {Path(args.events_output)}")
            return 0
        if args.command == "campaign":
            candidate_factory = _candidate_factory(args)
            with _CampaignOutputReservation(args.output_dir) as reservation:
                campaign_result = run_campaign(
                    candidate_factory,
                    profile=args.profile,
                    seed=args.seed,
                    traces=args.traces,
                    events_per_trace=args.events_per_trace,
                    max_minimize_runs=args.max_minimize_runs,
                )
                report_path = reservation.write(campaign_result)
            print(f"Campaign report written: {report_path}")
            print(
                f"Traces completed: {len(campaign_result.traces)}/"
                f"{campaign_result.requested_traces}"
            )
            if campaign_result.failure is not None:
                failure = campaign_result.failure
                minimized_path = Path(args.output_dir) / "failure" / "minimized.jsonl"
                print(
                    f"First divergence reduced from {len(failure.trace.events)} to "
                    f"{len(failure.minimization.events)} events: {minimized_path}"
                )
            return 0 if campaign_result.conformant else 1
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
