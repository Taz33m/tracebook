"""Command-line interface for reproducible Coinbase L3 corpora."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, List, Mapping, Optional

from .. import __version__
from ..events.coinbase_l3 import CoinbaseL3Error
from .coinbase import (
    MARKET_DATA_TERMS_URL,
    CoinbaseCorpusError,
    benchmark_coinbase_corpus,
    capture_coinbase_corpus,
    compare_corpus_benchmarks,
    copy_bundled_coinbase_corpus,
    prepare_coinbase_corpus,
    verify_coinbase_corpus,
    write_json_atomic,
)


def _emit_report(report: Mapping[str, Any], output: Optional[str]) -> None:
    if output:
        path = write_json_atomic(output, report)
        print(f"Report written to: {path}")
    else:
        print(json.dumps(report, sort_keys=True, indent=2, allow_nan=False))


def _add_report_output(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", help="Optional atomic JSON report path.")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture, prepare, verify, and benchmark Coinbase Exchange L3 corpora."
    )
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare",
        help="Sanitize recorded snapshot/feed files and build a corpus.",
    )
    prepare.add_argument("snapshot", help="Source REST L3 snapshot JSON path.")
    prepare.add_argument("feed", help="Source full/level3 JSON, JSONL, or NDJSON path.")
    prepare.add_argument("corpus", help="New corpus directory; it must not already exist.")
    prepare.add_argument("--product-id", required=True)
    prepare.add_argument("--tick-size", required=True)
    prepare.add_argument("--channel", choices=["full", "level3"], default="full")
    prepare.add_argument(
        "--source-classification",
        choices=["market_data", "synthetic"],
        default="market_data",
        help="Use synthetic only for data you authored; it controls rights metadata.",
    )
    prepare.add_argument(
        "--source-environment",
        help="Source label such as production, sandbox, or synthetic.",
    )
    prepare.add_argument(
        "--created-at",
        help="Optional RFC3339 manifest timestamp for deterministic fixture generation.",
    )

    capture = commands.add_parser(
        "capture",
        help="Capture a local-only public Coinbase L3 session.",
    )
    capture.add_argument("corpus", help="New corpus directory; it must not already exist.")
    capture.add_argument("--product-id", required=True)
    capture.add_argument("--tick-size", required=True)
    capture.add_argument("--channel", choices=["full", "level3"], default="level3")
    capture.add_argument(
        "--environment",
        choices=["production", "sandbox"],
        default="production",
    )
    capture.add_argument("--post-snapshot-seconds", type=float, default=10.0)
    capture.add_argument(
        "--max-messages",
        type=int,
        default=100_000,
        help=(
            "Maximum retained frames; capture fails if this limit is reached before "
            "the REST snapshot completes."
        ),
    )
    capture.add_argument("--snapshot-timeout", type=float, default=30.0)
    capture.add_argument(
        "--acknowledge-market-data-terms",
        action="store_true",
        help=f"Required after reviewing {MARKET_DATA_TERMS_URL}",
    )

    sample = commands.add_parser(
        "sample",
        help="Copy the bundled synthetic corpus to a new directory.",
    )
    sample.add_argument("corpus", help="New corpus directory; it must not already exist.")

    verify = commands.add_parser("verify", help="Verify hashes and reproduce golden state.")
    verify.add_argument("corpus")
    _add_report_output(verify)

    benchmark = commands.add_parser(
        "benchmark",
        help="Benchmark streaming import/replay and replay-only phases.",
    )
    benchmark.add_argument("corpus")
    benchmark.add_argument("--iterations", type=int, default=5)
    benchmark.add_argument("--warmups", type=int, default=1)
    _add_report_output(benchmark)

    compare = commands.add_parser(
        "compare",
        help="Compare two benchmark reports for the same corpus.",
    )
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    _add_report_output(compare)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            manifest = prepare_coinbase_corpus(
                args.snapshot,
                args.feed,
                args.corpus,
                product_id=args.product_id,
                tick_size=args.tick_size,
                channel=args.channel,
                source_classification=args.source_classification,
                source_environment=args.source_environment,
                created_at=args.created_at,
            )
            print(f"Corpus created: {Path(args.corpus)}")
            print(f"Corpus id: {manifest['corpus_id']}")
            return 0
        if args.command == "capture":
            manifest = capture_coinbase_corpus(
                args.corpus,
                product_id=args.product_id,
                tick_size=args.tick_size,
                channel=args.channel,
                environment=args.environment,
                post_snapshot_seconds=args.post_snapshot_seconds,
                max_messages=args.max_messages,
                snapshot_timeout=args.snapshot_timeout,
                acknowledge_market_data_terms=args.acknowledge_market_data_terms,
            )
            print(f"Local corpus created: {Path(args.corpus)}")
            print(f"Corpus id: {manifest['corpus_id']}")
            print("Redistribution status: not_granted")
            return 0
        if args.command == "sample":
            manifest = copy_bundled_coinbase_corpus(args.corpus)
            print(f"Synthetic corpus copied to: {Path(args.corpus)}")
            print(f"Corpus id: {manifest['corpus_id']}")
            return 0
        if args.command == "verify":
            _emit_report(verify_coinbase_corpus(args.corpus), args.output)
            return 0
        if args.command == "benchmark":
            report = benchmark_coinbase_corpus(
                args.corpus,
                iterations=args.iterations,
                warmups=args.warmups,
            )
            _emit_report(report, args.output)
            return 0
        if args.command == "compare":
            report = compare_corpus_benchmarks(args.baseline, args.candidate)
            _emit_report(report, args.output)
            return 0
        parser.error(f"unknown command: {args.command}")
    except KeyboardInterrupt:
        print("Corpus operation interrupted.", file=sys.stderr)
        return 130
    except (CoinbaseCorpusError, CoinbaseL3Error, OSError, ValueError) as exc:
        print(f"Corpus operation failed: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
