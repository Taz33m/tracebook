"""Command-line interface for normalized historical event replay."""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from .. import __version__
from ..core.order import SelfTradePolicy
from .market_replay import MarketReplayError, replay_market_event_file


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Replay normalized CSV/JSON/JSONL order events through tracebook."
    )
    parser.add_argument("input", help="Path to a normalized event file.")
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    parser.add_argument(
        "--algorithm",
        type=str.lower,
        choices=["fifo", "pro_rata"],
        default="fifo",
    )
    parser.add_argument("--tick-size", type=float, default=0.01)
    parser.add_argument(
        "--self-trade-policy",
        type=str.upper,
        choices=[policy.name for policy in SelfTradePolicy],
        default=SelfTradePolicy.NONE.name,
        help="Self-trade prevention policy applied to every replayed book.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Record rejected events and continue instead of failing fast.",
    )
    parser.add_argument("--depth-levels", type=int, default=5)
    parser.add_argument(
        "--include-trades",
        action="store_true",
        help="Include source-id annotated trade records in the JSON summary.",
    )
    parser.add_argument("--output", help="Optional JSON summary path.")
    args = parser.parse_args(argv)

    if args.depth_levels < 0:
        parser.error("--depth-levels must be non-negative")

    try:
        result = replay_market_event_file(
            args.input,
            matching_algorithm=args.algorithm,
            tick_size=args.tick_size,
            strict=not args.lenient,
            self_trade_policy=SelfTradePolicy[args.self_trade_policy],
        )
    except MarketReplayError as exc:
        print(f"Replay failed: {exc}", file=sys.stderr)
        return 2

    summary = result.to_dict(
        depth_levels=args.depth_levels,
        include_trades=args.include_trades,
    )
    rendered = json.dumps(summary, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(rendered + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Unable to write replay summary: {exc}", file=sys.stderr)
            return 2
        print(f"Replay summary written to: {output}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
