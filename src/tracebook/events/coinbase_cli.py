"""CLI for Coinbase Exchange level-3 snapshot and feed replay."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import IO, Iterator, List, Optional, Tuple

from .. import __version__
from .coinbase_l3 import (
    CoinbaseL3Adapter,
    CoinbaseL3Error,
    iter_coinbase_l3_messages,
    load_coinbase_l3_snapshot,
)
from .market_replay import MarketEvent, MarketReplayError, replay_market_events


def _temporary_text_file(target: Path) -> Tuple[IO[str], Path]:
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    return handle, Path(handle.name)


def _write_json_atomic(target: Path, payload: dict) -> None:
    handle, temporary = _temporary_text_file(target)
    try:
        with handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.write("\n")
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _tick_decimal(value: float) -> Decimal:
    try:
        tick = Decimal(str(value))
    except InvalidOperation as exc:
        raise CoinbaseL3Error("--tick-size must be a positive finite number") from exc
    if not tick.is_finite() or tick <= 0:
        raise CoinbaseL3Error("--tick-size must be a positive finite number")
    return tick


def _validate_event_tick(event: MarketEvent, tick: Decimal) -> None:
    if event.price is None or event.price == 0:
        return
    price = Decimal(str(event.price))
    if price % tick != 0:
        raise CoinbaseL3Error(f"price {event.price} is not aligned to --tick-size {tick}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Normalize and replay a Coinbase Exchange L3 snapshot plus recorded "
            "full/level3 feed."
        )
    )
    parser.add_argument("snapshot", help="REST level-3 snapshot JSON path.")
    parser.add_argument("feed", help="Recorded full/level3 JSON, JSONL, or NDJSON path.")
    parser.add_argument("--version", action="version", version=f"tracebook {__version__}")
    parser.add_argument(
        "--product-id",
        help="Coinbase product id; required when the snapshot has no product_id field.",
    )
    parser.add_argument(
        "--tick-size",
        type=float,
        required=True,
        help="The product quote_increment used by the replay book.",
    )
    parser.add_argument(
        "--lenient",
        action="store_true",
        help="Record feed problems and continue with sequence_complete=false.",
    )
    parser.add_argument("--depth-levels", type=int, default=5)
    parser.add_argument(
        "--include-trades",
        action="store_true",
        help="Include observed Coinbase matches in the JSON summary.",
    )
    parser.add_argument(
        "--include-id-map",
        action="store_true",
        help="Include normalized-to-Coinbase order ids (uses memory proportional to seen ids).",
    )
    parser.add_argument(
        "--events-output",
        help="Optional normalized MarketEvent JSONL output path.",
    )
    parser.add_argument("--output", help="Optional combined JSON summary path.")
    args = parser.parse_args(argv)

    if args.depth_levels < 0:
        parser.error("--depth-levels must be non-negative")

    source_paths = {Path(args.snapshot).resolve(), Path(args.feed).resolve()}
    destination_paths = [
        Path(path).resolve() for path in (args.events_output, args.output) if path is not None
    ]
    if any(path in source_paths for path in destination_paths):
        parser.error("output paths must not overwrite the snapshot or feed")
    if len(destination_paths) != len(set(destination_paths)):
        parser.error("--events-output and --output must be different paths")

    event_handle: Optional[IO[str]] = None
    event_temporary: Optional[Path] = None
    event_target = Path(args.events_output) if args.events_output else None

    try:
        tick = _tick_decimal(args.tick_size)
        snapshot = load_coinbase_l3_snapshot(args.snapshot)
        adapter = CoinbaseL3Adapter(
            snapshot,
            args.product_id,
            strict=not args.lenient,
            retain_id_map=args.include_id_map,
            retain_trades=args.include_trades,
        )
        messages = iter_coinbase_l3_messages(args.feed)

        if event_target is not None:
            event_handle, event_temporary = _temporary_text_file(event_target)

        def tapped_events() -> Iterator[MarketEvent]:
            for event in adapter.iter_events(messages):
                _validate_event_tick(event, tick)
                if event_handle is not None:
                    event_handle.write(
                        json.dumps(event.to_dict(), separators=(",", ":"), allow_nan=False) + "\n"
                    )
                yield event

        replay = replay_market_events(
            tapped_events(),
            matching_algorithm="fifo",
            tick_size=args.tick_size,
            strict=True,
        )
        if replay.trades:
            raise CoinbaseL3Error(
                "normalized Coinbase state generated a simulated trade; "
                "the snapshot/feed is crossed or the tick size is incorrect"
            )
        if event_handle is not None and event_temporary is not None and event_target is not None:
            event_handle.close()
            os.replace(event_temporary, event_target)
            event_handle = None
            event_temporary = None

        summary = {
            "schema_version": 1,
            "normalization": adapter.to_dict(
                include_trades=args.include_trades,
                include_id_map=args.include_id_map,
            ),
            "replay": replay.to_dict(depth_levels=args.depth_levels, include_trades=False),
        }
        if args.output:
            _write_json_atomic(Path(args.output), summary)
            print(f"Coinbase L3 replay summary written to: {args.output}")
        else:
            print(json.dumps(summary, indent=2, allow_nan=False))
        return 0
    except (CoinbaseL3Error, MarketReplayError, OSError, ValueError) as exc:
        print(f"Coinbase L3 replay failed: {exc}", file=sys.stderr)
        return 2
    finally:
        if event_handle is not None:
            event_handle.close()
        if event_temporary is not None:
            event_temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
