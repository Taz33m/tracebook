#!/usr/bin/env python3
"""Convert a Flash canonical divergence into a Tracebook workload prefix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Optional, Sequence

FLASH_ARTIFACT_TYPE = "matching-engine-benchmark.canonical-divergence"
FLASH_SCHEMA_VERSION = 1
WORKLOAD_MAGIC = 0x4D4542575F303031
WORKLOAD_VERSION = 1
WORKLOAD_MAX_RECORDS = 100_000_000
MAX_EXACT_FLOAT_INTEGER = (1 << 53) - 1
_HEADER = struct.Struct("<QII")
_RECORD = struct.Struct("<BBBBIQQqq")
_MAX_DIVERGENCE_ARTIFACT_BYTES = 1 << 20


class FlashBridgeError(ValueError):
    """Raised when an upstream artifact or workload violates its contract."""


@dataclass(frozen=True)
class ConversionResult:
    """Identity of one converted Flash workload prefix."""

    first_divergent_sequence: int
    event_count: int
    output_path: Path
    output_sha256: str


def _load_json_object(path: Path) -> Mapping[str, object]:
    try:
        if path.stat().st_size > _MAX_DIVERGENCE_ARTIFACT_BYTES:
            raise FlashBridgeError("divergence artifact exceeds the 1 MiB limit")
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FlashBridgeError(f"divergence artifact not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise FlashBridgeError(f"invalid divergence artifact JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise FlashBridgeError("divergence artifact must be a JSON object")
    return payload


def load_divergent_sequence(path: str | Path) -> int:
    """Validate Flash's versioned comparator artifact and return its sequence."""
    source = Path(path).expanduser()
    payload = _load_json_object(source)
    if payload.get("artifact_type") != FLASH_ARTIFACT_TYPE:
        raise FlashBridgeError("unsupported divergence artifact_type")
    if payload.get("schema_version") != FLASH_SCHEMA_VERSION:
        raise FlashBridgeError("unsupported divergence schema_version")
    if payload.get("conformant") is not False:
        raise FlashBridgeError("artifact must describe a canonical divergence")
    sequence = payload.get("first_divergent_sequence")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise FlashBridgeError("first_divergent_sequence must be a non-negative integer")
    return sequence


def _workload_count(handle, path: Path) -> int:
    header = handle.read(_HEADER.size)
    if len(header) != _HEADER.size:
        raise FlashBridgeError("workload is shorter than its 16-byte header")
    magic, version, count = _HEADER.unpack(header)
    if magic != WORKLOAD_MAGIC:
        raise FlashBridgeError("workload has the wrong magic value")
    if version != WORKLOAD_VERSION:
        raise FlashBridgeError(f"workload version {version} is unsupported")
    if count == 0 or count > WORKLOAD_MAX_RECORDS:
        raise FlashBridgeError("workload record count is outside the supported range")
    expected_size = _HEADER.size + count * _RECORD.size
    if path.stat().st_size != expected_size:
        raise FlashBridgeError(f"workload size does not match its declared {count} records")
    return count


def _event_from_record(raw: bytes, expected_sequence: int, symbol: str) -> dict:
    if len(raw) != _RECORD.size:
        raise FlashBridgeError(f"workload record {expected_sequence} is truncated")
    kind, side, ioc, padding, quantity, sequence, order_id, price, reserved = _RECORD.unpack(raw)
    if sequence != expected_sequence:
        raise FlashBridgeError(f"workload record {expected_sequence} carries sequence {sequence}")
    if kind not in {0, 1, 2}:
        raise FlashBridgeError(f"workload record {sequence} has invalid type {kind}")
    if side not in {0, 1}:
        raise FlashBridgeError(f"workload record {sequence} has invalid side {side}")
    if ioc not in {0, 1}:
        raise FlashBridgeError(f"workload record {sequence} has invalid IOC flag {ioc}")
    if padding != 0 or reserved != 0:
        raise FlashBridgeError(f"workload record {sequence} has non-zero reserved data")
    if order_id == 0:
        raise FlashBridgeError(f"workload record {sequence} has order id zero")
    if quantity == 0:
        raise FlashBridgeError(f"workload record {sequence} has quantity zero")
    if price <= 0:
        raise FlashBridgeError(f"workload record {sequence} has non-positive price")
    if price > MAX_EXACT_FLOAT_INTEGER:
        raise FlashBridgeError(
            f"workload record {sequence} price cannot be represented exactly by Tracebook"
        )

    if kind == 0:
        return {
            "op": "new",
            "order_id": order_id,
            "order_type": "IOC" if ioc else "LIMIT",
            "price": price,
            "quantity": quantity,
            "side": "BUY" if side == 0 else "SELL",
            "symbol": symbol,
        }
    if kind == 1:
        return {"op": "cancel", "order_id": order_id, "symbol": symbol}
    return {
        "op": "replace",
        "order_id": order_id,
        "price": price,
        "quantity": quantity,
        "symbol": symbol,
    }


def iter_workload_prefix(
    workload_path: str | Path,
    first_divergent_sequence: int,
    symbol: str = "FLASH",
) -> Iterator[dict]:
    """Yield normalized lifecycle events through the divergent Flash sequence."""
    if (
        isinstance(first_divergent_sequence, bool)
        or not isinstance(first_divergent_sequence, int)
        or first_divergent_sequence < 0
    ):
        raise FlashBridgeError("first divergent sequence must be a non-negative integer")
    if not isinstance(symbol, str) or not symbol.strip():
        raise FlashBridgeError("symbol must be a non-empty string")
    normalized_symbol = symbol.strip()
    source = Path(workload_path).expanduser()
    try:
        with source.open("rb") as handle:
            count = _workload_count(handle, source)
            if first_divergent_sequence >= count:
                raise FlashBridgeError(
                    "first divergent sequence is outside the workload record range"
                )
            for sequence in range(first_divergent_sequence + 1):
                yield _event_from_record(handle.read(_RECORD.size), sequence, normalized_symbol)
    except FileNotFoundError as exc:
        raise FlashBridgeError(f"workload not found: {source}") from exc


def convert_divergent_prefix(
    divergence_path: str | Path,
    workload_path: str | Path,
    output_path: str | Path,
    symbol: str = "FLASH",
) -> ConversionResult:
    """Atomically write the Flash prefix selected by a divergence artifact."""
    divergence = Path(divergence_path).expanduser().resolve()
    workload = Path(workload_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if len({divergence, workload, output}) != 3:
        raise FlashBridgeError("divergence, workload, and output paths must be distinct")
    sequence = load_divergent_sequence(divergence)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    digest = hashlib.sha256()
    count = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            for event in iter_workload_prefix(workload, sequence, symbol=symbol):
                line = (
                    json.dumps(
                        event,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                handle.write(line)
                digest.update(line.encode("utf-8"))
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return ConversionResult(
        first_divergent_sequence=sequence,
        event_count=count,
        output_path=output,
        output_sha256="sha256:" + digest.hexdigest(),
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="convert a Flash canonical divergence into Tracebook JSONL"
    )
    parser.add_argument("divergence", help="schema-v1 Flash divergence JSON")
    parser.add_argument("workload", help="Flash orders_*.bin workload")
    parser.add_argument("output", help="destination Tracebook JSONL prefix")
    parser.add_argument("--symbol", default="FLASH", help="symbol assigned to the workload")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        result = convert_divergent_prefix(
            args.divergence,
            args.workload,
            args.output,
            symbol=args.symbol,
        )
    except (FlashBridgeError, OSError) as exc:
        print(f"Flash bridge failed: {exc}", file=sys.stderr)
        return 2
    print(f"Flash divergence sequence: {result.first_divergent_sequence}")
    print(f"Workload prefix: {result.event_count} events")
    print(f"Tracebook trace hash: {result.output_sha256}")
    print(f"Tracebook JSONL: {result.output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
