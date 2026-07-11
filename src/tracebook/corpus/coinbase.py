"""Build, verify, capture, and benchmark Coinbase Exchange L3 corpora.

The checked artifacts are intentionally separate from the venue adapter. The
adapter stays offline and dependency-free; live capture imports ``websockets``
only when requested through the optional ``capture`` extra.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import math
import os
import platform
import secrets
import shutil
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from importlib import metadata, resources
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, IO, Iterable, List, Mapping, Optional, Sequence, Tuple

from .. import __version__
from ..core.order import normalize_symbol
from ..events.coinbase_l3 import (
    CoinbaseL3Adapter,
    RawCoinbaseMessage,
    iter_coinbase_l3_messages,
    load_coinbase_l3_snapshot,
)
from ..events.market_replay import (
    MarketEvent,
    MarketReplayResult,
    load_market_events,
    replay_market_events,
)

CORPUS_SCHEMA_VERSION = 1
GOLDEN_SCHEMA_VERSION = 1
BENCHMARK_SCHEMA_VERSION = 1
COMPARISON_SCHEMA_VERSION = 1

MARKET_DATA_TERMS_URL = "https://www.coinbase.com/legal/market_data"
PRODUCTION_WEBSOCKET_URL = "wss://ws-feed.exchange.coinbase.com"
SANDBOX_WEBSOCKET_URL = "wss://ws-feed-public.sandbox.exchange.coinbase.com"
PRODUCTION_REST_URL = "https://api.exchange.coinbase.com"
SANDBOX_REST_URL = "https://api-public.sandbox.exchange.coinbase.com"

_SNAPSHOT_FILE = "snapshot.json"
_FEED_FILE = "feed.jsonl"
_EVENTS_FILE = "events.jsonl"
_GOLDEN_FILE = "golden.json"
_MANIFEST_FILE = "manifest.json"
_BUNDLED_CORPUS_NAME = "coinbase-btcusd-synthetic-v1"
_EXPECTED_FILES = {
    "snapshot": _SNAPSHOT_FILE,
    "feed": _FEED_FILE,
    "events": _EVENTS_FILE,
    "golden": _GOLDEN_FILE,
}
_ORDER_ID_FIELDS = {"order_id", "maker_order_id", "taker_order_id"}
_GENERIC_SAFE_FIELDS = {"type", "product_id", "sequence", "time"}
_MESSAGE_FIELDS = {
    "received": _GENERIC_SAFE_FIELDS,
    "open": _GENERIC_SAFE_FIELDS | {"order_id", "side", "price", "remaining_size", "size"},
    "done": _GENERIC_SAFE_FIELDS | {"order_id", "side", "price", "remaining_size", "reason"},
    "match": _GENERIC_SAFE_FIELDS
    | {
        "trade_id",
        "maker_order_id",
        "taker_order_id",
        "side",
        "price",
        "size",
    },
    "change": _GENERIC_SAFE_FIELDS
    | {
        "order_id",
        "side",
        "price",
        "size",
        "old_price",
        "new_price",
        "old_size",
        "new_size",
        "reason",
    },
    "noop": _GENERIC_SAFE_FIELDS,
    "heartbeat": _GENERIC_SAFE_FIELDS | {"last_trade_id"},
    "subscriptions": {"type"},
    "error": {"type", "message", "reason"},
    "activate": _GENERIC_SAFE_FIELDS,
}
_MAX_SNAPSHOT_BYTES = 256 * 1024 * 1024
_MAX_WEBSOCKET_MESSAGE_BYTES = 4 * 1024 * 1024


class CoinbaseCorpusError(ValueError):
    """Raised when a corpus cannot be built or verified safely."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _manifest_time(value: Optional[str]) -> str:
    if value is None:
        return _utc_now()
    if not isinstance(value, str) or not value.strip():
        raise CoinbaseCorpusError("created_at must be an RFC3339 timestamp")
    raw = value.strip()
    parsed_value = raw[:-1] + "+00:00" if raw.endswith("Z") else raw
    try:
        parsed = datetime.fromisoformat(parsed_value)
    except ValueError as exc:
        raise CoinbaseCorpusError("created_at must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise CoinbaseCorpusError("created_at must include a timezone")
    return raw


def _product_id(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CoinbaseCorpusError("product_id must be a non-empty string")
    try:
        return normalize_symbol(value).upper()
    except ValueError as exc:
        raise CoinbaseCorpusError(str(exc)) from exc


def _tick_size(value: Any) -> str:
    if isinstance(value, bool):
        raise CoinbaseCorpusError("tick_size must be a positive finite number")
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise CoinbaseCorpusError("tick_size must be a positive finite number") from exc
    if not parsed.is_finite() or parsed <= 0:
        raise CoinbaseCorpusError("tick_size must be a positive finite number")
    return format(parsed.normalize(), "f")


def _positive_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise CoinbaseCorpusError(f"{field_name} must be a positive integer")
    return value


def _nonnegative_integer(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise CoinbaseCorpusError(f"{field_name} must be a non-negative integer")
    return value


def _positive_seconds(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number")
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number")
    return parsed


def _positive_finite_metric(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number")
    try:
        parsed = float(value)
    except OverflowError as exc:
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise CoinbaseCorpusError(f"{field_name} must be a positive finite number")
    return parsed


def _canonical_json_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _pretty_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _write_json(path: Path, payload: Any) -> None:
    path.write_bytes(_pretty_json_bytes(payload))


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write a JSON report atomically and return its path."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=target.parent,
        prefix=f".{target.name}.",
        suffix=".tmp",
        delete=False,
    )
    temporary = Path(handle.name)
    try:
        with handle:
            handle.write(_pretty_json_bytes(dict(payload)))
        os.replace(temporary, target)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return target


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_size_digest(path: Path) -> Tuple[int, str]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            size += len(chunk)
            digest.update(chunk)
    return size, digest.hexdigest()


def _file_record(path: Path, role: str) -> dict:
    size, digest = _file_size_digest(path)
    return {
        "role": role,
        "path": path.name,
        "bytes": size,
        "sha256": digest,
    }


@dataclass
class SanitizationStats:
    """Counters describing what the sanitizer retained and removed."""

    frames_seen: int = 0
    frames_written: int = 0
    frames_dropped_other_products: int = 0
    sequenced_frames: int = 0
    fields_removed: int = 0
    order_id_fields_pseudonymized: int = 0


class CoinbaseSanitizer:
    """Allowlist Coinbase L3 fields and pseudonymize venue order identifiers."""

    def __init__(self, product_id: str, *, id_key: Optional[bytes] = None) -> None:
        self.product_id = _product_id(product_id)
        key = secrets.token_bytes(32) if id_key is None else id_key
        if not isinstance(key, bytes) or not 16 <= len(key) <= 64:
            raise CoinbaseCorpusError("id_key must contain between 16 and 64 bytes")
        self._id_key = key
        self._source_schema: Optional[Dict[str, Tuple[str, ...]]] = None
        self._output_schema: Optional[Dict[str, Tuple[str, ...]]] = None
        self.stats = SanitizationStats()

    def sanitize_snapshot(self, snapshot: Mapping[str, Any]) -> dict:
        """Return the minimal replayable L3 snapshot with pseudonymized order ids."""
        if not isinstance(snapshot, Mapping):
            raise CoinbaseCorpusError("snapshot must be a JSON object")
        source_product = snapshot.get("product_id")
        if source_product is not None and _product_id(source_product) != self.product_id:
            raise CoinbaseCorpusError("snapshot product_id does not match the corpus product")

        sanitized: Dict[str, Any] = {
            "product_id": self.product_id,
            "sequence": snapshot.get("sequence"),
            "bids": self._sanitize_snapshot_rows(snapshot.get("bids"), "bids"),
            "asks": self._sanitize_snapshot_rows(snapshot.get("asks"), "asks"),
        }
        for field in ("time", "auction_mode"):
            if field in snapshot:
                sanitized[field] = snapshot[field]
        self.stats.fields_removed += len(set(snapshot) - set(sanitized))
        return sanitized

    def sanitize_message(self, raw: RawCoinbaseMessage) -> Optional[RawCoinbaseMessage]:
        """Return one safe replayable frame, or ``None`` for another product."""
        self.stats.frames_seen += 1
        sanitized: Optional[RawCoinbaseMessage]
        if isinstance(raw, Mapping):
            sanitized = self._sanitize_mapping(raw)
        elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
            sanitized = self._sanitize_compact(raw)
        else:
            raise CoinbaseCorpusError("feed frames must be JSON objects or compact arrays")

        if sanitized is None:
            self.stats.frames_dropped_other_products += 1
            return None
        self.stats.frames_written += 1
        return sanitized

    def _sanitize_snapshot_rows(self, rows: Any, field_name: str) -> list:
        if not isinstance(rows, list):
            raise CoinbaseCorpusError(f"snapshot.{field_name} must be an array")
        sanitized = []
        for index, row in enumerate(rows, 1):
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                raise CoinbaseCorpusError(
                    f"snapshot.{field_name}[{index}] must be [price, size, order_id]"
                )
            sanitized.append([row[0], row[1], self._pseudonymize(row[2], "order_id")])
        return sanitized

    def _sanitize_mapping(self, raw: Mapping[str, Any]) -> Optional[dict]:
        message_type = raw.get("type")
        if not isinstance(message_type, str) or not message_type.strip():
            raise CoinbaseCorpusError("feed frame type must be a non-empty string")
        message_type = message_type.strip().lower()

        if message_type == "level3" and "schema" in raw:
            schema = self._sanitize_schema(raw["schema"])
            self.stats.fields_removed += max(0, len(raw) - 2)
            return {"type": "level3", "schema": schema}

        product = raw.get("product_id")
        if product is not None and _product_id(product) != self.product_id:
            return None

        allowed = _MESSAGE_FIELDS.get(message_type, _GENERIC_SAFE_FIELDS)
        sanitized: Dict[str, Any] = {"type": message_type}
        for field in allowed:
            if field == "type" or field not in raw:
                continue
            value = raw[field]
            if field in _ORDER_ID_FIELDS:
                value = self._pseudonymize(value, field)
            sanitized[field] = value
        self.stats.fields_removed += len(set(raw) - set(sanitized))
        if "sequence" in sanitized:
            self.stats.sequenced_frames += 1
        return sanitized

    def _sanitize_schema(self, raw_schema: Any) -> dict:
        if not isinstance(raw_schema, Mapping):
            raise CoinbaseCorpusError("level3 schema must be a JSON object")
        source: Dict[str, Tuple[str, ...]] = {}
        output: Dict[str, Tuple[str, ...]] = {}
        rendered: Dict[str, List[str]] = {}
        for raw_type, raw_fields in raw_schema.items():
            if not isinstance(raw_type, str) or not isinstance(raw_fields, list):
                raise CoinbaseCorpusError("level3 schema entries must map names to field arrays")
            message_type = raw_type.strip().lower()
            if not message_type:
                raise CoinbaseCorpusError("level3 schema message names must not be empty")
            if not all(isinstance(field, str) and field.strip() for field in raw_fields):
                raise CoinbaseCorpusError("level3 schema fields must be non-empty strings")
            fields = tuple(field.strip() for field in raw_fields)
            if not fields or fields[0] != "type" or len(set(fields)) != len(fields):
                raise CoinbaseCorpusError(
                    f"level3 schema for {message_type!r} must start with unique type field"
                )
            allowed = _MESSAGE_FIELDS.get(message_type, _GENERIC_SAFE_FIELDS)
            safe_fields = tuple(field for field in fields if field in allowed)
            if not safe_fields or safe_fields[0] != "type":
                raise CoinbaseCorpusError(
                    f"level3 schema for {message_type!r} has no safe type field"
                )
            source[message_type] = fields
            output[message_type] = safe_fields
            rendered[message_type] = list(safe_fields)
            self.stats.fields_removed += len(fields) - len(safe_fields)
        self._source_schema = source
        self._output_schema = output
        return rendered

    def _sanitize_compact(self, raw: Sequence[Any]) -> Optional[list]:
        if self._source_schema is None or self._output_schema is None:
            raise CoinbaseCorpusError("compact level3 frame arrived before its schema")
        if not raw or not isinstance(raw[0], str):
            raise CoinbaseCorpusError("compact level3 frame must begin with a type string")
        message_type = raw[0].strip().lower()
        source_fields = self._source_schema.get(message_type)
        output_fields = self._output_schema.get(message_type)
        if source_fields is None or output_fields is None:
            raise CoinbaseCorpusError(f"level3 schema has no entry for {message_type!r}")
        if len(raw) != len(source_fields):
            raise CoinbaseCorpusError(
                f"compact {message_type} has {len(raw)} values; expected {len(source_fields)}"
            )
        mapping = dict(zip(source_fields, raw))
        product = mapping.get("product_id")
        if product is not None and _product_id(product) != self.product_id:
            return None
        sanitized = []
        for field in output_fields:
            value = mapping[field]
            if field in _ORDER_ID_FIELDS:
                value = self._pseudonymize(value, field)
            sanitized.append(value)
        if "sequence" in output_fields:
            self.stats.sequenced_frames += 1
        return sanitized

    def _pseudonymize(self, value: Any, field_name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise CoinbaseCorpusError(f"{field_name} must be a non-empty string")
        digest = hashlib.blake2b(
            value.strip().encode("utf-8"),
            digest_size=16,
            key=self._id_key,
            person=b"tb-cb-corpus-v1",
        ).hexdigest()
        self.stats.order_id_fields_pseudonymized += 1
        return f"cb_{digest}"


def _staging_directory(target: Path) -> Path:
    if target.exists() or target.is_symlink():
        raise CoinbaseCorpusError(f"corpus output already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    return Path(
        tempfile.mkdtemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
    )


def _write_sanitized_feed(
    path: Path,
    sanitizer: CoinbaseSanitizer,
    messages: Iterable[RawCoinbaseMessage],
) -> None:
    with path.open("wb") as handle:
        for message in messages:
            sanitized = sanitizer.sanitize_message(message)
            if sanitized is not None:
                handle.write(_canonical_json_bytes(sanitized))


def _validate_event_ticks(events: Iterable[MarketEvent], tick_size: str) -> None:
    tick = Decimal(tick_size)
    for event in events:
        if event.price is None or event.price == 0:
            continue
        price = Decimal(str(event.price))
        if price % tick != 0:
            raise CoinbaseCorpusError(
                f"normalized price {event.price} is not aligned to tick_size {tick_size}"
            )


def _normalize_replay(
    snapshot_path: Path,
    feed_path: Path,
    tick_size: str,
) -> Tuple[CoinbaseL3Adapter, List[MarketEvent], MarketReplayResult]:
    snapshot = load_coinbase_l3_snapshot(snapshot_path)
    adapter = CoinbaseL3Adapter(snapshot, strict=True, retain_trades=True)
    events = list(adapter.iter_events(iter_coinbase_l3_messages(feed_path)))
    _validate_event_ticks(events, tick_size)
    replay = replay_market_events(events, tick_size=float(Decimal(tick_size)), strict=True)
    if replay.trades:
        raise CoinbaseCorpusError(
            "normalized Coinbase lifecycle generated a simulated trade; corpus is crossed"
        )
    return adapter, events, replay


def _events_bytes(events: Iterable[MarketEvent]) -> bytes:
    return b"".join(_canonical_json_bytes(event.to_dict()) for event in events)


def _golden_payload(
    adapter: CoinbaseL3Adapter,
    events: List[MarketEvent],
    replay: MarketReplayResult,
    event_bytes: bytes,
) -> dict:
    books = {}
    for symbol, book in sorted(replay.manager.get_all_order_books().items()):
        market = book.get_market_data_snapshot()
        books[symbol] = {
            "bids": [list(level) for level in market.bid_levels],
            "asks": [list(level) for level in market.ask_levels],
            "active_orders": len(replay.active_order_ids.get(symbol, {})),
        }

    trades = [asdict(trade) for trade in adapter.exchange_trades]
    return {
        "schema_version": GOLDEN_SCHEMA_VERSION,
        "venue": "coinbase_exchange",
        "product_id": adapter.product_id,
        "normalization": {
            "snapshot_sequence": adapter.snapshot_sequence,
            "final_sequence": adapter.final_sequence,
            "sequence_complete": adapter.sequence_complete,
            "snapshot_orders": adapter.snapshot_orders,
            "messages_seen": adapter.messages_seen,
            "messages_sequenced": adapter.messages_sequenced,
            "messages_ignored": adapter.messages_ignored,
            "normalized_events": adapter.normalized_events,
            "active_orders": adapter.active_order_count,
            "exchange_trades_observed": adapter.exchange_trades_observed,
            "exchange_trades_sha256": _sha256_bytes(_canonical_json_bytes(trades)),
        },
        "events": {
            "count": len(events),
            "sha256": _sha256_bytes(event_bytes),
        },
        "replay": {
            "input_events": replay.input_events,
            "applied_events": replay.applied_events,
            "rejected_events": len(replay.rejections),
            "submissions": replay.submissions,
            "cancellations": replay.cancellations,
            "reductions": replay.reductions,
            "replacements": replay.replacements,
            "clears": replay.clears,
            "trades_executed": len(replay.trades),
            "books": books,
        },
    }


def _rights(source_classification: str) -> dict:
    if source_classification == "synthetic":
        return {
            "contains_market_data": False,
            "redistribution": "project_fixture",
            "terms_url": None,
        }
    if source_classification != "market_data":
        raise CoinbaseCorpusError("source_classification must be 'market_data' or 'synthetic'")
    return {
        "contains_market_data": True,
        "redistribution": "not_granted",
        "terms_url": MARKET_DATA_TERMS_URL,
        "notice": (
            "Pseudonymization does not grant redistribution rights. Keep this corpus local "
            "unless you have separate permission from Coinbase."
        ),
    }


def _validate_channel(channel: str) -> str:
    if not isinstance(channel, str) or channel.strip().lower() not in {"full", "level3"}:
        raise CoinbaseCorpusError("channel must be 'full' or 'level3'")
    return channel.strip().lower()


def _corpus_identity(manifest: Mapping[str, Any]) -> str:
    identity = {key: value for key, value in manifest.items() if key != "corpus_id"}
    return f"sha256:{_sha256_bytes(_canonical_json_bytes(identity))}"


def _finalize_staging_corpus(
    staging: Path,
    *,
    product_id: str,
    tick_size: str,
    channel: str,
    source_environment: str,
    source_classification: str,
    sanitizer: CoinbaseSanitizer,
    capture: Mapping[str, Any],
    created_at: str,
) -> dict:
    adapter, events, replay = _normalize_replay(
        staging / _SNAPSHOT_FILE,
        staging / _FEED_FILE,
        tick_size,
    )
    event_bytes = _events_bytes(events)
    (staging / _EVENTS_FILE).write_bytes(event_bytes)
    golden = _golden_payload(adapter, events, replay, event_bytes)
    _write_json(staging / _GOLDEN_FILE, golden)

    files = [_file_record(staging / filename, role) for role, filename in _EXPECTED_FILES.items()]
    source = {
        "venue": "coinbase_exchange",
        "product_id": product_id,
        "channel": channel,
        "environment": source_environment,
        "snapshot_sequence": adapter.snapshot_sequence,
        "final_sequence": adapter.final_sequence,
    }
    replay_config = {
        "matching_algorithm": "fifo",
        "tick_size": tick_size,
        "strict": True,
    }
    manifest: Dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "created_at": created_at,
        "tool": {
            "name": "tracebook",
            "version": __version__,
        },
        "source": source,
        "rights": _rights(source_classification),
        "sanitization": {
            "schema_version": 1,
            "field_policy": "replay_allowlist",
            "order_ids": "keyed_blake2b_128",
            "pseudonymization_scope": (
                "deterministic_fixture"
                if source_classification == "synthetic"
                else "random_local_corpus"
            ),
            "pseudonymization_key_persisted": False,
            **asdict(sanitizer.stats),
        },
        "capture": dict(capture),
        "replay": replay_config,
        "files": files,
    }
    manifest["corpus_id"] = _corpus_identity(manifest)
    _write_json(staging / _MANIFEST_FILE, manifest)
    return manifest


def prepare_coinbase_corpus(
    snapshot_path: str | Path,
    feed_path: str | Path,
    output_dir: str | Path,
    *,
    product_id: str,
    tick_size: Any,
    channel: str = "full",
    source_classification: str = "market_data",
    source_environment: Optional[str] = None,
    created_at: Optional[str] = None,
    id_key: Optional[bytes] = None,
) -> dict:
    """Sanitize recorded inputs and atomically build a verified corpus directory."""
    product = _product_id(product_id)
    tick = _tick_size(tick_size)
    normalized_channel = _validate_channel(channel)
    rights = _rights(source_classification)
    del rights  # Validate classification before touching the output directory.
    environment = source_environment or (
        "synthetic" if source_classification == "synthetic" else "unspecified"
    )
    if not isinstance(environment, str) or not environment.strip():
        raise CoinbaseCorpusError("source_environment must be a non-empty string")

    snapshot_source = Path(snapshot_path)
    feed_source = Path(feed_path)
    target = Path(output_dir)
    if target.resolve() in {snapshot_source.resolve(), feed_source.resolve()}:
        raise CoinbaseCorpusError("corpus output must not overwrite an input file")
    effective_key = id_key
    if effective_key is None and source_classification == "synthetic":
        effective_key = hashlib.sha256(
            f"tracebook-synthetic-corpus-v1\0{product}".encode("utf-8")
        ).digest()
    sanitizer = CoinbaseSanitizer(product, id_key=effective_key)
    staging = _staging_directory(target)
    try:
        snapshot = load_coinbase_l3_snapshot(snapshot_source)
        _write_json(staging / _SNAPSHOT_FILE, sanitizer.sanitize_snapshot(snapshot))
        _write_sanitized_feed(
            staging / _FEED_FILE,
            sanitizer,
            iter_coinbase_l3_messages(feed_source),
        )
        manifest = _finalize_staging_corpus(
            staging,
            product_id=product,
            tick_size=tick,
            channel=normalized_channel,
            source_environment=environment.strip().lower(),
            source_classification=source_classification,
            sanitizer=sanitizer,
            capture={
                "mode": "prepared_offline",
                "websocket_url": None,
                "rest_url": None,
                "post_snapshot_seconds": None,
                "max_messages": None,
            },
            created_at=_manifest_time(created_at),
        )
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_manifest(corpus_dir: Path) -> dict:
    path = corpus_dir / _MANIFEST_FILE
    if not path.is_file():
        raise CoinbaseCorpusError(f"manifest not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoinbaseCorpusError(f"invalid corpus manifest: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise CoinbaseCorpusError(f"manifest schema_version must be {CORPUS_SCHEMA_VERSION}")
    return payload


def _manifest_file_records(manifest: Mapping[str, Any]) -> List[dict]:
    records = manifest.get("files")
    if not isinstance(records, list) or not all(isinstance(record, dict) for record in records):
        raise CoinbaseCorpusError("manifest files must be an array of objects")
    by_role = {record.get("role"): record for record in records}
    if len(records) != len(_EXPECTED_FILES) or set(by_role) != set(_EXPECTED_FILES):
        raise CoinbaseCorpusError(f"manifest file roles must be {sorted(_EXPECTED_FILES)}")
    normalized = []
    for role, expected_path in _EXPECTED_FILES.items():
        record = by_role[role]
        raw_path = record.get("path")
        if not isinstance(raw_path, str):
            raise CoinbaseCorpusError(f"manifest file path for {role} must be a string")
        parsed = PurePosixPath(raw_path)
        if parsed.is_absolute() or ".." in parsed.parts or raw_path != expected_path:
            raise CoinbaseCorpusError(f"manifest file path for {role} is unsafe")
        if (
            isinstance(record.get("bytes"), bool)
            or not isinstance(record.get("bytes"), int)
            or record["bytes"] < 0
        ):
            raise CoinbaseCorpusError(f"manifest byte count for {role} is invalid")
        digest = record.get("sha256")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise CoinbaseCorpusError(f"manifest sha256 for {role} is invalid")
        normalized.append(record)
    return normalized


def verify_coinbase_corpus(corpus_dir: str | Path) -> dict:
    """Verify file integrity and reproduce normalized events and golden state."""
    root = Path(corpus_dir)
    manifest = _load_manifest(root)
    records = _manifest_file_records(manifest)
    for record in records:
        path = root / record["path"]
        if path.is_symlink():
            raise CoinbaseCorpusError(f"corpus files must not be symlinks: {path}")
        if not path.is_file():
            raise CoinbaseCorpusError(f"corpus file not found: {path}")
        size, digest = _file_size_digest(path)
        if size != record["bytes"]:
            raise CoinbaseCorpusError(f"corpus byte count mismatch: {record['path']}")
        if digest != record["sha256"]:
            raise CoinbaseCorpusError(f"corpus sha256 mismatch: {record['path']}")

    source = manifest.get("source")
    replay_config = manifest.get("replay")
    if not isinstance(source, dict) or not isinstance(replay_config, dict):
        raise CoinbaseCorpusError("manifest source and replay must be objects")
    expected_id = _corpus_identity(manifest)
    if manifest.get("corpus_id") != expected_id:
        raise CoinbaseCorpusError("manifest corpus_id does not match its file identity")
    tick = _tick_size(replay_config.get("tick_size"))

    adapter, events, replay = _normalize_replay(
        root / _SNAPSHOT_FILE,
        root / _FEED_FILE,
        tick,
    )
    event_bytes = _events_bytes(events)
    if event_bytes != (root / _EVENTS_FILE).read_bytes():
        raise CoinbaseCorpusError("normalized events do not reproduce exactly")
    golden = _golden_payload(adapter, events, replay, event_bytes)
    try:
        stored_golden = json.loads((root / _GOLDEN_FILE).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CoinbaseCorpusError(f"invalid golden JSON: {exc}") from exc
    if golden != stored_golden:
        raise CoinbaseCorpusError("golden replay state does not reproduce exactly")
    if (
        source.get("product_id") != adapter.product_id
        or source.get("snapshot_sequence") != adapter.snapshot_sequence
        or source.get("final_sequence") != adapter.final_sequence
    ):
        raise CoinbaseCorpusError("manifest source summary does not match replayed input")

    return {
        "schema_version": 1,
        "verified": True,
        "corpus_id": expected_id,
        "files_verified": len(records),
        "events_verified": len(events),
        "snapshot_sequence": adapter.snapshot_sequence,
        "final_sequence": adapter.final_sequence,
    }


def copy_bundled_coinbase_corpus(output_dir: str | Path) -> dict:
    """Copy the bundled synthetic corpus to a new directory and verify it."""
    target = Path(output_dir)
    staging = _staging_directory(target)
    resource_root = resources.files("tracebook.corpus.fixtures").joinpath(_BUNDLED_CORPUS_NAME)
    names = [*_EXPECTED_FILES.values(), _MANIFEST_FILE]
    try:
        for name in names:
            resource = resource_root.joinpath(name)
            if not resource.is_file():
                raise CoinbaseCorpusError(f"bundled corpus resource is missing: {name}")
            with resource.open("rb") as source, (staging / name).open("wb") as destination:
                shutil.copyfileobj(source, destination)
        verify_coinbase_corpus(staging)
        manifest = _load_manifest(staging)
        os.replace(staging, target)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _endpoint_urls(environment: str) -> Tuple[str, str]:
    if environment == "production":
        return PRODUCTION_WEBSOCKET_URL, PRODUCTION_REST_URL
    if environment == "sandbox":
        return SANDBOX_WEBSOCKET_URL, SANDBOX_REST_URL
    raise CoinbaseCorpusError("environment must be 'production' or 'sandbox'")


def _fetch_snapshot(product_id: str, environment: str, timeout: float) -> Mapping[str, Any]:
    _, rest_base = _endpoint_urls(environment)
    product_path = urllib.parse.quote(product_id, safe="-")
    url = f"{rest_base}/products/{product_path}/book?level=3"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": f"tracebook/{__version__}",
        },
    )
    try:
        # The scheme and host come only from the fixed Coinbase HTTPS endpoints above.
        with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
            payload = response.read(_MAX_SNAPSHOT_BYTES + 1)
    except (OSError, urllib.error.URLError) as exc:
        raise CoinbaseCorpusError(f"unable to fetch Coinbase L3 snapshot: {exc}") from exc
    if len(payload) > _MAX_SNAPSHOT_BYTES:
        raise CoinbaseCorpusError("Coinbase L3 snapshot exceeded the 256 MiB safety limit")
    try:
        decoded = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CoinbaseCorpusError(f"Coinbase L3 snapshot was not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise CoinbaseCorpusError("Coinbase L3 snapshot must be a JSON object")
    decoded["product_id"] = product_id
    return decoded


def _default_websocket_connector(url: str, **kwargs: Any) -> Any:
    try:
        from websockets.asyncio.client import connect
    except ImportError as exc:
        raise CoinbaseCorpusError(
            "live capture requires the optional dependency: "
            "python -m pip install 'tracebook-sim[capture]'"
        ) from exc
    return connect(url, **kwargs)


async def _capture_session(
    *,
    product_id: str,
    channel: str,
    environment: str,
    post_snapshot_seconds: float,
    max_messages: int,
    snapshot_timeout: float,
    sanitizer: CoinbaseSanitizer,
    feed_handle: IO[bytes],
    websocket_connector: Optional[Callable[..., Any]],
    snapshot_fetcher: Optional[Callable[[str, str, float], Mapping[str, Any]]],
) -> Tuple[Mapping[str, Any], dict]:
    websocket_url, rest_base = _endpoint_urls(environment)
    connector = websocket_connector or _default_websocket_connector
    fetcher = snapshot_fetcher or _fetch_snapshot
    started_at = _utc_now()
    subscribe = {
        "type": "subscribe",
        "product_ids": [product_id],
        "channels": [channel],
    }
    stop = asyncio.Event()
    reader_errors: List[BaseException] = []
    stop_reason = "connection_closed"
    snapshot_acquired = False

    try:
        connection = connector(
            websocket_url,
            open_timeout=snapshot_timeout,
            close_timeout=10,
            ping_interval=20,
            ping_timeout=20,
            max_size=_MAX_WEBSOCKET_MESSAGE_BYTES,
            max_queue=1024,
        )
    except CoinbaseCorpusError:
        raise
    except Exception as exc:
        raise CoinbaseCorpusError(f"unable to configure Coinbase WebSocket: {exc}") from exc

    async with connection as websocket:
        await websocket.send(json.dumps(subscribe, separators=(",", ":")))

        async def read_frames() -> None:
            nonlocal stop_reason
            try:
                async for payload in websocket:
                    if isinstance(payload, bytes):
                        payload = payload.decode("utf-8")
                    if not isinstance(payload, str):
                        raise CoinbaseCorpusError("Coinbase WebSocket frame must be text JSON")
                    try:
                        decoded = json.loads(payload)
                    except json.JSONDecodeError as exc:
                        raise CoinbaseCorpusError(
                            f"Coinbase WebSocket frame was not valid JSON: {exc}"
                        ) from exc
                    sanitized = sanitizer.sanitize_message(decoded)
                    if sanitized is not None:
                        feed_handle.write(_canonical_json_bytes(sanitized))
                    if sanitizer.stats.frames_written >= max_messages:
                        stop_reason = (
                            "message_limit" if snapshot_acquired else "pre_snapshot_message_limit"
                        )
                        stop.set()
                        return
            except asyncio.CancelledError:
                raise
            except BaseException as exc:
                reader_errors.append(exc)
            finally:
                stop.set()

        reader = asyncio.create_task(read_frames())
        try:
            snapshot = await asyncio.wait_for(
                asyncio.to_thread(fetcher, product_id, environment, snapshot_timeout),
                timeout=snapshot_timeout,
            )
        except Exception:
            reader.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await reader
            raise
        snapshot_acquired = True

        timed_out = False
        try:
            await asyncio.wait_for(stop.wait(), timeout=post_snapshot_seconds)
        except asyncio.TimeoutError:
            timed_out = True
            stop_reason = "duration"
            reader.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await reader
        feed_handle.flush()

        if reader_errors:
            error = reader_errors[0]
            if isinstance(error, CoinbaseCorpusError):
                raise error
            raise CoinbaseCorpusError(f"Coinbase WebSocket capture failed: {error}") from error
        if stop_reason == "pre_snapshot_message_limit":
            raise CoinbaseCorpusError(
                "max_messages was reached before the REST snapshot completed; "
                "increase the limit so capture includes a post-snapshot interval"
            )
        if not timed_out and stop_reason != "message_limit":
            raise CoinbaseCorpusError("Coinbase WebSocket closed before the capture completed")

    return snapshot, {
        "mode": "live_public",
        "started_at": started_at,
        "completed_at": _utc_now(),
        "websocket_url": websocket_url,
        "rest_url": f"{rest_base}/products/{product_id}/book?level=3",
        "post_snapshot_seconds": post_snapshot_seconds,
        "max_messages": max_messages,
        "stop_reason": stop_reason,
    }


async def capture_coinbase_corpus_async(
    output_dir: str | Path,
    *,
    product_id: str,
    tick_size: Any,
    channel: str = "level3",
    environment: str = "production",
    post_snapshot_seconds: float = 10.0,
    max_messages: int = 100_000,
    snapshot_timeout: float = 30.0,
    acknowledge_market_data_terms: bool = False,
    created_at: Optional[str] = None,
    id_key: Optional[bytes] = None,
    websocket_connector: Optional[Callable[..., Any]] = None,
    snapshot_fetcher: Optional[Callable[[str, str, float], Mapping[str, Any]]] = None,
) -> dict:
    """Capture one public Coinbase session and atomically build a local corpus."""
    if acknowledge_market_data_terms is not True:
        raise CoinbaseCorpusError(
            "live capture requires acknowledge_market_data_terms=True after reviewing "
            f"{MARKET_DATA_TERMS_URL}"
        )
    product = _product_id(product_id)
    tick = _tick_size(tick_size)
    normalized_channel = _validate_channel(channel)
    normalized_environment = environment.strip().lower() if isinstance(environment, str) else ""
    _endpoint_urls(normalized_environment)
    duration = _positive_seconds(post_snapshot_seconds, "post_snapshot_seconds")
    timeout = _positive_seconds(snapshot_timeout, "snapshot_timeout")
    message_limit = _positive_integer(max_messages, "max_messages")

    target = Path(output_dir)
    sanitizer = CoinbaseSanitizer(product, id_key=id_key)
    staging = _staging_directory(target)
    try:
        with (staging / _FEED_FILE).open("wb") as feed_handle:
            snapshot, capture = await _capture_session(
                product_id=product,
                channel=normalized_channel,
                environment=normalized_environment,
                post_snapshot_seconds=duration,
                max_messages=message_limit,
                snapshot_timeout=timeout,
                sanitizer=sanitizer,
                feed_handle=feed_handle,
                websocket_connector=websocket_connector,
                snapshot_fetcher=snapshot_fetcher,
            )
        _write_json(staging / _SNAPSHOT_FILE, sanitizer.sanitize_snapshot(snapshot))
        manifest = _finalize_staging_corpus(
            staging,
            product_id=product,
            tick_size=tick,
            channel=normalized_channel,
            source_environment=normalized_environment,
            source_classification="market_data",
            sanitizer=sanitizer,
            capture=capture,
            created_at=_manifest_time(created_at),
        )
        os.replace(staging, target)
        return manifest
    except CoinbaseCorpusError:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    except Exception as exc:
        shutil.rmtree(staging, ignore_errors=True)
        raise CoinbaseCorpusError(f"live capture failed: {exc}") from exc


def capture_coinbase_corpus(
    output_dir: str | Path,
    **kwargs: Any,
) -> dict:
    """Synchronous wrapper for :func:`capture_coinbase_corpus_async`."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        pass
    else:
        raise CoinbaseCorpusError(
            "capture_coinbase_corpus cannot run inside an event loop; "
            "await capture_coinbase_corpus_async instead"
        )
    return asyncio.run(capture_coinbase_corpus_async(output_dir, **kwargs))


def _distribution_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for package in ("tracebook-sim", "numpy", "psutil", "websockets"):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = __version__ if package == "tracebook-sim" else None
    return versions


def _benchmark_environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "dependency_versions": _distribution_versions(),
    }


def _phase_summary(samples_ns: List[int], event_count: int) -> dict:
    ordered = sorted(samples_ns)
    p95_index = max(0, math.ceil(len(ordered) * 0.95) - 1)
    median_ns = float(statistics.median(ordered))
    return {
        "event_count": event_count,
        "samples_ns": samples_ns,
        "min_ns": ordered[0],
        "median_ns": median_ns,
        "p95_ns": ordered[p95_index],
        "max_ns": ordered[-1],
        "events_per_second_median": event_count / (median_ns / 1_000_000_000),
    }


def _measure_phase(
    runner: Callable[[], int],
    *,
    event_count: int,
    iterations: int,
    warmups: int,
) -> dict:
    samples = []
    for index in range(iterations + warmups):
        started = time.perf_counter_ns()
        observed_count = runner()
        elapsed = time.perf_counter_ns() - started
        if observed_count != event_count:
            raise CoinbaseCorpusError(
                f"benchmark event count changed: expected {event_count}, got {observed_count}"
            )
        if index >= warmups:
            samples.append(elapsed)
    return _phase_summary(samples, event_count)


def benchmark_coinbase_corpus(
    corpus_dir: str | Path,
    *,
    iterations: int = 5,
    warmups: int = 1,
) -> dict:
    """Benchmark streaming import/replay and replay-only phases for one corpus."""
    measured_iterations = _positive_integer(iterations, "iterations")
    measured_warmups = _nonnegative_integer(warmups, "warmups")
    root = Path(corpus_dir)
    verification = verify_coinbase_corpus(root)
    manifest = _load_manifest(root)
    replay_config = manifest["replay"]
    tick = _tick_size(replay_config.get("tick_size"))
    events = load_market_events(root / _EVENTS_FILE)
    event_count = len(events)

    def stream_import_replay() -> int:
        _, normalized, _ = _normalize_replay(
            root / _SNAPSHOT_FILE,
            root / _FEED_FILE,
            tick,
        )
        return len(normalized)

    def replay_only() -> int:
        replay = replay_market_events(events, tick_size=float(Decimal(tick)), strict=True)
        if replay.trades:
            raise CoinbaseCorpusError("replay-only benchmark generated a simulated trade")
        return replay.input_events

    phases = {
        "stream_import_replay": _measure_phase(
            stream_import_replay,
            event_count=event_count,
            iterations=measured_iterations,
            warmups=measured_warmups,
        ),
        "replay_only": _measure_phase(
            replay_only,
            event_count=event_count,
            iterations=measured_iterations,
            warmups=measured_warmups,
        ),
    }
    manifest_bytes = (root / _MANIFEST_FILE).read_bytes()
    return {
        "schema_version": BENCHMARK_SCHEMA_VERSION,
        "measurement_model": "local_wall_clock",
        "generated_at": _utc_now(),
        "corpus": {
            "corpus_id": verification["corpus_id"],
            "manifest_sha256": _sha256_bytes(manifest_bytes),
            "product_id": manifest["source"]["product_id"],
            "event_count": event_count,
        },
        "config": {
            "iterations": measured_iterations,
            "warmups": measured_warmups,
            "clock": "time.perf_counter_ns",
        },
        "environment": _benchmark_environment(),
        "phases": phases,
    }


def _load_benchmark_report(value: str | Path | Mapping[str, Any], label: str) -> dict:
    if isinstance(value, Mapping):
        report = dict(value)
    else:
        path = Path(value)
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise CoinbaseCorpusError(f"invalid {label} benchmark report: {exc}") from exc
    if not isinstance(report, dict):
        raise CoinbaseCorpusError(f"{label} benchmark report must be a JSON object")
    if report.get("schema_version") != BENCHMARK_SCHEMA_VERSION:
        raise CoinbaseCorpusError(
            f"{label} benchmark schema_version must be {BENCHMARK_SCHEMA_VERSION}"
        )
    corpus = report.get("corpus")
    phases = report.get("phases")
    if not isinstance(corpus, dict) or not isinstance(phases, dict) or not phases:
        raise CoinbaseCorpusError(f"{label} benchmark is missing corpus or phases")
    corpus_id = corpus.get("corpus_id")
    if (
        not isinstance(corpus_id, str)
        or not corpus_id.startswith("sha256:")
        or len(corpus_id) != 71
        or any(character not in "0123456789abcdef" for character in corpus_id[7:])
    ):
        raise CoinbaseCorpusError(f"{label} benchmark corpus_id is invalid")
    manifest_digest = corpus.get("manifest_sha256")
    if (
        not isinstance(manifest_digest, str)
        or len(manifest_digest) != 64
        or any(character not in "0123456789abcdef" for character in manifest_digest)
    ):
        raise CoinbaseCorpusError(f"{label} benchmark manifest_sha256 is invalid")
    return report


def compare_corpus_benchmarks(
    baseline: str | Path | Mapping[str, Any],
    candidate: str | Path | Mapping[str, Any],
) -> dict:
    """Compare two reports while exposing machine and software differences."""
    baseline_report = _load_benchmark_report(baseline, "baseline")
    candidate_report = _load_benchmark_report(candidate, "candidate")
    baseline_corpus = baseline_report["corpus"]
    candidate_corpus = candidate_report["corpus"]
    if baseline_corpus.get("corpus_id") != candidate_corpus.get("corpus_id"):
        raise CoinbaseCorpusError("benchmark reports must reference the same corpus_id")

    baseline_environment = baseline_report.get("environment")
    candidate_environment = candidate_report.get("environment")
    if not isinstance(baseline_environment, dict) or not isinstance(candidate_environment, dict):
        raise CoinbaseCorpusError("benchmark reports must include environment objects")
    environment_keys = ("python", "platform", "processor", "machine")
    environment_differences = [
        key
        for key in environment_keys
        if baseline_environment.get(key) != candidate_environment.get(key)
    ]
    baseline_software = baseline_environment.get("dependency_versions", {})
    candidate_software = candidate_environment.get("dependency_versions", {})
    if not isinstance(baseline_software, dict) or not isinstance(candidate_software, dict):
        raise CoinbaseCorpusError("benchmark dependency_versions must be objects")
    software_keys = sorted(set(baseline_software) | set(candidate_software))
    software_differences = [
        key for key in software_keys if baseline_software.get(key) != candidate_software.get(key)
    ]

    baseline_phases = baseline_report["phases"]
    candidate_phases = candidate_report["phases"]
    if set(baseline_phases) != set(candidate_phases):
        raise CoinbaseCorpusError("benchmark reports must contain the same phases")
    comparisons = {}
    for phase in sorted(baseline_phases):
        before = baseline_phases[phase]
        after = candidate_phases[phase]
        if not isinstance(before, dict) or not isinstance(after, dict):
            raise CoinbaseCorpusError(f"benchmark phase {phase!r} must be an object")
        baseline_event_count = _positive_integer(
            before.get("event_count"), f"baseline benchmark phase {phase!r} event_count"
        )
        candidate_event_count = _positive_integer(
            after.get("event_count"), f"candidate benchmark phase {phase!r} event_count"
        )
        if baseline_event_count != candidate_event_count:
            raise CoinbaseCorpusError(f"benchmark phase {phase!r} event counts differ")
        baseline_median = _positive_finite_metric(
            before.get("median_ns"), f"baseline benchmark phase {phase!r} median_ns"
        )
        candidate_median = _positive_finite_metric(
            after.get("median_ns"), f"candidate benchmark phase {phase!r} median_ns"
        )
        baseline_rate = _positive_finite_metric(
            before.get("events_per_second_median"),
            f"baseline benchmark phase {phase!r} events_per_second_median",
        )
        candidate_rate = _positive_finite_metric(
            after.get("events_per_second_median"),
            f"candidate benchmark phase {phase!r} events_per_second_median",
        )
        comparisons[phase] = {
            "event_count": baseline_event_count,
            "baseline_median_ns": baseline_median,
            "candidate_median_ns": candidate_median,
            "duration_ratio_candidate_to_baseline": candidate_median / baseline_median,
            "speedup_baseline_over_candidate": baseline_median / candidate_median,
            "baseline_events_per_second": baseline_rate,
            "candidate_events_per_second": candidate_rate,
            "throughput_change_percent": ((candidate_rate / baseline_rate) - 1.0) * 100.0,
        }

    return {
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "corpus_id": baseline_corpus["corpus_id"],
        "manifest_match": baseline_corpus.get("manifest_sha256")
        == candidate_corpus.get("manifest_sha256"),
        "baseline": {
            "generated_at": baseline_report.get("generated_at"),
            "manifest_sha256": baseline_corpus.get("manifest_sha256"),
            "config": baseline_report.get("config"),
            "environment": baseline_environment,
        },
        "candidate": {
            "generated_at": candidate_report.get("generated_at"),
            "manifest_sha256": candidate_corpus.get("manifest_sha256"),
            "config": candidate_report.get("config"),
            "environment": candidate_environment,
        },
        "environment_match": not environment_differences,
        "environment_differences": environment_differences,
        "software_differences": software_differences,
        "phases": comparisons,
    }
