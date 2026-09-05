"""NDJSON protocol helpers for external L3 book-replay adapters."""

from __future__ import annotations

import json
import sys
from typing import Callable, Optional, Protocol, TextIO

from .model import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookReplayConfig,
    BookReplayEvent,
    BookReplayObservation,
    BookReplayState,
    EngineMetadata,
)


class BookReplayAdapter(Protocol):
    """Minimal interface shared by in-process and external book mirrors."""

    metadata: EngineMetadata

    def apply(self, event: BookReplayEvent, index: int) -> BookReplayObservation: ...

    def snapshot(self) -> BookReplayState: ...

    def close(self) -> None: ...


BookReplayAdapterFactory = Callable[[BookReplayConfig], BookReplayAdapter]


class _RequestError(ValueError):
    """Invalid client input, distinct from exceptions inside an adapter."""


def _message(line: str) -> dict:
    try:
        message = json.loads(line)
    except json.JSONDecodeError as exc:
        raise _RequestError(str(exc)) from exc
    if not isinstance(message, dict):
        raise _RequestError("protocol message must be an object")
    return message


def _index(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _RequestError(f"{name} must be a non-negative integer")
    return value


def _write_message(stream: TextIO, payload: dict) -> None:
    stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def serve_book_replay_stdio(
    adapter_factory: BookReplayAdapterFactory,
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    """Serve one adapter over the isolated book-replay protocol."""
    source = input_stream or sys.stdin
    sink = output_stream or sys.stdout
    adapter: Optional[BookReplayAdapter] = None
    last_index = 0
    try:
        first_line = source.readline()
        if not first_line:
            raise _RequestError("expected hello message")
        hello = _message(first_line)
        if hello.get("type") != "hello":
            raise _RequestError("first message must be hello")
        if hello.get("protocol") != PROTOCOL_NAME:
            raise _RequestError(f"protocol must be {PROTOCOL_NAME!r}")
        if _index(hello.get("protocol_version"), "protocol_version") != PROTOCOL_VERSION:
            raise _RequestError(f"protocol_version must be {PROTOCOL_VERSION}")
        try:
            config = BookReplayConfig.from_dict(hello.get("config", {}))
        except (TypeError, ValueError) as exc:
            raise _RequestError(str(exc)) from exc
        adapter = adapter_factory(config)
        _write_message(
            sink,
            {
                "type": "ready",
                "protocol": PROTOCOL_NAME,
                "protocol_version": PROTOCOL_VERSION,
                "engine": adapter.metadata.to_dict(),
            },
        )

        for line in source:
            if not line.strip():
                continue
            message = _message(line)
            message_type = message.get("type")
            if message_type == "event":
                index = _index(message.get("index"), "event index")
                if index != last_index + 1:
                    raise _RequestError("event indexes must be contiguous and start at 1")
                try:
                    event = BookReplayEvent.from_mapping(message.get("event", {}))
                except (TypeError, ValueError) as exc:
                    raise _RequestError(str(exc)) from exc
                observation = adapter.apply(event, index)
                if observation.index != index:
                    raise ValueError("adapter returned the wrong observation index")
                last_index = index
                _write_message(sink, observation.to_dict(include_type=True))
            elif message_type == "snapshot":
                if _index(message.get("index"), "snapshot index") != last_index:
                    raise _RequestError("snapshot index does not match the last event")
                _write_message(
                    sink,
                    {
                        "type": "snapshot",
                        "index": last_index,
                        "state": adapter.snapshot().to_dict(),
                    },
                )
            elif message_type == "finish":
                if _index(message.get("event_count"), "finish event_count") != last_index:
                    raise _RequestError("finish event_count does not match the last event")
                # Completion includes shutdown. Detach first so a failing close
                # produces one error and is never retried by the cleanup path.
                closing, adapter = adapter, None
                closing.close()
                _write_message(sink, {"type": "complete", "event_count": last_index})
                return 0
            else:
                raise _RequestError(f"unsupported protocol message type: {message_type!r}")
        raise _RequestError("protocol ended before finish")
    except (_RequestError, BrokenPipeError) as exc:
        try:
            _write_message(sink, {"type": "error", "code": "PROTOCOL_ERROR", "message": str(exc)})
        except BrokenPipeError:
            pass
        return 2
    except Exception as exc:
        try:
            _write_message(sink, {"type": "error", "code": "ADAPTER_ERROR", "message": str(exc)})
        except BrokenPipeError:
            pass
        return 2
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                print(f"Adapter close error: {exc}", file=sys.stderr)


__all__ = [
    "BookReplayAdapter",
    "BookReplayAdapterFactory",
    "serve_book_replay_stdio",
]
