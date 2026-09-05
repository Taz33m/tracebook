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
            raise ValueError("expected hello message")
        hello = json.loads(first_line)
        if not isinstance(hello, dict) or hello.get("type") != "hello":
            raise ValueError("first message must be hello")
        if hello.get("protocol") != PROTOCOL_NAME:
            raise ValueError(f"protocol must be {PROTOCOL_NAME!r}")
        if hello.get("protocol_version") != PROTOCOL_VERSION:
            raise ValueError(f"protocol_version must be {PROTOCOL_VERSION}")
        adapter = adapter_factory(BookReplayConfig.from_dict(hello.get("config", {})))
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
            message = json.loads(line)
            if not isinstance(message, dict):
                raise ValueError("protocol message must be an object")
            message_type = message.get("type")
            if message_type == "event":
                index = message.get("index")
                if isinstance(index, bool) or not isinstance(index, int) or index <= 0:
                    raise ValueError("event index must be a positive integer")
                if index != last_index + 1:
                    raise ValueError("event indexes must be contiguous and start at 1")
                event = BookReplayEvent.from_mapping(message.get("event", {}))
                observation = adapter.apply(event, index)
                if observation.index != index:
                    raise ValueError("adapter returned the wrong observation index")
                last_index = index
                _write_message(sink, observation.to_dict(include_type=True))
            elif message_type == "snapshot":
                if message.get("index") != last_index:
                    raise ValueError("snapshot index does not match the last event")
                _write_message(
                    sink,
                    {
                        "type": "snapshot",
                        "index": last_index,
                        "state": adapter.snapshot().to_dict(),
                    },
                )
            elif message_type == "finish":
                if message.get("event_count") != last_index:
                    raise ValueError("finish event_count does not match the last event")
                _write_message(sink, {"type": "complete", "event_count": last_index})
                return 0
            else:
                raise ValueError(f"unsupported protocol message type: {message_type!r}")
        raise ValueError("protocol ended before finish")
    except (BrokenPipeError, json.JSONDecodeError, TypeError, ValueError) as exc:
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
