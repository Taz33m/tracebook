"""Newline-delimited JSON protocol helpers for external engine adapters."""

from __future__ import annotations

import json
import sys
from typing import Callable, Optional, Protocol, TextIO

from ..events import MarketEvent
from .model import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookState,
    ConformanceConfig,
    EngineMetadata,
    Observation,
)


class EngineAdapter(Protocol):
    """Minimal interface implemented by in-process and external adapters."""

    metadata: EngineMetadata

    def apply(self, event: MarketEvent, index: int) -> Observation: ...

    def snapshot(self) -> BookState: ...

    def close(self) -> None: ...


AdapterFactory = Callable[[ConformanceConfig], EngineAdapter]


def _write_message(stream: TextIO, payload: dict) -> None:
    stream.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
    stream.flush()


def serve_stdio(
    adapter_factory: AdapterFactory,
    input_stream: Optional[TextIO] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    """Serve one adapter over the conformance NDJSON protocol.

    Adapter programs must reserve stdout for protocol messages. Diagnostics may
    be written to stderr.
    """
    source = input_stream or sys.stdin
    sink = output_stream or sys.stdout
    adapter: Optional[EngineAdapter] = None
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
        adapter = adapter_factory(ConformanceConfig.from_dict(hello.get("config", {})))
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
                event = MarketEvent.from_mapping(message.get("event", {}))
                observation = adapter.apply(event, index)
                if observation.index != index:
                    raise ValueError("adapter returned the wrong observation index")
                last_index = index
                _write_message(sink, observation.to_dict(include_type=True))
            elif message_type == "snapshot":
                requested_index = message.get("index")
                if requested_index != last_index:
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
                event_count = message.get("event_count")
                if event_count != last_index:
                    raise ValueError("finish event_count does not match the last event")
                _write_message(sink, {"type": "complete", "event_count": last_index})
                return 0
            else:
                raise ValueError(f"unsupported protocol message type: {message_type!r}")
        raise ValueError("protocol ended before finish")
    except (BrokenPipeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        try:
            _write_message(
                sink,
                {"type": "error", "code": "PROTOCOL_ERROR", "message": str(exc)},
            )
        except BrokenPipeError:
            pass
        return 2
    except Exception as exc:
        try:
            _write_message(
                sink,
                {"type": "error", "code": "ADAPTER_ERROR", "message": str(exc)},
            )
        except BrokenPipeError:
            pass
        return 2
    finally:
        if adapter is not None:
            try:
                adapter.close()
            except Exception as exc:
                print(f"Adapter close error: {exc}", file=sys.stderr)
