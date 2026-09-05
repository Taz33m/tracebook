"""External-process client for the isolated book-replay protocol."""

from __future__ import annotations

import json
import math
import queue
import subprocess  # nosec B404
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Mapping, Optional, Sequence

from ..conformance.model import ConformanceError
from .model import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookReplayConfig,
    BookReplayError,
    BookReplayEvent,
    BookReplayObservation,
    BookReplayState,
    EngineMetadata,
)

_END_OF_STREAM = object()


class BookReplayProtocolError(BookReplayError):
    """Raised when an external book adapter violates or times out on protocol."""


class ExternalBookReplayAdapter:
    """Drive one external book mirror over newline-delimited JSON."""

    def __init__(
        self,
        command: Sequence[str],
        config: BookReplayConfig,
        timeout_seconds: float = 5.0,
        cwd: Optional[str | Path] = None,
    ) -> None:
        if isinstance(command, (str, bytes)) or not command:
            raise BookReplayError("candidate command must be a non-empty argument list")
        normalized_command = []
        for argument in command:
            if not isinstance(argument, str) or not argument:
                raise BookReplayError("candidate command arguments must be non-empty strings")
            normalized_command.append(argument)
        try:
            normalized_timeout = float(timeout_seconds)
        except (TypeError, ValueError, OverflowError) as exc:
            raise BookReplayError("timeout_seconds must be a positive finite number") from exc
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(normalized_timeout)
            or normalized_timeout <= 0
        ):
            raise BookReplayError("timeout_seconds must be a positive finite number")

        self.command = tuple(normalized_command)
        self.timeout_seconds = normalized_timeout
        self._closed = False
        self._broken = False
        self._last_index = 0
        self._stdout_queue: queue.Queue[object] = queue.Queue(maxsize=64)
        self._stderr_lines: Deque[str] = deque(maxlen=64)
        self._stderr_lock = threading.Lock()
        try:
            self._process = subprocess.Popen(  # nosec B603
                self.command,
                cwd=str(cwd) if cwd is not None else None,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
            )
        except OSError as exc:
            raise BookReplayProtocolError(
                f"unable to start candidate command {self.command!r}: {exc}"
            ) from exc
        stdout = self._process.stdout
        stderr = self._process.stderr
        if stdout is None or stderr is None:
            self._process.kill()
            self._process.wait()
            raise BookReplayProtocolError("candidate process pipes were not created")
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(stdout,),
            name="tracebook-book-replay-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(stderr,),
            name="tracebook-book-replay-stderr",
            daemon=True,
        )
        self._stdout_thread.start()
        self._stderr_thread.start()

        try:
            self._send(
                {
                    "type": "hello",
                    "protocol": PROTOCOL_NAME,
                    "protocol_version": PROTOCOL_VERSION,
                    "config": config.to_dict(),
                }
            )
            ready = self._receive("ready")
            if ready.get("protocol") != PROTOCOL_NAME:
                raise BookReplayProtocolError("ready message reported the wrong protocol")
            if ready.get("protocol_version") != PROTOCOL_VERSION:
                raise BookReplayProtocolError("ready message reported the wrong protocol version")
            try:
                self.metadata = EngineMetadata.from_dict(ready.get("engine", {}))
            except ConformanceError as exc:
                raise BookReplayProtocolError(f"invalid ready engine metadata: {exc}") from exc
        except Exception:
            self._broken = True
            self._shutdown()
            raise

    def _read_stdout(self, stream) -> None:
        try:
            for line in stream:
                self._stdout_queue.put(line)
        finally:
            self._stdout_queue.put(_END_OF_STREAM)

    def _read_stderr(self, stream) -> None:
        for line in stream:
            with self._stderr_lock:
                self._stderr_lines.append(line[-4096:])

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            rendered = "".join(self._stderr_lines).strip()
        return rendered[-4000:] if rendered else ""

    def _send(self, payload: Mapping[str, object]) -> None:
        if self._closed or self._process.poll() is not None:
            raise BookReplayProtocolError(
                f"candidate process is not running{self._process_context()}"
            )
        process_input = self._process.stdin
        if process_input is None:
            self._broken = True
            raise BookReplayProtocolError("candidate process input pipe is unavailable")
        try:
            process_input.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
            process_input.flush()
        except (BrokenPipeError, OSError) as exc:
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate process closed its input{self._process_context()}"
            ) from exc

    def _receive(self, expected_type: str) -> dict:
        try:
            item = self._stdout_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate timed out waiting for {expected_type!r} after "
                f"{self.timeout_seconds:g}s{self._process_context()}"
            ) from exc
        if item is _END_OF_STREAM:
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate exited before {expected_type!r}{self._process_context()}"
            )
        if not isinstance(item, str):
            self._broken = True
            raise BookReplayProtocolError("candidate produced an invalid protocol frame")
        try:
            message = json.loads(item)
        except json.JSONDecodeError as exc:
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate stdout was not valid JSON: {item.strip()!r}"
            ) from exc
        if not isinstance(message, dict):
            self._broken = True
            raise BookReplayProtocolError("candidate protocol frame must be a JSON object")
        if message.get("type") == "error":
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate reported {message.get('code', 'ERROR')}: "
                f"{message.get('message', '')}"
            )
        if message.get("type") != expected_type:
            self._broken = True
            raise BookReplayProtocolError(
                f"expected candidate message {expected_type!r}, "
                f"received {message.get('type')!r}"
            )
        return message

    def _process_context(self) -> str:
        return_code = self._process.poll()
        details = f" (exit code {return_code})" if return_code is not None else ""
        stderr = self._stderr_tail()
        return details + (f"; stderr: {stderr}" if stderr else "")

    def apply(self, event: BookReplayEvent, index: int) -> BookReplayObservation:
        self._send({"type": "event", "index": index, "event": event.to_dict()})
        observation = BookReplayObservation.from_dict(self._receive("observation"))
        if observation.index != index:
            self._broken = True
            raise BookReplayProtocolError(
                f"candidate observation index {observation.index} does not match {index}"
            )
        self._last_index = index
        return observation

    def snapshot(self) -> BookReplayState:
        self._send({"type": "snapshot", "index": self._last_index})
        message = self._receive("snapshot")
        if message.get("index") != self._last_index:
            self._broken = True
            raise BookReplayProtocolError("candidate snapshot reported the wrong event index")
        return BookReplayState.from_dict(message.get("state", {}))

    def close(self) -> None:
        if self._closed:
            return
        close_error: Optional[BookReplayProtocolError] = None
        was_broken = self._broken
        try:
            if not self._broken and self._process.poll() is None:
                self._send({"type": "finish", "event_count": self._last_index})
                complete = self._receive("complete")
                if complete.get("event_count") != self._last_index:
                    raise BookReplayProtocolError(
                        "candidate complete message reported the wrong event_count"
                    )
            elif not self._broken:
                raise BookReplayProtocolError("candidate exited before finish")
        except (BookReplayProtocolError, OSError) as exc:
            self._broken = True
            close_error = (
                exc
                if isinstance(exc, BookReplayProtocolError)
                else BookReplayProtocolError(f"candidate shutdown failed: {exc}")
            )
        finally:
            self._closed = True
            shutdown_timed_out = self._shutdown(
                timeout_seconds=self.timeout_seconds if not self._broken else 0.5
            )
        if not was_broken and close_error is None:
            if shutdown_timed_out:
                close_error = BookReplayProtocolError(
                    f"candidate timed out exiting after complete ({self.timeout_seconds:g}s)"
                )
            elif self._process.returncode != 0:
                close_error = BookReplayProtocolError(
                    f"candidate exited with code {self._process.returncode} after complete"
                )
        if close_error is not None:
            raise close_error

    def _shutdown(self, *, timeout_seconds: float = 0.5) -> bool:
        timed_out = False
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=timeout_seconds)
            except subprocess.TimeoutExpired:
                timed_out = True
                self._process.terminate()
                try:
                    self._process.wait(timeout=0.5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=0.5)
        if self._process.stdin is not None:
            try:
                self._process.stdin.close()
            except OSError:
                pass
        self._stdout_thread.join(timeout=0.2)
        self._stderr_thread.join(timeout=0.2)
        for stream in (self._process.stdout, self._process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass
        return timed_out


class ExternalBookReplayAdapterFactory:
    """Start a fresh external book adapter for each comparison."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 5.0,
        cwd: Optional[str | Path] = None,
    ) -> None:
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def __call__(self, config: BookReplayConfig) -> ExternalBookReplayAdapter:
        return ExternalBookReplayAdapter(
            self.command,
            config,
            timeout_seconds=self.timeout_seconds,
            cwd=self.cwd,
        )


__all__ = [
    "BookReplayProtocolError",
    "ExternalBookReplayAdapter",
    "ExternalBookReplayAdapterFactory",
]
