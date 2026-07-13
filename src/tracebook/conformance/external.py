"""External-process adapter for Rust, C++, Java, Python, or any NDJSON peer."""

from __future__ import annotations

import json
import math
import queue
import subprocess  # nosec B404
import threading
from collections import deque
from pathlib import Path
from typing import Deque, Mapping, Optional, Sequence

from ..events import MarketEvent
from .model import (
    PROTOCOL_NAME,
    PROTOCOL_VERSION,
    BookState,
    ConformanceConfig,
    ConformanceError,
    EngineMetadata,
    Observation,
)

_END_OF_STREAM = object()


class AdapterProtocolError(ConformanceError):
    """Raised when an external adapter violates or times out on the protocol."""


class ExternalProcessAdapter:
    """Drive one external engine adapter over newline-delimited JSON."""

    def __init__(
        self,
        command: Sequence[str],
        config: ConformanceConfig,
        timeout_seconds: float = 5.0,
        cwd: Optional[str | Path] = None,
    ) -> None:
        if isinstance(command, (str, bytes)) or not command:
            raise ConformanceError("candidate command must be a non-empty argument list")
        normalized_command = []
        for argument in command:
            if not isinstance(argument, str) or not argument:
                raise ConformanceError("candidate command arguments must be non-empty strings")
            normalized_command.append(argument)
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, (int, float))
            or not math.isfinite(float(timeout_seconds))
            or timeout_seconds <= 0
        ):
            raise ConformanceError("timeout_seconds must be a positive finite number")

        self.command = tuple(normalized_command)
        self.timeout_seconds = float(timeout_seconds)
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
            raise AdapterProtocolError(
                f"unable to start candidate command {self.command!r}: {exc}"
            ) from exc
        stdout = self._process.stdout
        stderr = self._process.stderr
        if stdout is None or stderr is None:
            self._process.kill()
            self._process.wait()
            raise AdapterProtocolError("candidate process pipes were not created")
        self._stdout_thread = threading.Thread(
            target=self._read_stdout,
            args=(stdout,),
            name="tracebook-conformance-stdout",
            daemon=True,
        )
        self._stderr_thread = threading.Thread(
            target=self._read_stderr,
            args=(stderr,),
            name="tracebook-conformance-stderr",
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
                raise AdapterProtocolError("ready message reported the wrong protocol")
            if ready.get("protocol_version") != PROTOCOL_VERSION:
                raise AdapterProtocolError("ready message reported the wrong protocol version")
            self.metadata = EngineMetadata.from_dict(ready.get("engine", {}))
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
            text = "".join(self._stderr_lines).strip()
        if not text:
            return ""
        return text[-4000:]

    def _send(self, payload: Mapping[str, object]) -> None:
        if self._closed or self._process.poll() is not None:
            raise AdapterProtocolError(f"candidate process is not running{self._process_context()}")
        process_input = self._process.stdin
        if process_input is None:
            self._broken = True
            raise AdapterProtocolError("candidate process input pipe is unavailable")
        try:
            process_input.write(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n")
            process_input.flush()
        except (BrokenPipeError, OSError) as exc:
            self._broken = True
            raise AdapterProtocolError(
                f"candidate process closed its input{self._process_context()}"
            ) from exc

    def _receive(self, expected_type: str) -> dict:
        try:
            item = self._stdout_queue.get(timeout=self.timeout_seconds)
        except queue.Empty as exc:
            self._broken = True
            raise AdapterProtocolError(
                f"candidate timed out waiting for {expected_type!r} after "
                f"{self.timeout_seconds:g}s{self._process_context()}"
            ) from exc
        if item is _END_OF_STREAM:
            self._broken = True
            raise AdapterProtocolError(
                f"candidate exited before {expected_type!r}{self._process_context()}"
            )
        if not isinstance(item, str):
            self._broken = True
            raise AdapterProtocolError("candidate produced an invalid protocol frame")
        try:
            message = json.loads(item)
        except json.JSONDecodeError as exc:
            self._broken = True
            raise AdapterProtocolError(
                f"candidate stdout was not valid JSON: {item.strip()!r}"
            ) from exc
        if not isinstance(message, dict):
            self._broken = True
            raise AdapterProtocolError("candidate protocol frame must be a JSON object")
        if message.get("type") == "error":
            self._broken = True
            raise AdapterProtocolError(
                f"candidate reported {message.get('code', 'ERROR')}: "
                f"{message.get('message', '')}"
            )
        if message.get("type") != expected_type:
            self._broken = True
            raise AdapterProtocolError(
                f"expected candidate message {expected_type!r}, "
                f"received {message.get('type')!r}"
            )
        return message

    def _process_context(self) -> str:
        return_code = self._process.poll()
        details = f" (exit code {return_code})" if return_code is not None else ""
        stderr = self._stderr_tail()
        if stderr:
            details += f"; stderr: {stderr}"
        return details

    def apply(self, event: MarketEvent, index: int) -> Observation:
        self._send({"type": "event", "index": index, "event": event.to_dict()})
        message = self._receive("observation")
        observation = Observation.from_dict(message)
        if observation.index != index:
            self._broken = True
            raise AdapterProtocolError(
                f"candidate observation index {observation.index} does not match {index}"
            )
        self._last_index = index
        return observation

    def snapshot(self) -> BookState:
        self._send({"type": "snapshot", "index": self._last_index})
        message = self._receive("snapshot")
        if message.get("index") != self._last_index:
            self._broken = True
            raise AdapterProtocolError("candidate snapshot reported the wrong event index")
        return BookState.from_dict(message.get("state", {}))

    def close(self) -> None:
        if self._closed:
            return
        close_error: Optional[AdapterProtocolError] = None
        was_broken = self._broken
        try:
            if not self._broken and self._process.poll() is None:
                self._send({"type": "finish", "event_count": self._last_index})
                complete = self._receive("complete")
                if complete.get("event_count") != self._last_index:
                    raise AdapterProtocolError(
                        "candidate complete message reported the wrong event_count"
                    )
            elif not self._broken:
                raise AdapterProtocolError("candidate exited before finish")
        except (AdapterProtocolError, OSError) as exc:
            self._broken = True
            close_error = (
                exc
                if isinstance(exc, AdapterProtocolError)
                else AdapterProtocolError(f"candidate shutdown failed: {exc}")
            )
        finally:
            self._closed = True
            self._shutdown()
        if not was_broken and close_error is None and self._process.returncode != 0:
            close_error = AdapterProtocolError(
                f"candidate exited with code {self._process.returncode} after complete"
            )
        if close_error is not None:
            raise close_error

    def _shutdown(self) -> None:
        if self._process.poll() is None:
            try:
                self._process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
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


class ExternalProcessAdapterFactory:
    """Reusable factory that starts a fresh external process for every run."""

    def __init__(
        self,
        command: Sequence[str],
        timeout_seconds: float = 5.0,
        cwd: Optional[str | Path] = None,
    ) -> None:
        self.command = tuple(command)
        self.timeout_seconds = timeout_seconds
        self.cwd = cwd

    def __call__(self, config: ConformanceConfig) -> ExternalProcessAdapter:
        return ExternalProcessAdapter(
            self.command,
            config,
            timeout_seconds=self.timeout_seconds,
            cwd=self.cwd,
        )
