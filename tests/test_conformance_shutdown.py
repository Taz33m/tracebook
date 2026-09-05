"""A complete frame is valid evidence only when the peer exits cleanly in time."""

import json
import os
import sys
import time

import pytest

from tracebook.conformance import (
    AdapterProtocolError,
    ConformanceConfig,
    ExternalProcessAdapter,
    run_conformance,
)
from tracebook.conformance.exit_codes import exit_code_for_artifact


def _shutdown_peer(delay, exit_code, *, handle_terminate=False, broken_snapshot=False):
    script = """
import json
import signal
import sys
import time

delay, exit_code, handle_terminate, broken_snapshot = json.loads(sys.argv[1])
if handle_terminate:
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
hello = json.loads(sys.stdin.readline())
print(json.dumps({
    "type": "ready",
    "protocol": hello["protocol"],
    "protocol_version": hello["protocol_version"],
    "engine": {"name": "shutdown-test", "version": "1", "language": "Python"},
}), flush=True)
for line in sys.stdin:
    message = json.loads(line)
    if message["type"] == "snapshot":
        if broken_snapshot:
            print("broken snapshot", flush=True)
            time.sleep(60)
        else:
            print(json.dumps({"type": "snapshot", "index": 0, "state": {"books": []}}), flush=True)
    elif message["type"] == "finish":
        print(json.dumps({"type": "complete", "event_count": message["event_count"]}), flush=True)
        time.sleep(delay)
        sys.exit(exit_code)
"""
    return ExternalProcessAdapter(
        [
            sys.executable,
            "-I",
            "-u",
            "-c",
            script,
            json.dumps([delay, exit_code, handle_terminate, broken_snapshot]),
        ],
        ConformanceConfig(),
        timeout_seconds=10,
    )


def test_close_allows_configured_time_for_clean_exit():
    adapter = _shutdown_peer(0.75, 0)

    adapter.close()
    adapter.close()

    assert adapter._process.returncode == 0


def test_close_rejects_nonzero_exit_after_complete():
    adapter = _shutdown_peer(0, 7)

    with pytest.raises(AdapterProtocolError, match="exited with code 7 after complete"):
        adapter.close()

    assert adapter._process.returncode == 7


@pytest.mark.skipif(os.name == "nt", reason="requires a catchable POSIX SIGTERM")
def test_close_rejects_timeout_even_when_termination_exits_zero():
    adapter = _shutdown_peer(60, 0, handle_terminate=True)
    # Give startup its own generous budget before testing just the exit bound.
    adapter.timeout_seconds = 0.5

    with pytest.raises(AdapterProtocolError, match="timed out exiting after complete"):
        adapter.close()

    assert adapter._process.returncode == 0
    adapter.close()


def test_forced_shutdown_invalidates_otherwise_conformant_report():
    adapter = _shutdown_peer(60, 0, handle_terminate=True)
    adapter.timeout_seconds = 0.5

    report = run_conformance([], lambda config: adapter)

    assert report.conformant is False
    assert report.divergence.kind == "adapter_close_error"
    assert "timed out exiting after complete" in report.divergence.message
    assert report.operational_failure is True
    assert exit_code_for_artifact(report.to_dict()) == 2
    assert adapter._process.returncode is not None


def test_broken_stream_keeps_original_error_and_uses_prompt_cleanup():
    adapter = _shutdown_peer(60, 0, handle_terminate=True, broken_snapshot=True)
    started = time.monotonic()

    report = run_conformance([], lambda config: adapter)

    assert report.conformant is False
    assert report.divergence.kind == "adapter_error"
    assert "not valid JSON" in report.divergence.message
    assert report.divergence.close_error is None
    assert report.operational_failure is True
    assert exit_code_for_artifact(report.to_dict()) == 2
    assert adapter._process.returncode is not None
    # A broken stream should not consume the healthy peer's ten-second budget.
    assert time.monotonic() - started < 5
