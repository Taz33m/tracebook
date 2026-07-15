import hashlib
import json
import struct
from pathlib import Path

import pytest

from integrations.flash_benchmark.bridge import (
    FLASH_ARTIFACT_TYPE,
    FlashBridgeError,
    convert_divergent_prefix,
    load_divergent_sequence,
)
from tracebook.conformance import is_partial_fill_priority_probe
from tracebook.events import load_market_events

ROOT = Path(__file__).resolve().parents[1]
FLASH_INTEGRATION = ROOT / "integrations" / "flash_benchmark"
REAL_DIVERGENCE = FLASH_INTEGRATION / "artifacts" / "orderbook-rs-issue-88-divergence.json"
REAL_PROVENANCE = FLASH_INTEGRATION / "artifacts" / "orderbook-rs-issue-88-provenance.json"
REAL_REDUCED_TRACE = (
    ROOT / "integrations" / "orderbook_rs" / "regressions" / "flash-issue-88-reduced.jsonl"
)
HEADER = struct.Struct("<QII")
RECORD = struct.Struct("<BBBBIQQqq")


def _write_divergence(path: Path, sequence: int = 2, **overrides) -> None:
    payload = {
        "schema_version": 1,
        "artifact_type": FLASH_ARTIFACT_TYPE,
        "conformant": False,
        "first_divergent_sequence": sequence,
        "matching_sequences": sequence,
        "reference": {"path": "reference.txt", "reports": []},
        "candidate": {"path": "candidate.txt", "reports": []},
    }
    payload.update(overrides)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _record(
    kind: int,
    side: int,
    ioc: int,
    quantity: int,
    sequence: int,
    order_id: int,
    price: int,
) -> bytes:
    return RECORD.pack(kind, side, ioc, 0, quantity, sequence, order_id, price, 0)


def _write_workload(path: Path, records: list[bytes], tail: bytes = b"") -> None:
    path.write_bytes(HEADER.pack(0x4D4542575F303031, 1, len(records)) + b"".join(records) + tail)


def test_flash_bridge_converts_the_selected_dense_prefix(tmp_path):
    divergence = tmp_path / "divergence.json"
    workload = tmp_path / "orders_normal_s23_n4.bin"
    output = tmp_path / "prefix.jsonl"
    _write_divergence(divergence)
    _write_workload(
        workload,
        [
            _record(0, 0, 0, 10, 0, 101, 100),
            _record(2, 0, 0, 12, 1, 101, 101),
            _record(1, 0, 0, 10, 2, 101, 100),
            _record(0, 1, 1, 5, 3, 202, 99),
        ],
    )

    result = convert_divergent_prefix(divergence, workload, output, symbol="MEB")

    assert result.first_divergent_sequence == 2
    assert result.event_count == 3
    assert result.output_sha256 == "sha256:" + hashlib.sha256(output.read_bytes()).hexdigest()
    events = load_market_events(output)
    assert [event.op for event in events] == ["new", "replace", "cancel"]
    assert events[0].to_dict() == {
        "op": "new",
        "symbol": "MEB",
        "order_id": 101,
        "side": "BUY",
        "order_type": "LIMIT",
        "price": 100.0,
        "quantity": 10.0,
        "owner": -1,
        "timestamp_ns": None,
    }
    assert events[1].price == 101
    assert events[1].quantity == 12


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"schema_version": 2}, "schema_version"),
        ({"artifact_type": "unknown"}, "artifact_type"),
        ({"conformant": True}, "must describe a canonical divergence"),
        ({"first_divergent_sequence": None}, "non-negative integer"),
    ],
)
def test_flash_bridge_rejects_unsupported_divergence_artifacts(tmp_path, overrides, message):
    divergence = tmp_path / "divergence.json"
    _write_divergence(divergence, **overrides)

    with pytest.raises(FlashBridgeError, match=message):
        load_divergent_sequence(divergence)


def test_flash_bridge_rejects_trailing_or_truncated_workloads(tmp_path):
    divergence = tmp_path / "divergence.json"
    workload = tmp_path / "orders.bin"
    output = tmp_path / "prefix.jsonl"
    _write_divergence(divergence, sequence=0)
    records = [_record(0, 0, 0, 10, 0, 101, 100)]

    _write_workload(workload, records, tail=b"x")
    with pytest.raises(FlashBridgeError, match="size does not match"):
        convert_divergent_prefix(divergence, workload, output)

    workload.write_bytes(HEADER.pack(0x4D4542575F303031, 1, 1) + records[0][:-1])
    with pytest.raises(FlashBridgeError, match="size does not match"):
        convert_divergent_prefix(divergence, workload, output)


def test_flash_bridge_preserves_existing_output_on_invalid_sequence(tmp_path):
    divergence = tmp_path / "divergence.json"
    workload = tmp_path / "orders.bin"
    output = tmp_path / "prefix.jsonl"
    _write_divergence(divergence, sequence=1)
    _write_workload(
        workload,
        [
            _record(0, 0, 0, 10, 0, 101, 100),
            _record(1, 0, 0, 10, 7, 101, 100),
        ],
    )
    output.write_text("existing\n", encoding="utf-8")

    with pytest.raises(FlashBridgeError, match="carries sequence 7"):
        convert_divergent_prefix(divergence, workload, output)

    assert output.read_text(encoding="utf-8") == "existing\n"
    assert list(tmp_path.glob(f".{output.name}.*.tmp")) == []


def test_flash_bridge_rejects_a_price_that_would_lose_numeric_precision(tmp_path):
    divergence = tmp_path / "divergence.json"
    workload = tmp_path / "orders.bin"
    output = tmp_path / "prefix.jsonl"
    _write_divergence(divergence, sequence=0)
    _write_workload(workload, [_record(0, 0, 0, 1, 0, 101, 1 << 53)])

    with pytest.raises(FlashBridgeError, match="represented exactly"):
        convert_divergent_prefix(divergence, workload, output)


def test_real_flash_artifact_pins_the_first_orderbook_rs_divergence():
    artifact = json.loads(REAL_DIVERGENCE.read_text(encoding="utf-8"))

    assert load_divergent_sequence(REAL_DIVERGENCE) == 15738
    assert artifact["matching_sequences"] == 15738
    assert artifact["reference"]["first_divergent_line"] == 17449
    assert artifact["candidate"]["first_divergent_line"] == 17449
    assert artifact["reference"]["reports"][-2:] == [
        "1,15738,33532,55,146075,933919",
        "1,15738,33532,3,185199,933919",
    ]
    assert artifact["candidate"]["reports"][-2:] == [
        "1,15738,33532,14,185199,933919",
        "1,15738,33532,44,146075,933919",
    ]


def test_real_flash_reducer_output_is_a_crossed_partial_fill_priority_probe():
    events = load_market_events(REAL_REDUCED_TRACE)

    assert len(events) == 4
    assert is_partial_fill_priority_probe(events)
    assert events[2].price == 33531
    assert events[3].order_type.name == "IOC"


def test_real_flash_provenance_hashes_the_durable_evidence():
    provenance = json.loads(REAL_PROVENANCE.read_text(encoding="utf-8"))

    assert provenance["flash"]["commit"] == "eb6e89fbc4313f77cea7d424bab14a26093cf552"
    assert provenance["handoff"] == {
        "converted_prefix_event_count": 15739,
        "converted_prefix_trace_sha256": (
            "sha256:4e0bb497924a68dcca9575a5860954089bf5d137ab1f577618fba482a19877cd"
        ),
        "tracebook_divergence_event": 15739,
    }
    assert provenance["minimization"]["runs"] == 193
    assert provenance["minimization"]["one_minimal"] is True
    assert provenance["minimization"]["budget_exhausted"] is False
    assert (
        provenance["canonical_evidence"]["divergence_artifact_sha256"]
        == hashlib.sha256(REAL_DIVERGENCE.read_bytes()).hexdigest()
    )
    assert provenance["minimization"]["reduced_trace_sha256"] == (
        "sha256:" + hashlib.sha256(REAL_REDUCED_TRACE.read_bytes()).hexdigest()
    )
