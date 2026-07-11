import asyncio
import copy
import json
import time
from pathlib import Path

import pytest

from tracebook.corpus import (
    CoinbaseCorpusError,
    CoinbaseSanitizer,
    benchmark_coinbase_corpus,
    capture_coinbase_corpus_async,
    compare_corpus_benchmarks,
    copy_bundled_coinbase_corpus,
    prepare_coinbase_corpus,
    verify_coinbase_corpus,
)
from tracebook.corpus.cli import main

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "examples/data/coinbase_btcusd_l3_snapshot.json"
FEED = ROOT / "examples/data/coinbase_btcusd_full.jsonl"
CORPUS = ROOT / "src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1"
FIXED_KEY = b"tracebook-corpus-test-key-00001"


def _prepare(tmp_path, name="corpus"):
    output = tmp_path / name
    manifest = prepare_coinbase_corpus(
        SNAPSHOT,
        FEED,
        output,
        product_id="BTC-USD",
        tick_size="0.01",
        source_classification="synthetic",
        created_at="2026-07-10T12:00:00Z",
        id_key=FIXED_KEY,
    )
    return output, manifest


def test_sanitizer_allowlists_fields_and_pseudonymizes_consistently():
    sanitizer = CoinbaseSanitizer("BTC-USD", id_key=FIXED_KEY)
    venue_id = "11111111-1111-4111-8111-111111111111"
    opened = sanitizer.sanitize_message(
        {
            "type": "open",
            "product_id": "BTC-USD",
            "sequence": 11,
            "order_id": venue_id,
            "side": "buy",
            "price": "100",
            "remaining_size": "1",
            "time": "2026-01-01T00:00:00Z",
            "profile_id": "must-not-persist",
            "user_id": "must-not-persist",
            "client_oid": "must-not-persist",
        }
    )
    done = sanitizer.sanitize_message(
        {
            "type": "done",
            "product_id": "BTC-USD",
            "sequence": 12,
            "order_id": venue_id,
            "reason": "canceled",
        }
    )

    assert opened["order_id"].startswith("cb_")
    assert opened["order_id"] == done["order_id"]
    assert "profile_id" not in opened
    assert "user_id" not in opened
    assert "client_oid" not in opened
    assert (
        sanitizer.sanitize_message({"type": "noop", "product_id": "ETH-USD", "sequence": 99})
        is None
    )
    assert sanitizer.stats.fields_removed == 3
    assert sanitizer.stats.frames_dropped_other_products == 1


def test_compact_sanitizer_rewrites_schema_when_removing_fields():
    sanitizer = CoinbaseSanitizer("BTC-USD", id_key=FIXED_KEY)
    schema = sanitizer.sanitize_message(
        {
            "type": "level3",
            "schema": {
                "open": [
                    "type",
                    "product_id",
                    "sequence",
                    "order_id",
                    "side",
                    "price",
                    "size",
                    "time",
                    "client_oid",
                ]
            },
        }
    )
    frame = sanitizer.sanitize_message(
        [
            "open",
            "BTC-USD",
            "11",
            "11111111-1111-4111-8111-111111111111",
            "buy",
            "100",
            "1",
            "2026-01-01T00:00:00Z",
            "must-not-persist",
        ]
    )

    assert "client_oid" not in schema["schema"]["open"]
    assert len(frame) == len(schema["schema"]["open"])
    order_id_index = schema["schema"]["open"].index("order_id")
    assert frame[order_id_index].startswith("cb_")


def test_prepare_verify_and_tamper_detection(tmp_path):
    corpus, manifest = _prepare(tmp_path)

    assert set(path.name for path in corpus.iterdir()) == {
        "snapshot.json",
        "feed.jsonl",
        "events.jsonl",
        "golden.json",
        "manifest.json",
    }
    assert manifest["rights"]["contains_market_data"] is False
    assert manifest["rights"]["redistribution"] == "project_fixture"
    assert manifest["sanitization"]["pseudonymization_key_persisted"] is False
    assert manifest["sanitization"]["pseudonymization_scope"] == "deterministic_fixture"
    assert manifest["source"]["snapshot_sequence"] == 100
    assert manifest["source"]["final_sequence"] == 109
    verification = verify_coinbase_corpus(corpus)
    assert verification["verified"] is True
    assert verification["events_verified"] == 8

    rendered = "".join(path.read_text(encoding="utf-8") for path in corpus.iterdir())
    assert "22222222-2222-4222-8222-222222222222" not in rendered

    with (corpus / "feed.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(CoinbaseCorpusError, match="(byte count|sha256) mismatch"):
        verify_coinbase_corpus(corpus)


def test_checked_synthetic_corpus_reproduces_exactly():
    verification = verify_coinbase_corpus(CORPUS)

    assert verification == {
        "schema_version": 1,
        "verified": True,
        "corpus_id": "sha256:ca32b618dda7906e7fe85fbddd9ca03fd25075cc9cc0adc071c72226462a82ec",
        "files_verified": 4,
        "events_verified": 8,
        "snapshot_sequence": 100,
        "final_sequence": 109,
    }


def test_checked_synthetic_corpus_regenerates_byte_for_byte(tmp_path):
    regenerated = tmp_path / "regenerated"
    prepare_coinbase_corpus(
        SNAPSHOT,
        FEED,
        regenerated,
        product_id="BTC-USD",
        tick_size="0.01",
        channel="full",
        source_classification="synthetic",
        source_environment="synthetic",
        created_at="2026-07-10T12:00:00Z",
    )

    for expected in CORPUS.iterdir():
        assert (regenerated / expected.name).read_bytes() == expected.read_bytes()


def test_bundled_synthetic_corpus_can_be_materialized(tmp_path):
    output = tmp_path / "sample"
    manifest = copy_bundled_coinbase_corpus(output)

    assert manifest["corpus_id"].startswith("sha256:")
    assert verify_coinbase_corpus(output)["verified"] is True
    for expected in CORPUS.iterdir():
        assert (output / expected.name).read_bytes() == expected.read_bytes()


def test_prepare_is_atomic_and_refuses_existing_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(CoinbaseCorpusError, match="already exists"):
        prepare_coinbase_corpus(
            SNAPSHOT,
            FEED,
            output,
            product_id="BTC-USD",
            tick_size="0.01",
        )
    assert marker.read_text(encoding="utf-8") == "keep"


def test_manifest_rejects_unsafe_paths_and_duplicate_roles(tmp_path):
    corpus, manifest = _prepare(tmp_path)
    manifest_path = corpus / "manifest.json"

    unsafe = copy.deepcopy(manifest)
    unsafe["files"][0]["path"] = "../snapshot.json"
    manifest_path.write_text(json.dumps(unsafe), encoding="utf-8")
    with pytest.raises(CoinbaseCorpusError, match="unsafe"):
        verify_coinbase_corpus(corpus)

    duplicate = copy.deepcopy(manifest)
    duplicate["files"].append(copy.deepcopy(duplicate["files"][0]))
    manifest_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(CoinbaseCorpusError, match="file roles"):
        verify_coinbase_corpus(corpus)

    rights_tamper = copy.deepcopy(manifest)
    rights_tamper["rights"]["redistribution"] = "granted"
    manifest_path.write_text(json.dumps(rights_tamper), encoding="utf-8")
    with pytest.raises(CoinbaseCorpusError, match="corpus_id"):
        verify_coinbase_corpus(corpus)


def test_benchmark_report_and_environment_aware_comparison(tmp_path):
    corpus, _ = _prepare(tmp_path)
    baseline = benchmark_coinbase_corpus(corpus, iterations=2, warmups=0)
    candidate = copy.deepcopy(baseline)
    candidate["environment"]["python"] = "different"
    candidate["environment"]["dependency_versions"]["tracebook-sim"] = "candidate"
    for phase in candidate["phases"].values():
        phase["median_ns"] *= 0.5
        phase["events_per_second_median"] *= 2.0

    comparison = compare_corpus_benchmarks(baseline, candidate)

    assert baseline["measurement_model"] == "local_wall_clock"
    assert len(baseline["phases"]["replay_only"]["samples_ns"]) == 2
    assert comparison["environment_match"] is False
    assert comparison["manifest_match"] is True
    assert comparison["environment_differences"] == ["python"]
    assert comparison["software_differences"] == ["tracebook-sim"]
    assert comparison["phases"]["replay_only"]["speedup_baseline_over_candidate"] == 2.0
    assert comparison["phases"]["replay_only"]["throughput_change_percent"] == 100.0

    mismatch = copy.deepcopy(candidate)
    mismatch["corpus"]["corpus_id"] = "sha256:" + "0" * 64
    with pytest.raises(CoinbaseCorpusError, match="same corpus_id"):
        compare_corpus_benchmarks(baseline, mismatch)


def test_benchmark_comparison_rejects_malformed_reports(tmp_path, capsys):
    corpus, _ = _prepare(tmp_path)
    baseline = benchmark_coinbase_corpus(corpus, iterations=1, warmups=0)
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")
    array_path = tmp_path / "array.json"
    array_path.write_text("[]", encoding="utf-8")

    code = main(["compare", str(array_path), str(baseline_path)])

    assert code == 2
    assert "JSON object" in capsys.readouterr().err

    for field, value in (
        ("median_ns", "nan"),
        ("events_per_second_median", float("inf")),
        ("median_ns", 10**400),
    ):
        malformed = copy.deepcopy(baseline)
        malformed["phases"]["replay_only"][field] = value
        with pytest.raises(CoinbaseCorpusError, match="positive finite number"):
            compare_corpus_benchmarks(baseline, malformed)


class _FakeWebSocket:
    def __init__(self, frames):
        self.frames = list(frames)
        self.sent = []

    async def send(self, payload):
        self.sent.append(json.loads(payload))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.frames:
            await asyncio.sleep(60)
        return json.dumps(self.frames.pop(0))


class _FakeConnection:
    def __init__(self, websocket):
        self.websocket = websocket

    async def __aenter__(self):
        return self.websocket

    async def __aexit__(self, exc_type, exc, traceback):
        return False


def test_live_capture_queues_feed_before_snapshot_and_marks_rights(tmp_path):
    websocket = _FakeWebSocket(
        [
            {"type": "subscriptions", "channels": []},
            {
                "type": "open",
                "product_id": "BTC-USD",
                "sequence": 11,
                "order_id": "11111111-1111-4111-8111-111111111111",
                "side": "buy",
                "price": "100",
                "remaining_size": "1",
                "time": "2026-01-01T00:00:00Z",
                "user_id": "must-not-persist",
            },
        ]
    )

    def connector(url, **kwargs):
        assert url.startswith("wss://ws-feed.exchange.coinbase.com")
        assert kwargs["max_size"] > 0
        return _FakeConnection(websocket)

    def snapshot_fetcher(product_id, environment, timeout):
        assert product_id == "BTC-USD"
        assert environment == "production"
        assert timeout == 1.0
        return {
            "product_id": product_id,
            "sequence": 10,
            "bids": [],
            "asks": [],
        }

    corpus = tmp_path / "live"
    manifest = asyncio.run(
        capture_coinbase_corpus_async(
            corpus,
            product_id="BTC-USD",
            tick_size="0.01",
            channel="full",
            post_snapshot_seconds=0.01,
            max_messages=10,
            snapshot_timeout=1.0,
            acknowledge_market_data_terms=True,
            id_key=FIXED_KEY,
            websocket_connector=connector,
            snapshot_fetcher=snapshot_fetcher,
        )
    )

    assert websocket.sent == [
        {"type": "subscribe", "product_ids": ["BTC-USD"], "channels": ["full"]}
    ]
    assert manifest["capture"]["stop_reason"] == "duration"
    assert manifest["rights"]["redistribution"] == "not_granted"
    assert verify_coinbase_corpus(corpus)["final_sequence"] == 11
    assert "must-not-persist" not in (corpus / "feed.jsonl").read_text(encoding="utf-8")


def test_live_capture_rejects_a_pre_snapshot_message_limit(tmp_path):
    websocket = _FakeWebSocket([{"type": "subscriptions", "channels": []}])

    def connector(url, **kwargs):
        return _FakeConnection(websocket)

    def delayed_snapshot(product_id, environment, timeout):
        time.sleep(0.05)
        return {
            "product_id": product_id,
            "sequence": 10,
            "bids": [],
            "asks": [],
        }

    corpus = tmp_path / "too-short"
    with pytest.raises(CoinbaseCorpusError, match="before the REST snapshot"):
        asyncio.run(
            capture_coinbase_corpus_async(
                corpus,
                product_id="BTC-USD",
                tick_size="0.01",
                channel="full",
                post_snapshot_seconds=1.0,
                max_messages=1,
                snapshot_timeout=1.0,
                acknowledge_market_data_terms=True,
                id_key=FIXED_KEY,
                websocket_connector=connector,
                snapshot_fetcher=delayed_snapshot,
            )
        )

    assert not corpus.exists()
    assert list(tmp_path.iterdir()) == []


def test_capture_cli_requires_market_data_acknowledgement(tmp_path, capsys):
    code = main(
        [
            "capture",
            str(tmp_path / "capture"),
            "--product-id",
            "BTC-USD",
            "--tick-size",
            "0.01",
        ]
    )

    assert code == 2
    assert "acknowledge_market_data_terms" in capsys.readouterr().err
