import json
from pathlib import Path

import pytest

from tracebook.events import (
    CoinbaseL3Adapter,
    CoinbaseL3Error,
    coinbase_order_id,
    iter_coinbase_l3_messages,
    load_coinbase_l3_snapshot,
    normalize_coinbase_l3,
)
from tracebook.events.coinbase_cli import main
from tracebook.events.market_replay import replay_market_events

ROOT = Path(__file__).parents[1]
SNAPSHOT = ROOT / "examples/data/coinbase_btcusd_l3_snapshot.json"
FEED = ROOT / "examples/data/coinbase_btcusd_full.jsonl"


def _empty_snapshot(sequence=10):
    return {
        "product_id": "BTC-USD",
        "sequence": sequence,
        "bids": [],
        "asks": [],
    }


def _compact_schema():
    return {
        "type": "level3",
        "schema": {
            "change": ["type", "product_id", "sequence", "order_id", "size", "price", "time"],
            "done": ["type", "product_id", "sequence", "order_id", "time"],
            "match": [
                "type",
                "product_id",
                "sequence",
                "maker_order_id",
                "taker_order_id",
                "size",
                "price",
                "time",
            ],
            "noop": ["type", "product_id", "sequence", "time"],
            "open": [
                "type",
                "product_id",
                "sequence",
                "order_id",
                "side",
                "size",
                "price",
                "time",
            ],
        },
    }


def test_full_feed_fixture_normalizes_and_replays_exact_book_state():
    snapshot = load_coinbase_l3_snapshot(SNAPSHOT)
    adapter = CoinbaseL3Adapter(snapshot, strict=True, retain_id_map=True, retain_trades=True)
    events = list(adapter.iter_events(iter_coinbase_l3_messages(FEED)))
    replayed = replay_market_events(events, tick_size=0.01)
    summary = adapter.to_dict(include_trades=True, include_id_map=True)

    assert [event.op for event in events] == [
        "new",
        "new",
        "reduce",
        "new",
        "reduce",
        "replace",
        "cancel",
        "reduce",
    ]
    assert events[2].timestamp_ns == 1783684800000000002
    assert adapter.final_sequence == 109
    assert adapter.sequence_complete is True
    assert adapter.active_order_count == 1
    assert summary["exchange_trades_observed"] == 2
    assert summary["messages_seen"] == 10
    assert summary["messages_sequenced"] == 9
    assert summary["normalized_events"] == 8
    assert summary["id_map"]
    assert replayed.to_dict()["reductions"] == 3
    depth = replayed.manager.get_order_book("BTC-USD").get_order_book_depth()
    assert depth["bids"] == [(100.0, 2.0, 1)]
    assert depth["asks"] == []


def test_compact_level3_uses_announced_field_order():
    maker = "11111111-1111-4111-8111-111111111111"
    taker = "22222222-2222-4222-8222-222222222222"
    snapshot = {
        "product_id": "BTC-USD",
        "sequence": 20,
        "bids": [],
        "asks": [["101", "2", maker]],
    }
    messages = [
        _compact_schema(),
        ["match", "BTC-USD", "21", maker, taker, "0.5", "101", "2026-01-01T00:00:00Z"],
        ["change", "BTC-USD", "22", maker, "1.0", "101", "2026-01-01T00:00:01Z"],
        ["done", "BTC-USD", "23", maker, "2026-01-01T00:00:02Z"],
    ]

    adapter, events = normalize_coinbase_l3(snapshot, messages, retain_trades=True)

    assert [event.op for event in events] == ["new", "reduce", "reduce", "cancel"]
    assert [event.quantity for event in events[1:3]] == [0.5, 0.5]
    assert adapter.active_order_count == 0
    assert adapter.exchange_trades[0].normalized_taker_order_id == coinbase_order_id(
        "BTC-USD", taker
    )


def test_sequence_gap_is_fatal_by_default_and_explicit_in_lenient_output():
    message = {
        "type": "open",
        "product_id": "BTC-USD",
        "sequence": 12,
        "order_id": "11111111-1111-4111-8111-111111111111",
        "side": "buy",
        "price": "100",
        "remaining_size": "1",
        "time": "2026-01-01T00:00:00Z",
    }
    strict = CoinbaseL3Adapter(_empty_snapshot())
    with pytest.raises(CoinbaseL3Error, match="sequence gap"):
        list(strict.iter_events([message]))

    lenient = CoinbaseL3Adapter(_empty_snapshot(), strict=False)
    events = list(lenient.iter_events([message]))

    assert len(events) == 1
    assert lenient.sequence_complete is False
    assert lenient.final_sequence == 12
    assert lenient.issues[0].kind == "sequence_gap"


def test_duplicate_and_other_product_messages_do_not_break_target_sequence():
    target = {
        "type": "noop",
        "product_id": "BTC-USD",
        "sequence": 11,
        "time": "2026-01-01T00:00:00Z",
    }
    messages = [
        {**target, "product_id": "ETH-USD", "sequence": 999},
        target,
        target,
    ]
    adapter, events = normalize_coinbase_l3(_empty_snapshot(), messages)

    assert events == []
    assert adapter.final_sequence == 11
    assert adapter.sequence_complete is True
    assert adapter.messages_ignored == 3


def test_unknown_message_types_are_sequence_checked_then_ignored():
    messages = [
        {
            "type": "future_message",
            "product_id": "BTC-USD",
            "sequence": 11,
            "time": "2026-01-01T00:00:00Z",
            "future_field": "ignored",
        },
        {
            "type": "noop",
            "product_id": "BTC-USD",
            "sequence": 12,
            "time": "2026-01-01T00:00:01Z",
        },
    ]

    adapter, events = normalize_coinbase_l3(_empty_snapshot(), messages)

    assert events == []
    assert adapter.final_sequence == 12
    assert adapter.sequence_complete is True
    assert adapter.messages_sequenced == 2
    assert adapter.messages_ignored == 2

    with pytest.raises(CoinbaseL3Error, match="sequence gap"):
        normalize_coinbase_l3(
            _empty_snapshot(),
            [{"type": "future_message", "product_id": "BTC-USD", "sequence": 12}],
        )

    unsequenced, events = normalize_coinbase_l3(
        _empty_snapshot(), [{"type": "future_control_message"}]
    )
    assert events == []
    assert unsequenced.final_sequence == 10
    assert unsequenced.messages_ignored == 1


def test_compact_messages_require_a_complete_preceding_schema():
    adapter = CoinbaseL3Adapter(_empty_snapshot())
    with pytest.raises(CoinbaseL3Error, match="before its schema"):
        list(adapter.iter_events([["noop", "BTC-USD", "11", "2026-01-01T00:00:00Z"]]))

    broken = _compact_schema()
    del broken["schema"]["match"][-1]
    adapter = CoinbaseL3Adapter(_empty_snapshot())
    with pytest.raises(CoinbaseL3Error, match="missing"):
        list(adapter.iter_events([broken]))


def test_nonresting_done_and_change_are_ignored_but_unknown_match_is_not():
    missing = "11111111-1111-4111-8111-111111111111"
    inactive = [
        {
            "type": "change",
            "product_id": "BTC-USD",
            "sequence": 11,
            "order_id": missing,
            "new_size": "1",
        },
        {
            "type": "done",
            "product_id": "BTC-USD",
            "sequence": 12,
            "order_id": missing,
            "reason": "filled",
        },
    ]
    adapter, events = normalize_coinbase_l3(_empty_snapshot(), inactive)
    assert events == []
    assert adapter.messages_ignored == 2

    match = {
        "type": "match",
        "product_id": "BTC-USD",
        "sequence": 11,
        "maker_order_id": missing,
        "taker_order_id": "22222222-2222-4222-8222-222222222222",
        "price": "100",
        "size": "1",
    }
    with pytest.raises(CoinbaseL3Error, match="non-resting maker"):
        normalize_coinbase_l3(_empty_snapshot(), [match])


def test_snapshot_rejects_l2_rows_crosses_and_auctions():
    l2 = {**_empty_snapshot(), "bids": [["100", "1", 2]]}
    with pytest.raises(CoinbaseL3Error, match="not level 3"):
        CoinbaseL3Adapter(l2)

    crossed = {
        **_empty_snapshot(),
        "bids": [["101", "1", "a"]],
        "asks": [["101", "1", "b"]],
    }
    with pytest.raises(CoinbaseL3Error, match="crossed or locked"):
        CoinbaseL3Adapter(crossed)

    with pytest.raises(CoinbaseL3Error, match="auction-mode"):
        CoinbaseL3Adapter({**_empty_snapshot(), "auction_mode": True})


def test_change_increase_or_price_move_becomes_replace():
    venue_id = "11111111-1111-4111-8111-111111111111"
    snapshot = {**_empty_snapshot(), "bids": [["100", "1", venue_id]]}
    change = {
        "type": "change",
        "product_id": "BTC-USD",
        "sequence": 11,
        "order_id": venue_id,
        "side": "buy",
        "reason": "modify_order",
        "old_price": "100",
        "new_price": "99",
        "old_size": "1",
        "new_size": "2",
        "time": "2026-01-01T00:00:00.123456789Z",
    }

    _, events = normalize_coinbase_l3(snapshot, [change])

    assert events[-1].op == "replace"
    assert events[-1].price == 99
    assert events[-1].quantity == 2
    assert events[-1].timestamp_ns == 1767225600123456789


def test_json_loaders_support_wrapped_json_and_bom(tmp_path):
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text("\ufeff" + json.dumps(_empty_snapshot()), encoding="utf-8")
    feed_path = tmp_path / "feed.json"
    feed_path.write_text(
        json.dumps(
            {
                "messages": [
                    {
                        "type": "noop",
                        "product_id": "BTC-USD",
                        "sequence": 11,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_coinbase_l3_snapshot(snapshot_path)["sequence"] == 10
    assert list(iter_coinbase_l3_messages(feed_path))[0]["type"] == "noop"

    bad = tmp_path / "feed.txt"
    bad.write_text("{}", encoding="utf-8")
    with pytest.raises(CoinbaseL3Error, match="must end"):
        list(iter_coinbase_l3_messages(bad))

    empty = tmp_path / "empty.json"
    empty.write_text("[]", encoding="utf-8")
    assert list(iter_coinbase_l3_messages(empty)) == []


def test_cli_streams_normalized_events_and_combined_summary(tmp_path, capsys):
    events_path = tmp_path / "normalized.jsonl"
    summary_path = tmp_path / "summary.json"

    code = main(
        [
            str(SNAPSHOT),
            str(FEED),
            "--tick-size",
            "0.01",
            "--events-output",
            str(events_path),
            "--output",
            str(summary_path),
            "--include-trades",
            "--include-id-map",
        ]
    )

    assert code == 0
    assert "written to" in capsys.readouterr().out
    normalized = [json.loads(line) for line in events_path.read_text().splitlines()]
    summary = json.loads(summary_path.read_text())
    assert len(normalized) == 8
    assert any(event["op"] == "reduce" for event in normalized)
    assert summary["normalization"]["exchange_trades_observed"] == 2
    assert summary["replay"]["schema_version"] == 2
    assert summary["replay"]["books"]["BTC-USD"]["bids"] == [[100.0, 2.0, 1]]


def test_cli_rejects_wrong_tick_without_replacing_existing_event_output(tmp_path, capsys):
    events_path = tmp_path / "normalized.jsonl"
    events_path.write_text("keep-me\n", encoding="utf-8")

    code = main(
        [
            str(SNAPSHOT),
            str(FEED),
            "--tick-size",
            "0.03",
            "--events-output",
            str(events_path),
        ]
    )

    assert code == 2
    assert "not aligned" in capsys.readouterr().err
    assert events_path.read_text(encoding="utf-8") == "keep-me\n"


def test_cli_rejects_feed_that_would_generate_a_simulated_trade(tmp_path, capsys):
    feed = tmp_path / "crossed.jsonl"
    feed.write_text(
        json.dumps(
            {
                "type": "open",
                "product_id": "BTC-USD",
                "sequence": 101,
                "order_id": "99999999-9999-4999-8999-999999999999",
                "side": "sell",
                "price": "99.00",
                "remaining_size": "0.5",
                "time": "2026-07-10T12:00:00.000000001Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    code = main([str(SNAPSHOT), str(feed), "--tick-size", "0.01"])

    assert code == 2
    assert "generated a simulated trade" in capsys.readouterr().err


def test_adapter_is_single_use_and_id_map_is_opt_in():
    adapter = CoinbaseL3Adapter(_empty_snapshot())
    assert list(adapter.iter_events([])) == []
    with pytest.raises(CoinbaseL3Error, match="only normalize one"):
        list(adapter.iter_events([]))
    with pytest.raises(CoinbaseL3Error, match="retention"):
        adapter.to_dict(include_id_map=True)
    with pytest.raises(CoinbaseL3Error, match="trade retention"):
        adapter.to_dict(include_trades=True)

    venue_id = "11111111-1111-4111-8111-111111111111"
    assert coinbase_order_id("BTC-USD", venue_id) == coinbase_order_id("btc-usd", venue_id)
