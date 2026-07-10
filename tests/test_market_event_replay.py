import json

import pytest

from tracebook.events import (
    MarketEvent,
    MarketReplayError,
    load_market_events,
    replay_market_event_file,
    replay_market_events,
)
from tracebook.events.cli import main
from tracebook import SelfTradePolicy


def test_jsonl_replay_supports_multiple_symbols_and_lifecycle_events(tmp_path):
    source = tmp_path / "events.jsonl"
    rows = [
        {
            "op": "new",
            "symbol": "BTCUSD",
            "order_id": 1,
            "side": "BUY",
            "price": 100,
            "quantity": 2,
        },
        {
            "op": "new",
            "symbol": "ETHUSD",
            "order_id": 1,
            "side": "SELL",
            "price": 200,
            "quantity": 3,
        },
        {"op": "replace", "symbol": "BTCUSD", "order_id": 1, "price": 101},
        {"op": "cancel", "symbol": "ETHUSD", "order_id": 1},
        {"op": "clear", "symbol": "BTCUSD"},
    ]
    source.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    result = replay_market_event_file(source)
    summary = result.to_dict()

    assert summary["input_events"] == 5
    assert summary["applied_events"] == 5
    assert summary["rejected_events"] == 0
    assert summary["books"]["BTCUSD"]["bids"] == []
    assert summary["books"]["ETHUSD"]["asks"] == []


def test_csv_and_json_event_loaders_share_the_normalized_schema(tmp_path):
    csv_path = tmp_path / "events.csv"
    csv_path.write_text(
        "op,symbol,order_id,side,order_type,price,quantity,owner,timestamp_ns\n"
        "new,BTCUSD,10,BUY,LIMIT,100.0,1.5,4,123\n",
        encoding="utf-8",
    )
    json_path = tmp_path / "events.json"
    json_path.write_text(
        json.dumps(
            {
                "events": [
                    {
                        "op": "new",
                        "symbol": "BTCUSD",
                        "order_id": 10,
                        "side": "BUY",
                        "order_type": "LIMIT",
                        "price": 100.0,
                        "quantity": 1.5,
                        "owner": 4,
                        "timestamp_ns": 123,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert load_market_events(csv_path) == load_market_events(json_path)


def test_lenient_replay_collects_rejections_and_continues():
    events = [
        MarketEvent.from_mapping({"op": "cancel", "symbol": "BTCUSD", "order_id": 99}),
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 1,
                "side": "BUY",
                "price": 100,
                "quantity": 1,
            }
        ),
    ]

    result = replay_market_events(events, strict=False)

    assert result.input_events == 2
    assert result.applied_events == 1
    assert len(result.rejections) == 1
    assert result.manager.get_order_book("BTCUSD").get_best_bid() == pytest.approx(100.0)


def test_replace_keeps_source_id_addressable_for_later_cancel():
    events = [
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 50,
                "side": "BUY",
                "price": 100,
                "quantity": 1,
            }
        ),
        MarketEvent.from_mapping(
            {"op": "replace", "symbol": "BTCUSD", "order_id": 50, "price": 101}
        ),
        MarketEvent.from_mapping({"op": "cancel", "symbol": "BTCUSD", "order_id": 50}),
    ]

    result = replay_market_events(events)

    assert result.applied_events == 3
    assert result.replacements == 1
    assert result.cancellations == 1
    assert result.resolve_order_id("BTCUSD", 50) is None
    assert result.manager.get_order_book("BTCUSD").get_active_order_ids() == []
    with pytest.raises(MarketReplayError, match="source_order_id"):
        result.resolve_order_id("BTCUSD", 0)


def test_replace_preserves_source_timestamp_on_the_new_engine_order():
    events = [
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 12,
                "side": "BUY",
                "price": 100,
                "quantity": 1,
                "timestamp_ns": 10,
            }
        ),
        MarketEvent.from_mapping(
            {
                "op": "replace",
                "symbol": "BTCUSD",
                "order_id": 12,
                "price": 101,
                "timestamp_ns": 20,
            }
        ),
    ]

    result = replay_market_events(events)
    engine_id = result.resolve_order_id("BTCUSD", 12)

    assert engine_id is not None
    assert result.manager.get_order_book("BTCUSD").get_order(engine_id).timestamp == 20


def test_trade_records_use_source_ids_and_market_price_can_be_omitted():
    events = [
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 700,
                "side": "SELL",
                "price": 100,
                "quantity": 1,
            }
        ),
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 900,
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 1,
            }
        ),
    ]

    result = replay_market_events(events)
    payload = result.to_dict(include_trades=True)

    assert len(result.trades) == 1
    assert result.trades[0].buy_order_id == 900
    assert result.trades[0].sell_order_id == 700
    assert payload["trades_included"] is True
    assert payload["trades"][0]["event_index"] == 2


def test_replay_exposes_self_trade_policy_and_validates_global_configuration():
    events = [
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 1,
                "side": "SELL",
                "price": 100,
                "quantity": 1,
                "owner": 5,
            }
        ),
        MarketEvent.from_mapping(
            {
                "op": "new",
                "symbol": "BTCUSD",
                "order_id": 2,
                "side": "BUY",
                "order_type": "MARKET",
                "quantity": 1,
                "owner": 5,
            }
        ),
    ]

    result = replay_market_events(events, self_trade_policy=SelfTradePolicy.CANCEL_INCOMING)

    assert result.trades == []
    assert result.to_dict()["replay_config"]["self_trade_policy"] == "CANCEL_INCOMING"
    with pytest.raises(MarketReplayError, match="tick_size"):
        replay_market_events([], tick_size=float("nan"))
    with pytest.raises(MarketReplayError, match="depth_levels"):
        result.to_dict(depth_levels=1.5)


def test_lenient_unknown_symbol_cancel_does_not_create_a_ghost_book():
    event = MarketEvent.from_mapping({"op": "cancel", "symbol": "BTCUSD", "order_id": 99})

    result = replay_market_events([event], strict=False)

    assert result.manager.get_all_symbols() == []


def test_strict_replay_fails_with_event_context():
    event = MarketEvent.from_mapping({"op": "cancel", "symbol": "BTCUSD", "order_id": 99})

    with pytest.raises(MarketReplayError, match=r"Event 1 \(cancel BTCUSD\)"):
        replay_market_events([event])


def test_direct_market_event_construction_is_normalized_and_validated():
    event = MarketEvent(
        op="submit",
        symbol=" BTCUSD ",
        order_id="7",
        side="buy",
        price="100",
        quantity="1.5",
    )

    assert event.op == "new"
    assert event.symbol == "BTCUSD"
    assert event.order_id == 7
    assert event.side.name == "BUY"
    assert event.price == pytest.approx(100.0)

    with pytest.raises(MarketReplayError, match="op must be"):
        MarketEvent(op="upsert", symbol="BTCUSD")


def test_lenient_replay_rejects_non_event_items_with_index_context():
    result = replay_market_events([object()], strict=False)

    assert result.input_events == 1
    assert result.applied_events == 0
    assert result.rejections[0].event_index == 1
    assert "Expected MarketEvent" in result.rejections[0].reason


@pytest.mark.parametrize(
    "mapping",
    [
        {"op": "unknown", "symbol": "BTCUSD"},
        {
            "op": "new",
            "symbol": "BTCUSD",
            "order_id": 1.5,
            "side": "BUY",
            "price": 1,
            "quantity": 1,
        },
        {
            "op": "new",
            "symbol": "BTCUSD",
            "order_id": 1,
            "side": "SIDEWAYS",
            "price": 1,
            "quantity": 1,
        },
        {
            "op": "new",
            "symbol": "BTCUSD",
            "order_id": 1,
            "side": "BUY",
            "price": float("inf"),
            "quantity": 1,
        },
        {"op": "replace", "symbol": "BTCUSD", "order_id": 1},
    ],
)
def test_market_event_parser_rejects_ambiguous_input(mapping):
    with pytest.raises((MarketReplayError, ValueError)):
        MarketEvent.from_mapping(mapping)


def test_replay_cli_writes_machine_readable_summary(tmp_path, capsys):
    source = tmp_path / "events.json"
    output = tmp_path / "nested" / "summary.json"
    source.write_text(
        json.dumps(
            [
                {
                    "op": "new",
                    "symbol": "BTCUSD",
                    "order_id": 1,
                    "side": "BUY",
                    "price": 100,
                    "quantity": 1,
                }
            ]
        ),
        encoding="utf-8",
    )

    assert (
        main(
            [
                str(source),
                "--self-trade-policy",
                "NONE",
                "--include-trades",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["schema_version"] == 1
    assert payload["trades_included"] is True
    assert payload["books"]["BTCUSD"]["bids"] == [[100.0, 1.0, 1]]
    assert "Replay summary written" in capsys.readouterr().out
