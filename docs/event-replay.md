# Normalized Event Replay

`tracebook-replay` processes order-level events through the same validated books
used by the Python API. It accepts `.json`, `.jsonl`/`.ndjson`, and `.csv` files,
preserves source-file order, and supports independent books per symbol.

This is an order-event (L3-style) interface. Aggregated L2 snapshots do not
contain enough information to reconstruct queue priority and should be converted
only when the source provides explicit, documented assumptions.

## Install And Run

```bash
python -m pip install tracebook-sim
tracebook-replay examples/data/sample_events.jsonl --output replay-summary.json
```

Use `--algorithm pro_rata`, `--tick-size 0.0001`,
`--self-trade-policy CANCEL_INCOMING`, or `--lenient` as needed. Strict mode is
the default and exits at the first rejected event. Lenient mode continues and
writes each rejection into the summary. Add `--include-trades` to include the
source-id annotated trade tape in the JSON output.

## Event Schema

Every event requires `op` and `symbol`.

| Operation | Required fields | Optional fields |
| --- | --- | --- |
| `new` | `order_id`, `side`, `quantity`; `price` except for market orders | `order_type`, `price` for market orders, `owner`, `timestamp_ns` |
| `cancel` | `order_id` | `timestamp_ns` |
| `replace` | `order_id`, and `price` and/or `quantity` | `timestamp_ns` |
| `clear` | none | `timestamp_ns` |

`side` accepts `BUY`, `SELL`, `1`, or `-1`. `order_type` accepts `LIMIT`,
`MARKET`, `IOC`, or `FOK`; it defaults to `LIMIT`. A market-order price may be
omitted and is normalized to `0`.

`order_id` is the stable source identifier. The replay engine allocates its own
ids, maps lifecycle events back to the source id, and keeps that mapping across
cancel-and-new replacement. A later `cancel` therefore continues to address the
same source order after any number of replacements. Included trade records carry
both source ids and engine ids; `result.resolve_order_id(symbol, source_id)`
returns the current active engine id when one exists.

`timestamp_ns` is retained on new and replacement engine orders. It is metadata;
event application and queue priority follow source-file order.

Example JSONL:

```json
{"op":"new","symbol":"BTCUSD","order_id":1,"side":"BUY","price":50000,"quantity":1.0,"owner":10,"timestamp_ns":1000}
{"op":"new","symbol":"BTCUSD","order_id":2,"side":"SELL","price":50000,"quantity":0.5,"owner":20,"timestamp_ns":2000}
{"op":"cancel","symbol":"BTCUSD","order_id":1,"timestamp_ns":3000}
```

## Python API

```python
from tracebook import SelfTradePolicy, load_market_events, replay_market_events

events = load_market_events("events.jsonl")
result = replay_market_events(
    events,
    matching_algorithm="fifo",
    self_trade_policy=SelfTradePolicy.CANCEL_INCOMING,
    strict=True,
)

book = result.manager.get_order_book("BTCUSD")
print(result.to_dict(depth_levels=10, include_trades=True))
print(book.get_best_bid(), book.get_best_ask())
```

Exchange-specific adapters should emit `MarketEvent` objects and keep venue
sequence validation, checksum handling, and feed-specific semantics in the
adapter rather than weakening this normalized contract.
