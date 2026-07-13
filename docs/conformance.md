# Matching-Engine Conformance

For a complete maintained adapter, see the
[PythonMatchingEngine integration](../integrations/python_matching_engine/README.md).
To make conformance a pull-request gate, use the
[copy-paste GitHub Actions workflow](ci.md).

Candidate commands execute with the caller's operating-system permissions.
Protocol timeouts bound waiting and clean up stalled processes; they are not a
sandbox. Run untrusted engines inside an appropriately isolated container or VM.

Tracebook can run one normalized order-event trace through its inspectable
reference engine and an external candidate, compare every observable result,
stop at the first disagreement, and reduce that failure to a smaller trace.
The candidate may be written in any language that can read and write one JSON
object per line.

## Quick Start

Copy the bundled synthetic suite and run the example adapter:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v1

tracebook-conformance suite \
  /tmp/tracebook-conformance-v1 \
  --output /tmp/conformance-suite.json \
  --candidate python examples/conformance_adapter.py
```

`--candidate` and its arguments must be last because every remaining argument
belongs to the candidate process. A conformant suite exits `0`; a semantic
divergence exits `1`; an invalid trace, manifest, command, or protocol exits `2`.

Run one normalized CSV, JSON, JSONL, or NDJSON trace:

```bash
tracebook-conformance run events.jsonl \
  --algorithm fifo \
  --tick-size 0.01 \
  --self-trade-policy NONE \
  --output conformance-report.json \
  --candidate ./my-engine-adapter --config adapter.toml
```

Reduce a divergence while preserving event order:

```bash
tracebook-conformance minimize events.jsonl \
  --events-output minimal.jsonl \
  --output minimization-report.json \
  --max-runs 100 \
  --candidate ./my-engine-adapter
```

The minimizer truncates events after the first failure, then uses deterministic
delta debugging. It retains a candidate subsequence only when the same failure
category still occurs. The report sets `one_minimal` only after every individual
deletion in the final round no longer reproduces that category. If `--max-runs`
stops the search first, `budget_exhausted` is true and no minimality claim is
made.

## Differential Campaigns

Campaigns generate stateful traces, compare them one at a time, and stop at the
first divergence. The next generated event depends only on the reference
engine's active orders. Candidate output never feeds back into generation.

```bash
tracebook-conformance campaign \
  --output-dir /tmp/tracebook-campaign \
  --profile fifo-limit-v1 \
  --seed 20260713 \
  --traces 25 \
  --events-per-trace 100 \
  --max-minimize-runs 100 \
  --candidate ./my-engine-adapter
```

Profiles are named and versioned because their generated semantic surface is a
public reproducibility boundary:

| Profile | Generated surface |
| --- | --- |
| `fifo-limit-v1` | FIFO limit orders, partial fills, cancel, reduce, replace, clear, duplicate active IDs, inactive lifecycle requests, and two symbols |
| `fifo-full-v1` | Everything in `fifo-limit-v1`, plus market, IOC, and FOK instructions |

Generator version 1 specifies SplitMix64 independently of Python's `random`
module. The same profile, generator version, unsigned 64-bit seed, trace count,
and events-per-trace value produce the same campaign ID and trace hashes across
supported Python versions. Changing candidate metadata or behavior does not
change that identity.

The output directory is created atomically and must not already exist. Every
run writes `campaign.json`. A divergent run also writes:

| Path | Contents |
| --- | --- |
| `failure/original.jsonl` | Complete generated trace containing the first drift |
| `failure/original-report.json` | Exact first-divergence semantic report |
| `failure/minimized.jsonl` | Reduced trace that preserves the failure category |
| `failure/minimization.json` | Reduction budget, run count, hashes, and minimality claim |

The command exits `0` when all requested traces conform, `1` on semantic drift,
and `2` for invalid configuration, adapter, protocol, or filesystem errors.
Promote the minimized JSONL file into a regression suite after reviewing the
candidate and reference semantics.

## What Is Compared

After every source event, the candidate reports:

- whether the event was applied or rejected;
- a stable rejection reason code when rejected;
- ordered trades using source order IDs;
- the number of resting orders;
- a SHA-256 digest of every resting order and its queue position.

Tracebook compares those fields immediately. On a state-digest mismatch it asks
the candidate for a full snapshot and reports the first differing symbol, side,
queue position, order ID, price, remaining quantity, owner, or order type. A
final full snapshot is always requested, even after all per-event hashes agree.

Wall-clock timestamps, internal engine IDs, latency, and implementation-specific
metadata are excluded from semantic comparison. Performance is measured by the
separate benchmark tools.

## Canonical Semantics

The protocol uses normalized `MarketEvent` objects documented in
[`event-replay.md`](event-replay.md). Source `order_id` values remain stable
through replacement even when an engine allocates a new internal ID.

State ordering is part of the contract:

1. Books are sorted lexicographically by symbol.
2. Bids are ordered from highest price to lowest.
3. Asks are ordered from lowest price to highest.
4. Orders at one price are listed in matching-priority order.
5. A symbol first encountered by `new` or `clear` remains present after its book
   becomes empty. This includes a structurally valid `new` that reaches the book
   but is rejected by engine validation. An invalid cancel/reduce/replace for an
   unknown symbol does not create a book.

Only limit orders may appear in resting state. Prices and quantities are decimal
strings, never binary floating-point JSON numbers. Prices use the canonical
tick-grid value. Quantities are rounded half-even to the configured
`quantity_decimal_places` (default `12`) before trailing zeroes are removed.
Zero is always `"0"`; exponent notation is forbidden.

Stable rejection codes in protocol version 1 are:

| Code | Meaning |
| --- | --- |
| `ORDER_NOT_ACTIVE` | Cancel, reduce, or replace could not find the source order |
| `DUPLICATE_ORDER_ID` | A new source ID is already active for that symbol |
| `INVALID_REPLACEMENT` | Replacement parameters failed validation |
| `INVALID_CANCEL` | A cancel request was otherwise invalid |
| `INVALID_ORDER` | Submission or lifecycle input failed another engine rule |

The optional human-readable rejection `message` is retained in reports but is
not compared. This avoids requiring a Rust or Java engine to reproduce Python
exception text.

## Protocol Version 1

The adapter is a long-running child process. Stdin and stdout carry UTF-8 NDJSON.
Stdout is reserved for protocol frames; diagnostics belong on stderr. Tracebook
starts the command as an argument list with no shell interpolation.

Host to candidate, once:

```json
{"type":"hello","protocol":"tracebook.conformance","protocol_version":1,"config":{"matching_algorithm":"fifo","tick_size":"0.01","self_trade_policy":"NONE","quantity_decimal_places":12}}
```

Candidate to host:

```json
{"type":"ready","protocol":"tracebook.conformance","protocol_version":1,"engine":{"name":"my-engine","version":"1.4.2","language":"Rust"}}
```

Host to candidate for each event:

```json
{"type":"event","index":1,"event":{"op":"new","symbol":"TEST","order_id":10,"side":"BUY","order_type":"LIMIT","price":100.0,"quantity":2.0,"owner":7,"timestamp_ns":1}}
```

Candidate to host after applying it:

```json
{"type":"observation","index":1,"outcome":{"status":"applied","reason":null,"message":null},"trades":[],"state_hash":"18de97d9cb98a7ca4dbc2215a748f7fded7607f05782a33338fcc6f6f0b91c62","resting_order_count":1}
```

A trade object has exactly the comparable source fields:

```json
{"symbol":"TEST","buy_order_id":10,"sell_order_id":11,"price":"100","quantity":"0.5"}
```

When Tracebook needs the full state, it sends:

```json
{"type":"snapshot","index":1}
```

The candidate responds:

```json
{
  "type": "snapshot",
  "index": 1,
  "state": {
    "books": [
      {
        "symbol": "TEST",
        "bids": [
          {"order_id":10,"price":"100","remaining_quantity":"2","owner":7,"order_type":"LIMIT"}
        ],
        "asks": []
      }
    ]
  }
}
```

After the final event the host sends `{"type":"finish","event_count":N}` and
the candidate responds with `{"type":"complete","event_count":N}` before
exiting.

The state hash is SHA-256 over the UTF-8 bytes of the `state` object serialized
as compact JSON with object keys sorted lexicographically, arrays retained in
their specified order, and no ASCII escaping requirement for ordinary symbols.
The Python equivalent is:

```python
json.dumps(
    state,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=False,
    allow_nan=False,
).encode("utf-8")
```

The compact hash keeps protocol traffic constant per event, but producing a
canonical hash may still require visiting the current resting book. A simple
adapter can therefore spend O(book size) per event on conformance state. These
runs are correctness checks, not latency benchmarks; large-engine adapters may
maintain their own equivalent incremental state representation.

## Python Adapter SDK

Python adapters can implement the small `EngineAdapter` protocol and let
`serve_stdio` handle framing and validation:

```python
from tracebook.conformance import EngineMetadata, serve_stdio


class MyAdapter:
    def __init__(self, config):
        self.metadata = EngineMetadata("my-engine", "1.0", "Python")
        self.config = config

    def apply(self, event, index):
        # Apply event and return Observation(...).
        ...

    def snapshot(self):
        # Return BookState(...), preserving matching priority.
        ...

    def close(self):
        ...


raise SystemExit(serve_stdio(MyAdapter))
```

[`examples/conformance_adapter.py`](../examples/conformance_adapter.py) is a
runnable reference. Non-Python adapters implement the same frames directly.

## Bundled Suite

`tracebook-conformance-v1` is synthetic and redistributable. Every event file is
SHA-256 locked, and `suite_hash` binds the manifest's case configs, tags, file
hashes, description, and suite ID. The suite currently covers:

- FIFO partial fills, reduction, cancel-and-new replacement, queue-priority
  loss, crossed input, and rejected lifecycle events;
- market, IOC, fillable FOK, and unfillable FOK instructions;
- `CANCEL_RESTING` and `CANCEL_INCOMING` self-trade prevention;
- pro-rata allocation and subsequent lifecycle operations;
- independent source-ID domains across multiple symbols;
- off-grid tick snapping and a price that snaps to a non-positive tick;
- a deeper two-sided book with cancellation, reduction, replacement, and a
  multi-level sweep.

Suite reports preserve every case report rather than stopping after the first
failed case, making them suitable for CI artifacts.

## Artifact Contracts

Single-run reports use `artifact_type = "tracebook.conformance.report"` and
include protocol/schema versions, trace SHA-256, exact config, engine metadata,
the number of compared events, final state hash, and the first divergence.

Minimization reports use
`artifact_type = "tracebook.conformance.minimization"` and include original and
minimized counts, run count, reduction percentage, minimality/budget status,
minimized trace hash, target failure category, and the final conformance report.

Suite reports use `artifact_type = "tracebook.conformance.suite_report"` and
retain the suite hash plus each case's tags, fixture hash, and full report. All
three artifact schemas start at version `1`.

Campaign reports use `artifact_type = "tracebook.conformance.campaign"` and
include generator/profile versions, campaign identity, requested and completed
work, candidate metadata, every trace seed and hash, and relative failure-bundle
paths. Campaign artifacts also start at schema version `1`.

## Boundaries

Conformance means agreement with Tracebook's documented semantics for the given
trace and configuration. It does not prove exchange certification, thread
safety, durability, risk controls, network behavior, or production latency. An
adapter is also responsible for faithfully translating its engine's internal
state into source IDs and canonical snapshots; the final snapshot/hash check
detects inconsistent adapter output but cannot prove an adapter is honest.
