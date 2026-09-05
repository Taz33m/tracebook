# L3 Book-Replay Conformance

`l3-book-replay-v1` is a separate conformance surface for deterministic L3
order-book mirrors. It exists for libraries that ingest exogenous market data
and answer depth or simulated-fill questions but do not cross submitted orders
against one another.

It is deliberately outside `tracebook.conformance` protocol v1. A successful
book-replay report says nothing about FIFO venue matching, IOC/FOK behavior,
self-trade prevention, trade identity, or candidate order lifecycle.

## Profile Semantics

Every JSONL record has an `op` and `symbol`. Numbers are normalized into finite,
positive, non-exponent decimal strings.

### Add

```json
{"op":"add","symbol":"TEST","order_id":1,"side":"BUY","price":"100","quantity":"2"}
```

The order is appended at its price level. An active ID can occur only once per
symbol; a duplicate is rejected as `DUPLICATE_ORDER_ID`.

### Update

```json
{"op":"update","symbol":"TEST","order_id":1,"side":"BUY","price":"100","quantity":"4"}
```

An update at the existing side and price replaces quantity in place, including
an increase, and retains queue priority. A side or price change removes the old
entry and appends it at the destination level. An unknown ID is rejected as
`ORDER_NOT_ACTIVE`.

### Delete

```json
{"op":"delete","symbol":"TEST","order_id":1}
```

Deletion is keyed by active source order ID. An unknown ID is rejected as
`ORDER_NOT_ACTIVE`.

### Probe

```json
{"op":"probe","symbol":"TEST","side":"SELL","price":"99","quantity":"8"}
```

`side` is the hypothetical aggressive side. A buy walks asks from low to high;
a sell walks bids from high to low. The walk stops outside the limit price,
visits orders FIFO within a level, and returns ordered `{price, quantity}` fill
segments. A probe never mutates state.

This is execution *simulation against mirrored liquidity*. It is not an order
submission and creates no trade IDs, positions, accounts, or order events.

## Observable State

After every event, an adapter returns:

- normalized applied/rejected outcome;
- ordered simulated fills, if any;
- mirrored order count; and
- SHA-256 of the canonical full snapshot.

Snapshots contain symbols in lexical order, bids in descending price order,
asks in ascending price order, and orders FIFO within each level. The runner
requests the full snapshot when a state hash differs so the report localizes
the first order or queue-position mismatch.

Reports use `artifact_type = "tracebook.book-replay.report"`. The wire handshake
uses `protocol = "tracebook.book-replay"`, version `1`. Neither identity is an
alias for `tracebook.conformance`.

## Run It

Book replay is unreleased and is not present in the published 0.6.0 wheel.
Install this source checkout (`python -m pip install -e .`) or its built wheel,
and record the exact source revision with any evidence.

Copy the bundled trace:

```bash
python -m tracebook.book_replay sample /tmp/tracebook-book-replay
```

Run an external adapter:

```bash
python -m tracebook.book_replay run \
  /tmp/tracebook-book-replay/l3-book-replay-v1.jsonl \
  --output /tmp/book-replay-report.json \
  --candidate-cmd './book-adapter --tracebook-book-replay-stdio'
```

Exit `0` means semantic agreement, `1` means the first semantic divergence was
recorded, and `2` means input, process, or protocol failure.

File outputs must be new paths. Existing files, directories, and symlinks are
refused with exit `2`; input/output aliases and outputs nested inside one
another are also refused. All requested destinations are reserved before the
candidate starts, including a campaign's optional reduced trace even when the
campaign ultimately passes. A passing campaign leaves no reduced-trace file.

Reports and traces are staged completely before atomic publication without
overwriting another writer's file. When producing a pair, the reduced trace is
published first and the report last; a caught publication failure rolls back
files created by that attempt. These are individual file commits, not a
filesystem-wide transaction: abrupt process termination can leave a complete
trace without its report. Only use the pair after the command completes.
Sibling `.NAME.tracebook-in-progress` reservations and `.tracebook-stage-*`
private staging directories are removed on normal completion, with cleanup attempted on
handled errors. Filesystem cleanup failures or competing replacements can
leave sidecars. After a forced stop or cleanup failure, retain the artifacts
and use fresh output names for a new invocation; do not remove a live process's
reservations.

Staged payloads live in owner-only, read-only directories and publication uses
the open directory descriptor, so replacing a staging directory's parent entry
cannot substitute another payload. Published files are owner-readable. This
protects against ordinary concurrent file writers, not privileged processes or
same-user code deliberately changing permissions or writing through an already
open descriptor. Use a trusted output directory; these filesystem guards are
not a sandbox for hostile code running as your user.

## Generated Campaigns And Reduction

The v1 generator uses a specified SplitMix64 stream rather than Python's
runtime PRNG. Each trace draws its prefix from an 18-event semantic scaffold;
at 18 events or longer it reaches all 11 declared capabilities, then continues
with state-aware generated adds, updates, deletes, and probes. Candidate
behavior never feeds back into event generation.

Run a deterministic campaign:

```bash
python -m tracebook.book_replay campaign \
  --output /tmp/book-replay-campaign.json \
  --reduced-events-output /tmp/book-replay-reduced.jsonl \
  --seed 20260831 \
  --traces 25 \
  --events-per-trace 100 \
  --candidate-cmd './book-adapter --tracebook-book-replay-stdio'
```

The runner stops at the first divergence and automatically delta-debugs its
compared prefix. Reduction preserves the first-divergence category, rejects a
new protocol failure while reducing a semantic failure, records its candidate
run budget, and reports whether the result is one-minimal. Campaign artifacts
include the generator version, campaign ID, per-trace seeds and hashes,
capability coverage, original evidence, minimized evidence, and a stable
failure ID.

An already captured divergent trace can be minimized directly:

```bash
python -m tracebook.book_replay minimize trace.jsonl \
  --events-output /tmp/reduced.jsonl \
  --output /tmp/minimization.json \
  --candidate-cmd './book-adapter --tracebook-book-replay-stdio'
```

`campaign` artifacts use `artifact_type = "tracebook.book-replay.campaign"`;
standalone reductions use
`artifact_type = "tracebook.book-replay.minimization"`.

Python adapters can use the server helper:

```python
from tracebook.book_replay import serve_book_replay_stdio

raise SystemExit(serve_book_replay_stdio(MyBookAdapter))
```

The complete reference-backed example is
[`examples/book_replay_adapter.py`](../examples/book_replay_adapter.py).

## NautilusTrader Boundary Proof

The maintained optional adapter targets NautilusTrader's native Rust-backed
`L3_MBO` book and direct `simulate_fills(BookOrder)` Python binding. It passes
all 17 bundled events and a pinned 2,500-event generated campaign at the
`2.0.0rc3` release candidate. See the
[`integrations/nautilus_trader`](../integrations/nautilus_trader) proof,
provenance, exact command, and LGPL distribution warning.

NautilusTrader's broader backtest matching engine remains out of scope here.
That engine fills candidate orders against book state supplied by market data;
it does not turn this delta profile into a submitted-order CLOB.
