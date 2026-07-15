# Matching-Engine Conformance

For complete maintained adapters, start with the native Rust
[`orderbook-rs` integration](../integrations/orderbook_rs/README.md), then see
the narrower
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
tracebook-conformance sample /tmp/tracebook-conformance-v2

tracebook-conformance suite \
  /tmp/tracebook-conformance-v2 \
  --output /tmp/conformance-suite.json \
  --candidate python examples/conformance_adapter.py
```

Use `--candidate-cmd './adapter --flag value'` when Tracebook options follow the
candidate command. The legacy `--candidate ./adapter --flag value` form remains
available, but it must be last because every remaining argument belongs to the
candidate process. A conformant suite exits `0`; a semantic divergence exits
`1`; an invalid trace, manifest, command, or protocol exits `2`.

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
made. Every fresh candidate process must report the same engine metadata as the
initial failing run; a binary or adapter identity change aborts minimization.

Replay a corpus reproducer and verify its exact stored divergence:

```bash
tracebook-conformance reproduce \
  .tracebook/corpus/failure-bc8b19d3e0e3441a98db/reduced.jsonl \
  --output reproduction.json \
  --junit-output reproduction.xml \
  --candidate-cmd './my-engine-adapter --tracebook-stdio'
```

When `failure.json` is beside the trace, `reproduce` loads its config and
requires the same failure class, event, structural path, reference value, and
candidate value. It exits `0` only for that exact reproduction and exits `1`
when the trace conforms or fails differently. Without metadata it accepts any
semantic divergence, which is useful for older minimized traces.

## Differential Campaigns

Campaigns generate stateful traces, compare them one at a time, and stop at the
first divergence. The next generated event depends only on the reference
engine's active orders. Candidate output never feeds back into generation.

```bash
tracebook-conformance campaign \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 1000 \
  --events-per-trace 200 \
  --max-minimize-runs 100 \
  --candidate-cmd ./my-engine-adapter \
  --corpus-dir .tracebook/corpus \
  --stop-after-first \
  --junit-output .tracebook/campaign.xml
```

Profiles are named and versioned because their generated semantic surface is a
public reproducibility boundary:

| Profile | Generated surface |
| --- | --- |
| `fifo-limit-v1` | FIFO limit orders, partial fills, cancel, reduce, replace, clear, duplicate active IDs, inactive lifecycle requests, multiple symbols, and a structured queue-priority probe |
| `fifo-full-v1` | Everything in `fifo-limit-v1`, plus market, IOC, and FOK instructions |
| `fifo-partial-fill-v1` | The `fifo-limit-v1` lifecycle surface with a four-event probe that verifies a partially filled maker keeps priority over a later maker |

### `fifo-limit-v1` Capability Profile

`fifo-limit-v1` is the smallest portable external-engine contract maintained by
Tracebook. A candidate claiming this profile is expected to agree on:

| Capability | Observable contract |
| --- | --- |
| Limit orders | BUY and SELL limit submission, crossing, resting, and partial fills |
| FIFO priority | Price priority followed by insertion order at one price level |
| Cancel | Active cancellation applies; inactive cancellation rejects with `ORDER_NOT_ACTIVE` |
| Reduce | Quantity decreases in place and retains queue priority; a full reduction removes the order |
| Replace | Validated cancel-and-new under the same source ID; replacement loses queue priority |
| Clear | Removes one symbol's resting state without mutating other symbols |
| Active IDs | A duplicate source ID rejects with `DUPLICATE_ORDER_ID` while active |
| Multiple symbols | Source-ID domains and queue state remain independent per symbol |
| Numeric normalization | Tick size `0.01`, half-even price snapping, and 12 quantity decimal places by default |
| Output | Applied/rejected outcome, ordered source-ID trades, resting count, and complete priority-state hash after every event |

The profile deliberately excludes market, IOC, FOK, pro-rata, and self-trade
prevention semantics. It also excludes in-place quantity increases: `reduce`
is a decrement of remaining quantity, while `replace` is cancel-and-new. Use
`fifo-full-v1` for the three additional order instructions. STP and pro-rata
remain covered by the fixed standard suite, not by either generated profile.

`fifo-partial-fill-v1` keeps the same portable capability boundary but changes
the structured probe. Two makers rest at one price, one taker partially fills
the first maker, and a second taker verifies that the first maker's remainder
still executes before the later maker. The profile exists separately so adding
this real-world regression does not change `fifo-limit-v1` traces, hashes, or
failure IDs.

That upsize exclusion defines the versioned profile surface; it is no longer a
limitation of the maintained Rust adapter. In
[`orderbook-rs` #203](https://github.com/joaquinbejar/OrderBook-rs/issues/203),
the maintainer confirmed that `orderbook-rs 0.12.0` with `pricelevel 0.9.1`
materializes snapshots in queue-consumption order, including after an in-place
quantity increase receives a fresh insertion sequence. Tracebook locks that
guarantee with a native snapshot-and-next-trade regression. Any future profile
that adds upsize still needs a new versioned capability contract and portable
regression rather than silently changing `fifo-limit-v1` traces and hashes.

Generator version 2 specifies SplitMix64 independently of Python's `random`
module and adds a five-event FIFO priority probe in an isolated symbol. Its end
position is `min(events_per_trace, 133 + trace_seed % 43)`, so campaign seed 42
puts the first trace's probe at events 169-173. The probe rests two makers at one
price, reduces and replaces the first maker, then crosses the level. Correct
cancel-and-new replacement semantics fill the second maker first.

For `fifo-partial-fill-v1`, generator version 2 places a four-event probe at the
same deterministic end position. Campaign seed 42 therefore ends the first
trace's probe at event 173. The first three events leave aggregate depth
unchanged between conforming and tail-requeue implementations; the fourth
event exposes the wrong maker ID and reduces to those four causal events.

The same profile, generator version, unsigned 64-bit seed, requested trace
count, and events-per-trace value produce the same campaign ID and trace hashes
across supported Python versions. Changing candidate metadata or behavior does
not change that identity.

The output path must not already exist. Tracebook reserves that exact directory
before starting candidate work and keeps its inode open. It installs every
artifact with descriptor-relative, exclusive creates, writes `campaign.json`,
then removes `.tracebook-campaign-reservation` as the final commit signal. It
never replaces or mutates a re-resolved destination path. A directory that
still contains the reservation marker is incomplete and must be explicitly
removed before retrying; this conservative rule also applies after handled
candidate or write failures. Bundle publication fails closed before creating
the output on platforms where Python lacks descriptor-relative directory
operations; `run_campaign` generation and comparison remain available. A
divergent `--output-dir` run also writes its compatibility layout:

| Path | Contents |
| --- | --- |
| `failure/original.jsonl` | Trace prefix ending at the first drift |
| `failure/original-report.json` | Exact first-divergence semantic report |
| `failure/minimized.jsonl` | Reduced trace that preserves the failure category |
| `failure/minimization.json` | Reduction budget, run count, hashes, and minimality claim |

`--corpus-dir` stores the run under a deterministic failure ID. The bundle adds
top-level `original.jsonl`, `reduced.jsonl`, and `failure.json`; the latter pins
the campaign seed and hash, profile config, candidate identity, original event,
failure class, reduced trace hash, and exact expected reduced divergence. A
failure ID binds the campaign, original and reduced trace hashes, failure class,
and exact reduced divergence. A conformant campaign is stored under a
deterministic `campaign-*` ID instead.

Campaign JSON includes `semantic_coverage`. Coverage is calculated by replaying
only candidate-compared events through the reference engine. It reports
declared, covered, and uncovered capabilities; operation counts; attempted and
applied instruction counts; applied and rejected outcomes; rejection reasons; symbols; trades;
partial-fill evidence; and structured queue-priority probes. It is workload
semantic coverage, not source-code coverage and not evidence that a candidate
supports semantics it declares out of scope.

The command exits `0` when all requested traces conform, `1` on semantic drift,
and `2` for invalid configuration, adapter, protocol, or filesystem errors.
Promote `reduced.jsonl` into a regression suite after reviewing the
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

For FIFO `CANCEL_RESTING`, self-trade prevention is encounter-based. Matching
walks makers in priority order, cancels a same-owner maker when the sweep
reaches it, and continues. Once the incoming quantity is filled, deeper makers
are not inspected or canceled. An engine that pre-cancels every same-owner
maker at a touched price level implements a different, valid policy and should
report that boundary rather than claiming this case.

Only limit orders may appear in resting state. Prices and quantities are decimal
strings, never binary floating-point JSON numbers. Prices use the canonical
tick-grid value, but the reference first divides the parsed IEEE-754 binary64
price by the binary64 tick size and then rounds ties to even. This is not exact
decimal division: with tick size `0.01`, `1.015` snaps to `1.01`, not `1.02`.
Quantities are rounded half-even to the configured
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

`tracebook-conformance-v2` is the default synthetic, redistributable suite.
Every event file is SHA-256 locked, and `suite_hash` binds the manifest's case
configs, tags, file hashes, description, and suite ID. Version 2 adds a
four-event FIFO case proving that `CANCEL_RESTING` does not remove a deeper
same-owner maker that the sweep never reaches. The suite currently covers:

- FIFO partial fills, reduction, cancel-and-new replacement, queue-priority
  loss, crossed input, and rejected lifecycle events;
- market, IOC, fillable FOK, and unfillable FOK instructions;
- encounter-based `CANCEL_RESTING`, including a deeper same-owner maker, and
  `CANCEL_INCOMING` self-trade prevention;
- pro-rata allocation and subsequent lifecycle operations;
- independent source-ID domains across multiple symbols;
- off-grid tick snapping and a price that snaps to a non-positive tick;
- a deeper two-sided book with cancellation, reduction, replacement, and a
  multi-level sweep.

Suite reports preserve every case report rather than stopping after the first
failed case, making them suitable for CI artifacts.

Bundled suites are immutable once published. `sample` copies v2 by default;
use `tracebook-conformance sample ./suite --suite-version v1` to reproduce the
original eight-case suite and its historical hash.

## Artifact Contracts

Single-run reports use `artifact_type = "tracebook.conformance.report"` and
include protocol/schema versions, trace SHA-256, exact config, engine metadata,
the number of compared events, final state hash, and the first divergence.

Minimization reports use
`artifact_type = "tracebook.conformance.minimization"` and include original and
minimized counts, run count, reduction percentage, minimality/budget status,
minimized trace hash, target failure category, and the final conformance report.

Suite reports use `artifact_type = "tracebook.conformance.suite_report"` and
retain the suite hash plus each case's tags, fixture hash, and full report.

Campaign reports use `artifact_type = "tracebook.conformance.campaign"` and
include generator/profile versions, campaign identity, requested and completed
work, candidate metadata, every trace seed and hash, and relative failure-bundle
paths. Campaign artifacts also start at schema version `1`.

Corpus metadata uses `artifact_type = "tracebook.conformance.failure"`.
Reproduction reports use `artifact_type =
"tracebook.conformance.reproduction"` and preserve expected and observed
failure details plus the full conformance report. All JSON artifact schemas
start at version `1`.

`run`, `suite`, `minimize`, `campaign`, and `reproduce` accept
`--junit-output`. JUnit is a projection of the canonical JSON: divergences are
test failures, a successful minimization is a passing case, and an exact known
failure reproduction is a passing case. Campaign JUnit properties include the
semantic coverage ratio and counts. JSON remains the lossless contract.

## Boundaries

Conformance means agreement with Tracebook's documented semantics for the given
trace and configuration. It does not prove exchange certification, thread
safety, durability, risk controls, network behavior, or production latency. An
adapter is also responsible for faithfully translating its engine's internal
state into source IDs and canonical snapshots; the final snapshot/hash check
detects inconsistent adapter output but cannot prove an adapter is honest.
