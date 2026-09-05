# intrepidkarthi/orderbook CLOB Integration

This adapter checks a genuine submitted-order versus submitted-order central
limit order book. Unlike Tracebook's NautilusTrader book-replay integration,
both sides of every trade are orders sent through the candidate engine.

The external input is pinned exactly:

- Go module: `github.com/intrepidkarthi/orderbook v0.26.0`;
- tag: `v0.26.0`;
- source commit: `51d480cdb68b9989febb0b075d291cf891f425b3`;
- repository: <https://github.com/intrepidkarthi/orderbook>; and
- upstream license: MIT.

`go.mod` and `go.sum` lock the module graph. No upstream source or compiled
binary is vendored in Tracebook.

## What The Adapter Exercises

There is one native `matching.Engine` per symbol. New limit, market, IOC, and
FOK orders call `Engine.Process`; cancel, reduce, and replace call the native
methods of the same names. Trades come directly from the native
`matching.MatchResult`, and snapshots traverse native `RestingOrders`.
Tracebook source IDs are mapped to engine-generated IDs without changing the
engine's matching decisions.

The protocol configuration maps FIFO and pro-rata allocation plus `NONE`,
`CANCEL_RESTING`, and `CANCEL_INCOMING` self-trade policies to their native
settings. Prices are converted exactly to integer ticks. Quantities are first
normalized at the protocol precision and then represented as integer native
lots. The adapter accepts at most six non-zero quantity decimal places because
the upstream pro-rata implementation multiplies two `int64` lot values; values
that cannot be represented exactly at that boundary are explicitly rejected.

The local Go tests include a submitted-order crossing test: two same-price
bids are reduced and then consumed by an incoming sell in native FIFO order.
That guards the central claim that this is a matching-engine integration, not
an order-book mirror or a reimplemented reference engine.

## Qualification Results

Using Tracebook 0.6.0, generator version 2, seed `42`, 25 traces of 200 events,
and the pinned candidate:

| Profile | Fixed | Generated | Events | Coverage | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| `fifo-limit-v1` | 3/3 | 25/25 | 5,000 | 10/10 | pass |
| `fifo-partial-fill-v1` | 3/3 | 25/25 | 5,000 | 10/10 | pass |
| `fifo-full-v1` | 4/5 | stopped at trace 1, event 7 | 7 before stop | n/a | expected divergence |

The passing proof IDs are:

- FIFO limit qualification `sha256:f8db775baabe7e665d45bbb920a67e43d405acf445de4951617df68ffa24eb69`,
  campaign `sha256:59d70645ff13f12fa4af23af69631714df22ef5f25cd1104b0c1124f98f71f6a`;
- FIFO partial-fill qualification `sha256:08bf641bfa57da1892d744c08bca0be00f5ac1ade580ae8fcf7f195f0fcad6eb`,
  campaign `sha256:297f895b99ac26f92287ababd1dcc25b68b0be98787e9b9d814d5cd241769cf9`.

The nine-case standard suite passes 6/9 cases. All three stopped cases reach
the same exact difference: an unfillable FOK is `applied` with no reason in the
Tracebook reference contract, while the native engine returns a rejected
status and its `ErrFOKCannotFill` reason. The adapter preserves that native
result as `rejected / INVALID_ORDER`; it does not rewrite the result to obtain
a green report. This is a contract difference, not by itself proof of an
upstream defect.

## Retained Divergence

The full-profile campaign reduces the FOK difference from seven events to the
single event in
[`regressions/fok-rejection-reduced.jsonl`](regressions/fok-rejection-reduced.jsonl).
The retained
[`regressions/fok-rejection-failure.json`](regressions/fok-rejection-failure.json)
records failure ID `failure-1ce2954857800b3a068d`, the original and reduced
trace hashes, and `$.observation.outcome.reason` as the semantic path. CI
regenerates and byte-compares both artifacts.

## Reproduce

From this directory:

```bash
go test ./...
go build -trimpath -o /tmp/tracebook-intrepid-orderbook .

tracebook-conformance qualify \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 25 \
  --events-per-trace 200 \
  --max-minimize-runs 100 \
  --candidate-cmd /tmp/tracebook-intrepid-orderbook \
  --output-dir /tmp/intrepid-fifo-limit
```

The tag is reproducible evidence, not a production-readiness certification.
This integration makes no claim about upstream adoption, operational maturity,
latency, durability, or suitability for live trading.
