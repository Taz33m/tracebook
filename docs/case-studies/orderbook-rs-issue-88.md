# Case Study: Reducing a Real FIFO Defect to Four Events

This case study imports a historical matching defect found by the
[`matching-engine-benchmark`](https://github.com/flash1-dev/matching-engine-benchmark)
project into a deterministic Tracebook regression. It does not claim that
Tracebook discovered the defect, and it does not apply to current
`orderbook-rs` releases.

Flash reported the original problem as
[`orderbook-rs` issue #88](https://github.com/joaquinbejar/OrderBook-rs/issues/88).
Upstream fixed it through
[`orderbook-rs` PR #131](https://github.com/joaquinbejar/OrderBook-rs/pull/131).
Tracebook preserves both sides of that history:

- affected `orderbook-rs 0.8.0` commit
  `53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf` with `pricelevel 0.7.0`;
- maintained `orderbook-rs 0.11.0` with `pricelevel 0.8.4`.

## The Defect

At one price level, the historical queue removed the head maker before
matching. After a partial fill it pushed the maker's remainder onto the tail.
That preserved aggregate depth but moved a later same-price maker ahead of the
older maker's remainder.

The source snapshot path sorted resting orders by admission timestamp, so it
could still display the older maker first after the internal queue had moved it
to the back. Aggregate depth and that timestamp-oriented snapshot therefore
looked plausible. The next trade's maker ID exposed the actual consumption
order.

## Reproduce the Campaign

Build the current and historical candidates from the same adapter source:

```bash
python3 -m pip install -e .

cargo build --release --locked \
  --manifest-path integrations/orderbook_rs/Cargo.toml

cargo build --release --locked \
  --manifest-path integrations/orderbook_rs/Cargo.toml \
  --no-default-features \
  --features historical-issue-88 \
  --bin orderbook-rs-issue-88-adapter
```

Run 200 generated lifecycle events per trace. Candidate behavior does not
influence trace generation:

```bash
tracebook-conformance campaign \
  --profile fifo-partial-fill-v1 \
  --seed 42 \
  --traces 1000 \
  --events-per-trace 200 \
  --max-minimize-runs 200 \
  --candidate-cmd ./integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter \
  --corpus-dir .tracebook/corpus \
  --stop-after-first \
  --junit-output .tracebook/orderbook-rs-issue-88.xml
```

The first trace reports:

```text
Semantic coverage: 10/10 capabilities
Divergence detected at original event 173
Failure class: queue-priority drift
Original trace: 173 events
Reduced reproducer: 4 events
Campaign seed: 42
Campaign hash: sha256:e8e158af0223b4e61dbb7efeab10cfd1b34b0d3b478b3e086c12bea008c0b4aa
Failure id: failure-7dd023c684cdb2d0fc0e
```

The exact divergence is:

```text
Path: $.observation.trades[0].sell_order_id
Reference maker: 9100000001
Historical candidate maker: 9100000002
```

## Four Causal Events

The one-minimal trace is committed as
[`issue-88-reduced.jsonl`](../../integrations/orderbook_rs/regressions/issue-88-reduced.jsonl):

1. Maker A rests 5 units at 100.
2. Maker B rests 5 units at 100.
3. Taker C buys 2 units, partially filling maker A.
4. Taker D buys 1 unit. FIFO requires maker A again; the affected engine uses maker B.

Replay the exact stored failure against the historical candidate:

```bash
tracebook-conformance reproduce \
  .tracebook/corpus/failure-7dd023c684cdb2d0fc0e/reduced.jsonl \
  --candidate-cmd ./integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter
```

Then use the committed trace as a regression gate for the fixed release:

```bash
tracebook-conformance run \
  integrations/orderbook_rs/regressions/issue-88-reduced.jsonl \
  --candidate-cmd ./integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The first command verifies the exact maker-ID mismatch. The second exits `0`.
CI rebuilds both upstream versions, regenerates and minimizes the 173-event
failure, checks the stored JSON and JUnit evidence, and requires the fixed
engine to conform on the four-event regression.

## Why This Matters

Flash demonstrates broad differential discovery across many engines.
Tracebook's useful complementary role is failure forensics: preserve the exact
affected provenance, localize the first semantic difference, reduce the prefix
to a reviewable reproducer, and turn that reproducer into a stable CI test.

The practical next interoperability step is a deterministic, machine-readable
export of one Flash candidate's canonical report stream and first divergent
prefix. That would let a discrepancy move from broad discovery into Tracebook
reduction without hand-transcribing the engine-specific evidence.
