# Case Study: Reducing a Real FIFO Defect to Four Events

This case study imports a historical matching defect found by the
[`matching-engine-benchmark`](https://github.com/flash1-dev/matching-engine-benchmark)
project into a deterministic Tracebook regression. It does not claim that
Tracebook discovered the defect, and it does not apply to current
`orderbook-rs` releases.

The handoff uses the canonical-output boundary merged in
[`matching-engine-benchmark` PR #4](https://github.com/flash1-dev/matching-engine-benchmark/pull/4):
Flash localizes a report-stream disagreement, then Tracebook reconstructs and
minimizes the corresponding lifecycle prefix.

Flash reported the original problem as
[`orderbook-rs` issue #88](https://github.com/joaquinbejar/OrderBook-rs/issues/88).
Upstream fixed it through
[`orderbook-rs` PR #131](https://github.com/joaquinbejar/OrderBook-rs/pull/131).
Tracebook preserves both sides of that history:

- affected `orderbook-rs 0.8.0` commit
  `53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf` with `pricelevel 0.7.0`;
- maintained `orderbook-rs 0.12.0` with `pricelevel 0.9.1`.

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

## Direct Flash Handoff

At Flash commit `eb6e89fbc4313f77cea7d424bab14a26093cf552`, run the
canonical `normal`, seed `23` workload against the affected dependency graph
and export the candidate stream:

```bash
./harness \
  --engine /path/to/affected-orderbook-rs-adapter.so \
  --scenario normal \
  --mode perf \
  --write-canonical-output /tmp/orderbook-rs-0.8.0-candidate.txt

scripts/explain_divergence.py \
  reference/canonical_output.txt.gz \
  /tmp/orderbook-rs-0.8.0-candidate.txt \
  --json-output /tmp/divergence.json
```

The adapter must use `orderbook-rs 0.8.0` commit
`53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf` and unpatched
`pricelevel 0.7.0`. Flash's maintained working adapter applies the upstream fix;
the exact affected build is retained in Tracebook's Linux workflow.

The upstream comparator reports:

```text
First divergent sequence: 15738
Matching sequences: 15738
First divergent canonical line: 17449
Reference next makers: 146075, then 185199
Candidate next makers: 185199, then 146075
```

The exact schema-v1 JSON is committed as
[`orderbook-rs-issue-88-divergence.json`](../../integrations/flash_benchmark/artifacts/orderbook-rs-issue-88-divergence.json).
Flash sequences are zero-based and dense, so sequence `15738` selects a prefix
of 15,739 workload messages. Convert that prefix without parsing logs or
reimplementing Flash's comparator:

```bash
python integrations/flash_benchmark/bridge.py \
  integrations/flash_benchmark/artifacts/orderbook-rs-issue-88-divergence.json \
  /path/to/orders_normal_s23_n1000000.bin \
  /tmp/orderbook-rs-issue-88-prefix.jsonl

tracebook-conformance minimize \
  /tmp/orderbook-rs-issue-88-prefix.jsonl \
  --tick-size 1 \
  --quantity-decimal-places 0 \
  --events-output /tmp/orderbook-rs-issue-88-reduced.jsonl \
  --output /tmp/orderbook-rs-issue-88-minimization.json \
  --max-runs 200 \
  --candidate-cmd ./integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter
```

Tracebook finds the same semantic boundary at event `15739`. Delta debugging
uses 193 fresh candidate runs and reduces the imported prefix to four events.
The result is one-minimal and does not exhaust the run budget.

## Independent Generated Campaign

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

## Two Four-Event Regressions

The direct Flash import preserves the canonical workload's actual IDs, prices,
quantities, and IOC instruction in
[`flash-issue-88-reduced.jsonl`](../../integrations/orderbook_rs/regressions/flash-issue-88-reduced.jsonl):

1. Buy maker A rests 68 units at 33532.
2. Buy maker B rests 14 units at 33532.
3. A sell for 13 at 33531 partially fills maker A, leaving 55 units. The
   affected queue moves that remainder behind maker B.
4. A sell IOC for 75 at 33517 should consume maker A before maker B. The
   affected engine consumes maker B first.

The generated campaign independently reaches the same causal shape with small,
normalized values. Its one-minimal trace is committed as
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

The first command verifies the normalized maker-ID mismatch. The second exits
`0`. CI also replays the Flash-derived four-event trace against both engines.

These traces are causally equivalent, not byte-identical: one comes directly
from Flash's canonical workload; the other comes from Tracebook's portable
campaign profile.

## Why This Matters

Flash demonstrates broad differential discovery across many engines.
Tracebook's useful complementary role is failure forensics: preserve the exact
affected provenance, localize the first semantic difference, reduce the prefix
to a reviewable reproducer, and turn that reproducer into a stable CI test.

The bridge is deliberately an integration helper rather than a new public
Tracebook command. One real handoff now proves the boundary; more engine cases
should reveal whether a broader import API is warranted.
