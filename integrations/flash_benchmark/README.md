# Flash Benchmark Handoff

This bounded integration connects
[`flash1-dev/matching-engine-benchmark`](https://github.com/flash1-dev/matching-engine-benchmark)
to Tracebook without making either project depend on the other.

Flash owns broad workload generation, canonical report-stream consensus, and
first-divergence localization. Tracebook consumes the resulting sequence number,
reconstructs the corresponding lifecycle prefix, minimizes it, and stores a
deterministic regression.

## Stable Boundary

Flash PR
[`#4`](https://github.com/flash1-dev/matching-engine-benchmark/pull/4)
established two upstream-owned interfaces:

- `--write-canonical-output <path>` exports candidate report bytes after timing;
- `scripts/explain_divergence.py` emits schema-v1
  `matching-engine-benchmark.canonical-divergence` JSON.

The local [`bridge.py`](bridge.py) reads only that JSON and Flash workload
version 1. It does not parse benchmark logs, reproduce Flash's comparator, or
add a Tracebook dependency to Flash.

## Real orderbook-rs Result

The committed
[`orderbook-rs-issue-88-divergence.json`](artifacts/orderbook-rs-issue-88-divergence.json)
was emitted by Flash's merged comparator for its canonical `normal`, seed `23`
workload and the affected `orderbook-rs 0.8.0` plus `pricelevel 0.7.0` adapter.

It records:

```text
Matching sequences: 15,738
First divergent sequence: 15,738
First divergent canonical line: 17,449
Reference next makers: 146075, then 185199
Candidate next makers: 185199, then 146075
```

Flash sequence numbers are zero-based and dense. Sequence `15738` therefore
selects a Tracebook prefix of `15,739` events.

## Convert And Reduce

After Flash generates `orders_normal_s23_n1000000.bin`:

```bash
python integrations/flash_benchmark/bridge.py \
  integrations/flash_benchmark/artifacts/orderbook-rs-issue-88-divergence.json \
  /path/to/matching-engine-benchmark/orders_normal_s23_n1000000.bin \
  /tmp/orderbook-rs-issue-88-prefix.jsonl

tracebook-conformance minimize \
  /tmp/orderbook-rs-issue-88-prefix.jsonl \
  --tick-size 1 \
  --quantity-decimal-places 0 \
  --events-output /tmp/orderbook-rs-issue-88-reduced.jsonl \
  --output /tmp/orderbook-rs-issue-88-minimization.json \
  --max-runs 200 \
  --candidate-cmd integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter
```

The conversion hash is
`sha256:4e0bb497924a68dcca9575a5860954089bf5d137ab1f577618fba482a19877cd`.
Delta debugging takes 193 candidate runs, reaches a one-minimal four-event
trace, and does not exhaust its budget. The exact result is committed as
[`flash-issue-88-reduced.jsonl`](../orderbook_rs/regressions/flash-issue-88-reduced.jsonl).

Replay the affected engine and gate the fixed one:

```bash
tracebook-conformance reproduce \
  integrations/orderbook_rs/regressions/flash-issue-88-reduced.jsonl \
  --tick-size 1 \
  --quantity-decimal-places 0 \
  --candidate-cmd integrations/orderbook_rs/target/release/orderbook-rs-issue-88-adapter

tracebook-conformance run \
  integrations/orderbook_rs/regressions/flash-issue-88-reduced.jsonl \
  --tick-size 1 \
  --quantity-decimal-places 0 \
  --candidate-cmd integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The historical engine reports `queue-priority drift`; the maintained 0.12.0
engine exits `0`.

## Reproducibility

[`orderbook-rs-issue-88-provenance.json`](artifacts/orderbook-rs-issue-88-provenance.json)
pins the Flash commit, affected dependency graph, workload identity, source
artifact hash, converted prefix hash, minimization result, and fixed-engine
gate. The maintained
[`orderbook-rs` workflow](../../.github/workflows/orderbook-rs.yml) rebuilds the
Linux handoff from the pinned upstream commit.

The workload binary and canonical streams are not vendored. They are large,
reproducible upstream artifacts; the small divergence JSON and four-event
regression are the durable review surfaces.
