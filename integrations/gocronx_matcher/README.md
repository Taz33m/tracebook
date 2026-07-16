# gocronx/matcher Adapter

This integration drives [`gocronx/matcher`](https://github.com/gocronx/matcher)
through Tracebook's version 1 NDJSON process protocol. The dependency is pinned
to commit `b8d48356c8a2677e0d8a1965d754e3c4884bb947` (`matcher 0.2.0`) and the
toolchain is pinned to Rust 1.88.0.

The adapter qualifies for `fifo-limit-v1`: FIFO limit orders, partial fills,
cancellation, quantity reduction, priority-losing replacement, clear, duplicate
active IDs, inactive lifecycle requests, and independent symbol books.

## Reproduce The Qualification

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
cargo build --release --locked

tracebook-conformance qualify \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 25 \
  --events-per-trace 200 \
  --candidate-cmd ./target/release/tracebook-gocronx-matcher \
  --output-dir .tracebook/qualification
```

The frozen run produces 3/3 passing fixed cases, 25/25 conformant generated
traces, 10/10 covered capabilities, and 28 candidate process runs.

## Translation Boundary

| Tracebook operation | `matcher` API |
| --- | --- |
| `new` | `OrderBook::submit_events` |
| `cancel` | `OrderBook::cancel_events` |
| `reduce` | quantity-only `OrderBook::amend` |
| `replace` | validated `cancel` followed by `submit_events` under the same ID |
| `clear` | replace the symbol's `OrderBook` |
| full queue state | decode snapshot format version 1, then canonicalize bid price order |

The adapter scales prices into integer ticks and quantities into configured
fixed-point units. Source owners are retained in adapter metadata because
`matcher` does not store them, then reconciled against the native snapshot after
every event.

## Open Contract Questions

The adapter is intentionally explicit about two assumptions awaiting an upstream
answer in [`gocronx/matcher` issue #7](https://github.com/gocronx/matcher/issues/7):

1. `OrderBook::snapshot()` format version 1 is used as the complete FIFO
   observation surface. The upstream source documents FIFO emission and
   round-trip preservation, but has not yet confirmed the byte layout as a
   stable public inspection contract.
2. Tracebook replacement is modeled as cancel followed by resubmission under the
   same `OrderId`. This produces the required priority loss but has not yet been
   confirmed as the maintainer's preferred external representation.

The qualification proves behavior at the pinned revision. It does not convert
those assumptions into promises about future upstream releases.

## Scope

The public qualification claim is only `fifo-limit-v1`. The adapter can translate
MARKET, IOC, and FOK submissions, but `fifo-full-v1` is not part of this evidence
bundle. `matcher` has no pro-rata or Tracebook STP policy surface, so those
profiles are rejected during the protocol handshake rather than silently
approximated.
