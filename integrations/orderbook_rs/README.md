# orderbook-rs Integration

This is a native Rust candidate adapter for the MIT-licensed
[`orderbook-rs`](https://github.com/joaquinbejar/OrderBook-rs) matching engine.
It speaks Tracebook protocol v1 directly over stdin/stdout; the candidate
process does not import or execute Tracebook's Python engine.

The integration pins:

- `orderbook-rs = 0.10.4` (upstream tag commit
  `92db5927ac59bf5f68ebdea011e6d7fe9a8ecb64`)
- `pricelevel = 0.8.4`
- Rust `1.88.0` (the first release that compiles upstream's let-chain usage)
- the complete transitive dependency graph in `Cargo.lock`

## Architecture

```mermaid
flowchart LR
    T["Tracebook runner"] -->|"protocol v1 JSONL"| W["Rust wire server"]
    W --> A["semantic adapter"]
    A --> E["orderbook-rs 0.10.4"]
    E --> A
    A -->|"outcome, trades, state hash"| T
    T --> D["reference comparison"]
    D --> R["report or minimized trace"]
```

`wire.rs` owns framing and canonical SHA-256 state serialization. `adapter.rs`
owns numeric conversion, source IDs, owners, lifecycle operations, trade
translation, and complete queue snapshots. `orderbook-rs` performs all matching.

## Run The Proof

From a Tracebook checkout:

```bash
python3 -m pip install -e .

cd integrations/orderbook_rs
cargo build --release --locked
cd ../..

tracebook-conformance run \
  integrations/orderbook_rs/fifo-compatible.jsonl \
  --output /tmp/orderbook-rs-report.json \
  --candidate integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The command exits `0` after 13 events and produces:

```json
{
  "candidate_engine": {
    "language": "Rust",
    "name": "orderbook-rs FIFO adapter",
    "version": "0.10.4"
  },
  "compared_events": 13,
  "conformant": true,
  "final_state_hash": "21a9606e7c77c3b239259f5032245c6330ddcd1d3f7fa25394612d9818becee3"
}
```

The trace covers FIFO fills, a decimal partial fill, priority-preserving reduce,
cancel-and-new replace, an inactive cancellation rejection, clear, and multiple
symbols.

## Compatibility Profile

The unmodified engine agrees with seven of Tracebook's eight standard cases:

| Standard case | Result |
| --- | --- |
| `fifo-lifecycle` | Conformant |
| `order-instructions` | Conformant |
| `stp-cancel-resting` | Conformant |
| `stp-cancel-incoming` | Conformant |
| `multi-symbol` | Conformant |
| `tick-grid` | Conformant |
| `deep-cancellation` | Conformant |
| `pro-rata-allocation` | Expected difference: upstream is FIFO |

Run and retain the complete matrix:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v1
tracebook-conformance suite \
  /tmp/tracebook-conformance-v1 \
  --output /tmp/orderbook-rs-suite.json \
  --candidate integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

The suite exits `1` because pro-rata is deliberately unsupported. The maintained
workflow asserts the exact `7/8` profile and suite ID instead of disguising that
boundary.

The broader generated gate exercises all FIFO instructions:

```bash
tracebook-conformance campaign \
  --output-dir /tmp/orderbook-rs-campaign \
  --profile fifo-full-v1 \
  --seed 20260713 \
  --traces 10 \
  --events-per-trace 100 \
  --candidate integrations/orderbook_rs/target/release/tracebook-orderbook-rs
```

That deterministic 1,000-event campaign is conformant with campaign ID
`sha256:3042184192ea03c666dd2120d8b8acc728b2805678c5fb5fdd849bf97a00925d`.

## Detection Control

The binary has one explicit test-only fault mode. It drops the first reported
trade while leaving the native book mutation intact:

```bash
tracebook-conformance run \
  integrations/orderbook_rs/fifo-compatible.jsonl \
  --output /tmp/orderbook-rs-drift.json \
  --candidate integrations/orderbook_rs/target/release/tracebook-orderbook-rs \
  --test-fault=drop-first-trade
```

This must exit `1` with a `trades` divergence at event 3. The scheduled workflow
runs this negative control so a green integration proves both agreement and the
ability to detect Rust-side drift.

## Translation Contract

- Source IDs map to `orderbook-rs` sequential IDs and must fit in `u64`.
- Prices are snapped with Tracebook's half-even tick rule, stored as integer
  ticks, and restored to canonical decimal strings in observations.
- Quantities use fixed-point `u64` units at `quantity_decimal_places`; a value
  that rounds to zero or overflows that range is rejected.
- Real owners map deterministically to `Hash32`. Anonymous owners receive a
  unique per-order identity so STP does not make unrelated anonymous orders
  self-match or reject them for a missing user ID.
- `CANCEL_RESTING` maps to `CancelMaker`; `CANCEL_INCOMING` maps to
  `CancelTaker`.
- Replacement is translated as validated cancel-and-new, preserving the source
  ID and owner while losing queue priority.
- Symbols are independent books, sorted in canonical snapshots.

This adapter tests behavior, not latency. Its process timing includes JSON,
pipes, translation, snapshots, and OS scheduling.

## CI

The maintained, copyable workflow is
[`../../.github/workflows/orderbook-rs.yml`](../../.github/workflows/orderbook-rs.yml).
Its core Rust gate is:

```yaml
- uses: dtolnay/rust-toolchain@1.88.0
  with:
    components: rustfmt, clippy
- run: cargo fmt --check
- run: cargo clippy --locked --all-targets -- -D warnings
- run: cargo test --locked
- run: cargo build --release --locked
```

When adapting another Rust engine, retain `wire.rs`, replace the native calls in
`adapter.rs`, update engine metadata and the compatibility matrix, then keep the
fixed trace, generated campaign, and negative control as three separate gates.
Candidate stdout is reserved for protocol frames; diagnostics belong on stderr.
