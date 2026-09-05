# NautilusTrader L3 Book-Replay Integration

This optional adapter checks NautilusTrader's native Rust `OrderBook` through
Tracebook's separate `l3-book-replay-v1` surface. It does **not** run the
matching-engine protocol and does not claim that NautilusTrader is a submitted
order versus submitted order CLOB.

The integration pins the pre-release Python binding whose public API exposes
`OrderBook.simulate_fills(BookOrder)` directly:

- package: `nautilus-trader==2.0.0rc3`;
- tag: `v2.0.0rc3`;
- source commit: `648970ce64a304d93da0a29320cb6e19b905fa39`;
- repository: <https://github.com/nautechsystems/nautilus_trader>.

The stable `1.231.0` release and the v2 release candidate are different Python
binding generations. This adapter intentionally fails closed on any installed
version other than the pin above.

## What The Proof Exercises

The adapter creates one native `L3_MBO` `OrderBook` per trace symbol. `add`,
`update`, and `delete` events call the corresponding native book methods. A
`probe` constructs a native `BookOrder` and calls the native
`OrderBook.simulate_fills` method. Snapshots traverse `book.bids()`,
`book.asks()`, and each level's FIFO `get_orders()` result.

The 17-event bundled trace covers:

- same-price FIFO insertion;
- a size increase that retains its current queue slot;
- ordered partial simulation across orders and price levels;
- a price-changing update that requeues at its destination level;
- deletion by source order ID;
- an inactive-delete rejection at the normalized adapter boundary;
- independent symbols; and
- probes that leave the native book unchanged.

No trace order is submitted to NautilusTrader's execution engine. The only
liquidity is the exogenous L3 state built from deltas.

The scheduled proof also runs generator version 1 with seed `20260831`: 25
independent traces of 100 events each. The stable campaign ID is
`sha256:09b1599eaeb474d98617acc7869ace26759ed5ab8350803f06197ef572864bab`.
It reaches all 11 declared semantic capabilities and passes all 2,500 events.

## Run The Pinned Proof

Use a separate Python 3.12-3.14 environment because the v2 candidate does not
support Tracebook's Python 3.10 floor:

```bash
python3.14 -m venv /tmp/tracebook-nautilus
/tmp/tracebook-nautilus/bin/python -m pip install -e . \
  "nautilus-trader==2.0.0rc3"

/tmp/tracebook-nautilus/bin/python -m tracebook.book_replay run \
  src/tracebook/book_replay/fixtures/l3-book-replay-v1.jsonl \
  --output /tmp/nautilus-book-replay.json \
  --candidate /tmp/tracebook-nautilus/bin/python \
    integrations/nautilus_trader/adapter.py
```

The command exits `0` and the pinned candidate produces:

```json
{
  "candidate_engine": {
    "language": "Rust/Python",
    "name": "NautilusTrader L3 OrderBook",
    "version": "2.0.0rc3"
  },
  "compared_events": 17,
  "conformant": true,
  "final_state_hash": "9e0af6be935dce940a87497788c7a9a799c71f05e34a0204c9d294fce611b002"
}
```

## Negative Control And Retained Evidence

[`faulty_adapter.py`](faulty_adapter.py) subclasses the real pinned native
adapter and injects one deliberate fault: a same-price size increase is
implemented as delete-plus-add, moving the order to the back of its level.
This file is a harness negative control, not a candidate integration.

The generated campaign must exit `1`, localize the first mismatch to
`$.state.books[0].bids[0].order_id`, and reduce it to the three events in
[`regressions/upsize-requeue-reduced.jsonl`](regressions/upsize-requeue-reduced.jsonl).
The retained metadata in
[`regressions/upsize-requeue-failure.json`](regressions/upsize-requeue-failure.json)
pins failure ID `failure-dfe5c23848b63211b655`, the reduced trace hash, and
the expected localized divergence. CI regenerates both forms and compares them
to the retained evidence. This proves the test surface can reject a known
native queue-priority defect; it does not imply the unmodified upstream has
that defect.

## License Boundary

NautilusTrader declares `LGPL-3.0-only` in its Cargo workspace. It is not a
Tracebook dependency, no Nautilus source or binary is vendored here, and the
normal Tracebook wheel does not import it. The optional adapter loads a
user-installed Nautilus wheel in the candidate process and communicates with
the Tracebook runner over NDJSON.

That process boundary avoids adding an LGPL Rust crate to Tracebook's build,
but it does not waive obligations for someone who redistributes a combined
bundle. In particular, a distributed binary statically linking the Nautilus
Rust crates needs an explicit LGPL section 4 compliance design. This note is a
distribution warning, not legal advice.
