# Shared Rust Adapter Protocol

This private crate owns Tracebook protocol v1 framing, validation, canonical
book-state serialization, hashing, and the stdin/stdout server loop used by the
native Rust engine adapters.

The crate also supplies the exact `QuantityEncoding` boundary shared by
`orderbook_rs` and `gocronx_matcher`. Quantities become integer native lots at a
fixed twelve decimal places, independently of `quantity_decimal_places`.
That protocol setting controls half-even rounding of output quantities only;
it never changes submitted order, reduction, or replacement quantities.

The individual input range is positive `u64` lots: from `0.000000000001` through
`18446744.073709551615`. Inputs with additional non-zero decimal places or a
larger scaled value are rejected before native submission. Scientific notation
and insignificant trailing zeros are handled exactly. Parsing uses checked
integer arithmetic rather than a decimal parser that could silently round a
long mantissa into range. Each native engine retains its own capacity limits.

The fixed twelve-place scale preserves the native lots used by existing
default-precision qualification workloads. Other numeric configurations remain
unqualified. For example, output precision eighteen exposes the native exact
remainder `0.2` after `0.3 - 0.1`, while the binary64 reference reports
`0.19999999999999998`. The adapters preserve native results, including such
differences; they do not imitate the reference's arithmetic to obtain a pass.

Engine-specific crates implement `EngineAdapter` and provide a constructor.
They must not copy protocol frames or server behavior locally. This keeps every
Rust candidate on the same observation contract while leaving matching and
translation semantics inside each adapter.

Run its isolated checks with:

```bash
cargo fmt --check
cargo clippy --locked --all-targets -- -D warnings
cargo test --locked
```
