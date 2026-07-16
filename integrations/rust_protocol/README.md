# Shared Rust Adapter Protocol

This private crate owns Tracebook protocol v1 framing, validation, canonical
book-state serialization, hashing, and the stdin/stdout server loop used by the
native Rust engine adapters.

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
