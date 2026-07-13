#!/usr/bin/env python3
"""Minimal Python stdio adapter implementing the conformance protocol."""

from tracebook.conformance import ReferenceEngineAdapter, serve_stdio


def main() -> int:
    return serve_stdio(
        lambda config: ReferenceEngineAdapter(
            config,
            engine_name="example-python-adapter",
        )
    )


if __name__ == "__main__":
    raise SystemExit(main())
