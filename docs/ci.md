# Conformance In CI

The workflow below turns matching semantics into a pull-request gate. It copies
Tracebook's versioned suite, runs the candidate as a separate process, and
uploads the complete report even when a divergence fails the job.

The candidate command is executable code, and Tracebook does not sandbox it.
Keep workflow permissions minimal and use an isolated runner for untrusted code.

Copy [`examples/github-actions/conformance.yml`](../examples/github-actions/conformance.yml)
to `.github/workflows/conformance.yml` in the candidate engine repository. Two
lines are engine-specific:

1. Replace `make build` with the candidate's build command.
2. Replace `./build/matching-engine --tracebook-stdio` with its adapter command.

```yaml
name: Matching engine conformance

on:
  pull_request:
  push:
    branches: [main]

permissions:
  contents: read

jobs:
  conformance:
    runs-on: ubuntu-latest
    timeout-minutes: 10

    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"

      - run: python -m pip install "tracebook-sim==0.3.0"
      - run: make build

      - name: Compare matching semantics
        run: |
          tracebook-conformance sample "$RUNNER_TEMP/tracebook-suite"
          mkdir -p artifacts
          tracebook-conformance suite \
            "$RUNNER_TEMP/tracebook-suite" \
            --output artifacts/conformance.json \
            --candidate ./build/matching-engine --tracebook-stdio

      - uses: actions/upload-artifact@v7
        if: always()
        with:
          name: matching-engine-conformance
          path: artifacts/conformance.json
          if-no-files-found: error
```

`tracebook-conformance suite` exits `0` only when every case agrees, `1` on a
semantic divergence, and `2` for an invalid trace, adapter, or protocol error.
Because the report is written before exit, `if: always()` preserves the exact
first disagreement for failed builds.

Projects with a deliberately narrower contract should maintain a suite that
matches their declared capabilities and run selected standard traces as a
separate compatibility profile. Do not mark unsupported semantics as conformant
inside the adapter.
