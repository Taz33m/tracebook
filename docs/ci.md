# Conformance In CI

The workflow below turns matching semantics into a pull-request gate. It
installs the public PyPI release, generates the `fifo-limit-v1` workload, runs
the candidate as a separate process, and uploads JSON, JUnit, and any minimized
failure corpus even when a divergence fails the job.

The candidate command is executable code, and Tracebook does not sandbox it.
Keep workflow permissions minimal and use an isolated runner for untrusted code.

Copy [`examples/github-actions/conformance.yml`](../examples/github-actions/conformance.yml)
to `.github/workflows/conformance.yml` in the candidate engine repository after
the candidate has a tested Tracebook stdio adapter. The workflow exposes two
obvious placeholders:

1. Replace `make build` with the candidate's build command.
2. Replace `./build/matching-engine --tracebook-stdio` with its adapter command.

Those two substitutions are not the whole onboarding cost. A real adapter must
translate lifecycle semantics, numeric representation, source IDs, trades, and
priority-ordered snapshots without masking unsupported behavior. In the frozen
[design-partner measurements](qualification-design-partners.md), the two Rust
adapter files were 672 and 1,056 lines, while the Go experiment required 865
adapter lines plus 130 test lines. The
[17-line Python example](../examples/conformance_adapter.py) demonstrates
protocol framing around Tracebook's own reference adapter, not a typical native
integration.

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

      - run: python -m pip install "tracebook-conformance==0.6.0"
      - run: make build

      - name: Compare matching semantics
        run: |
          tracebook-conformance qualify \
            --profile fifo-limit-v1 \
            --seed 42 \
            --traces 25 \
            --events-per-trace 200 \
            --candidate-cmd './build/matching-engine --tracebook-stdio' \
            --output-dir artifacts/qualification

      - uses: actions/upload-artifact@v7
        if: always()
        with:
          name: matching-engine-conformance
          path: artifacts
          if-no-files-found: error
```

`tracebook-conformance qualify` exits `0` only when the selected fixed cases,
generated traces, and declared semantic coverage pass. It exits `1` on a
semantic divergence or incomplete coverage and `2` for invalid configuration,
adapter, protocol, or filesystem errors. The atomic bundle contains JSON,
JUnit, the selected suite, the campaign, and any minimized first disagreement,
so `if: always()` preserves reviewable evidence for failed builds.

The command also prints a concise review summary to the job log: profile, fixed
case count, generated trace count, semantic coverage, PASS/FAIL, and—when
present—the failure class and reduced-trace path. The sample workflow uploads
JUnit as part of the evidence bundle. JUnit does not annotate a pull request by
itself; projects that want annotations must configure a compatible test-report
consumer.

The published release matrix tests this evidence path on Ubuntu with Python
3.10-3.13. Atomic campaign and qualification publication requires
descriptor-relative filesystem operations and fails closed when they are
unavailable. Windows qualification artifact publication is not currently
supported; macOS has manual measurements but is not a release-gated target.

Projects with a deliberately narrower contract should maintain a suite that
matches their declared capabilities and run selected standard traces as a
separate compatibility profile. Do not mark unsupported semantics as conformant
inside the adapter.
