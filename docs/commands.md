# Command Guide

The maintained native Rust adapter and its pinned commands live in
[`integrations/orderbook_rs/`](../integrations/orderbook_rs/README.md). The
narrower Python adapter lives in
[`integrations/python_matching_engine/`](../integrations/python_matching_engine/README.md).
A generic GitHub Actions gate is documented in [`docs/ci.md`](ci.md).

This guide collects the commands a reviewer, contributor, or benchmark author is expected to use.

## Setup

If this environment already contains `tracebook-sim` 0.5.x, remove that
legacy package owner before installing either 0.6.0 distribution:

```bash
python -m pip uninstall -y tracebook-sim
```

Conformance and core-library install:

```bash
python -m pip install "tracebook-conformance==0.6.0"
```

This ordinary install has no mandatory runtime dependencies and does not
resolve NumPy or psutil. Install the compatibility facade for simulator and
workbench commands:

```bash
python -m pip install "tracebook-sim==0.6.0"
```

For the one-time full-surface migration from `tracebook-sim` 0.5.x, continue
after the uninstall above with:

```bash
python -m pip install "tracebook-sim==0.6.0"
```

See the [package-boundary decision](../packaging/lightweight-conformance.md)
for the ownership rationale and recovery command.

Contributor install:

```bash
python -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev]"
python -m pip install -e "./packaging/tracebook-sim[dashboard]"
```

Add `capture` when live public Coinbase WebSocket input is needed:

```bash
python -m pip install -e "./packaging/tracebook-sim[capture,dashboard]"
```

Alternative contributor setup:

```bash
make setup
```

## Tests And Quality

| Command | Purpose |
| --- | --- |
| `python -m pytest --cov=tracebook --cov-fail-under=75` | Run tests and enforce the coverage baseline |
| `python -m black --check src tests examples integrations experiments tools` | Check formatting |
| `python -m flake8 src tests examples integrations experiments tools` | Run lint checks |
| `python -m compileall -q src tests examples integrations experiments tools` | Check source compilation |
| `make build` | Build and validate both distribution wheels and sdists |
| `make verify-distribution-split` | Prove version, dependency, command, file-ownership, uninstall, and migration contracts |
| `python tools/smoke_conformance_wheel.py dist/conformance/*.whl` | Prove normal conformance-only installation and qualification without NumPy or psutil |
| `python -m pip check` | Validate installed dependency consistency |

## Simulation CLI

Basic FIFO run:

```bash
tracebook-sim --duration 5 --throughput 500 --algorithm FIFO
```

Deterministic run with lifecycle events:

```bash
tracebook-sim \
  --duration 5 \
  --throughput 500 \
  --algorithm FIFO \
  --seed 1337 \
  --cancel-ratio 0.05 \
  --replace-ratio 0.02 \
  --warmup-seconds 0.05 \
  --output benchmark_results/simulation.json
```

Options:

| Option | Meaning |
| --- | --- |
| `--duration` | Simulation duration in seconds |
| `--throughput` | Target new orders per second |
| `--algorithm` | `FIFO` or `PRO_RATA` |
| `--seed` | Deterministic synthetic order-flow seed |
| `--output` | Optional JSON result path |
| `--cancel-ratio` | Probability of a cancel lifecycle event after a new order |
| `--replace-ratio` | Probability of a replace lifecycle event after a new order |
| `--warmup-seconds` | Interpreter/cache warmup excluded from measured run |
| `--magic-trace` | Enable magic-trace integration or fallback tracing |

## Matching-Engine Conformance

Produce one profile-scoped qualification bundle:

```bash
tracebook-conformance qualify \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 25 \
  --events-per-trace 200 \
  --candidate-cmd './engine-adapter' \
  --output-dir .tracebook/qualification
```

This runs profile-relevant immutable suite cases, a generated campaign, and a
semantic coverage gate. The output contains canonical JSON, JUnit, and any
automatically minimized failure. Use this command for a first integration or a
profile-level CI claim; use `suite` when intentionally comparing every broader
fixed semantic surface.

Prepare and verify a release-grade two-run evidence pair:

```bash
tracebook-conformance evidence-init /path/to/candidate \
  --workspace .tracebook/release-evidence \
  --candidate-name owner/repository \
  --candidate-revision REVISION

# Run the canonical qualification from evidence-plan.json in both generated
# roots, with all three --candidate-* identity flags, then:
tracebook-conformance evidence-verify \
  .tracebook/release-evidence/evidence-plan.json
```

`evidence-init` refuses an existing workspace and creates independent candidate,
adapter, build, cache, and qualification paths for `run-1` and `run-2`.
`evidence-verify` writes `evidence-manifest.json` only after both canonical
bundles are qualified, byte-identical, and bound to the unchanged pinned
candidate. See [Captured Two-Run Evidence](conformance.md#captured-two-run-evidence)
for the exact qualification command and trust boundary.

Copy and run the standard suite:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v2 --suite-version v2

tracebook-conformance suite \
  /tmp/tracebook-conformance-v2 \
  --output /tmp/conformance-suite.json \
  --candidate python examples/conformance_adapter.py
```

Suite v2 is the current default. Pass `--suite-version v1` only when
reproducing the original immutable eight-case suite and its historical hash.

Compare one trace:

```bash
tracebook-conformance run events.jsonl \
  --algorithm fifo \
  --tick-size 0.01 \
  --self-trade-policy NONE \
  --output conformance-report.json \
  --candidate ./engine-adapter
```

Minimize a failure:

```bash
tracebook-conformance minimize events.jsonl \
  --events-output minimal.jsonl \
  --output minimization.json \
  --max-runs 100 \
  --candidate ./engine-adapter
```

Generate stateful traces and minimize the first drift:

```bash
tracebook-conformance campaign \
  --profile fifo-limit-v1 \
  --seed 42 \
  --traces 1000 \
  --events-per-trace 200 \
  --max-minimize-runs 100 \
  --candidate-cmd ./engine-adapter \
  --corpus-dir .tracebook/corpus \
  --stop-after-first \
  --junit-output .tracebook/campaign.xml
```

Replay the exact saved failure:

```bash
tracebook-conformance reproduce \
  .tracebook/corpus/failure-bc8b19d3e0e3441a98db/reduced.jsonl \
  --output reproduction.json \
  --junit-output reproduction.xml \
  --candidate-cmd ./engine-adapter
```

Campaign output contains canonical JSON, semantic coverage, and, on failure,
the original prefix plus a minimized JSONL reproducer. `--corpus-dir` stores a
deterministically named bundle; `--output-dir` retains the single-run directory
layout. A selected bundle path must not already exist.

`--candidate` must be the final option; all remaining values are passed to the
adapter command. Prefer `--candidate-cmd` when Tracebook options follow the
candidate. The suite carries its own algorithm, tick size, self-trade
policy, and quantity normalization per case. See `docs/conformance.md` for the
stdio protocol, campaign profiles, and artifact contracts.

## Benchmark CLI

Smoke benchmark:

```bash
tracebook-benchmark --scenario smoke --seed 1337 --warmup-seconds 0.01
```

Full local report:

```bash
tracebook-benchmark \
  --scenario all \
  --seed 1337 \
  --warmup-seconds 0.05 \
  --output benchmark_results/local.json
```

Options:

| Option | Meaning |
| --- | --- |
| `--scenario` | Any documented scenario (`smoke`, FIFO/pro-rata baselines, lifecycle/deep/multi-symbol), or `all` |
| `--seed` | Base random seed |
| `--warmup-seconds` | Warmup excluded from results |
| `--duration` | Override scenario duration |
| `--throughput` | Override scenario target throughput |
| `--output` | Optional JSON benchmark report path |

## Dashboard CLI

```bash
tracebook-dashboard --port 8050 --demo-simulation --demo-throughput 200 --seed 1337
```

Options:

| Option | Meaning |
| --- | --- |
| `--port` | Local dashboard port |
| `--host` | Bind host |
| `--allow-remote` | Allow the unauthenticated dashboard to bind to a non-loopback host |
| `--update-interval` | Dashboard update interval in milliseconds |
| `--demo-simulation` | Start a background simulation for live data |
| `--demo-duration` | Demo simulation duration |
| `--demo-throughput` | Demo target orders per second |
| `--seed` | Demo simulation seed |

The dashboard binds to loopback by default. Binding to a non-loopback address requires
`--allow-remote` because the demo dashboard does not provide authentication.

## Historical Event Replay

```bash
tracebook-replay examples/data/sample_events.jsonl \
  --algorithm fifo \
  --tick-size 0.01 \
  --self-trade-policy NONE \
  --include-trades \
  --output replay-summary.json
```

Use `--lenient` to record invalid or inapplicable events and continue. Strict
mode is the default. `--include-trades` adds source-id annotated executions to
the otherwise compact summary. See `docs/event-replay.md` for the normalized
schema.

## Coinbase Exchange L3 Replay

```bash
tracebook-coinbase \
  examples/data/coinbase_btcusd_l3_snapshot.json \
  examples/data/coinbase_btcusd_full.jsonl \
  --tick-size 0.01 \
  --events-output /tmp/coinbase-events.jsonl \
  --include-trades \
  --output /tmp/coinbase-replay.json
```

The command accepts recorded Coinbase `full` objects and compact `level3`
arrays, validates per-product sequence continuity, and keeps observed exchange
matches separate from simulator-generated trades. `--tick-size` must equal the
product's Coinbase `quote_increment`. See `docs/coinbase-l3.md` for replay
synchronization, strictness, and limitation details.

## Coinbase Corpus CLI

Verify the checked synthetic corpus:

```bash
tracebook-corpus sample /tmp/tracebook-sample-corpus
tracebook-corpus verify /tmp/tracebook-sample-corpus
```

Produce a machine-attributed report and compare two runs:

```bash
tracebook-corpus benchmark \
  /tmp/tracebook-sample-corpus \
  --iterations 10 \
  --warmups 2 \
  --output benchmark_results/corpus-baseline.json

tracebook-corpus compare \
  benchmark_results/corpus-baseline.json \
  benchmark_results/corpus-candidate.json
```

`capture` and `prepare` create a new directory atomically and refuse to
overwrite an existing corpus. Live capture requires the `capture` extra and an
explicit market-data-terms acknowledgement. See `docs/corpora.md` for the full
workflow, rights boundary, artifact schemas, and regeneration command.

## Benchmark Claim Checklist

Do not publish a performance claim unless the report includes:

- exact command
- seed
- warmup duration
- machine and OS
- Python version
- dependency versions
- scenario config
- generated JSON report path

Use `docs/performance.md` as the baseline reporting format.
