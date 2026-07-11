# Command Guide

This guide collects the commands a reviewer, contributor, or benchmark author is expected to use.

## Setup

User install:

```bash
python -m pip install tracebook-sim
```

Contributor install:

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,dashboard]"
```

Add `capture` when live public Coinbase WebSocket input is needed:

```bash
pip install -e ".[dev,dashboard,capture]"
```

Alternative contributor setup:

```bash
make setup
```

## Tests And Quality

| Command | Purpose |
| --- | --- |
| `python -m pytest --cov=tracebook --cov-fail-under=75` | Run tests and enforce the coverage baseline |
| `python test_system.py` | Run integration smoke checks |
| `python -m black --check src tests examples install_deps.py test_system.py` | Check formatting |
| `python -m flake8 src tests examples install_deps.py test_system.py` | Run lint checks |
| `python -m compileall -q src tests examples install_deps.py test_system.py` | Check source compilation |
| `python -m build --sdist --wheel --outdir dist` | Build package artifacts |
| `python -m twine check dist/*` | Validate distribution metadata and README rendering |
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
