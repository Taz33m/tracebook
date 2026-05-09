# Command Guide

This guide collects the commands a reviewer, contributor, or benchmark author is expected to use.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,dashboard]"
```

Alternative contributor setup:

```bash
make setup
```

## Tests And Quality

| Command | Purpose |
| --- | --- |
| `python -m pytest` | Run the unit test suite |
| `python test_system.py` | Run integration smoke checks |
| `python -m black --check src tests examples install_deps.py` | Check formatting |
| `python -m flake8 src tests examples install_deps.py` | Run lint checks |
| `python -m compileall -q src tests examples install_deps.py` | Check source compilation |
| `python -m build --sdist --wheel --outdir dist` | Build package artifacts |
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
| `--warmup-seconds` | JIT warmup excluded from measured run |
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
| `--scenario` | `smoke`, `fifo_baseline`, `pro_rata_baseline`, `cancellation_mix`, or `all` |
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
