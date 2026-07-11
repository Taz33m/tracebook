# Contributing

Thanks for helping make `tracebook` better. This project is early alpha, so the best contributions are focused, tested, and honest about behavior.

## Development Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -e ".[dev,dashboard]"
```

## Local Checks

Run these before opening a pull request:

```bash
python -m black --check src tests examples install_deps.py test_system.py
python -m flake8 src tests examples install_deps.py test_system.py
python -m pytest --cov=tracebook --cov-report=term-missing --cov-fail-under=75
python test_system.py
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/local-smoke.json
tracebook-dashboard --demo-simulation --help
tracebook-replay examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json
tracebook-coinbase examples/data/coinbase_btcusd_l3_snapshot.json examples/data/coinbase_btcusd_full.jsonl --tick-size 0.01 --output /tmp/tracebook-coinbase.json
tracebook-corpus verify src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1
```

For packaging changes, also run:

```bash
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
```

Normalized feed adapters should emit `tracebook.MarketEvent` values and keep
venue-specific parsing, sequence checks, and assumptions outside the core replay
engine. Include a small fixture and strict rejection tests with each adapter.
Fixtures must be synthetic or explicitly redistributable, and adapter semantics
must cite the venue's primary documentation.

Do not commit a live market-data corpus unless the repository has explicit
redistribution permission. Pseudonymized captures are still market data. Corpus
format changes must update schema tests, regenerate the synthetic fixture, and
document an intentional corpus-ID change.

The command reference in `docs/commands.md` is the source of truth for reviewer-facing smoke paths.

## Pull Requests

- Keep changes scoped to one clear theme.
- Add or update tests for user-visible behavior.
- Keep performance claims tied to a reproducible benchmark command and machine context.
- Prefer public APIs under `tracebook.*`; do not reintroduce old `src/core` style imports.
- If a change affects matching semantics, include a small example in the PR description.

## Commit Identity

GitHub contributor attribution depends on the commit author email being verified on your GitHub account. Check before committing:

```bash
git config user.name
git config user.email
```

## Reporting Bugs

Please include:

- Python version and OS
- install command used
- minimal code or CLI command that reproduces the issue
- expected vs actual behavior
- relevant benchmark JSON when reporting performance regressions
