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
python -m flake8 src tests
python -m pytest
python test_system.py
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/local-smoke.json
tracebook-dashboard --demo-simulation --help
```

For packaging changes, also run:

```bash
python -m build --sdist --wheel --outdir dist
```

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
