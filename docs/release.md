# Release Checklist

`tracebook` is still alpha software. Use this checklist before cutting any public release or tagged benchmark claim.

## Version And Changelog

1. Update `src/tracebook/_version.py`.
2. Update `CHANGELOG.md` with user-visible changes.
3. Confirm README commands still match the installed console scripts.
4. Confirm benchmark claims cite command, seed, machine, Python version, dependency versions, and report path.

## Local Verification

```bash
python -m pip install -e ".[dev,dashboard]"
python -m black --check src tests examples install_deps.py test_system.py
python -m flake8 src tests examples install_deps.py test_system.py
python -m bandit -q -r src
python -m pytest --cov=tracebook --cov-report=term-missing --cov-fail-under=75
python test_system.py
tracebook-sim --duration 1 --throughput 50 --algorithm FIFO --seed 1337 --warmup-seconds 0.01
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/release-smoke.json
tracebook-dashboard --demo-simulation --help
tracebook-replay examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json
tracebook-coinbase examples/data/coinbase_btcusd_l3_snapshot.json examples/data/coinbase_btcusd_full.jsonl --tick-size 0.01 --output /tmp/tracebook-coinbase.json
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
python -m pip check
```

## Remote Verification

- Push to a branch and wait for GitHub Actions on Python 3.10 through 3.13.
- Check generated package metadata in the build artifact.
- Check that Dependabot has no urgent security update waiting.
- Confirm no README badge or proof table overstates the current CI/test state.

## PyPI Trusted Publishing

The distribution name is `tracebook-sim`; the Python import is `tracebook`.
Configure a PyPI Trusted Publisher for:

- owner/repository: `Taz33m/tracebook`
- workflow: `release.yml`
- environment: `pypi`

Repository settings enforce pull requests and the Python 3.10-3.13 CI matrix on
`main`. The `pypi` deployment environment accepts only `v*` tags; keep those
protections aligned if workflow or check names change.

Publishing a GitHub release whose tag matches `v<package-version>` builds,
validates, and publishes the wheel and sdist. The workflow rejects a mismatched
tag before requesting a PyPI token.

After publishing, verify from a clean environment:

```bash
python -m pip install tracebook-sim==0.2.0
python -c "import tracebook; print(tracebook.__version__)"
tracebook-replay --help
tracebook-coinbase --help
```

## Release Notes

Release notes should include:

- user-visible matching or simulation changes
- CLI changes
- benchmark report schema changes
- dashboard changes
- compatibility notes
- known limitations

Do not publish universal latency or throughput claims from local smoke benchmarks.
