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
python -m black --check src tests examples integrations install_deps.py test_system.py
python -m flake8 src tests examples integrations install_deps.py test_system.py
python -m bandit -q -r src integrations
python -m pytest --cov=tracebook --cov-report=term-missing --cov-fail-under=75
python test_system.py
tracebook-sim --duration 1 --throughput 50 --algorithm FIFO --seed 1337 --warmup-seconds 0.01
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/release-smoke.json
tracebook-dashboard --demo-simulation --help
tracebook-replay examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json
tracebook-coinbase examples/data/coinbase_btcusd_l3_snapshot.json examples/data/coinbase_btcusd_full.jsonl --tick-size 0.01 --output /tmp/tracebook-coinbase.json
tracebook-corpus verify src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1
python -m build --sdist --wheel --outdir dist
python -m twine check dist/*
python -m pip check
(
  cd integrations/orderbook_rs
  cargo fmt --check
  cargo clippy --locked --all-targets -- -D warnings
  cargo test --locked
)
```

## Remote Verification

- Push to a branch and wait for GitHub Actions on Python 3.10 through 3.13.
- Confirm the native `orderbook-rs` integration passes its fixed trace, `7/9`
  suite profile, generated campaign, and intentional-drift negative control.
- Confirm the pinned PythonMatchingEngine integration workflow passes.
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
python -m pip install --no-cache-dir tracebook-sim==0.5.0
python -c "import tracebook; print(tracebook.__version__)"
tracebook-conformance --help
python -m pip download --no-deps --no-binary=:all: tracebook-sim==0.5.0
```

Extract the downloaded sdist in a blank repository, build
`integrations/orderbook_rs` with `cargo build --release --locked`, and run the
seed-42 faulty campaign plus `reproduce` using only the installed public command
and extracted Rust source. Require event 173, a five-event reduced trace, exact
reproduction, semantic coverage, JSON, JUnit, and conformance of the correct
binary on the reduced regression case.

## Release Notes

Release notes should include:

- user-visible matching or simulation changes
- CLI changes
- benchmark, corpus, manifest, or golden-state schema changes
- dashboard changes
- compatibility notes
- known limitations

For 0.5.0, begin with [`docs/releases/0.5.0.md`](releases/0.5.0.md). The
[0.4.1 notes](releases/0.4.1.md) document the first upstream semantic review,
and the [0.4.0 notes](releases/0.4.0.md) document the failure-corpus release.

Do not publish universal latency or throughput claims from local smoke benchmarks.
