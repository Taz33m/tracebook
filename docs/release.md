# Release Checklist

`tracebook` is still alpha software. Use this checklist before cutting any public release or tagged benchmark claim.

## Version And Changelog

1. Update `src/tracebook/_version.py`.
2. Update `CHANGELOG.md` with user-visible changes.
3. In the final release commit, update `CITATION.cff` to the version, date, and
   `tracebook-conformance` PyPI URL that will actually be published, and mark
   that minor line supported in `SECURITY.md`. Keep both at the latest public
   release while a version remains Unreleased.
4. Confirm README commands still match the installed console scripts.
5. Confirm benchmark claims cite command, seed, machine, Python version,
   dependency versions, and report path.

## Local Verification

```bash
python -m pip install -e ".[dev]"
python -m pip install -e "./packaging/tracebook-sim[dashboard]"
python -m black --check src tests examples integrations experiments tools
python -m flake8 src tests examples integrations experiments tools
python -m mypy --python-version "$(python -c 'import sys; print("%d.%d" % sys.version_info[:2])')" src/tracebook experiments tools
python -m bandit -q -r src integrations tools
python -m compileall -q src tests examples integrations experiments tools
python -m pytest --cov=tracebook --cov-report=term-missing --cov-fail-under=75
tracebook-sim --duration 1 --throughput 50 --algorithm FIFO --seed 1337 --warmup-seconds 0.01
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/release-smoke.json
tracebook-dashboard --demo-simulation --help
tracebook-replay examples/data/sample_events.jsonl --output /tmp/tracebook-replay.json
tracebook-coinbase examples/data/coinbase_btcusd_l3_snapshot.json examples/data/coinbase_btcusd_full.jsonl --tick-size 0.01 --output /tmp/tracebook-coinbase.json
tracebook-corpus verify src/tracebook/corpus/fixtures/coinbase-btcusd-synthetic-v1
python -m build --sdist --wheel --outdir dist/conformance .
python -m build --sdist --wheel --outdir dist/simulator packaging/tracebook-sim
python tools/verify_distribution_privacy.py dist/conformance/* dist/simulator/*
python -m twine check dist/conformance/* dist/simulator/*
python -m pip download \
  --no-deps \
  --only-binary=:all: \
  --dest dist/legacy \
  "tracebook-sim==0.5.0"
python tools/verify_distribution_split.py \
  --expected-version 0.6.0 \
  --conformance-wheel dist/conformance/*.whl \
  --sim-wheel dist/simulator/*.whl \
  --legacy-wheel dist/legacy/tracebook_sim-0.5.0-py3-none-any.whl \
  --legacy-wheel-sha256 d190e1c2af83e5d853b0734b4d9627b1a8f6707e0fbab391015d2d94437cd4da \
  --resolver-runtime-checks
python tools/verify_sdist_wheel_agreement.py \
  --conformance-wheel dist/conformance/*.whl \
  --conformance-sdist dist/conformance/*.tar.gz \
  --sim-wheel dist/simulator/*.whl \
  --sim-sdist dist/simulator/*.tar.gz \
  --expected-version 0.6.0
python tools/smoke_conformance_wheel.py dist/conformance/*.whl
python -m pip check
(
  cd integrations/orderbook_rs
  cargo fmt --check
  cargo clippy --locked --all-targets -- -D warnings
  cargo test --locked
)
```

## Remote Verification

- Review the [PR #84 disposition](pr-84-review-disposition.md): release-code
  fixes do not close the separately frozen research findings or authorize a release.
- Push to a branch and wait for Ubuntu GitHub Actions on Python 3.10 through
  3.13.
- Confirm the native `orderbook-rs` integration passes its fixed trace, `7/9`
  suite profile, generated campaign, and intentional-drift negative control.
- Confirm the pinned PythonMatchingEngine integration workflow passes.
- Confirm the pinned Go CLOB's FIFO qualifications and retained FOK divergence,
  and the separate Nautilus L3 campaign and injected queue-priority control.
  These are distinct contracts, not interchangeable qualification claims.
- Check both generated package metadata sets and their disjoint wheel RECORDs.
- Check that Dependabot has no urgent security update waiting.
- Confirm no README badge or proof table overstates the current CI/test state.

## Coordinated PyPI Publishing

`tracebook-conformance` owns the importable `tracebook` package and the
qualification command. `tracebook-sim` owns no Python package; it is the
compatibility facade for the seven simulator/workbench commands and depends on
the exact matching conformance version. Read
[`packaging/lightweight-conformance.md`](../packaging/lightweight-conformance.md)
before publishing.

Configure a PyPI Trusted Publisher for both distribution names:

- owner/repository: `Taz33m/tracebook`
- workflow: `release.yml`
- environment: `pypi`

`tracebook-conformance` is a new project name in 0.6.0, so create its pending
Trusted Publisher on PyPI before publishing the first release. Do not publish
the facade until that exact conformance version can be downloaded from the
public index.

Repository settings enforce pull requests and the Ubuntu/Python 3.10-3.13 CI
matrix on `main`. The `pypi` deployment environment accepts only `v*` tags;
keep those protections aligned if workflow or check names change.

Publishing a GitHub release whose tag matches `v<package-version>` builds,
validates, and uploads both wheels and sdists from the same revision. The
workflow rejects a mismatched tag before requesting a PyPI token, publishes
`tracebook-conformance` first, waits until its exact wheel is visible on PyPI,
and only then publishes `tracebook-sim`.

After publishing, verify the lightweight path from a clean environment:

```bash
python -m pip install --no-cache-dir "tracebook-conformance==0.6.0"
python -c "import tracebook; print(tracebook.__version__)"
tracebook-conformance --help
python -m pip check
python -m pip download --no-deps --no-binary=:all: "tracebook-conformance==0.6.0"
```

Extract the downloaded sdist in a blank repository, build
`integrations/orderbook_rs` with `cargo build --release --locked`, and run the
seed-42 faulty campaign plus `reproduce` using only the installed public command
and extracted Rust source. Require event 173, a five-event reduced trace, exact
reproduction, semantic coverage, JSON, JUnit, and conformance of the correct
binary on the reduced regression case.

Verify the full facade independently in a second clean environment:

```bash
python -m pip install --no-cache-dir "tracebook-sim==0.6.0"
python -c "import importlib.metadata as m; assert m.version('tracebook-conformance') == m.version('tracebook-sim') == '0.6.0'"
tracebook-sim --version
tracebook-benchmark --version
tracebook-dashboard --version
tracebook-web --version
tracebook-replay --help
tracebook-coinbase --help
tracebook-corpus --help
python -m pip check
```

Finally, test the supported 0.5.x ownership migration. A direct in-place
upgrade is unsafe because the old `tracebook-sim` uninstall record owns paths
that move to `tracebook-conformance`:

```bash
python -m pip install "tracebook-sim==0.5.0"
python -m pip uninstall -y tracebook-sim
python -m pip install "tracebook-sim==0.6.0"
python -m pip check
```

## Release Notes

Release notes should include:

- user-visible matching or simulation changes
- CLI changes
- benchmark, corpus, manifest, or golden-state schema changes
- dashboard changes
- compatibility notes
- known limitations

For 0.6.0, begin with [`docs/releases/0.6.0.md`](releases/0.6.0.md). The
[0.5.0 notes](releases/0.5.0.md) document qualification as a public contract,
the [0.4.1 notes](releases/0.4.1.md) document the first upstream semantic
review, and the [0.4.0 notes](releases/0.4.0.md) document the failure-corpus
release.

Do not publish universal latency or throughput claims from local smoke benchmarks.
