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
python -m black src tests
python -m flake8 src tests
python -m pytest
python test_system.py
tracebook-sim --duration 1 --throughput 50 --algorithm FIFO --seed 1337 --warmup-seconds 0.01
tracebook-benchmark --scenario smoke --duration 1 --throughput 50 --seed 1337 --warmup-seconds 0.01 --output benchmark_results/release-smoke.json
tracebook-dashboard --demo-simulation --help
python -m build --sdist --wheel --outdir dist
python -m pip check
```

## Remote Verification

- Push to a branch and wait for GitHub Actions on Python 3.10 and 3.11.
- Check generated package metadata in the build artifact.
- Check that Dependabot has no urgent security update waiting.
- Confirm no README badge or proof table overstates the current CI/test state.

## Release Notes

Release notes should include:

- user-visible matching or simulation changes
- CLI changes
- benchmark report schema changes
- dashboard changes
- compatibility notes
- known limitations

Do not publish universal latency or throughput claims from local smoke benchmarks.
