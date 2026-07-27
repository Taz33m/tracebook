# tracebook-sim

`tracebook-sim` is the compatibility distribution for Tracebook's simulation,
benchmarking, replay, corpus, profiling, and visualization commands.

Beginning with version 0.6.0, this distribution contains no importable Python
packages. It depends on the exact matching `tracebook-conformance` release,
which owns the `tracebook` package and the `tracebook-conformance` command.
This facade owns only the established non-conformance console entry points:

- `tracebook-sim`
- `tracebook-benchmark`
- `tracebook-dashboard`
- `tracebook-web`
- `tracebook-replay`
- `tracebook-coinbase`
- `tracebook-corpus`

Install it in a fresh environment when you use the simulator or broader local
tooling:

If this environment contains `tracebook-sim` 0.5.x, do not run either 0.6.0
install yet. Remove the legacy package owner first:

```bash
python -m pip uninstall -y tracebook-sim
```

```bash
python -m pip install "tracebook-sim==0.6.0"
```

## Required Upgrade Step From 0.5.x

Do not directly upgrade an environment containing `tracebook-sim` 0.5.x. That
release owned the entire `tracebook` package; its uninstall record can remove
files newly installed by `tracebook-conformance` 0.6.0.

After removing the legacy owner as shown above, install the new facade:

```bash
python -m pip install "tracebook-sim==0.6.0"
```

If a direct upgrade already left `tracebook` unimportable, repair the new
source owner:

```bash
python -m pip install --force-reinstall --no-deps "tracebook-conformance==0.6.0"
python -m pip check
```

NumPy and psutil remain mandatory dependencies of this compatibility
distribution so an ordinary `tracebook-sim` installation preserves the
pre-0.6 simulator experience. Users who only need matching-engine conformance
can instead install the lightweight source-owning distribution:

```bash
python -m pip install "tracebook-conformance==0.6.0"
```

The existing runtime extras remain available:

- `tracebook-sim[analysis]` adds dataframe, plotting, and columnar analysis
  dependencies.
- `tracebook-sim[capture]` adds live WebSocket capture support.
- `tracebook-sim[dashboard]` adds the interactive dashboard dependencies.

Project documentation and source are maintained at
<https://github.com/Taz33m/tracebook>.
