# Lightweight conformance package boundary

Decision date: 2026-07-24.

## Decision

Version 0.6.0 uses two coordinated distributions with no overlapping installed
files:

1. `tracebook-conformance` owns the `tracebook` Python package, bundled
   fixtures, `py.typed`, and the `tracebook-conformance` console script. It has
   no mandatory runtime dependencies, so a normal installation does not resolve
   NumPy or psutil.
2. `tracebook-sim` is a package-less compatibility facade. It depends on the
   exact matching `tracebook-conformance` version plus NumPy and psutil, and
   owns the seven simulation, benchmark, profiling, visualization, replay, and
   corpus console scripts.

The implementation remains one source tree and existing `tracebook.*` imports
do not change. Only one distribution owns that tree. The facade does not copy,
vendor, or install any Python packages.

## Install contracts

Uninstall the legacy owner before installing either 0.6.0 distribution when an
environment contains the monolithic `tracebook-sim` 0.5.x wheel:

```bash
python -m pip uninstall -y tracebook-sim
```

For matching-engine qualification:

```bash
python -m pip install "tracebook-conformance==0.6.0"
```

For the full simulator and workbench command surface:

```bash
python -m pip install "tracebook-sim==0.6.0"
```

The simulator dependency is pinned exactly because its console scripts invoke
modules owned by the conformance distribution from the same source revision.
Publishing either half at a different version would create an untested
combination.

## One-time migration from 0.5.x

`tracebook-sim` 0.5.x owned the entire `tracebook` package and every console
script. During a direct in-place upgrade, pip may install the new conformance
dependency and then uninstall 0.5.x. The old wheel's uninstall record can
delete files that the newly installed conformance wheel now owns. Distribution
metadata may then say 0.6.0 is installed even though the shared import package
has been removed.

After uninstalling the legacy owner, install the required 0.6.0 surface:

```bash
python -m pip install "tracebook-sim==0.6.0"
```

For a conformance-only migration, replace the second command with:

```bash
python -m pip install "tracebook-conformance==0.6.0"
```

If a direct upgrade has already left the environment damaged, repair the
package owner without resolving the simulator stack again:

```bash
python -m pip install --force-reinstall --no-deps "tracebook-conformance==0.6.0"
python -m pip check
```

This is a one-release ownership handoff, not a continuing install-order
requirement. Fresh 0.6.0 environments have disjoint wheel records.

## Why an extra cannot provide the boundary

Python package extras add dependencies; they cannot remove mandatory
dependencies. Keeping NumPy and psutil mandatory would leave qualification
heavy. Moving them into a conventional optional extra would make an ordinary
`pip install tracebook-sim` expose simulator commands whose imports fail.

[PEP 771](https://peps.python.org/pep-0771/) proposes default extras and an
explicit minimal-install syntax, but it remained a draft on the decision date.
The public contract therefore uses distribution ownership that works with
released pip versions.

## Release and verification invariants

The two artifacts are built from the same source revision and version.
`tracebook-conformance` must be published first, and `tracebook-sim` only after
that exact dependency is available. CI and the release workflow prove:

- the conformance wheel contains the `tracebook` package and only the
  `tracebook-conformance` entry point;
- the simulator wheel contains no Python package and only the seven
  compatibility entry points;
- the two wheel RECORDs have no overlapping paths;
- a normal conformance-only install contains neither NumPy nor psutil and
  completes 3/3 fixed cases, 25/25 generated traces, 5,000 events, and 10/10
  semantic capabilities;
- a fresh simulator install makes every previously published simulator command
  available, resolves NumPy and psutil normally, passes `pip check`, and loads
  all seven command modules;
- removing the simulator facade leaves conformance functional;
- the documented uninstall-first 0.5.x migration leaves one healthy package
  owner, with the final release gate using the SHA-256-pinned public 0.5.0
  wheel rather than only a synthetic ownership model;
- each sdist safely rebuilds a wheel whose complete logical contents agree with
  the directly built wheel; and
- commit-derived timestamps plus pinned build backends make release wheels
  byte-for-byte reproducible across a partial-publication retry.

`--no-deps` remains useful only in the explicit recovery command above and in
artifact-level verification where dependencies are installed separately. It is
not the supported way to obtain lightweight conformance.
