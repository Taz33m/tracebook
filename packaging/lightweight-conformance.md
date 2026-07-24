# Lightweight conformance package boundary

Decision date: 2026-07-24.

## Decision

Do not claim a lightweight public installation mode until the distribution
ownership split described below ships. The existing `tracebook-sim`
distribution continues to require NumPy and psutil so a fresh installation
continues to support its simulation, benchmark, and profiling commands.

The release gate runs `tools/smoke_conformance_wheel.py` in an isolated virtual
environment. It deliberately installs the wheel without dependencies and proves
that qualification itself completes 3/3 fixed cases, 25/25 generated traces,
5,000 events, and 10/10 semantic capabilities with neither NumPy nor psutil
present. This is architectural evidence only; `--no-deps` is not a supported
end-user installation mode.

## Why an extra is not the answer

Python package extras add dependencies to a distribution's required
dependencies; they cannot remove NumPy and psutil from `tracebook-sim`.
Consequently, an empty `conformance` extra would be misleading. Moving the two
dependencies into a conventional `simulation` extra would make a fresh
`pip install tracebook-sim` leave the existing simulator commands unusable.

[PEP 771](https://peps.python.org/pep-0771/) proposes default extras and an
explicit minimal-install syntax, but it remained a draft on the decision date.
Depending on it would not provide a broadly supported pip contract.

## Rejected interim distribution

Do not publish a second distribution that installs the existing `tracebook`
package tree or the existing `tracebook-conformance` console script alongside
`tracebook-sim`. The two distributions would own the same installed files.
Install order could silently replace files, and uninstalling either distribution
could damage the other. Vendoring a second renamed copy would avoid file
ownership overlap but would create two independently importable copies of the
conformance implementation and its process-global types.

## Target release architecture

Use a coordinated two-distribution release:

1. `tracebook-conformance` owns the `tracebook` Python package, bundled
   conformance fixtures, typing marker, and `tracebook-conformance` console
   script. Its mandatory runtime dependency list excludes NumPy and psutil.
2. `tracebook-sim` remains the compatibility name for simulator users. It
   depends on the exact matching `tracebook-conformance` release plus NumPy and
   psutil, and owns the simulation, benchmark, profiling, visualization, replay,
   and corpus console scripts.
3. The conformance distribution is published first. The matching simulator
   distribution is published only after the conformance artifact is available.
4. Both artifacts come from the same source revision and version. Existing
   `tracebook.*` import paths remain unchanged; source files are not copied or
   installed by both distributions.

This migration is ready to ship only when CI proves all of the following:

- installing only `tracebook-conformance` resolves no NumPy or psutil package;
- the full qualification smoke passes from the installed conformance wheel;
- installing `tracebook-sim` in an empty environment still makes every
  previously published simulator command usable;
- installing and uninstalling the two distributions in either order leaves no
  shared-file damage;
- release automation builds, validates, and publishes both distributions in the
  required order; and
- sdist and wheel metadata agree on dependency and file ownership.
