# Tracebook Project Plan

## Product Position

`tracebook` is the conformance-testing and reproducible failure-analysis toolkit
for matching engines. It runs canonical order-event traces against its
inspectable Python reference engine and external engines, localizes semantic
drift, reduces failures, and keeps correctness and performance artifacts
reproducible. Beginning with 0.6.0, the dependency-light `tracebook-conformance`
distribution owns the `tracebook` import package and qualification command.
`tracebook-sim` is a package-less compatibility facade that adds the simulator
commands and their NumPy/psutil dependencies at the exact matching version.

The project optimizes for deterministic behavior and auditability before raw
speed. Its optional public capture is research tooling, not a production feed
handler or trading infrastructure.

## Maintained Surfaces

```text
src/tracebook/
  core/             matching, lifecycle, snapshots, deterministic replay
  conformance/      adapters, campaigns, semantic diffing, minimization, standard suite
  book_replay/      separate L3 delta/probe contract; never venue-order matching
  events/           normalized event replay and offline venue adapters
  corpus/           safe local capture, manifests, golden state, import benchmarks
  simulation/       synthetic order flow and paced workload execution
  benchmarks/       reproducible local scenario reports
  profiling/        metrics, magic-trace, and selected-function fallback tracing
  visualization/    Dash dashboard and dependency-free live web frontend
integrations/
  python_matching_engine/  pinned external adapter and compatibility trace
  orderbook_rs/             native Rust adapter, faulty example, regression proof
  gocronx_matcher/          pinned Rust adapter and profile qualification
  intrepid_orderbook/      pinned Go CLOB adapter and explicit FOK divergence
  nautilus_trader/         pinned L3 book replay, not CLOB qualification
```

## External Validation

The first external-validation milestone was achieved on 2026-07-14. The
`orderbook-rs` maintainer reviewed and confirmed Tracebook's adapter semantics
in [issue #203](https://github.com/joaquinbejar/OrderBook-rs/issues/203), then
merged public priority documentation and property tests in
[PR #204](https://github.com/joaquinbejar/OrderBook-rs/pull/204). The review
also exposed an upsize snapshot-round-trip discrepancy tracked in
[`orderbook-rs` #205](https://github.com/joaquinbejar/OrderBook-rs/issues/205)
and repaired at the lower `PriceLevel` layer in
[PR #110](https://github.com/joaquinbejar/PriceLevel/pull/110). Tracebook did
not automatically generate that discrepancy; it surfaced through independent
review of the profile boundary.

The second external-validation milestone is also complete. Flash's canonical
divergence export let Tracebook reduce a real historical `orderbook-rs`
priority defect from 15,739 workload messages to a four-event regression. Flash
is the discovery source; Tracebook provides localization, reduction, replay,
and CI evidence.

A second profile qualification now passes against pinned `gocronx/matcher`
source. Tracebook CI preserves the JSON and JUnit bundle, while maintainer review
of the snapshot observation surface, replacement representation, and possible
upstream CI adoption remains pending. That distinction is part of the evidence,
not a reason to label the upstream contract settled.

The first maintainer-directed public-package qualification was exercised in
[`geseq/orderbook` draft PR #30](https://github.com/geseq/orderbook/pull/30).
The maintainer selected a native amendment API after the profile review exposed
that cancel plus create could not preserve reduction priority. From a fresh
clone, the 0.5.0 public command passed 3/3 fixed cases, 25/25 generated traces,
5,000 events, and 10/10 capabilities. The bundled draft was later closed in
favor of the reviewable, core-only
[`geseq/orderbook` PR #31](https://github.com/geseq/orderbook/pull/31), which
merged on 2026-07-25. The 865-line adapter and optional CI workflow remain out
of tree. The core semantic change is therefore retained upstream, but the
maintainer has not adopted Tracebook or its qualification workflow; this is
evidence of a serious onboarding attempt rather than independent adoption.

The first native-regression retention pilot reached three additional engines on
2026-07-20. StockSharp merged a focused FIFO-priority repair and regression in
[PR #684](https://github.com/StockSharp/StockSharp/pull/684), and
`inv2004/orderbook-rs` approved and merged its last-order panic repair and
regression in [PR #3](https://github.com/inv2004/orderbook-rs/pull/3). The
equivalent OPEX lifecycle repair remains open in
[PR #690](https://github.com/opexdev/core/pull/690). The public
[experiment ledger](https://github.com/Taz33m/tracebook/issues/66) records the
2/3 retention result and its key limitation: Flash had already reduced all
three source reports, so this validates maintainer demand for small native
regressions, not Tracebook's discovery or minimization value.

The proposed untouched Flash handoff did not yield a raw divergence. Every
eligible duplicate repository tested against the oracle reproduced the
consensus, while the remaining long-tail candidates were incomplete or lacked
the cancel/modify and price-time semantics required by the workload. The raw
case remains a useful opportunity when a genuine divergence appears, but it is
not a dependable near-term milestone or a reason to delay independent
onboarding work.

## Current Product Hypothesis

The corrected, held-out [24-run agent study](docs/agent-qualification-generalization-v2-results.md)
completed on 2026-08-29. The frozen skill raised safe passes from 1/12 to 5/12,
entirely through better profile-boundary decisions; compatible-engine evidence
passed in neither arm (0/6 each). The now-implemented captured two-run workflow
addresses candidate identity and artifact freshness mechanically. Its effect on
independent onboarding remains unmeasured; the study did not test that workflow.

The product question is whether a developer can obtain a trustworthy answer
from the installed package. The historical FIFO proof exercises the complete
path from workload to first difference, reduced trace, and passing control.

## Next Milestones

1. Ship independently verified qualification evidence and replay integrity fixes in the coordinated 0.7.3 distributions.
2. Keep the four-event proof runnable from an empty directory with one command.
   Measure the installed path on each candidate release.
3. Track first divergent event, reduced size, minimizer runs and budget,
   one-minimality, and fixed-engine controls in the Rust integration scorecard.
   Keep historical defects and injected controls labeled separately.
4. Choose the next engineering change from an observed missed defect, poor
   reduction or repeatable adapter problem. Attach a failing example and a
   measurable acceptance criterion before expanding profiles or protocols.

## Next Release Gate

- The exact release candidate passes the supported Python matrix, native
  integration checks, distribution ownership, privacy, migration, and
  wheel/source agreement checks in the [release checklist](docs/release.md).
- A clean install of the built wheel runs `tracebook-conformance fifo-proof`
  from an empty directory. It must regenerate the four-event Flash trace from
  15,739 messages, reproduce the maker mismatch on the affected revision, and
  pass the fixed engine. The injected control has its own row in the scorecard.
- CI verifies two byte-identical FIFO qualification bundles for each pinned Go
  release and retains the manifests.
- After publication, verify clean installs of both public packages, the exact
  facade dependency, and the FIFO proof from the public wheel.

External runs, retained native tests, upstream CI, documentation links, and
first-time adapter-author timings remain adoption measurements. Record them
when observed. Release scheduling and feature selection can proceed using the
repository-owned checks above. The [Go onboarding record](docs/independent-onboarding.md)
preserves the maintainer's review and decision about local execution.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Conformance protocol changes require cross-process tests, artifact schema
  tests, a version decision, and migration notes.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
