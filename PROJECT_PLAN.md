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

The immediate bottleneck is adoption, not another matching algorithm. An engine
author should be able to move from an adapter command to a trustworthy,
profile-scoped compatibility artifact in one invocation. Qualification version
1 combines relevant immutable suite cases, a deterministic generated campaign,
semantic coverage, JSON, JUnit, and any minimized failure without testing
features outside the profile the engine claims.

The retention pilot sharpens the adoption path. Tracebook should earn upstream
ownership by first delivering a reviewable semantic result: a localized
divergence, a minimized trace, or a focused native regression. Tracebook can
carry the adapter and scheduled qualification out of tree until the engine
maintainer independently chooses to retain the tool or workflow. A passing
artifact produced only by this repository remains technical evidence, not
adoption.

The 0.6.0 distribution split removes installation weight from that path without
broadening it. Tracebook discovers, localizes, and reduces; maintainers review
small native regressions. Adapter or CI ownership is offered only after the
maintainer has seen concrete semantic value.

## How The Roadmap Is Chosen

Roadmap decisions use an evidence ladder:

1. **Maintainer behavior:** Did an external author run the tool, understand the
   result, accept the semantic boundary, and keep the regression in CI?
2. **Workflow friction:** Measure time to first qualification, adapter size,
   failed command attempts, protocol questions, and manual artifact edits.
3. **Discovery yield:** Under an equal candidate-run budget, compare unique
   semantic transitions, time to first divergence, and minimized reproducer
   quality on held-out real and injected defects.
4. **Repeatability:** Prefer results independently reproduced from the public
   package over repository-local demos, stars, benchmark volume, or test count.

A feature moves into the maintained product only when it improves one of those
measures for an external engine. Paper-derived techniques begin as controlled
experiments rather than presumed roadmap wins.

## Next Milestones

1. Complete one independent public-package onboarding. Start from a concrete,
   tested semantic artifact, ask one narrow ownership question, and record
   whether an external maintainer runs the tool or retains its output. Do not
   count Tracebook-owned scheduled integrations as adoption.
2. Keep the raw Flash-to-Tracebook forensic handoff opportunistic. If an
   untriaged divergence appears, measure localization, minimization, translation
   into a native regression, and maintainer retention honestly; do not keep
   searching incompatible or unfinished repositories solely to manufacture the
   gate.
3. Await an OPEX maintainer decision on the focused lifecycle repair after one
   concise status follow-up. Do not add repeated hypothetical integration
   outreach to an inactive review. Any new outreach should begin with a tested
   native artifact and one narrow semantic question.
4. Observe at least two external onboarding attempts before building an adapter
   scaffold. The 865-line Go adapter is the first concrete signal that protocol
   framing, canonical hashing, state translation, and decimal handling may
   dominate the work. Extract only pieces repeated by a second author while
   leaving engine-specific semantics explicit.
5. Keep guided exploration research-only. The first frozen held-out comparison
   improved one injected defect and regressed on the historical defect plus a
   second injected defect, so it failed the product gate.
6. Revisit protocol v2, additional semantic profiles, and candidate benchmarking
   only after qualification evidence identifies a repeated external need.

## Next Release Gate

The next feature release is gated by external use, not a calendar date. Before
cutting it, require:

- two independently maintained engines qualified with the public package;
- one candidate repository retaining a qualification or reduced regression in
  its own CI;
- an observed time to first qualification under 30 minutes for a new adapter
  author; and
- evidence that any new adapter helper removes repeated friction rather than
  merely reducing Tracebook's own integration code.

The north-star measure is the number of independently maintained candidate
repositories that retain Tracebook-derived evidence. A recurring CI job is the
strongest form; a native regression counts only when a raw failure actually
passed through Tracebook localization or minimization. Downloads, clones, stars,
local adapters, passing campaigns, and regressions translated from already
reduced third-party reports are awareness or technical signals, not adoption by
themselves.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Conformance protocol changes require cross-process tests, artifact schema
  tests, a version decision, and migration notes.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
