# Tracebook Project Plan

## Product Position

`tracebook` is the conformance-testing and reproducible failure-analysis toolkit
for matching engines. It runs canonical order-event traces against its
inspectable Python reference engine and external engines, localizes semantic
drift, reduces failures, and keeps correctness and performance artifacts
reproducible. It is distributed as `tracebook-sim` and imported as `tracebook`.

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

## Current Product Hypothesis

The immediate bottleneck is adoption, not another matching algorithm. An engine
author should be able to move from an adapter command to a trustworthy,
profile-scoped compatibility artifact in one invocation. Qualification version
1 combines relevant immutable suite cases, a deterministic generated campaign,
semantic coverage, JSON, JUnit, and any minimized failure without testing
features outside the profile the engine claims.

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

1. Run a time-boxed adoption sprint with independently maintained engines.
   Complete the `gocronx/matcher` review loop and recruit at least two additional
   authors or contributors. Record successful and blocked attempts through the
   public engine-qualification report rather than counting repository-local
   integrations as adoption.
2. Observe at least two external onboarding attempts before building an adapter
   scaffold. If adapter mechanics repeatedly dominate the work, extract only
   the repeated protocol server, canonical-state, fixture, and CI pieces while
   leaving engine-specific semantics explicit.
3. Keep guided exploration research-only. The first frozen held-out comparison
   improved one injected defect and regressed on the historical defect plus a
   second injected defect, so it failed the product gate.
4. Revisit protocol v2, additional semantic profiles, and candidate benchmarking
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

The north-star measure is the number of externally maintained engine CI jobs
that run Tracebook and retain its evidence. Downloads, clones, stars, local
adapters, and passing campaign counts are awareness or technical signals, not
adoption by themselves.

## Decision Rules

- Matching behavior changes require executable semantic and invariant tests.
- Conformance protocol changes require cross-process tests, artifact schema
  tests, a version decision, and migration notes.
- Exported schema changes require artifact tests and a changelog entry.
- Performance claims require a command, seed, environment, and JSON artifact.
- New dependencies need a measurable benefit and must remain optional unless
  they are required by the core order-processing path.
