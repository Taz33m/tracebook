# Field Note 001: From Matching-Engine Divergence to Four Events

**Published:** 2026-07-19<br>
**Software:** Tracebook 0.5.0<br>
**Scope:** FIFO matching-engine conformance and failure forensics

## Finding

A broad correctness harness and a failure-forensics tool solve different parts
of the same maintenance problem. A harness can establish that two matching
engines disagree under a large workload. A maintainer still needs to know the
first semantic difference, the lifecycle rule it implicates, the smallest trace
that preserves it, and whether that trace can become a deterministic regression.

Tracebook tested this boundary on a historical `orderbook-rs` FIFO defect first
reported by
[`matching-engine-benchmark`](https://github.com/flash1-dev/matching-engine-benchmark).
Flash remains the discovery source. Tracebook imported its versioned divergence
artifact, selected the exact workload prefix, classified the disagreement, and
reduced 15,739 messages to four events in 193 fresh candidate runs.

An independent Tracebook campaign reached the same causal shape at event 173
and also reduced it to four events. Both traces prove that a partially filled
maker must retain priority over a later order at the same price.

## The Four-Event Proof

1. Maker A rests at a price.
2. Maker B rests later at the same price.
3. A taker partially fills maker A, leaving a remainder.
4. A second taker arrives. FIFO requires maker A's remainder to trade before
   maker B; the affected engine selects maker B.

Aggregate depth can remain correct throughout this sequence. The observable
failure is the maker identity on the fourth event, which is why best-price and
depth-only checks are insufficient for queue-priority conformance.

The stored regressions are:

- [`flash-issue-88-reduced.jsonl`](../../integrations/orderbook_rs/regressions/flash-issue-88-reduced.jsonl),
  preserving values from Flash's canonical workload;
- [`issue-88-reduced.jsonl`](../../integrations/orderbook_rs/regressions/issue-88-reduced.jsonl),
  preserving the normalized campaign failure.

## Reproduction Contract

The historical candidate is pinned to `orderbook-rs 0.8.0` commit
`53b4d2b0a657f4260e316d3a8ac3f0df0fc068bf` and `pricelevel 0.7.0`.
The maintained comparison uses `orderbook-rs 0.12.0` and `pricelevel 0.9.1`.
The reduced trace must diverge against the historical dependency graph and pass
against the maintained release.

```bash
tracebook-conformance reproduce \
  .tracebook/corpus/failure-7dd023c684cdb2d0fc0e/reduced.jsonl \
  --candidate-cmd ./orderbook-rs-issue-88-adapter

tracebook-conformance run \
  integrations/orderbook_rs/regressions/issue-88-reduced.jsonl \
  --candidate-cmd ./tracebook-orderbook-rs
```

The full provenance, build commands, hashes, and exact divergence path are in
the [case study](../case-studies/orderbook-rs-issue-88.md).

## External Review Changed the Contract

The maintained `orderbook-rs` author later reviewed Tracebook's adapter
assumptions. That review confirmed priority-preserving quantity reduction,
priority-losing replacement, and faithful maker/taker identifiers. It also
surfaced a historical difference between snapshot order and consumption order
after in-place quantity increases. `orderbook-rs 0.12.0` removed that caveat by
materializing snapshots in queue-consumption order.

This matters because a passing adapter is not enough. The candidate author must
agree that the adapter observes the same queue the engine will consume. Profile
questions are therefore recorded as evidence, not treated as integration noise.

## Qualification Evidence

The same `fifo-limit-v1` qualification contract has been exercised against two
repository-local Rust adapters and one public-package Go integration proposal.
Each qualification combines three profile-relevant fixed cases, 25 generated
traces of 200 events, 10 semantic capabilities, JSON, and JUnit output.

The Go integration against `geseq/orderbook` passed all 5,000 generated events
from the public Tracebook 0.5.0 command. Its upstream
[draft PR #30](https://github.com/geseq/orderbook/pull/30) remains a proposal
until the maintainer reviews it and chooses whether to retain the CI job.
Repository-local success is not labeled adoption.

## What Did Not Ship

A controlled experiment compared deterministic generation with guided suffix
regeneration under equal candidate-run budgets. Guidance improved one injected
defect but took longer on the historical defect and a second injected defect.
Tracebook therefore did not add a guided campaign mode. The negative result is
preserved in the
[qualification design-partner study](../qualification-design-partners.md).

## Product Consequence

The useful category is not another general matching engine or another broad
leaderboard. It is the reproducible path from a discovered disagreement to a
small semantic proof that an engine author can review and keep in CI.

The next gate is external retention: one candidate repository must accept a
qualification or reduced regression into its own CI, and a second independent
onboarding must determine whether adapter protocol work repeats across engines.
Only repeated friction should become a new SDK or protocol surface.

## Claim Boundary

- The historical defect was discovered by Flash, not Tracebook.
- A passing qualification applies only to the named profile, candidate revision,
  and recorded workload.
- The reference engine is inspectable research infrastructure, not a production
  exchange.
- Process-level conformance timing is not matching-engine latency.
