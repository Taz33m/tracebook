# Agentic Qualification Generalization V2 Results

**Protocol:** `agent-qualification-generalization-v2`  
**Public protocol commit:** `50ac3b63c934ec9f3c18869cc5c68b221b73fdd2`  
**Execution completed:** 2026-08-02  
**Aggregate grading completed:** 2026-08-03
**Validity:** invalidated on 2026-08-25 after execution-harness review; retained
only as a historical audit record pending a corrected 24-run recollection

## Historical Result (Invalidated)

The figures below describe the original execution. They must not be used for
inference: the corrected harness restores fixture-mutation validation, Claude
provider-catalog auditing, and the restricted toolchain environment, among
other isolation and evidence-capture fixes.

All 24 preregistered runs received valid pre-score technical verdicts before
semantic outcomes were inspected. The frozen skill improved the autonomous
safe-pass rate by 50 percentage points over public documentation alone.

| Condition | Safe passes | Runs | Rate |
| --- | ---: | ---: | ---: |
| Public docs | 1 | 12 | 8.3% |
| Frozen skill | 7 | 12 | 58.3% |

The result exceeds the preregistered directional-product-signal threshold of
+20 percentage points. The skill condition produced zero unsupported
candidate-defect accusations, and no measured run changed tracked production
matching behavior.

## Required Breakdowns

### By agent family

| Agent | Docs | Skill | Delta |
| --- | ---: | ---: | ---: |
| Claude | 1/6 (16.7%) | 3/6 (50.0%) | +33.3 pp |
| Codex | 0/6 (0.0%) | 4/6 (66.7%) | +66.7 pp |

### By held-out case class

| Case | Docs | Skill | Delta |
| --- | ---: | ---: | ---: |
| Compatible engine | 0/6 (0.0%) | 1/6 (16.7%) | +16.7 pp |
| Profile boundary | 1/6 (16.7%) | 6/6 (100.0%) | +83.3 pp |

The intervention generalized strongly to boundary recognition: every skill
run correctly refused to manufacture Tracebook qualification evidence for the
system that did not select identified resting venue makers. Compatible-engine
release evidence remained the bottleneck. Only one of six skill runs satisfied
every clean-build, candidate-identity, canonical-bundle, and byte-equality
gate.

## Interpretation

This is a positive directional product result, not a statistically conclusive
model comparison. It supports three narrow claims:

1. an explicit qualification skill materially improved safe autonomous
   workflow execution in this held-out cohort;
2. the improvement was driven mainly by profile-boundary judgment; and
3. compatible-engine release evidence still required machine-enforced gates
   rather than more prose.

It does not establish broad model superiority, production adoption, product-
market fit, or reliable autonomous qualification across matching engines. The
adaptive v2 cohort must not be pooled with v1 as one untouched confirmatory
study, revised after grading, or rerun to improve the result.

## Product Consequence

The highest-leverage follow-up is a public evidence workflow that:

- pins candidate name, revision, and source snapshot before adapter readiness;
- creates two separate candidate, adapter, build, cache, and output roots;
- requires two canonical qualification bundles with matching deterministic
  identifiers, counts, coverage, and bytes; and
- emits one concise, grading-ready manifest.

That follow-up is implemented separately in
[PR #71](https://github.com/Taz33m/tracebook/pull/71). It is not part of the
frozen intervention and must be evaluated on another untouched candidate.

## Audit Trail

Evaluator-only case identities, gold, authorizations, provider transcripts,
workspaces, and per-run grading remain under the ignored private experiment
tree. They are intentionally not published with the protocol. The aggregate
values above were transcribed from the completed private scorecard whose
SHA-256 is:

```text
7741a127949b0bf8881b8827e1ec46ee12082a40c84f7bc8ba6b25ba409269b7
```

The private narrative report used for the transcription has SHA-256:

```text
cfbf53af662407bbe8df74970cbaf107573d74d3d3ce8881976f3103ca2de78d
```

These hashes bind the public aggregate report to the preserved evaluator
record without publishing restricted experiment materials.
