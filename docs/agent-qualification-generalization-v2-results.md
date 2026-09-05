# Agentic Qualification Generalization V2 Results

**Protocol:** `agent-qualification-generalization-v2`  
**Public protocol commit:** `50ac3b63c934ec9f3c18869cc5c68b221b73fdd2`  
**Corrected execution completed:** 2026-08-29
**Blinded aggregate grading completed:** 2026-08-29
**Validity:** corrected 24-run recollection complete; this result supersedes
the invalidated 2026-08-02 execution

The preregistered protocol file remains byte-frozen; its pending-recollection
status line is therefore a historical execution note, not the current result.

## Corrected Result

All 24 preregistered runs received valid pre-score technical verdicts before
semantic outcomes were inspected. Private gold stayed local. Grading used
opaque evidence IDs, and condition/provider labels were revealed only after the
24 verdicts were sealed.

The frozen skill improved the autonomous safe-pass rate by **33.3 percentage
points** over public documentation alone.

| Condition | Safe passes | Runs | Rate |
| --- | ---: | ---: | ---: |
| Public docs | 1 | 12 | 8.3% |
| Frozen skill | 5 | 12 | 41.7% |

The result exceeds the preregistered directional-product-signal threshold of
+20 percentage points. The skill condition produced zero unsupported
candidate-defect accusations, and no measured run changed tracked production
matching behavior.

## Required Breakdowns

### By agent family

| Agent | Docs | Skill | Delta |
| --- | ---: | ---: | ---: |
| Claude | 1/6 (16.7%) | 2/6 (33.3%) | +16.7 pp |
| Codex | 0/6 (0.0%) | 3/6 (50.0%) | +50.0 pp |

### By held-out case class

| Case | Docs | Skill | Delta |
| --- | ---: | ---: | ---: |
| Compatible engine | 0/6 (0.0%) | 0/6 (0.0%) | +0.0 pp |
| Profile boundary | 1/6 (16.7%) | 5/6 (83.3%) | +66.7 pp |

The intervention generalized to boundary recognition: five of six skill runs
correctly refused to manufacture Tracebook qualification evidence for a system
that does not select identified resting venue makers. One additional skill run
selected the right boundary class but failed the mechanically fresh native-pair
gate.

Compatible-engine release evidence remained the bottleneck. Several runs
reached the expected behavioral counts twice, but none preserved a pair whose
candidate metadata and qualification identity matched the frozen canonical
gold. Other runs omitted the canonical pair or selected the wrong class.

## Matched Safe-Pass Timing

Only `claude__c5-boundary__r1` safely passed in both conditions. The docs run
took 685.14 seconds and the skill run took 655.93 seconds: 29.21 seconds, or
4.3%, faster. No other timing comparison is reported.

## Interpretation

This is a positive directional product result, not a statistically conclusive
model comparison. It supports three narrow claims:

1. the frozen skill improved safe autonomous execution in this corrected
   held-out cohort;
2. the gain came entirely from profile-boundary judgment; and
3. compatible-engine qualification still requires machine-enforced identity
   and artifact gates.

It does not establish broad model superiority, production adoption,
product-market fit, or reliable autonomous qualification across matching
engines. The adaptive v2 cohort must not be pooled with v1 or the invalidated
execution as one untouched confirmatory study.

## Product Consequence

The public [captured two-run evidence workflow](conformance.md#captured-two-run-evidence)
now enforces canonical candidate name, revision, and snapshot binding, then
requires two fresh byte-matching qualification bundles with fixed campaign and
qualification identities. It was implemented separately from the frozen study;
these results do not measure that workflow's effect.

The next gate is independent use: test the consolidated workflow with an
external maintainer and record time to first verified pair, human corrections,
and whether the maintainer retains its output or CI workflow.

## Corrected Harness Binding

The measured recollection used:

- execution plan SHA-256:
  `1d3753bf1ea7e46b261e56ebb36fdf726c56192cb19d50c2dd00b6213ca68d1f`;
- execution freeze SHA-256:
  `82b037d86723fc8003424a33c8b84030224746495b9736c664347aafb1f81416`;
- execution runner SHA-256:
  `728937c53c26ade2703014bd44bd440e82bd59e6dfebac65977a2a765dfd209f`; and
- frozen skill SHA-256:
  `0aec0785111a9e4b67d2323684d6372ccaba03b84ddbecd850d4b658df1cdee2`.

## Audit Trail

Evaluator-only identities, gold, authorizations, transcripts, workspaces,
blinded dossiers, and per-run grading remain under the ignored private
experiment tree. They are intentionally not published with the protocol.

The completed private scorecard has SHA-256:

```text
789f8cae51ea57f541a938770e5cc12c5099690c02c736260b2300d0062a8f1a
```

The private semantic report has SHA-256:

```text
33b769d1d093609786dc1ab5a12d062c18d86eed462a2035e750721ecba1ca02
```

These hashes bind the public aggregate report to the preserved evaluator
record without publishing restricted experiment materials.
