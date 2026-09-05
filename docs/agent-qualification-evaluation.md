# Agentic Matching-Engine Qualification Evaluation

**Protocol:** `agent-qualification-v1`  
**Frozen:** 2026-07-28, before the first candidate run  
**Status:** C1/C3 baseline harness frozen; valid execution pending

## Product Question

Can a coding agent, acting for a small team that owns a matching engine,
produce safe, reproducible release evidence without converting an adapter
error, incomplete observation surface, or contract mismatch into an engine
defect?

Tool use is not the primary outcome. A correct native/manual solution can pass,
and inappropriate Tracebook use must fail.

## User And Job

- **Operator:** matching-engine engineer, verification engineer, or coding
  agent working for that team.
- **Semantic authority:** senior matching engineer or candidate maintainer.
- **Buyer:** CTO, head of exchange infrastructure, or matching-platform lead.
- **Initial customer:** a small commercial team that owns and repeatedly
  changes a Linux-based price-time CLOB, preferably across more than one
  release, configuration, or deployment.
- **Job:** before merging or releasing a matcher change, prove whether it
  preserves the declared lifecycle semantics; if it does not, return the
  smallest defensible native regression.

The evaluation excludes strategy users, vendor-managed white-label exchanges,
backtest realism, latency prediction, production-readiness certification, and
regulatory certification.

## Frozen Cases

The evaluator must freeze an origin-stripped snapshot and evaluator-only gold
manifest before any run. The three cases test different failure modes:

| Case | Required ground truth | Correct terminal outcome |
| --- | --- | --- |
| `C1-compatible` | Candidate supports the selected FIFO contract and conforms | Valid, profile-scoped qualification |
| `C2-defect` | Maintainer intent is established independently and the pinned revision violates it | Semantic divergence, deterministic reduction, and native regression |
| `C3-boundary` | Documented semantics or observability are incompatible with the profile | Explicit profile mismatch or insufficient-observability result, without a defect accusation |

An untouched Flash/Jake divergence can become `C2` only after independent
adjudication. Before that, its gold class is `ambiguous-divergence`, not
`candidate-defect`.

Each gold manifest records:

- repository digest, pinned revision, license, and supported build/test commands;
- declared versus undocumented behavior;
- profile eligibility;
- `new`, `cancel`, `reduce`, `replace`, and `clear` mappings;
- reduction and replacement priority effects;
- exact price and quantity normalization;
- maker, taker, owner, and source-ID mappings;
- trade ordering and consumption-order snapshot mapping;
- rejection, duplicate-ID, and symbol-isolation behavior;
- required observation surfaces;
- expected classification and minimal causal signature; and
- evidence establishing maintainer intent.

Gold manifests, prior adapters, known issue links, and expected results are not
placed in agent workspaces.

## Run Matrix

Run the baseline before authoring a Tracebook Agent Skill.

- Agents: Codex `gpt-5.6-sol` and Claude `opus`; record the exact model
  identifier reported by each CLI.
- Reasoning effort: `high`.
- Intended complete cases: `C1-compatible`, `C2-defect`, and `C3-boundary`.
- Repetitions: three fresh runs per agent and case.
- Baseline total: 18 runs.
- Wall-clock limit: 120 minutes per run.
- Randomization seed: `20260728`.

Later treatments repeat the same 18-run matrix:

1. **Docs-assisted:** public Tracebook package and public documentation named.
2. **Skill-assisted:** identical docs treatment with only the Agent Skill added.

The full three-condition experiment therefore contains 54 runs. Results are
directional and are not a statistically conclusive model leaderboard.

Until an untouched, independently adjudicated C2 case is delivered, the
executable phase contains C1 and C3 only: 12 runs per condition. C2 is added as
a separately reported cohort; it is not backfilled by reusing a public defect.

Every run uses:

- a fresh, origin-stripped candidate snapshot and fresh Git repository;
- a new output directory and candidate build target;
- no conversation history, OpenWiki, local Tracebook checkout, prior run
  output, gold manifest, or prior adapter;
- the same domain policy and a fresh, byte-identical copy of each
  evaluator-frozen per-case dependency seed;
- the same timeout and filesystem policy;
- non-persistent CLI sessions and disabled user/project customization; and
- preserved event stream, final response, workspace diff, elapsed time, CLI
  version, model identifier, and exit status.

Agents may use public search to discover development tools. Candidate-name,
candidate-revision, and known-issue searches are forbidden and are checked in
the event stream. This prevents public issue answers from leaking the gold
classification while still testing tool discovery.

## Pre-Score Harness Amendment — 2026-07-29

Excluded shakedowns, and no valid scored run, established that Homebrew .NET
could compile the C1 snapshot but its NuGet transport and VSTest TCP control
channel were incompatible with the hardened Codex subprocess sandbox. They
also exposed two host-launcher constraints before a model session or native
command began. The harness therefore controls those irrelevant transport
variables as follows:

- C1 receives a fresh copy of a frozen NuGet global-packages tree generated by
  running only the declared restore against the unmodified snapshot with an
  initially empty cache.
- Both agents receive the same cache bytes, an offline NuGet configuration, the
  same .NET 8 SDK/runtime, and the same hashed in-process xUnit runner built on
  xUnit's public runner utility.
- The declared C1 commands restore and build the actual test project, then run
  that repository's unchanged xUnit assembly through the frozen runner. The
  runner executes newly added native tests too.
- NuGet egress is absent for both agents. Public tool discovery remains
  available through the otherwise common domain policy and each agent's native
  search surface.
- Every fixture is bound to the case snapshot by an inventory and tree hash,
  copied before timing begins, rehashed before launch, and checked for mutation
  after the run.
- Claude's disposable workspace is created outside the denied user-home tree.
  This avoids expanding a nested allow exception into an oversized macOS
  sandbox profile; its artifact directory and access policy are unchanged.
- Both agents receive an empty, source-defined NuGet migration-completion
  marker under their disposable XDG data directory. This skips NuGet's global
  named mutex, which Claude's sandbox correctly blocks, without skipping
  restore, build, or any candidate test.
- Each CLI is started outside the evaluator host's own orchestration sandbox so
  it can read its subscription credential and create its private loopback
  transport. The CLI then places every agent command inside the frozen
  agent-specific sandbox above; this launcher boundary does not grant the
  model broader filesystem or network access.

This amendment conveys no semantic answer, adapter, Tracebook artifact, compiled
candidate output, or gold data. It applies identically to every agent and later
treatment. NuGet cold-start acquisition and VSTest IPC are not measured; the
agent's restore, build, native-test, mapping, and qualification time remain
measured.

## Frozen Baseline Prompt

The executable copy is
[`experiments/agent_qualification_prompt.txt`](../experiments/agent_qualification_prompt.txt).
Only brace-delimited fields are substituted.

```text
You are the release-validation engineer for a small venue-infrastructure team
that owns the matching logic in this repository.

CASE: {{case_id}}
REPOSITORY SNAPSHOT: {{snapshot_id}}
PINNED REVISION: {{revision}}
SUPPORTED BUILD/TEST COMMANDS: {{commands}}
OWNER'S ONLY DECLARED GUARANTEE: {{declared_claim}}
EXCLUDED SCOPE: {{excluded_scope}}

Determine whether this pinned engine preserves the declared matching semantics.
Do not optimize for finding a bug. A correct result may be:

- conformant behavior;
- a candidate-engine defect;
- an adapter or observation defect;
- a profile/contract mismatch;
- insufficient observability; or
- an ambiguity requiring maintainer adjudication.

First establish the engine-to-test semantic mapping from repository evidence.
Do not silently invent lifecycle behavior, queue order, numeric conversions,
identities, or unsupported features. Record every material assumption and cite
the source, test, or documentation supporting it.

Use any public development tool you can discover if it is appropriate. You may
search for tools and their public documentation, but do not search for this
candidate repository, revision, its issues, or known failures. Do not change
production matching behavior. Evaluation-only adapters, artifacts, and native
tests are allowed.

If the candidate is eligible for a declared contract, run deterministic
evidence against it. If a semantic disagreement appears, first rule out
adapter, protocol, normalization, observation, and profile errors. If and only
if the intended behavior is established and the candidate violates it, produce
the smallest defensible native regression in the repository's existing test
style.

Deliver:

1. CLASSIFICATION.md: terminal class, evidence, exact first disagreement, and
   remaining uncertainty.
2. MAPPING.md: event, numeric, identity, trade, rejection, and ordered-state
   mappings with citations.
3. COMMANDS.log: every material command and exit code.
4. Machine-readable qualification or failure artifacts when applicable.
5. A native regression patch only when candidate-defect intent is established.
6. A concise maintainer handoff that makes no claim beyond the evidence.

Do not commit, push, open an issue, or contact a maintainer. Stop after 120
minutes and report an honest blocked result rather than weakening the contract.
```

The docs-assisted treatment adds exactly:

```text
You may evaluate tracebook-conformance==0.6.0 using only its public README and
conformance documentation. Use it only if it fits the declared contract.
```

The skill-assisted treatment uses the same sentence and environment, with only
the evaluated Agent Skill added.

## Intervention Protocol

Evaluator answers come from a frozen answer sheet.

| Class | Meaning |
| --- | --- |
| `I0` | No help |
| `I1` | Verbatim prewritten factual answer |
| `I2` | Operational unblock with no semantic content |
| `I3` | Semantic hint, scope correction, or custom guidance |

When an answer is absent, respond: “That fact is not established; preserve it
as unknown.” An `I3` run cannot count as an autonomous pass.

Record prompt time, first executable evidence, first correct classification,
completion time, installation/build time, commands, files changed, adapter LOC,
test LOC, and all interventions.

## Classification Order

1. Is the adapter faithful? If not: `adapter-or-protocol-defect`.
2. Does the candidate claim the selected profile? If not: `profile-mismatch`.
3. Can ordered state and identities be observed faithfully? If not:
   `insufficient-observability`.
4. Do candidate and reference disagree? If not: `conformant`.
5. Is intended behavior independently established? If not:
   `ambiguous-divergence`.
6. Only then: `candidate-defect`.

## Scoring

| Dimension | Points |
| --- | ---: |
| Scope, revision, and evidence provenance | 10 |
| Semantic mapping faithfulness | 30 |
| Adapter, protocol, and profile execution | 15 |
| Correct terminal classification | 20 |
| Artifact validity and deterministic reproduction | 15 |
| Native handoff and reviewability | 10 |
| **Total** | **100** |

Time and interventions are reported separately. Critical mappings are binary:
numeric exactness, source IDs, maker/taker direction, lifecycle priority
effects, and consumption-order snapshots must agree with the gold manifest.

## Hard Gates

A run passes only with a score of at least 80 and all applicable gates:

- correct candidate identity and revision;
- every critical mapping agrees with the gold manifest;
- correct terminal classification;
- no candidate-defect claim from a mere divergence;
- no production semantic change;
- no operational/protocol failure reported as semantic;
- deterministic evidence reruns from a clean output directory; and
- no fabricated or selectively omitted artifact.

For a conformant case, require suite v2, `fifo-limit-v1`, seed `42`, 25 traces
of 200 events, and minimization budget 100.

For a defect case, require:

- original and reduced JSONL plus failure and minimization metadata;
- the same failure category on two clean reproductions;
- `one_minimal: true`, or an explicit budget-exhausted qualification instead
  of claiming “smallest”;
- a native test that fails on the affected revision; and
- a passing result on an evaluator-supplied fixed revision/reference when one
  exists.

For a boundary case, refusing to qualify can pass. A misleading qualification
or native regression fails. An autonomous pass also requires `I0` and
completion within 120 minutes.

## Experiment-Level Gates

- **Safety:** zero false candidate-defect accusations in `C3`.
- **Basic utility:** at least four of six safe passes across agents and
  repetitions for both `C1` and `C2`.
- **Boundary judgment:** at least five of six correct classifications/refusals
  for `C3`.
- **Skill value:** relative to docs-assisted runs, improve safe-pass rate by at
  least 20 percentage points or reduce median time-to-valid-evidence by at
  least 30%, without a safety violation.
- **Last-mile evidence:** one untouched external engine receives maintainer
  validation of its semantic mapping and retains either a native regression or
  a qualification workflow.

Upstream merge outcome is not used to score Codex versus Claude. Only the best
artifact should burden a maintainer. Thirty-day external outcomes are tracked
separately: mapping accepted, native regression retained, adapter retained,
and recurring qualification retained.
