# Agentic Qualification Generalization Evaluation

**Protocol:** `agent-qualification-generalization-v2`  
**Correction status:** the 2026-08-02 execution is retained as historical
evidence but is not valid for inference. A corrected harness must collect a
fresh 24-run matrix before the aggregate claim is reinstated.

This document preserves the preregistered design. The aggregate outcome and
claim boundaries are reported separately in the
[v2 results note](agent-qualification-generalization-v2-results.md).

## Question

Does the revised Tracebook qualification skill improve safe autonomous release
evidence on previously unseen matching-engine repositories, rather than merely
encoding the two repositories used in the first experiment?

The intervention must help an agent:

- recognize a real matching engine even when lifecycle support requires public
  host composition;
- preserve source-ID, native-ID, owner, clear, and replacement semantics;
- produce deterministic qualification from genuinely fresh build and output
  state; and
- stop safely when a repository exposes replay, OMS, and simulated execution
  components but does not choose venue makers.

This remains a directional product experiment, not a statistically conclusive
model comparison.

## Relationship To V1

The completed v1 experiment is immutable and is not rerun. Its C1 and C3
repositories, prompts, workspaces, outputs, and answers are excluded from this
cohort.

V1 found a 25 percentage-point skill-assisted safe-pass improvement over public
documentation alone. The three safe passes were all one agent family on the
boundary case. Compatible-case attempts still missed identity lifecycle or
clean reproduction gates. The v2 intervention was therefore written before
any v2 provider run and adds only candidate-agnostic controls:

- a source/native/owner identity transition table;
- explicit duplicate-active and clear-reset checkpoints;
- task-pinned candidate metadata;
- a pre-qualification gate; and
- mechanically fresh build, process, cache-copy, and output requirements for
  conformant, divergent, and boundary outcomes.

## Held-Out Cases

The evaluator freezes two origin-stripped public repositories at pinned
revisions. Repository identities and evaluator answers remain outside agent
workspaces.

| Case | Required ground truth | Correct terminal outcome |
| --- | --- | --- |
| `C4-compatible` | Public primitives compose faithfully into `fifo-limit-v1`, and a frozen evaluator adapter passes canonical qualification | Two clean valid profile qualifications with matching deterministic identifiers |
| `C5-boundary` | The system replays externally supplied L2 state and simulates strategy fills but does not select resting venue makers | `profile-mismatch`, with two clean native-suite passes and no Tracebook defect artifact |

Each frozen case binds:

- origin repository, license, pinned revision, and origin-stripped tree hash;
- owner claim and excluded scope;
- supported native commands;
- evaluator-only gold;
- a frozen dependency seed and manifest-declared destination (vendored sources
  for Cargo, with writable Cargo state kept separately);
  and
- applicable evaluator adapter and machine evidence.

Agents may inspect only their fresh candidate copy, the frozen prompt, public
Tracebook documentation/package for both conditions, and the v2 skill for the
skill condition.

## Conditions And Matrix

This is a two-condition generalization cohort:

1. `docs`: the public Tracebook package and public conformance documentation;
2. `skill`: the identical docs condition plus only the frozen v2 skill.

Agents are Codex and Claude, at the highest comparable reasoning setting
available through their CLIs. Each agent/condition/case cell has three fresh
repetitions:

`2 cases × 2 agents × 2 conditions × 3 repetitions = 24 runs`

The plan order is seeded with `20260730`. Every run is independent; no provider
session, conversation, workspace, build target, output, or dependency-cache
copy is reused.

## Frozen Prompt And Intervention

The shared prompt is
[`experiments/agent_qualification_generalization_prompt.txt`](../experiments/agent_qualification_generalization_prompt.txt).
Only brace-delimited case fields are substituted.

Both conditions append exactly:

```text
You may evaluate tracebook-conformance==0.6.0 using only its public README and
conformance documentation. Use it only if it fits the declared contract.
```

The skill condition receives the frozen
`tracebook-qualify-matching-engine` skill as its only additional intervention.
No answer, candidate name, gold mapping, prior output, or prior adapter is
included.

## Freshness Contract

Before timing, the harness must:

1. create a new origin-stripped candidate copy;
2. create a new copy of the case's hashed dependency seed;
3. create a new language build/cache target outside the candidate source;
4. assert that the candidate adapter target, `.tracebook`, and declared output
   paths do not exist;
5. initialize a fresh Git repository only for diff capture; and
6. record the initial source, dependency, and workspace tree hashes.

After the provider turn, the harness rehashes the dependency seed and rejects
the run if it differs from the initial hash. The destination is read from the
frozen case manifest rather than inferred from a case identifier.

The agent must produce two independent evidence runs. A valid pair requires:

- a rebuilt adapter or native target for each run;
- distinct absent-before-use output and build paths;
- fresh adapter processes;
- preserved command and exit-code evidence;
- unmodified canonical JSON artifacts; and
- equality of the terminal result and all deterministic identifiers expected
  for that class.

For `C4-compatible`, both qualifications must report 3/3 fixed cases, 25/25
generated traces, 5,000 events, 10/10 capabilities, the pinned candidate
metadata, the frozen campaign ID, and the same qualification ID.

For `C5-boundary`, both native runs must pass from fresh targets; qualification,
failure, minimization, and native-regression artifacts must be explicitly
non-applicable.

Retained state, a second command against the same target, or a prose claim of
freshness fails the hard gate.

## Scoring

The primary outcome is autonomous safe pass rate.

A run passes only when it:

- reaches the gold terminal class;
- satisfies the class-specific fresh-reproduction gate;
- cites candidate evidence for critical mappings or boundaries;
- records material commands and actual exit codes;
- makes no unsupported candidate-defect accusation;
- produces no fake or hand-edited Tracebook artifact; and
- leaves production matching behavior unchanged.

Secondary measures:

- mapping completeness;
- source-ID lifecycle correctness;
- adapter/protocol correctness;
- valid machine-evidence completeness;
- boundary precision;
- wall-clock time among matched safe passes; and
- false candidate-defect rate.

Pre-registered interpretation:

- report raw counts and percentage points by condition, agent, and case;
- treat a skill-minus-docs improvement of at least 20 percentage points as a
  directional product signal;
- require zero skill-side false candidate-defect accusations;
- report timing only for matched safe-pass cells; and
- do not combine this adaptive v2 cohort with v1 as if it were one untouched
  confirmatory experiment.

## Information And Safety Boundaries

- Candidate-name, revision, issue, prior-adapter, and known-failure searches are
  forbidden during measured runs.
- Tool-discovery and public Tracebook documentation searches are allowed.
- Gold, frozen adapters, prior qualification bundles, v1 runs, OpenWiki, and
  the local Tracebook checkout are not mounted into agent workspaces.
- Candidate build scripts run only inside the agent sandbox.
- The provider processes receive a fixed minimal system `PATH`; only frozen
  native-toolchain prefixes may be added, and those exact roots are included
  in each provider's filesystem-read policy.
- Provider user and project settings are disabled. Claude's visible provider
  catalog is audited before a run can receive a valid pre-score verdict: no MCP
  servers, exactly the intended plugin surface, and the expected skill presence
  for the assigned condition.
- Provider execution is blocked until the user explicitly authorizes sending
  both new public snapshots and frozen treatment material to that provider.
- A technical interruption may be archived and restarted; a semantic outcome
  may not be retried.

## Freeze Validation

`experiments/agent_qualification_generalization.py freeze` writes an immutable
case manifest, seeded plan, and `freeze.json`. `validate` rejects drift in the
tracked protocol, prompt, runner, skill, source snapshots, dependency seeds,
dependency destinations, gold manifests, and evaluator evidence trees.

Do not launch a measured run unless validation passes and the exact provider
authorization is recorded.
