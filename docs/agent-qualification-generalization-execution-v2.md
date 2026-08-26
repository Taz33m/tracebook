# Agentic Qualification V2 Execution Addendum

**Protocol:** `agent-qualification-generalization-v2`
**Layer:** separately frozen execution harness

This addendum defines how the already frozen C4/C5 cohort is executed. It does
not change the case snapshots, gold data, prompt, skill, matrix, scoring, or
seeded order in the primary protocol.

## Preconditions

Before the execution layer may be frozen, the harness must:

1. validate the primary `freeze.json` and every bound candidate, cache, gold,
   prompt, protocol, runner, and skill hash;
2. bind the exact host, provider-client binaries, requested models, runner
   dependencies, and native toolchain versions;
3. prove with a synthetic canary that the skill is visible only in each
   provider's skill condition; and
4. leave provider authorization outside the immutable freeze so authorization
   can be recorded later without changing experimental inputs.

The Claude binding probe and every Claude run explicitly disable the
Bedrock, Vertex, and Foundry routing flags in their isolated settings. The
harness still refuses to freeze unless `claude auth status` identifies an
authenticated `firstParty` route.

Every provider run also starts without user or project configuration. The
harness supplies a fixed minimal system `PATH`, adds only the toolchain prefixes
bound by the execution freeze, and grants read access to those exact roots.

The synthetic canary contains no candidate source, Tracebook guidance, gold
data, or measured prompt material.

## Authorization

Measured execution is denied by default. Each provider requires its own new,
case-specific authorization record covering both public C4/C5 snapshots, the
frozen prompt, both treatment conditions, all preregistered runs for that
provider, and restarts after genuine technical interruptions. Authorization
for the earlier C1/C3 cohort does not apply.

## Isolation And Order

Every run receives a new origin-stripped workspace, new dependency-seed copy,
new writable tool state and build target, isolated provider state, and
condition-specific native skill surface. For Maven cases, the hashed fixture is
an immutable repository seed copied into a separate writable local repository.
For Cargo cases, the hashed fixture is an immutable vendored-source tree while
Cargo home and target directories are separate fresh writable paths. The local
Tracebook checkout, private gold, prior workspaces, and prior transcripts are
not mounted into the workspace.
Fresh candidate clones, cache copies, build products, tool state, and virtual
environments belong under the supplied external `FRESH_*` paths so the final
workspace evidence contains only source-sized audit artifacts and deliverables.

Before accepting a Claude turn, the harness audits the provider catalog and
requires no MCP servers, exactly one intended plugin, and skill visibility that
matches the assigned condition. The audit is preserved with the run.

The harness executes only the next entry in the frozen 24-run order. A later
entry is rejected until every earlier entry has a valid pre-score technical
verdict. Existing run or external-workspace paths are never overwritten.

## Technical Verdicts And Restarts

The harness records prompt, environment and binary bindings, provider JSONL,
stderr, workspace diff, final workspace, fixture state, terminal state, and
reported model before semantic grading. A pre-score verdict covers technical
validity only; it must not inspect whether the agent reached the correct
qualification outcome.

Fixture validity requires the final dependency-seed hash to equal its frozen
initial hash. Tool-generated usage metadata lives outside that seed. Workspace
inventory and patch capture include ignored candidate
files, using a temporary Git index so the evaluator's real index is unchanged.

A completed provider turn, refusal, timeout, or semantic failure is not
restartable. Only a genuine technical interruption that lacks a completed
provider turn may be archived intact and restarted. Archives remain under the
ignored private experiment tree.

If execution fails before a technical verdict exists, the harness removes the
new external workspace and quarantines the incomplete run root under the
private interruption archive. A documented clean restart is therefore possible
without overwriting evidence.

## Commands

The public executor is
`experiments/agent_qualification_generalization_runs.py`:

```text
python experiments/agent_qualification_generalization_runs.py shakedown
python experiments/agent_qualification_generalization_runs.py freeze
python experiments/agent_qualification_generalization_runs.py validate
python experiments/agent_qualification_generalization_runs.py authorize --provider codex --statement-file FILE
python experiments/agent_qualification_generalization_runs.py run --run-id RUN_ID
python experiments/agent_qualification_generalization_runs.py validate-run --run-id RUN_ID
```

`shakedown` and `freeze` must run on the same host and provider-client versions
used for measurement. `authorize` is an evidence-recording operation, not a
substitute for explicit user consent.
