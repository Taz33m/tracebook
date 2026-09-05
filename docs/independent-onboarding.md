# Independent Onboarding: Captured Go Evidence

## Status And Scope

This packet is prepared for one independent maintainer attempt against
`intrepidkarthi/orderbook` v0.26.0, source commit
`51d480cdb68b9989febb0b075d291cf891f425b3`. A Tracebook-owned rehearsal is not
independent onboarding, adoption, or an upstream decision.

The request is deliberately narrow: can the maintainer reproduce and understand
one verified FIFO-limit evidence pair, and would they retain the result or its
CI check? The native unfillable-FOK rejection is a documented contract
difference, not a bug accusation or a request to change native matching.

The captured evidence workflow and this adapter are unreleased. The published
0.6.0 wheel does not contain the new evidence commands. Use a wheel built from
the exact Tracebook checkout below, not `pip install tracebook-conformance==0.6.0`.
The built wheel still carries version 0.6.0: record its SHA-256 and source commit
so it cannot be confused with that release. Do not upload it to PyPI.

## Reproduce From A Reviewed Checkout

Prerequisites: Git, Go >=1.23.5, Python >=3.10, and a POSIX shell. Start at the
root of a clean, reviewed Tracebook checkout containing this packet. Review the
adapter before running it; candidate execution is not sandboxed. These commands
download only public source and Go modules, and never use experiment material.

Create and inspect the conformance wheel following the repository's build and
privacy gates. Then set `WHEEL` to that artifact's absolute path. Run the
following with Bash; all candidate, cache, build, and qualification outputs stay
under a fresh directory, which is preserved on failure.

```bash
set -euo pipefail
TRACEBOOK_ROOT=$PWD
TRACEBOOK_REV=$(git rev-parse HEAD)
test -z "$(git status --porcelain --untracked-files=no)"
WHEEL="$TRACEBOOK_ROOT/dist/conformance/tracebook_conformance-0.6.0-py3-none-any.whl"
test -f "$WHEEL"
shasum -a 256 "$WHEEL"

ATTEMPT_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/tracebook-onboarding.XXXXXX")
export ATTEMPT_ROOT
python3 -m venv "$ATTEMPT_ROOT/tool"
PYTHON="$ATTEMPT_ROOT/tool/bin/python"
"$PYTHON" -m pip install "$WHEEL"
"$PYTHON" -m tracebook.conformance.cli --help

CANDIDATE_REV=51d480cdb68b9989febb0b075d291cf891f425b3
git clone --depth 1 --branch v0.26.0 \
  https://github.com/intrepidkarthi/orderbook.git "$ATTEMPT_ROOT/source"
test "$(git -C "$ATTEMPT_ROOT/source" rev-parse HEAD)" = "$CANDIDATE_REV"

WORKSPACE="$ATTEMPT_ROOT/evidence"
"$PYTHON" -m tracebook.conformance.cli evidence-init "$ATTEMPT_ROOT/source" \
  --workspace "$WORKSPACE" \
  --candidate-name 'intrepidkarthi/orderbook CLOB' \
  --candidate-revision "$CANDIDATE_REV"
SNAPSHOT=$("$PYTHON" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["candidate"]["snapshot_id"])' \
  "$WORKSPACE/evidence-plan.json")

for RUN_NUMBER in 1 2; do
  RUN_ROOT="$WORKSPACE/runs/run-$RUN_NUMBER"
  git -C "$TRACEBOOK_ROOT" archive "$TRACEBOOK_REV" integrations/intrepid_orderbook \
    | tar -x --strip-components=2 -C "$RUN_ROOT/adapter"
  (
    cd "$RUN_ROOT/adapter"
    export GOWORK=off GOCACHE="$RUN_ROOT/cache/build"
    export GOMODCACHE="$RUN_ROOT/cache/modules" GOPATH="$RUN_ROOT/cache/gopath"
    go mod edit "-replace=github.com/intrepidkarthi/orderbook=$RUN_ROOT/candidate"
    go mod download
    go build -mod=readonly -trimpath \
      -ldflags "-X main.engineRevision=$CANDIDATE_REV -X main.engineSnapshot=$SNAPSHOT" \
      -o "$RUN_ROOT/build/adapter" .
    go version -m "$RUN_ROOT/build/adapter"
  )
  "$PYTHON" -m tracebook.conformance.cli qualify \
    --profile fifo-limit-v1 --suite-version v2 \
    --seed 42 --traces 25 --events-per-trace 200 --max-minimize-runs 100 \
    --timeout 10 \
    --candidate-name 'intrepidkarthi/orderbook CLOB' \
    --candidate-revision "$CANDIDATE_REV" --candidate-snapshot "$SNAPSHOT" \
    --output-dir "$RUN_ROOT/qualification" \
    --candidate "$RUN_ROOT/build/adapter"
done

"$PYTHON" -m tracebook.conformance.cli evidence-verify "$WORKSPACE/evidence-plan.json"
```

Expected result: each run passes 3/3 fixed cases, 25/25 generated traces, 5,000
generated events, and 10/10 capabilities. Verification emits
`evidence-manifest.json` only when both candidate trees remain unchanged and
their canonical bundles are byte-identical. Identity-bound qualification IDs
differ from the integration README's unbound runs; do not substitute one for
the other.

The source replacement above is essential: embedding a snapshot ID in an
otherwise ordinary module-cache build does not prove that captured source was
used. Each run gets its own module and compiler caches. This remains an
auditable process, not a hermetic build or proof against a dishonest adapter;
see the [evidence trust boundary](conformance.md#captured-two-run-evidence).

## Measurement Record

Before the independent attempt, record its operator, relationship to the engine,
start time, Tracebook commit and wheel hash, candidate commit and snapshot,
operating system, Python and Go versions. Start the clock before environment
setup; do not exclude dependency downloads or time spent asking questions.

Record each failed command and each human correction, including who suggested
it, what changed, and minutes spent. Preserve failed outputs in separate roots;
do not edit a failed bundle into a passing one. Record elapsed time to the first
verified pair or the first blocker, even if no pair succeeds. A provided adapter
measures time to evidence, not time to author an adapter.

The result record must contain:

- operator and relationship: **not yet observed**;
- first verified pair or blocker time: **not yet observed**;
- failed attempts and human corrections: **not yet observed**;
- plan, manifest, qualification and campaign IDs: **not yet observed**;
- semantic questions or disagreements: **not yet observed**; and
- retention decision (retained, declined, undecided) plus a public link if
  retained: **not yet observed**.

An external maintainer running the packet establishes independent use, not
retention. A promise to add CI is not retained CI. A Tracebook-owned run or job
is neither. Report the attempt through the existing Engine qualification issue
form, adding the wheel hash, evidence manifest and human-correction log. Remove
credentials, proprietary traces and machine-specific paths before publication.

## Proposed Maintainer Invitation (Not Sent)

> We have a pinned, out-of-tree Tracebook adapter for your v0.26.0 CLOB. FIFO
> limit qualification passes; the broader FOK profile stops on your native
> unfillable-FOK rejection, which we preserve as a contract difference, not a
> defect. Would you be willing to try one captured two-run FIFO evidence packet
> and tell us whether its result is useful enough to retain? No matching-engine
> change or in-tree adapter commitment is requested. We would record setup
> time, corrections, and your retention decision, including a failed attempt.

Attach the reviewed Tracebook source revision and artifact hashes before
sending. A maintainer response and any independent result must be recorded as
new evidence; this prepared invitation establishes neither.
