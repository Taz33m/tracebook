# PR #84 Review Disposition

This records the 30 comments in the initial automated review of
[PR #84](https://github.com/Taz33m/tracebook/pull/84), against `0c98e18`.
It is a disposition of that review, not release approval or a new experiment
result. Source and regression tests, not the review's severity labels, determine
the implemented changes.

## Release-code fixes

The following 17 comments are addressed in the release implementation. Comment
IDs identify the original GitHub review comments. Existing qualified-profile
identities and retained evidence are not rewritten to accommodate a change.

| Comment ID | Change | Regression / verification surface |
| --- | --- | --- |
| 3939088988 | Close the L3 adapter before emitting `complete`; a failed close is one terminal adapter error, with no retry. | `test_server_close_failure_is_one_terminal_adapter_error` |
| 3939088991 | Both Rust adapters fail before emitting a positive fill/remainder normalized to zero. Inputs remain exact native lots. | Shared quantity tests and both adapters' `positive_fills_and_remainders_cannot_be_emitted_as_zero` |
| 3939089005 | Share the faulty FIFO-reordering fixture between in-process and external tests; retain each existing engine identity. | Existing campaign, reduction, and failure-ID regressions |
| 3939089008 | Validate typed L3 outcomes, fills, snapshot entries, and book collections during construction. | `test_invalid_in_process_observation_is_classified_as_protocol_failure` |
| 3939089018 | Correct the orderbook-rs translation contract: 12 native decimal places, independent output precision. | Quantity conversion regressions and README review |
| 3939089021 | Exact FIFO probe subtraction independent of ambient Decimal precision; remove context-rounding from shared canonical formatting too. | `test_probe_subtraction_and_canonical_values_are_exact_at_any_context_precision` |
| 3939089023 | Separate client decoding/validation failures from exceptions raised inside adapters. | Factory/apply/snapshot exception tests and malformed-payload test |
| 3939089028 | Require actual non-negative integer snapshot/finish counters; reject booleans, missing values, null, floats, and strings. | Parameterized session-counter tests |
| 3939089029 | Require Go snapshot/finish counters and an object event payload, without zero-value defaults accepting missing/null fields. | `TestServerRejectsMissingNullAndMistypedRequiredFields` |
| 3939089032 | Normalize timeout conversion errors, including oversized integers, into the public L3 error contract. | `test_external_timeout_overflow_is_a_public_validation_error` |
| 3939089038 | Wrap invalid ready-frame engine metadata as a book-replay protocol error and reap the child. | Invalid-metadata/child-reaping regression |
| 3939089052 | Stage in a private read-only directory and link relative to its open descriptor. Protect the payload entry from ordinary competing writers and anchor publication across directory replacement. | Stage-entry and stage-directory replacement tests, plus existing rollback/competing-destination tests |
| 3939089055 | Count price relocation only when the price actually changes, not on side-only updates. | `test_side_only_update_does_not_claim_price_relocation_coverage` |
| 3939089057 | Apply the documented conservative same-side aggregate envelope to IOC/FOK as well as GTC limit orders; retain the market exemption. | `TestSideAggregateEnvelopeIncludesIOCAndFOKButNotMarket` |
| 3939089063 | Report symlink-loop path resolution as a controlled CLI error before starting a candidate. | Symlink-loop CLI regression |
| 3939089067 | Handle corrupt deflated source inventories as a failed distribution privacy check. | Corrupt real ZIP-block regression |
| 3939089071 | Bound the workflow path reader to `on.pull_request.paths`, allowing comments, blank lines, different indentation, and other trigger keys. | Positive layout variants and negative job/other-trigger regressions |

Two limits matter when interpreting these fixes:

- Insufficient output precision is an operational error, not a semantic
  rejection or a successful qualification. Native matching inputs are not
  rounded, inflated, retried, or silently removed. The Python CLOB reference
  also rejects zero normalized positive quantities; its positive-value wire
  contract has not been relaxed.
- Artifact publication is not a security sandbox against root or same-user
  code deliberately changing permissions or writing through existing file
  descriptors. Use trusted output directories. Normal concurrent writers must
  not overwrite destinations, and rollback preserves competing files. The
  directory-permission test is intentionally skipped when running as root.

## Frozen research findings — open, not silently patched

These 13 findings are in the historical/hash-bound research harness. The
accepted scope leaves that code, frozen inputs, evaluator material, and retained
results unchanged. They remain open pending a separate research disposition;
the release fixes above do not resolve them.

| Comment ID | Reported issue | Required next decision/check |
| --- | --- | --- |
| 3939088982 | V2 Claude auth-route probe inherits ambient environment. | Audit route-binding evidence locally; use a sanitized, explicit environment in a newly versioned runner. |
| 3939088984 | V2 finalization failure before verdict can quarantine a completed provider turn. | Audit lifecycle receipts without rerunning providers; a future runner needs a durable provider-completion barrier before finalization. |
| 3939088987 | V2 frozen snapshots permit symbolic links. | Assess snapshot inventories and provider read boundaries locally; preregister an explicit link policy for future runs. |
| 3939088994 | Baseline live copy is not checked against its frozen source before launch. | Audit existing snapshot bindings; bind and verify copied workspaces in a future runner. |
| 3939089000 | Baseline command success can outlive a failed/timed-out provider. | Audit lifecycle exit receipts; separate transport completion and semantic outcomes in a new runner. |
| 3939089002 | Treatment technical verdict handling exempts some unsuccessful provider exits/timeouts. | Audit treatment lifecycle verdicts separately from V2, which has its own completion checks. |
| 3939089010 | Missing/unreadable V2 bound file can escape as `OSError`. | Add controlled validation errors in a future version, preserving frozen hash bindings. |
| 3939089043 | Baseline missing reported model falls back to `opus`. | Treat absent model evidence as missing, not observed identity, in a separately versioned audit/fix. |
| 3939089045 | Baseline permits an execution-time timeout override. | Compare receipts to the preregistered budget before deciding whether existing runs are affected. |
| 3939089048 | Baseline candidate/gold paths lack private-root confinement. | Audit declared paths locally; enforce resolved boundaries in a new runner. |
| 3939089050 | Baseline case IDs admit path components. | Audit manifest IDs; enforce a safe identifier grammar in a new runner. |
| 3939089056 | Treatment validation omits some workspace evidence artifacts. | Audit artifact completeness locally and specify required evidence in a new validator version. |
| 3939089060 | Baseline quarantine checks a verdict sentinel that baseline does not write. | Distinguish completed turns from incomplete interruptions using lifecycle receipts, not a missing sentinel. |

These are source-level risks, not proof that a particular completed cohort was
contaminated. Do not pool different harness generations, modify an old frozen
runner in place, reinterpret missing metadata as observed evidence, or rerun a
completed provider turn. An approved audit should be local and preserve every
interruption artifact; any necessary future collection needs a new versioned
protocol and explicit authorization. No private gold or evaluator artifacts
belong in a provider request or this public document.

## Release gate

Local verification of the fixes passed: Python 3.13 had 651 passing tests,
one skip, and 82.38% coverage; the focused Python 3.10 regressions passed too.
Formatting, lint, type checking, static security checks, pinned Rust 1.88/Go
1.23.5 checks, distribution privacy, and sdist/wheel agreement passed.
Fresh 5,000-event qualifications retained the exact orderbook-rs, gocronx, Go
FIFO-limit, and Go partial-fill identities. The Go full-profile FOK failure and
the Nautilus 2,500-event positive campaign/three-event negative control retained
their expected identities and reduced trace. These are local checks; consult
the PR for the latest remote CI and review state.

Keep the research conversations open until their disposition is agreed. Passing
release tests does not justify dismissing those comments, bypassing repository
protections, merging the PR, or publishing a tag. The previous maintainer
invitation and its immutable source/wheel references remain unchanged.
