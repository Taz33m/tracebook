# Qualification Design-Partner Study

Status at `2026-07-19T23:09:03Z`: two independently authored Rust engines have
passing repository-local `fifo-limit-v1` artifacts. One Rust maintainer review
loop is complete; the other is awaiting an upstream reply. A third,
maintainer-directed Go integration now passes from the public package and is
proposed in the candidate repository as draft PR
[`geseq/orderbook` #30](https://github.com/geseq/orderbook/pull/30). It remains
an adoption proposal until the maintainer reviews it and decides whether to
retain the workflow.

## Frozen Protocol

The measurement contract was fixed before either qualification result:

- profile: `fifo-limit-v1`;
- suite selection: qualification version 1;
- seed: `42`;
- generated workload: 25 traces x 200 events;
- minimization budget: 100 runs if a divergence appears;
- primary time: empty Cargo target through committed qualification directory;
- failed attempt: nonzero build/protocol exit, excluding a valid semantic exit 1;
- adapter size: physical lines in the integration's `adapter.rs`;
- environment: Apple M2, macOS Darwin 24.4.0, Python 3.10.5, Rust 1.88.0.

Cargo's registry cache was warm, but each release target directory and
qualification destination was empty. Build and qualification time are reported
separately so this limitation stays visible.

## Results

| Measure | `orderbook-rs 0.12.0` | `gocronx/matcher 0.2.0@b8d48356` |
| --- | ---: | ---: |
| Empty-target release build | 34.41 s | 22.26 s |
| Qualification command | 4.71 s | 5.80 s |
| Build to first artifact | 39.12 s | 28.06 s |
| Candidate process runs | 28 | 28 |
| Fixed cases | 3/3 | 3/3 |
| Generated traces | 25/25 | 25/25 |
| Generated events | 5,000 | 5,000 |
| Semantic capabilities | 10/10 | 10/10 |
| Adapter file | 1,056 lines | 672 lines |
| Failed build attempts | 0 | 0 |
| Failed qualification attempts | 0 | 0 |

The adapter-size comparison is directional. The older `orderbook-rs` file also
contains historical-revision compatibility and two explicit fault modes; the
new `matcher` file contains only its maintained candidate path.

The qualification IDs differ because engine identity is bound into the artifact:

- `orderbook-rs`: `sha256:5597e97cf8dbf42d25dedb035bb6e1fd05dbdb2ea31d2576a802fd9b3d0efee4`
- `gocronx/matcher`: `sha256:f702a24e4e0113b3591107aab40f2aec189daee4f87486273c791481db622591`

Both share campaign ID
`sha256:59d70645ff13f12fa4af23af69631714df22ef5f25cd1104b0c1124f98f71f6a`,
which confirms that the generated workload was identical.

A separate blank-directory proof created a fresh Python 3.13 virtual environment,
installed only the built wheel, extracted the built sdist, compiled the pinned
adapter from that archive, and reproduced the same 28-run qualification ID.

## Author Friction

| Measure | `orderbook-rs` | `gocronx/matcher` |
| --- | --- | --- |
| Protocol/semantic questions | 4 asked, 4 answered | 3 asked, 3 pending |
| First maintainer response | About 7 h 27 m after issue creation | Pending after more than 37 h |
| Profile-boundary disagreements | 2 surfaced and resolved | None recorded; 2 assumptions await review |
| Upstream effect | Public priority docs, property tests, and snapshot-order repair | None yet |
| Qualification/reduced trace in Tracebook CI | Yes / historical reduced trace also yes | Yes / no failure to reduce |
| Qualification artifact in upstream CI | No | Pending maintainer answer |

The two `orderbook-rs` boundary corrections were useful. First, in-place upsize
was outside `fifo-limit-v1` because snapshot order once differed from consumption
order; upstream later repaired and tested that observation surface. Second, an
initial `8/8` suite claim included a scope mismatch; Tracebook recorded the
FIFO-only adapter's pro-rata failure instead of broadening the claim.

The main `matcher` friction is not process transport. It is whether consumers
should decode public snapshot bytes to observe every resting order, and whether
cancel plus resubmit is the preferred representation of Tracebook's broader
replacement operation. The adapter pins snapshot version 1 and labels both
assumptions while review is pending.

## Public-Package Qualification Proposal

The `geseq/orderbook` maintainer answered the three initial contract questions
and then proposed adding a native amendment API when Tracebook showed that
cancel plus create could not honestly satisfy priority-preserving reduction.
Draft PR #30 implements that maintainer-selected direction, a detached FIFO
snapshot, an adapter example, and a public-PyPI qualification job.

| Measure | `geseq/orderbook@88e80980` |
| --- | ---: |
| Maintainer API direction to first local artifact | About 43 h 24 m |
| Public-package qualification after candidate build | 17.30 s |
| Candidate process runs | 28 |
| Fixed cases | 3/3 |
| Generated traces / events | 25/25 / 5,000 |
| Semantic capabilities | 10/10 |
| Adapter source / adapter tests | 865 / 130 lines |
| Semantic qualification failures | 0 |
| Failed clean-package attempts before candidate launch | 1 |
| Qualification retained in candidate CI | Proposed; maintainer decision pending |

The final qualification ID is
`sha256:4216d817384360e6edc50bf6b801b80fa6306b4e2910f5b9bed968b4daaf79d9`.
It was reproduced from a fresh remote clone with the `tracebook-sim==0.5.0`
wheel fetched from PyPI. The local environment used `--no-deps` after repeated
network timeouts transferring NumPy; the proposed upstream job performs the
normal dependency-resolving install and is still awaiting its first run.

The single failed clean-package attempt was useful product evidence: public
version 0.4.1 did not yet expose `qualify`, even though the repository did. That
release-integrity gap directly triggered 0.5.0. No candidate protocol or
semantic attempt failed after the public command became available.

The integration surfaced four boundaries rather than hiding them:

- priority-preserving reduction required a native engine operation;
- replacement semantics were initially left to the engine, then implemented in
  the draft as priority-losing remove and re-entry;
- the engine's native eight-decimal quantity representation required an exact,
  workload-bounded bridge to Tracebook's twelve-decimal wire values;
- the 865-line adapter includes protocol framing, canonical hashing, state
  translation, and that decimal bridge because no shared Go adapter SDK exists.

That file size is a hypothesis for reusable Go protocol support, not yet a
feature decision. A second independent Go onboarding must show the same burden
before Tracebook extracts an SDK. There is no reduced trace for this candidate
because all compared events conformed.

## Diversity Experiment

The second experiment removed all built-in priority probes, then compared fresh
deterministic generation with structure-preserving suffix regeneration retained
by novel black-box semantic transition tuples. Each strategy received 40
candidate runs per trial, 120 events per run, and 12 trials per defect. Guided
selection examined eight reference-side mutations per candidate run, so
candidate budgets were equal while reference-side work was intentionally not.

| Held-out defect | Deterministic median runs / events | Guided median runs / events | Result |
| --- | ---: | ---: | --- |
| Historical `orderbook-rs` issue #88 | 1 / 84.5 | 2 / 200.5 | Guided worse |
| Injected reduction requeues | 2 / 182.5 | 1 / 88.0 | Guided better |
| Injected replacement keeps priority | 1 / 69.5 | 2 / 204.0 | Guided worse |

Every strategy found every defect in all 12 trials, but guidance improved only
one of three time-to-divergence distributions. Tracebook therefore does **not**
ship guided exploration. The public generator, hashes, profiles, CLI, and
artifact contracts remain unchanged.

This result also exposes a structural limitation: before the first divergence,
a strict differential oracle sees identical reference/candidate behavior.
Relative-behavior diversity therefore collapses to reference semantic diversity
on conformant prefixes. A future experiment needs a stronger pre-divergence
signal or a defect family where deterministic generation is demonstrably weak.

The reproducible harness is
[`experiments/guided_diversity.py`](https://github.com/Taz33m/tracebook/blob/main/experiments/guided_diversity.py).
The frozen 72-trial artifact is
[`guided-diversity-v1.json`](https://github.com/Taz33m/tracebook/blob/main/experiments/results/guided-diversity-v1.json),
SHA-256 `e798d014dc8ea63cd7714aedf838678d1249354ba9e7adb3f965def9289c9a6c`.
Runtime timing is excluded from the canonical payload; two complete reruns
produced this same byte-for-byte hash.

## Product Decision

The next product step is not generator expansion or a speculative Go SDK. It is
getting a maintainer decision on the `geseq/orderbook` draft and observing a
second independent public-package onboarding. The first result tells us whether
the artifact belongs in an engine's CI; the second tells us whether the large Go
adapter reflects a repeated protocol burden or only this engine's boundaries.

New attempts should use the public
[engine qualification report](https://github.com/Taz33m/tracebook/issues/new?template=engine_qualification.yml)
to record time to evidence, engine-specific adapter size, failed attempts,
protocol questions, profile-boundary disagreements, and the upstream CI
decision. Repository-local integration work does not count as adoption until an
independent engine author runs or accepts the workflow.
