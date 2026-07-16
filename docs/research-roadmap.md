# Research-Grounded Roadmap

This note records what recent matching-engine and stateful-testing work changes
about Tracebook's roadmap. It is a product decision record, not a claim that the
project implements every cited technique.

## What The Evidence Says

- The 2026 [Flash matching-engine study](https://arxiv.org/abs/2606.01183)
  drives 247 engines through one harness. Only 47 are correct as shipped, and
  the study reports 181 upstream issues. Its large cancel-dominated workload,
  full report-stream hash, and unpredictable live book audits show that broad
  reproducible discovery can find semantic drift that hand-written cases miss.
- [RESTler](https://arxiv.org/abs/1806.09739) shows that stateful generators need
  producer-consumer dependencies and response feedback to avoid spending their
  budget on invalid sequences. Tracebook already derives lifecycle operations
  from reference-active orders and records applied/rejected outcomes.
- [TCP-Fuzz](https://www.usenix.org/conference/atc21/presentation/zou) found that
  transition coverage was more useful than counting states and combined it with
  dependency-aware generation and a differential oracle. That makes semantic
  lifecycle transitions a better future experiment than raw book-state counts.
- [NEZHA](https://www.cs.columbia.edu/~suman/docs/nezha.pdf) retains inputs that
  create new relative behavior across implementations. This supports a future
  black-box differential-diversity experiment using Tracebook observations,
  without requiring candidate source instrumentation.
- AFLNet's 2025
  [five-year retrospective](https://arxiv.org/abs/2412.20324) warns that state
  selection strategies often produce small or statistically insignificant
  gains and that state coverage does not necessarily predict code coverage.
  Tracebook should benchmark guidance before making it a public contract.
- The 2026 [FEST paper](https://www.usenix.org/conference/nsdi26/presentation/li)
  reports higher behavior and scenario coverage from feedback-guided mutation
  of recorded executions. Its controlled-mutation and explicit-scenario ideas
  are promising for lifecycle traces, but its distributed-schedule results do
  not transfer automatically to matching engines.
- [Probabilistic Monotonicity Assessment](https://arxiv.org/abs/2506.11614)
  reduces delta-debugging executions, but can filter failure-inducing subsets
  when practical assumptions fail. Tracebook keeps deterministic one-minimal
  reduction until candidate runtime data shows minimization is the bottleneck.
- [ABIDES](https://arxiv.org/abs/1904.12066),
  [JAX-LOB](https://arxiv.org/abs/2308.13289), and the 2024
  [LOB simulation review](https://arxiv.org/abs/2402.17359) cover realistic
  market simulation, agent behavior, and accelerated research workloads. They
  reinforce Tracebook's decision not to become another trading simulator.

## The Next Product Experiment

Qualification version 1 is the immediate move. One command runs only the fixed
cases inside a declared profile, follows with deterministic stateful generation,
checks semantic coverage, and emits a reviewable JSON/JUnit/failure bundle.

We will test this with external engine authors and record:

| Measure | Why it matters |
| --- | --- |
| Time to first qualification | Direct measure of integration friction |
| Adapter source lines | Approximation of translation burden |
| Failed attempts and questions | Finds unclear protocol and documentation |
| Profile boundary disputes | Finds missing or over-broad semantics |
| CI adoption | Stronger demand signal than a successful demo |
| Reduced regressions accepted | Evidence that the output is useful in maintenance |

## The Next Technical Experiment

After qualification has external use, compare two generators under an identical
candidate-run budget:

1. current deterministic reference-driven generation;
2. structure-preserving mutation that retains traces exposing a new semantic
   transition tuple or a new reference/candidate behavior tuple.

Use held-out defects, publish seeds and artifacts, and compare time to first
divergence plus reduced trace size. Do not mutate existing profile versions or
campaign hashes. A successful guided mode receives a new generator version and
artifact contract; an inconclusive result remains research code.

## 2026-07-16 Result

The qualification and guidance experiments are recorded in the
[design-partner study](qualification-design-partners.md). Both pinned Rust
candidates passed the same 28-run `fifo-limit-v1` qualification. The
`orderbook-rs` maintainer loop is complete; `gocronx/matcher` review and upstream
CI adoption remain pending.

Guided suffix regeneration improved time-to-divergence for one injected defect
but regressed on the historical defect and a second injected defect. It therefore
remains research-only. No generator version, profile, CLI, or artifact contract
changed.
