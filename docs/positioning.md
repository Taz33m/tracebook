# Product Position

Tracebook is the conformance-testing and reproducible failure-analysis toolkit
for matching engines. Its reference engine exists to make semantics executable
and inspectable; it is not trying to become a production exchange or a complete
trading platform.

## Why This Wedge

Strong open-source systems already own broader categories:

- [ABIDES](https://github.com/abides-sim/abides) is an agent-based discrete-event
  market simulator designed for AI research, large agent populations, an
  exchange agent, and configurable network latency.
- [PythonMatchingEngine](https://github.com/Surbeivol/PythonMatchingEngine)
  focuses on replaying historical Level 3 orders and latency-aware HFT strategy
  simulation.
- [NautilusTrader](https://nautilustrader.io/docs/latest/concepts/order_book/)
  provides a Rust L1-L3 order book inside a much larger backtesting and live
  trading platform.

Building a smaller general-purpose simulator beside those projects would leave
Tracebook between categories: too slow and incomplete for production exchange
infrastructure, less realistic than dedicated HFT replay systems, and more
complex than a basic order-book tutorial.

Correctness work has a clearer user and a sharper question:

> Given the same event trace, where does this engine first disagree with the
> documented matching semantics, and what is the smallest trace that proves it?

## Jobs Tracebook Owns

| Job | Tracebook capability |
| --- | --- |
| Learn matching semantics | Small Python reference path, detached queue snapshots, and executable FIFO/pro-rata/IOC/FOK/STP cases |
| Reproduce a matching bug | Normalized traces, deterministic replay, exact first-divergence reports, and regression artifacts |
| Test another engine | Language-neutral stdio adapters, source-ID trade comparison, stable rejection codes, and full queue-state hashing |
| Reduce a failure | Deterministic delta debugging that emits JSONL and distinguishes one-minimal output from budget exhaustion |
| Guard performance | Machine-attributed benchmark artifacts with generation, matching, lifecycle, memory, and monitoring boundaries |
| Share workloads | Hash-locked synthetic conformance suite plus verified local historical-data corpora |

## Deliberate Non-Goals

- production exchange deployment;
- broker, venue, or risk-management connectivity;
- portfolio accounting and strategy lifecycle management;
- a realistic multi-agent market economy;
- a claim that Python latency predicts C++ or Rust production latency;
- reconstructing L3 queue priority from aggregated L2 data;
- redistributing exchange market data without explicit rights.

## Product Test

A feature belongs in the core roadmap when it improves at least one of these:

1. semantic coverage across matching engines;
2. localization or reduction of a failure;
3. reproducibility of a correctness or performance claim;
4. adapter interoperability without coupling an engine to Python internals.

Features that primarily make Tracebook a broader trading platform should remain
outside the core unless they are necessary to test a matching-engine contract.
