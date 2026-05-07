# Tracebook Project Plan

## Current Scope

`tracebook` is an alpha Python order book simulator focused on matching semantics, synthetic order-flow experiments, and profiling hooks. The code is packaged as `tracebook` under `src/tracebook`.

## Current Structure

```text
src/tracebook/
  core/
    order.py
    orderbook.py
    matching_engine.py
    price_level.py
  algorithms/
    fifo.py
    pro_rata.py
  profiling/
    magic_trace_wrapper.py
    performance_monitor.py
    trace_analyzer.py
    trace_visualizer.py
  simulation/
    order_generator.py
    simulation_engine.py
  visualization/
    dashboard.py
tests/
  test_orderbook_semantics.py
examples/
  full_simulation_demo.py
```

## Near-Term Work

- Expand benchmark coverage beyond the current smoke/FIFO/pro-rata/cancellation scenarios.
- Add measured local baseline reports under `benchmark_results/` when comparing optimization work.
- Revisit data structures after correctness stabilizes; the current Python dict/list implementation is clear but not yet a low-latency final form.
- Consider fixed-point price/quantity representation only after benchmark data shows float handling is a bottleneck.

## Performance Goals

- Establish reproducible benchmark baselines before publishing throughput or latency claims.
- Keep profiling overhead visible in reports.
- Prefer correctness and deterministic matching semantics before deeper optimization.
