"""Compare unguided generation with semantic-transition diversity guidance.

This experiment deliberately lives outside ``src/tracebook``. A positive result
is required before any guided mode is added to the public CLI or artifact contract.
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import asdict, dataclass
from decimal import Decimal
from pathlib import Path
from typing import Iterable, Sequence

from tracebook.conformance import (
    AdapterFactory,
    BookState,
    ConformanceConfig,
    EngineMetadata,
    ExternalProcessAdapter,
    Observation,
    Outcome,
    ReferenceEngineAdapter,
    RestingOrder,
    TradeFill,
    run_conformance,
)
from tracebook.conformance.campaign import (
    _ActiveOrder,
    _SplitMix64,
    _active_orders,
    _generated_event,
    get_campaign_profile,
)
from tracebook.conformance.model import trace_sha256
from tracebook.core.order import OrderSide, OrderType
from tracebook.events import MarketEvent

Transition = tuple[str, ...]

DEFECT_METADATA = {
    "historical-orderbook-rs-issue-88": {
        "kind": "historical",
        "candidate": "historical orderbook-rs issue 88 adapter",
        "candidate_version": "0.8.0@53b4d2b0+pricelevel-0.7.0",
        "upstream_issue": "https://github.com/joaquinbejar/OrderBook-rs/issues/88",
    },
    "injected-reduce-requeues": {
        "kind": "injected",
        "fault": "A partial quantity reduction cancels and re-adds the order.",
    },
    "injected-replace-keeps-priority": {
        "kind": "injected",
        "fault": "A same-price non-increasing replacement stays in place.",
    },
}


@dataclass(frozen=True)
class TrialResult:
    defect: str
    strategy: str
    trial: int
    seed: int
    found: bool
    candidate_runs: int
    candidate_events: int
    divergence_event: int | None
    divergence_category: str | None


class _InjectedLifecycleAdapter:
    metadata = EngineMetadata("injected lifecycle fault", "1", "Python")

    def __init__(self, config: ConformanceConfig) -> None:
        self._inner = ReferenceEngineAdapter(config)

    def snapshot(self) -> BookState:
        return self._inner.snapshot()

    def close(self) -> None:
        self._inner.close()

    def _resting(self, event: MarketEvent) -> tuple[OrderSide, RestingOrder] | None:
        for book in self.snapshot().books:
            if book.symbol != event.symbol:
                continue
            for side, orders in ((OrderSide.BUY, book.bids), (OrderSide.SELL, book.asks)):
                for order in orders:
                    if order.order_id == event.order_id:
                        return side, order
        return None

    def _current_observation(
        self,
        index: int,
        outcome: Outcome,
        trades: Sequence[TradeFill] = (),
    ) -> Observation:
        state = self.snapshot()
        return Observation(index, outcome, tuple(trades), state.digest(), state.order_count)


class ReduceRequeuesAdapter(_InjectedLifecycleAdapter):
    """Injected defect: a partial quantity reduction forfeits FIFO priority."""

    metadata = EngineMetadata("injected reduce-requeues fault", "1", "Python")

    def apply(self, event: MarketEvent, index: int) -> Observation:
        resting = self._resting(event)
        if event.op != "reduce" or resting is None or event.quantity is None:
            return self._inner.apply(event, index)
        side, order = resting
        remaining = Decimal(order.remaining_quantity)
        reduction = Decimal(str(event.quantity))
        if reduction <= 0 or reduction >= remaining:
            return self._inner.apply(event, index)

        cancel_observation = self._inner.apply(
            MarketEvent(op="cancel", symbol=event.symbol, order_id=event.order_id),
            index,
        )
        if cancel_observation.outcome.status != "applied":
            return cancel_observation
        observation = self._inner.apply(
            MarketEvent(
                op="new",
                symbol=event.symbol,
                order_id=event.order_id,
                side=side,
                order_type=OrderType.LIMIT,
                price=float(Decimal(order.price)),
                quantity=float(remaining - reduction),
                owner=order.owner,
                timestamp_ns=event.timestamp_ns,
            ),
            index,
        )
        return self._current_observation(index, observation.outcome, observation.trades)


class SamePriceReplaceKeepsPriorityAdapter(_InjectedLifecycleAdapter):
    """Injected defect: same-price replacement incorrectly retains priority."""

    metadata = EngineMetadata("injected replace-keeps-priority fault", "1", "Python")

    def apply(self, event: MarketEvent, index: int) -> Observation:
        resting = self._resting(event)
        if event.op != "replace" or resting is None:
            return self._inner.apply(event, index)
        _, order = resting
        old_price = Decimal(order.price)
        new_price = old_price if event.price is None else Decimal(str(event.price))
        remaining = Decimal(order.remaining_quantity)
        new_quantity = remaining if event.quantity is None else Decimal(str(event.quantity))
        if new_price != old_price or new_quantity <= 0 or new_quantity > remaining:
            return self._inner.apply(event, index)
        outcome = Outcome("applied")
        if new_quantity < remaining:
            outcome = self._inner.apply(
                MarketEvent(
                    op="reduce",
                    symbol=event.symbol,
                    order_id=event.order_id,
                    quantity=float(remaining - new_quantity),
                    timestamp_ns=event.timestamp_ns,
                ),
                index,
            ).outcome
        return self._current_observation(index, outcome)


def generate_unprobed_trace(seed: int, event_count: int) -> tuple[MarketEvent, ...]:
    """Generate the fifo-limit workload without its built-in queue probe."""
    profile = get_campaign_profile("fifo-limit-v1")
    rng = _SplitMix64(seed)
    next_ids = {symbol: 1 for symbol in profile.symbols}
    reference = ReferenceEngineAdapter(profile.config)
    events: list[MarketEvent] = []
    active: tuple[_ActiveOrder, ...] = ()
    try:
        for index in range(1, event_count + 1):
            event = _generated_event(rng, profile, next_ids, active, index)
            events.append(event)
            reference.apply(event, index)
            active = _active_orders(reference.snapshot())
    finally:
        reference.close()
    return tuple(events)


def regenerate_suffix(
    source: Sequence[MarketEvent],
    seed: int,
) -> tuple[MarketEvent, ...]:
    """Keep a valid prefix and regenerate a dependency-aware suffix."""
    if len(source) < 2:
        return tuple(source)
    profile = get_campaign_profile("fifo-limit-v1")
    rng = _SplitMix64(seed)
    cut = 1 + rng.randbelow(len(source) - 1)
    prefix = tuple(source[:cut])
    next_ids = {symbol: 1 for symbol in profile.symbols}
    for event in prefix:
        if event.symbol in next_ids and event.order_id is not None:
            next_ids[event.symbol] = max(next_ids[event.symbol], event.order_id + 1)

    reference = ReferenceEngineAdapter(profile.config)
    events = list(prefix)
    try:
        for index, event in enumerate(prefix, 1):
            reference.apply(event, index)
        active = _active_orders(reference.snapshot())
        for index in range(cut + 1, len(source) + 1):
            event = _generated_event(rng, profile, next_ids, active, index)
            events.append(event)
            reference.apply(event, index)
            active = _active_orders(reference.snapshot())
    finally:
        reference.close()
    return tuple(events)


def semantic_transitions(events: Sequence[MarketEvent]) -> frozenset[Transition]:
    """Return black-box lifecycle transition tuples for diversity retention."""
    config = get_campaign_profile("fifo-limit-v1").config
    reference = ReferenceEngineAdapter(config)
    transitions: set[Transition] = set()
    previous_count = 0
    try:
        for index, event in enumerate(events, 1):
            observation = reference.apply(event, index)
            delta = observation.resting_order_count - previous_count
            previous_count = observation.resting_order_count
            side = event.side.name if event.side is not None else "NONE"
            order_type = event.order_type.name if event.op == "new" else "NONE"
            transitions.add(
                (
                    event.op,
                    side,
                    order_type,
                    observation.outcome.status,
                    observation.outcome.reason or "NONE",
                    str(min(len(observation.trades), 3)),
                    str(max(-2, min(delta, 2))),
                    _count_bucket(observation.resting_order_count),
                )
            )
    finally:
        reference.close()
    return frozenset(transitions)


def _count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    if count <= 4:
        return "2-4"
    return "5+"


def _guided_trace(
    rng: _SplitMix64,
    corpus: Sequence[tuple[MarketEvent, ...]],
    seen: set[Transition],
    event_count: int,
    pool_size: int,
) -> tuple[tuple[MarketEvent, ...], frozenset[Transition]]:
    candidates = []
    for _ in range(pool_size):
        seed = rng.next_u64()
        if corpus:
            source = corpus[rng.randbelow(len(corpus))]
            trace = regenerate_suffix(source, seed)
        else:
            trace = generate_unprobed_trace(seed, event_count)
        signature = semantic_transitions(trace)
        score = (len(signature - seen), len(signature), trace_sha256(trace))
        candidates.append((score, trace, signature))
    _, trace, signature = max(candidates, key=lambda item: item[0])
    return trace, signature


def run_trial(
    defect: str,
    strategy: str,
    trial: int,
    seed: int,
    candidate_factory: AdapterFactory,
    *,
    budget: int,
    event_count: int,
    pool_size: int,
) -> TrialResult:
    config = get_campaign_profile("fifo-limit-v1").config
    rng = _SplitMix64(seed)
    seen: set[Transition] = set()
    corpus: list[tuple[MarketEvent, ...]] = []
    compared_events = 0
    for run in range(1, budget + 1):
        if strategy == "deterministic":
            trace = generate_unprobed_trace(rng.next_u64(), event_count)
            signature = semantic_transitions(trace)
        elif strategy == "guided":
            trace, signature = _guided_trace(rng, corpus, seen, event_count, pool_size)
        else:
            raise ValueError(f"unknown strategy {strategy!r}")

        report = run_conformance(
            trace,
            candidate_factory,
            config=config,
            trace_name=f"{defect}:{strategy}:{trial}:{run}",
        )
        compared_events += report.compared_events
        if not report.conformant:
            divergence = report.divergence
            return TrialResult(
                defect=defect,
                strategy=strategy,
                trial=trial,
                seed=seed,
                found=True,
                candidate_runs=run,
                candidate_events=compared_events,
                divergence_event=divergence.event_index if divergence else None,
                divergence_category=divergence.category if divergence else None,
            )

        novel = signature - seen
        seen.update(signature)
        if strategy == "guided" and (novel or not corpus):
            corpus.append(trace)

    return TrialResult(
        defect=defect,
        strategy=strategy,
        trial=trial,
        seed=seed,
        found=False,
        candidate_runs=budget,
        candidate_events=compared_events,
        divergence_event=None,
        divergence_category=None,
    )


def summarize(results: Sequence[TrialResult], budget: int, event_count: int) -> dict:
    successes = sum(result.found for result in results)
    censored_runs = [result.candidate_runs if result.found else budget + 1 for result in results]
    event_limit = budget * event_count
    censored_events = [
        result.candidate_events if result.found else event_limit + 1 for result in results
    ]
    return {
        "trials": len(results),
        "discoveries": successes,
        "discovery_rate": successes / len(results),
        "median_candidate_runs_censored": statistics.median(censored_runs),
        "median_candidate_events_censored": statistics.median(censored_events),
    }


def guidance_decision(summaries: dict[str, dict[str, dict]]) -> dict:
    checks = {}
    for defect, strategies in summaries.items():
        baseline = strategies["deterministic"]
        guided = strategies["guided"]
        checks[defect] = {
            "discovery_rate_not_lower": guided["discovery_rate"] >= baseline["discovery_rate"],
            "candidate_runs_not_higher": guided["median_candidate_runs_censored"]
            <= baseline["median_candidate_runs_censored"],
            "candidate_events_improved": guided["median_candidate_events_censored"]
            < baseline["median_candidate_events_censored"],
        }
        checks[defect]["passed"] = all(checks[defect].values())
    ship = all(check["passed"] for check in checks.values())
    return {
        "ship_guided_exploration": ship,
        "rule": (
            "For every held-out defect: discovery rate must not fall, median censored "
            "candidate runs must not rise, and median censored candidate events must fall."
        ),
        "checks": checks,
    }


def run_experiment(args: argparse.Namespace) -> dict:
    historical_argument = args.historical_adapter
    historical = Path(args.historical_adapter).resolve()
    if not historical.is_file():
        raise FileNotFoundError(f"historical adapter not found: {historical}")
    defects: dict[str, AdapterFactory] = {
        "historical-orderbook-rs-issue-88": lambda config: ExternalProcessAdapter(
            [str(historical)], config
        ),
        "injected-reduce-requeues": ReduceRequeuesAdapter,
        "injected-replace-keeps-priority": SamePriceReplaceKeepsPriorityAdapter,
    }
    results: list[TrialResult] = []
    for defect_index, (defect, factory) in enumerate(defects.items(), 1):
        for trial in range(1, args.trials + 1):
            trial_seed = args.seed + defect_index * 1_000_003 + trial * 97
            for strategy in ("deterministic", "guided"):
                results.append(
                    run_trial(
                        defect,
                        strategy,
                        trial,
                        trial_seed,
                        factory,
                        budget=args.budget,
                        event_count=args.events,
                        pool_size=args.pool_size,
                    )
                )

    summaries = {
        defect: {
            strategy: summarize(
                [
                    result
                    for result in results
                    if result.defect == defect and result.strategy == strategy
                ],
                args.budget,
                args.events,
            )
            for strategy in ("deterministic", "guided")
        }
        for defect in defects
    }
    return {
        "artifact_type": "tracebook.experiment.guided-diversity",
        "artifact_version": 1,
        "method": {
            "profile": "fifo-limit-v1",
            "built_in_priority_probes": False,
            "candidate_run_budget_per_trial": args.budget,
            "events_per_candidate_run": args.events,
            "guided_mutation_pool": args.pool_size,
            "trials_per_defect": args.trials,
            "base_seed": args.seed,
            "candidate_run_budget_equal": True,
            "reference_side_work_equal": False,
            "runtime_timings_in_artifact": False,
            "guidance": (
                "Structure-preserving suffix regeneration retained by novel black-box "
                "semantic transition tuples. Conformant prefixes make the relative "
                "reference/candidate tuple identical until the first divergence."
            ),
        },
        "historical_adapter": historical_argument,
        "defects": DEFECT_METADATA,
        "summaries": summaries,
        "decision": guidance_decision(summaries),
        "trials": [asdict(result) for result in results],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--historical-adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--budget", type=int, default=40)
    parser.add_argument("--events", type=int, default=120)
    parser.add_argument("--pool-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260716)
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if min(args.trials, args.budget, args.events, args.pool_size) <= 0:
        raise SystemExit("trials, budget, events, and pool-size must be positive")
    payload = run_experiment(args)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Experiment report written: {destination}")
    print(f"Ship guided exploration: {payload['decision']['ship_guided_exploration']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
