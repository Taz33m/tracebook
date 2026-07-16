import json
from pathlib import Path

from experiments.guided_diversity import (
    DEFECT_METADATA,
    generate_unprobed_trace,
    guidance_decision,
    regenerate_suffix,
    semantic_transitions,
)
from tracebook.conformance import ReferenceEngineAdapter, run_conformance

ROOT = Path(__file__).parents[1]


def test_research_generator_is_deterministic_and_has_no_forced_probe():
    first = generate_unprobed_trace(42, 40)
    second = generate_unprobed_trace(42, 40)

    assert first == second
    assert len(first) == 40
    assert all("PROBE" not in event.symbol for event in first)
    assert semantic_transitions(first)


def test_suffix_regeneration_preserves_a_valid_reference_trace():
    source = generate_unprobed_trace(42, 40)
    mutated = regenerate_suffix(source, 99)
    report = run_conformance(mutated, ReferenceEngineAdapter)

    assert len(mutated) == len(source)
    assert mutated != source
    assert report.conformant is True


def test_guidance_gate_requires_every_held_out_defect_to_improve():
    summaries = {
        "improved": {
            "deterministic": {
                "discovery_rate": 1.0,
                "median_candidate_runs_censored": 4,
                "median_candidate_events_censored": 300,
            },
            "guided": {
                "discovery_rate": 1.0,
                "median_candidate_runs_censored": 3,
                "median_candidate_events_censored": 200,
            },
        },
        "regressed": {
            "deterministic": {
                "discovery_rate": 1.0,
                "median_candidate_runs_censored": 2,
                "median_candidate_events_censored": 100,
            },
            "guided": {
                "discovery_rate": 1.0,
                "median_candidate_runs_censored": 3,
                "median_candidate_events_censored": 90,
            },
        },
    }

    decision = guidance_decision(summaries)

    assert decision["checks"]["improved"]["passed"] is True
    assert decision["checks"]["regressed"]["passed"] is False
    assert decision["ship_guided_exploration"] is False


def test_frozen_held_out_result_rejects_guidance_without_local_paths():
    payload = json.loads(
        (ROOT / "experiments" / "results" / "guided-diversity-v1.json").read_text()
    )

    assert payload["defects"] == DEFECT_METADATA
    assert payload["decision"]["ship_guided_exploration"] is False
    assert (
        payload["summaries"]["injected-reduce-requeues"]["guided"][
            "median_candidate_events_censored"
        ]
        == 88.0
    )
    assert (
        payload["summaries"]["historical-orderbook-rs-issue-88"]["guided"][
            "median_candidate_events_censored"
        ]
        == 200.5
    )
    assert "/Users/" not in json.dumps(payload)
