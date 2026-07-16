import hashlib
import json
from pathlib import Path

from experiments.guided_diversity import (
    DEFECT_METADATA,
    ReduceRequeuesAdapter,
    generate_unprobed_trace,
    guidance_decision,
    regenerate_suffix,
    run_trial,
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


def test_trial_results_are_byte_stable_inputs_without_runtime_timings():
    arguments = {
        "defect": "injected-reduce-requeues",
        "strategy": "deterministic",
        "trial": 1,
        "seed": 42,
        "candidate_factory": ReduceRequeuesAdapter,
        "budget": 2,
        "event_count": 40,
        "pool_size": 2,
    }

    assert run_trial(**arguments) == run_trial(**arguments)


def test_frozen_held_out_result_rejects_guidance_without_local_paths():
    artifact = ROOT / "experiments" / "results" / "guided-diversity-v1.json"
    payload = json.loads(artifact.read_text())

    assert hashlib.sha256(artifact.read_bytes()).hexdigest() == (
        "e798d014dc8ea63cd7714aedf838678d1249354ba9e7adb3f965def9289c9a6c"
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
    assert "wall_seconds" not in json.dumps(payload)
    assert payload["method"]["runtime_timings_in_artifact"] is False
