"""Summarize the independently checked cases from the orderbook-rs CI workflow."""

import argparse
import json
from pathlib import Path


def scorecard(root: Path) -> dict:
    def load(path: str) -> dict:
        return json.loads((root / path).read_text(encoding="utf-8"))

    rows = []
    for name, origin, case, control in (
        ("Flash workload", "historical", "flash-handoff", "flash-handoff/fixed.json"),
        (
            "Seed 42 partial fill",
            "historical",
            "historical-issue-88-corpus/failure-7dd023c684cdb2d0fc0e",
            "historical-issue-88-fixed.json",
        ),
        (
            "Seed 42 replacement",
            "injected",
            "queue-priority-corpus/failure-bc8b19d3e0e3441a98db",
            "queue-priority-regression.json",
        ),
    ):
        if not (root / control).exists():
            rows.append(
                {"case": name, "origin": origin, "status": "incomplete", "missing": control}
            )
            continue
        if case == "flash-handoff":
            first = load(f"{case}/divergence.json")["first_divergent_sequence"] + 1
            budget = 200
            reduced = load(f"{case}/minimization.json")
        else:
            campaign = load(f"{case}/campaign.json")
            first = campaign["failure"]["original_divergence_event"]
            budget = campaign["max_minimize_runs"]
            reduced = load(f"{case}/failure/minimization.json")
        fixed = load(control)
        report = reduced["conformance_report"]
        if (
            fixed["trace"]["sha256"] != reduced["minimized_trace_sha256"]
            or fixed["compared_events"] != reduced["minimized_event_count"]
        ):
            raise ValueError(f"{name}: control did not cover the reduced trace")
        rows.append(
            {
                "case": name,
                "origin": origin,
                "status": "measured",
                "candidate": report["candidate_engine"],
                "first_divergent_event": first,
                "reduced_events": reduced["minimized_event_count"],
                "minimizer_runs": reduced["runs"],
                "minimizer_budget": budget,
                "one_minimal": reduced["one_minimal"],
                "budget_exhausted": reduced["budget_exhausted"],
                "fixed_control_conformant": fixed["conformant"],
                "trace_sha256": fixed["trace"]["sha256"],
            }
        )
    return {"schema_version": 1, "cases": rows}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", type=Path)
    print(json.dumps(scorecard(parser.parse_args().reports), indent=2))
