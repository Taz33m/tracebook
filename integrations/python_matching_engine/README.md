# PythonMatchingEngine Integration

This adapter runs the MIT-licensed
[PythonMatchingEngine](https://github.com/Surbeivol/PythonMatchingEngine) as a
Tracebook candidate process. The integration is pinned to upstream commit
`f94150294a85d7b415ca4518590b5a661d6f9958` so a future upstream change cannot
silently alter the comparison.

## Supported Native Surface

- FIFO limit-order matching
- cancellation
- priority-preserving quantity reduction
- cancel-and-new replacement
- clear
- multiple independent symbols
- ordered source-ID trades and full queue snapshots

The adapter intentionally does not emulate pro-rata allocation, market/IOC/FOK
instructions, self-trade prevention, or Tracebook's fixed tick grid. Running
those cases should produce a conformance divergence. A difference means the two
configured contracts disagree; it does not by itself mean either project is
incorrect.

Queue snapshots traverse PythonMatchingEngine's linked price levels and orders,
which are private upstream attributes. The scheduled integration workflow is
therefore pinned and acts as the compatibility alarm for that boundary.

## Run It

```bash
git clone https://github.com/Surbeivol/PythonMatchingEngine.git /tmp/PythonMatchingEngine
git -C /tmp/PythonMatchingEngine checkout f94150294a85d7b415ca4518590b5a661d6f9958
python -m pip install -e . "pandas>=2.3.3" "PyYAML>=6.0.2"

export PYTHON_MATCHING_ENGINE_PATH=/tmp/PythonMatchingEngine
tracebook-conformance run \
  integrations/python_matching_engine/fifo-compatible.jsonl \
  --output /tmp/python-matching-engine-report.json \
  --timeout 20 \
  --candidate python integrations/python_matching_engine/adapter.py
```

The command exits `0` and the report contains:

```json
{
  "candidate_engine": {
    "language": "Python",
    "name": "PythonMatchingEngine FIFO/LIMIT",
    "version": "f94150294a85"
  },
  "conformant": true,
  "compared_events": 13
}
```

Profile the complete standard contract separately:

```bash
tracebook-conformance sample /tmp/tracebook-conformance-v1
tracebook-conformance suite \
  /tmp/tracebook-conformance-v1 \
  --output /tmp/python-matching-engine-suite.json \
  --timeout 20 \
  --candidate python integrations/python_matching_engine/adapter.py
```

This command intentionally exits `1`. The pinned profile is:

| Result | Cases |
| --- | --- |
| Native agreement | `fifo-lifecycle`, `deep-cancellation` |
| Expected contract difference | `order-instructions`, `stp-cancel-resting`, `stp-cancel-incoming`, `pro-rata-allocation`, `multi-symbol`, `tick-grid` |

The profile is an interoperability map, not a quality ranking. Run the
compatible trace as the integration's pass/fail gate and retain the complete
suite report as evidence of the unsupported boundary.

The upstream package metadata pins dependencies from its original Python 3.6
environment. This integration deliberately avoids installing those pins and
uses current `pandas`, `numpy`, and `PyYAML` versions instead. It is an optional
interoperability check, not a Tracebook runtime dependency. The explicit
20-second timeout accommodates a cold import of that scientific Python stack;
it does not change Tracebook's five-second default for other adapters.
