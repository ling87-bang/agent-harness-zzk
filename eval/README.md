# Golden evaluation cases

## Live eval (requires API key)

```bash
zzk eval --cases-file eval/cases.json --report-out eval/report.json
```

After a failure, inspect the matching trace:

```bash
zzk trace list
zzk trace show <run_id>
```

`cases.json` includes answer checks and tool-path smoke cases (`expected_tools`).

## CI / offline eval

`cases.ci.json` is executed in pytest with a mocked provider (`tests/test_ci_eval_report.py`) and writes `eval/report-ci.json` for GitHub Actions artifacts.

Deterministic regression: `pytest` (full suite). Live eval may vary with the model.

## Report metrics

| Field | Meaning |
|-------|---------|
| `task_success_rate` | `passed / total` |
| `tool_error_rate` | `error_cases / total` (execution errors; excludes `parse_failed`) |
| `parse_failed_rate` | `degraded_cases / total` (`observed_error_code == parse_failed`) |
| `avg_wall_clock_ms` | Mean end-to-end wall time per case |
| `avg_latency_ms` | Alias of `avg_wall_clock_ms` |
| `avg_step_latency_ms` | Mean sum of trace `step.latency_ms` per case |
