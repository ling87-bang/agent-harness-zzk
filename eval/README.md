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
