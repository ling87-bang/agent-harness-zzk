"""Generate deterministic eval report for CI artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harness.cli.commands import run_eval_async
from harness.config import Settings, get_settings
from harness.state import StreamEvent, TraceStep

REPO_ROOT = Path(__file__).resolve().parents[1]
CI_CASES_FILE = REPO_ROOT / "eval" / "cases.ci.json"
CI_REPORT_FILE = REPO_ROOT / "eval" / "report-ci.json"


@pytest.mark.asyncio()
async def test_generate_ci_eval_report(monkeypatch, tmp_path) -> None:
    get_settings.cache_clear()
    monkeypatch.setattr(
        "harness.cli.commands.get_settings",
        lambda: Settings(deepseek_api_key="ci-key"),
    )

    class _FakeProvider:
        name = "fake"
        model = "fake-model"

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        if query == "__ci_file_reader__":
            trace.append_step(
                TraceStep(
                    step=1,
                    step_type="skill_execution",
                    run_id=trace.run_id,
                    skill="file_reader",
                    status="succeeded",
                )
            )
            yield StreamEvent(
                event_type="token",
                content="file content ready",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done")
            return
        if query == "__ci_knowledge_search__":
            trace.append_step(
                TraceStep(
                    step=1,
                    step_type="skill_execution",
                    run_id=trace.run_id,
                    skill="knowledge_search",
                    status="succeeded",
                )
            )
            yield StreamEvent(
                event_type="token",
                content="knowledge summary",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done")
            return
        if query == "__ci_parse_failed__":
            yield StreamEvent(
                event_type="token",
                content="fallback",
                error_code="parse_failed",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done", error_code="parse_failed")
            return
        if query == "__ci_path_denied__":
            trace.append_step(
                TraceStep(
                    step=1,
                    step_type="skill_execution",
                    run_id=trace.run_id,
                    skill="file_reader",
                    status="failed",
                    error_code="path_denied",
                    error="path denied",
                )
            )
            yield StreamEvent(
                event_type="token",
                content="path access denied",
                error_code="path_denied",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done", error_code="path_denied")
            return
        if query == "__ci_unknown_tool__":
            trace.append_step(
                TraceStep(
                    step=1,
                    step_type="skill_execution",
                    run_id=trace.run_id,
                    skill="missing_tool",
                    status="failed",
                    error_code="unknown_tool",
                    error="unknown tool",
                )
            )
            yield StreamEvent(
                event_type="token",
                content="unknown tool requested",
                error_code="unknown_tool",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done", error_code="unknown_tool")
            return
        yield StreamEvent(event_type="token", content="unexpected", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    code = await run_eval_async(
        CI_CASES_FILE,
        output_path=CI_REPORT_FILE,
        trace_dir=tmp_path / "traces",
    )
    assert code == 0

    report = json.loads(CI_REPORT_FILE.read_text(encoding="utf-8"))
    assert report["total"] == 5
    assert report["passed"] == 5
    assert report["task_success_rate"] == 1.0
    assert all(case["passed"] for case in report["cases"])
