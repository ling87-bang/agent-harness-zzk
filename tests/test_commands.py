import json
from pathlib import Path

import pytest

from harness.cli.commands import (
    _build_eval_report,
    _is_case_passed,
    _load_eval_cases,
    _read_trace_metrics,
    _summarize_eval,
    _write_eval_report,
    EvalCase,
    EvalCaseResult,
    run_chat_async,
    run_eval_async,
    run_query_async,
)
from harness.engine.trace import TraceRecorder
from harness.state import TraceStep
from harness.config import Settings, get_settings
from harness.engine.context import ConversationManager
from harness.state import StreamEvent


def _patch_settings_without_api_key(monkeypatch) -> None:
    get_settings.cache_clear()
    monkeypatch.setattr(
        "harness.cli.commands.get_settings",
        lambda: Settings(deepseek_api_key=""),
    )


@pytest.mark.asyncio()
async def test_run_query_async_without_api_key(monkeypatch) -> None:
    _patch_settings_without_api_key(monkeypatch)

    code = await run_query_async("hello")
    assert code == 1


@pytest.mark.asyncio()
async def test_run_query_async_returns_nonzero_on_degraded(monkeypatch) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(event_type="token", content="hello", error_code="parse_failed")
        yield StreamEvent(event_type="done", error_code="parse_failed")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    code = await run_query_async("hello")
    assert code == 1


@pytest.mark.asyncio()
async def test_run_chat_async_exits_cleanly(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(
        *,
        query,
        provider,
        trace,
        registry=None,
        messages=None,
        conversation_id="",
        system_prompt=None,
        **kwargs,
    ):
        yield StreamEvent(event_type="token", content="hello", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    prompts = iter(["hi", "exit"])
    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)
    monkeypatch.setattr(
        "harness.cli.commands.ConversationManager",
        lambda: ConversationManager(storage_dir=tmp_path),
    )
    monkeypatch.setattr("harness.cli.commands.typer.prompt", lambda _: next(prompts))

    code = await run_chat_async("conv-test")
    assert code == 0


@pytest.mark.asyncio()
async def test_run_eval_async_without_api_key(monkeypatch, tmp_path) -> None:
    _patch_settings_without_api_key(monkeypatch)
    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([{"id": "c1", "query": "hello"}]), encoding="utf-8")

    code = await run_eval_async(cases_file)
    assert code == 1


@pytest.mark.asyncio()
async def test_run_eval_async_with_mixed_results(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        if query == "ok":
            yield StreamEvent(event_type="token", content="done", metadata={"step_type": "answer"})
            yield StreamEvent(event_type="done")
            return
        if query == "degraded":
            yield StreamEvent(
                event_type="token",
                content="fallback",
                error_code="parse_failed",
                metadata={"step_type": "answer"},
            )
            yield StreamEvent(event_type="done", error_code="parse_failed")
            return
        yield StreamEvent(event_type="token", content="wrong", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps(
            [
                {"id": "pass-case", "query": "ok", "expected_contains": "done"},
                {"id": "degraded-case", "query": "degraded", "expected_error_code": "parse_failed"},
                {"id": "fail-case", "query": "bad", "expected_contains": "done"},
            ]
        ),
        encoding="utf-8",
    )

    code = await run_eval_async(cases_file)
    assert code == 1


@pytest.mark.asyncio()
async def test_run_eval_async_writes_report(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        name = "fake"
        model = "fake-model"

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        trace.append_step(
            TraceStep(
                step=1,
                step_type="llm_call",
                run_id=trace.run_id,
                status="succeeded",
                latency_ms=12.5,
            )
        )
        trace.append_step(
            TraceStep(
                step=1,
                step_type="skill_execution",
                run_id=trace.run_id,
                skill="file_reader",
                status="succeeded",
            )
        )
        trace.finish_run(final_status="success")
        yield StreamEvent(event_type="token", content="done", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps([{"id": "pass-case", "query": "ok", "expected_contains": "done"}]),
        encoding="utf-8",
    )
    report_path = tmp_path / "report.json"
    trace_dir = tmp_path / "traces"

    code = await run_eval_async(cases_file, output_path=report_path, trace_dir=trace_dir)
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["total"] == 1
    assert report["passed"] == 1
    assert report["cases"][0]["run_id"].startswith("run-")
    assert report["cases"][0]["tools_used"] == ["file_reader"]
    assert report["cases"][0]["step_latency_ms"] == 12.5
    assert report["cases"][0]["wall_clock_ms"] >= 0.0
    assert "pass_rate" not in report
    assert "avg_step_latency_ms" in report
    assert "avg_wall_clock_ms" in report


@pytest.mark.asyncio()
async def test_run_eval_async_report_write_failure(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(event_type="token", content="ok", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([{"id": "c1", "query": "q"}]), encoding="utf-8")
    output_path = tmp_path / "blocked" / "report.json"

    def _fail_write(path: Path, report: dict) -> str | None:
        return "eval_report_io_error"

    monkeypatch.setattr("harness.cli.commands._write_eval_report", _fail_write)

    code = await run_eval_async(cases_file, output_path=output_path)
    assert code == 1


def test_read_trace_metrics_sums_latency_and_tools(tmp_path) -> None:
    trace = TraceRecorder(trace_dir=tmp_path)
    trace.start_run(query="q", llm_provider="fake", llm_model="fake-model")
    trace.append_step(
        TraceStep(step=1, step_type="llm_call", run_id=trace.run_id, status="succeeded", latency_ms=10.0)
    )
    trace.append_step(
        TraceStep(
            step=1,
            step_type="skill_execution",
            run_id=trace.run_id,
            skill="knowledge_search",
            status="succeeded",
        )
    )
    trace.finish_run(final_status="success")

    latency_ms, tools = _read_trace_metrics(trace.path)
    assert latency_ms == 10.0
    assert tools == ("knowledge_search",)


def test_build_eval_report_includes_failures() -> None:
    results = [
        EvalCaseResult(
            case_id="ok",
            passed=True,
            observed_error_code=None,
            answer="done",
            expected_contains="done",
            expected_error_code=None,
            expected_tools=(),
            run_id="run-abc",
            trace_path="/tmp/run-abc.jsonl",
            step_latency_ms=5.0,
            wall_clock_ms=100.0,
        ),
        EvalCaseResult(
            case_id="bad",
            passed=False,
            observed_error_code="parse_failed",
            answer="x",
            expected_contains="done",
            expected_error_code=None,
            expected_tools=(),
            run_id="run-def",
            trace_path="/tmp/run-def.jsonl",
            step_latency_ms=0.0,
            wall_clock_ms=50.0,
        ),
    ]
    summary = _summarize_eval(results)
    report = _build_eval_report(
        cases_file=Path("eval/cases.json"),
        summary=summary,
        results=results,
    )
    assert report["failed"] == 1
    assert len(report["failures"]) == 1
    assert report["failures"][0]["case_id"] == "bad"
    assert report["task_success_rate"] == 0.5
    assert report["tool_error_rate"] == 0.0
    assert report["parse_failed_rate"] == 0.5
    assert report["avg_latency_ms"] == report["avg_wall_clock_ms"]


def test_write_eval_report_persists_json(tmp_path) -> None:
    output_path = tmp_path / "nested" / "report.json"
    error_code = _write_eval_report(output_path, {"total": 1})
    assert error_code is None
    assert json.loads(output_path.read_text(encoding="utf-8"))["total"] == 1


def test_load_eval_cases_rejects_invalid_payload(tmp_path) -> None:
    bad_cases_file = tmp_path / "bad_cases.json"
    bad_cases_file.write_text('{"query":"not-list"}', encoding="utf-8")

    cases, error_code = _load_eval_cases(bad_cases_file)
    assert not cases
    assert error_code == "eval_cases_parse_failed"


def test_load_eval_cases_handles_io_error(tmp_path) -> None:
    missing_cases_file = tmp_path / "missing.json"

    cases, error_code = _load_eval_cases(missing_cases_file)
    assert not cases
    assert error_code == "eval_cases_io_error"


def test_load_eval_cases_empty_list_returns_eval_cases_empty(tmp_path) -> None:
    cases_file = tmp_path / "empty.json"
    cases_file.write_text("[]", encoding="utf-8")

    cases, error_code = _load_eval_cases(cases_file)
    assert cases == []
    assert error_code == "eval_cases_empty"


def test_load_eval_cases_expected_contains_null_means_empty_string(tmp_path) -> None:
    cases_file = tmp_path / "null_contains.json"
    cases_file.write_text(
        json.dumps([{"id": "c1", "query": "hello", "expected_contains": None}]),
        encoding="utf-8",
    )

    cases, error_code = _load_eval_cases(cases_file)
    assert error_code is None
    assert len(cases) == 1
    assert cases[0].expected_contains == ""


def test_is_case_passed_expected_error_code_takes_precedence_over_contains() -> None:
    case = EvalCase(
        case_id="c",
        query="q",
        expected_contains="never-present",
        expected_error_code="parse_failed",
    )
    assert _is_case_passed(case, "parse_failed", "answer without expected substring", ()) is True
    assert _is_case_passed(case, None, "never-present", ()) is False


def test_is_case_passed_expected_tools() -> None:
    case = EvalCase(
        case_id="tools",
        query="q",
        expected_tools=("file_reader", "knowledge_search"),
    )
    assert _is_case_passed(case, None, "ok", ("file_reader", "knowledge_search")) is True
    assert _is_case_passed(case, None, "ok", ("file_reader",)) is False


def test_load_eval_cases_parses_expected_tools(tmp_path) -> None:
    cases_file = tmp_path / "tools.json"
    cases_file.write_text(
        json.dumps([{"id": "t1", "query": "q", "expected_tools": ["file_reader"]}]),
        encoding="utf-8",
    )
    cases, error_code = _load_eval_cases(cases_file)
    assert error_code is None
    assert cases[0].expected_tools == ("file_reader",)


def test_eval_rate_metrics_empty_summary() -> None:
    from harness.cli.commands import EvalSummary, _eval_rate_metrics

    summary = EvalSummary(
        total_cases=0,
        passed_cases=0,
        failed_cases=0,
        degraded_cases=0,
        error_cases=0,
    )
    assert _eval_rate_metrics(summary) == {"tool_error_rate": 0.0, "parse_failed_rate": 0.0}


def test_summarize_eval_empty_results() -> None:
    summary = _summarize_eval([])
    assert summary.total_cases == 0
    assert summary.passed_cases == 0
    assert summary.failed_cases == 0
    assert summary.degraded_cases == 0
    assert summary.error_cases == 0
    assert summary.task_success_rate == 0.0


@pytest.mark.asyncio()
async def test_run_eval_async_missing_cases_file(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    missing = tmp_path / "does-not-exist.json"
    code = await run_eval_async(missing)
    assert code == 1


@pytest.mark.asyncio()
async def test_run_eval_async_expected_error_code_mismatch_observed_none(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(event_type="token", content="ok", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps([{"id": "expect-degraded", "query": "q", "expected_error_code": "parse_failed"}]),
        encoding="utf-8",
    )

    code = await run_eval_async(cases_file)
    assert code == 1


@pytest.mark.asyncio()
async def test_run_eval_async_expected_error_code_mismatch_wrong_code(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(
            event_type="token",
            content="x",
            error_code="parse_failed",
            metadata={"step_type": "answer"},
        )
        yield StreamEvent(event_type="done", error_code="parse_failed")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps([{"id": "wrong-code", "query": "q", "expected_error_code": "llm_error"}]),
        encoding="utf-8",
    )

    code = await run_eval_async(cases_file)
    assert code == 1


def test_summarize_eval_aggregates_metrics() -> None:
    summary = _summarize_eval(
        [
            EvalCaseResult(
                case_id="c1",
                passed=True,
                observed_error_code=None,
                answer="ok",
                expected_contains="ok",
                expected_error_code=None,
                expected_tools=(),
                run_id="run-1",
                trace_path="",
                step_latency_ms=0.0,
                wall_clock_ms=0.0,
            ),
            EvalCaseResult(
                case_id="c2",
                passed=False,
                observed_error_code="parse_failed",
                answer="fallback",
                expected_contains="",
                expected_error_code=None,
                expected_tools=(),
                run_id="run-2",
                trace_path="",
                step_latency_ms=0.0,
                wall_clock_ms=0.0,
            ),
            EvalCaseResult(
                case_id="c3",
                passed=False,
                observed_error_code="llm_error",
                answer="",
                expected_contains="",
                expected_error_code=None,
                expected_tools=(),
                run_id="run-3",
                trace_path="",
                step_latency_ms=0.0,
                wall_clock_ms=0.0,
            ),
        ]
    )
    assert summary.total_cases == 3
    assert summary.passed_cases == 1
    assert summary.failed_cases == 2
    assert summary.degraded_cases == 1
    assert summary.error_cases == 1
    assert summary.task_success_rate == pytest.approx(1 / 3)


@pytest.mark.asyncio()
async def test_run_eval_async_compare_prompts(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(event_type="token", content="ok", metadata={"step_type": "answer"})
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(json.dumps([{"id": "c1", "query": "q", "expected_contains": "ok"}]), encoding="utf-8")
    report_path = tmp_path / "compare.json"

    code = await run_eval_async(
        cases_file,
        output_path=report_path,
        compare_prompt_versions=("v1", "v2"),
    )
    assert code == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "prompt_comparison" in report
    assert "v1" in report["prompt_comparison"]
    assert "v2" in report["prompt_comparison"]


@pytest.mark.asyncio()
async def test_run_eval_async_workers_parallel(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None, system_prompt=None, **kwargs):
        yield StreamEvent(
            event_type="token",
            content=f"answer-{query}",
            metadata={"step_type": "answer"},
        )
        yield StreamEvent(event_type="done")

    monkeypatch.setattr("harness.cli.commands.DeepSeekProvider", lambda settings: _FakeProvider())
    monkeypatch.setattr("harness.cli.commands.run_single_turn", _fake_stream)

    cases_file = tmp_path / "cases.json"
    cases_file.write_text(
        json.dumps(
            [
                {"id": "c1", "query": "one", "expected_contains": "answer-one"},
                {"id": "c2", "query": "two", "expected_contains": "answer-two"},
            ]
        ),
        encoding="utf-8",
    )

    code = await run_eval_async(cases_file, workers=2)
    assert code == 0


def test_resolve_runtime_includes_user_skill_tools() -> None:
    from harness.cli.commands import _resolve_runtime
    from harness.config import Settings

    repo_root = Path(__file__).resolve().parents[1]
    settings = Settings(deepseek_api_key="k", enable_user_skills=True)
    registry, prompt = _resolve_runtime(
        settings,
        prompt_version="v2",
        enable_user_skills=True,
        user_skill_dirs=(repo_root / "examples" / "skills",),
    )
    assert registry.get("weather") is not None
    assert "weather" in prompt
