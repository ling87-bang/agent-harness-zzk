"""CLI command implementations."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path

import typer

from harness.cli.formatter import render_stream_event
from harness.config import Settings, get_settings
from harness.engine.context import ConversationManager
from harness.engine.loop import run_single_turn
from harness.engine.trace import TraceRecorder
from harness.llm.deepseek import DeepSeekProvider
from harness.llm.prompts import PROMPT_VERSIONS, get_system_prompt
from harness.skills.registry import SkillRegistry
from harness.state import Message


@dataclass(frozen=True, slots=True)
class EvalCase:
    """A single golden evaluation case."""

    case_id: str
    query: str
    expected_contains: str = ""
    expected_error_code: str | None = None
    expected_tools: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalCaseResult:
    """Execution outcome for one evaluation case."""

    case_id: str
    passed: bool
    observed_error_code: str | None
    answer: str
    expected_contains: str
    expected_error_code: str | None
    expected_tools: tuple[str, ...]
    run_id: str
    trace_path: str
    step_latency_ms: float
    wall_clock_ms: float
    tools_used: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EvalSummary:
    """Aggregated metrics for one eval run."""

    total_cases: int
    passed_cases: int
    failed_cases: int
    degraded_cases: int
    error_cases: int

    @property
    def task_success_rate(self) -> float:
        """Fraction of cases passed, in [0.0, 1.0]."""

        if self.total_cases == 0:
            return 0.0
        return self.passed_cases / self.total_cases


def _load_eval_cases(cases_file: Path) -> tuple[list[EvalCase], str | None]:
    try:
        payload = json.loads(cases_file.read_text(encoding="utf-8"))
    except OSError:
        return ([], "eval_cases_io_error")
    except json.JSONDecodeError:
        return ([], "eval_cases_parse_failed")

    if not isinstance(payload, list):
        return ([], "eval_cases_parse_failed")

    cases: list[EvalCase] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            return ([], "eval_cases_parse_failed")
        query_value = item.get("query")
        if not isinstance(query_value, str) or not query_value.strip():
            return ([], "eval_cases_parse_failed")
        case_id_value = item.get("id")
        expected_contains_value = item.get("expected_contains")
        if expected_contains_value is None:
            expected_contains_value = ""
        elif not isinstance(expected_contains_value, str):
            return ([], "eval_cases_parse_failed")
        expected_error_code_value = item.get("expected_error_code")
        if case_id_value is not None and not isinstance(case_id_value, str):
            return ([], "eval_cases_parse_failed")
        if expected_error_code_value is not None and not isinstance(expected_error_code_value, str):
            return ([], "eval_cases_parse_failed")
        expected_tools_value = item.get("expected_tools")
        if expected_tools_value is None:
            expected_tools_tuple: tuple[str, ...] = ()
        elif not isinstance(expected_tools_value, list) or not all(
            isinstance(tool, str) for tool in expected_tools_value
        ):
            return ([], "eval_cases_parse_failed")
        else:
            expected_tools_tuple = tuple(expected_tools_value)
        cases.append(
            EvalCase(
                case_id=case_id_value or f"case-{index + 1}",
                query=query_value,
                expected_contains=expected_contains_value,
                expected_error_code=expected_error_code_value,
                expected_tools=expected_tools_tuple,
            )
        )
    if not cases:
        return ([], "eval_cases_empty")
    return (cases, None)


def _is_case_passed(
    case: EvalCase,
    observed_error_code: str | None,
    answer: str,
    tools_used: tuple[str, ...],
) -> bool:
    if case.expected_error_code is not None:
        return observed_error_code == case.expected_error_code
    if case.expected_tools and not all(tool in tools_used for tool in case.expected_tools):
        return False
    if observed_error_code is not None:
        return False
    if case.expected_contains:
        return case.expected_contains in answer
    return bool(answer.strip())


def _read_trace_metrics(trace_path: Path) -> tuple[float, tuple[str, ...]]:
    """Sum per-step latency_ms from trace steps (not end-to-end wall clock)."""

    if not trace_path.is_file():
        return (0.0, ())
    total_latency_ms = 0.0
    tools: list[str] = []
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") != "step":
                continue
            latency_value = row.get("latency_ms")
            if isinstance(latency_value, (int, float)):
                total_latency_ms += float(latency_value)
            if row.get("step_type") == "skill_execution":
                skill_name = row.get("skill")
                if isinstance(skill_name, str) and skill_name:
                    tools.append(skill_name)
    except (OSError, json.JSONDecodeError):
        return (0.0, ())
    return (total_latency_ms, tuple(tools))


def _eval_rate_metrics(summary: EvalSummary) -> dict[str, float]:
    """Derived eval rates (tool_error excludes parse_failed degradation)."""
    total = summary.total_cases
    if total == 0:
        return {"tool_error_rate": 0.0, "parse_failed_rate": 0.0}
    return {
        "tool_error_rate": round(summary.error_cases / total, 4),
        "parse_failed_rate": round(summary.degraded_cases / total, 4),
    }


def _build_eval_report(
    *,
    cases_file: Path,
    summary: EvalSummary,
    results: list[EvalCaseResult],
) -> dict[str, object]:
    step_latencies = [result.step_latency_ms for result in results if result.step_latency_ms > 0]
    avg_step_latency_ms = sum(step_latencies) / len(step_latencies) if step_latencies else 0.0
    wall_clocks = [result.wall_clock_ms for result in results if result.wall_clock_ms > 0]
    avg_wall_clock_ms = sum(wall_clocks) / len(wall_clocks) if wall_clocks else 0.0
    failures = [
        {
            "case_id": result.case_id,
            "error_code": result.observed_error_code,
            "run_id": result.run_id,
            "trace_path": result.trace_path,
        }
        for result in results
        if not result.passed
    ]
    rate_metrics = _eval_rate_metrics(summary)
    avg_wall_clock_rounded = round(avg_wall_clock_ms, 2)
    return {
        "cases_file": str(cases_file),
        "total": summary.total_cases,
        "passed": summary.passed_cases,
        "failed": summary.failed_cases,
        "degraded_cases": summary.degraded_cases,
        "error_cases": summary.error_cases,
        "task_success_rate": round(summary.task_success_rate, 4),
        "tool_error_rate": rate_metrics["tool_error_rate"],
        "parse_failed_rate": rate_metrics["parse_failed_rate"],
        "avg_step_latency_ms": round(avg_step_latency_ms, 2),
        "avg_wall_clock_ms": avg_wall_clock_rounded,
        "avg_latency_ms": avg_wall_clock_rounded,
        "cases": [
            {
                "case_id": result.case_id,
                "passed": result.passed,
                "observed_error_code": result.observed_error_code,
                "expected_contains": result.expected_contains,
                "expected_error_code": result.expected_error_code,
                "expected_tools": list(result.expected_tools),
                "run_id": result.run_id,
                "trace_path": result.trace_path,
                "step_latency_ms": round(result.step_latency_ms, 2),
                "wall_clock_ms": round(result.wall_clock_ms, 2),
                "tools_used": list(result.tools_used),
            }
            for result in results
        ],
        "failures": failures,
    }


def _write_eval_report(output_path: Path, report: dict[str, object]) -> str | None:
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
    except OSError:
        return "eval_report_io_error"
    return None


def _resolve_runtime(
    settings: Settings,
    *,
    prompt_version: str | None,
    enable_user_skills: bool | None,
    user_skill_dirs: tuple[Path, ...],
) -> tuple[SkillRegistry, str]:
    version = (prompt_version or settings.prompt_version).strip().lower()
    use_user_skills = (
        settings.enable_user_skills if enable_user_skills is None else enable_user_skills
    )
    registry = SkillRegistry.from_settings(
        settings,
        enable_user_skills=use_user_skills,
        user_skill_dirs=user_skill_dirs,
    )
    system_prompt = get_system_prompt(
        version,
        extra_tools=registry.extra_tool_descriptions(),
    )
    return registry, system_prompt


def _count_trace_steps(trace_path: Path) -> int:
    if not trace_path.is_file():
        return 0
    count = 0
    try:
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("record_type") == "step":
                count += 1
    except (OSError, json.JSONDecodeError):
        return 0
    return count


def _summarize_prompt_version(
    summary: EvalSummary,
    results: list[EvalCaseResult],
) -> dict[str, float | int]:
    step_counts = [_count_trace_steps(Path(result.trace_path)) for result in results]
    avg_steps = sum(step_counts) / len(step_counts) if step_counts else 0.0
    rate_metrics = _eval_rate_metrics(summary)
    wall_clocks = [result.wall_clock_ms for result in results if result.wall_clock_ms > 0]
    avg_wall_clock_ms = sum(wall_clocks) / len(wall_clocks) if wall_clocks else 0.0
    avg_wall_clock_rounded = round(avg_wall_clock_ms, 2)
    return {
        "total": summary.total_cases,
        "passed": summary.passed_cases,
        "failed": summary.failed_cases,
        "degraded_cases": summary.degraded_cases,
        "error_cases": summary.error_cases,
        "task_success_rate": round(summary.task_success_rate, 4),
        "tool_error_rate": rate_metrics["tool_error_rate"],
        "parse_failed_rate": rate_metrics["parse_failed_rate"],
        "avg_steps": round(avg_steps, 2),
        "avg_wall_clock_ms": avg_wall_clock_rounded,
        "avg_latency_ms": avg_wall_clock_rounded,
    }


def _build_prompt_compare_report(
    *,
    cases_file: Path,
    version_metrics: dict[str, dict[str, float | int]],
) -> dict[str, object]:
    return {
        "cases_file": str(cases_file),
        "prompt_comparison": version_metrics,
    }


def _summarize_eval(results: list[EvalCaseResult]) -> EvalSummary:
    total_cases = len(results)
    passed_cases = sum(1 for result in results if result.passed)
    degraded_cases = sum(1 for result in results if result.observed_error_code == "parse_failed")
    error_cases = sum(
        1
        for result in results
        if result.observed_error_code is not None and result.observed_error_code != "parse_failed"
    )
    failed_cases = total_cases - passed_cases
    return EvalSummary(
        total_cases=total_cases,
        passed_cases=passed_cases,
        failed_cases=failed_cases,
        degraded_cases=degraded_cases,
        error_cases=error_cases,
    )


async def run_query_async(
    query: str,
    *,
    prompt_version: str | None = None,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
) -> int:
    """Execute one query and stream terminal output."""

    settings = get_settings()
    if not settings.deepseek_api_key:
        typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
        return 1

    provider = DeepSeekProvider(settings=settings)
    registry, system_prompt = _resolve_runtime(
        settings,
        prompt_version=prompt_version,
        enable_user_skills=enable_user_skills,
        user_skill_dirs=user_skill_dirs,
    )
    trace = TraceRecorder()
    has_error = False
    async for event in run_single_turn(
        query=query,
        provider=provider,
        trace=trace,
        registry=registry,
        system_prompt=system_prompt,
    ):
        render_stream_event(event)
        if event.error_code:
            has_error = True
    return 1 if has_error else 0


async def _run_eval_case(
    case: EvalCase,
    *,
    provider: DeepSeekProvider,
    registry: SkillRegistry,
    system_prompt: str,
    trace_dir: Path | None,
) -> EvalCaseResult:
    trace = TraceRecorder(trace_dir=trace_dir)
    answer_chunks: list[str] = []
    observed_error_code: str | None = None
    started = time.perf_counter()
    async for event in run_single_turn(
        query=case.query,
        provider=provider,
        trace=trace,
        registry=registry,
        system_prompt=system_prompt,
    ):
        if event.event_type == "token" and event.metadata.get("step_type") == "answer":
            answer_chunks.append(event.content)
        if event.error_code is not None:
            observed_error_code = event.error_code
    wall_clock_ms = (time.perf_counter() - started) * 1000
    answer = "".join(answer_chunks).strip()
    step_latency_ms, tools_used = _read_trace_metrics(trace.path)
    passed = _is_case_passed(
        case=case,
        observed_error_code=observed_error_code,
        answer=answer,
        tools_used=tools_used,
    )
    return EvalCaseResult(
        case_id=case.case_id,
        passed=passed,
        observed_error_code=observed_error_code,
        answer=answer,
        expected_contains=case.expected_contains,
        expected_error_code=case.expected_error_code,
        expected_tools=case.expected_tools,
        run_id=trace.run_id,
        trace_path=str(trace.path),
        step_latency_ms=step_latency_ms,
        wall_clock_ms=wall_clock_ms,
        tools_used=tools_used,
    )


async def _run_eval_cases(
    cases: list[EvalCase],
    *,
    provider: DeepSeekProvider,
    registry: SkillRegistry,
    system_prompt: str,
    trace_dir: Path | None,
    workers: int,
) -> list[EvalCaseResult]:
    bounded_workers = max(1, workers)
    if bounded_workers == 1:
        results: list[EvalCaseResult] = []
        for case in cases:
            results.append(
                await _run_eval_case(
                    case,
                    provider=provider,
                    registry=registry,
                    system_prompt=system_prompt,
                    trace_dir=trace_dir,
                )
            )
        return results

    semaphore = asyncio.Semaphore(bounded_workers)

    async def _guarded(case: EvalCase) -> EvalCaseResult:
        async with semaphore:
            return await _run_eval_case(
                case,
                provider=provider,
                registry=registry,
                system_prompt=system_prompt,
                trace_dir=trace_dir,
            )

    return list(await asyncio.gather(*[_guarded(case) for case in cases]))


def _echo_eval_case_results(results: list[EvalCaseResult]) -> None:
    for result in results:
        status = "PASS" if result.passed else "FAIL"
        typer.echo(
            f"[eval:{status}] {result.case_id} error_code={result.observed_error_code or 'none'} "
            f"run_id={result.run_id}"
        )
        if not result.passed:
            typer.echo(f"[eval:hint] zzk trace show {result.run_id}")


async def run_eval_async(
    cases_file: Path,
    output_path: Path | None = None,
    trace_dir: Path | None = None,
    *,
    prompt_version: str | None = None,
    compare_prompt_versions: tuple[str, ...] | None = None,
    workers: int = 1,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
) -> int:
    """Execute golden cases and print aggregated eval metrics."""

    settings = get_settings()
    if not settings.deepseek_api_key:
        typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
        return 1

    cases, load_error_code = _load_eval_cases(cases_file)
    if load_error_code is not None:
        typer.echo(f"[eval:error:{load_error_code}] failed to load cases: {cases_file}")
        return 1

    provider = DeepSeekProvider(settings=settings)

    if compare_prompt_versions:
        versions = tuple(version.strip().lower() for version in compare_prompt_versions if version.strip())
        if not versions:
            typer.echo("[eval:error:invalid_prompt_versions] --compare-prompts requires at least one version")
            return 1
        for version in versions:
            if version not in PROMPT_VERSIONS:
                supported = ", ".join(PROMPT_VERSIONS)
                typer.echo(f"[eval:error:invalid_prompt_version] {version!r} not in {supported}")
                return 1

        version_metrics: dict[str, dict[str, float | int]] = {}
        any_failed = False
        for version in versions:
            registry, system_prompt = _resolve_runtime(
                settings,
                prompt_version=version,
                enable_user_skills=enable_user_skills,
                user_skill_dirs=user_skill_dirs,
            )
            typer.echo(f"[eval:prompt] version={version}")
            results = await _run_eval_cases(
                cases,
                provider=provider,
                registry=registry,
                system_prompt=system_prompt,
                trace_dir=trace_dir,
                workers=workers,
            )
            _echo_eval_case_results(results)
            summary = _summarize_eval(results)
            version_metrics[version] = _summarize_prompt_version(summary, results)
            typer.echo(
                f"[eval:summary:{version}] "
                f"task_success_rate={summary.task_success_rate:.4f} "
                f"parse_failed_rate={version_metrics[version]['parse_failed_rate']:.4f} "
                f"avg_steps={version_metrics[version]['avg_steps']:.2f}"
            )
            if summary.failed_cases:
                any_failed = True

        if output_path is not None:
            report = _build_prompt_compare_report(
                cases_file=cases_file,
                version_metrics=version_metrics,
            )
            report_error_code = _write_eval_report(output_path, report)
            if report_error_code is not None:
                typer.echo(f"[eval:error:{report_error_code}] failed to write report: {output_path}")
                return 1
            typer.echo(f"[eval:report] {output_path}")

        return 1 if any_failed else 0

    registry, system_prompt = _resolve_runtime(
        settings,
        prompt_version=prompt_version,
        enable_user_skills=enable_user_skills,
        user_skill_dirs=user_skill_dirs,
    )
    active_version = (prompt_version or settings.prompt_version).strip().lower()
    typer.echo(f"[eval:prompt] version={active_version} workers={max(1, workers)}")

    results = await _run_eval_cases(
        cases,
        provider=provider,
        registry=registry,
        system_prompt=system_prompt,
        trace_dir=trace_dir,
        workers=workers,
    )
    _echo_eval_case_results(results)

    summary = _summarize_eval(results)
    typer.echo(
        "[eval:summary] "
        f"total={summary.total_cases} passed={summary.passed_cases} failed={summary.failed_cases} "
        f"degraded={summary.degraded_cases} error={summary.error_cases} "
        f"task_success_rate={summary.task_success_rate:.4f}"
    )

    if output_path is not None:
        report = _build_eval_report(cases_file=cases_file, summary=summary, results=results)
        report["prompt_version"] = active_version
        report_error_code = _write_eval_report(output_path, report)
        if report_error_code is not None:
            typer.echo(f"[eval:error:{report_error_code}] failed to write report: {output_path}")
            return 1
        typer.echo(f"[eval:report] {output_path}")

    return 0 if summary.failed_cases == 0 else 1


async def run_chat_async(
    conversation_id: str | None = None,
    *,
    prompt_version: str | None = None,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
) -> int:
    """Run interactive multi-turn chat session."""

    settings = get_settings()
    if not settings.deepseek_api_key:
        typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
        return 1

    provider = DeepSeekProvider(settings=settings)
    registry, system_prompt = _resolve_runtime(
        settings,
        prompt_version=prompt_version,
        enable_user_skills=enable_user_skills,
        user_skill_dirs=user_skill_dirs,
    )
    manager = ConversationManager()
    active_conversation_id = conversation_id or manager.new_conversation_id()
    history = manager.load_history(active_conversation_id)
    history = await manager.compress_history_async(
        history,
        mode=settings.memory_compress_mode,
        provider=provider if settings.memory_compress_mode == "llm" else None,
        summary_max_tokens=settings.memory_summary_max_tokens,
    )

    typer.echo(f"[chat] conversation_id={active_conversation_id}")
    has_error = False

    while True:
        user_input = typer.prompt("you")
        if user_input.strip().lower() in {"exit", "quit", ":q"}:
            break

        turn_messages = [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_input),
        ]
        trace = TraceRecorder()
        answer_chunks: list[str] = []
        async for event in run_single_turn(
            query=user_input,
            provider=provider,
            trace=trace,
            registry=registry,
            messages=turn_messages,
            conversation_id=active_conversation_id,
            system_prompt=system_prompt,
        ):
            render_stream_event(event)
            if event.error_code:
                has_error = True
            if event.event_type == "token" and event.metadata.get("step_type") == "answer":
                answer_chunks.append(event.content)

        assistant_text = "".join(answer_chunks).strip()
        updated_history = [*history, Message(role="user", content=user_input)]
        if assistant_text:
            updated_history = [*updated_history, Message(role="assistant", content=assistant_text)]
        history = await manager.compress_history_async(
            updated_history,
            mode=settings.memory_compress_mode,
            provider=provider if settings.memory_compress_mode == "llm" else None,
            summary_max_tokens=settings.memory_summary_max_tokens,
        )
        if not manager.save_history(active_conversation_id, history):
            typer.echo("[warn] failed to persist conversation history.")

    return 1 if has_error else 0


def run_query(
    query: str,
    *,
    prompt_version: str | None = None,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
) -> int:
    """Sync wrapper for Typer commands."""

    return asyncio.run(
        run_query_async(
            query,
            prompt_version=prompt_version,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=user_skill_dirs,
        )
    )


def run_chat(
    conversation_id: str | None = None,
    *,
    prompt_version: str | None = None,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
) -> int:
    """Sync wrapper for interactive chat."""

    return asyncio.run(
        run_chat_async(
            conversation_id=conversation_id,
            prompt_version=prompt_version,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=user_skill_dirs,
        )
    )


def run_eval(
    cases_file: Path,
    output_path: Path | None = None,
    *,
    prompt_version: str | None = None,
    compare_prompt_versions: tuple[str, ...] | None = None,
    workers: int = 1,
    enable_user_skills: bool | None = None,
    user_skill_dirs: tuple[Path, ...] = (),
    trace_dir: Path | None = None,
) -> int:
    """Sync wrapper for eval command."""

    return asyncio.run(
        run_eval_async(
            cases_file=cases_file,
            output_path=output_path,
            trace_dir=trace_dir,
            prompt_version=prompt_version,
            compare_prompt_versions=compare_prompt_versions,
            workers=workers,
            enable_user_skills=enable_user_skills,
            user_skill_dirs=user_skill_dirs,
        )
    )
