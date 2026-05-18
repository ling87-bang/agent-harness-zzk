"""ReAct loop for phase 2."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import AsyncIterator

from harness.engine.trace import TraceRecorder
from harness.errors import ERROR_LLM_ERROR, ERROR_MAX_STEPS, ERROR_UNKNOWN_TOOL
from harness.llm.base import LLMProvider, parse_llm_response
from harness.llm.prompts import SYSTEM_PROMPT
from harness.skills.registry import SkillRegistry
from harness.state import Message, StreamEvent, TraceStep


async def run_single_turn(
    query: str,
    provider: LLMProvider,
    trace: TraceRecorder,
    registry: SkillRegistry | None = None,
    max_steps: int = 4,
    messages: list[Message] | None = None,
    conversation_id: str = "",
) -> AsyncIterator[StreamEvent]:
    """Phase 2 loop: LLM -> optional tool -> observe -> answer."""

    skill_registry = registry or SkillRegistry.with_builtins()
    turn_messages = messages or [
        Message(role="system", content=SYSTEM_PROMPT),
        Message(role="user", content=query),
    ]
    trace.start_run(
        query=query,
        llm_provider=provider.name,
        llm_model=provider.model,
        conversation_id=conversation_id,
    )

    try:
        for step in range(1, max_steps + 1):
            started = time.perf_counter()
            raw_text = ""
            streamed_answer_content = ""
            async for event in provider.chat_stream(turn_messages):
                if event.event_type == "error":
                    trace.append_step(
                        TraceStep(
                            step=step,
                            step_type="llm_call",
                            run_id=trace.run_id,
                            status="failed",
                            error=event.content,
                            error_code=event.error_code or ERROR_LLM_ERROR,
                        )
                    )
                    trace.finish_run(final_status="error")
                    yield event
                    return
                if event.event_type != "token":
                    continue
                raw_text += event.content
                partial_content = _extract_answer_content_progress(raw_text)
                if partial_content is None:
                    continue
                if len(partial_content) > len(streamed_answer_content):
                    delta = partial_content[len(streamed_answer_content) :]
                    streamed_answer_content = partial_content
                    if delta:
                        yield StreamEvent(
                            event_type="token",
                            content=delta,
                            metadata={"step_type": "answer"},
                        )
            elapsed_ms = (time.perf_counter() - started) * 1000
            parsed = parse_llm_response(raw_text)

            trace.append_step(
                TraceStep(
                    step=step,
                    step_type="llm_call",
                    run_id=trace.run_id,
                    status="succeeded",
                    latency_ms=elapsed_ms,
                    model=provider.model,
                )
            )
            trace.append_step(
                TraceStep(
                    step=step,
                    step_type="parse_result",
                    run_id=trace.run_id,
                    status="failed" if parsed.error_code else "succeeded",
                    error_code=parsed.error_code,
                )
            )

            if parsed.action == "answer":
                if len(parsed.content) > len(streamed_answer_content):
                    remaining = parsed.content[len(streamed_answer_content) :]
                    for chunk in _chunk_text(remaining):
                        yield StreamEvent(
                            event_type="token",
                            content=chunk,
                            error_code=parsed.error_code,
                            metadata={"step_type": "answer"},
                        )
                yield StreamEvent(event_type="done", error_code=parsed.error_code)
                trace.finish_run(final_status="success" if parsed.error_code is None else "error")
                return

            tool_name = parsed.name or ""
            tool_args = dict(parsed.args)
            tool_args.setdefault("cwd", str(Path.cwd()))
            yield StreamEvent(
                event_type="token",
                content=f"\n[tool:{tool_name}] running...\n",
                metadata={"step_type": "tool"},
            )
            skill_result = await skill_registry.execute(tool_name, tool_args)
            trace.append_step(
                TraceStep(
                    step=step,
                    step_type="skill_execution",
                    run_id=trace.run_id,
                    skill=tool_name,
                    status="failed" if skill_result.error_code else "succeeded",
                    error_code=skill_result.error_code,
                    error=skill_result.output if skill_result.error_code else None,
                )
            )

            tool_payload = {
                "tool": tool_name or "unknown",
                "error_code": skill_result.error_code,
                "metadata": skill_result.metadata,
                "output": skill_result.output,
            }
            assistant_payload = {
                "action": "tool",
                "name": tool_name or "unknown",
                "args": tool_args,
                "reasoning": parsed.reasoning,
            }
            turn_messages = [
                *turn_messages,
                Message(role="assistant", content=json.dumps(assistant_payload, ensure_ascii=True)),
                Message(role="tool", content=json.dumps(tool_payload, ensure_ascii=True)),
            ]

            if skill_result.error_code == ERROR_UNKNOWN_TOOL:
                trace.append_step(
                    TraceStep(
                        step=step,
                        step_type="parse_result",
                        run_id=trace.run_id,
                        status="failed",
                        error=f"unknown tool requested: {tool_name}",
                        error_code=ERROR_UNKNOWN_TOOL,
                    )
                )
                yield StreamEvent(
                    event_type="token",
                    content=f"Unknown tool requested: {tool_name}",
                    error_code=ERROR_UNKNOWN_TOOL,
                    metadata={"step_type": "answer"},
                )
                yield StreamEvent(event_type="done", error_code=ERROR_UNKNOWN_TOOL)
                trace.finish_run(final_status="error")
                return

        message = "Reached maximum steps before final answer."
        yield StreamEvent(
            event_type="token",
            content=message,
            error_code=ERROR_MAX_STEPS,
            metadata={"step_type": "answer"},
        )
        yield StreamEvent(event_type="done", error_code=ERROR_MAX_STEPS)
        trace.append_step(
            TraceStep(
                step=max_steps,
                step_type="parse_result",
                run_id=trace.run_id,
                status="failed",
                error=message,
                error_code=ERROR_MAX_STEPS,
            )
        )
        trace.finish_run(final_status="error")
    except Exception as exc:
        trace.append_step(
            TraceStep(
                step=1,
                step_type="llm_call",
                run_id=trace.run_id,
                status="failed",
                error=str(exc),
                error_code=ERROR_LLM_ERROR,
            )
        )
        trace.finish_run(final_status="error")
        yield StreamEvent(event_type="error", content=str(exc), error_code=ERROR_LLM_ERROR)


def _chunk_text(content: str, chunk_size: int = 10) -> list[str]:
    return [content[index : index + chunk_size] for index in range(0, len(content), chunk_size)] or [""]


def _extract_answer_content_progress(raw_text: str) -> str | None:
    action_value = _extract_json_string_value(raw_text, "action")
    if action_value != "answer":
        return None
    return _extract_json_string_value(raw_text, "content")


def _extract_json_string_value(raw_text: str, key: str) -> str | None:
    marker = f'"{key}"'
    key_index = raw_text.find(marker)
    if key_index == -1:
        return None
    colon_index = raw_text.find(":", key_index + len(marker))
    if colon_index == -1:
        return None

    value_start = colon_index + 1
    while value_start < len(raw_text) and raw_text[value_start] in {" ", "\t", "\r", "\n"}:
        value_start += 1
    if value_start >= len(raw_text) or raw_text[value_start] != '"':
        return None
    return _decode_json_string_partial(raw_text, value_start)


def _decode_json_string_partial(raw_text: str, quote_index: int) -> str:
    chars: list[str] = []
    index = quote_index + 1
    while index < len(raw_text):
        char = raw_text[index]
        if char == '"':
            break
        if char != "\\":
            chars.append(char)
            index += 1
            continue
        if index + 1 >= len(raw_text):
            break
        esc = raw_text[index + 1]
        escape_map = {
            '"': '"',
            "\\": "\\",
            "/": "/",
            "b": "\b",
            "f": "\f",
            "n": "\n",
            "r": "\r",
            "t": "\t",
        }
        if esc in escape_map:
            chars.append(escape_map[esc])
            index += 2
            continue
        if esc == "u":
            if index + 5 >= len(raw_text):
                break
            code = raw_text[index + 2 : index + 6]
            try:
                chars.append(chr(int(code, 16)))
            except ValueError:
                pass
            index += 6
            continue
        chars.append(esc)
        index += 2
    return "".join(chars)
