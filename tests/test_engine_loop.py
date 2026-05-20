import json
from pathlib import Path

import pytest

from harness.engine.loop import run_single_turn
from harness.engine.trace import TraceRecorder
from harness.errors import ERROR_LLM_ERROR, ERROR_MAX_STEPS, ERROR_UNKNOWN_TOOL
from harness.skills.registry import SkillRegistry
from harness.skills.target import MockKnowledgeTarget
from harness.state import StreamEvent
from tests.conftest import FakeProvider


@pytest.mark.asyncio()
async def test_run_single_turn_success(tmp_path) -> None:
    provider = FakeProvider(
        stream_text='{"action":"answer","content":"hello world","reasoning":"ok","citations":[]}'
    )
    trace = TraceRecorder(trace_dir=tmp_path)

    events: list[StreamEvent] = []
    async for item in run_single_turn(query="hello", provider=provider, trace=trace):
        events.append(item)

    tokens = "".join(event.content for event in events if event.event_type == "token")
    assert "hello world" in tokens
    done_event = [event for event in events if event.event_type == "done"][-1]
    assert done_event.error_code is None


@pytest.mark.asyncio()
async def test_run_single_turn_parse_fallback_records_error_code(tmp_path) -> None:
    provider = FakeProvider(stream_text="not json")
    trace = TraceRecorder(trace_dir=tmp_path)

    events: list[StreamEvent] = []
    async for item in run_single_turn(query="hello", provider=provider, trace=trace):
        events.append(item)

    done_event = [event for event in events if event.event_type == "done"][-1]
    assert done_event.error_code == "parse_failed"

    trace_files = list(tmp_path.glob("*.jsonl"))
    assert len(trace_files) == 1
    rows = [
        json.loads(line)
        for line in trace_files[0].read_text(encoding="utf-8").splitlines()
    ]
    parse_rows = [row for row in rows if row.get("step_type") == "parse_result"]
    assert parse_rows[0]["error_code"] == "parse_failed"


@pytest.mark.asyncio()
async def test_run_single_turn_llm_error(tmp_path) -> None:
    provider = FakeProvider(should_error=True)
    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []

    async for item in run_single_turn(query="x", provider=provider, trace=trace):
        events.append(item)

    error_event = [event for event in events if event.event_type == "error"][-1]
    assert error_event.error_code == "llm_error"


@pytest.mark.asyncio()
async def test_run_single_turn_tool_then_answer(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    sample = tmp_path / "note.txt"
    sample.write_text("hello from file", encoding="utf-8")

    provider = FakeProvider(
        scripted_texts=(
            '{"action":"tool","name":"file_reader","args":{"path":"note.txt"},"reasoning":"read file"}',
            '{"action":"answer","content":"done","reasoning":"r","citations":[]}',
        )
    )
    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []

    async for item in run_single_turn(query="read note", provider=provider, trace=trace):
        events.append(item)

    tokens = "".join(event.content for event in events if event.event_type == "token")
    assert "done" in tokens
    rows = [
        json.loads(line)
        for line in next(Path(tmp_path).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    skill_rows = [row for row in rows if row.get("step_type") == "skill_execution"]
    assert len(skill_rows) == 1
    assert skill_rows[0]["status"] == "succeeded"


@pytest.mark.asyncio()
async def test_run_single_turn_hits_max_steps(tmp_path) -> None:
    provider = FakeProvider(
        scripted_texts=(
            '{"action":"tool","name":"file_reader","args":{"path":"x.txt"},"reasoning":"r"}',
        )
    )
    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []

    async for item in run_single_turn(
        query="loop",
        provider=provider,
        trace=trace,
        max_steps=1,
    ):
        events.append(item)

    done = [event for event in events if event.event_type == "done"][-1]
    assert done.error_code == ERROR_MAX_STEPS


@pytest.mark.asyncio()
async def test_run_single_turn_with_knowledge_search_integration(tmp_path) -> None:
    provider = FakeProvider(
        scripted_texts=(
            '{"action":"tool","name":"knowledge_search","args":{"query":"phase4","top_k":2},"reasoning":"need kb"}',
            '{"action":"answer","content":"knowledge ready","reasoning":"ok","citations":[]}',
        )
    )
    registry = SkillRegistry.with_builtins(
        knowledge_target=MockKnowledgeTarget(
            items=[{"source": "kb-doc", "snippet": "phase4 details"}]
        )
    )
    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []

    async for item in run_single_turn(
        query="search knowledge",
        provider=provider,
        trace=trace,
        registry=registry,
    ):
        events.append(item)

    tokens = "".join(event.content for event in events if event.event_type == "token")
    assert "knowledge ready" in tokens
    rows = [
        json.loads(line)
        for line in next(Path(tmp_path).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    skill_rows = [row for row in rows if row.get("step_type") == "skill_execution"]
    assert skill_rows
    assert skill_rows[0]["skill"] == "knowledge_search"
    assert skill_rows[0]["status"] == "succeeded"


@pytest.mark.asyncio()
async def test_run_single_turn_unknown_tool_branch(tmp_path) -> None:
    provider = FakeProvider(
        scripted_texts=(
            '{"action":"tool","name":"missing_tool","args":{"x":1},"reasoning":"r"}',
        )
    )
    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []

    async for item in run_single_turn(query="unknown", provider=provider, trace=trace):
        events.append(item)

    done = [event for event in events if event.event_type == "done"][-1]
    assert done.error_code == ERROR_UNKNOWN_TOOL
    rows = [
        json.loads(line)
        for line in next(Path(tmp_path).glob("*.jsonl")).read_text(encoding="utf-8").splitlines()
    ]
    assert any(
        row.get("step_type") == "parse_result" and row.get("error_code") == ERROR_UNKNOWN_TOOL
        for row in rows
    )


@pytest.mark.asyncio()
async def test_run_single_turn_catches_unexpected_exception(tmp_path) -> None:
    class BrokenProvider:
        name = "broken"
        model = "broken-model"

        async def chat_stream(self, messages, *, temperature=0.1, max_tokens=4096):
            raise RuntimeError("explode")
            yield  # pragma: no cover

        async def chat(self, messages, *, temperature=0.1, max_tokens=4096):
            return ""

    trace = TraceRecorder(trace_dir=tmp_path)
    events: list[StreamEvent] = []
    async for item in run_single_turn(query="x", provider=BrokenProvider(), trace=trace):
        events.append(item)

    error = [event for event in events if event.event_type == "error"][-1]
    assert error.error_code == ERROR_LLM_ERROR
