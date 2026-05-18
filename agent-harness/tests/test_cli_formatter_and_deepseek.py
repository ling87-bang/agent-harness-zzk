from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
from typer.testing import CliRunner

from harness.cli.app import app
from harness.cli.formatter import render_stream_event
from harness.config import Settings
from harness.llm.deepseek import DeepSeekProvider
from harness.state import Message, StreamEvent


def test_formatter_renders_done_and_error(capsys) -> None:
    render_stream_event(StreamEvent(event_type="token", content="hi"))
    render_stream_event(StreamEvent(event_type="done", error_code="parse_failed"))
    render_stream_event(StreamEvent(event_type="error", content="oops", error_code="llm_error"))

    output = capsys.readouterr().out
    assert "hi" in output
    assert "[降级:parse_failed]" in output
    assert "[error:llm_error] oops" in output


def test_cli_requires_command() -> None:
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 1
    assert "Usage: zzk run" in result.stdout


def test_cli_run_alias(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("harness.cli.app.run_query", lambda query, **kwargs: 0)
    result = runner.invoke(app, ["run", "hello"])
    assert result.exit_code == 0


def test_cli_run_requires_query() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["run"])
    assert result.exit_code == 2


def test_cli_chat_alias(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("harness.cli.app.run_chat", lambda conversation_id=None, **kwargs: 0)
    result = runner.invoke(app, ["chat", "--conversation-id", "conv-123"])
    assert result.exit_code == 0


def test_cli_eval_alias(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    cases_file = tmp_path / "cases.json"
    cases_file.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        "harness.cli.app.run_eval",
        lambda cases_file, output_path=None, **kwargs: 0,
    )

    result = runner.invoke(app, ["eval", "--cases", str(cases_file)])
    assert result.exit_code == 0


def test_cli_eval_report_out_option(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    cases_file = tmp_path / "cases.json"
    cases_file.write_text("[]", encoding="utf-8")
    report_file = tmp_path / "report.json"
    captured: dict[str, object] = {}

    def _fake_eval(cases_file, output_path=None, **kwargs):
        captured["output_path"] = output_path
        return 0

    monkeypatch.setattr("harness.cli.app.run_eval", _fake_eval)
    result = runner.invoke(
        app,
        ["eval", "--cases-file", str(cases_file), "--report-out", str(report_file)],
    )
    assert result.exit_code == 0
    assert captured["output_path"] == report_file


def test_cli_eval_legacy_out_alias(monkeypatch, tmp_path) -> None:
    runner = CliRunner()
    cases_file = tmp_path / "cases.json"
    cases_file.write_text("[]", encoding="utf-8")
    report_file = tmp_path / "report.json"
    captured: dict[str, object] = {}

    def _fake_eval(cases_file, output_path=None, **kwargs):
        captured["output_path"] = output_path
        return 0

    monkeypatch.setattr("harness.cli.app.run_eval", _fake_eval)
    result = runner.invoke(
        app,
        ["eval", "--cases", str(cases_file), "--out", str(report_file)],
    )
    assert result.exit_code == 0
    assert captured["output_path"] == report_file


def test_cli_eval_requires_cases_option() -> None:
    runner = CliRunner()
    result = runner.invoke(app, ["eval"])
    assert result.exit_code == 2


@dataclass(frozen=True, slots=True)
class FakeResponse:
    payload: dict[str, Any]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload

    async def __aenter__(self) -> "FakeResponse":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def aiter_lines(self):
        for line in self.payload.get("lines", []):
            yield line


@dataclass(frozen=True, slots=True)
class FakeClient:
    response_payload: dict[str, Any]
    should_raise_post: bool = False
    should_raise_stream: bool = False

    async def __aenter__(self) -> "FakeClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        if self.should_raise_post:
            raise httpx.ConnectError("post failed")
        return FakeResponse(payload=self.response_payload)

    def stream(self, method: str, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
        if self.should_raise_stream:
            raise httpx.ConnectError("stream failed")
        return FakeResponse(payload=self.response_payload)


def _make_settings() -> Settings:
    return Settings(
        deepseek_api_key="key",
        deepseek_base_url="https://example.com",
        deepseek_model="deepseek-chat",
        deepseek_timeout_seconds=5.0,
    )


@pytest.mark.asyncio()
async def test_deepseek_chat_success(monkeypatch) -> None:
    payload = {"choices": [{"message": {"content": "ok"}}]}
    monkeypatch.setattr("harness.llm.deepseek.httpx.AsyncClient", lambda timeout: FakeClient(payload))
    provider = DeepSeekProvider(settings=_make_settings())

    text = await provider.chat([Message(role="user", content="hello")])
    assert text == "ok"


@pytest.mark.asyncio()
async def test_deepseek_chat_raises_runtime_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.llm.deepseek.httpx.AsyncClient",
        lambda timeout: FakeClient({}, should_raise_post=True),
    )
    provider = DeepSeekProvider(settings=_make_settings())

    with pytest.raises(RuntimeError):
        await provider.chat([Message(role="user", content="hello")])


@pytest.mark.asyncio()
async def test_deepseek_chat_raises_on_malformed_payload(monkeypatch) -> None:
    payload = {"choices": []}
    monkeypatch.setattr("harness.llm.deepseek.httpx.AsyncClient", lambda timeout: FakeClient(payload))
    provider = DeepSeekProvider(settings=_make_settings())

    with pytest.raises(RuntimeError, match="llm_error"):
        await provider.chat([Message(role="user", content="hello")])


@pytest.mark.asyncio()
async def test_deepseek_chat_stream_normalizes_tool_messages(monkeypatch) -> None:
    captured_payloads: list[dict[str, Any]] = []

    class CapturingClient:
        def __init__(self, timeout: float) -> None:
            pass

        async def __aenter__(self) -> "CapturingClient":
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        def stream(self, method: str, url: str, headers: dict[str, str], json: dict[str, Any]) -> FakeResponse:
            captured_payloads.append(json)
            return FakeResponse(payload={"lines": ["data: [DONE]"]})

    monkeypatch.setattr("harness.llm.deepseek.httpx.AsyncClient", CapturingClient)
    provider = DeepSeekProvider(settings=_make_settings())
    messages = [
        Message(role="system", content="s"),
        Message(role="user", content="u"),
        Message(role="assistant", content='{"action":"tool","name":"file_reader"}'),
        Message(role="tool", content='{"output":"denied"}'),
    ]

    events = []
    async for item in provider.chat_stream(messages):
        events.append(item)

    assert captured_payloads
    api_messages = captured_payloads[0]["messages"]
    assert all(message["role"] != "tool" for message in api_messages)
    assert api_messages[-1]["role"] == "user"
    assert "[tool_observation]" in api_messages[-1]["content"]


@pytest.mark.asyncio()
async def test_deepseek_chat_stream_success(monkeypatch) -> None:
    payload = {
        "lines": [
            'data: {"choices":[{"delta":{"content":"hel"}}]}',
            'data: {"choices":[{"delta":{"content":"lo"}}]}',
            "data: [DONE]",
        ]
    }
    monkeypatch.setattr("harness.llm.deepseek.httpx.AsyncClient", lambda timeout: FakeClient(payload))
    provider = DeepSeekProvider(settings=_make_settings())

    events = []
    async for item in provider.chat_stream([Message(role="user", content="hello")]):
        events.append(item)
    assert "".join(event.content for event in events if event.event_type == "token") == "hello"
    assert events[-1].event_type == "done"


@pytest.mark.asyncio()
async def test_deepseek_chat_stream_http_error(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.llm.deepseek.httpx.AsyncClient",
        lambda timeout: FakeClient({}, should_raise_stream=True),
    )
    provider = DeepSeekProvider(settings=_make_settings())

    events = []
    async for item in provider.chat_stream([Message(role="user", content="hello")]):
        events.append(item)
    assert events[0].event_type == "error"
    assert events[0].error_code == "llm_error"
