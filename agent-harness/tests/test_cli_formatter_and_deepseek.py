from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

from harness.cli.app import _parse_conversation_id, app, default
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


def test_cli_default_callback(monkeypatch) -> None:
    runner = CliRunner()
    monkeypatch.setattr("harness.cli.app.run_query", lambda query: 0)
    result = runner.invoke(app, ["hello"])
    assert result.exit_code == 0


def test_cli_requires_query() -> None:
    runner = CliRunner()
    result = runner.invoke(app, [])
    assert result.exit_code == 1
    assert "Provide a query" in result.stdout


def test_default_dispatches_run_alias(monkeypatch) -> None:
    monkeypatch.setattr("harness.cli.app.run_query", lambda query: 0)

    class _Ctx:
        invoked_subcommand = None
        args = ["hello"]

    with pytest.raises(typer.Exit) as exc:
        default(_Ctx(), query="run")
    assert exc.value.exit_code == 0


def test_default_dispatches_chat_alias(monkeypatch) -> None:
    monkeypatch.setattr("harness.cli.app.run_chat", lambda conversation_id=None: 0)

    class _Ctx:
        invoked_subcommand = None
        args = ["--conversation-id", "conv-123"]

    with pytest.raises(typer.Exit) as exc:
        default(_Ctx(), query="chat")
    assert exc.value.exit_code == 0


def test_parse_conversation_id_option() -> None:
    assert _parse_conversation_id(["--conversation-id", "conv-123"]) == "conv-123"
    assert _parse_conversation_id([]) is None


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
