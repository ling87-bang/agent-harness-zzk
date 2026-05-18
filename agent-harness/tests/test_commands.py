import pytest

from harness.cli.commands import run_chat_async, run_query_async
from harness.config import get_settings
from harness.engine.context import ConversationManager
from harness.state import StreamEvent


@pytest.mark.asyncio()
async def test_run_query_async_without_api_key(monkeypatch) -> None:
    monkeypatch.delenv("ZZK_DEEPSEEK_API_KEY", raising=False)
    get_settings.cache_clear()

    code = await run_query_async("hello")
    assert code == 1


@pytest.mark.asyncio()
async def test_run_query_async_returns_nonzero_on_degraded(monkeypatch) -> None:
    monkeypatch.setenv("ZZK_DEEPSEEK_API_KEY", "k")
    get_settings.cache_clear()

    class _FakeProvider:
        pass

    async def _fake_stream(*, query, provider, trace, registry=None):
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

    async def _fake_stream(*, query, provider, trace, registry=None, messages=None, conversation_id=""):
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
