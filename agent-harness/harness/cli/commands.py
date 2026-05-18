"""CLI command implementations."""

from __future__ import annotations

import asyncio

import typer

from harness.cli.formatter import render_stream_event
from harness.config import get_settings
from harness.engine.context import ConversationManager
from harness.engine.loop import run_single_turn
from harness.engine.trace import TraceRecorder
from harness.llm.deepseek import DeepSeekProvider
from harness.llm.prompts import SYSTEM_PROMPT
from harness.skills.registry import SkillRegistry
from harness.skills.target import HttpKnowledgeTarget
from harness.state import Message


async def run_query_async(query: str) -> int:
    """Execute one query and stream terminal output."""

    settings = get_settings()
    if not settings.deepseek_api_key:
        typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
        return 1

    provider = DeepSeekProvider(settings=settings)
    registry = SkillRegistry.with_builtins(
        knowledge_target=HttpKnowledgeTarget(
            base_url=settings.knowledge_base_url,
            api_key=settings.knowledge_api_key or None,
            timeout_seconds=settings.knowledge_timeout_seconds,
        )
    )
    trace = TraceRecorder()
    has_error = False
    async for event in run_single_turn(query=query, provider=provider, trace=trace, registry=registry):
        render_stream_event(event)
        if event.error_code:
            has_error = True
    return 1 if has_error else 0


async def run_chat_async(conversation_id: str | None = None) -> int:
    """Run interactive multi-turn chat session."""

    settings = get_settings()
    if not settings.deepseek_api_key:
        typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
        return 1

    provider = DeepSeekProvider(settings=settings)
    registry = SkillRegistry.with_builtins(
        knowledge_target=HttpKnowledgeTarget(
            base_url=settings.knowledge_base_url,
            api_key=settings.knowledge_api_key or None,
            timeout_seconds=settings.knowledge_timeout_seconds,
        )
    )
    manager = ConversationManager()
    active_conversation_id = conversation_id or manager.new_conversation_id()
    history = manager.load_history(active_conversation_id)
    history = manager.compress_history(history)

    typer.echo(f"[chat] conversation_id={active_conversation_id}")
    has_error = False

    while True:
        user_input = typer.prompt("you")
        if user_input.strip().lower() in {"exit", "quit", ":q"}:
            break

        turn_messages = [
            Message(role="system", content=SYSTEM_PROMPT),
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
        history = manager.compress_history(updated_history)
        if not manager.save_history(active_conversation_id, history):
            typer.echo("[warn] failed to persist conversation history.")

    return 1 if has_error else 0


def run_query(query: str) -> int:
    """Sync wrapper for Typer commands."""

    return asyncio.run(run_query_async(query))


def run_chat(conversation_id: str | None = None) -> int:
    """Sync wrapper for interactive chat."""

    return asyncio.run(run_chat_async(conversation_id=conversation_id))
