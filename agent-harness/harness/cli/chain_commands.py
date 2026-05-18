"""CLI commands for chain orchestration."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer

from harness.chain import ChainContext, SequentialChain, list_chains
from harness.chain.nodes import LLMNode, PassThroughNode, SkillNode, TransformNode
from harness.chain.tracing import finish_chain_trace, start_chain_trace
from harness.config import get_settings
from harness.engine.trace import TraceRecorder
from harness.errors import ERROR_CODE_MAP
from harness.llm.deepseek import DeepSeekProvider
from harness.skills.registry import SkillRegistry


def parse_chain_steps(steps: str) -> tuple[object, ...]:
    """Parse comma-separated step tokens into chain nodes."""

    nodes: list[object] = []
    for raw in steps.split(","):
        token = raw.strip()
        if not token:
            continue
        if token == "llm":
            nodes.append(LLMNode())
            continue
        if token == "passthrough":
            nodes.append(PassThroughNode())
            continue
        if token.startswith("skill:"):
            skill_name = token.split(":", 1)[1].strip()
            if not skill_name:
                raise ValueError(f"invalid skill step: {token}")
            nodes.append(SkillNode(skill_name=skill_name))
            continue
        if token == "transform:upper":
            nodes.append(TransformNode(transform=str.upper, node_name="transform:upper"))
            continue
        if token.startswith("transform:truncate:"):
            limit_text = token.split(":", 2)[2]
            limit = int(limit_text)

            def _truncate(value: str, max_len: int = limit) -> str:
                return value[:max_len]

            nodes.append(TransformNode(transform=_truncate, node_name=token))
            continue
        raise ValueError(f"unknown step token: {token}")
    return tuple(nodes)


def _steps_need_llm(steps: str) -> bool:
    return any(part.strip() == "llm" for part in steps.split(","))


def _build_runtime_context(steps: str, trace: TraceRecorder | None = None) -> ChainContext | None:
    settings = get_settings()
    needs_llm = _steps_need_llm(steps)
    provider = None
    if needs_llm:
        if not settings.deepseek_api_key:
            typer.echo(f"Missing DeepSeek API key for {settings.app_name}. Set ZZK_DEEPSEEK_API_KEY.")
            return None
        provider = DeepSeekProvider(settings=settings)
    registry = SkillRegistry.from_settings(settings)
    return ChainContext(provider=provider, registry=registry, cwd=str(Path.cwd()), trace=trace)


def run_chain_list() -> int:
    """Print registered chains and step syntax."""

    names = list_chains()
    if names:
        typer.echo("Registered chains:")
        for name in names:
            typer.echo(f"  - {name}")
    else:
        typer.echo("No pre-registered chains (use `zzk chain run sequential`).")
    typer.echo("")
    typer.echo("Step tokens: llm | passthrough | skill:<name> | transform:upper | transform:truncate:N")
    return 0


async def run_chain_sequential_async(steps: str, input_text: str) -> int:
    try:
        nodes = parse_chain_steps(steps)
    except ValueError as exc:
        typer.echo(f"[chain:error] {exc}")
        return 1

    trace = TraceRecorder()
    context = _build_runtime_context(steps, trace=trace)
    if context is None:
        return 1

    chain = SequentialChain(nodes=nodes)
    start_chain_trace(trace, input_text, context, chain_name=chain.name)
    result = await chain.run(input_text, context)
    finish_chain_trace(trace, error_code=result.error_code)
    typer.echo(f"[chain:trace] run_id={trace.run_id} path={trace.path}")

    if result.error_code:
        message = ERROR_CODE_MAP.get(result.error_code, result.error_code)
        typer.echo(f"[chain:error:{result.error_code}] {message}")
        typer.echo(f"[chain:hint] zzk trace show {trace.run_id}")
        if result.output:
            typer.echo(result.output)
        return 1
    typer.echo(result.output)
    return 0


def run_chain_sequential(steps: str, input_text: str) -> int:
    """Run a sequential chain from CLI."""

    return asyncio.run(run_chain_sequential_async(steps, input_text))
