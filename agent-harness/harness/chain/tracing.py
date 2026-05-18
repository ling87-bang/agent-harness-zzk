"""Trace helpers for chain orchestration."""

from __future__ import annotations

import time

from harness.chain.base import ChainContext, ChainNode, ChainResult
from harness.engine.trace import TraceRecorder
from harness.state import TraceStep


def start_chain_trace(trace: TraceRecorder, query: str, context: ChainContext, *, chain_name: str) -> None:
    """Write run header for a chain execution."""

    provider = context.provider
    trace.start_run(
        query=query,
        llm_provider=provider.name if provider is not None else "chain",
        llm_model=provider.model if provider is not None else chain_name,
    )


def append_chain_node_trace(
    trace: TraceRecorder,
    *,
    step: int,
    node_name: str,
    result: ChainResult,
    latency_ms: float,
) -> None:
    """Record one chain node step."""

    trace.append_step(
        TraceStep(
            step=step,
            step_type="chain_node",
            run_id=trace.run_id,
            skill=node_name,
            status="failed" if result.error_code else "succeeded",
            error_code=result.error_code,
            error=result.output[:500] if result.error_code else None,
            latency_ms=latency_ms,
        )
    )


def finish_chain_trace(trace: TraceRecorder, *, error_code: str | None) -> None:
    """Finalize chain run status."""

    trace.finish_run(final_status="error" if error_code else "success")


async def run_node_timed(node: ChainNode, input_text: str, context: ChainContext) -> tuple[ChainResult, float]:
    """Run a node and return result with elapsed milliseconds."""

    started = time.perf_counter()
    result = await node.run(input_text, context)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return result, elapsed_ms
