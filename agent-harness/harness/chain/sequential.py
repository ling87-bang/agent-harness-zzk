"""Sequential chain: pipe each node output into the next."""

from __future__ import annotations

from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainNode, ChainResult
from harness.chain.tracing import append_chain_node_trace, run_node_timed


@dataclass(frozen=True, slots=True)
class SequentialChain:
    """Run nodes in order; stop on first error_code."""

    nodes: tuple[ChainNode, ...]
    chain_name: str = "sequential"

    @property
    def name(self) -> str:
        return self.chain_name

    async def run(self, input_text: str, context: ChainContext | None = None) -> ChainResult:
        ctx = context or ChainContext()
        if not self.nodes:
            return ChainResult(output=input_text, metadata={"chain": self.name, "node_count": 0})

        current = input_text
        collected: list[ChainResult] = []
        for index, node in enumerate(self.nodes, start=1):
            result, elapsed_ms = await run_node_timed(node, current, ctx)
            if ctx.trace is not None:
                append_chain_node_trace(
                    ctx.trace,
                    step=index,
                    node_name=node.name,
                    result=result,
                    latency_ms=elapsed_ms,
                )
            collected.append(result)
            if result.error_code is not None:
                return ChainResult(
                    output=result.output,
                    error_code=result.error_code,
                    metadata={"chain": self.name, "failed_node": node.name},
                    sub_results=tuple(collected),
                )
            current = result.output

        return ChainResult(
            output=current,
            metadata={"chain": self.name, "node_count": len(self.nodes)},
            sub_results=tuple(collected),
        )
