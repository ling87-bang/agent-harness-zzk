"""Router chain: select a branch by predicate."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainNode, ChainResult
from harness.chain.tracing import append_chain_node_trace, run_node_timed
from harness.errors import ERROR_CHAIN_ROUTE_MISS


@dataclass(frozen=True, slots=True)
class RouterChain:
    """Route input to one branch node based on a predicate key."""

    routes: Mapping[str, ChainNode]
    predicate: Callable[[str, ChainContext], str]
    default_key: str | None = None
    chain_name: str = "router"

    @property
    def name(self) -> str:
        return self.chain_name

    async def run(self, input_text: str, context: ChainContext | None = None) -> ChainResult:
        ctx = context or ChainContext()
        route_key = self.predicate(input_text, ctx)
        node = self.routes.get(route_key)
        if node is None and self.default_key is not None:
            route_key = self.default_key
            node = self.routes.get(route_key)
        if node is None:
            return ChainResult(
                output=input_text,
                error_code=ERROR_CHAIN_ROUTE_MISS,
                metadata={"chain": self.name, "route_key": route_key},
            )
        result, elapsed_ms = await run_node_timed(node, input_text, ctx)
        if ctx.trace is not None:
            append_chain_node_trace(
                ctx.trace,
                step=1,
                node_name=node.name,
                result=result,
                latency_ms=elapsed_ms,
            )
        return ChainResult(
            output=result.output,
            error_code=result.error_code,
            metadata={**result.metadata, "chain": self.name, "route_key": route_key},
            sub_results=(result,),
        )
