"""Pass-through chain node for debugging."""

from __future__ import annotations

from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainResult


@dataclass(frozen=True, slots=True)
class PassThroughNode:
    """Return input unchanged."""

    @property
    def name(self) -> str:
        return "passthrough"

    async def run(self, input_text: str, context: ChainContext) -> ChainResult:
        return ChainResult(output=input_text, metadata={"node": self.name})
