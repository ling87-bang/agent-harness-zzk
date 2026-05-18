"""Transform chain node."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainResult
from harness.errors import ERROR_TOOL_CRASH


@dataclass(frozen=True, slots=True)
class TransformNode:
    """Apply a synchronous text transform."""

    transform: Callable[[str], str]
    node_name: str = "transform"

    @property
    def name(self) -> str:
        return self.node_name

    async def run(self, input_text: str, context: ChainContext) -> ChainResult:
        try:
            output = self.transform(input_text)
        except Exception as exc:
            return ChainResult(
                output=str(exc),
                error_code=ERROR_TOOL_CRASH,
                metadata={"node": self.name},
            )
        return ChainResult(output=output, metadata={"node": self.name})
