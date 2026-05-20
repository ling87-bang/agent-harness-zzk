"""LLM chain node."""

from __future__ import annotations

from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainResult
from harness.errors import ERROR_LLM_ERROR
from harness.state import Message

LLM_NODE_SYSTEM_PROMPT = "You are a chain step. Respond with plain text."


@dataclass(frozen=True, slots=True)
class LLMNode:
    """Invoke LLM once with plain-text output (no ReAct JSON schema)."""

    system_prompt: str = LLM_NODE_SYSTEM_PROMPT

    @property
    def name(self) -> str:
        return "llm"

    async def run(self, input_text: str, context: ChainContext) -> ChainResult:
        if context.provider is None:
            return ChainResult(output="", error_code=ERROR_LLM_ERROR)
        messages = [
            Message(role="system", content=self.system_prompt),
            Message(role="user", content=input_text),
        ]
        try:
            output = await context.provider.chat(messages)
        except Exception as exc:
            return ChainResult(
                output=str(exc),
                error_code=ERROR_LLM_ERROR,
                metadata={"node": self.name},
            )
        return ChainResult(output=output, metadata={"node": self.name, "model": context.provider.model})
