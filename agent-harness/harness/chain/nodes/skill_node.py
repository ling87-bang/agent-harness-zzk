"""Skill chain node."""

from __future__ import annotations

from dataclasses import dataclass

from harness.chain.base import ChainContext, ChainResult
from harness.errors import ERROR_UNKNOWN_TOOL


@dataclass(frozen=True, slots=True)
class SkillNode:
    """Execute one registered skill; input_text maps to skill-specific args."""

    skill_name: str

    @property
    def name(self) -> str:
        return f"skill:{self.skill_name}"

    async def run(self, input_text: str, context: ChainContext) -> ChainResult:
        if context.registry is None:
            return ChainResult(output="", error_code=ERROR_UNKNOWN_TOOL, metadata={"node": self.name})
        args = _args_for_skill(self.skill_name, input_text, context.cwd)
        result = await context.registry.execute(self.skill_name, args)
        return ChainResult(
            output=result.output,
            error_code=result.error_code,
            metadata={"node": self.name, **result.metadata},
        )


def _args_for_skill(skill_name: str, input_text: str, cwd: str) -> dict[str, object]:
    if skill_name == "file_reader":
        return {"path": input_text, "cwd": cwd}
    if skill_name in {"knowledge_search", "web_search"}:
        return {"query": input_text}
    return {"query": input_text}
