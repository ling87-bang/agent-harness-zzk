"""Builtin knowledge search skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.errors import ERROR_TOOL_CRASH
from harness.skills.base import SkillResult
from harness.skills.target import KnowledgeTarget


@dataclass(frozen=True, slots=True)
class _KnowledgeSearchSkill:
    """Search remote/local knowledge target."""

    target: KnowledgeTarget

    @property
    def name(self) -> str:
        return "knowledge_search"

    @property
    def description(self) -> str:
        return "Search the knowledge backend and return matched snippets."

    async def execute(self, **kwargs: Any) -> SkillResult:
        query = kwargs.get("query")
        top_k = kwargs.get("top_k", 5)
        if not isinstance(query, str) or not query.strip():
            return SkillResult(output="", error_code=ERROR_TOOL_CRASH)
        if not isinstance(top_k, int):
            top_k = 5
        result = await self.target.search(query=query, top_k=max(1, min(top_k, 20)))
        if result.error_code:
            return SkillResult(output=result.error or "", error_code=result.error_code)
        return SkillResult(
            output=_format_items(result.items),
            metadata={"hit_count": result.hit_count, "items": result.items},
        )


def make_knowledge_search_skill(target: KnowledgeTarget) -> _KnowledgeSearchSkill:
    """Build knowledge_search skill from target implementation."""

    return _KnowledgeSearchSkill(target=target)


def _format_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no knowledge results"
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        source = str(item.get("source", f"item-{index}"))
        snippet = str(item.get("snippet", item.get("content", "")))
        lines.append(f"[{index}] {source}: {snippet[:240]}")
    return "\n".join(lines)
