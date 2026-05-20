"""Builtin web search skill."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.errors import ERROR_TOOL_CRASH
from harness.skills.base import SkillResult
from harness.skills.target import SearchTarget


@dataclass(frozen=True, slots=True)
class _WebSearchSkill:
    """Search the web via SearchTarget adapter."""

    target: SearchTarget

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web and return summarized results."

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


def make_web_search_skill(target: SearchTarget) -> _WebSearchSkill:
    """Build web_search skill from target implementation."""

    return _WebSearchSkill(target=target)


def _format_items(items: list[dict[str, Any]]) -> str:
    if not items:
        return "no web search results"
    lines: list[str] = []
    for index, item in enumerate(items, start=1):
        title = str(item.get("title", f"result-{index}"))
        url = str(item.get("url", ""))
        snippet = str(item.get("snippet", ""))
        prefix = f"[{index}] {title}"
        if url:
            prefix = f"{prefix} ({url})"
        lines.append(f"{prefix}: {snippet[:240]}")
    return "\n".join(lines)
