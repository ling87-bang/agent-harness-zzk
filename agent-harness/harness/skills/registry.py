"""Skill registry and execution helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from harness.errors import ERROR_TOOL_CRASH, ERROR_TOOL_TIMEOUT, ERROR_UNKNOWN_TOOL
from harness.skills.base import Skill, SkillResult
from harness.skills.builtins import file_reader, make_knowledge_search_skill
from harness.skills.target import KnowledgeTarget, MockKnowledgeTarget


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """Registry storing skills by name."""

    skills: dict[str, Skill] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    @classmethod
    def with_builtins(cls, knowledge_target: KnowledgeTarget | None = None) -> "SkillRegistry":
        target = knowledge_target or MockKnowledgeTarget(items=[])
        knowledge_search = make_knowledge_search_skill(target)
        return cls(
            skills={
                file_reader.name: file_reader,
                knowledge_search.name: knowledge_search,
            }
        )

    def get(self, name: str) -> Skill | None:
        return self.skills.get(name)

    async def execute(self, name: str, args: dict[str, Any]) -> SkillResult:
        skill = self.get(name)
        if skill is None:
            return SkillResult(output=f"unknown tool: {name}", error_code=ERROR_UNKNOWN_TOOL)
        try:
            return await asyncio.wait_for(skill.execute(**args), timeout=self.timeout_seconds)
        except asyncio.TimeoutError:
            return SkillResult(output=f"tool timeout: {name}", error_code=ERROR_TOOL_TIMEOUT)
        except Exception:
            return SkillResult(output=f"tool crashed: {name}", error_code=ERROR_TOOL_CRASH)
