"""Skill registry and execution helpers."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harness.errors import ERROR_TOOL_CRASH, ERROR_TOOL_TIMEOUT, ERROR_UNKNOWN_TOOL
from harness.skills.base import Skill, SkillResult
from harness.config import Settings, get_settings
from harness.skills.builtins import (
    file_reader,
    file_writer,
    make_knowledge_search_skill,
    make_web_search_skill,
)
from harness.skills.discovery import discover_user_skills
from harness.skills.target import (
    KnowledgeTarget,
    MockKnowledgeTarget,
    MockSearchTarget,
    SearchTarget,
    build_knowledge_target,
    build_search_target,
)


@dataclass(frozen=True, slots=True)
class SkillRegistry:
    """Registry storing skills by name."""

    skills: dict[str, Skill] = field(default_factory=dict)
    timeout_seconds: float = 30.0

    @classmethod
    def with_builtins(
        cls,
        knowledge_target: KnowledgeTarget | None = None,
        search_target: SearchTarget | None = None,
        settings: Settings | None = None,
    ) -> "SkillRegistry":
        knowledge = knowledge_target or MockKnowledgeTarget(items=[])
        knowledge_search = make_knowledge_search_skill(knowledge)
        if search_target is None:
            runtime = settings or get_settings()
            search_target = build_search_target(
                provider=runtime.search_provider,
                api_key=runtime.search_api_key,
                timeout_seconds=runtime.search_timeout_seconds,
            )
        web_search = make_web_search_skill(search_target)
        return cls(
            skills={
                file_reader.name: file_reader,
                file_writer.name: file_writer,
                knowledge_search.name: knowledge_search,
                web_search.name: web_search,
            }
        )

    @classmethod
    def from_settings(
        cls,
        settings: Settings,
        *,
        enable_user_skills: bool = False,
        user_skill_dirs: tuple[Path, ...] = (),
    ) -> "SkillRegistry":
        """Build builtin registry from resolved settings (single settings source)."""

        registry = cls.with_builtins(
            knowledge_target=build_knowledge_target(
                provider=settings.knowledge_provider,
                base_url=settings.knowledge_base_url,
                api_key=settings.knowledge_api_key,
                timeout_seconds=settings.knowledge_timeout_seconds,
                sqlite_path=settings.knowledge_sqlite_path,
            ),
            search_target=build_search_target(
                provider=settings.search_provider,
                api_key=settings.search_api_key,
                timeout_seconds=settings.search_timeout_seconds,
            ),
            settings=settings,
        )
        if not enable_user_skills:
            return registry
        user_skills = discover_user_skills(user_skill_dirs or None)
        return registry.with_user_skills(user_skills)

    def with_user_skills(self, user_skills: dict[str, Skill]) -> "SkillRegistry":
        """Merge user skills; builtin names are not overwritten."""

        merged = dict(self.skills)
        for name, skill in user_skills.items():
            if name in merged:
                continue
            merged[name] = skill
        return SkillRegistry(skills=merged, timeout_seconds=self.timeout_seconds)

    def extra_tool_descriptions(self) -> tuple[tuple[str, str], ...]:
        """Return (name, description) for skills beyond the three builtins."""

        builtin_names = {"file_reader", "file_writer", "knowledge_search", "web_search"}
        return tuple(
            (skill.name, skill.description)
            for skill in self.skills.values()
            if skill.name not in builtin_names
        )

    @classmethod
    def with_builtins_for_tests(
        cls,
        knowledge_items: list[dict[str, object]] | None = None,
        search_items: list[dict[str, object]] | None = None,
    ) -> "SkillRegistry":
        """Build registry with mock targets for offline tests."""

        return cls.with_builtins(
            knowledge_target=MockKnowledgeTarget(items=knowledge_items),
            search_target=MockSearchTarget(items=search_items),
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
