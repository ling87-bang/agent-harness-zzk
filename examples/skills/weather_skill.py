"""Example user skill — copy to ~/.zzk/skills/ and run with --enable-user-skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from harness.skills.base import SkillResult


@dataclass(frozen=True, slots=True)
class _WeatherSkill:
    """Return mock weather for a city (demo only, no external API)."""

    @property
    def name(self) -> str:
        return "weather"

    @property
    def description(self) -> str:
        return "Get current weather for a city (demo mock data)."

    async def execute(self, **kwargs: Any) -> SkillResult:
        city = kwargs.get("city")
        if not isinstance(city, str) or not city.strip():
            return SkillResult(output="city is required", error_code="tool_crash")
        normalized = city.strip()
        return SkillResult(
            output=f"Weather in {normalized}: sunny, 24C, light wind (demo data).",
            metadata={"city": normalized, "source": "mock"},
        )


skill = _WeatherSkill()
