"""Skill protocol definitions."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class SkillResult:
    """Standard skill execution result."""

    output: str
    metadata: dict[str, Any] = field(default_factory=dict)
    error_code: str | None = None


class Skill(Protocol):
    """Skill contract for plugin implementations."""

    @property
    def name(self) -> str:
        """Skill name."""

    @property
    def description(self) -> str:
        """Skill description."""

    async def execute(self, **kwargs: Any) -> SkillResult:
        """Execute skill with keyword arguments."""
