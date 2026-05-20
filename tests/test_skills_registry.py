import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from harness.errors import ERROR_TOOL_CRASH, ERROR_TOOL_TIMEOUT, ERROR_UNKNOWN_TOOL
from harness.skills.base import SkillResult
from harness.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class _TimeoutSkill:
    name: str = "timeout_skill"
    description: str = "sleep"

    async def execute(self, **kwargs: Any) -> SkillResult:
        await asyncio.sleep(0.05)
        return SkillResult(output="late")


@dataclass(frozen=True, slots=True)
class _CrashSkill:
    name: str = "crash_skill"
    description: str = "crash"

    async def execute(self, **kwargs: Any) -> SkillResult:
        raise RuntimeError("boom")


@pytest.mark.asyncio()
async def test_registry_unknown_tool() -> None:
    registry = SkillRegistry.with_builtins()
    result = await registry.execute("missing", {})
    assert result.error_code == ERROR_UNKNOWN_TOOL


@pytest.mark.asyncio()
async def test_registry_timeout() -> None:
    registry = SkillRegistry(skills={"timeout_skill": _TimeoutSkill()}, timeout_seconds=0.01)
    result = await registry.execute("timeout_skill", {})
    assert result.error_code == ERROR_TOOL_TIMEOUT


@pytest.mark.asyncio()
async def test_registry_crash() -> None:
    registry = SkillRegistry(skills={"crash_skill": _CrashSkill()})
    result = await registry.execute("crash_skill", {})
    assert result.error_code == ERROR_TOOL_CRASH
