from pathlib import Path

import pytest

from harness.skills.discovery import discover_skills_from_directory
from harness.skills.registry import SkillRegistry


@pytest.mark.asyncio()
async def test_discover_example_weather_skill() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples" / "skills"
    discovered = discover_skills_from_directory(example_dir)
    assert "weather" in discovered

    registry = SkillRegistry.with_builtins().with_user_skills(discovered)
    result = await registry.execute("weather", {"city": "Shanghai"})
    assert result.error_code is None
    assert "Shanghai" in result.output


def test_discover_skips_private_modules(tmp_path: Path) -> None:
    (tmp_path / "_hidden.py").write_text("skill = None\n", encoding="utf-8")
    assert discover_skills_from_directory(tmp_path) == {}
