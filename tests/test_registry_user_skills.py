from pathlib import Path

from harness.skills.registry import SkillRegistry


def test_with_user_skills_does_not_override_builtins() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    example_dir = repo_root / "examples" / "skills"
    from harness.skills.discovery import discover_skills_from_directory

    user_skills = discover_skills_from_directory(example_dir)
    fake_builtin = dict(user_skills)
    fake_builtin["file_reader"] = user_skills["weather"]

    registry = SkillRegistry.with_builtins().with_user_skills(fake_builtin)
    assert registry.get("file_reader") is not None
    assert registry.get("file_reader").name == "file_reader"


def test_extra_tool_descriptions_excludes_builtins() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    from harness.skills.discovery import discover_skills_from_directory

    user_skills = discover_skills_from_directory(repo_root / "examples" / "skills")
    registry = SkillRegistry.with_builtins().with_user_skills(user_skills)
    names = {name for name, _ in registry.extra_tool_descriptions()}
    assert names == {"weather"}
