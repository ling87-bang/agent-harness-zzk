import pytest

from harness.llm.prompts import (
    DEFAULT_PROMPT_VERSION,
    PROMPT_CHANGELOG,
    PROMPT_VERSIONS,
    get_system_prompt,
)


def test_prompt_versions_exist() -> None:
    assert "v1" in PROMPT_VERSIONS
    assert "v2" in PROMPT_VERSIONS
    assert DEFAULT_PROMPT_VERSION == "v2"


def test_v2_includes_json_examples() -> None:
    prompt = get_system_prompt("v2")
    assert "Examples:" in prompt
    assert '"action":"tool"' in prompt


def test_v1_is_shorter_than_v2() -> None:
    assert len(get_system_prompt("v1")) < len(get_system_prompt("v2"))


def test_extra_tools_appended() -> None:
    prompt = get_system_prompt("v2", extra_tools=(("weather", "Mock weather lookup"),))
    assert "User-provided tools:" in prompt
    assert "weather" in prompt


def test_unknown_prompt_version_raises() -> None:
    with pytest.raises(ValueError, match="Unknown prompt version"):
        get_system_prompt("v99")


def test_changelog_has_entries() -> None:
    assert PROMPT_CHANGELOG["v1"]
    assert PROMPT_CHANGELOG["v2"]
