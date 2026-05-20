from dataclasses import FrozenInstanceError

import pytest

from harness.skills.base import SkillResult
from harness.state import Message


def test_message_is_frozen() -> None:
    item = Message(role="user", content="hello")
    with pytest.raises(FrozenInstanceError):
        item.content = "changed"


def test_skill_result_metadata_default_factory() -> None:
    first = SkillResult(output="a")
    second = SkillResult(output="b")
    first.metadata["k"] = "v"

    assert second.metadata == {}
