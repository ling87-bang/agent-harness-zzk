from __future__ import annotations

import pytest

from harness.chain.base import ChainContext
from harness.chain.nodes import LLMNode, PassThroughNode, SkillNode, TransformNode
from harness.errors import ERROR_LLM_ERROR, ERROR_TOOL_CRASH, ERROR_UNKNOWN_TOOL
from harness.skills.registry import SkillRegistry
from harness.skills.target import MockSearchTarget
from tests.conftest import FakeProvider


@pytest.mark.asyncio()
async def test_passthrough_node() -> None:
    node = PassThroughNode()
    result = await node.run("hello", ChainContext())
    assert result.output == "hello"
    assert result.error_code is None


@pytest.mark.asyncio()
async def test_transform_node_upper() -> None:
    node = TransformNode(transform=str.upper, node_name="upper")
    result = await node.run("abc", ChainContext())
    assert result.output == "ABC"


@pytest.mark.asyncio()
async def test_transform_node_crash_sets_error_code() -> None:
    def _boom(_: str) -> str:
        raise RuntimeError("boom")

    node = TransformNode(transform=_boom)
    result = await node.run("x", ChainContext())
    assert result.error_code == ERROR_TOOL_CRASH


@pytest.mark.asyncio()
async def test_llm_node_without_provider() -> None:
    node = LLMNode()
    result = await node.run("hi", ChainContext())
    assert result.error_code == ERROR_LLM_ERROR


@pytest.mark.asyncio()
async def test_llm_node_custom_system_prompt() -> None:
    provider = FakeProvider(stream_text="ok")
    node = LLMNode(system_prompt="custom")
    result = await node.run("question", ChainContext(provider=provider))
    assert result.output == "ok"
    assert result.error_code is None


@pytest.mark.asyncio()
async def test_skill_node_web_search_mock() -> None:
    registry = SkillRegistry.with_builtins(
        search_target=MockSearchTarget(items=[{"title": "t", "url": "u", "snippet": "s"}])
    )
    node = SkillNode(skill_name="web_search")
    result = await node.run("ai news", ChainContext(registry=registry))
    assert result.error_code is None
    assert "t" in result.output


@pytest.mark.asyncio()
async def test_skill_node_unknown_without_registry() -> None:
    node = SkillNode(skill_name="missing")
    result = await node.run("x", ChainContext())
    assert result.error_code == ERROR_UNKNOWN_TOOL
