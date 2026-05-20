import pytest

from harness.errors import ERROR_TARGET_UNREACHABLE
from harness.skills.builtins.knowledge_search import make_knowledge_search_skill
from harness.skills.registry import SkillRegistry
from harness.skills.target import KnowledgeResult, MockKnowledgeTarget


@pytest.mark.asyncio()
async def test_knowledge_search_skill_success() -> None:
    target = MockKnowledgeTarget(items=[{"source": "doc-1", "snippet": "hello"}])
    skill = make_knowledge_search_skill(target)
    result = await skill.execute(query="hello", top_k=5)
    assert result.error_code is None
    assert "doc-1" in result.output
    assert result.metadata["hit_count"] == 1


@pytest.mark.asyncio()
async def test_knowledge_search_skill_error() -> None:
    class _FailTarget:
        async def search(self, query: str, top_k: int = 5) -> KnowledgeResult:
            return KnowledgeResult(items=[], hit_count=0, error="down", error_code=ERROR_TARGET_UNREACHABLE)

    skill = make_knowledge_search_skill(_FailTarget())
    result = await skill.execute(query="hello")
    assert result.error_code == ERROR_TARGET_UNREACHABLE


@pytest.mark.asyncio()
async def test_registry_with_builtins_contains_knowledge_search() -> None:
    registry = SkillRegistry.with_builtins()
    result = await registry.execute("knowledge_search", {"query": "x"})
    assert result.error_code is None
