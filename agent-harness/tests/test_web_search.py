from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from harness.errors import ERROR_SEARCH_MISCONFIGURED, ERROR_TARGET_UNREACHABLE, ERROR_TOOL_CRASH
from harness.skills.builtins.web_search import make_web_search_skill
from harness.skills.registry import SkillRegistry
from harness.skills.target import (
    DuckDuckGoTarget,
    MockSearchTarget,
    SerpApiTarget,
    build_search_target,
)


@dataclass(frozen=True, slots=True)
class _FakeResponse:
    payload: dict[str, Any]

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


@dataclass(frozen=True, slots=True)
class _FakeClient:
    payload: dict[str, Any]
    should_fail: bool = False

    async def __aenter__(self) -> "_FakeClient":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    async def get(self, url: str, params: dict[str, Any] | None = None) -> _FakeResponse:
        if self.should_fail:
            raise httpx.ConnectError("offline")
        return _FakeResponse(payload=self.payload)


@pytest.mark.asyncio()
async def test_mock_search_target_and_skill() -> None:
    target = MockSearchTarget(items=[{"title": "News", "url": "https://x", "snippet": "body"}])
    skill = make_web_search_skill(target)
    result = await skill.execute(query="ai", top_k=3)
    assert result.error_code is None
    assert "News" in result.output


@pytest.mark.asyncio()
async def test_web_search_invalid_query() -> None:
    skill = make_web_search_skill(MockSearchTarget())
    result = await skill.execute(query="  ")
    assert result.error_code == ERROR_TOOL_CRASH


@pytest.mark.asyncio()
async def test_duckduckgo_target_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.skills.target.httpx.AsyncClient",
        lambda timeout: _FakeClient(
            payload={"AbstractText": "summary", "Heading": "AI", "AbstractURL": "https://example.com"}
        ),
    )
    target = DuckDuckGoTarget(timeout_seconds=1.0, min_interval_seconds=0.0)
    result = await target.search("ai", top_k=2)
    assert result.error_code is None
    assert result.hit_count >= 1


@pytest.mark.asyncio()
async def test_duckduckgo_target_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.skills.target.httpx.AsyncClient",
        lambda timeout: _FakeClient(payload={}, should_fail=True),
    )
    target = DuckDuckGoTarget(min_interval_seconds=0.0)
    result = await target.search("ai", top_k=2)
    assert result.error_code == ERROR_TARGET_UNREACHABLE


@pytest.mark.asyncio()
async def test_serpapi_misconfigured_without_key() -> None:
    target = SerpApiTarget(api_key="")
    result = await target.search("ai", top_k=2)
    assert result.error_code == ERROR_SEARCH_MISCONFIGURED


@pytest.mark.asyncio()
async def test_serpapi_target_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.skills.target.httpx.AsyncClient",
        lambda timeout: _FakeClient(
            payload={"organic_results": [{"title": "T", "link": "https://t", "snippet": "S"}]}
        ),
    )
    target = SerpApiTarget(api_key="secret")
    result = await target.search("ai", top_k=2)
    assert result.error_code is None
    assert result.hit_count == 1


@pytest.mark.asyncio()
async def test_registry_with_builtins_contains_web_search() -> None:
    registry = SkillRegistry.with_builtins_for_tests(search_items=[{"title": "x", "snippet": "y"}])
    result = await registry.execute("web_search", {"query": "zzk"})
    assert result.error_code is None
    assert "x" in result.output


def test_build_search_target_serpapi() -> None:
    target = build_search_target(provider="serpapi", api_key="k")
    assert isinstance(target, SerpApiTarget)
