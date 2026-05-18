from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from harness.errors import ERROR_TARGET_UNREACHABLE
from harness.skills.target import HttpKnowledgeTarget, MockKnowledgeTarget


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

    async def post(self, url: str, headers: dict[str, str], json: dict[str, Any]) -> _FakeResponse:
        if self.should_fail:
            raise httpx.ConnectError("offline")
        return _FakeResponse(payload=self.payload)


@pytest.mark.asyncio()
async def test_http_target_success(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.skills.target.httpx.AsyncClient",
        lambda timeout: _FakeClient(payload={"items": [{"source": "doc", "snippet": "x"}]}),
    )
    target = HttpKnowledgeTarget(base_url="http://localhost:8000/knowledge")
    result = await target.search("hello", top_k=3)
    assert result.error_code is None
    assert result.hit_count == 1


@pytest.mark.asyncio()
async def test_http_target_unreachable(monkeypatch) -> None:
    monkeypatch.setattr(
        "harness.skills.target.httpx.AsyncClient",
        lambda timeout: _FakeClient(payload={}, should_fail=True),
    )
    target = HttpKnowledgeTarget(base_url="http://localhost:8000/knowledge")
    result = await target.search("hello", top_k=3)
    assert result.error_code == ERROR_TARGET_UNREACHABLE


@pytest.mark.asyncio()
async def test_mock_target_returns_items() -> None:
    target = MockKnowledgeTarget(items=[{"source": "a"}, {"source": "b"}])
    result = await target.search("hi", top_k=1)
    assert result.hit_count == 1
    assert result.items[0]["source"] == "a"
