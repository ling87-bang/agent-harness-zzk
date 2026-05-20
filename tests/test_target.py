from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx
import pytest

from harness.errors import ERROR_TARGET_UNREACHABLE
from harness.skills.target import (
    AutoKnowledgeTarget,
    HttpKnowledgeTarget,
    MockKnowledgeTarget,
    SqliteKnowledgeTarget,
    build_knowledge_target,
    ensure_knowledge_database,
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


@pytest.mark.asyncio()
async def test_sqlite_target_search(tmp_path) -> None:
    db_path = tmp_path / "knowledge.db"
    ensure_knowledge_database(db_path)
    target = SqliteKnowledgeTarget(db_path=db_path)
    result = await target.search("Chain", top_k=2)
    assert result.error_code is None
    assert result.hit_count >= 1
    assert any("Chain" in str(item.get("snippet", "")) for item in result.items)


def test_build_knowledge_target_sqlite(tmp_path) -> None:
    target = build_knowledge_target(provider="sqlite", sqlite_path=str(tmp_path / "knowledge.db"))
    assert isinstance(target, SqliteKnowledgeTarget)


def test_build_knowledge_target_auto() -> None:
    target = build_knowledge_target(provider="auto")
    assert isinstance(target, AutoKnowledgeTarget)


@pytest.mark.asyncio()
async def test_auto_target_falls_back_to_sqlite_on_unreachable(monkeypatch, tmp_path) -> None:
    db_path = tmp_path / "knowledge.db"
    ensure_knowledge_database(db_path)
    http = HttpKnowledgeTarget(base_url="http://localhost:9/knowledge", timeout_seconds=0.1)
    auto = AutoKnowledgeTarget(http=http, sqlite=SqliteKnowledgeTarget(db_path=db_path))
    result = await auto.search("Chain", top_k=2)
    assert result.error_code is None
    assert result.hit_count >= 1
    second = await auto.search("Chain", top_k=2)
    assert second.hit_count >= 1
